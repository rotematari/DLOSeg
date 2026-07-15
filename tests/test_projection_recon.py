"""Synthetic success-criteria tests for the projection-based reconstruction.

These lift the mock-stereo pattern from `bspline_3d_recon.py`'s __main__ into
parametrizable fixtures (see conftest): a known 3D curve projected through mock
P1/P2 into two views, reconstructed, and compared to ground truth.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep

from dloseg.recon3d.bspline_3d_recon import triangulate_and_reconstruct
from dloseg.recon3d.projection_recon import reconstruct_projection_based
from dloseg.recon3d.reprojection import flag_epipolar_degenerate, project_points


def _dense(curve: np.ndarray, n: int = 800) -> np.ndarray:
    """Densely resample a curve (linear interp in its own parameter)."""
    src = np.linspace(0.0, 1.0, len(curve))
    dst = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(dst, src, curve[:, j]) for j in range(3)])


def _max_curve_error(recon: np.ndarray, gt: np.ndarray) -> float:
    """Max over recon samples of nearest 3D distance to the ground-truth curve."""
    dense = _dense(gt)
    d = np.linalg.norm(recon[:, None, :] - dense[None, :, :], axis=-1)
    return float(d.min(axis=1).max())


def _init_from_triangulation(scene):
    """Triangulation baseline spline used as the projection initializer."""
    _, tri_func = triangulate_and_reconstruct(
        scene.calib, scene.left_pts, scene.right_pts, matching="arclength"
    )
    assert tri_func is not None  # synthetic scenes always yield >= 4 matched points
    return tri_func


def test_projection_recovers_smooth_curve(smooth_scene) -> None:
    # Criterion 1: max 3D error < 5 mm and RMS reprojection < 0.5 px per view.
    tri_func = _init_from_triangulation(smooth_scene)
    proj_func, _, info = reconstruct_projection_based(
        smooth_scene.calib, smooth_scene.left_pts, smooth_scene.right_pts, tri_func
    )
    u = np.linspace(0.0, 1.0, 100)
    recon = proj_func(u)

    max_err_mm = _max_curve_error(recon, smooth_scene.curve3d) * 1000.0
    assert max_err_mm < 5.0, f"max 3D error {max_err_mm:.2f} mm"

    proj_l = project_points(smooth_scene.P1, recon)
    proj_r = project_points(smooth_scene.P2, recon)
    from dloseg.recon3d.reprojection import point_to_polyline_dist

    rms_l = np.sqrt(np.mean(point_to_polyline_dist(proj_l, smooth_scene.left_pts) ** 2))
    rms_r = np.sqrt(np.mean(point_to_polyline_dist(proj_r, smooth_scene.right_pts) ** 2))
    assert rms_l < 0.5 and rms_r < 0.5, f"reproj RMS L={rms_l:.3f} R={rms_r:.3f} px"
    assert info["runtime_s"] < 15.0


def test_hairpin_is_flagged_and_projection_beats_triangulation(hairpin_scene) -> None:
    # Criterion 2: the diagnostic flags the near-horizontal span, and the
    # projection method's max 3D error on that span is strictly lower than the
    # triangulation baseline's.
    tri_func = _init_from_triangulation(hairpin_scene)
    u = np.linspace(0.0, 1.0, 100)
    tri_curve = tri_func(u)

    flagged, frac = flag_epipolar_degenerate(
        project_points(hairpin_scene.P1, tri_curve), angle_deg=15.0
    )
    assert frac > 0.0 and flagged.any(), "hairpin span was not flagged"

    proj_func, _, _ = reconstruct_projection_based(
        hairpin_scene.calib, hairpin_scene.left_pts, hairpin_scene.right_pts, tri_func
    )
    proj_curve = proj_func(u)

    tri_flag_err = _max_curve_error(tri_curve[flagged], hairpin_scene.curve3d)
    proj_flag_err = _max_curve_error(proj_curve[flagged], hairpin_scene.curve3d)
    assert proj_flag_err < tri_flag_err, (
        f"projection flagged-span error {proj_flag_err * 1000:.2f} mm not below "
        f"triangulation {tri_flag_err * 1000:.2f} mm"
    )


def test_init_curve_reprojects_near_baseline(smooth_scene) -> None:
    # Step 5a sanity: the LSQ-initialized control points reproject within ~1 px
    # of the baseline spline's own reprojection before any optimization.
    from scipy.interpolate import BSpline

    from dloseg.recon3d.projection_recon import (
        _clamped_uniform_knots,
        _init_control_points,
    )

    tri_func = _init_from_triangulation(smooth_scene)
    u = np.linspace(0.0, 1.0, 100)
    base = tri_func(u)

    knots = _clamped_uniform_knots(40, 3)
    init = np.asarray(tri_func(np.linspace(0.0, 1.0, 160)), dtype=np.float64)
    c0 = _init_control_points(init, knots, 3)
    approx = BSpline(knots, c0, 3)(u)

    proj_base = project_points(smooth_scene.P1, base)
    proj_approx = project_points(smooth_scene.P1, approx)
    assert np.max(np.linalg.norm(proj_base - proj_approx, axis=1)) < 1.0


def test_returns_tck_like_and_evaluable(smooth_scene) -> None:
    tri_func = _init_from_triangulation(smooth_scene)
    spline_func, tck_like, info = reconstruct_projection_based(
        smooth_scene.calib, smooth_scene.left_pts, smooth_scene.right_pts, tri_func
    )
    knots, ctrl, degree = tck_like
    assert degree == 3
    assert ctrl.shape == (40, 3)
    assert knots.shape[0] == 40 + 3 + 1
    out = spline_func(np.linspace(0.0, 1.0, 10))
    assert out.shape == (10, 3)
    assert set(info) >= {"cost", "nfev", "runtime_s", "success"}


def test_reference_mock_stereo_smoke() -> None:
    # The bspline_3d_recon.py __main__ mock pattern still round-trips: a known
    # disparity-shifted stereo pair triangulates to a plausible depth.
    y = np.linspace(200, 800, 10)
    x_l = 800 + 100 * np.sin(y / 200)
    disparity = 60 - (y / 800) * 40
    left = np.vstack([x_l, y]).T
    right = np.vstack([x_l - disparity, y]).T
    fx, cx, cy, baseline = 1050.0, 960.0, 540.0, 0.12
    calib = {
        "P1": np.array([[fx, 0, cx, 0], [0, fx, cy, 0], [0, 0, 1, 0]], dtype=np.float64),
        "P2": np.array(
            [[fx, 0, cx, -fx * baseline], [0, fx, cy, 0], [0, 0, 1, 0]], dtype=np.float64
        ),
    }
    tck, _ = splprep([left[:, 0], left[:, 1]], s=0, k=3)
    _ = splev(0.5, tck)  # spline fit does not raise
    points_3d, _ = triangulate_and_reconstruct(calib, left, right, matching="arclength")
    assert points_3d.shape[1] == 3
    assert np.all(points_3d[:, 2] > 0.0)
