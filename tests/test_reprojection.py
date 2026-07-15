"""Unit tests for the reprojection diagnostics and curve metrics."""

from __future__ import annotations

import numpy as np

from dloseg.recon3d.reprojection import (
    arclength_std,
    bending_energy_D4,
    flag_epipolar_degenerate,
    point_to_polyline_dist,
    project_points,
    reprojection_report,
)


def test_project_points_matches_manual_pinhole() -> None:
    fx, fy, cx, cy = 1000.0, 1000.0, 640.0, 360.0
    P = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64)
    pts3d = np.array([[0.0, 0.0, 2.0], [0.1, -0.2, 4.0]], dtype=np.float64)
    got = project_points(P, pts3d)
    # u = fx * X/Z + cx, v = fy * Y/Z + cy
    expected = np.array(
        [
            [cx, cy],
            [fx * 0.1 / 4.0 + cx, fy * -0.2 / 4.0 + cy],
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


def test_point_to_polyline_dist_perpendicular_and_endpoint() -> None:
    polyline = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    query = np.array(
        [
            [5.0, 3.0],  # perpendicular foot inside segment -> 3.0
            [-4.0, 0.0],  # beyond the start endpoint -> 4.0
            [10.0, 0.0],  # on the far endpoint -> 0.0
        ]
    )
    got = point_to_polyline_dist(query, polyline)
    np.testing.assert_allclose(got, [3.0, 4.0, 0.0], atol=1e-9)


def test_point_to_polyline_dist_uses_nearest_segment() -> None:
    # An L-shaped polyline: nearest segment differs per query.
    polyline = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    query = np.array([[10.0, 5.0], [5.0, -2.0]])
    got = point_to_polyline_dist(query, polyline)
    np.testing.assert_allclose(got, [0.0, 2.0], atol=1e-9)


def test_flag_epipolar_degenerate_horizontal_flags_vertical_does_not() -> None:
    horiz = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])
    vert = np.column_stack([np.zeros(20), np.linspace(0, 100, 20)])

    flag_h, frac_h = flag_epipolar_degenerate(horiz, angle_deg=15.0)
    flag_v, frac_v = flag_epipolar_degenerate(vert, angle_deg=15.0)

    assert flag_h.all()
    assert frac_h == 1.0
    assert not flag_v.any()
    assert frac_v == 0.0


def test_flag_epipolar_degenerate_partial_span() -> None:
    # Diagonal ends with a horizontal middle -> only the middle is flagged.
    left = np.column_stack([np.linspace(0, 30, 15), np.linspace(0, 30, 15)])
    mid = np.column_stack([np.linspace(30, 70, 20), np.full(20, 30.0)])
    right = np.column_stack([np.linspace(70, 100, 15), np.linspace(30, 60, 15)])
    poly = np.vstack([left, mid, right])
    flagged, frac = flag_epipolar_degenerate(poly, angle_deg=15.0)
    assert flagged.any() and not flagged.all()
    assert 0.0 < frac < 1.0


def test_bending_energy_zero_for_straight_line() -> None:
    line = np.column_stack([np.linspace(0, 1, 25), np.zeros(25), np.zeros(25)])
    assert bending_energy_D4(line) == 0.0


def test_bending_energy_right_angle() -> None:
    # Two segments meeting at 90 deg -> one turning angle of pi/2.
    pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0]], dtype=np.float64)
    assert bending_energy_D4(pts) == float((np.pi / 2) ** 2)


def test_arclength_std_zero_for_uniform_samples() -> None:
    uniform = np.column_stack([np.linspace(0, 1, 40), np.zeros(40), np.zeros(40)])
    assert arclength_std(uniform) < 1e-12


def test_arclength_std_positive_for_nonuniform() -> None:
    x = np.array([0.0, 0.1, 0.5, 0.55, 1.0])
    pts = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    assert arclength_std(pts) > 1e-6


def test_reprojection_report_zero_residual_on_ground_truth(smooth_scene) -> None:
    # Feeding the true 3D curve back through its own projections must yield
    # near-zero reprojection residuals.
    report = reprojection_report(
        frame="synthetic",
        method="ground_truth",
        curve3d=smooth_scene.curve3d,
        P1=smooth_scene.P1,
        P2=smooth_scene.P2,
        left_poly=smooth_scene.left_pts,
        right_poly=smooth_scene.right_pts,
        runtime_s=0.0,
    )
    assert report.rms_reproj_L_px < 1e-6
    assert report.rms_reproj_R_px < 1e-6
    assert report.arclen_std_mm >= 0.0
