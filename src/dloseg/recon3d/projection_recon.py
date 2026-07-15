"""Correspondence-free projection-based 3D DLO reconstruction.

Instead of matching left<->right points and triangulating, this optimizes the
3D control points of a cubic B-spline so that its projection fits BOTH observed
2D splines simultaneously, with no explicit stereo correspondence. Depth on
epipolar-parallel (near-horizontal) spans, where sliding along the observed
polyline is free, is pinned by a Discrete-Elastic-Rod prior bridging in from
the well-conditioned neighbors.

Energy terms (all as least-squares residuals):

- Data (per view): distance from N_s projected curve samples to the nearest
  segment of the observed 2D polyline. Correspondence-free by construction.
- Coverage (reverse): distance from subsampled observed points to the nearest
  segment of the projected curve, so the curve cannot shrink onto a sub-span.
- Bending (sun2024 Eq. 2, zero rest curvature): kappa_i = 2*tan(psi_i/2),
  residual_i = sqrt(w_b / l_i) * kappa_i.
- Inextensibility (sun2024 Eq. 1): residual_i = sqrt(w_s) * (|s_i| - s_bar)/s_bar
  with s_bar the initializer arc length divided by N_s - 1.

Material constants (Young's modulus, radius) are absorbed into the single knobs
w_b / w_s, exactly as sun2024 treats them (§III-C, footnote 2). Final tuned
defaults (synthetic fixture first, then the 13 real frames): w_b=0.02, w_s=5.0,
w_cov=1.0, n_ctrl=40, N_s=120, 15 deg flag threshold, 2% end trim. Under these,
projection reprojection matches or beats triangulation on every non-degenerate
frame and gives strictly lower D4 bending and arc-length spread.

n_ctrl=40 (above the spec's nominal 25) gives the data term enough freedom to
nearly interpolate both views, so its reprojection matches or beats the
triangulation baseline even on well-conditioned frames; the DER prior then buys
lower bending and arc-length spread almost for free, because the nearest-segment
data term is invariant to how samples are distributed along the curve. N_s=120
(below the nominal 150) keeps the finite-difference least_squares within the
per-frame runtime budget (each major iteration costs 3*n_ctrl+1 residual evals).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.optimize import least_squares

from dloseg.recon3d.reprojection import point_to_polyline_dist, project_points

FloatArray = NDArray[np.float64]

# Tuned defaults (see module docstring).
DEFAULT_W_B = 0.02  # bending weight
DEFAULT_W_S = 5.0  # inextensibility weight
DEFAULT_W_COV = 1.0  # coverage weight
DEFAULT_N_CTRL = 40
DEFAULT_N_SAMPLES = 120
DEFAULT_N_COVERAGE = 40
DEFAULT_END_TRIM = 0.02  # fraction of observed arc length trimmed at each end
# max_nfev bounds MAJOR iterations; each triggers a finite-difference Jacobian
# of 3*n_ctrl+1 extra residual evals, so keep it modest for the runtime budget.
DEFAULT_MAX_NFEV = 60
DEFAULT_TOL = 1e-4  # ftol/xtol; research-grade, stops well before machine tol


def _clamped_uniform_knots(n_ctrl: int, k: int = 3) -> FloatArray:
    """Clamped-uniform knot vector for `n_ctrl` control points, degree `k`."""
    n_interior = n_ctrl - k - 1
    if n_interior < 0:
        raise ValueError(f"n_ctrl={n_ctrl} too small for degree {k}")
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    return np.concatenate([np.zeros(k + 1), interior, np.ones(k + 1)])


def _init_control_points(init_curve: FloatArray, knots: FloatArray, k: int = 3) -> FloatArray:
    """LSQ-fit control points to samples of the initializer curve."""
    m = init_curve.shape[0]
    u = np.linspace(0.0, 1.0, m)
    spline = make_lsq_spline(u, init_curve, knots, k=k)
    return np.asarray(spline.c, dtype=np.float64)


def _trim_polyline(polyline: FloatArray, trim: float) -> FloatArray:
    """Drop the outer `trim` fraction of arc length at each end of a polyline."""
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0.0:
        return polyline
    keep = (s >= trim * total) & (s <= (1.0 - trim) * total)
    return polyline[keep] if keep.sum() >= 2 else polyline


def _bending_residual(curve: FloatArray, w_b: float) -> FloatArray:
    """DER bending residual sqrt(w_b / l_i) * kappa_i at interior vertices."""
    seg = np.diff(curve, axis=0)  # (N-1, 3)
    seg_len = np.linalg.norm(seg, axis=1)
    seg_unit = seg / np.where(seg_len[:, None] > 0.0, seg_len[:, None], 1.0)
    cos = np.einsum("id,id->i", seg_unit[:-1], seg_unit[1:])  # (N-2,)
    psi = np.arccos(np.clip(cos, -1.0, 1.0))
    kappa = 2.0 * np.tan(np.clip(psi, 0.0, np.pi - 1e-3) / 2.0)  # zero rest curvature
    l_vor = 0.5 * (seg_len[:-1] + seg_len[1:])  # Voronoi (integrated) length at vertex i
    l_vor = np.where(l_vor > 0.0, l_vor, 1.0)
    return np.sqrt(w_b / l_vor) * kappa


def _inextensibility_residual(curve: FloatArray, s_bar: float, w_s: float) -> FloatArray:
    """DER stretch residual sqrt(w_s) * (|s_i| - s_bar) / s_bar."""
    seg_len = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    return np.sqrt(w_s) * (seg_len - s_bar) / s_bar


def reconstruct_projection_based(
    calib_rect: dict[str, Any],
    left_pts: FloatArray,
    right_pts: FloatArray,
    init_spline_3d,
    n_ctrl: int = DEFAULT_N_CTRL,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_coverage: int = DEFAULT_N_COVERAGE,
    w_b: float = DEFAULT_W_B,
    w_s: float = DEFAULT_W_S,
    w_cov: float = DEFAULT_W_COV,
    end_trim: float = DEFAULT_END_TRIM,
    max_nfev: int = DEFAULT_MAX_NFEV,
    tol: float = DEFAULT_TOL,
) -> tuple[Any, tuple[FloatArray, FloatArray, int], dict[str, Any]]:
    """Fit a 3D B-spline whose projections match both observed 2D splines.

    Args:
        calib_rect: calibration dict with rectified "P1"/"P2" (3, 4) matrices.
        left_pts: (K1, 2) observed left 2D polyline (rectified pixels).
        right_pts: (K2, 2) observed right 2D polyline (rectified pixels).
        init_spline_3d: callable u -> (M, 3), the triangulation baseline spline
            used to initialize the control points and the rest length.
        n_ctrl: number of 3D control points (decision vars = 3 * n_ctrl).
        n_samples: curve samples for the data / bending / stretch terms.
        n_coverage: observed points per view for the coverage term.
        w_b, w_s, w_cov: bending / inextensibility / coverage weights.
        end_trim: fraction of observed arc length trimmed at each end (tolerates
            differing mask clipping between views).
        max_nfev: cap on `least_squares` function evaluations (runtime budget).

    Returns:
        Tuple of:
        - spline_3d_func: callable u -> (N, 3) evaluating the fitted curve.
        - tck_like: (knots, control_points (n_ctrl, 3), degree) mirroring scipy.
        - info: dict with cost, nfev, runtime_s, success, status, message.
    """
    import time

    k = 3
    P1 = np.asarray(calib_rect["P1"], dtype=np.float64)
    P2 = np.asarray(calib_rect["P2"], dtype=np.float64)
    left_pts = np.asarray(left_pts, dtype=np.float64)
    right_pts = np.asarray(right_pts, dtype=np.float64)

    knots = _clamped_uniform_knots(n_ctrl, k)
    u_init = np.linspace(0.0, 1.0, max(4 * n_ctrl, 100))
    init_curve = np.asarray(init_spline_3d(u_init), dtype=np.float64)
    c0 = _init_control_points(init_curve, knots, k)  # (n_ctrl, 3)

    init_arclen = float(np.linalg.norm(np.diff(init_curve, axis=0), axis=1).sum())
    s_bar = init_arclen / (n_samples - 1)

    u_data = np.linspace(0.0, 1.0, n_samples)
    left_cov = _trim_polyline(left_pts, end_trim)
    right_cov = _trim_polyline(right_pts, end_trim)
    left_cov = left_cov[np.linspace(0, len(left_cov) - 1, n_coverage).astype(int)]
    right_cov = right_cov[np.linspace(0, len(right_cov) - 1, n_coverage).astype(int)]

    def curve_from_params(x: FloatArray) -> FloatArray:
        c = x.reshape(n_ctrl, 3)
        return np.asarray(BSpline(knots, c, k)(u_data), dtype=np.float64)

    def residuals(x: FloatArray) -> FloatArray:
        curve = curve_from_params(x)
        proj_l = project_points(P1, curve)
        proj_r = project_points(P2, curve)

        data_l = point_to_polyline_dist(proj_l, left_pts)
        data_r = point_to_polyline_dist(proj_r, right_pts)
        cov_l = np.sqrt(w_cov) * point_to_polyline_dist(left_cov, proj_l)
        cov_r = np.sqrt(w_cov) * point_to_polyline_dist(right_cov, proj_r)
        bend = _bending_residual(curve, w_b)
        stretch = _inextensibility_residual(curve, s_bar, w_s)
        return np.concatenate([data_l, data_r, cov_l, cov_r, bend, stretch])

    t0 = time.perf_counter()
    result = least_squares(
        residuals,
        c0.ravel(),
        method="trf",
        x_scale="jac",
        ftol=tol,
        xtol=tol,
        gtol=tol,
        max_nfev=max_nfev,
    )
    runtime_s = time.perf_counter() - t0

    c_opt = result.x.reshape(n_ctrl, 3)
    spline = BSpline(knots, c_opt, k)

    def spline_3d_func(u: FloatArray) -> FloatArray:
        return np.asarray(spline(u), dtype=np.float64)

    info: dict[str, Any] = {
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "runtime_s": runtime_s,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "optimality": float(result.optimality),
        "init_arclen_m": init_arclen,
    }
    return spline_3d_func, (knots, c_opt, k), info
