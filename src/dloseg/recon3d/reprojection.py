"""Reprojection diagnostics and curve metrics for 3D DLO reconstruction.

These utilities judge a reconstructed 3D spline against the observed 2D
evidence in a rectified stereo pair, independent of how the 3D curve was
produced (triangulation baseline or projection-based fit):

- `project_points`:            pinhole projection through a 3x4 matrix.
- `point_to_polyline_dist`:    vectorized point-to-segment distance.
- `flag_epipolar_degenerate`:  mark near-horizontal (epipolar-parallel) samples
                               where arc-length y-matching is ill-conditioned.
- `bending_energy_D4`:         sun2024 §IV-A3d smoothness metric.
- `arclength_std`:             segment-length spread (inextensibility proxy).
- `mask_distance_rms`:         RMS distance-transform residual to a mask.
- `reprojection_report`:       per-frame/per-method dataclass of all columns.

All pixel quantities are in the rectified image frame, so the epipolar lines
are horizontal and "epipolar-parallel" means "tangent near horizontal".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def project_points(P: FloatArray, pts3d: FloatArray) -> FloatArray:
    """Project 3D points into the image via a 3x4 projection matrix.

    Args:
        P: (3, 4) projection matrix (rectified P1 or P2).
        pts3d: (N, 3) points in the projection matrix's reference frame (meters).

    Returns:
        (N, 2) pixel coordinates.
    """
    pts3d = np.asarray(pts3d, dtype=np.float64)
    hom = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])  # (N, 4)
    proj = hom @ P.T  # (N, 3)
    return proj[:, :2] / proj[:, 2:3]


def point_to_polyline_dist(query_pts: FloatArray, polyline: FloatArray) -> FloatArray:
    """Distance from each query point to the nearest segment of a polyline.

    Vectorized point-to-segment distance (not point-to-vertex): each query is
    projected onto every segment, clamped to the segment, and the minimum
    distance is returned. This is the correspondence-free data residual: it
    does not assume which polyline vertex a query maps to.

    Args:
        query_pts: (M, 2) query points.
        polyline: (K, 2) ordered polyline vertices, K >= 2.

    Returns:
        (M,) nearest-segment distances.
    """
    query_pts = np.asarray(query_pts, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)
    if polyline.shape[0] < 2:
        raise ValueError("polyline needs at least 2 vertices")

    seg_a = polyline[:-1]  # (S, 2)
    seg_d = polyline[1:] - seg_a  # (S, 2)
    seg_len2 = np.einsum("sd,sd->s", seg_d, seg_d)  # (S,)
    seg_len2 = np.where(seg_len2 > 0.0, seg_len2, 1.0)  # guard zero-length segments

    v = query_pts[:, None, :] - seg_a[None, :, :]  # (M, S, 2)
    t = np.einsum("msd,sd->ms", v, seg_d) / seg_len2[None, :]  # (M, S)
    t = np.clip(t, 0.0, 1.0)
    # Work in squared distance and take sqrt only on the per-query minimum: the
    # inner (M, S) sqrt is the optimizer hotspot, so avoid it S-fold.
    delta = v - t[:, :, None] * seg_d[None, :, :]  # (M, S, 2) foot-to-query offset
    dist2 = np.einsum("msd,msd->ms", delta, delta)  # (M, S)
    return np.sqrt(dist2.min(axis=1))


def _voronoi_arclength_weights(polyline: FloatArray) -> FloatArray:
    """Per-vertex arc-length weight (half of each adjacent segment length)."""
    seg_len = np.linalg.norm(np.diff(polyline, axis=0), axis=1)  # (K-1,)
    w = np.zeros(polyline.shape[0], dtype=np.float64)
    w[:-1] += 0.5 * seg_len
    w[1:] += 0.5 * seg_len
    return w


def flag_epipolar_degenerate(
    polyline: FloatArray, angle_deg: float = 15.0
) -> tuple[NDArray[np.bool_], float]:
    """Flag samples whose tangent is within `angle_deg` of horizontal.

    On a rectified image the epipolar lines are horizontal, so a near-horizontal
    tangent means many x-positions share one y: the arc-length y-matching used
    by the triangulation baseline is ill-conditioned there (hairpins, sagging
    spans). Tangents are central differences along the polyline.

    Args:
        polyline: (K, 2) ordered 2D samples in rectified pixel coords, K >= 2.
        angle_deg: threshold; a sample is flagged when |atan2(|dy|, |dx|)| is
            below this many degrees.

    Returns:
        Tuple of (per-sample boolean mask of shape (K,), arc-length fraction
        flagged in [0, 1]).
    """
    polyline = np.asarray(polyline, dtype=np.float64)
    dx = np.gradient(polyline[:, 0])
    dy = np.gradient(polyline[:, 1])
    tangent_deg = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))
    flagged = tangent_deg < angle_deg

    w = _voronoi_arclength_weights(polyline)
    total = w.sum()
    frac = float(w[flagged].sum() / total) if total > 0.0 else 0.0
    return flagged, frac


def bending_energy_D4(pts3d: FloatArray) -> float:
    """Discrete bending metric D4 from sun2024 §IV-A3d.

    D4 = sum_i angle(seg_i, seg_{i+1})^2 over consecutive segments, with
    angle the turning angle between adjacent segment vectors (radians). Lower
    is smoother. Zero for a straight polyline.

    Args:
        pts3d: (N, 3) ordered curve samples, N >= 3.

    Returns:
        Sum of squared inter-segment turning angles (rad^2).
    """
    pts3d = np.asarray(pts3d, dtype=np.float64)
    if pts3d.shape[0] < 3:
        return 0.0
    seg = np.diff(pts3d, axis=0)  # (N-1, 3)
    seg_norm = np.linalg.norm(seg, axis=1, keepdims=True)
    seg_unit = seg / np.where(seg_norm > 0.0, seg_norm, 1.0)
    cos = np.einsum("id,id->i", seg_unit[:-1], seg_unit[1:])  # (N-2,)
    angles = np.arccos(np.clip(cos, -1.0, 1.0))
    return float(np.sum(angles**2))


def arclength_std(pts3d: FloatArray) -> float:
    """Standard deviation of segment lengths along a curve.

    Zero when samples are equidistant. A proxy for how well an inextensible
    (uniform-segment) prior is satisfied. Returned in the input distance unit.

    Args:
        pts3d: (N, 3) ordered curve samples, N >= 2.

    Returns:
        Population standard deviation of the N-1 segment lengths.
    """
    pts3d = np.asarray(pts3d, dtype=np.float64)
    seg_len = np.linalg.norm(np.diff(pts3d, axis=0), axis=1)
    return float(np.std(seg_len))


def mask_distance_rms(mask: NDArray[np.uint8], pts2d: FloatArray) -> float:
    """RMS distance from projected points to the wire mask (diagnostic only).

    Builds the distance transform of the inverted mask (zero on the wire,
    growing into the background) and samples it at the projected pixel
    locations. This is a diagnostic column, never an optimizer term.

    Args:
        mask: (H, W) binary/grayscale rectified mask; nonzero = wire.
        pts2d: (N, 2) projected pixel coordinates.

    Returns:
        RMS distance-transform value (pixels) at the projected points; points
        outside the image are ignored.
    """
    h, w = mask.shape[:2]
    background = (mask == 0).astype(np.uint8)  # 1 where background, 0 on the wire
    dist_field = cv2.distanceTransform(background, cv2.DIST_L2, 3)

    cols = np.round(pts2d[:, 0]).astype(int)
    rows = np.round(pts2d[:, 1]).astype(int)
    inside = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    if not np.any(inside):
        return float("nan")
    sampled = dist_field[rows[inside], cols[inside]]
    return float(np.sqrt(np.mean(sampled.astype(np.float64) ** 2)))


@dataclass
class ReprojectionReport:
    """Per-frame, per-method reprojection diagnostics (one CSV row).

    Distances are pixels unless noted. `flagged` metrics restrict to samples
    marked epipolar-degenerate in the left view (`flag_epipolar_degenerate`).
    """

    frame: str
    method: str
    rms_reproj_L_px: float
    rms_reproj_R_px: float
    max_reproj_L_px: float
    max_reproj_R_px: float
    rms_reproj_flagged_px: float
    mask_rms_px: float
    bending_D4: float
    arclen_std_mm: float
    flagged_frac: float
    runtime_s: float

    def as_row(self) -> dict[str, object]:
        """Return the report as a plain dict for CSV writing."""
        return asdict(self)


def _rms(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def reprojection_report(
    frame: str,
    method: str,
    curve3d: FloatArray,
    P1: FloatArray,
    P2: FloatArray,
    left_poly: FloatArray,
    right_poly: FloatArray,
    runtime_s: float,
    mask_left: NDArray[np.uint8] | None = None,
    mask_right: NDArray[np.uint8] | None = None,
    angle_deg: float = 15.0,
) -> ReprojectionReport:
    """Score a sampled 3D curve against both observed 2D polylines.

    The curve is projected into each rectified view; residuals are nearest-
    segment distances to the observed polyline. Epipolar-degenerate samples are
    flagged from the left-view projection tangent (both curves trace the same
    wire, so the flagged span is the physically hard span for either method).

    Args:
        frame: frame id (e.g. "img_05").
        method: method label (e.g. "triangulation" / "projection").
        curve3d: (N, 3) uniform-u samples of the reconstructed 3D curve (meters).
        P1: (3, 4) rectified left projection matrix.
        P2: (3, 4) rectified right projection matrix.
        left_poly: (K1, 2) observed left 2D polyline (rectified pixels).
        right_poly: (K2, 2) observed right 2D polyline (rectified pixels).
        runtime_s: wall-clock seconds spent producing `curve3d`.
        mask_left / mask_right: optional rectified masks for the mask-distance
            diagnostic column; omitted -> NaN.
        angle_deg: epipolar-degenerate tangent threshold.

    Returns:
        A populated `ReprojectionReport`. `bending_D4` and `arclen_std_mm` are
        computed on `curve3d` directly, so callers pass a fixed sample count
        (100) for a matched-density comparison between methods.
    """
    curve3d = np.asarray(curve3d, dtype=np.float64)
    proj_l = project_points(P1, curve3d)
    proj_r = project_points(P2, curve3d)

    res_l = point_to_polyline_dist(proj_l, left_poly)
    res_r = point_to_polyline_dist(proj_r, right_poly)

    flagged, frac = flag_epipolar_degenerate(proj_l, angle_deg=angle_deg)
    flagged_res = np.concatenate([res_l[flagged], res_r[flagged]])

    mask_rms = float("nan")
    if mask_left is not None and mask_right is not None:
        rms_l = mask_distance_rms(mask_left, proj_l)
        rms_r = mask_distance_rms(mask_right, proj_r)
        mask_rms = float(np.sqrt(np.nanmean([rms_l**2, rms_r**2])))

    return ReprojectionReport(
        frame=frame,
        method=method,
        rms_reproj_L_px=_rms(res_l),
        rms_reproj_R_px=_rms(res_r),
        max_reproj_L_px=float(res_l.max()),
        max_reproj_R_px=float(res_r.max()),
        rms_reproj_flagged_px=_rms(flagged_res),
        mask_rms_px=mask_rms,
        bending_D4=bending_energy_D4(curve3d),
        arclen_std_mm=arclength_std(curve3d) * 1000.0,
        flagged_frac=frac,
        runtime_s=runtime_s,
    )
