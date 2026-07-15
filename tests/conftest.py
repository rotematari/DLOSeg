"""Shared synthetic-stereo fixtures for the recon3d tests.

The fixtures build a known smooth 3D curve, a mock rectified stereo rig
(P1/P2 with a horizontal baseline, as in `bspline_3d_recon.py`'s __main__),
and the two 2D projections that a perfect segmenter would report. Tests then
reconstruct from those projections and compare against the known ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class StereoScene:
    """A synthetic stereo scene with ground-truth 3D and its two projections."""

    P1: FloatArray
    P2: FloatArray
    calib: dict[str, object]
    curve3d: FloatArray  # (N, 3) ground-truth samples, meters
    left_pts: FloatArray  # (N, 2) left projection, pixels
    right_pts: FloatArray  # (N, 2) right projection, pixels


def _mock_calib(fx: float = 1050.0, baseline: float = 0.12) -> dict[str, object]:
    """Mock rectified ZED-like calibration (matches bspline_3d_recon __main__)."""
    fy, cx, cy = 1050.0, 960.0, 540.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    P1 = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64)
    P2 = np.array([[fx, 0, cx, -fx * baseline], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64)
    return {
        "K1": K,
        "D1": np.zeros(5),
        "K2": K,
        "D2": np.zeros(5),
        "R": np.identity(3),
        "T": np.array([[-baseline], [0], [0]], dtype=np.float64),
        "P1": P1,
        "P2": P2,
        "baseline": baseline,
    }


def _project(P: FloatArray, pts3d: FloatArray) -> FloatArray:
    hom = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
    proj = hom @ P.T
    return proj[:, :2] / proj[:, 2:3]


def _make_scene(curve3d: FloatArray) -> StereoScene:
    calib = _mock_calib()
    P1 = np.asarray(calib["P1"], dtype=np.float64)
    P2 = np.asarray(calib["P2"], dtype=np.float64)
    return StereoScene(
        P1=P1,
        P2=P2,
        calib=calib,
        curve3d=curve3d,
        left_pts=_project(P1, curve3d),
        right_pts=_project(P2, curve3d),
    )


@pytest.fixture
def smooth_scene() -> StereoScene:
    """A smooth, well-conditioned 3D curve (no epipolar-degenerate span).

    A gentle helix-like arc that curves in depth; its image tangent stays well
    away from horizontal, so triangulation and projection should both do well.
    """
    n = 60
    t = np.linspace(0.0, 1.0, n)
    x = 0.10 * np.sin(2.2 * t)
    y = -0.25 + 0.30 * t  # mostly vertical in the image -> well-conditioned
    z = 1.20 + 0.15 * t + 0.05 * np.cos(2.0 * t)
    curve3d = np.column_stack([x, y, z])
    return _make_scene(curve3d)


@pytest.fixture
def hairpin_scene() -> StereoScene:
    """A 3D curve whose image contains a near-horizontal hairpin span.

    The middle third runs horizontally in the image (tangent within ~15 deg of
    the epipolar direction) while curving in depth, so arc-length y-matching is
    ill-conditioned there. Built in the image plane first, then lifted to 3D by
    assigning a smooth depth profile with a bump over the hairpin.
    """
    n = 90
    t = np.linspace(0.0, 1.0, n)
    # Image-plane y stays nearly constant across the middle (horizontal span),
    # rising at the two ends -> a hairpin sitting on its side.
    y_img = -0.18 + 0.16 * np.cos(np.pi * t)  # meters in the world y (image v)
    x = -0.18 + 0.36 * t
    # Depth bumps over the hairpin so the flagged span carries real 3D signal
    # that the triangulation baseline cannot recover from y-matching alone.
    z = 1.10 + 0.18 * np.exp(-((t - 0.5) ** 2) / 0.03)
    curve3d = np.column_stack([x, y_img, z])
    return _make_scene(curve3d)
