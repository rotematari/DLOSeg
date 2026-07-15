"""Shared frame loading for the recon3d demo and evaluation scripts.

Both `scripts/recon3d_demo.py` and `scripts/recon3d_eval.py` need the same
front half of the pipeline: read a frame's left/right wire masks, rectify them
with the ZED factory calibration, and turn each rectified mask into a full-res
2D B-spline. This module owns that logic so the scripts share one loader
instead of importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from dloseg.graph.pipeline import get_spline
from dloseg.recon3d.stereo import get_zed_calibration, rectify_stereo_pair

# recon3d/frame_io.py -> recon3d -> dloseg -> src -> repo root
_PKG_DIR = Path(__file__).resolve().parents[1]  # .../src/dloseg
REPO_ROOT = _PKG_DIR.parents[1]  # .../repo
DEFAULT_MASK_DIR = REPO_ROOT / "outputs" / "seg_data_720_15fps"
DEFAULT_CALIB_PATH = _PKG_DIR / "zed" / "calibration_data" / "zed_2i_cal_data.yaml"

# Graph-pipeline config tuned for the 720p ZED masks (shared by demo + eval so
# both methods see identical 2D input).
DEFAULT_GRAPH_CONFIG: dict[str, Any] = {
    "padding_size": 0,
    "max_prune_length": 5,
    "dilate_iterations": 1,
    "erode_iterations": 1,
    "max_dist_to_connect_leafs": 60,
    "max_dist_to_connect_nodes": 5,
    "spline": {
        "k": 3,
        "smoothing": 20,
        "final_smoothing": None,
        "n_points": 200,
        "max_num_points": 100,
    },
    "on_mask": False,
    "show_initial_graph": False,
    "show_pruned_graph": False,
    "show_spline_graph": False,
    "show_dlo_graph": False,
    "node_size_small": 1,
    "node_size_large": 5,
}


@dataclass
class FrameData:
    """A rectified stereo frame ready for 2D spline extraction.

    Attributes:
        frame_id: e.g. "img_05".
        rect_left / rect_right: rectified left/right masks (H, W), binary.
        P1 / P2: (3, 4) projection matrices of the rectified frame.
        calib_rect: full calibration dict with P1/P2 swapped in for the
            rectified frame (ready for `triangulate_and_reconstruct`).
    """

    frame_id: str
    rect_left: NDArray[np.uint8]
    rect_right: NDArray[np.uint8]
    P1: NDArray[np.float64]
    P2: NDArray[np.float64]
    calib_rect: dict[str, Any]


def mask_to_splines_fullres(
    mask: NDArray[np.uint8], config: dict[str, Any], proc_size: int = 512
) -> list[NDArray[np.float64]]:
    """Run the 2D pipeline on a mask, returning splines in full-res pixel coords.

    proc_size=512 (not 256): rectification resampling pinches thin strands at
    tight knots, and 256px processing then loses them — measured 96% vs 100%
    skeleton coverage on the 720p ZED masks.

    Args:
        mask: (H, W) binary mask.
        config: DLOGraph pipeline config (see `DEFAULT_GRAPH_CONFIG`).
        proc_size: square size the mask is resampled to before graph processing.

    Returns:
        List of (K, 2) spline point arrays in the mask's full-resolution pixels.
    """
    orig_h, orig_w = mask.shape
    mask_proc = cv2.resize(mask, (proc_size, proc_size), interpolation=cv2.INTER_NEAREST)
    graph = get_spline(mask_proc, config=config)
    scale = np.array([orig_w / proc_size, orig_h / proc_size])
    pad = np.array([graph.padding_size * 2, graph.padding_size * 2])
    return [pts * scale - pad for pts in graph.full_bsplines]


def load_frame(
    frame_id: str,
    mask_dir: str | Path = DEFAULT_MASK_DIR,
    calib_path: str | Path = DEFAULT_CALIB_PATH,
    res: str = "720p",
) -> FrameData:
    """Load and rectify one frame's left/right wire masks.

    Args:
        frame_id: frame stem, e.g. "img_05" (masks are
            `{frame_id}_{left,right}_mask_0.png`).
        mask_dir: directory holding the mask PNGs.
        calib_path: ZED calibration YAML.
        res: calibration resolution tag passed to `get_zed_calibration`.

    Returns:
        A `FrameData` with rectified masks and rectified projection matrices.

    Raises:
        FileNotFoundError: if either mask is missing.
    """
    mask_dir = Path(mask_dir)
    left_path = mask_dir / f"{frame_id}_left_mask_0.png"
    right_path = mask_dir / f"{frame_id}_right_mask_0.png"
    mask_l = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    mask_r = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if mask_l is None or mask_r is None:
        raise FileNotFoundError(f"Missing masks for {frame_id} in {mask_dir}")

    calib = get_zed_calibration(str(calib_path), res=res)
    rect_l, rect_r, P1, P2 = rectify_stereo_pair(
        mask_l,
        mask_r,
        calib,
        alpha=0,
        interpolation=cv2.INTER_NEAREST,  # keep masks binary
        verbose=False,
    )
    calib_rect = dict(calib, P1=P1, P2=P2)
    return FrameData(
        frame_id=frame_id,
        rect_left=rect_l,
        rect_right=rect_r,
        P1=np.asarray(P1, dtype=np.float64),
        P2=np.asarray(P2, dtype=np.float64),
        calib_rect=calib_rect,
    )


def _arclen(pts: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def extract_wire_splines(
    frame: FrameData, config: dict[str, Any] = DEFAULT_GRAPH_CONFIG
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract the single-wire 2D B-splines from a rectified frame.

    Single-wire scene: the longest spline on each side is taken as the wire.

    Args:
        frame: a loaded `FrameData`.
        config: DLOGraph pipeline config.

    Returns:
        (left_pts, right_pts), each (K, 2) full-res rectified pixel coords.

    Raises:
        RuntimeError: if the 2D pipeline yields no spline on a view.
    """
    splines_l = mask_to_splines_fullres(frame.rect_left, config)
    splines_r = mask_to_splines_fullres(frame.rect_right, config)
    if not splines_l or not splines_r:
        raise RuntimeError(f"2D pipeline produced no splines on one view of {frame.frame_id}")
    left_pts = max(splines_l, key=_arclen)
    right_pts = max(splines_r, key=_arclen)
    return left_pts, right_pts
