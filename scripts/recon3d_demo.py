"""3D DLO reconstruction from a real ZED stereo frame.

End-to-end demo of the recon3d pipeline on captured data:

1. Load + rectify the left/right wire masks for one frame (dloseg.recon3d.frame_io).
2. Run the 2D graph pipeline on each rectified mask -> 2D B-splines.
3. Reconstruct the 3D spline by triangulation and save figures.

Outputs figures to outputs/recon3d/. Headless-safe:
    MPLBACKEND=Agg uv run scripts/recon3d_demo.py [frame] [smoothing_mm]
    # e.g.: ... recon3d_demo.py img_05 5
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from dloseg.recon3d.bspline_3d_recon import triangulate_and_reconstruct, visualize_reconstruction
from dloseg.recon3d.frame_io import REPO_ROOT, extract_wire_splines, load_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", nargs="?", default="img_01", help="frame stem, e.g. img_05")
    parser.add_argument(
        "smoothing_mm",
        nargs="?",
        type=float,
        default=3.0,
        help="final 3D spline smoothing budget (RMS mm per triangulated point)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frame = args.frame
    out_dir = os.path.join(REPO_ROOT, "outputs", "recon3d")
    os.makedirs(out_dir, exist_ok=True)

    frame_data = load_frame(frame)
    left_pts, right_pts = extract_wire_splines(frame_data)
    print(f"{frame}: left/right 2D splines extracted ({len(left_pts)}, {len(right_pts)} pts)")

    # 2D sanity figure: splines over the rectified masks
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, m, pts, name in [
        (axes[0], frame_data.rect_left, left_pts, "left"),
        (axes[1], frame_data.rect_right, right_pts, "right"),
    ]:
        ax.imshow(m, cmap="gray")
        ax.plot(pts[:, 0], pts[:, 1], "r-", lw=2)
        ax.set_title(f"{frame} {name} (rectified) + 2D B-spline")
        ax.axis("off")
    fig.savefig(os.path.join(out_dir, f"{frame}_2d_splines.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Triangulation ---------------------------------------------------------
    points_3d, spline_3d_func = triangulate_and_reconstruct(
        frame_data.calib_rect,
        left_pts,
        right_pts,
        z_range=(0.3, 5.0),  # plausible tabletop depth window for the ZED
        smoothing_mm=args.smoothing_mm,
    )
    z = points_3d[:, 2]
    print(
        f"Triangulated {len(points_3d)} points; depth Z in [{z.min():.3f}, {z.max():.3f}] m "
        f"(median {np.median(z):.3f} m)"
    )

    visualize_reconstruction(
        left_pts,
        right_pts,
        points_3d,
        spline_3d_func,
        save_path=os.path.join(out_dir, f"{frame}_3d_recon.png"),
    )
    np.save(os.path.join(out_dir, f"{frame}_points3d.npy"), points_3d)

    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
