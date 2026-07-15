"""3D DLO reconstruction from a real ZED stereo frame.

End-to-end demo of the recon3d pipeline on captured data:

1. Load the left/right wire masks for one frame
   (outputs/seg_data_720_15fps/img_NN_{left,right}_mask_0.png).
2. Parse the ZED factory calibration and rectify both masks
   (dloseg.recon3d.stereo) — the returned P1/P2 are used for triangulation.
3. Run the 2D graph pipeline (dloseg.graph) on each mask -> 2D B-splines.
4. Epipolar-match + triangulate (dloseg.recon3d.bspline_3d_recon) -> 3D spline.

Outputs figures to outputs/recon3d/. Headless-safe:
    MPLBACKEND=Agg uv run scripts/recon3d_demo.py
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dloseg.graph.pipeline import get_spline
from dloseg.recon3d.stereo import get_zed_calibration, rectify_stereo_pair
from dloseg.recon3d.bspline_3d_recon import triangulate_and_reconstruct, visualize_reconstruction

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

FRAME = 'img_04'

config = {
    # Graph processing parameters (tuned for 256x256 masks)
    'padding_size': 0,
    'max_prune_length': 5,
    'dilate_iterations': 1,
    'erode_iterations': 1,
    'max_dist_to_connect_leafs': 60,
    'max_dist_to_connect_nodes': 5,

    # Spline fitting parameters
    'spline': {
        'k': 3,
        'smoothing': 20,
        'n_points': 200,
        'max_num_points': 100,
    },

    # Visualization settings (off — we save our own figures)
    'on_mask': False,
    'show_initial_graph': False,
    'show_pruned_graph': False,
    'show_spline_graph': False,
    'show_dlo_graph': False,
    'node_size_small': 1,
    'node_size_large': 5,
}


def mask_to_splines_fullres(mask, config, proc_size=256):
    """Run the 2D pipeline on a mask, returning splines in full-res pixel coords."""
    orig_h, orig_w = mask.shape
    mask_proc = cv2.resize(mask, (proc_size, proc_size), interpolation=cv2.INTER_NEAREST)
    G = get_spline(mask_proc, config=config)
    scale = np.array([orig_w / proc_size, orig_h / proc_size])
    pad = np.array([G.padding_size * 2, G.padding_size * 2])
    return [pts * scale - pad for pts in G.full_bsplines]


if __name__ == '__main__':
    out_dir = os.path.join(REPO_ROOT, 'outputs', 'recon3d')
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Load masks -------------------------------------------------------
    mask_dir = os.path.join(REPO_ROOT, 'outputs', 'seg_data_720_15fps')
    mask_l = cv2.imread(os.path.join(mask_dir, f'{FRAME}_left_mask_0.png'), cv2.IMREAD_GRAYSCALE)
    mask_r = cv2.imread(os.path.join(mask_dir, f'{FRAME}_right_mask_0.png'), cv2.IMREAD_GRAYSCALE)
    if mask_l is None or mask_r is None:
        raise FileNotFoundError(f"Missing masks for {FRAME} in {mask_dir}")

    # --- 2. Rectify with the ZED factory calibration -------------------------
    calib_path = os.path.join(REPO_ROOT, 'src/dloseg/zed/calibration_data/zed_2i_cal_data.yaml')
    calib = get_zed_calibration(calib_path, res='720p')
    rect_l, rect_r, P1, P2 = rectify_stereo_pair(
        mask_l, mask_r, calib, alpha=0,
        interpolation=cv2.INTER_NEAREST,  # keep masks binary
        verbose=False,
    )
    # Triangulation must use the projection matrices of the rectified frame
    calib_rect = dict(calib, P1=P1, P2=P2)

    # --- 3. Masks -> 2D B-splines --------------------------------------------
    splines_l = mask_to_splines_fullres(rect_l, config)
    splines_r = mask_to_splines_fullres(rect_r, config)
    print(f"Left: {len(splines_l)} spline(s), Right: {len(splines_r)} spline(s)")
    if not splines_l or not splines_r:
        raise RuntimeError("2D pipeline produced no splines on one of the views")

    # Single-wire scene: take the longest spline on each side
    def arclen(p):
        return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    left_pts = max(splines_l, key=arclen)
    right_pts = max(splines_r, key=arclen)

    # 2D sanity figure: splines over the rectified masks
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, m, pts, name in [(axes[0], rect_l, left_pts, 'left'), (axes[1], rect_r, right_pts, 'right')]:
        ax.imshow(m, cmap='gray')
        ax.plot(pts[:, 0], pts[:, 1], 'r-', lw=2)
        ax.set_title(f'{FRAME} {name} (rectified) + 2D B-spline')
        ax.axis('off')
    fig.savefig(os.path.join(out_dir, f'{FRAME}_2d_splines.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 4. Triangulate -> 3D spline ------------------------------------------
    points_3d, spline_3d_func = triangulate_and_reconstruct(
        calib_rect, left_pts, right_pts,
        z_range=(0.3, 5.0),  # plausible tabletop depth window for the ZED
    )

    z = points_3d[:, 2]
    print(f"Triangulated {len(points_3d)} points; depth Z in [{z.min():.3f}, {z.max():.3f}] m "
          f"(median {np.median(z):.3f} m)")

    visualize_reconstruction(left_pts, right_pts, points_3d, spline_3d_func,
                             save_path=os.path.join(out_dir, f'{FRAME}_3d_recon.png'))

    # Save the 3D points for downstream use
    np.save(os.path.join(out_dir, f'{FRAME}_points3d.npy'), points_3d)
    print(f"Results in {out_dir}")
