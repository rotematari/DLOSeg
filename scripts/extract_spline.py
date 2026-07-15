"""Single-image DLO spline extraction demo.

Loads one binary segmentation mask (left image of a stereo pair), resizes it
to 256x256, and runs the full DLOGraph pipeline:
mask -> skeleton -> graph -> prune -> per-branch spline fit -> DLO
reconstruction -> global B-spline fit. The resulting B-splines are rescaled
back to the original resolution and plotted next to the real RGB image.

The 2D pipeline lives in dloseg/graph/pipeline.py; stereo helpers
(get_zed_calibration, rectify_stereo_pair, ...) in dloseg/recon3d/stereo.py.

Usage: edit the `config` dict in __main__ (mask paths, spline params,
visualization flags), then run `python scripts/extract_spline.py`.
"""
import os
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

from dloseg.graph.pipeline import get_spline

# Repo root (one level up from scripts/) so relative dataset paths work
# no matter where the script is launched from.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if __name__ == '__main__':

    config = {
        # File paths (relative to the repo root)
        'mask_l_path': os.path.join(REPO_ROOT, 'outputs/seg_data_720_15fps/img_04_right_mask_0.png'),
        'mask_r_path': os.path.join(REPO_ROOT, 'outputs/seg_data_720_15fps/img_04_left_mask_0.png'),
        'img_real_path': os.path.join(REPO_ROOT, 'DATASETS/data_720_15fps/img_04.png'),
        'zed_calib_path': os.path.join(REPO_ROOT, 'src/dloseg/zed/calibration_data/zed_2i_cal_data.yaml'),

        # ZED calibration settings
        'zed': {
            'resolution': '720p'
        },

        # Rectification settings
        'rectification': {
            'alpha': 0
        },

        # Graph processing parameters
        'statistic': 'mean',
        'min_cluster_factor': 1.0,
        'padding_size': 0,
        'max_prune_length': 5,
        'dilate_iterations': 1,  # Number of iterations for dilation
        'erode_iterations': 1,  # Number of iterations for erosion
        'max_dist_to_connect_leafs': 60,  # Maximum distance to connect leaf nodes
        'max_dist_to_connect_nodes': 5,  # Maximum distance to connect internal nodes

        # Spline fitting parameters
        'spline': {
            'k': 3,  # B-spline degree
            'smoothing': 20,
            'n_points': 500,
            'max_num_points': 500  # Maximum number of points in the spline
        },

        # Visualization settings
        'on_mask': False,  # Whether to draw the mask as background
        'show_initial_graph': True,
        'show_pruned_graph': True,
        'show_spline_graph': True,
        'show_dlo_graph': True,

        'show_rectification': True,
        'show_final_plots': False,
        'node_size_small': 1,
        'node_size_large': 5,
        'figure_size': (12, 10),

        # Processing settings
        'processing': {
            'process_right_image': False  # Set to True to process right image as well
        }
    }

    # For stereo processing (helpers available in dloseg.recon3d.stereo):
    # from dloseg.recon3d.stereo import get_zed_calibration, get_images_from_stereo_pair, rectify_stereo_pair, visualize_rectification
    # zed_calibration = get_zed_calibration(config['zed_calib_path'], config['zed']['resolution'])
    # img_real_l, img_real_r = get_images_from_stereo_pair(config['mask_l_path'], config['mask_r_path'])
    # rect_left, rect_right = rectify_stereo_pair(img_real_l, img_real_r, zed_calibration, alpha=0)
    # visualize_rectification(img_real_l, img_real_r, rect_left, rect_right)

    rect_left = cv2.imread(config['mask_l_path'], cv2.IMREAD_GRAYSCALE)
    if rect_left is None:
        raise FileNotFoundError(f"Could not read mask image: {config['mask_l_path']}")

    plt.figure()
    plt.imshow(rect_left, cmap='gray')
    plt.title('Rectified Left Image')
    plt.pause(0.1)

    # resize image to 256
    orig_h, orig_w = rect_left.shape
    new_h, new_w = 256, 256
    if orig_h > 256 or orig_w > 256:
        print(f"Resizing images from {orig_h}x{orig_w} to 256x256")
        rect_left = cv2.resize(rect_left, (new_h, new_w), interpolation=cv2.INTER_LINEAR)
    else:
        print("Images are already small enough, no resizing needed")

    print("-----------left-------------\n")
    start_total = time.time()
    G_left = get_spline(rect_left, config=config, verbose=True)
    print(f"Time to process left image: {time.time() - start_total:.3f} seconds")

    # resize to original size
    rect_left = cv2.resize(rect_left, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    # compute scale factors
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    # plot full_bsplines over the mask and next to the real image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
    ax1.imshow(rect_left, cmap='gray')
    for i, bspline_pts in enumerate(G_left.full_bsplines):
        bspline_pts_orig = bspline_pts * np.array([scale_x, scale_y]) - np.array([G_left.padding_size*2, G_left.padding_size*2])
        ax1.plot(bspline_pts_orig[:, 0], bspline_pts_orig[:, 1], label=f'B-spline {i}')
    ax1.axis('off')
    ax1.legend(loc="upper right", fontsize=18)

    ax2.axis('off')
    real_image = cv2.imread(os.path.join(REPO_ROOT, 'DATASETS/SBHC/S3/images/img83.jpg'), cv2.IMREAD_COLOR_RGB)
    ax2.imshow(real_image)
    plt.tight_layout()
    plt.show()
