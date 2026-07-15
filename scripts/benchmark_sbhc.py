"""Batch DLO spline extraction over the full SBHC dataset.

Iterates over every ground-truth mask in DATASETS/SBHC/{S1,S2,S3}/gt_images,
resizes each to 256x256, and runs the DLOGraph pipeline
(mask -> skeleton -> graph -> prune -> spline fit -> DLO reconstruction ->
B-spline fit) on each image. Reports average per-image processing time / FPS
and collects the paths of images that failed.

The (commented-out) saving block can dump per-wire spline predictions as .npy
files to spline_preds/ and overlay plots to plot_preds/ for later evaluation.

The pipeline itself (`get_spline`) lives in dloseg/graph/pipeline.py. Run:
`python scripts/benchmark_sbhc.py` (dataset paths are resolved from the repo root).
"""

import os
import time

import cv2

from dloseg.graph.pipeline import get_spline

# Repo root (one level up from scripts/) so relative dataset paths work
# no matter where the script is launched from.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


if __name__ == "__main__":
    config = {
        # Graph processing parameters
        "statistic": "mean",
        "min_cluster_factor": 1.0,
        "padding_size": 0,
        "max_prune_length": 5,
        "dilate_iterations": 1,  # Number of iterations for dilation
        "erode_iterations": 1,  # Number of iterations for erosion
        "max_dist_to_connect_leafs": 60,  # Maximum distance to connect leaf nodes
        "max_dist_to_connect_nodes": 5,  # Maximum distance to connect internal nodes
        # Spline fitting parameters
        "spline": {
            "k": 3,  # B-spline degree
            "smoothing": 20,
            "n_points": 200,
            "max_num_points": 50,  # Maximum number of points in the spline
        },
        # Visualization settings
        "on_mask": False,  # Whether to draw the mask as background
        "show_initial_graph": False,
        "show_pruned_graph": False,
        "show_spline_graph": False,
        "show_dlo_graph": False,
        "show_rectification": False,
        "show_final_plots": False,
        "node_size_small": 1,
        "node_size_large": 5,
        "figure_size": (12, 10),
    }

    total_time = 0
    count = 0
    errors = 0
    error_paths = []
    for folder in ["S1", "S2", "S3"]:
        folder_dir = os.path.join(REPO_ROOT, "DATASETS/SBHC", folder)
        wire_count = int(folder[-1])  # S1 -> 1 wire, S2 -> 2 wires, ...
        os.makedirs(os.path.join(folder_dir, "spline_preds"), exist_ok=True)
        os.makedirs(os.path.join(folder_dir, "plot_preds"), exist_ok=True)
        for img_path in os.listdir(os.path.join(folder_dir, "gt_images")):
            if img_path.endswith(".png"):
                config["img_path"] = os.path.join(folder_dir, "gt_images", img_path)
                image = cv2.imread(config["img_path"], cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Could not read image: {config['img_path']}")
                    errors += 1
                    error_paths.append(config["img_path"])
                    continue

                orig_h, orig_w = image.shape
                new_h, new_w = 256, 256
                if orig_h > 256 or orig_w > 256:
                    image = cv2.resize(image, (new_h, new_w), interpolation=cv2.INTER_LINEAR)

                try:
                    start_time = time.time()
                    G = get_spline(image, config=config)
                    total_time += time.time() - start_time
                    count += 1

                    # save the path graph as npy file
                    image = cv2.resize(image, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    # compute scale factors
                    scale_x = orig_w / new_w
                    scale_y = orig_h / new_h

                    # Optional: save per-wire spline predictions and overlay plots
                    # fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
                    # ax1.imshow(image, cmap='gray')
                    # for i, bspline_pts in enumerate(G.full_bsplines):
                    #     bspline_pts_orig = bspline_pts * np.array([scale_x, scale_y]) - np.array([G.padding_size*2, G.padding_size*2])
                    #     ax1.plot(bspline_pts_orig[:, 0], bspline_pts_orig[:, 1], label=f'Full B-spline {i}')
                    #     np.save(os.path.join(folder_dir, 'spline_preds', f'{img_path[:-4]}_{i}.npy'), bspline_pts)
                    # ax1.set_title('Full B-spline')
                    # ax1.axis('equal')
                    # ax1.legend()
                    # plt.savefig(os.path.join(folder_dir, 'plot_preds', f'{img_path[:-4]}_full_bspline.png'))
                    # plt.close(fig)
                except Exception as e:
                    print(f"Error processing image {count}: {e}")
                    errors += 1
                    error_paths.append(config["img_path"])
                    continue

        if count > 0:
            print(
                f"Average time to process image: {total_time / count:.3f} seconds and in FPS: {1 / (total_time / count):.3f} FPS with {count} images processed and {errors} errors"
            )
        print(f"Error paths: {error_paths}")
