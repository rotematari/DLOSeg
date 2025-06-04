import cv2
import matplotlib.pyplot as plt
import time
from graph.dlo_graph import DLOGraph  # Assuming you place the class in graph/dlo_graph.py
import logging
import numpy as np


if __name__ == '__main__':
    # Start timing
    start_total = time.time()
    read_time = time.time()
    # Load the mask
    mask_l_path = '/home/admina/segmetation/DLOSeg/outputs/seg_data_720_15fps/img_06_left_mask_0.png'
    mask_l = cv2.imread(mask_l_path, cv2.IMREAD_GRAYSCALE)
    mask_r_path = '/home/admina/segmetation/DLOSeg/outputs/seg_data_720_15fps/img_06_right_mask_0.png'
    # mask_r = cv2.imread(mask_r_path, cv2.IMREAD_GRAYSCALE)
    img_real_path ='/home/admina/segmetation/DLOSeg/src/graph/data_720_15fps/img_06.png'
    img_real = cv2.imread(img_real_path, cv2.IMREAD_GRAYSCALE)
    # split to left right
    img_real_l, img_real_r = img_real[:, :img_real.shape[1] // 2], img_real[:, img_real.shape[1] // 2:]

    print(f"Time to read images: {time.time() - read_time:.3f} seconds")
    # Normalize mask to 0-1 range
    setup_time = time.time()
    img_real_l = img_real_l * mask_l  # Normalize mask to 0-1 range
    img_real_l[img_real_l < img_real_l[img_real_l > 0].mean()] = 0

    # plt.imshow(img_real_l, cmap='gray')
    # plt.show()
    # Create graph instance and load mask
    graph = DLOGraph()
    print(f"Time to setup graph: {time.time() - setup_time:.3f} seconds")
    load_time = time.time()
    graph.load_from_mask(
                    img_real_l,
                        statistic="mean",
                        min_cluster_factor=0.50,
                        padding_size=40,
                    )
    print(f"Time to load mask: {time.time() - load_time:.3f} seconds")

    # Visualize initial graph
    # graph.visualize(node_size=5, with_labels=False, title="Initial Tree Graph")
    prune_time = time.time()
    graph.prune_short_branches_and_delete_junctions(max_length=5)
    print(f"Time to prune branches: {time.time() - prune_time:.3f} seconds")
    

    
    # graph.visualize(node_size=1, with_labels=False, title="Pruned Graph")
    start_fit = time.time()
    # Fit spline to branches
    graph.fit_spline_to_branches(smoothing=20, max_num_points=None)
    print(f"Time to fit spline to branches: {time.time() - start_fit:.3f} seconds")
    time_to_prune = time.time()
    
    # graph.visualize(node_size=1, with_labels=True, title="Graph After Spline Fitting")
    
    start_dlo = time.time()
    graph.reconstruct_dlo()
    print(f"Time to reconstruct DLO: {time.time() - start_dlo:.3f} seconds")
    # graph.visualize(node_size=1, with_labels=False, title="Graph After DLO Reconstruction")

    print(f"Total processing time: {time.time() - start_total:.3f} seconds")
    print("Done")