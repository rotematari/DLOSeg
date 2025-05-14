import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree

import cv2
from graph.utils.utils import *
import time 
from scipy.interpolate import splrep, splev, splprep

from networkx.algorithms.approximation import traveling_salesman_problem, greedy_tsp
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G, title="Graph"):
    plt.figure()
    # add title
    plt.title(title)
    plt.gca().invert_yaxis()
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    plt.pause(0.01)

if __name__ == '__main__':
    start = time.time()
    start_total = time.time()
    mask = cv2.imread('/home/admina/segmetation/DLOSeg/outputs/grounded_sam2_local_demo/groundingdino_mask_0.png', cv2.IMREAD_GRAYSCALE)
    print("Time taken to read image:", time.time() - start)
    height, width = mask.shape
    print("Image hight:", height)
    print("Image width:", width)
    down_sample = 1
    start_rs = time.time()
    width = int(width / down_sample)
    height = int(height / down_sample)
    if down_sample > 1:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # mask = cv2.resize(mask, (width // down_sample, height // down_sample), interpolation=cv2.INTER_AREA)
    mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
    # SHOW MASK
    # plt.imshow(mask, cmap='gray')
    # plt.pause(0.01)
    height, width = mask.shape
    print("Time taken to resize:", time.time() - start_rs)
    
    start_m2g = time.time()
    G_full = binary_mask_to_graph_indexed(mask)
    

    print("Time taken to convert to graph:", time.time() - start_m2g)

    # Draw full graph on the mask
    # pos = nx.get_node_attributes(G_full, 'pos')
    # nx.draw(G_full, pos=pos, node_size=1, edge_color='gray')
    # plt.gca().invert_yaxis()
    # plt.pause(0.01)
    
    pos = nx.get_node_attributes(G_full, 'pos')
    centerline_pts = np.asarray(list(pos.values()), dtype=np.int32)
    width_px = dlo_width_from_mask(centerline_pts, mask, "max")
    print("Width in px:", width_px)
    start_smip = time.time()
    G = simplify_graph_by_grid(G=G_full, 
                            grid_size=width_px,
                            min_cluster_size=width_px*2  ,
                            )
    print("Time taken to simplify graph:", time.time() - start_smip)
    # get the node positions

    # draw simplified graph 
    visualize_graph(G, title="Simplified graph")
    start_trav = time.time()
    G = knn_graph_from_bending_energy_2(G,G_full, k=10,
                                      be_limit=1.50,
                                      radius=width_px*1.5,
                                      )
    print("Time taken to create knn graph:", time.time() - start_trav)
    # draw knn graph
    visualize_graph(G, title="KNN graph")
    

    start_con = time.time()
    G = connect_leaf_nodes(G,be_limit=5.0)
    print("Time taken to connect leaf nodes:", time.time() - start_con)
    # draw connected graph
    visualize_graph(G, title="Connected graph")
    start_prune = time.time()
    G = prune_short_branches(G, min_length=3)
    print("Time taken to prune short branches:", time.time() - start_prune)

    
    print("total time taken:", time.time() - start_total)
    # draw pruned graph 
    visualize_graph(G, title="Pruned graph")

    print("Done")
