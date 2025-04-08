import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree
import cv2
from graph.utils.utils import *
import time 



if __name__ == '__main__':
    start = time.time()
    mask = cv2.imread('/home/admina/segmetation/DLOSeg/outputs/grounded_sam2_local_demo/groundingdino_mask_0.png', cv2.IMREAD_GRAYSCALE)
    print("Time taken to read image:", time.time() - start)
    height, width = mask.shape
    down_sample = 6
    start_rs = time.time()
    mask = cv2.resize(mask, (width // down_sample, height // down_sample), interpolation=cv2.INTER_AREA)
    mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
    plt.imshow(mask, cmap='gray')
    height, width = mask.shape
    print("Time taken to resize:", time.time() - start_rs)
    # mask = cv2.ximgproc.thinning(mask)
    start_m2g = time.time()
    G = binary_mask_to_graph_indexed(mask,connectivity=8)
    print("Time taken to convert to graph:", time.time() - start_m2g)
    # Draw with saved node positions
    pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos=pos, node_size=1, edge_color='gray')
    # plt.gca().invert_yaxis()
    # plt.show()
    # draw_nx_graph(G)
    start_sif = time.time()
    # G = simplify_intersections_dbscan(G)
    # G = simplify_intersections_grid(G, grid_size=5)
    # G = simplify_intersections_unionfind(G, threshold= 10.0)
    G = simplify_intersections_fast(G, threshold=2.0)
    print("Time taken to simplify intersections:", time.time() - start_sif)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    # plt.gca().invert_yaxis()
    plt.pause(0.1)
    start_smooth = time.time()
    # path = find_smoothest_path(G, start_node=232, goal_node=15)
    path = find_dlo(G, start_node=1026, goal_node=70)
    print("Time taken to smooth path:", time.time() - start_smooth)
    subG = G.subgraph(path)
    pos = nx.get_node_attributes(subG, 'pos')
    nx.draw(subG,pos=pos, with_labels=False,node_size=5)
    # plt.gca().invert_yaxis()
    plt.show()

