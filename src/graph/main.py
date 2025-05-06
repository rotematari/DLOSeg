import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree

import cv2
from graph.utils.utils import *
import time 
from scipy.interpolate import splrep, splev, splprep

from networkx.algorithms.approximation import traveling_salesman_problem, greedy_tsp

if __name__ == '__main__':
    start = time.time()
    start_total = time.time()
    mask = cv2.imread('/home/admina/segmetation/DLOSeg/outputs/grounded_sam2_local_demo/groundingdino_mask_0.png', cv2.IMREAD_GRAYSCALE)
    print("Time taken to read image:", time.time() - start)
    height, width = mask.shape
    print("Image hight:", height)
    print("Image width:", width)
    down_sample = 4
    start_rs = time.time()
    width = int(width / down_sample)
    height = int(height / down_sample)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    # mask = cv2.resize(mask, (width // down_sample, height // down_sample), interpolation=cv2.INTER_AREA)
    mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
    # SHOW MASK
    plt.imshow(mask, cmap='gray')
    # plt.pause(0.01)
    height, width = mask.shape
    print("Time taken to resize:", time.time() - start_rs)
    
    start_m2g = time.time()
    G_full = binary_mask_to_graph_indexed(mask)
    
    print("Time taken to convert to graph:", time.time() - start_m2g)
    
    # Draw full graph on the mask
    # pos = nx.get_node_attributes(G_full, 'pos')
    # nx.draw(G_full, pos=pos, node_size=1, edge_color='gray')
    # # plt.gca().invert_yaxis()
    # plt.pause(0.01)
    start_sif = time.time()
    pos = nx.get_node_attributes(G_full, 'pos')
    centerline_pts = np.asarray(list(pos.values()), dtype=np.int32)
    width_px = dlo_width_from_mask(centerline_pts, mask, "max")
    print("Width in px:", width_px)
    G = simplify_graph_by_grid(G=G_full, 
                            grid_size=width_px,
                            min_cluster_size=width_px*2  ,
                            )
    
    # get the node positions

    # draw simplified graph 

    # plt.gca().invert_yaxis()
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    # plt.pause(0.01)
    G = knn_graph_from_bending_energy_2(G,G_full, k=10,
                                      be_limit=1.50,
                                      radius=width_px*1.5,
                                      )
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    # # plt.gca().invert_yaxis()
    # plt.pause(0.01)
    
    
    
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    # plt.gca().invert_yaxis()
    plt.pause(0.01)
    
    G = connect_leaf_nodes(G,be_limit=3.0)
    
    G = prune_short_branches(G, min_length=3)
    # G = prune_short_branches(G, min_length=20)
    
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    # # plt.gca().invert_yaxis()
    # plt.pause(0.01)
    
    # G = prune_short_branches(G, min_length=20)
    # G = connect_leaf_nodes(G,be_limit=10.0)
    
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    # plt.gca().invert_yaxis()
    plt.pause(0.01)
    
    
    G = knn_graph_from_pos(
        G_in=G,
        k= 5,
        in_place= False
    )
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    G = add_bending_energy_to_node_and_prune( G, 
                            alpha = 1.0)
    
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    # plt.pause(0.01)
    G = connect_nodes_by_bending_energy(G,10)
    # plt.figure()
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    # plt.gca().invert_yaxis()
    # plt.pause(0.01)
    # G = prune_graph(G,2)

    # G = connect_nodes_by_bending_energy(G,10)
    # G = add_bending_energy_to_node_and_prune( G, 
    #                     alpha = 1.0)
    # G = connect_nodes_by_bending_energy(G,10)
    # G = prune_graph(G,3)
    # G = connect_nodes_by_bending_energy(G,10)
    # plt.figure()
    pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    plt.gca().invert_yaxis()

    print("Time taken to simplify intersections:", time.time() - start_sif)

    
    plt.pause(0.01)

    start_smooth = time.time()
    path = traverse_graph_by_smallest_bending_energy(G)
    subG = G.subgraph(path)
    pos = nx.get_node_attributes(subG, 'pos')
    plt.figure(figsize=(10, 10))
    nx.draw(subG, pos=pos, node_size=5, edge_color='red', with_labels=False)
    plt.gca().invert_yaxis()
    plt.show()
    print("Time taken to smooth path:", time.time() - start_smooth)
    print("Total time taken:", time.time() - start_total)
    # subG = G.subgraph(path)
    # pos = nx.get_node_attributes(subG, 'pos')
    # plt.figure(figsize=(10, 10))
    # nx.draw(subG,pos=pos, with_labels=False,node_size=5)
    # plt.gca().invert_yaxis()


