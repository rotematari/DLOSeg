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
    down_sample = 4
    start_rs = time.time()
    width = int(width / down_sample)
    height = int(height / down_sample)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    # mask = cv2.resize(mask, (width // down_sample, height // down_sample), interpolation=cv2.INTER_AREA)
    mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
    plt.imshow(mask, cmap='gray')
    height, width = mask.shape
    print("Time taken to resize:", time.time() - start_rs)
    # mask = cv2.ximgproc.thinning(mask)
    start_m2g = time.time()
    G_full = binary_mask_to_graph_indexed(mask,connectivity=8)
    
    
    print("Time taken to convert to graph:", time.time() - start_m2g)
    # Draw with saved node positions
    # pos = nx.get_node_attributes(G_full, 'pos')
    # nx.draw(G_full, pos=pos, node_size=1, edge_color='gray')
    # plt.gca().invert_yaxis()
    
    # draw_nx_graph(G)
    start_sif = time.time()

    # G = simplify_intersections_fast_2(G, dist_threshold=2.0)
    G,dic = simplify_graph_by_grid(G=G_full, 
                            grid_size=5,
                            min_cluster_size=10 ,
                            connect_radius_factor=2.5)
    
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    G = knn_graph_from_pos(
        G_in=G,
        k= 5,
        in_place= False
    )
    G = add_bending_energy_to_node_and_prune( G, 
                            alpha = 1.0)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    plt.pause(0.01)
    G = connect_nodes_by_bending_energy(G,10)
    plt.figure()
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    plt.gca().invert_yaxis()
    # TODO : prone nodes with high turning angles
    # Find leaf nodes (nodes with only one neighbor)
    leaf_nodes = [n for n in G.nodes if G.degree[n] == 1]
    # print("Leaf nodes:", leaf_nodes)
    paths = []
    list_of_paths = []
    if len(leaf_nodes) > 0:
        for node in leaf_nodes:
            # print("Node:", node)
            
            path   = sparse_greedy_path(G,
                                        node
                                        )
            
            if node not in paths:
                list_of_paths.append(path)
                paths.extend(path)
            # subG = G.subgraph(path)
            # pos = nx.get_node_attributes(subG, 'pos')
            # nx.draw(subG, pos=pos, node_size=5, edge_color='red', with_labels=True)
    
    # nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
    print("Time taken to simplify intersections:", time.time() - start_sif)
    # paths = extend_paths(   paths = list_of_paths,
    #                         G_full = G_full,
    #                         G = G,
    #                         max_dist = 50.0,
    #                         max_line_dist = 1.0,
    #                         line_agreement = 0.950,
    #                         k_candidates = 10
    #                         )
    # plt.figure()
    # for path in paths:
        
    #     subG = G.subgraph(path)
    #     pos = nx.get_node_attributes(subG, 'pos')
    #     nx.draw(subG, pos=pos, node_size=5, edge_color='red', with_labels=False)
    
    plt.pause(0.01)
    # plt.gca().invert_yaxis()
    # plt.show()
    start_smooth = time.time()

    print("Time taken to smooth path:", time.time() - start_smooth)
    print("Total time taken:", time.time() - start_total)
    # subG = G.subgraph(path)
    # pos = nx.get_node_attributes(subG, 'pos')
    # plt.figure(figsize=(10, 10))
    # nx.draw(subG,pos=pos, with_labels=False,node_size=5)
    # plt.gca().invert_yaxis()


