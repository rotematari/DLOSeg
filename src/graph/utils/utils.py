import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist


import math
from scipy.spatial import cKDTree
import heapq
# from scipy.spatial import ccKDTree
from collections import deque
import torch
import torch.nn.functional as F
from collections import defaultdict
from networkx.algorithms.approximation import traveling_salesman_problem, greedy_tsp
import heapq

def draw(G):
    pos = nx.get_node_attributes(G, "pos")
    nx.draw(G, pos, with_labels=True, node_size=5)
    plt.gca().set_aspect("equal")
    plt.show()
def binary_mask_to_graph_indexed(mask, connectivity=4):
    assert connectivity in (4, 8), "Connectivity must be 4 or 8"

    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    G = nx.Graph()

    ys, xs = np.nonzero(mask)
    coords = list(zip(ys, xs))

    # Step 1: assign an index to each (y, x)
    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}

    # Step 2: add nodes with index and position attribute
    for coord, idx in coord_to_index.items():
        y, x = coord
        G.add_node(idx, pos=(x, y))  # flip to (x, y) for drawing

    # # Step 3: define neighbors
    # if connectivity == 4:
    #     # offsets = [(0, 1), (1, 0)]
    #     offsets = [(1, 1), (1, -1)]
        
    # else:
    #     offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]

    # # Step 4: find and add valid edges using indices
    # for dy, dx in offsets:
    #     for y, x in coords:
    #         ny, nx_ = y + dy, x + dx
    #         neighbor = (ny, nx_)
    #         if 0 <= ny < h and 0 <= nx_ < w and mask[ny, nx_]:
    #             src_idx = coord_to_index[(y, x)]
    #             dst_idx = coord_to_index[neighbor]

    #             G.add_edge(src_idx, dst_idx)

    return G
def knn_graph_from_pos(
        G_in: nx.Graph,
        k: int = 2,
        in_place: bool = False
    ) -> nx.Graph:
    """
    Build a k‑nearest‑neighbour graph from an existing graph *G_in*
    whose nodes have a `"pos"` attribute = (x, y).

    Parameters
    ----------
    G_in     : nx.Graph or nx.DiGraph
        Source graph with node attribute ``"pos": (x, y)``.
    k        : int, default 2
        Each node is connected to its k nearest neighbours.
    in_place : bool, default False
        If True, add the k‑NN edges to *G_in* and return it.
        If False, create and return a fresh nx.Graph that
        contains only the k‑NN edges (node attributes copied).

    Returns
    -------
    G_out : nx.Graph
        Undirected k‑NN graph with edge attribute ``weight`` =
        Euclidean distance and node attribute ``pos``.
    """
    if k < 1:
        raise ValueError("k must be ≥ 1")

    # ---- 1. Extract node IDs and their positions ---------------------------
    try:
        nodes   = np.fromiter(G_in.nodes, dtype=int)
        
        coords  = np.array([G_in.nodes[n]["pos"] for n in nodes], dtype=float)
    except KeyError as e:
        raise ValueError(f"Node {e.args[0]} is missing a 'pos' attribute")

    # ---- 2. Build KD‑tree for O(N log N) neighbour look‑ups ---------------
    tree = cKDTree(coords)

    # ---- 3. Decide whether to modify in place or start fresh --------------
    G_out = G_in if in_place else nx.Graph()
    if not in_place:
        # Copy nodes (including all attributes) into the new graph
        G_out.add_nodes_from((n, G_in.nodes[n]) for n in nodes)

    # ---- 4. Add edges to the k nearest neighbours -------------------------
    for idx, pos in enumerate(coords):
        dists, nbr_idx = tree.query(pos, k=k + 1)        # self + k
        for j, dist in zip(nbr_idx[1:], dists[1:]):      # skip self
            u, v = int(nodes[idx]), int(nodes[j])
            if not G_out.has_edge(u, v):
                v_nbs = [n for n in G_out.neighbors(v) if n != v]
                u_nbs = [n for n in G_out.neighbors(u) if n != u]
                vec = np.array(G_out.nodes[v]['pos']) - np.array(G_out.nodes[u]['pos'])
                dist = np.linalg.norm(vec)
                # if dist < 7.5:
                #     # Add edge if both nodes are not already connected
                #     G_out.add_edge(u, v, weight=float(dist),vector=vec)
                #     continue
                if not len(v_nbs) >= 2 and not len(u_nbs) >= 2:
                    # Add edge if both nodes are not already connected
                    G_out.add_edge(u, v, weight=float(dist),vector=vec)

    return G_out

def sparse_greedy_path_legth(G, start=None):
    

    if start is None:
        start = list(G.nodes)[0]

    visited = set([start])
    path = [start]
    current = start

    while len(visited) < len(G.nodes):
        neighbors = [(G[current][n]["weight"], n) for n in G.neighbors(current) if n not in visited]
        if not neighbors:
            break  # disconnected graph
        _, next_node = min(neighbors)
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path

# --------------------------------------------------------------------------- #
def line_points(p1: tuple[float, float],
                p2: tuple[float, float],
                num: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return `num` evenly‑spaced points on the segment p1–p2."""
    x1, y1 = p1
    x2, y2 = p2
    t = np.linspace(0.0, 1.0, num, dtype=float)
    x = (1.0 - t) * x1 + t * x2
    y = (1.0 - t) * y1 + t * y2
    return x, y
# --------------------------------------------------------------------------- #
def extend_paths(   paths      : list[list[int]],
                    G_full          : nx.Graph,
                    G            : nx.Graph,
                    max_dist   : float = 15.0,
                    max_line_dist: float = 5.0,
                    line_agreement: float = 0.5,
                    k_candidates  : int = 10) -> list[list[int]]:
    """
    Try to extend each path at its start/end by attaching the closest
    non‑neighbour node whose straight line is mostly supported by DLO nodes.

    Parameters
    ----------
    paths : list of paths (each path = list[int])
    G     : graph with node attribute "pos" = (x, y)
    max_dist       : search radius for candidate nodes (pixels / units)
    max_line_dist  : tolerance when probing the interpolated line
    line_agreement : required fraction of the sampled line that must run
                     within `max_line_dist` of an existing node (0‑1)
    k_candidates   : how many nearest nodes to test per side

    Returns
    -------
    new_paths : same length as `paths`, each possibly extended
    """
    # ---- KD‑tree on all node positions ------------------------------------
    pos_dict_full   = nx.get_node_attributes(G_full, "pos")
    nodes_full      = np.fromiter(pos_dict_full.keys(), dtype=int)
    coords_full     = np.array([pos_dict_full[n] for n in nodes_full], dtype=float)
    tree_full       = cKDTree(coords_full)
    sentinel_full   = len(coords_full)          # index returned when no neighbour found
    id_of_idx_full  = {i: node for i, node in enumerate(nodes_full)}
    
    pos_dict   = nx.get_node_attributes(G, "pos")
    nodes      = np.fromiter(pos_dict.keys(), dtype=int)
    coords     = np.array([pos_dict[n] for n in nodes], dtype=float)
    tree       = cKDTree(coords)
    sentinel   = len(coords)          # index returned when no neighbour found
    id_of_idx  = {i: node for i, node in enumerate(nodes)}

    new_paths = []

    for path in paths:
        path_set      = set(path)
        extended_path = path[:]       # copy

        # We treat the two ends separately
        for side, ref_node in (("start", path[0]), ("end", path[-1])):
            ref_pos = pos_dict[ref_node]

            # 1️⃣  Find candidate nodes near the reference node
            _, idxs = tree.query(ref_pos,
                                 k=min(k_candidates + 1, len(nodes)),
                                 distance_upper_bound=max_dist)
            cands = [id_of_idx[idx] for idx in idxs if idx != sentinel ]
            cands = [c for c in cands if c not in path_set]
            # print(f"Ref node: {ref_node} Candidates: {cands}")
            # Go through candidates in ascending distance order
            for cand in cands:
                if G.has_edge(ref_node, cand):
                    continue

                # 2️⃣  Line‑of‑sight / DLO support test
                cand_pos = pos_dict[cand]
                xs, ys   = line_points(ref_pos, cand_pos,num=100)
                hits     = 0
                
                for x, y in zip(xs, ys):
                    # single‑nearest lookup
                    line_pos = (x, y)
                    _, near_idxs = tree_full.query(line_pos,
                                                k=3,
                                                )
                    line_cands = [id_of_idx_full[idx] for idx in near_idxs if idx != sentinel_full]
                    line_cands = [c for c in line_cands if c not in path_set]
                    for line_cand in line_cands:
                        line_cand_pos = pos_dict_full[line_cand]
                        edge_len = np.linalg.norm(np.array(line_pos) - np.array(line_cand_pos))
                        if edge_len <= max_line_dist:
                            # Found a valid DLO node
                            hits += 1
                            break
                            

                line_score = hits / len(xs)
                # print(f"Line score: {line_score} for {ref_node} to {cand}")
                if line_score <= line_agreement:
                    continue            # line not well supported → try next cand

                # 3️⃣  All tests passed → connect and update path
                edge_len = np.linalg.norm(np.array(ref_pos) - np.array(cand_pos))
                G.add_edge(ref_node, cand, weight=float(edge_len))

                if side == "start":
                    extended_path.insert(0, cand)
                else:
                    extended_path.append(cand)

                # Only extend once per side
                break

        new_paths.append(extended_path)

    return new_paths

def prune_edges(G:nx.Graph, max_bending_energy:float = 5.0) -> nx.Graph:
    
    
    return G

def simplify_graph_by_grid(
    G: nx.Graph,
    grid_size: float = 3.0,
    *,
    min_cluster_size: int = 1,
    connect_radius_factor: float = 1.5,
) -> tuple[nx.Graph, dict]:
    """
    Down‑sample a 2‑D geometric graph by clustering nodes into square grid cells
    of size `grid_size` and replacing each cluster with its centroid‑nearest node.

    Parameters
    ----------
    G : nx.Graph
        Original graph. Each node must have a `pos=(x, y)` attribute (floats).
    grid_size : float, default 3.0
        Side length of each grid cell in the same units as the node positions.
    min_cluster_size : int, default 1
        Clusters smaller than this are ignored.
    connect_radius_factor : float, default 1.5
        Edges are (re‑)created between representative nodes whose Euclidean
        distance ≤ `connect_radius_factor * grid_size`.

    Returns
    -------
    simplified_G : nx.DiGraph
        The reduced graph with representative nodes and bidirectional edges.
    node_to_rep : dict
        Mapping from every original node to its representative.
    """
    # --- gather positions ---------------------------------------------------
    pos_dict = nx.get_node_attributes(G, "pos")
    if len(pos_dict) != len(G):
        raise ValueError("Every node needs a 'pos' attribute")

    node_list = list(pos_dict)                         # preserve order
    coords = np.array([pos_dict[n] for n in node_list])  # (N, 2)
    node_index = {n: i for i, n in enumerate(node_list)}

    # --- assign each node to a grid cell ------------------------------------
    cell_indices = np.floor_divide(coords, grid_size).astype(int)  # (N, 2)

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node, cell in zip(node_list, cell_indices):
        buckets[tuple(cell)].append(node)

    clusters = [c for c in buckets.values() if len(c) >= min_cluster_size]

    # --- pick a representative per cluster ----------------------------------
    rep_pos = {}
    node_to_rep = {}

    for cluster in clusters:
        cluster_rows = [node_index[n] for n in cluster]          # safe indices
        cluster_coords = coords[cluster_rows]
        centroid = cluster_coords.mean(axis=0)

        # vectorised distance to centroid
        dists = np.linalg.norm(cluster_coords - centroid, axis=1)
        best_idx = int(dists.argmin())
        rep_node = cluster[best_idx]

        rep_pos[rep_node] = pos_dict[rep_node]
        for n in cluster:
            node_to_rep[n] = rep_node

    # --- build simplified graph ---------------------------------------------
    simplified_G = nx.Graph()
    simplified_G.add_nodes_from(
        [(n, {"pos": rep_pos[n]}) for n in rep_pos]
    )

    rep_nodes = list(rep_pos)
    rep_coords = np.array([rep_pos[n] for n in rep_nodes])
    rep_tree = cKDTree(rep_coords)

    # radius = connect_radius_factor * grid_size
    # for i, node_i in enumerate(rep_nodes):
    #     pos_i = rep_coords[i]
    #     for j in rep_tree.query_ball_point(pos_i, r=radius):
    #         if j <= i:
    #             continue  # avoid duplicates
    #         node_j = rep_nodes[j]
    #         pos_j = rep_coords[j]

    #         dx, dy = pos_j - pos_i
    #         angle_ij = math.degrees(math.atan2(dy, dx))
    #         angle_ji = (angle_ij + 180.0) % 360.0

    #         simplified_G.add_edge(node_i, node_j, angle_deg=angle_ij)
    #         simplified_G.add_edge(node_j, node_i, angle_deg=angle_ji)

    return simplified_G, node_to_rep
def add_bending_energy_to_node_and_prune(G:nx.Graph, alpha:float = 1.0) -> nx.Graph:
    """
    Add bending energy to the graph edges.
    as in discrite elastic rod model
    
    :param
    G: graph
    alpha: weight of the bending energy
    :return: graph with bending energy
    """
    
    node_dict = nx.get_node_attributes(G, 'pos')
    
    for i,node in enumerate(node_dict.items()):
        
        neighbors = list(G.neighbors(node[0]))
        if len(neighbors) < 2:
            print("Node {} has less than 2 neighbors".format(node[0]))
            continue                                    # leaf or isolated
        # get the edges
        # to the node
        edge_to_node = np.array(node_dict[node[0]])- np.array(node_dict[neighbors[0]])
        # frome the node to the next node
        edge_from_node = np.array(node_dict[neighbors[1]]) - np.array(node_dict[node[0]])

        be_value = float(bending_energy(edge_to_node, edge_from_node, alpha=alpha))
        if be_value > 2:
            # drop the node and the edges
            G.remove_node(node[0])
            continue
        # G.nodes[node[0]]['bending_energy'] = (neighbors[0],neighbors[1]),be
        G.nodes[node[0]].setdefault('bending_energy', []).append(((neighbors[0],neighbors[1]),be_value))
    
    # prune the graph
    # get all one degree nodes
    one_degree_nodes = [n for n in G.nodes if G.degree[n] == 1]
    for node in one_degree_nodes:
        # if the neighbor also has degree 1, remove the node
        neighbors = list(G.neighbors(node))
        one_degree_neighbors = [n for n in neighbors if G.degree[n] == 1]
        for n in one_degree_neighbors:
            # remove the node and the edge
            G.remove_node(node)
            
            break
    return G

def same_line_same_direction(v, w, tol=1e-9):
    """
    Return True if v and w are colinear and point in the same direction.
    v, w are length-2 numpy arrays (or lists / tuples).
    """
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    if np.linalg.norm(v) < tol or np.linalg.norm(w) < tol:
        raise ValueError("Zero-length vector")

    # scalar z-component of the 2-D cross product
    cross_z = v[0]*w[1] - v[1]*w[0]

    colinear     = abs(cross_z) < tol          # nearly zero ⇒ parallel
    same_signed  = np.dot(v, w) > 0            # >0 ⇒ same direction

    return colinear and not same_signed

def connect_nodes_by_bending_energy(G: nx.Graph, number_of_nb=10) -> nx.Graph:
    """
    Connect nodes with degree 1 (endpoints) to other nodes with minimal bending energy.
    
    Parameters:
    -----------
    G : nx.Graph
        Input graph with node attribute 'pos'
    number_of_nb : int, default 10
        Number of nearest neighbors to consider for each endpoint
        
    Returns:
    --------
    G : nx.Graph
        Modified graph with new edges added to endpoints
    """
    nodes = np.fromiter(G.nodes, dtype=int)
    
    coords = np.array([G.nodes[n]["pos"] for n in nodes], dtype=float)
    tree = cKDTree(coords)
    # get 1 degree nodes
    one_degree_nodes = [n for n in G.nodes if G.degree[n] == 1]
    
    # ---- 2. Build KD‑tree for O(N log N) neighbour look‑ups ---------------
    
    for node in one_degree_nodes:
        # print("\n-------------------------------------------------")
        # print("Working on Node {}".format(node))
        if G.degree[node] > 1:
            print("Node {} has degree > 1".format(node))
            continue
        node_pos = G.nodes[node]["pos"]
        # get n closest nodes
        _, idxs = tree.query(G.nodes[node]["pos"],
                            k=number_of_nb,
                            )
        bending_energys = {}
        # print("Closest nodes to {}: {}".format(node, nodes[idxs]))
        for idx in idxs:
            # get the node
            candidate_node = nodes[idx]
            
            # Skip self or already connected nodes
            if candidate_node == node or G.has_edge(node, candidate_node):
                # print("Node {} is already connected to {}".format(node, candidate_node))
                continue
                

            candidate_pos = G.nodes[candidate_node]["pos"]
            # get first node neighbor
            first_node_nb = list(G.neighbors(node))[0]
            first_node_nb_pos = G.nodes[first_node_nb]["pos"]
            # get the edges
            # to the node
            edge_to_node = np.array(first_node_nb_pos) - np.array(node_pos)
            # frome the node to the next node
            edge_from_node = np.array(node_pos) - np.array(candidate_pos)
            if same_line_same_direction(edge_to_node, edge_from_node):
                # print("Node {} and {} are on the same line".format(node, candidate_node))
                continue
            be = float(bending_energy(edge_to_node, edge_from_node, alpha=1.0))
            bending_energys[candidate_node] = (first_node_nb,candidate_node),be
        
        # Find the neighbor node with the lowest bending energy and add edge
        if bending_energys:
            # print("Bending energys: {}".format(bending_energys))
            best_candidate = min(bending_energys, key=lambda x: bending_energys[x][1])
            best_energy = bending_energys[best_candidate]
            if best_energy[1] > 10:
                continue
            
            # Add edge with appropriate attributes
            edge_weight = np.linalg.norm(
                np.array(G.nodes[node]["pos"]) - np.array(G.nodes[best_candidate]["pos"])
            )
            edge_vector = np.array(G.nodes[best_candidate]["pos"]) - np.array(G.nodes[node]["pos"])
            
            if G.has_edge(node, best_candidate):
                # print("Node {} and {} are already connected".format(node, best_candidate))
                continue

            G.add_edge(
                node, 
                best_candidate, 
                weight=float(edge_weight),
                vector=edge_vector,
            )
            # print("Added edge between {} and {} with weight {}".format(node, best_candidate, edge_weight))
            # add bending_energy to the node
            G.nodes[node].setdefault('bending_energy', []).append(best_energy)
            
            # add bending_energy to the best candidate node
            # from node to best candidate
            to_best_candidate = np.array(G.nodes[best_candidate]['pos']) - np.array(G.nodes[node]['pos'])
            
            best_candidate_nbrs = list(G.neighbors(best_candidate))
            
            if len(best_candidate_nbrs) > 0:
                for best_candidate_nbr in best_candidate_nbrs:
                    # from best candidate to the neighbor
                    from_best_candidate = np.array(G.nodes[best_candidate_nbr]['pos']) - np.array(G.nodes[best_candidate]['pos'])
                    be = float(bending_energy(from_best_candidate, to_best_candidate, alpha=1.0))
                    if be > 2:
                        continue
                    if best_candidate_nbr == node:
                        continue
                    G.nodes[best_candidate].setdefault('bending_energy', []).append(((node,best_candidate_nbr),be))
                    

                    
    return G
            
            
            
def bending_energy(prev_edges,next_edges, alpha=1.0):
    """
    Compute the bending energy between two edges.
    as in discrite elastic rod model
    
    :param
    e_prev: previous edge vector
    :param e_next: next edge vector
    :param alpha: weight of the bending energy
    :return: bending energy
    """
    # make sure the edges are 2D vectors
    if prev_edges.ndim == 1:
        prev_edges = prev_edges.reshape(1, -1)
    if next_edges.ndim == 1:
        next_edges = next_edges.reshape(1, -1)
    # points: (N,2/3) tensor (requires_grad=True if torch)
    e_prev = torch.from_numpy(prev_edges).float()  # (N,2/3)
    e_next = torch.from_numpy(next_edges).float()  # (N,2/3)
    if e_prev.size(1) == 2:
        # N×2 → N×3   (append a zero z–component)
        e_prev3 = F.pad(e_prev, (0, 1))        # (1,3)
        e_next3 = F.pad(e_next, (0, 1))        # (1,3)

    cross   = torch.cross(e_prev3, e_next3, dim=1)   # (1,3)
    dot     = (e_prev3 * e_next3).sum(dim=1)         # (1,)

    L_prev  = torch.linalg.norm(e_prev3, dim=1)
    L_next  = torch.linalg.norm(e_next3, dim=1)

    denom   = (L_prev * L_next + dot).clamp_min(1e-12)
    kappa_b = 2.0 * cross / denom.unsqueeze(1)       # (1,3)

    kappa2  = (kappa_b**2).sum(dim=1)                # (1,)
    l_i     = 0.5 * (L_prev + L_next)                # (1,)
    # E_i     = kappa2 * l_i                           # (1,) 
    E_i =  kappa2                   
    return E_i

def prune_graph(G:nx.Graph, max_degree:int = 3) -> nx.Graph:
    """
    Prune the graph by removing nodes with bending energy above a threshold.
    
    Parameters:
    -----------
    G : nx.Graph
        Input graph with node attribute 'pos'
    max_degree : int, default 3
        Maximum allowed degree for nodes to be retained
        
    Returns:
    --------
    G : nx.Graph
        Pruned graph
    """
    print("\n-------------------------------------------------\n")
    print("Pruning graph with max degree {}".format(max_degree))
    # get all one degree nodes
    big_degree_nodes = [n for n in G.nodes if G.degree[n] > max_degree]
    big_degree_nodes_set = set(big_degree_nodes)
    
    for node in big_degree_nodes:
        if node in big_degree_nodes_set:
            print("\n-------------------------------------------------")
            print("Working on Node {}".format(node))
            big_degree_nodes_set.remove(node)
            # prune node if it has more bendig energys than neighbors
            if 'bending_energy' in G.nodes[node]:
                bending_energys = G.nodes[node]['bending_energy']
                if G.degree(node) > len(bending_energys)  :


                    # connect the neighbors
                    neighbors = list(G.neighbors(node))
                    G.remove_node(node)
                    print("Removed node {} with too many bending energys".format(node))
                    for neighbor_1 in neighbors:
                        for neighbor_2 in neighbors:
                            if neighbor_1 == neighbor_2:
                                continue
                            # check if the neighbor is not already connected
                            if G.has_edge(neighbor_1, neighbor_2):
                                print("Node {} and {} are already connected".format(neighbor_1, neighbor_2))
                                continue
                            # connect the neighbors
                            
                            if G.degree[neighbor_1] == 1 and G.degree[neighbor_2] == 1:
                                G.add_edge(neighbor_1, neighbor_2)
                                if neighbor_1 in big_degree_nodes_set:
                                    big_degree_nodes_set.remove(neighbor_1)
                                    print("Removed node {} from big degree nodes set".format(neighbor_1))
                                if neighbor_2 in big_degree_nodes_set:
                                    big_degree_nodes_set.remove(neighbor_2)
                                    print("Removed node {} from big degree nodes set".format(neighbor_2))
                                print("Connected node {} to neighbor {}".format(neighbor_1, neighbor_2))
                            

                    continue
        
        # prune nodes with bad bending energy
        nodes_to_remove = []
        for node in list(G.nodes):
            if 'bending_energy' in G.nodes[node]:
                # get the bending energy
                bending_energys = G.nodes[node]['bending_energy']
                # check if the bending energy is too high
                for bending_energy in bending_energys:
                    if bending_energy[1] > 2:
                        nodes_to_remove.append(node)
                        print("Marked node {} for removal due to bad bending energy".format(node))
                        break
        
        # Remove nodes after the iteration
        for node in nodes_to_remove:
            if node in G.nodes:  # Check if still in graph
                G.remove_node(node)
                print("Removed node {} with bad bending energy".format(node))
    return G
def traverse_graph_by_smallest_bending_energy(G:nx.Graph):
    
    """
    Traverse a graph by always choosing the edge that creates the smallest bending energy.
    
    Parameters:
    -----------
    G : nx.Graph
        Input graph with node attribute 'pos'
        
    Returns:
    --------
    paths : list of lists
        List of paths (each path is a list of node IDs) found by traversing the graph
    """
    
    # get start and end nodes 
    # nodes with degree 1
    leaf_nodes = [n for n in G.nodes if G.degree[n] == 1]
    start_node = leaf_nodes[0]
    goal_node = leaf_nodes[-1]
    # print("Start node: {}, Goal node: {}".format(start_node, goal_node))
    
    # get all nodes 
    nodes_dict = nx.get_node_attributes(G, 'bending_energy')
    
    traverse = True
    visited_nodes = set()
    path = []
    
    # start from the first node
    visited_nodes.add(start_node)
    path.append(start_node)
    first_nbr = list(G.neighbors(start_node))[0]
    # print("First neighbor: {}".format(first_nbr))
    visited_nodes.add(first_nbr)
    path.append(first_nbr)
    current_node = first_nbr
    last_node = start_node
    while traverse:
        # check if the current node is the goal node
        if current_node == goal_node:
            print("Goal node reached: {}".format(current_node))
            traverse = False
            break
        bending_energys = nodes_dict[current_node]
        print("\n-------------------------------------------------")
        print("Current node: {}, Last node {} , Bending energys: {}".format(current_node,last_node, bending_energys))
        # find possible candidates
        # if the path last->current_node->candidate is in the bending energy list
        # find all the energies with the last node
        
        candidate_list = []
        for bending_energy in bending_energys:
            if last_node in bending_energy[0]:
                # check if the candidate is not already visited
                # if last_node not in visited_nodes:
                candidate_list.append(bending_energy)
        # get min bending energy
        sorted_candidate_list = sorted(candidate_list, key=lambda x: x[1])
        print("Sorted candidate list: {}".format(sorted_candidate_list))
        # min_bending_energy = min(candidate_list, key=lambda x: x[1])
        # print("With Best {}".format(min_bending_energy))
        
        # get the best candidate
        # best_candidate_tuple = min_bending_energy[0]
        stop = False
        for best_candidate_tuple,_ in sorted_candidate_list:
            for best_candidate in best_candidate_tuple:
                if best_candidate not in visited_nodes :
                    # check if the candidate is in the graph
                    # if not, continue
                    if best_candidate not in G.nodes:
                        print("Candidate {} not in graph".format(best_candidate))
                        continue
                    # if the 
                    # add the node to the path
                    path.append(best_candidate)
                    visited_nodes.add(best_candidate)
                    last_node = current_node
                    current_node = best_candidate
                    stop = True
                    break
                else:
                    print("Candidate {} already visited".format(best_candidate))
            if stop:
                break
        print("Path: {}".format(path))

    return path
    

#     edge_angles = nx.get_edge_attributes(G, 'angle_rad')

#     # Each state is (cost, current_node, incoming_angle, path)
#     heap = [(0, start_node, None, [start_node])]
#     visited = dict()

#     while heap:
#         cost, current, in_angle, path = heapq.heappop(heap)

#         # Check goal
#         if current == goal_node:
#             return path

#         state_key = (current, in_angle if in_angle is not None else None)
#         if state_key in visited and visited[state_key] <= cost:
#             continue
#         visited[state_key] = cost

#         for neighbor in G.neighbors(current):
#             # Edge can be (current, neighbor) or (neighbor, current)
#             edge = (current, neighbor) if (current, neighbor) in edge_angles else (neighbor, current)
#             out_angle = edge_angles[edge]

#             if in_angle is None:
#                 angle_cost = 0  # First step, no turning cost
#             else:
#                 angle_cost = angle_diff(in_angle, out_angle)

#             total_cost = cost + angle_cost
#             heapq.heappush(heap, (total_cost, neighbor, out_angle, path + [neighbor]))

#     return None  # No path found


# # find the smothest path between length n from a node 
# def find_smoothest_path_length(G, start_node, length, goal_node, last_path=None):
#     # pos = nx.get_node_attributes(G, 'pos')
#     edge_angles = nx.get_edge_attributes(G, 'angle_deg')

#     # Each state is (cost, current_node, incoming_angle, path)
#     heap = [(0, start_node, None, [start_node])]
#     visited = dict()
#     candidate_paths = []  # To store candidate paths as tuples (cost, path)
#     while heap:
#         cost, current, in_angle, path = heapq.heappop(heap)

#         # Check if the current path qualifies as a candidate.
#         is_candidate = False
#         if goal_node is not None:
#             if current == goal_node:
#                 is_candidate = True
        
#         if len(path) == length:
#             is_candidate = True

#         if is_candidate:
#             candidate_paths.append((cost, path))
#             # Do not expand this path further.
#             continue

#         state_key = (current, in_angle if in_angle is not None else None)
#         if state_key in visited and visited[state_key] <= cost:
#             continue
#         visited[state_key] = cost

#         for neighbor in G.neighbors(current):
#             # # Skip if the neighbor is the last node in the path
#             if len(path) > 1:
#                 if neighbor in path:
#                     continue
#             if last_path is not None:
#                 if neighbor in last_path:
#                     continue

#             # Edge can be (current, neighbor) or (neighbor, current)
#             edge = (current, neighbor)
#             out_angle = edge_angles[edge]

#             if in_angle is None:
#                 angle_cost = 0  # First step, no turning cost
#             else:
#                 angle_cost = angle_diff(in_angle, out_angle)

#             total_cost = cost + angle_cost
#             candidate_path = path + [neighbor]
#             # Check if the candidate path is valid
#             if candidate_path == last_path:
#                 continue
#             heapq.heappush(heap, (total_cost, neighbor, out_angle,candidate_path ))
#         # After exploring all possibilities, choose the candidate with the smallest cost.
#     if candidate_paths:
#         best_candidate = min(candidate_paths, key=lambda x: x[0])[1]
#         return best_candidate
#     print("No path found in the heap")
#     return None  # No path found

# # traverse the graph in n size steps
# def find_dlo(G, start_node, goal_node):
#     path = []
#     traverse = True
#     branch_length = 10
#     small_path = find_smoothest_path_length(G, start_node, branch_length,goal_node,last_path=path)
#     if small_path is None:
#         print("No path found")
#         traverse = False
#     else:
#         path.extend(small_path)
#         start_node = small_path[-1]
#     while traverse:
#         revers_path = small_path[::-1]
#         small_path = find_smoothest_path_length(G, start_node, branch_length,goal_node,last_path=path)
#         if small_path is None:
#             traverse = False
#             print("No path found")
#         else:
#             path.extend(small_path[1:])
#             start_node = small_path[-1]
#         if start_node == goal_node :
#             traverse = False
#             print("Goal reached")
    
#     return path




# def find_longest_path(G: nx.DiGraph, start: int, max_angle: float) -> list[int]:
#     """
#     Finds the longest simple path in a directed graph G starting from the 
#     specified node, under the constraint that the difference between the angles of 
#     consecutive edges does not exceed max_angle.
    
#     The graph is assumed to have an edge attribute 'angle_deg' giving the direction
#     of the edge in degrees. For the first step (from the start node), no angle
#     constraint is applied.
    
#     Args:
#         G (nx.DiGraph): A directed graph with edges that have an 'angle_deg' attribute.
#         start (int): The starting node.
#         max_angle (float): Maximum allowed difference (in degrees) between consecutive edges.
    
#     Returns:
#         list[int]: A list of nodes representing the longest valid path found. If no 
#                    extension is possible, returns the trivial path [start].
#     """
#     best_path = [start]  # Global variable to store the longest valid path found
    
#     def dfs(current: int, current_path: list[int], prev_angle: float | None):
#         nonlocal best_path
        
#         # Update best_path if the current path is longer than the previously found path.
#         if len(current_path) > len(best_path):
#             best_path = current_path.copy()
        
#         # Iterate over outgoing neighbors of the current node.
#         for neighbor in G.successors(current):
#             # Avoid cycles by not revisiting nodes already in the current path.
#             if neighbor in current_path:
#                 continue
            
#             # Retrieve the angle of the edge (current, neighbor); skip the edge if not found.
#             if 'angle_deg' in G[current][neighbor]:
#                 edge_angle = G[current][neighbor]['angle_deg']
#             else:
#                 continue  # or assign a default value
            
#             # For the first edge there's no previous angle
#             if prev_angle is not None:
#                 # Check if the turning angle is within the allowed maximum.
#                 if angle_diff(prev_angle, edge_angle) > max_angle:
#                     continue  # Skip this edge as it violates the angle constraint
            
#             # Recursively extend the current path.
#             current_path.append(neighbor)
#             dfs(neighbor, current_path, edge_angle)
#             current_path.pop()

#     dfs(start, [start], None)
#     return best_path


