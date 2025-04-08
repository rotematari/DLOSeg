import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
def convert_mask_to_graph(mask):
    """
    Convert a skeleton mask to a graph representation.
    
    Args:
        mask_graph (numpy.ndarray): A binary mask where the skeleton is represented by 1s.
        
    Returns:
        edge_list (list): List of edges in the graph.
        node_indices (dict): Mapping from node coordinates to indices.
    """
    # Ensure the input is a binary mask
    mask = (mask > 0).astype(np.uint8)

    # Check if the input is a valid binary mask
    if not np.any(mask):
        raise ValueError("The input mask must contain at least one non-zero element.")
        
    # Convert skeleton mask to graph
    nodes = np.argwhere(mask == 1)
    node_indices = {tuple(node): idx for idx, node in enumerate(nodes)}

    edge_list = []
    directions = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),          (0, 1),
                (1, -1),  (1, 0), (1, 1)]

    for y, x in nodes:
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                if mask[ny, nx]:
                    edge_list.append([node_indices[(y, x)], node_indices[(ny, nx)]])
                    
    return edge_list, node_indices



def convert_mask_to_graph_fast(mask):
    """
    Convert a binary skeleton mask to a graph (edges + node indices).
    
    Args:
        mask (numpy.ndarray): Binary 2D mask (skeleton).
        
    Returns:
        edge_list (list of [int, int]): List of edges (as index pairs).
        node_indices (dict): Mapping from (y, x) coordinates to node index.
    """
    mask = (mask > 0).astype(np.uint8)
    if not np.any(mask):
        raise ValueError("The input mask must contain at least one non-zero element.")

    coords = np.argwhere(mask)
    node_indices = {tuple(coord): idx for idx, coord in enumerate(coords)}
    
    edge_list = []
    for y, x in coords:
        # Iterate over 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in node_indices:
                    edge_list.append([node_indices[(y, x)], node_indices[(ny, nx)]])

    return edge_list, node_indices

def binary_mask_to_graph(mask, connectivity=4):
    """
    Converts a binary mask into a graph using given connectivity.

    Parameters:
        mask (np.ndarray): 2D binary mask (dtype=bool or 0/1)
        connectivity (int): 4 or 8 (pixel connectivity)

    Returns:
        G (networkx.Graph): Graph with pixel (row, col) as nodes
    """
    assert connectivity in (4, 8), "Connectivity must be 4 or 8"
    
    hight, width = mask.shape
    mask = np.asarray(mask).astype(bool)

    nodes = list(zip(*np.nonzero(mask))) 
    # node_indices = {node: idx for idx, node in enumerate(nodes)}

    G = nx.Graph()
    G.add_nodes_from(nodes)

    edge_list = []

    for y,x in nodes:
        if y == 0 or y == hight-1 or x == 0 or x == width-1:
            continue
        # Check the 3x3 neighborhood
        view = mask[y - 1:y + 2, x - 1:x + 2]
        neighbors = np.argwhere(view)
        for neighbor in neighbors:
            ny, nx_ = neighbor
            ny += y - 1
            nx_ += x - 1
            # Skip the center pixel itself
            if (ny, nx_) == (y, x):
                continue
            if 0 <= ny < hight and 0 <= nx_ < width:
                if mask[ny, nx_]:
                    edge_list.append([(y, x), (ny, nx_)])
    G.add_edges_from(edge_list)

    return G


def binary_mask_to_graph_fast(mask, connectivity=4):
    assert connectivity in (4, 8), "Connectivity must be 4 or 8"

    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    G = nx.Graph()

    ys, xs = np.nonzero(mask)
    nodes = list(zip(ys, xs))
    G.add_nodes_from(nodes)

    edges = []

    # Define neighbor offsets based on connectivity
    if connectivity == 4:
        offsets = [(0, 1), (1, 0)]  # Only right and bottom neighbors
    else:  # connectivity == 8
        offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dy, dx in offsets:
        y_neighbors, x_neighbors = ys + dy, xs + dx

        valid = (
            (y_neighbors >= 0) & (y_neighbors < h) &
            (x_neighbors >= 0) & (x_neighbors < w)
        )

        src = np.array(nodes)[valid]
        dst = np.column_stack((y_neighbors[valid], x_neighbors[valid]))

        mask_src = mask[src[:, 0], src[:, 1]]
        mask_dst = mask[dst[:, 0], dst[:, 1]]

        edge_mask = mask_src & mask_dst
        edges.extend(zip(map(tuple, src[edge_mask]), map(tuple, dst[edge_mask])))

    G.add_edges_from(edges)

    return G
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

    # Step 3: define neighbors
    if connectivity == 4:
        offsets = [(0, 1), (1, 0)]
    else:
        offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]

    # Step 4: find and add valid edges using indices
    for dy, dx in offsets:
        for y, x in coords:
            ny, nx_ = y + dy, x + dx
            neighbor = (ny, nx_)
            if 0 <= ny < h and 0 <= nx_ < w and mask[ny, nx_]:
                src_idx = coord_to_index[(y, x)]
                dst_idx = coord_to_index[neighbor]
                # Get positions
                # pos_u = np.array((x, y))
                # pos_v = np.array((nx_, ny))

                # # Compute angle
                # vec = pos_v - pos_u
                # angle_rad = np.arctan2(vec[1], vec[0])
                # angle_deg = np.degrees(angle_rad)
                
                G.add_edge(src_idx, dst_idx)

    return G


def binary_mask_to_graph_2(mask, connectivity=4):
    assert connectivity in (4, 8), "Connectivity must be 4 or 8"

    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    G = nx.Graph()

    ys, xs = np.nonzero(mask)
    nodes = list(zip(ys, xs))
    G.add_nodes_from(nodes)

    edges = []

    # Define proper neighbor offsets
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else \
              [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),          (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    node_set = set(nodes)

    for y, x in nodes:
        for dy, dx in offsets:
            ny, nx_ = y + dy, x + dx
            neighbor = (ny, nx_)
            if neighbor in node_set:
                if (y, x) < neighbor:  # avoid duplicates
                    edges.append([(y, x), neighbor])

    G.add_edges_from(edges)
    return G
def draw_nx_graph(graph, img_size=(621, 1104)):
    plt.figure(figsize=(8, 8))

    nodes = np.asarray(graph.nodes())
    plt.scatter(nodes[:, 1], img_size[0] - nodes[:, 0], s=1, c='red')

    # Plot edges individually
    for (y1, x1), (y2, x2) in graph.edges():
        plt.plot([x1, x2], [img_size[0] - y1, img_size[0] - y2], color='blue', linewidth=0.5)

    plt.axis('equal')
    plt.show()
    

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from collections import deque
def get_angle_rad(pos_from:tuple[int,int],pos_to:tuple[int,int])-> float:
    dx = pos_from[0] - pos_to[0]
    dy = pos_from[1] - pos_to[1]
    return np.arctan2(abs(dy), abs(dx))


def simplify_intersections_fast_2(G: nx.Graph, dist_threshold=1.0):
    
    

    
    simplified_G = None
    return simplified_G
def simplify_intersections_fast(G: nx.Graph, threshold=1.0):
    # Extract node indices and their positions
    node_indices = np.array(G.nodes())
    positions = np.array([G.nodes[n]['pos'] for n in node_indices])

    # Build KD-tree using positions
    tree = cKDTree(positions)

    visited = np.zeros(len(positions), dtype=bool)
    clusters = []

    # # Identify clusters based on threshold distance
    # for i in range(len(positions)):
    #     if visited[i]:
    #         continue
    #     cluster_idx = tree.query_ball_point(positions[i], threshold)
    #     clusters.append(cluster_idx)
    #     visited[cluster_idx] = True
        # Flood-fill clustering
    for i in range(len(positions)):
        if visited[i]:
            continue
        cluster = []
        queue = deque([i])
        while queue:
            idx = queue.popleft()
            if visited[idx]:
                continue
            visited[idx] = True
            cluster.append(idx)
            # Get all neighbors within threshold for the current node
            neighbors = tree.query_ball_point(positions[idx], threshold)
            for nb in neighbors:
                if not visited[nb]:
                    queue.append(nb)
        clusters.append(cluster)
    # Compute new node positions (cluster centroids)
    old_to_new = {}
    new_nodes = []
    # for cluster in clusters:
    #     cluster_pos = positions[cluster]
    #     centroid = np.round(cluster_pos.mean(axis=0)).astype(int)
    #     new_node_idx = len(new_nodes)
    #     new_nodes.append((new_node_idx, {'pos': tuple(centroid)}))
    #     for idx in cluster:
    #         old_to_new[node_indices[idx]] = new_node_idx
    for cluster in clusters:
        # Get positions of nodes in the cluster
        cluster_pos = positions[cluster]
        
        # Compute the centroid (using the actual mean, not rounded)
        centroid = cluster_pos.mean(axis=0)
        
        # Compute Euclidean distances from each node in the cluster to the centroid
        distances = np.linalg.norm(cluster_pos - centroid, axis=1)
        
        # Identify the index of the node closest to the centroid within the cluster
        min_index = np.argmin(distances)
        representative_idx = cluster[min_index]
        
        new_node_idx = len(new_nodes)
        
        # Use the position of the representative node (convert to int if needed)
        rep_pos = positions[representative_idx]
        new_nodes.append((new_node_idx, {'pos': tuple(rep_pos.astype(int))}))
        
        # Map all original nodes in this cluster to the new node index
        for idx in cluster:
            old_to_new[node_indices[idx]] = new_node_idx
    
    # Rebuild edges with updated nodes
    # new_edges = set()
    simplified_G = nx.Graph()
    for src, dst in G.edges():
        new_src = old_to_new[src]
        new_dst = old_to_new[dst]
        if new_src != new_dst:
            angle_rad = get_angle_rad(G.nodes[new_src]['pos'], G.nodes[new_dst]['pos'])
            edge = tuple(sorted((new_src, new_dst)))
            # new_edges.add(edge)
            simplified_G.add_edge(*edge, angle_rad=angle_rad)

    # Construct simplified graph
    
    simplified_G.add_nodes_from(new_nodes)
    # simplified_G.add_edges_from(new_edges)

    return simplified_G

def draw_graph(edge_list, node_indices):
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # Reverse node_indices to get positions
    pos = {idx: coord[::-1] for coord, idx in node_indices.items()}  # flip (y, x) to (x, y) for display

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, node_size=30, with_labels=False, edge_color='gray')
    # plt.axis("equal")
    plt.show()
    
import heapq
import numpy as np
import networkx as nx

def angle_diff(a1, a2):
    """Returns smallest absolute angle difference in radians."""
    return abs(a2 - a1 )
    # """Returns the directed angle difference in radians in the range [0, 2pi).
    
    # Here, if a2 equals a1 then the result is 0. 
    # A positive result means a counter-clockwise turn from a1 to a2.
    # """
    # return (a2 - a1) % (2 * np.pi)

def find_smoothest_path(G, start_node, goal_node):
    # pos = nx.get_node_attributes(G, 'pos')
    edge_angles = nx.get_edge_attributes(G, 'angle_rad')

    # Each state is (cost, current_node, incoming_angle, path)
    heap = [(0, start_node, None, [start_node])]
    visited = dict()

    while heap:
        cost, current, in_angle, path = heapq.heappop(heap)

        # Check goal
        if current == goal_node:
            return path

        state_key = (current, in_angle if in_angle is not None else None)
        if state_key in visited and visited[state_key] <= cost:
            continue
        visited[state_key] = cost

        for neighbor in G.neighbors(current):
            # Edge can be (current, neighbor) or (neighbor, current)
            edge = (current, neighbor) if (current, neighbor) in edge_angles else (neighbor, current)
            out_angle = edge_angles[edge]

            if in_angle is None:
                angle_cost = 0  # First step, no turning cost
            else:
                angle_cost = angle_diff(in_angle, out_angle)

            total_cost = cost + angle_cost
            heapq.heappush(heap, (total_cost, neighbor, out_angle, path + [neighbor]))

    return None  # No path found


# find the smothest path between length n from a node 
def find_smoothest_path_length(G, start_node, length, goal_node, last_path=None):
    # pos = nx.get_node_attributes(G, 'pos')
    edge_angles = nx.get_edge_attributes(G, 'angle_rad')

    # Each state is (cost, current_node, incoming_angle, path)
    heap = [(0, start_node, None, [start_node])]
    visited = dict()

    while heap:
        cost, current, in_angle, path = heapq.heappop(heap)


        # Check goal
        if len(path) == length or (current == goal_node and goal_node is not None):
            return path

        state_key = (current, in_angle if in_angle is not None else None)
        if state_key in visited and visited[state_key] <= cost:
            continue
        visited[state_key] = cost

        for neighbor in G.neighbors(current):
            # # Skip if the neighbor is the last node in the path
            if len(path) > 1:
                if neighbor in path:
                    continue
            if last_path is not None:
                if neighbor in last_path:
                    continue
            # Edge can be (current, neighbor) or (neighbor, current)
            edge = (current, neighbor) if (current, neighbor) in edge_angles else (neighbor, current)
            out_angle = edge_angles[edge]

            if in_angle is None:
                angle_cost = 0  # First step, no turning cost
            else:
                angle_cost = angle_diff(in_angle, out_angle)

            total_cost = cost + angle_cost
            candidate_path = path + [neighbor]
            # Check if the candidate path is valid
            if candidate_path == last_path:
                continue
            heapq.heappush(heap, (total_cost, neighbor, out_angle,candidate_path ))

    return None  # No path found

# traverse the graph in n size steps
def find_dlo(G, start_node, goal_node):
    path = []
    traverse = True
    small_path = find_smoothest_path_length(G, start_node, 5,goal_node)
    if path is None:
        print("No path found")
        traverse = False
    else:
        path.extend(small_path)
        start_node = small_path[-1]
    while traverse:
        revers_path = small_path[::-1]
        small_path = find_smoothest_path_length(G, start_node, 5,goal_node,last_path=revers_path)
        if small_path is None:
            traverse = False
            print("No path found")
        else:
            path.extend(small_path[1:])
            start_node = small_path[-1]
        if start_node == goal_node:
            traverse = False
            print("Goal reached")
    
    return path