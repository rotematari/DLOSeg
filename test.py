import networkx as nx
import numpy as np
import math
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean

def simplify_intersections_fast_2(G: nx.Graph, dist_threshold=1.0) -> nx.Graph:
    """
    Simplifies a graph by clustering nodes based on proximity and representing
    each cluster with its most central node (closest to the geometric centroid).

    Steps:
    1. Extract node positions and filter nodes without 'x' or 'y' attributes.
    2. Cluster nodes using DBSCAN based on Euclidean distance and dist_threshold.
    3. For each cluster, find the node closest to the geometric centroid.
    4. Create a new graph with these representative nodes.
    5. Add edges between representative nodes if their original clusters were connected in G.
    6. Calculate the angle (in degrees) for each edge in the simplified graph.

    Args:
        G: The input NetworkX graph. Nodes are expected to have 'x' and 'y'
           attributes representing their coordinates.
        dist_threshold: The maximum distance between nodes for them to be
                        considered part of the same cluster (epsilon for DBSCAN).

    Returns:
        A new NetworkX graph containing the simplified representation.
        Nodes in the new graph will have 'x', 'y', and 'original_nodes' attributes.
        Edges will have an 'angle' attribute. Returns an empty graph if no
        nodes with coordinates are found.
    """
    # --- 1. Extract Node Positions ---
    nodes_with_coords = []
    node_coords = {}
    node_list = list(G.nodes(data=True)) # Get nodes with data

    for node, data in node_list:
        # Check if 'x' and 'y' attributes exist and are numeric
        x = data.get('x')
        y = data.get('y')
        if x is not None and y is not None and isinstance(x, (int, float)) and isinstance(y, (int, float)):
            nodes_with_coords.append(node)
            node_coords[node] = (x, y)
        # else:
        #     print(f"Warning: Node {node} skipped due to missing/invalid coordinates.") # Optional warning

    if not nodes_with_coords:
        print("Warning: No nodes with valid 'x' and 'y' coordinates found.")
        return nx.Graph() # Return empty graph if no valid nodes

    # Create a mapping from index in coords_array back to node ID
    index_to_node = {i: node for i, node in enumerate(nodes_with_coords)}
    coords_array = np.array([node_coords[node] for node in nodes_with_coords])

    # --- 2. Cluster Nodes using DBSCAN ---
    # min_samples=1 ensures that every point is assigned to a cluster (or noise -1)
    db = DBSCAN(eps=dist_threshold, min_samples=1).fit(coords_array)
    labels = db.labels_ # Cluster labels for each point

    # Group nodes by cluster label
    clusters = {}
    for i, label in enumerate(labels):
        node = index_to_node[i]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)

    # --- 3. Find Central Node per Cluster ---
    representative_nodes = {} # Map cluster label to representative node ID
    original_to_representative = {} # Map original node ID to its representative
    simplified_G = nx.Graph()

    for label, cluster_nodes in clusters.items():
        if label == -1: # Handle noise points (each becomes its own representative)
            for node in cluster_nodes:
                representative_node_id = f"repr_{node}" # Create a unique ID
                representative_nodes[node] = representative_node_id # Use original node as 'cluster label' key
                original_to_representative[node] = representative_node_id
                # Add node to simplified graph with original coords and original node list
                simplified_G.add_node(
                    representative_node_id,
                    x=node_coords[node][0],
                    y=node_coords[node][1],
                    original_nodes=[node] # List containing only itself
                )
        else: # Handle actual clusters
            if not cluster_nodes:
                continue

            cluster_coords = np.array([node_coords[node] for node in cluster_nodes])
            # Calculate geometric centroid
            centroid = np.mean(cluster_coords, axis=0)

            # Find the node closest to the centroid
            min_dist = float('inf')
            central_node = cluster_nodes[0] # Default to first node
            for node in cluster_nodes:
                dist = euclidean(node_coords[node], centroid)
                if dist < min_dist:
                    min_dist = dist
                    central_node = node

            # Use the central node as the representative for this cluster
            representative_node_id = f"repr_{central_node}" # Use central node's ID in the new name
            representative_nodes[label] = representative_node_id
            for node in cluster_nodes:
                original_to_representative[node] = representative_node_id

            # Add representative node to the simplified graph
            simplified_G.add_node(
                representative_node_id,
                x=node_coords[central_node][0],
                y=node_coords[central_node][1],
                original_nodes=list(cluster_nodes) # Store all original nodes
            )

    # --- 4. Add Edges based on Original Connectivity ---
    added_edges = set() # Keep track of added edges to avoid duplicates

    for u_orig, v_orig in G.edges():
        # Find representatives for the original edge's nodes
        u_repr = original_to_representative.get(u_orig)
        v_repr = original_to_representative.get(v_orig)

        # Check if both nodes have representatives and they are different
        if u_repr is not None and v_repr is not None and u_repr != v_repr:
            # Ensure edge order doesn't matter for the check
            edge_tuple = tuple(sorted((u_repr, v_repr)))

            if edge_tuple not in added_edges:
                # Get coordinates of representative nodes
                coords_u = (simplified_G.nodes[u_repr]['x'], simplified_G.nodes[u_repr]['y'])
                coords_v = (simplified_G.nodes[v_repr]['x'], simplified_G.nodes[v_repr]['y'])

                # Calculate angle
                delta_x = coords_v[0] - coords_u[0]
                delta_y = coords_v[1] - coords_u[1]
                angle = math.degrees(math.atan2(delta_y, delta_x))

                # Add edge with angle attribute
                simplified_G.add_edge(u_repr, v_repr, angle=angle)
                added_edges.add(edge_tuple) # Mark edge as added

    return simplified_G

# --- Example Usage ---
if __name__ == '__main__':
    # Create a sample graph
    G_orig = nx.Graph()
    # Cluster 1
    G_orig.add_node(1, x=1.0, y=1.0)
    G_orig.add_node(2, x=1.1, y=1.2)
    G_orig.add_node(3, x=0.9, y=0.8)
    # Cluster 2
    G_orig.add_node(4, x=5.0, y=5.0)
    G_orig.add_node(5, x=5.1, y=4.9)
    # Outlier
    G_orig.add_node(6, x=10.0, y=10.0)
    # Node without coords
    G_orig.add_node(7)
    # Node with partial coords
    G_orig.add_node(8, x=15.0)


    # Add edges
    G_orig.add_edge(1, 2) # Within cluster 1
    G_orig.add_edge(1, 3) # Within cluster 1
    G_orig.add_edge(4, 5) # Within cluster 2
    G_orig.add_edge(2, 4) # Between cluster 1 and 2
    G_orig.add_edge(3, 5) # Between cluster 1 and 2
    G_orig.add_edge(5, 6) # Between cluster 2 and outlier
    G_orig.add_edge(1, 7) # Edge involving node without coords (will be ignored in simplification)
    G_orig.add_edge(6, 8) # Edge involving node with partial coords (will be ignored)


    print(f"Original Graph:")
    print(f"Nodes: {G_orig.nodes(data=True)}")
    print(f"Edges: {G_orig.edges()}")
    print("-" * 20)

    # Simplify the graph
    simplified_graph = simplify_intersections_fast_2(G_orig, dist_threshold=0.5)

    print(f"Simplified Graph:")
    print(f"Nodes: {simplified_graph.nodes(data=True)}")
    print(f"Edges: {simplified_graph.edges(data=True)}") # Print edges with angle data

    # --- Verification ---
    # Expected representative nodes (approximate, depends on exact centroid/closest logic):
    # Cluster 1 (nodes 1, 2, 3) -> centroid near (1.0, 1.0), node 1 is likely closest. Repr: repr_1
    # Cluster 2 (nodes 4, 5) -> centroid near (5.05, 4.95), node 5 is likely closest. Repr: repr_5
    # Outlier (node 6) -> treated as its own cluster. Repr: repr_6
    # Nodes 7, 8 ignored.

    # Expected edges in simplified graph:
    # repr_1 <-> repr_5 (because 2-4 and 3-5 existed)
    # repr_5 <-> repr_6 (because 5-6 existed)

    # Check number of nodes
    expected_nodes = 3 # repr_1, repr_5, repr_6
    print(f"\nExpected simplified nodes: {expected_nodes}, Found: {simplified_graph.number_of_nodes()}")
    assert simplified_graph.number_of_nodes() == expected_nodes

    # Check number of edges
    expected_edges = 2 # (repr_1, repr_5), (repr_5, repr_6)
    print(f"Expected simplified edges: {expected_edges}, Found: {simplified_graph.number_of_edges()}")
    assert simplified_graph.number_of_edges() == expected_edges

    # Check specific edges and angles (angles depend on chosen representatives)
    if simplified_graph.has_edge('repr_1', 'repr_5'):
        print("Edge (repr_1, repr_5) exists.")
        print(f"  Angle: {simplified_graph.edges['repr_1', 'repr_5']['angle']:.2f} degrees")
    if simplified_graph.has_edge('repr_5', 'repr_6'):
        print("Edge (repr_5, repr_6) exists.")
        print(f"  Angle: {simplified_graph.edges['repr_5', 'repr_6']['angle']:.2f} degrees")
