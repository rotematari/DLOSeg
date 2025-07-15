import numpy as np
import networkx as nx
import cv2
import time
from typing import Optional, List, Dict, Any,Tuple
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import CubicHermiteSpline

from graph import bspline_fitting
from itertools import combinations


class DLOGraph:
    """
    A class representing a Deformable Linear Object (DLO) as a graph structure.
    Optimized for processing speed and memory efficiency.
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize a DLOGraph from a binary mask or create an empty graph.

        Parameters:
        -----------
        mask : np.ndarray, optional
            Binary mask where True/1 represents the DLO
        """
        self.config = config if config is not None else {}
        self.G = nx.Graph()
        self._full_coords_array = None
        self.width_px = None
        self.grid_size = None
        self._kdtree = None
        self._coords_array = None
        self._nodes_array = None
        self._mask_bool = None
        self.mask_origin = None
        self.full_bsplins = []
        
        
        self.padding_size = config.get('padding_size', 4)
        self.dialate_iterations = config.get('dialate_iterations', 2)
        self.erode_iterations = config.get('erode_iterations', 2)

        self.max_dist_to_connect_leafs = config.get('max_dist_to_connect_leafs', 20.0)  # Default distance if not provided
        self.max_dist_to_connect_nodes = config.get('max_dist_to_connect_nodes', 5.0)  # Default distance if not provided


    def load_from_mask(self, mask: np.ndarray, config: Dict[str, Any] = None) -> None:
        """
        Load graph from binary mask image.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask where True/1 represents the DLO
        downsample : int
            Downsampling factor for the mask
        """
        # print("\n-----Loading graph from mask...-----\n")
        # pad_time = time.time()
        self.mask_origin = mask.copy()
        mask = cv2.copyMakeBorder(mask, self.padding_size, self.padding_size, self.padding_size, self.padding_size,
                                  cv2.BORDER_CONSTANT, value=0)

        # print(f"Padding completed in {time.time() - pad_time:.3f} seconds")
        
        # --- Dilation and Erosion ---

        # 1. Define a kernel
        # This is a structuring element that determines the nature of the operation.
        # A common choice is a rectangular or elliptical kernel.
        # cv2.getStructuringElement(shape, size)
        # Shape can be: cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE
        
        # erode_dialate_start = time.time()
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # # 2. Apply Dilation
        # # Dilation expands the white regions in the mask.
        
        mask = cv2.dilate(mask, kernel, iterations=self.dialate_iterations)

        # # 3. Apply Erosion
        # # Erosion shrinks the white regions in the mask.
        
        mask = cv2.erode(mask, kernel, iterations=self.erode_iterations)
        # # print(f"Dilation and erosion completed in {time.time() - erode_dialate_start:.3f} seconds")
        # plt.figure(figsize=(10, 8))
        # plt.imshow(self.mask_bool, cmap='gray')
        # plt.title("Processed Mask")
        # plt.show()

        skeletonization_start = time.time()

        # skeletonize 

        # should get 0-255
        mask[mask > 0] = 255
        mask[mask <= 0] = 0
        
        # plt.imshow(mask, cmap='gray')
        # plt.title("Processed Mask")
        # plt.axis('off')
        # plt.show()
        skel = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_GUOHALL)


        self.mask_bool = (skel>0).astype(np.uint8)  # Convert to uint8 for visualization
        # plt.imshow(self.mask_bool, cmap='gray')
        # plt.title("Skeletonized Mask")
        # plt.axis('off')
        # plt.show()
        
        # print(f"Skeletonization completed in {time.time() - skeletonization_start:.3f} seconds")
        # start_get_coords = time.time()
        ys, xs = np.nonzero(self.mask_bool)
        self._coords_array = np.column_stack((xs, ys))
        # print(f"Coordinates extraction completed in {time.time() - start_get_coords:.3f} seconds")
        # plt.imshow(self.skeleton, cmap='gray')
        # plt.title("Skeletonized Mask")
        # plt.show()
        # Create initial graph
        # graph_creation_start = time.time()
        self.G = self._mask_to_simplified_graph()
        # print(f"Graph creation completed in {time.time() - graph_creation_start:.3f} seconds")
        
        # Start timing for spatial index refresh
        # print("Done loading mask\n")
        # self._refresh_spatial_index()


    def _refresh_spatial_index(self) -> None:
        """Rebuild the KD-tree spatial index for the graph nodes."""
        self._pos_dict = nx.get_node_attributes(self.G, "pos")
        self._nodes_array = np.array(list(self._pos_dict.keys()), dtype=int)
        self._coords_array = np.array([self._pos_dict[n] for n in self._nodes_array], dtype=int)

    def _gen_tree(self,G:nx.Graph) -> None:
        """Generate a minimum spanning tree from the graph."""
        return nx.minimum_spanning_tree(G, algorithm='prim')  # 'kruskal', 'prim', or 'boruvka'

    def _mask_to_simplified_graph(self) -> nx.Graph:
        
        edges_start = time.time()
        coords = self._coords_array          # (N,2) array of [x,y]
        H, W = self.mask_bool.shape
        M = coords.shape[0]
        k = 3
        
        tree = KDTree(coords)
        dists, idxs = tree.query(coords, k+1,distance_upper_bound=self.max_dist_to_connect_nodes)  # idxs[:,0] is self
        # Filter out invalid neighbors (distances that are inf or indices that are >= M)
        
        valid_edges = []
        for i in range(M):
            for j in range(1, k+1):  # skip self (j=0)
                neighbor_idx = idxs[i, j]
                neighbor_dist = dists[i, j]
                
                # Check if neighbor is valid (not inf distance and valid index)
                if not np.isinf(neighbor_dist) and neighbor_idx < M:
                    valid_edges.append((i, neighbor_idx, neighbor_dist))
        # print(f"Edge filtering completed in {time.time() - edges_start:.3f} seconds")
        # Create a graph from the valid edges
        graph_time = time.time()
        G = nx.Graph()
        # add all reps as nodes (storing their 2D positions)
        node_creation_start = time.time()
        G.add_nodes_from((i, {"pos": tuple(coords[i])}) for i in range(M))
        # print(f"Node creation completed in {time.time() - node_creation_start:.3f} seconds")
        
        # add weighted edges from the valid k-NN connections
        edge_addition_start = time.time()
        G.add_weighted_edges_from(valid_edges)
        # print(f"Edge addition completed in {time.time() - edge_addition_start:.3f} seconds")
        
        # extract the minimum‐spanning‐tree
        tree_start = time.time()
        T = self._gen_tree(G)  # 'kruskal', 'prim', or 'boruvka'
        # print(f"Tree extraction completed in {time.time() - tree_start:.3f} seconds")
        
        # print(f"Graph creation from edges completed in {time.time() - graph_time:.3f} seconds")
        return T


    def _prune_path_between_3_degree_nodes(self, nodes: list) -> None:
        """
        Prunes paths between nodes of degree 3 in the graph.

        Args:
            nodes: A list of node IDs with degree 3.
            max_length: The maximum length of the path to keep.
        """
        # find all pairs of nodes with degree 3

        available_nodes = set(nodes.copy())
        for node in nodes:
            # print(f"Processing node {node}")
            paths = []
            for target in nodes:
                if node == target:
                    continue
                try:
                    # Find the shortest path between the two nodes
                    if node in available_nodes and target in available_nodes:
                        # Use NetworkX to find the shortest path
                        # print(f"Finding path from {node} to {target}")
                        path = nx.shortest_path(self.G, source=node, target=target)
                        paths.append(path)
                except nx.NetworkXNoPath:
                    continue
                
            # If there are no paths, continue
            if not paths:
                continue
            # Find the shortest path
            shortest_path = min(paths, key=len)
            self.G.remove_nodes_from(shortest_path)  # Remove intermediate nodes
            available_nodes.discard(shortest_path[0])  # Remove the current node from available nodes
            available_nodes.discard(shortest_path[-1])  # Remove the target node from available nodes
            # print(f"Pruned path between {shortest_path[0]} and {shortest_path[-1]}")
            # print(f"Available nodes after pruning: {available_nodes}")

    def _prune_short_branches(self, G: nx.Graph, min_length: int) -> None:
        """
        Prunes short, dead-end branches from a graph.

        A branch is a path of nodes with degree 2, ending in a leaf node (degree 1).
        This function removes branches where the path length is less than `min_length`.

        Args:
            G: The graph to prune, modified in-place.
            min_length: The minimum number of nodes a branch must have to be kept.
                        A branch of length 1 is a single leaf connected to a junction.
        """
        # Use a set for efficient additions and membership testing
        nodes_to_prune = set()
        
        # A set to ensure we don't traverse the same branch from different ends
        visited_branch_nodes = set()

        # Find all leaf nodes (degree 1) to start our traversals from
        leaf_nodes = [n for n in G.nodes if G.degree[n] == 1]

        for leaf in leaf_nodes:
            if leaf in visited_branch_nodes:
                continue

            path = [leaf]
            curr = leaf
            prev = None

            while True:
                # Mark the current node as visited for this branch traversal
                visited_branch_nodes.add(curr)
                
                # Find the next node in the path that isn't the previous one
                neighbors = [n for n in G.neighbors(curr) if n != prev]

                # A proper branch node has exactly one "forward" neighbor and a degree of 2.
                # The starting leaf has a degree of 1.
                # If we find more than one neighbor, or the neighbor has a degree > 2,
                # we've hit a junction and the end of the branch.
                if len(neighbors) != 1 or G.degree(neighbors[0]) != 2:
                    # End of the branch found
                    break

                prev = curr
                curr = neighbors[0]
                path.append(curr)

            # After finding the full branch, check if it's too short
            if len(path) < min_length:
                nodes_to_prune.update(path)

        # Remove all identified short-branch nodes at once
        G.remove_nodes_from(nodes_to_prune)
    def prune_short_branches_and_delete_junctions(self,max_length: int) -> None:


        """
        prune leaf branches longer than max_length
        """

            
        
        # connect leaf nodes if they close enough

        leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]
        
        leaf_positions = np.array([self.G.nodes[n]["pos"] for n in leaf_nodes])
        all_poses = np.array([self.G.nodes[n]["pos"] for n in self.G.nodes])
        leaf_tree = KDTree(leaf_positions)
        # find pairs of leaf nodes that are close enough
        dist, pairs_index = leaf_tree.query(leaf_positions, k=2, distance_upper_bound=self.max_dist_to_connect_leafs)
        pairs = [(leaf_nodes[i], leaf_nodes[j]) for i, j in pairs_index if j < len(leaf_nodes) and i < len(leaf_nodes)]
        
        # connect leaf nodes that are close enough
        for i, j in pairs:
            if self.G.has_edge(i, j):
                continue
            norm = np.linalg.norm(all_poses[i] - all_poses[j])
            if norm <= 3:
                self.G.add_edge(i, j)

        self._prune_short_branches(self.G, max_length)
        leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]
        
        # save leaf nodes positions for later use
        self.ends_leaf_nodes = leaf_nodes
        self.leaf_nodes_poses = np.array([self.G.nodes[n]["pos"] for n in leaf_nodes])
        

        three_degree_nodes = [n for n in self.G.nodes if self.G.degree[n] == 3]
        if len(three_degree_nodes) > 0:
            self._prune_path_between_3_degree_nodes(three_degree_nodes)

        # remove junction nodes (nodes with degree > 2)
        junction_nodes = [n for n in self.G.nodes if self.G.degree[n] > 2]
        self.G.remove_nodes_from(junction_nodes)
        self._prune_short_branches(self.G, max_length)
        # prune 0 degree nodes
        zero_degree_nodes = [n for n in self.G.nodes if self.G.degree[n] == 0]
        self.G.remove_nodes_from(zero_degree_nodes)


    def _extract_branches(self) -> List[List[int]]:
        """
        Extract branches from the graph.
        A branch is defined as a path between nodes with degree == 1.
        
        the input Graph should have only nodes with degree 1 or 2.
        """
        # find all nodes with degree 1
        leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]
        branches = []
        visited = set()
        for leaf in leaf_nodes:
            if leaf in visited:
                continue
            branch = nx.dfs_tree(self.G, source=leaf)
            branches.append(list(branch))
            visited.update(branch)

        return branches
    def _create_graph_from_branches(self, branches: List[np.ndarray]) -> nx.Graph:
        """
        Create a graph from a list of 2D branches, as fast as possible.

        branches : List of (Ni×2) arrays of (x,y) points
        """
        """
        branches : List of (Ni×2) arrays of (x,y) points
        """
        nodes = []
        edges = []
        offset = 0
        leaf_nodes = set()
        for branch in branches:
            L = branch.shape[0]
            # 1) add L nodes, with positions in their attr dict
            for i in range(L):
                node_id = offset + i
                pos = tuple(branch[i])
                nodes.append((node_id, {'pos':pos }))
                # check if it is a leaf node
                for leaf_pos in self.leaf_nodes_poses:
                    if i == 0 or i == L - 1:
                        dist = np.linalg.norm(np.array(pos) - np.array(leaf_pos))
                        if dist < 3:
                            # if the node is close enough to a leaf node, mark it as a leaf node
                            leaf_nodes.add(node_id)
                            
                

            # 2) add the L-1 edges along this branch
            #    using a list comprehension that yields tuples
            edges.extend([
                (offset + i, offset + i + 1)
                for i in range(L - 1)
            ])

            offset += L

        # 3) bulk‐create the graph
        G = nx.Graph()
        G.add_nodes_from(nodes)  # nodes is a List[(int, dict)]
        G.add_edges_from(edges)  # edges is a List[(int, int)]
        self.ends_leaf_nodes = list(leaf_nodes)

        return G
    
    def fit_spline_to_branches(self,
                               smoothing: float = 2.0,
                               max_num_points: Optional[int] = None) -> None:

        """
        Fit B-spline to each branch in the graph.
        """
        # ex_time = time.time()
        # branches = self._extract_branches()
        # print(f"Extracted {len(branches)} branches in {time.time() - ex_time:.3f} seconds")
        # ex_time = time.time()
        branches = self._extract_branches()
        # print(f"Extracted_0 {len(branches)} branches in {time.time() - ex_time:.3f} seconds")
        
        branches_coords = []
        for branch in branches:
            coords = np.array([self.G.nodes[n]["pos"] for n in branch])
            branches_coords.append(coords)
        
        # smooth_time = time.time()
        # Fit B-spline to each branch
        smoothed_branches = []
        
        for i, coords in enumerate(branches_coords):
            if len(coords) < 5:
                # Skip branches that are too short to fit a spline
                continue
            smooth_2d_branch = bspline_fitting.smooth_2d_branch_splprep(coords=coords,
                s=smoothing,
                num_samples=max_num_points
            )
            smoothed_branches.append(smooth_2d_branch)
        
        # print(f"Fitted {len(smoothed_branches)} branches in {time.time() - smooth_time:.3f} seconds")
        
        start_generation = time.time()
        # Generate new graph from the smoothed branches
        self.G = self._create_graph_from_branches(smoothed_branches)
        # print(f"Generated graph from branches in {time.time() - start_generation:.3f} seconds")

    
    def _get_unit_vector(self,v: np.ndarray) -> np.ndarray:
        """
            Normalize a vector to unit length.
        """
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _process_leaf_cluster(self, cluster: List[int]) -> None:
        """
        Process a cluster of leaf nodes to find the best configuration for connecting them.
        
        Parameters:
        -----------
        cluster : List[int]
            List of node IDs representing the cluster of leaf nodes
            
        Returns:
        --------
        None
        """
        if len(cluster) < 2:
            return
        
        # i have 2^len(cluster) (witch will be 4) configurations
        # so i want to caculate the sum of angels for each configuration 
        # for each node caculate the vector to the neighbour 
        def _calc_angle_sum(node_1,node_2):
                node_1_pos = np.array(self.G.nodes[node_1]["pos"])
                node_1_nbr = next(iter(self.G.neighbors(node_1)), None)
                if node_1_nbr is None:
                    raise ValueError(f"Node {node_1} has no neighbors.")
                node_1_nbr_pos = np.array(self.G.nodes[node_1_nbr]["pos"])

                tan0 = self._get_unit_vector(node_1_pos - node_1_nbr_pos)
                # tangent vector from the start node to the candidate.
                node_2_pos = np.array(self.G.nodes[node_2]["pos"])
                tan1 = self._get_unit_vector(node_2_pos - node_1_pos)
                cand_nbr = next(iter(self.G.neighbors(node_2)), None)
                
                # tangent vector from the candidate to its neighbor.
                tan2 = self._get_unit_vector(np.array(self.G.nodes[cand_nbr]["pos"]) - node_2_pos)

                angle1 = np.rad2deg(np.arccos(np.clip(np.dot(tan0, tan1), -1.0, 1.0)))
                angle2 = np.rad2deg(np.arccos(np.clip(np.dot(tan1, tan2), -1.0, 1.0)))
                return angle1 + angle2

        # get all configurations 

        combs = list(combinations(cluster, 2))
        # find the 2 best combinations
        angle_sums = {}
        # find the best combination
        combs_of_combs = list(combinations(combs, 2))
        # keep only those where the two pairs are disjoint
        combs_of_combs = [
            (p1, p2)
            for p1, p2 in combs_of_combs
            if len({*p1, *p2}) == 4
        ]
        for comb in combs_of_combs:
            angle_sum = _calc_angle_sum(comb[0][0], comb[0][1]) + _calc_angle_sum(comb[1][0], comb[1][1])
            angle_sums[comb] = angle_sum
            # print(f"Combination: {comb}, Angle Sum: {angle_sum}")
        best_comb = min(angle_sums, key=angle_sums.get)
        # print(f"Best Combination: {best_comb}, Angle Sum: {angle_sums[best_comb]}")
        for comb in best_comb:
            if not self.G.has_edge(comb[0], comb[1]) and comb[0] != comb[1]:
                if comb[0] not in self.ends_leaf_nodes and comb[1] not in self.ends_leaf_nodes:
                    self.G.add_edge(comb[0], comb[1])
        # print("done")

    def reconstruct_dlo_2(self):
        
        """
        Find clusters of leaf nodes and find the best configuration for connecting them.
        """
        leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1 
                      and n not in self.ends_leaf_nodes]
        leaf_poses = np.array([self.G.nodes[n]["pos"] for n in leaf_nodes])
        
        # cluster leaf nodes 
        
        tree = KDTree(leaf_poses)
        # query for clusters
        dists, cluster_labels = tree.query(leaf_poses, k=4,distance_upper_bound=self.max_dist_to_connect_leafs)
        cluster_labels = [sorted(cluster) for i,cluster in enumerate(cluster_labels) if all(dists[i] < self.max_dist_to_connect_leafs )]
        unique = [list(t) for t in set(tuple(lst) for lst in cluster_labels)]
        leaf_clusters = []
        for cluster in unique:
            cluster_nodes = [leaf_nodes[i] for i in cluster]
            leaf_clusters.append(cluster_nodes)

        for cluster in leaf_clusters:
            self._process_leaf_cluster(cluster)



    def fit_bspline_to_graph(self):
        leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]
        comb = list(combinations(leaf_nodes, 2))
        ends_pairs = []
        
        for u, v in comb:
            try:
                dlo_path = np.array(list(nx.shortest_path(self.G, source=u, target=v)))
            except nx.NetworkXNoPath:
                dlo_path = np.array([])
                # print(f"No path found between {u} and {v}.")
                continue
            
            if dlo_path.ndim > 1:
                dlo_path = dlo_path[0]
            path_coords = np.array([self.G.nodes[n]["pos"] for n in dlo_path])
        
            # spl, fitted_points = bspline_fitting.fit_bspline(path_coords,n_points=500,k=5)

            self.full_bsplins.append(path_coords)
        # print(f"Fitted B-spline with {len(fitted_points)} points.")

    def visualize(self, figsize: Tuple[int, int] = (10, 10), 
                node_size: int = 5, with_labels: bool = False,title: str="") -> None:
        """
        Visualize the graph.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        node_size : int
            Size of nodes in visualization
        with_labels : bool
            Whether to show node labels
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)

        pos = nx.get_node_attributes(self.G, 'pos')
        if self.config['on_mask']:
            # 1. Draw the mask as background
            ax.imshow(self.mask_origin, cmap='gray')
        else:
            ax.invert_yaxis()  # Flip Y axis to match image coordinates
        nx.draw(self.G, pos=pos, node_size=node_size, node_color='green',
                    edge_color='red', with_labels=with_labels, ax=ax)
        ax.set_axis_on()
        # 3. Draw grid underneath
        ax.set_axisbelow(True)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
        plt.pause(0.01)
        return fig, ax
    
        
    def visualize_on_mask(self, figsize: Tuple[int, int] = (10, 10), 
                node_size: int = 5, with_labels: bool = False,title: str="") -> None:
        """
        Visualize the graph.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        node_size : int
            Size of nodes in visualization
        with_labels : bool
            Whether to show node labels
        """
        # fig, ax = plt.subplots(figsize=figsize)
        # ax.set_title(title)
        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, pos=pos, node_size=node_size, 
               edge_color='red', with_labels=with_labels)
        # ax.invert_yaxis()  # Flip Y axis to match image coordinates
        plt.pause(0.01)
        return 

