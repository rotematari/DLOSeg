import numpy as np
import networkx as nx
import cv2
import time
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Literal, Set, Any, Union

from scipy.spatial import cKDTree
from skimage.draw import line
import torch
import torch.nn.functional as F
import logging
from graph import bspline_fitting
import math
from scipy.interpolate import splprep, splev
from dataclasses import dataclass
@dataclass
class _EndPoint:
    node: int           # graph node id
    pos:  np.ndarray    # (2,) or (3,)
    frag: int           # connected component id
    

class DLOGraph:
    """
    A class representing a Deformable Linear Object (DLO) as a graph structure.
    Optimized for processing speed and memory efficiency.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize a DLOGraph from a binary mask or create an empty graph.

        Parameters:
        -----------
        mask : np.ndarray, optional
            Binary mask where True/1 represents the DLO
        """
        self.G = nx.Graph()
        self._full_coords_array = None
        self.width_px = None
        self.grid_size = None
        self._kdtree = None
        self._coords_array = None
        self._nodes_array = None
        self._mask_bool = None



    def load_from_mask(self, mask: np.ndarray,
                       statistic: Literal["median", "mean", "min", "max"] = "mean",
                       min_cluster_factor: float = 2.0,
                       padding_size: int = 20) -> None:
        """
        Load graph from binary mask image.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask where True/1 represents the DLO
        downsample : int
            Downsampling factor for the mask
        """
        # Timer for performance tracking
        start_time = time.time()
        h, w = mask.shape

        # pad the mask
        # Pad 10 pixels on all sides with black
        padd_time = time.time()
        mask = cv2.copyMakeBorder(mask, padding_size, padding_size, padding_size, padding_size,
                                  cv2.BORDER_CONSTANT, value=0)
        self.mask_bool = (mask>0).view(np.uint8)
        # print(f"Mask padding completed in {time.time() - padd_time:.3f} seconds")
        
        
        
        # Extract foreground pixels
        # self.dlo_pixels = np.column_stack(np.nonzero(self.mask_bool))
        # Make it [x, y] order to match coords
        start_time = time.time()
        
        # ys, xs = np.nonzero(self.mask_bool)
        # self._full_coords_array = np.column_stack((xs, ys))
        
        # print(f"Mask coordinates extraction completed in {time.time() - start_time:.3f} seconds")
        # Pre-compute this once in the initialization
        # Start timing for width estimation

        # Estimate DLO width
        width_estimation_start = time.time()
        self.width_px , self.mask_bool = self._estimate_dlo_width(statistic=statistic)
        # self.width_px = h/150
        ys, xs = np.nonzero(self.mask_bool)
        self._coords_array = np.column_stack((xs, ys))
        # print(f"DLO width estimation completed in {time.time() - width_estimation_start:.3f} seconds")

        
        # Start timing for graph creation
        graph_creation_start = time.time()
        
        # Create initial graph
        self.G = self._mask_to_simplified_graph_new(min_cluster_factor=min_cluster_factor)

        # print(f"Graph creation completed in {time.time() - graph_creation_start:.3f} seconds")
        
        # Start timing for spatial index refresh
        # spatial_index_start = time.time()
        self._refresh_spatial_index()
        # print(f"Spatial index refresh completed in {time.time() - spatial_index_start:.3f} seconds")

        # print(f"Mask loaded in {time.time() - start_time:.3f} seconds. Width: {self.width_px} px")
    def _estimate_dlo_width(
            self,
            statistic: str = "max",
    ) -> float:

        dist_type  = cv2.DIST_L1
        mask_size  = 3                     # 3×3 kernel is the fast path

        dist_small = cv2.distanceTransform(self.mask_bool, dist_type, mask_size)

        if statistic == "median":
            median = np.median(dist_small[dist_small > 0.0])
            return float(median*2) , (dist_small > median).astype(np.uint8)
        if statistic == "mean":
            mean = np.mean(dist_small[dist_small > 0.0])
            return float(mean*2) , (dist_small > mean).astype(np.uint8)
        if statistic == "min":
            min = np.min(dist_small[dist_small > 0.0])
            return float(min*2) , (dist_small > min).astype(np.uint8)
        if statistic == "max":
            max = np.max(dist_small[dist_small > 0.0])
            return float(max*2) , (dist_small > max).astype(np.uint8)
        raise ValueError("statistic must be 'median', 'mean', 'min' or 'max'")
    def _mask_to_simplified_graph_new(self, min_cluster_factor: float = 2.0) -> nx.Graph:
        coords = self._coords_array          # (N,2) array of [x,y]
        H, W = self.mask_bool.shape
        gs = max(self.width_px, 1.0)         # grid size
        min_size = int(self.width_px * min_cluster_factor)

        # 1) assign each point to a 2D grid cell
        cells = np.floor(coords / gs).astype(int)
        n_bins_x = math.ceil(W / gs)
        cell_id = cells[:,1] * n_bins_x + cells[:,0]

        # 2) find unique cells, inverse indices, and their counts
        uniq_ids, inverse, counts = np.unique(cell_id,
                                            return_inverse=True,
                                            return_counts=True)

        # 3) select only the “big enough” bins
        large_cells = np.where(counts >= min_size)[0]
        if large_cells.size == 0:
            return nx.Graph()  # nothing to add

        # 4) vectorized centroid computation per cell
        sum_x = np.bincount(inverse, weights=coords[:,0])
        sum_y = np.bincount(inverse, weights=coords[:,1])
        centroids = np.vstack((sum_x[large_cells]/counts[large_cells],
                            sum_y[large_cells]/counts[large_cells])).T

        # 5) for each large cell, pick the one pixel closest to its centroid
        rep_coords = []
        for ci, centroid in zip(large_cells, centroids):
            members = np.nonzero(inverse == ci)[0]
            pts     = coords[members]
            dists   = np.linalg.norm(pts - centroid, axis=1)
            winner  = members[np.argmin(dists)]
            rep_coords.append(coords[winner].astype(int))

        rep_coords = np.asarray(rep_coords)  # shape (M,2)
        M = rep_coords.shape[0]

        # 7) build k-NN graph on reps and take its MST
        #    (each rep is a node 0..M-1)
        
        k = 3
        tree = cKDTree(rep_coords)
        dists, idxs = tree.query(rep_coords, k+1,distance_upper_bound=gs*5)  # idxs[:,0] is self
        # Filter out invalid neighbors (distances that are inf or indices that are >= M)
        valid_edges = []
        for i in range(M):
            for j in range(1, k+1):  # skip self (j=0)
                neighbor_idx = idxs[i, j]
                neighbor_dist = dists[i, j]
                
                # Check if neighbor is valid (not inf distance and valid index)
                if not np.isinf(neighbor_dist) and neighbor_idx < M:
                    valid_edges.append((i, neighbor_idx, neighbor_dist))
        # rows   = np.repeat(np.arange(M), k)
        # cols   = idxs[:, 1:].ravel()
        # weights= dists[:,1:].ravel()

        G = nx.Graph()
        # add all reps as nodes (storing their 2D positions)
        G.add_nodes_from((i, {"pos": tuple(rep_coords[i])}) for i in range(M))
        
        # add weighted edges from the valid k-NN connections
        G.add_weighted_edges_from(valid_edges)
        # add all reps as nodes (storing their 2D positions)
        # G.add_nodes_from((i, {"pos": tuple(rep_coords[i])}) for i in range(M))
        # add weighted edges from the k-NN
        # G.add_weighted_edges_from(zip(rows, cols, weights))

        # extract the minimum‐spanning‐tree
        T = nx.minimum_spanning_tree(G)
        T.graph["grid_size"] = gs
        return T


    def prune_short_branches_and_delete_junctions(self,max_length: int) -> None:


        """
        prune leaf branches longer than max_length
        """
        def _prune_short_branches(G: nx.Graph, max_length: int) -> None:
            """
            Prune branches longer than max_length
            """

            to_remove = []
            # get leaf nodes
            leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]

            
            for leaf in leaf_nodes:
                path = [leaf]
                while True:
                    # traverse the branch
                    neighbor = list(self.G.neighbors(path[-1]))
                    if len(neighbor) == 1 and self.G.degree[neighbor[0]] == 2:
                        path.append(neighbor[0])
                    else:
                        break
                    if len(path) > max_length:
                        break
                if len(path) < max_length:
                    to_remove.extend(path)
            # remove the nodes
            self.G.remove_nodes_from(to_remove)
            
        _prune_short_branches(self.G, max_length)
        junction_nodes = [n for n in self.G.nodes if self.G.degree[n] > 2]
        self.G.remove_nodes_from(junction_nodes)
        _prune_short_branches(self.G, max_length)
        # prune 0 degree nodes
        zero_degree_nodes = [n for n in self.G.nodes if self.G.degree[n] == 0]
        self.G.remove_nodes_from(zero_degree_nodes)

    def _extract_branches_0(self) -> List[List[int]]:
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

    def _extract_branches(self) -> List[List[int]]:
        # 1) gather all leaves (deg == 1)
        leaves = [n for n, d in self.G.degree() if d == 1]

        branches = []
        visited = set()
        for leaf in leaves:
            if leaf in visited:
                continue
            # 2) find the “other” leaf in the same component
            comp = nx.node_connected_component(self.G, leaf)
            other_leaves = [ℓ for ℓ in comp if self.G.degree(ℓ) == 1 and ℓ != leaf]

            if other_leaves:
                target = other_leaves[0]
                # 3) BFS‐based shortest_path is O(n+m) but on a chain it's just the unique path
                path = nx.shortest_path(self.G, source=leaf, target=target)
            else:
                # isolated single‐node component
                path = [leaf]

            branches.append(path)
            visited.update(path)

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

        for branch in branches:
            L = branch.shape[0]
            # 1) add L nodes, with positions in their attr dict
            for i in range(L):
                node_id = offset + i
                nodes.append((node_id, {'pos': tuple(branch[i])}))

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
        ex_time = time.time()
        branches = self._extract_branches_0()
        print(f"Extracted_0 {len(branches)} branches in {time.time() - ex_time:.3f} seconds")
        
        branches_coords = []
        for branch in branches:
            coords = np.array([self.G.nodes[n]["pos"] for n in branch])
            branches_coords.append(coords)
        
        smooth_time = time.time()
        # Fit B-spline to each branch
        smoothed_branches = []
        
        for i, coords in enumerate(branches_coords):
            if len(coords) < 3:
                # Skip branches that are too short to fit a spline
                continue
            smooth_2d_branch = bspline_fitting.smooth_2d_branch(
                coords=coords,
                s=smoothing,
                num_samples=max_num_points
            )
            smoothed_branches.append(smooth_2d_branch)
        
        print(f"Fitted {len(smoothed_branches)} branches in {time.time() - smooth_time:.3f} seconds")
        
        start_generation = time.time()
        # Generate new graph from the smoothed branches
        self.G = self._create_graph_from_branches(smoothed_branches)
        print(f"Generated graph from branches in {time.time() - start_generation:.3f} seconds")




    def _unit(self,v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _estimate_tangent(self, G: nx.Graph, node: int) -> np.ndarray:
        """tangent = (node → its neighbour) or (prev → node) depending on degree"""
        nbrs = list(G.neighbors(node))
        if not nbrs:
            return np.zeros(2)
        j = nbrs[0]
        return self._unit(np.array(G.nodes[node]["pos"]) -np.array(G.nodes[j]["pos"]))

    def _cubic_bezier(self, p0, d0, p1, d1, n=50, λ=0.3):
        gap = np.linalg.norm(p1 - p0)
        b0, b1 = p0, p0 + λ * gap * d0
        b2, b3 = p1 - λ * gap * d1, p1
        t = np.linspace(0., 1., n)[:, None]
        pts = ((1 - t) ** 3) * b0 + 3 * ((1 - t) ** 2) * t * b1 \
            + 3 * (1 - t) * (t ** 2) * b2 + (t ** 3) * b3
        return pts
    

    def _find_most_isolated_leaf_node_by_nearest_neighbor(self) -> int:
        """
        Find the leaf node that has the maximum distance to its nearest leaf neighbor.
        
        Returns:
        --------
        int
            Node ID of the most isolated leaf node
        """
        leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]
        
        if len(leaf_nodes) < 2:
            return leaf_nodes[0] if leaf_nodes else None
        
        # Get positions of all leaf nodes
        leaf_positions = np.array([self.G.nodes[n]["pos"] for n in leaf_nodes])
        
        # For each leaf node, find the minimum distance to any other leaf node
        max_min_distance = -1
        most_isolated_node = None
        
        for i, node in enumerate(leaf_nodes):
            node_pos = leaf_positions[i]
            other_positions = np.delete(leaf_positions, i, axis=0)
            
            # Find minimum distance to any other leaf node
            distances = np.linalg.norm(other_positions - node_pos, axis=1)
            min_distance = np.min(distances)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                most_isolated_node = node
        
        return most_isolated_node
    
    
    def reconstruct_dlo(self):
        
        
        # 1. Extract leaf nodes pairs on the same fragment
        comp_id: Dict[int, int] = {n: i for i, c in enumerate(nx.connected_components(self.G))
                        for n in c}
        
        leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]
        leaf_poses = np.array([self.G.nodes[n]["pos"] for n in leaf_nodes])
        leaf_comps = [comp_id[n] for n in leaf_nodes]
        leaf_pairs = {n0: n1 for i, n0 in enumerate(leaf_nodes)
                      for n1 in leaf_nodes
                      if leaf_comps[i] == leaf_comps[leaf_nodes.index(n1)]
                      and n0 != n1
                      }
                      
        
        # print(f"Found {len(leaf_pairs)} leaf node pairs in {len(set(leaf_comps))} fragments.")
        most_isolated_leaf = self._find_most_isolated_leaf_node_by_nearest_neighbor()
        # print(f"Most isolated leaf node: {most_isolated_leaf}")
        # most_isolated_leaf_pos = np.array(self.G.nodes[most_isolated_leaf]["pos"])
        # 2. start traversal 
        # from the most isolated leaf node
        # to the its pair leaf node
        # to the end point from the neighbor
        tree = cKDTree(leaf_poses)
        start_node = leaf_pairs[most_isolated_leaf]
        ln = leaf_nodes.copy()
        while len(ln) >2 :
            
            start_node_nbr = list(self.G.neighbors(start_node))[0]
            start_pos = np.array(self.G.nodes[start_node]["pos"])
            start_node_nbr_pos = np.array(self.G.nodes[start_node_nbr]["pos"])

            tan0 = self._unit(start_pos - start_node_nbr_pos)
            k= 10
            results = tree.query_ball_point(start_pos,r=self.width_px * 20.0,return_sorted=False)
            candidate_matches = [leaf_nodes[i] for i in results if leaf_nodes[i] != start_node and leaf_nodes[i] in ln]
            theta_deg_sums = {}
            for cand in candidate_matches:
                cand_pos = np.array(self.G.nodes[cand]["pos"])
                tan1 = self._unit(cand_pos - start_pos)
                # from the candidate match to its neighbor
                cand_nbr = list(self.G.neighbors(cand))[0]
                cand_nbr_pos = np.array(self.G.nodes[cand_nbr]["pos"])
                tan2 = self._unit(cand_nbr_pos - cand_pos)

                theta_deg_sums[cand] = np.rad2deg(np.arccos(np.dot(tan0, tan1)))\
                            + np.rad2deg(np.arccos(np.dot(tan1, tan2)))
                # print(f"Sum angle between {start_node} and {cand}: {theta_deg_sum:.2f}°")


            best_cand = min(theta_deg_sums, key=theta_deg_sums.get)
            self.G.add_edge(start_node, best_cand)
            start_node = leaf_pairs[best_cand]
            ln = [n for n in self.G.nodes if self.G.degree[n] == 1]
        # print(f"Best candidate for {start_node} is {best_cand} with angle sum {theta_deg_sums[best_cand]:.2f}°")

    def connect_leaf_nodes_by_bezier_curves(
            self,
            max_dist_factor: float = 2.0,
            k: int = 6,
            λ_ratio: float = 0.30,
            n_bridge: int = 50,
            do_resample: bool = False,
            resample_N: int = 60,
        ):
        """
        Auto–connect leaf nodes with C¹-continuous Bézier bridges.

        Parameters
        ----------
        max_dist_factor : float
            distance threshold = factor × self.width_px
        k : int
            Number of nearest endpoints examined for each endpoint
        λ_ratio : float
            Control-point length = λ_ratio × gap  (0.25-0.35 good range)
        n_bridge : int
            #points sampled along each Bézier bridge
        do_resample : bool
            If True, refits a global B-spline and re-samples it at resample_N points
        resample_N : int
            Number of points in the output polyline if do_resample is True
        """
        print("➤ Connecting leaf nodes with Bézier curves …")

        # ------------------------------------------------------------------
        # 1. gather endpoints (= all degree-1 nodes) and their tangents
        # ------------------------------------------------------------------
        
        
        
        leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]
        endpoints: List[_EndPoint] = []
        start_time = time.time()
        comp_id: Dict[int, int] = {n: i for i, c in enumerate(nx.connected_components(self.G))
                                for n in c}
        print(f"  found {len(leaf_nodes)} leaf nodes in {time.time() - start_time:.3f} seconds")
        # find start leaf node
        # by finding the most distant node in all of leaf nodes

        most_isolated_leaf = self._find_most_isolated_leaf_node_by_nearest_neighbor()
        # for n in leaf_nodes:
        #     pos = np.asarray(self.G.nodes[n]["pos"], dtype=float)
        #     tan = self._estimate_tangent(self.G, n)
        #     endpoints.append(_EndPoint(n, pos, tan, comp_id[n]))

        # if len(endpoints) < 2:
        #     print("  nothing to connect.")
        #     return

        # ------------------------------------------------------------------
        # 2. KD-tree: candidate search
        # ------------------------------------------------------------------
        P = np.array([ep.pos for ep in endpoints])
        tree = cKDTree(P)
        R_max = self.width_px * max_dist_factor
        pairs = {}                      # node_id -> (best_cost, best_match_id)
        visited_pairs = set()  # to avoid duplicates
          # node_id -> sum of angles in degrees
        best_pairs = set()  # to store the best pairs of endpoints
        best_theta_sums = {}  # to store the best angle sums for each pair
        for i, ep in enumerate(endpoints):
            dists, idxs = tree.query(ep.pos, k=min(k + 1, len(endpoints)))
            theta_deg_sums = {}
            for dist, j in zip(dists[1:], idxs[1:]):      # skip itself (0)
                ej = endpoints[j]
                if ep.frag == ej.frag or dist > R_max:
                    continue
                # if (ep.node, ej.node) in visited_pairs or (ej.node, ep.node) in visited_pairs:
                #     continue    
                # ------------------------------------------------------
                # 3. simple cost: distance + bending energy of 3 tangents
                #    to the node then to the match then from the match to the neighbor 
                # ------------------------------------------------------
                # to the end point from the neighbor
                tan0 = ep.tan 
                # to the candidate match from the end point
                tan1 = self._unit(ej.pos - ep.pos)  # unit vector
                # from the candidate match to its neighbor
                cand_nbr = list(self.G.neighbors(ej.node))[0]
                tan2 = self._unit(np.array(self.G.nodes[cand_nbr]["pos"]) - ej.pos)


                theta_deg_sums[tuple(sorted((ep.node, ej.node)))] = np.rad2deg(np.arccos(np.dot(tan0, tan1)))\
                                + np.rad2deg(np.arccos(np.dot(tan1, tan2)))
                visited_pairs.add((ep.node, ej.node))
                # sum_theta_deg = np.rad2deg(np.arccos(np.dot(tan0, tan1)))\
                #                 + np.rad2deg(np.arccos(np.dot(tan1, tan2)))
                # print(f"Sum angle between {ep.node} and {ej.node}: {sum_theta_deg:.2f}°")
            if theta_deg_sums:
                best_pair = min(theta_deg_sums, key=theta_deg_sums.get)
                if theta_deg_sums[best_pair] < 50.0:  # threshold for angle sum
                    best_pairs.add(best_pair)
                    best_theta_sums[best_pair] = theta_deg_sums[best_pair]
        print(f"best_pairs: {best_pairs}")
        # print(f"best_theta_sums: {best_theta_sums}")

        # ------------------------------------------------------------------
        # 5. build & insert bridges
        # ------------------------------------------------------------------
        next_id = max(self.G.nodes) + 1 if self.G.nodes else 0
        for n0, n1 in best_pairs:
            ep0 = next(ep for ep in endpoints if ep.node == n0)
            ep1 = next(ep for ep in endpoints if ep.node == n1)
            pts = self._cubic_bezier(ep0.pos, ep0.tan, ep1.pos, ep1.tan,
                                n=n_bridge, λ=λ_ratio)

            # add intermediate nodes (ignore first/last – they already exist)
            new_nodes = list(range(next_id, next_id + len(pts) - 2))
            next_id += len(new_nodes)
            for pid, p in zip(new_nodes, pts[1:-1]):
                self.G.add_node(pid, pos=p)
            # connect polyline: n0 — new_nodes — n1
            chain = [n0] + new_nodes + [n1]
            self.G.add_edges_from(zip(chain[:-1], chain[1:]))

        # # ------------------------------------------------------------------
        # # 6. optional global spline refit and uniform re-sampling
        # # ------------------------------------------------------------------
        # if do_resample:
        #     print("  refitting global B-spline …")
        #     xyz = np.array([self.G.nodes[n]["pos"] for n in self.G.nodes]).T
        #     tck, _ = splprep(xyz, s=0, k=3)
        #     u_eq = np.linspace(0, 1, resample_N)
        #     pts_eq = np.asarray(splev(u_eq, tck)).T

        #     # rebuild the graph from scratch with uniform spacing
        #     self.G.clear()
        #     self.G.add_nodes_from(
        #         (i, {"pos": p}) for i, p in enumerate(pts_eq)
        #     )
        #     self.G.add_edges_from((i, i + 1) for i in range(len(pts_eq) - 1))

        # print("  ✓ Bézier bridging completed.")




    def _bending_energy(self,prev_edges,next_edges, alpha=1.0):
        """
        Compute the bending energy between two edges.
        as in discrete elastic rod model

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
    
    def _are_same_line(self, v1: np.ndarray, v2: np.ndarray, tol: float = 1e-9) -> bool:
        """
        Return True if v and w are colinear and point in the same direction.
        v, w are length-2 numpy arrays (or lists / tuples).
        """
        v = np.asarray(v1, dtype=float)
        w = np.asarray(v2, dtype=float)

        if np.linalg.norm(v) < tol or np.linalg.norm(w) < tol:
            raise ValueError("Zero-length vector")

        # scalar z-component of the 2-D cross product
        cross_z = v[0]*w[1] - v[1]*w[0]

        colinear     = abs(cross_z) < tol          # nearly zero ⇒ parallel
        same_signed  = np.dot(v, w) > 0            # >0 ⇒ same direction

        return colinear and not same_signed
    def _calculate_bending_energy(self, prev_node: int, curr_node: int, 
                                next_node: int) -> float:
        """Calculate the bending energy between three nodes."""
        prev = np.asarray(self._pos_dict[prev_node])
        curr = np.asarray(self._pos_dict[curr_node])
        next = np.asarray(self._pos_dict[next_node])
        # Check if any points are the same
        prev_edges, next_edges = curr - prev, next - curr
        
        
        if np.all(prev_edges == 0) or np.all(next_edges == 0):
            return np.inf
        return self._bending_energy(prev_edges, next_edges) , self._are_same_line(prev_edges, next_edges)
    
    def add_bending_energy_and_prune(self, be_limit: float = 1.5) -> None:
        """
        Add bending energy to the graph and prune edges with high bending energy.
        
        Parameters:
        -----------
        be_limit : float
            Maximum bending energy allowed
        """
        print("Adding bending energy...")
        to_remove = []
        for node in self.G.nodes:
            if self.G.degree[node] > 1:
                neighbors = list(self.G.neighbors(node))
                for i in range(len(neighbors)-1):
                    prev_node = neighbors[i]
                    curr_node = node
                    next_node = neighbors[i+1]
                    be_val, same_line = self._calculate_bending_energy(prev_node, curr_node, next_node)
                    if be_val > be_limit or same_line:
                        to_remove.append(curr_node)
                        if not self.G.has_edge(next_node, prev_node):
                            self.G.add_edge(next_node, prev_node, weight=be_val)
                        continue
                    self.G.nodes[curr_node].setdefault("bending_energy", []).append(
                            ((prev_node, next_node), be_val)
                        )
        # add 0 degree nodes to the list
        
        
        self.G.remove_nodes_from(to_remove)
        zero_degree_nodes = [n for n in self.G.nodes if self.G.degree[n] == 0]
        self.G.remove_nodes_from(zero_degree_nodes)


    def _calc_sparse_distance_matrix(self,sparse_radius_factor) -> None:

        # Get sparse distance matrix for faster neighbor lookup
        self.sparse_radius = self.width_px * sparse_radius_factor
        print(f"Calculating sparse distance matrix with radius {self.sparse_radius}")
        S = self._kdtree.sparse_distance_matrix(self._kdtree, self.sparse_radius, output_type="coo_matrix")
        # Calculate degree heuristic
        if S.nnz:
            # 1) “S” is a scipy.sparse.coo_matrix where each non‐zero entry
            #    corresponds to a pair of nodes (i,j) that lie within your search radius.
            # 2) `S.nnz` is the total number of non‐zero entries—i.e. the total number
            #    of “edges” (including both directions) in your radius graph.
            
            # S.row and S.col are two arrays, each of shape (nnz,):
            #   - S.row[k] = i
            #   - S.col[k] = j
            # means “node i has node j as a neighbor.”
            #
            # Because the sparse_distance_matrix is symmetric, you’ll see both (i,j)
            # and (j,i) in those lists.
            idx_all = np.hstack((S.row, S.col))
            # Now idx_all is a single long array of length 2*nnz:
            #    [ i0, i1, i2, …, j0, j1, j2, … ]
            #
            # Each index appears once for each neighbor‐relationship it belongs to.
            neighbor_count = np.bincount(idx_all, minlength=len(self._nodes_array))
            # neighbor_count[p] = number of neighbors point p has 
            # (counting both directions, but that’s fine since deg(i)=deg(j)).
        else:
            neighbor_count = np.zeros(len(self._nodes_array), dtype=int)
            
        return neighbor_count
    





    
    def _refresh_spatial_index(self) -> None:
        """Rebuild the KD-tree spatial index for the graph nodes."""
        self._pos_dict = nx.get_node_attributes(self.G, "pos")
        self._nodes_array = np.array(list(self._pos_dict.keys()), dtype=int)
        self._coords_array = np.array([self._pos_dict[n] for n in self._nodes_array], dtype=int)
        self._kdtree = cKDTree(self._coords_array)
    
    def _clean_close_neighbors(self, nodes: np.ndarray, min_dist: float) -> np.ndarray:
        """
        Remove nodes that lie within `min_dist` of any previously selected node.

        Args:
            nodes:        Array of node‐indices (e.g. your `isolated_nodes`).
            min_dist:     Minimum Euclidean distance required between kept nodes.

        Returns:
            A filtered array of node‐indices with no two closer than min_dist.
        """
        cleaned = []
        # get actual coordinates for those nodes
        coords = self._coords_array[nodes]  # shape (N, 3) or (N,2) depending on dim

        for idx, pt in zip(nodes, coords):
            # check distance to all already-kept points
            if all(np.linalg.norm(pt - self._coords_array[other]) > min_dist
                for other in cleaned):
                cleaned.append(idx)

        return np.array(cleaned, dtype=nodes.dtype)

    def _get_isolated_nodes(self, radius_factor) -> np.ndarray:

        
        print("Finding isolated nodes...")
        neighbor_count = self._calc_sparse_distance_matrix(sparse_radius_factor=radius_factor)
        # Get best starting nodes
        isolated_nodes = self._nodes_array[np.argsort(neighbor_count)[:min(50, len(neighbor_count))]]
        # if the nodes are close to each other, take only one
        isolated_nodes = self._clean_close_neighbors(isolated_nodes, min_dist=self.sparse_radius)

        return isolated_nodes


    def remove_crowded_nodes(self,sparse_radius_factor) -> None:
        """
        Remove crowded nodes from the graph.
        
        Parameters:
        -----------
        crowded_idxs : np.ndarray
            Indices of crowded nodes to be removed
        """
        print("Removing crowded nodes...")
        neighbor_count = self._calc_sparse_distance_matrix(sparse_radius_factor=sparse_radius_factor)
        crowded_idxs = np.argsort(neighbor_count)[-100:]
        self._nodes_array = np.delete(self._nodes_array, crowded_idxs)
        self._coords_array = np.delete(self._coords_array, crowded_idxs, axis=0)
        self.G.remove_nodes_from(crowded_idxs)
        self.G = nx.convert_node_labels_to_integers(
                    self.G, first_label=0, ordering='sorted'
                )
    
    
    def connect_all_nodes(self, radius_factor: float = 2.0) -> None:
        """
        Connect all nodes in the graph.
        
        Parameters:
        -----------
        radius_factor : float
            Multiplier for search radius based on grid size
        """
        if not hasattr(self, 'width_px') or self.width_px is None:
            raise ValueError("Width not calculated. Load a mask first.")
            
        radius = self.width_px * radius_factor
        print(f"Connecting all nodes with radius {radius}")
        self._refresh_spatial_index()


        # Track visited nodes and paths
        visited = set()
        paths = []
        
        # Main connection loop
        for node in self.G.nodes:
            node_degree = self.G.degree[node]
            if node_degree > 2:
                continue
            qpos = self._coords_array[node]
            dist, idx = self._kdtree.query(qpos, k=5, distance_upper_bound=radius)
            try:
                # Get the indices of the nearest neighbors
                valid = idx < len(self._nodes_array)
                good_idxs = idx[valid]
                candidate_nodes = self._nodes_array[good_idxs]
                # remove the node itself from the candidates
                candidate_nodes = candidate_nodes[candidate_nodes != node]
                # remove the nodes that are already connected

                for c in candidate_nodes:
                    if self.G.has_edge(node, c):
                        candidate_nodes = candidate_nodes[candidate_nodes != c]

                for c in candidate_nodes:
                    if self.G.degree[c] > 2:
                        candidate_nodes = candidate_nodes[candidate_nodes != c]
                if len(candidate_nodes) == 0:
                    continue
                
                dist = np.linalg.norm(self._coords_array[candidate_nodes[0]] - self._coords_array[node])
                self.G.add_edge(node, candidate_nodes[0], weight=float(dist))
                if node_degree == 0 and len(candidate_nodes) > 1:
                    dist = np.linalg.norm(self._coords_array[candidate_nodes[1]] - self._coords_array[node])
                    self.G.add_edge(node, candidate_nodes[1], weight=float(dist))
                    
            except ValueError:
                # Handle the case where no neighbors are found
                print(f"No neighbors found for node {node}")
                continue

            
        print("Connecting nodes...")
    
    def connect_close_leaf_nodes(self, radius_factor: float = 2.0) -> None:
        
        # get the leaf nodes
        leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]
        
        if not leaf_nodes:
            print("No leaf nodes found.")
            return
        
        if not hasattr(self, 'width_px') or self.width_px is None:
            raise ValueError("Width not calculated. Load a mask first.")
        radius = self.width_px * radius_factor
        
        print(f"Connecting close leaf nodes with radius {radius}")
        
        self._refresh_spatial_index()
        
        # leaf_node_tree = cKDTree(self._coords_array[leaf_nodes])
        
        while len(leaf_nodes) > 0:
            node = leaf_nodes.pop(0)
            qpos = self._coords_array[node]
            dist, idx = self._kdtree.query(qpos, k=10, distance_upper_bound=radius)

            # Get the indices of the nearest neighbors
            valid = idx < len(self._nodes_array)
            good_idxs = idx[valid]
            candidate_nodes = self._nodes_array[good_idxs]
            # remove the node itself from the candidates
            candidate_nodes = candidate_nodes[candidate_nodes != node]
            candidate_nodes = candidate_nodes[np.isin(candidate_nodes, leaf_nodes)]
            # add edge to the graph
            if len(candidate_nodes) > 0:
                candidate_node = candidate_nodes[0]
                if not self.G.has_edge(node, candidate_node):
                    dist = np.linalg.norm(self._coords_array[candidate_node] - self._coords_array[node])
                    self.G.add_edge(node, candidate_node, weight=float(dist))
                    leaf_nodes.pop(leaf_nodes.index(candidate_node))
        print("Connecting close leaf nodes...")

    def find_all_paths(self)-> List[List[int]]:
        """
        Find all paths in the graph.

        Returns:
        --------
        List[List[int]]
            List of paths, where each path is a list of node indices
        """
        paths = []
        for node in self.G.nodes:
            for neighbor in self.G.neighbors(node):
                if node < neighbor:
                    path = nx.shortest_path(self.G, source=node, target=neighbor)
                    paths.append(path)
        return paths
    def longest_path_with_bending_energy(self, k: int = 10, be_limit: float = 1.5,
                                  radius_factor: float = 1.5,
                                  show_log: bool = False,
                                  iso_radius_factor: float = 10.0) -> None:
        """
        Connect nodes while minimizing bending energy.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors to consider
        be_limit : float
            Maximum bending energy allowed
        radius_factor : float
            Multiplier for search radius based on grid size
        """
        start_time = time.time()
        
        if not hasattr(self, 'width_px') or self.width_px is None:
            raise ValueError("Width not calculated. Load a mask first.")
            
        radius = self.width_px * radius_factor
        self._refresh_spatial_index()
        if show_log:
            print(f"Radius: {radius}")



        best_n_ids = self._get_isolated_nodes(radius_factor=iso_radius_factor).tolist()

        sub_G = self.G.subgraph(best_n_ids)


        pos = nx.get_node_attributes(sub_G, 'pos')
        nx.draw(sub_G, pos=pos, node_size=5, node_color='red',
               edge_color='red', with_labels=False)
        # 3. Draw grid underneath
        plt.gca().set_axisbelow(True)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
        # plt.show()
        # Track visited nodes and paths
        visited = set()
        paths = []
        new_path = True
        prev_node = None
        current_node = None
        path = []
        paths = []
        graph_size = len(self.G.nodes)
        # Main connection loop
        while len(best_n_ids) > 0:


            if new_path:
                if show_log and len(path) > 1:
                    print(f"PATH: {path}")
                    print(f"PATH LENGTH: {len(path)}")
                    

                path = []
                if show_log:
                    print(f"---------------------------NEW PATH---------------------------")
                    
                prev_node = best_n_ids.pop(0)

                
                path.append(prev_node)
                # visited.add(prev_node)
                
                # Find nearest unvisited node
                qpos = self._coords_array[prev_node]
                dist, idx = self._kdtree.query(qpos, k=2)  # self + 1
                candidate_nodes = self._nodes_array[idx]
                candidates = [(candidate_node, dist[i]) for i, candidate_node in enumerate(candidate_nodes)
                                if candidate_node != prev_node
                                    and dist[i] < radius*2]

                if not candidates :
                    if show_log:
                        print(f"No candidates found for node for NEW PATH {prev_node}")
                    continue

                current_node, current_node_dist = candidates[0]
                self.G.add_edge(prev_node, current_node, weight=float(current_node_dist))
                path.append(current_node)

                new_path = False
                if show_log:
                    print(f"from {prev_node} to {current_node}")
                
            else:
                # Extend current path
                qpos = self._coords_array[current_node]
                dist, idx = self._kdtree.query(qpos, k=k+1)
                candidate_nodes = self._nodes_array[idx]
                candidates = [(candidate_node, dist[i]) for i, candidate_node in enumerate(candidate_nodes)
                              if candidate_node != current_node
                              and dist[i] < radius*2]
                # in best_n_ids
                is_in_best_n_ids = [True for c in candidates if c[0] in best_n_ids]
                if not candidates or any(is_in_best_n_ids) :
                    paths.append(path)
                    new_path = True
                    if show_log:
                        print(f"No candidates found for node ENDS PATH {current_node}")
                    continue
                    
                # Select candidates with acceptable bending energy
                be_vals = {}
                for i, candidate in enumerate(candidates):
                    candidate, dist = candidate

                    be_val, same_line = self._calculate_bending_energy(
                        prev_node, current_node, candidate)
                    if be_val <= be_limit and not same_line:
                        # combine dist and be_val
                        be_vals[candidate] = be_val

                
                if not be_vals:
                    paths.append(path)
                    new_path = True
                    if show_log:
                        print(f"No VALID candidates found for node ENDS PATH {current_node}")
                    continue
                best = min(be_vals, key=be_vals.get)
                best_list = sorted(be_vals, key=be_vals.get)
                while len(best_list) > 0:
                    best = best_list.pop(0)

                    # if the new edge is not passing through the graph dont add it
                    current_node_pose = self._coords_array[current_node]
                    best_candidate_pose = self._coords_array[best]
                    rows,cols = line(current_node_pose[0], current_node_pose[1],best_candidate_pose[0], best_candidate_pose[1])

                    # checks if the line is in the graph
                    if self.mask_bool[cols, rows].all():
                        if not self.G.has_edge(current_node, best):
                            dist = np.linalg.norm(best_candidate_pose - current_node_pose)
                            self.G.add_edge(current_node, best, weight=float(dist))
                            self.G.nodes[current_node].setdefault("bending_energy", []).append(
                                ((prev_node, best), be_vals[best])
                            )

                        path.append(best)
                        visited.add(best)
                        prev_node, current_node = current_node, best
                        break
                    else:
                        # print("Line between {} and {} is not in the graph.".format(current_node, best))
                        continue
                if len(best_list) == 0:
                    paths.append(path)
                    new_path = True
                    if show_log:
                        print(f"No VALID candidates found for node ENDS PATH {current_node}")
        

        
        print(f"Connected with bending energy in {time.time() - start_time:.3f} seconds")
        return paths
    def connect_with_bending_energy(self, k: int = 10, be_limit: float = 1.5,
                                  radius_factor: float = 1.5,
                                  show_log: bool = False) -> None:
        """
        Connect nodes while minimizing bending energy.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors to consider
        be_limit : float
            Maximum bending energy allowed
        radius_factor : float
            Multiplier for search radius based on grid size
        """
        start_time = time.time()
        
        if not hasattr(self, 'width_px') or self.width_px is None:
            raise ValueError("Width not calculated. Load a mask first.")
            
        radius = self.width_px * radius_factor
        self._refresh_spatial_index()
        if show_log:
            print(f"Radius: {radius}")

        best_n_ids = self._get_isolated_nodes(radius_factor=radius_factor).tolist()
        # Track visited nodes and paths
        visited = set()
        paths = []
        new_path = True
        prev_node = None
        current_node = None
        path = []
        paths = []

        # Main connection loop
        while True:

            zero_deg = [n for n in self.G.nodes
                        if self.G.degree[n] == 0
                        and n not in visited]
            
            if len(zero_deg) <= 1:
                break
                
            if new_path:
                if show_log and len(path) > 1:
                    print(f"PATH: {path}")
                    print(f"PATH LENGTH: {len(path)}")
                    
                if len(path) == 2:
                    self.G.remove_node(path[1])
                    if show_log:
                        print(f"Removed node {path[1]} from path")
                
                path = []
                if show_log:
                    print(f"---------------------------NEW PATH---------------------------")
                # Pick a starting node
                if len(best_n_ids) == 0:
                    prev_node = zero_deg[0]
                else:
                    while best_n_ids:
                        prev_node = best_n_ids.pop(0)
                        if prev_node in zero_deg:
                            break
                    else:
                        prev_node = zero_deg[0]
                
                path.append(prev_node)
                visited.add(prev_node)
                
                # Find nearest unvisited node
                qpos = self._coords_array[prev_node]
                dist, idx = self._kdtree.query(qpos, k=2)  # self + 1
                candidate_nodes = self._nodes_array[idx]
                candidates = [(candidate_node, dist[i]) for i, candidate_node in enumerate(candidate_nodes)
                              if candidate_node not in visited
                              and dist[i] < radius*2]

                if not candidates:
                    if show_log:
                        print(f"No candidates found for node for NEW PATH {prev_node}")
                    continue

                current_node, current_node_dist = candidates[0]
                self.G.add_edge(prev_node, current_node, weight=float(current_node_dist))
                path.append(current_node)
                visited.add(current_node)
                new_path = False
                if show_log:
                    
                    print(f"from {prev_node} to {current_node}")
                
            else:
                # Extend current path
                qpos = self._coords_array[current_node]
                dist, idx = self._kdtree.query(qpos, k=k+1)
                candidate_nodes = self._nodes_array[idx]
                candidates = [(candidate_node, dist[i]) for i, candidate_node in enumerate(candidate_nodes)
                              if candidate_node not in visited
                              and dist[i] < radius*2]

                if not candidates:
                    paths.append(path)
                    new_path = True
                    if show_log:
                        print(f"No candidates found for node ENDS PATH {current_node}")
                    continue
                    
                # Select candidates with acceptable bending energy
                be_vals = {}
                for i, candidate in enumerate(candidates):
                    candidate, dist = candidate

                    be_val, same_line = self._calculate_bending_energy(
                        prev_node, current_node, candidate)
                    if be_val <= be_limit and not same_line:
                        # combine dist and be_val
                        be_vals[candidate] = be_val

                
                if not be_vals:
                    paths.append(path)
                    new_path = True
                    if show_log:
                        print(f"No VALID candidates found for node ENDS PATH {current_node}")
                    continue
                


                best = min(be_vals, key=be_vals.get)
                best_list = sorted(be_vals, key=be_vals.get)
                while len(best_list) > 0:
                    best = best_list.pop(0)

                    # if the new edge is not passing through the graph dont add it
                    current_node_pose = self._coords_array[current_node]
                    best_candidate_pose = self._coords_array[best]
                    rows,cols = line(current_node_pose[0], current_node_pose[1],best_candidate_pose[0], best_candidate_pose[1])

                    # checks if the line is in the graph
                    if self.mask_bool[cols, rows].all():
                        if not self.G.has_edge(current_node, best):
                            dist = np.linalg.norm(best_candidate_pose - current_node_pose)
                            self.G.add_edge(current_node, best, weight=float(dist))
                            self.G.nodes[current_node].setdefault("bending_energy", []).append(
                                ((prev_node, best), be_vals[best])
                            )

                        path.append(best)
                        visited.add(best)
                        prev_node, current_node = current_node, best
                        break
                    else:
                        # print("Line between {} and {} is not in the graph.".format(current_node, best))
                        continue
                if len(best_list) == 0:
                    paths.append(path)
                    new_path = True
                    if show_log:
                        print(f"No VALID candidates found for node ENDS PATH {current_node}")
        
        # Remove isolated nodes
        # self.G.remove_nodes_from([n for n in self.G.nodes if self.G.degree[n] == 0])
        
        print(f"Connected with bending energy in {time.time() - start_time:.3f} seconds")


    
    def connect_nodes_by_topology(self, be_limit: float = 5.0, max_dist_factor: float = 2.0, show_log: bool = True) -> None:
        """
        Connect nodes in the graph by topology in two phases:
        1. First connects isolated nodes (degree 0) to leaf nodes (degree 1)
        2. Then connects leaf nodes (degree 1) to other leaf nodes based on bending energy
        
        Parameters:
        -----------
        be_limit : float
            Maximum allowed bending energy for leaf-to-leaf connections
        max_dist_factor : float
            Maximum distance factor for connecting nodes (multiplied by width_px)
        """
        start_time = time.time()
        
        # Prepare spatial data
        self._refresh_spatial_index()
        
        # Track connection counts
        isolated_connected = 0
        leaf_connected = 0
        
        # ---- PHASE 1: Connect isolated nodes to leaf nodes ----
        max_dist = self.width_px * max_dist_factor
                # Main connection loop
        if show_log:
            print(f"----------------------------PHASE 2------------------------------\n")
        leaf_connected = self.connect_leaf_nodes_by_bending_energy(be_limit, max_dist_factor, show_log)
        if show_log:
            print(f"Max distance for connection: {max_dist}")
            print(f"----------------------------PHASE 1------------------------------\n")
        isolated_connected = self.connect_zero_deg_nodes(be_limit, show_log, max_dist)


        # if show_log:
        #     print(f"Phase 1: Connected {isolated_connected} isolated nodes in {time.time() - start_time:.3f} seconds")
        if show_log:
            print(f"----------------------------PHASE 2------------------------------\n")
        leaf_connected = self.connect_leaf_nodes_by_bending_energy(be_limit*3, max_dist_factor, show_log)
        # ---- PHASE 2: Connect leaf nodes to other leaf nodes ----
        phase2_time = time.time()

        total_time = time.time() - start_time
        # if show_log:
            # print(f"Phase 2: Connected {leaf_connected} leaf node pairs in {time.time() - phase2_time:.3f} seconds")
            # print(f"Total: Connected {isolated_connected + leaf_connected} node pairs in {total_time:.3f} seconds")
        
        return self 

    def connect_zero_deg_nodes(self, be_limit, show_log, max_dist):
        while True:
            # Get current isolated and leaf nodes
            isolated_nodes = [n for n in self.G.nodes if self.G.degree[n] == 0]
            leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]
            
            # Stop if no more isolated nodes or leaf nodes
            if not isolated_nodes or not leaf_nodes:
                break
            
            # Process each isolated node
            connections_made = False
            
            for isolated_node in isolated_nodes:
                best_leaf = None
                best_dist = float('inf')
                
                # Find closest leaf node
                isolated_pos = np.array(self._pos_dict[isolated_node])
                leaf_nodes_to_connect = []
                for i, leaf_node in enumerate(leaf_nodes):
                    def check_to_connect(from_node, to_node, be_limit):
                        from_node_pos = np.array(self._pos_dict[from_node])
                        to_node_pos = np.array(self._pos_dict[to_node])
                        dist = np.linalg.norm(from_node_pos - to_node_pos)
                        
                        if dist > max_dist:
                            return False
                        
                        rows, cols = line(
                        int(isolated_pos[0]), int(isolated_pos[1]),
                        int(to_node_pos[0]), int(to_node_pos[1])
                        )
                        # line_poses = set(zip(rows, cols))
                        
                        # if line_poses.issubset(self._full_coords_set):
                        if not self.mask_bool[cols, rows].all():
                            return False
                        
                        # check if the resolting connection make a good bending energy
                        leaf_nb = list(self.G.neighbors(to_node))[0]
                        be, same_line = self._calculate_bending_energy(leaf_nb, to_node, from_node)
                        if same_line or be > be_limit:
                            return False
                        
                        return True
                    leaf_nodes_to_connect.append(check_to_connect(isolated_node, leaf_node, be_limit))

                # Connect to best leaf node if found
                # if best_leaf is not None:
                leaf_nodes_to_connect = [leaf_nodes[i] for i, leaf_node in enumerate(leaf_nodes_to_connect) if leaf_node]
                if len(leaf_nodes_to_connect) == 2:
                    if show_log:
                        print(f"Connected isolated node {isolated_node} to leaf nodes {leaf_nodes_to_connect}")
                    self.G.add_edge(isolated_node, leaf_nodes_to_connect[0], weight=float(best_dist))
                    self.G.add_edge(isolated_node, leaf_nodes_to_connect[1], weight=float(best_dist))
                    # isolated_connected += 1
                    connections_made = True
                    break
                
                if len(leaf_nodes_to_connect) == 1:
                    best_leaf = leaf_nodes_to_connect[0]
                    self.G.add_edge(isolated_node, best_leaf, weight=float(best_dist))
                    if show_log:
                        print(f"Connected isolated node {isolated_node} to leaf node {best_leaf}")
                    self.G.add_edge(isolated_node, best_leaf, weight=float(best_dist))

                    # # calculate bending energy
                    # be, same_line = self._calculate_bending_energy(
                    #     list(self.G.neighbors(best_leaf))[0],
                    #     best_leaf,
                    #     isolated_node
                    # )
                    # if be <= be_limit and not same_line:
                    #     # Store bending energy information
                    #     self.G.nodes[isolated_node].setdefault('bending_energy', []).append(
                    #         ((list(self.G.neighbors(best_leaf))[0], best_leaf), be))
                    #     self.G.nodes[best_leaf].setdefault('bending_energy', []).append(
                    #         ((isolated_node, list(self.G.neighbors(best_leaf))[0]), be))
                    #     if show_log:
                    #         print(f"Added bending energy: {self.G.nodes[isolated_node]['bending_energy']}")
                    #         print(f"Added bending energy: {self.G.nodes[best_leaf]['bending_energy']}")
                    # self.G.nodes[best_leaf]['bending_energy'] = be

                    # isolated_connected += 1
                    connections_made = True
                    # Only make one connection per iteration to ensure proper updating
                    break
                    
            # If no connections were made, exit phase 1
            if not connections_made:
                break
        return 

    def connect_leaf_nodes_by_bending_energy(self, be_limit, max_dist_factor, show_log):
        processed = set()  # Track processed leaf nodes
        while True:
            # Get current leaf nodes (recalculated each iteration)
            leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1 and n not in processed]
            
            # Stop if no more unprocessed leaf nodes
            if not leaf_nodes:
                break
                
            # Take the first unprocessed leaf node
            node_1 = leaf_nodes[0]
            processed.add(node_1)
            
            node_1_pos = np.array(self._pos_dict[node_1])
            
            # Find candidates with acceptable bending energy
            candidates = []
            
            # Check other leaf nodes
            for node_2 in leaf_nodes:
                if node_2 == node_1 or self.G.has_edge(node_1, node_2):
                    continue
                    
                node_2_pos = np.array(self._pos_dict[node_2])
                dist = np.linalg.norm(node_2_pos - node_1_pos)
                
                # Skip if too far
                if dist > self.width_px * max_dist_factor:
                    continue
                    
                # Calculate bending energy
                nb_1 = list(self.G.neighbors(node_1))[0]
                be, same_line = self._calculate_bending_energy(nb_1, node_1, node_2)
                
                # Skip if nodes would create a straight line
                if same_line:
                    continue
            
                # Store candidate with its bending energy
                candidates.append((node_2, be, dist))
            if not candidates:
                continue
            # Sort candidates by bending energy
            candidates.sort(key=lambda x: x[1])
            
            # Try to connect to best candidate
            if candidates[0][1] <= be_limit:
                best_candidate, best_be, dist = candidates[0]
                
                # Get the current neighbors of best candidate
                bc_nb_1 = list(self.G.neighbors(best_candidate))[0]
                
                # Check bending energy for the best_candidate node 
                best_candidate_be, best_candidate_same_line = self._calculate_bending_energy(bc_nb_1, best_candidate, node_1)
                
                # Verify the connection is valid from both ends
                if not best_candidate_same_line and best_candidate_be <= be_limit:
                    # Check if path through mask is valid
                    rows, cols = line(
                        int(node_1_pos[0]), int(node_1_pos[1]),
                        int(self._pos_dict[best_candidate][0]), int(self._pos_dict[best_candidate][1])
                    )

                    if self.mask_bool[cols, rows].all():
                        # Add edge with attributes
                        self.G.add_edge(node_1, best_candidate, weight=float(dist))
                        if show_log:
                            print(f"Connected leaf node {node_1} to leaf node {best_candidate}")
                        # Store bending energy information
                        self.G.nodes[node_1].setdefault('bending_energy', []).append(
                            ((nb_1, best_candidate), best_be))
                        self.G.nodes[best_candidate].setdefault('bending_energy', []).append(
                            ((bc_nb_1, node_1), best_candidate_be))
                        # print the bending energy added
                        if show_log:
                            print(f"Added bending energy: {self.G.nodes[node_1]['bending_energy']}")
                            print(f"Added bending energy: {self.G.nodes[best_candidate]['bending_energy']}")

                        
                        # leaf_connected += 1
                        
                        # Processed nodes are no longer leaf nodes, so remove from processed
                        processed.discard(node_1)
                        processed.discard(best_candidate)
                    else:
                        if show_log:
                            print(f"Line between {node_1} and {best_candidate} is not in the graph.")
        return  

    def _vector_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate the angle between two vectors."""
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-9 or v2_norm < 1e-9:
            return np.inf
            
        dot = np.dot(v1, v2)
        cosang = np.clip(dot / (v1_norm * v2_norm), -1, 1)
        return np.arccos(cosang)




    
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
        nx.draw(self.G, pos=pos, node_size=node_size, node_color='green',
                edge_color='red', with_labels=with_labels, ax=ax)
        ax.invert_yaxis()  # Flip Y axis to match image coordinates
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
    
    def get_nodes_as_array(self) -> np.ndarray:
        """
        Get node positions as a numpy array.
        
        Returns:
        --------
        np.ndarray
            Array of node positions, shape (N, 2)
        """
        pos = nx.get_node_attributes(self.G, 'pos')
        return np.array(list(pos.values()), dtype=np.int32)
    
    def fit_spline(self, path: List[int], smoothing: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a spline to a path through the graph.
        
        Parameters:
        -----------
        path : List[int]
            Ordered list of node IDs
        smoothing : float
            Smoothing factor for the spline
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            x and y coordinates of the spline points
        """
        pos = nx.get_node_attributes(self.G, 'pos')
        points = np.array([pos[node] for node in path])
        
        # Fit spline
        tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing)
        
        # Evaluate spline at more points for smooth curve
        u_new = np.linspace(0, 1, 100)
        x_new, y_new = splev(u_new, tck)
        
        return x_new, y_new
    def visualize_path(self, path: List[int], figsize: Tuple[int, int] = (10, 10),
                node_size: int = 5, with_labels: bool = False) -> None:
        """
        Visualize a specific path through the graph.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Path Visualization")
        subG = self.G.subgraph(path)
        pos = nx.get_node_attributes(subG, 'pos')
        nx.draw(subG, pos=pos, node_size=node_size, edge_color='red', with_labels=with_labels, ax=ax)
        # Highlight the path
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(subG, pos=pos, edgelist=path_edges, edge_color='blue', width=2, ax=ax)
        ax.invert_yaxis()  # Flip Y axis to match image coordinates
        plt.pause(0.01)
        return fig, ax

    def fit_bspline(self, n_control_points: int = 10, degree: int = 4, smoothing: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a B-spline to a path through the graph.
        
        Parameters:
        -----------
        n_control_points : int
            Number of control points for the B-spline
        degree : int
            Degree of the B-spline (typically 3 for cubic splines)
        smoothing : float or None
            Smoothing factor. None for interpolation, larger values for smoother fit
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            x and y coordinates of the fitted B-spline points
        """
        pos = nx.get_node_attributes(self.G, 'pos')
        points = np.array([pos[node] for node in pos.keys()])
        
        # Fit B-spline
        bspline_fitting.fit_bspline(points,
                                    # n_control_points=n_control_points, degree=degree, smoothing=smoothing
                                    )
        
        tck, fitted_points = bspline_fitting.fit_bspline(points,
                                                        #  n_control_points, degree, smoothing
                                                         )
    
        # Extract control points
        # control_points = bspline_fitting.extract_control_points(tck)

        control_points = None
        # Plot results
        bspline_fitting.plot_results(points, fitted_points, control_points)
        


    def get_longest_path(self) -> List[int]:
        """
        Get the longest path in the graph.
        
        Returns:
        --------
        List[int]
            Ordered list of node IDs forming the longest path
        """
        longest_path = max(nx.all_pairs_dijkstra(self.G), key=lambda x: x[1][0])[0]
        return longest_path