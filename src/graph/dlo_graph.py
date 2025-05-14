import numpy as np
import networkx as nx
import cv2
import time
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Literal, Set, Any, Union
from collections import defaultdict
from scipy.spatial import cKDTree
from skimage.draw import line
import torch
import torch.nn.functional as F
import logging

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
        self.logger = logger if logger else logging.getLogger(__name__)


    def load_from_mask(self, mask: np.ndarray,
                       downsample: int = 1,
                       statistic: Literal["median", "mean", "min", "max"] = "max",
                       connect_radius_factor: float = 2.0,
                       min_cluster_factor: float = 2.0,
                       ds_factor: int = 5) -> None:
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
        
        # Preprocess mask
        if downsample > 1:
            h, w = mask.shape
            new_h, new_w = h // downsample, w // downsample
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        self.mask_bool = mask > 0
        
        # Extract foreground pixels
        # self.dlo_pixels = np.column_stack(np.nonzero(self.mask_bool))
        # Make it [x, y] order to match coords
        ys, xs = np.nonzero(self.mask_bool)
        self._full_coords_array = np.column_stack((xs, ys))

        # Pre-compute this once in the initialization
        # Start timing for width estimation
        set_build_start = time.time()
        # self._full_coords_set = set(tuple(p) for p in self._full_coords_array)
        print(f"Set build completed in {time.time() - set_build_start:.3f} seconds")
        # Estimate DLO width
        width_estimation_start = time.time()
        self.width_px , mask_bool = self._estimate_dlo_width(self.mask_bool, self._coords_array, statistic=statistic, ds_factor=ds_factor)
        print(f"DLO width estimation completed in {time.time() - width_estimation_start:.3f} seconds")
        ys, xs = np.nonzero(mask_bool)
        self._coords_array = np.column_stack((xs, ys))
        
        # Start timing for graph creation
        graph_creation_start = time.time()
        
        # Create initial graph
        self.G = self._mask_to_simplified_graph(mask_bool,min_cluster_factor=min_cluster_factor)
        
        print(f"Graph creation completed in {time.time() - graph_creation_start:.3f} seconds")
        
        # Start timing for spatial index refresh
        spatial_index_start = time.time()
        self._refresh_spatial_index()
        print(f"Spatial index refresh completed in {time.time() - spatial_index_start:.3f} seconds")

        print(f"Mask loaded in {time.time() - start_time:.3f} seconds. Width: {self.width_px} px")
    def _estimate_dlo_width(
            self,
            mask_bin: np.ndarray,
            coords: np.ndarray,
            statistic: str = "max",
            ds_factor: int = 2,                # ↓ mask by 2 × (adjust to taste)
            use_l1: bool = False,              # True ⇒ L1 distance
            sample_frac: float = 1.0,         # only use a subset of coords
    ) -> float:
        """
        Fast width estimator – up to ~20× quicker than the original.

        • Down-samples the mask before the distance-transform  
        • Optionally switches to the L1 (city-block) metric  
        • Samples only a fraction of centre-line points
        """
        mask_bin = mask_bin.astype(np.uint8)
        # ---------------------------------------------------------
        # 2.  Fast distance transform
        # ---------------------------------------------------------
        dist_type  = cv2.DIST_L1 if use_l1 else cv2.DIST_L2
        mask_size  = 5                     # 3×3 kernel is the fast path
        h, w = mask_bin.shape
        dist_small = cv2.distanceTransform(mask_bin, dist_type, mask_size)



        # diam = dist_small        # back to original-resolution diameter

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


    def _mask_to_simplified_graph(self, mask_bool: np.ndarray,
                                    min_cluster_factor: float = 2.0) -> nx.Graph:
            """Convert binary mask to a simplified graph using grid-based clustering."""
            # Create graph
            G = nx.Graph()
            
            # Get mask dimensions
            H, W = mask_bool.shape
            
            # Use the already computed dlo_pixels instead of recalculating
            coords = self._coords_array
            
            # For node IDs, we need to calculate the flat indices
            # meaning the pixel index from the full coordinate system in the mask in a 1D array
            # This is done by using the formula: flat_index = x + y * width
            flat_indices = coords[:, 0] + coords[:, 1] * W
            
            # Calculate grid size based on DLO width
            self.grid_size = max(self.width_px, 1.0)
            
            # Assign pixels to grid cells
            # by dividing by grid size and flooring to get cell indices
            cells = np.floor(coords / self.grid_size).astype(int)
            
            # Create a dictionary to group pixels by cell
            # Using tuple of cell indices as keys
            # eacth cell will have a list of pixel indices
            cell_dict = {}
            for i, cell in enumerate(map(tuple, cells)):
                if cell in cell_dict:
                    cell_dict[cell].append(i)
                else:
                    cell_dict[cell] = [i]
            
            # Choose representatives for each cell
            reps, rep_coords = [], []
            min_cluster_size = int(self.width_px * min_cluster_factor)
            clusters = [c for c in cell_dict.values() if len(c) >= min_cluster_size]
            for pix_rows in clusters:
                pts = coords[pix_rows]
                centroid = pts.mean(0)
                winner_idx = np.argmin(np.linalg.norm(pts - centroid, axis=1))
                winner_row = pix_rows[winner_idx]
                rep_id = int(flat_indices[winner_row])  # Use precalculated flat indices
                reps.append(rep_id)
                rep_coords.append(coords[winner_row])
            
            rep_coords = np.asarray(rep_coords, int)
            
            # # Add nodes to graph
            # for rid, p in zip(reps, rep_coords):
            #     G.add_node(rid, pos=tuple(map(int, p)))
            # Add nodes to graph
            for rid, p in enumerate(rep_coords):
                G.add_node(rid, pos=tuple(map(int, p)))

            # Store graph metadata
            G.graph.update(grid_size=self.grid_size)
            
            return G
    
    def _refresh_spatial_index(self) -> None:
        """Rebuild the KD-tree spatial index for the graph nodes."""
        self._pos_dict = nx.get_node_attributes(self.G, "pos")
        self._nodes_array = np.array(list(self._pos_dict.keys()), dtype=int)
        self._coords_array = np.array([self._pos_dict[n] for n in self._nodes_array], dtype=int)
        self._kdtree = cKDTree(self._coords_array)
        
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
        # Get sparse distance matrix for faster neighbor lookup
        S = self._kdtree.sparse_distance_matrix(self._kdtree, radius * 3.0, output_type="coo_matrix")
        
        # Calculate degree heuristic
        if S.nnz:
            idx_all = np.hstack((S.row, S.col))
            deg = np.bincount(idx_all, minlength=len(self._nodes_array))
        else:
            deg = np.zeros(len(self._nodes_array), dtype=int)
        
        # Get best starting nodes
        best_n_ids = list(self._nodes_array[np.argsort(deg)[:min(10, len(deg))]])
        
        # Track visited nodes and paths
        visited = set()
        paths = []
        new_path = True
        prev_node = None
        current_node = None
        path = []
        
        # Get position dictionary for fast access
        # pos_dict = self.pos_dict

        # Main connection loop
        while True:

            # Get isolated nodes
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
    # def connect_with_bending_energy(self, k: int = 10, be_limit: float = 1.5,
    #                               radius_factor: float = 1.5,
    #                               show_log: bool = False) -> None:
    #     """
    #     Connect nodes while minimizing bending energy.
        
    #     Parameters:
    #     -----------
    #     k : int
    #         Number of nearest neighbors to consider
    #     be_limit : float
    #         Maximum bending energy allowed
    #     radius_factor : float
    #         Multiplier for search radius based on grid size
    #     """
    #     start_time = time.time()
        
    #     if not hasattr(self, 'width_px') or self.width_px is None:
    #         raise ValueError("Width not calculated. Load a mask first.")
            
    #     radius = self.width_px * radius_factor
    #     self._refresh_spatial_index()
    #     if show_log:
    #         print(f"Radius: {radius}")
    #     # Get sparse distance matrix for faster neighbor lookup
    #     S = self._kdtree.sparse_distance_matrix(self._kdtree, radius * 3.0, output_type="coo_matrix")
        
    #     # Calculate degree heuristic
    #     if S.nnz:
    #         idx_all = np.hstack((S.row, S.col))
    #         deg = np.bincount(idx_all, minlength=len(self._nodes_array))
    #     else:
    #         deg = np.zeros(len(self._nodes_array), dtype=int)
        
    #     # Get best starting nodes
    #     best_n_ids = list(self._nodes_array[np.argsort(deg)[:min(10, len(deg))]])
        
    #     # Track visited nodes and paths
    #     visited = set()
    #     paths = []
    #     new_path = True
    #     prev_node = None
    #     current_node = None
    #     path = []
        
    #     # Get position dictionary for fast access
    #     # pos_dict = self.pos_dict

    #     # Main connection loop
    #     while True:

    #         # Get isolated nodes
    #         zero_deg = [n for n in self.G.nodes 
    #                     if self.G.degree[n] == 0
    #                     and n not in visited]
    #         if len(zero_deg) <= 1:
    #             break
                
    #         if new_path:
    #             if show_log and len(path) > 1:
    #                 print(f"PATH: {path}")
    #                 print(f"PATH LENGTH: {len(path)}")
    #             if len(path) == 2:
    #                 self.G.remove_node(path[1])
    #                 if show_log:
    #                     print(f"Removed node {path[1]} from path")
    #             path = []
    #             print(f"---------------------------NEW PATH---------------------------")
    #             # Pick a starting node
    #             if len(best_n_ids) == 0:
    #                 prev_node = zero_deg[0]
    #             else:
    #                 while best_n_ids:
    #                     prev_node = best_n_ids.pop(0)
    #                     if prev_node in zero_deg:
    #                         break
    #                 else:
    #                     prev_node = zero_deg[0]
                
    #             path.append(prev_node)
    #             visited.add(prev_node)
                
    #             # Find nearest unvisited node
    #             qpos = self._coords_array[prev_node]
    #             dist, idx = self._kdtree.query(qpos, k=2)  # self + 1

    #             candidates = [self._nodes_array[n] for i,n in enumerate(idx)
    #                             if self._nodes_array[n] not in visited
    #                             and dist[i] < radius*2
    #             ]

    #             if not candidates:
    #                 if show_log:
    #                     print(f"No candidates found for node for NEW PATH {prev_node}")
    #                 continue
                    
    #             current_node = candidates[0]
    #             current_node_dist = dist[0]
    #             self.G.add_edge(prev_node, current_node, weight=float(current_node_dist))
    #             path.append(current_node)
    #             visited.add(current_node)
    #             new_path = False
    #             if show_log:
                    
    #                 print(f"from {prev_node} to {current_node}")
                
    #         else:
    #             # Extend current path
    #             qpos = self._coords_array[current_node]
    #             dist, idx = self._kdtree.query(qpos, k=k+1)
                
    #             candidates = [self._nodes_array[n] for i,n in enumerate(idx) 
    #                             if self._nodes_array[n] not in visited
    #                             and dist[i] < radius*2
    #             ]

                
    #             if not candidates:
    #                 paths.append(path)
    #                 new_path = True
    #                 if show_log:
    #                     print(f"No candidates found for node ENDS PATH {current_node}")
    #                 continue
                
    #             # Select candidates with acceptable bending energy
    #             be_vals = {}
    #             for candidate in candidates:
    #                 be_val, same_line = self._calculate_bending_energy(
    #                     prev_node, current_node, candidate)
    #                 if be_val <= be_limit and not same_line:
    #                     be_vals[candidate] = be_val
                
    #             if not be_vals:
    #                 paths.append(path)
    #                 new_path = True
    #                 if show_log:
    #                     print(f"No VALID candidates found for node ENDS PATH {current_node}")
    #                 continue
                
    #             best = min(be_vals, key=be_vals.get)
    #             best_list = sorted(be_vals, key=be_vals.get)
    #             while len(best_list) > 0:
    #                 best = best_list.pop(0)

    #                 # if the new edge is not passing through the graph dont add it
    #                 current_node_pose = self._coords_array[current_node]
    #                 best_candidate_pose = self._coords_array[best]
    #                 rows,cols = line(current_node_pose[0], current_node_pose[1],best_candidate_pose[0], best_candidate_pose[1])

    #                 # checks if the line is in the graph
    #                 if self.mask_bool[cols, rows].all():
    #                     if not self.G.has_edge(current_node, best):
    #                         dist = np.linalg.norm(best_candidate_pose - current_node_pose)
    #                         self.G.add_edge(current_node, best, weight=float(dist))
    #                         self.G.nodes[current_node].setdefault("bending_energy", []).append(
    #                             ((prev_node, best), be_vals[best])
    #                         )

    #                     path.append(best)
    #                     visited.add(best)
    #                     prev_node, current_node = current_node, best
    #                     break
    #                 else:
    #                     # print("Line between {} and {} is not in the graph.".format(current_node, best))
    #                     continue
    #             if len(best_list) == 0:
    #                 paths.append(path)
    #                 new_path = True
    #                 if show_log:
    #                     print(f"No VALID candidates found for node ENDS PATH {current_node}")
        
    #     # Remove isolated nodes
    #     # self.G.remove_nodes_from([n for n in self.G.nodes if self.G.degree[n] == 0])
        
    #     print(f"Connected with bending energy in {time.time() - start_time:.3f} seconds")
        
        
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
    
    def prune_short_branches(self, min_length: int = 3) -> None:
        """
        Remove branches shorter than min_length.
        
        Parameters:
        -----------
        min_length : int
            Minimum path length to keep
        """
        start_time = time.time()
        
        # prune 0 degree nodes
        self.G.remove_nodes_from([n for n in self.G.nodes if self.G.degree[n] == 0])
        
        # get all leaf nodes
        leaf_nodes = [n for n in self.G.nodes if self.G.degree[n] == 1]

        # for each leaf node, check the longest path to the next leaf node
        path = []
        visited = set()
        for node_1 in leaf_nodes:
            if node_1 in visited:
                continue
            visited.add(node_1)
            for node_2 in leaf_nodes:
                if node_2 in visited:
                    continue
                try:
                    path = nx.shortest_path(self.G, source=node_1, target=node_2)
                except nx.NetworkXNoPath:
                    continue # no path between the two nodes
            # print("Path{}".format(path))
            visited.add(path[-1])
            if len(path) <= min_length:
                # remove the path
                for node in path:
                    if self.G.has_node(node):
                        self.G.remove_node(node)

                continue
        
        print(f"Pruned short branches in {time.time() - start_time:.3f} seconds")
    
    def optimize_path(self) -> List[int]:
        """
        Find an optimal path through the graph using TSP.
        
        Returns:
        --------
        List[int]
            Ordered list of node IDs forming the optimal path
        """
        start_time = time.time()
        
        # Use greedy TSP with precomputed paths
        try:
            path = traveling_salesman_problem(
                self.G, weight='weight', method='greedy', cycle=False)
        except:
            print("TSP optimization failed, returning empty path")
            return []
            
        print(f"Path optimization completed in {time.time() - start_time:.3f} seconds")
        return path
    
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
        nx.draw(self.G, pos=pos, node_size=node_size, 
               edge_color='red', with_labels=with_labels, ax=ax)
        ax.invert_yaxis()  # Flip Y axis to match image coordinates
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

