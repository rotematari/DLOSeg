import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

import itertools
from collections import defaultdict
from typing import Tuple, Literal

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
import cv2
from skimage.draw import line 
import numpy as np
import networkx as nx
from collections import defaultdict

import numpy as np
import networkx as nx
import cv2
from collections import defaultdict
from typing import Literal
import numpy as np
import networkx as nx
import cv2
from collections import defaultdict
from typing import Literal, Tuple, Optional
from scipy.spatial import cKDTree
from skimage.draw import line  # fast Bresenham


# # ╔═══════════════════════════════════════════════════════════════════════╗
# # ║  1.  Diameter estimate from binary mask                              ║
# # ╚═══════════════════════════════════════════════════════════════════════╝

# def dlo_width_from_mask(
#     poses: np.ndarray,
#     mask_bin: np.ndarray,
#     statistic: Literal["median", "mean", "min", "max"] = "median",
# ) -> float:
#     """Estimate DLO diameter (px) from mask via distance‑transform."""
#     if mask_bin.dtype != np.uint8:
#         mask_bin = (mask_bin > 0).astype(np.uint8)

#     dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
#     ys, xs = poses[:, 0].astype(int), poses[:, 1].astype(int)
#     inside = (
#         (ys >= 0) & (ys < mask_bin.shape[0]) &
#         (xs >= 0) & (xs < mask_bin.shape[1])
#     )
#     radii = dist[ys[inside], xs[inside]]
#     if radii.size == 0:
#         raise ValueError("No valid points for width estimation.")

#     diam = radii * 2.0
#     stats = {
#         "median": np.median,
#         "mean":   np.mean,
#         "min":    np.min,
#         "max":    np.max,
#     }
#     try:
#         return float(stats[statistic](diam))
#     except KeyError:
#         raise ValueError("statistic must be one of 'median', 'mean', 'min', 'max'")

# # ╔═══════════════════════════════════════════════════════════════════════╗
# # ║  2.  Mask  ➜  simplified representative graph                        ║
# # ╚═══════════════════════════════════════════════════════════════════════╝

# def mask_to_simplified_graph(
#     mask: np.ndarray,
#     *,
#     width_stat: Literal["median", "mean", "min", "max"] = "max",
#     grid_multiplier: float = 3.0,
#     connect_radius_factor: float = 1.5,
# ) -> nx.Graph:
#     """Vectorised mask → sparsified graph (auto‑scales by DLO diameter)."""
#     mask_bool = mask.astype(bool)
#     if not mask_bool.any():
#         return nx.Graph()

#     fg_yx = np.column_stack(np.nonzero(mask_bool))
#     width_px = dlo_width_from_mask(fg_yx, mask_bool, width_stat)
#     grid_size = max(grid_multiplier * width_px, 1.0)

#     H, W = mask.shape
#     idx = np.flatnonzero(mask_bool.ravel())
#     xs, ys = idx % W, idx // W
#     coords = np.column_stack((xs, ys))

#     cells = np.floor(coords / grid_size).astype(int)
#     # cells = np.floor_divide(coords, grid_size, dtype=int)
#     buckets: dict[Tuple[int, int], list[int]] = defaultdict(list)
#     for i, cell in enumerate(map(tuple, cells)):
#         buckets[cell].append(i)

#     reps, rep_coords = [], []
#     for rows in buckets.values():
#         pts = coords[rows]
#         centroid = pts.mean(0)
#         winner_row = rows[np.linalg.norm(pts - centroid, axis=1).argmin()]
#         rep_id = int(idx[winner_row])
#         reps.append(rep_id)
#         rep_coords.append(coords[winner_row])

#     rep_coords = np.asarray(rep_coords, int)

#     G = nx.Graph()
#     G.add_nodes_from((rid, {"pos": tuple(map(int, p))}) for rid, p in zip(reps, rep_coords))
#     G.graph.update(grid_size=grid_size, connect_radius_factor=connect_radius_factor)

#     if len(reps) > 1:
#         diffs = rep_coords[:, None, :] - rep_coords[None, :, :]
#         sqd = (diffs * diffs).sum(2)
#         thresh2 = (connect_radius_factor * grid_size) ** 2
#         u, v = np.where(np.triu(sqd, 1) <= thresh2)
#         G.add_edges_from((reps[i], reps[j]) for i, j in zip(u, v))

#     return G

# # ╔═══════════════════════════════════════════════════════════════════════╗
# # ║  3.  k‑NN growth driven by bending‑energy minimisation                ║
# # ╚═══════════════════════════════════════════════════════════════════════╝

# def knn_graph_from_bending_energy_fast(
#     G: nx.Graph,
#     mask_bool: np.ndarray,
#     *,
#     k: int = 2,
#     be_limit: float = 2.0,
#     radius: Optional[float] = None,
# ) -> nx.Graph:
#     """Connect isolated nodes under a bending‑energy constraint (fast)."""
#     if k < 1:
#         raise ValueError("k must be ≥ 1")

#     grid_size = G.graph.get("grid_size", 3.0)
#     if radius is None:
#         radius = 2.0 * grid_size

#     node_pos = nx.get_node_attributes(G, "pos")
#     nodes = np.fromiter(node_pos, int)
#     coords = np.asarray([node_pos[n] for n in nodes], float)
#     tree = cKDTree(coords)

#     # sparse degree heuristic (correct call signature)
#     S = tree.sparse_distance_matrix(tree, radius * 3.0, output_type="coo_matrix")
#     if S.nnz:
#         idx_all = np.hstack((S.row, S.col))
#         deg = np.bincount(idx_all, minlength=len(nodes))
#     else:
#         deg = np.zeros(len(nodes), int)
#     best_n_ids = list(nodes[np.argsort(deg)[: min(10, len(deg))]])

#     visited: set[int] = set()
#     alive: set[int] = set(G.nodes)

#     def add_edge(u: int, v: int):
#         G.add_edge(u, v)

#     def bending_energy(prev_: int, curr_: int, next_: int) -> float:
#         p0, p1, p2 = map(np.asarray, (node_pos[prev_], node_pos[curr_], node_pos[next_]))
#         v1, v2 = p0 - p1, p2 - p1
#         if np.allclose(v1, 0) or np.allclose(v2, 0):
#             return np.inf
#         cosang = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
#         return np.arccos(cosang)

#     def free_path(p0: Tuple[int, int], p1: Tuple[int, int]) -> bool:
#         rr, cc = line(p0[1], p0[0], p1[1], p1[0])
#         return mask_bool[rr, cc].all()

#     new_path = True
#     prev_node = current_node = None

#     while True:
#         zero_deg = [n for n in alive if G.degree[n] == 0]
#         if len(zero_deg) <= 1:
#             break

#         if new_path:
#             # pick a starting isolated node (low degree heuristic first)
#             prev_node = best_n_ids.pop(0) if best_n_ids else zero_deg[0]
#             if prev_node not in zero_deg:
#                 prev_node = zero_deg[0]
#             visited.add(prev_node)

#             # nearest unvisited neighbour
#             qpos = node_pos[prev_node]
#             dist, idx = tree.query(qpos, k=min(2, len(nodes)))
#             dist = np.atleast_1d(dist)
#             idx = np.atleast_1d(idx)
#             cand = [nodes[i] for i, d in zip(idx, dist) if nodes[i] not in visited and d < radius]
#             if not cand:
#                 G.remove_node(prev_node)
#                 alive.discard(prev_node)
#                 continue
#             current_node = cand[0]
#             add_edge(prev_node, current_node)
#             visited.add(current_node)
#             new_path = False
#             continue

#         # extend path ----------------------------------------------------
#         qpos = node_pos[current_node]
#         dist, idx = tree.query(qpos, k=min(k + 1, len(nodes)))
#         dist = np.atleast_1d(dist)
#         idx = np.atleast_1d(idx)
#         cand = [nodes[i] for i, d in zip(idx, dist) if nodes[i] not in visited and d < radius]
#         if not cand:
#             new_path = True
#             continue

#         choices = [n for n in cand if bending_energy(prev_node, current_node, n) <= be_limit]
#         if not choices:
#             new_path = True
#             continue

#         for nxt in sorted(choices, key=lambda n: bending_energy(prev_node, current_node, n)):
#             if free_path(node_pos[current_node], node_pos[nxt]):
#                 add_edge(current_node, nxt)
#                 visited.add(nxt)
#                 prev_node, current_node = current_node, nxt
#                 break
#         else:
#             new_path = True

#     G.remove_nodes_from([n for n in G.nodes if G.degree[n] == 0])
#     return G

# # alias for backward compatibility
# knn_graph_from_bending_energy_2 = knn_graph_from_bending_energy_fast

# # ╔═══════════════════════════════════════════════════════════════════════╗
# # ║  4.  End‑to‑end convenience wrapper                                   ║
# # ╚═══════════════════════════════════════════════════════════════════════╝

# def dlo_path_graph_from_mask(
#     mask: np.ndarray,
#     *,
#     width_stat: str = "max",
#     grid_multiplier: float = 3.0,
#     connect_radius_factor: float = 1.5,
#     k: int = 2,
#     be_limit: float = 2.0,
#     radius: Optional[float] = None,
# ) -> nx.Graph:
#     """High‑level convenience: *binary mask* ➜ *final path graph*.

#     Parameters
#     ----------
#     mask : (H, W) ndarray, bool / uint8
#         Binary image where *True/1* marks the DLO pixels.
#     width_stat : str, default "max"
#         Statistic used to compute cable diameter (``dlo_width_from_mask``).
#     grid_multiplier : float, default 3.0
#         Grid‑cell size = ``grid_multiplier × cable_diameter``.
#     connect_radius_factor : float, default 1.5
#         Two representatives within ``factor × grid_size`` get an edge.
#     k : int, default 2
#         How many nearest neighbours to *evaluate* when extending the path.
#     be_limit : float, default 2.0 [rad]
#         Maximum bending angle allowed when adding an edge.
#     radius : float or None, default None
#         Search radius for neighbours; defaults to ``2 × grid_size``.

#     Returns
#     -------
#     nx.Graph
#         The final path/curve graph through the DLO.
#     """
#     # 1.  coarse representation
#     G_simpl = mask_to_simplified_graph(
#         mask,
#         width_stat=width_stat,
#         grid_multiplier=grid_multiplier,
#         connect_radius_factor=connect_radius_factor,
#     )

#     # 2.  bending‑energy‑guided connection (works on mask boolean array)
#     G_paths = knn_graph_from_bending_energy_fast(
#         G_simpl,
#         mask.astype(bool),
#         k=k,
#         be_limit=be_limit,
#         radius=radius,
#     )

#     return G_paths

# # ─────────────────────────────────────────────────────────────────────────
# # __all__ helper for clean imports
# # ─────────────────────────────────────────────────────────────────────────
# __all__ = [
#     "dlo_width_from_mask",
#     "mask_to_simplified_graph",
#     "knn_graph_from_bending_energy_fast",
#     "knn_graph_from_bending_energy_2",
#     "dlo_path_graph_from_mask",
# ]


# # # ────────────────────────────────────────────────────────────────────────────────
# # # Helper – diameter estimate from the binary mask
# # # ────────────────────────────────────────────────────────────────────────────────

# # def dlo_width_from_mask(
# #     poses: np.ndarray,          # (N, 2) (y, x) pixel coords – ideally centre‑line
# #     mask_bin: np.ndarray,       # H×W uint8 / bool, 1 == object
# #     statistic: Literal["median", "mean", "min", "max"] = "median",
# # ) -> float:
# #     """Return characteristic *diameter* of a DLO inside ``mask_bin``.

# #     We sample the Euclidean distance‑transform at the given ``poses`` and
# #     convert radii → diameters.  The chosen statistic (median/mean/min/max)
# #     controls robustness.
# #     """
# #     if mask_bin.dtype != np.uint8:
# #         mask_bin = (mask_bin > 0).astype(np.uint8)

# #     # radius (distance) map
# #     dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)  # float32 radii

# #     # sample at requested coordinates
# #     ys, xs = poses[:, 0].astype(int), poses[:, 1].astype(int)
# #     valid = (
# #         (ys >= 0) & (ys < mask_bin.shape[0]) &
# #         (xs >= 0) & (xs < mask_bin.shape[1])
# #     )
# #     radii = dist[ys[valid], xs[valid]]
# #     diameters = radii * 2.0

# #     if diameters.size == 0:
# #         raise ValueError("No valid centre‑line points inside mask.")

# #     if statistic == "median":
# #         return float(np.median(diameters))
# #     if statistic == "mean":
# #         return float(np.mean(diameters))
# #     if statistic == "min":
# #         return float(np.min(diameters))
# #     if statistic == "max":
# #         return float(np.max(diameters))
# #     raise ValueError("statistic must be one of 'median', 'mean', 'min', 'max'")

# # # ────────────────────────────────────────────────────────────────────────────────
# # # Main – "mask ➜ simplified graph" with *automatic* grid sizing
# # # ────────────────────────────────────────────────────────────────────────────────

# # def mask_to_simplified_graph(
# #     mask: np.ndarray,
# #     *,
# #     width_stat: Literal["median", "mean", "min", "max"] = "max",
# #     grid_multiplier: float = 3.0,
# #     connect_radius_factor: float | None = None,
# #     return_mapping: bool = False,
# # ):
# #     """Fastest route from binary *mask* ➜ sparsified graph *with auto params*.

# #     The grid cell size is derived automatically from the estimated cable
# #     diameter:

# #     ``grid_size = grid_multiplier × width_px``

# #     If ``connect_radius_factor`` is not given, it defaults to **1.5**.

# #     Parameters
# #     ----------
# #     mask : (H, W) bool / uint8 array
# #         Foreground == True / 1.
# #     width_stat : {"median", "mean", "min", "max"}, default "max"
# #         Statistic used when estimating the cable diameter.
# #     grid_multiplier : float, default 3.0
# #         How many *diameters* wide each grid cell should be.
# #     connect_radius_factor : float or None, default None
# #         Representatives closer than ``factor × grid_size`` are connected.
# #         If *None*, defaults to **1.5**.
# #     return_mapping : bool, default False
# #         If True, also return ``{flat_pixel_index: rep_node_id}``.

# #     Returns
# #     -------
# #     G : nx.Graph
# #         Graph over representative pixels only.
# #     node_to_rep : dict[int, int]        # only if return_mapping=True
# #     """
# #     mask_bool = mask.astype(bool)
# #     if not mask_bool.any():
# #         return (nx.Graph(), {}) if return_mapping else nx.Graph()

# #     # ── 1. estimate cable diameter (in *pixels*) --------------------------
# #     foreground_yx = np.column_stack(np.nonzero(mask_bool))  # (N, 2) (y, x)
# #     width_px = dlo_width_from_mask(foreground_yx, mask_bool.astype(np.uint8), width_stat)

# #     # grid size & connection radius
# #     grid_size = max(grid_multiplier * width_px, 1.0)
# #     if connect_radius_factor is None:
# #         connect_radius_factor = 1.5

# #     # ── 2. foreground pixel coordinates -----------------------------------
# #     H, W = mask.shape
# #     idx  = np.flatnonzero(mask_bool.ravel())                # 1‑D indices  (N,)
# #     xs   = idx % W
# #     ys   = idx // W
# #     coords = np.column_stack((xs, ys))                      # (N, 2), int64

# #     # ── 3. assign each pixel to a grid cell -------------------------------
# #     cells = np.floor_divide(coords, grid_size,
# #                             # dtype=int
# #                             )   # (N, 2)
# #     buckets = defaultdict(list)
# #     for i, cell in enumerate(map(tuple, cells)):
# #         buckets[cell].append(i)

# #     # ── 4. choose one representative per populated cell -------------------
# #     reps, rep_coords = [], []              # node IDs & positions
# #     node_to_rep = {} if return_mapping else None

# #     for pix_rows in buckets.values():
# #         pts = coords[pix_rows]
# #         centroid = pts.mean(axis=0)
# #         dists = np.linalg.norm(pts - centroid, axis=1)
# #         winner_row = pix_rows[int(dists.argmin())]
# #         rep_id = int(idx[winner_row])      # flat index is the node id
# #         reps.append(rep_id)
# #         rep_coords.append(coords[winner_row])
# #         if return_mapping:
# #             for r in pix_rows:
# #                 node_to_rep[int(idx[r])] = rep_id

# #     rep_coords = np.asarray(rep_coords, dtype=int)          # (M, 2)

# #     # ── 5. build graph over representatives only --------------------------
# #     G = nx.Graph()
# #     G.add_nodes_from(
# #         (rep, {"pos": tuple(int(v) for v in xy)})
# #         for rep, xy in zip(reps, rep_coords)
# #     )

# #     # # connect close representatives
# #     # if len(reps) > 1:
# #     #     diffs   = rep_coords[:, None, :] - rep_coords[None, :, :]
# #     #     sq_dist = (diffs * diffs).sum(axis=2)
# #     #     thresh2 = (connect_radius_factor * grid_size) ** 2
# #     #     u, v = np.where(np.triu(sq_dist, k=1) <= thresh2)
# #     #     G.add_edges_from((reps[i], reps[j]) for i, j in zip(u, v))

# #     return (G, node_to_rep) if return_mapping else G



def binary_mask_to_graph_indexed(mask):
    
    mask = np.asarray(mask, dtype=bool)
    
    G = nx.Graph()

    ys, xs = np.nonzero(mask)
    coords = list(zip(ys, xs))

    # Step 1: assign an index to each (y, x)
    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}

    # Step 2: add nodes with index and position attribute
    for coord, idx in coord_to_index.items():
        y, x = coord
        G.add_node(idx, pos=(x, y))  # image coordinates (x, y) [width, height]

    return G

def add_edges(G: nx.Graph,node_1,node_2) -> nx.Graph:
    """
    Add edges to the graph G_out from the graph G.
    """
    
    if not G.has_edge(node_1, node_2):
        G.add_edge(node_1, node_2)
    
    return G

def calc_be(G: nx.Graph,node : int, neighbour_1: int,neighbour_2: int) -> float:

    to_node = np.array(G.nodes[neighbour_1]['pos']) - np.array(G.nodes[node]['pos'])
    from_node = np.array(G.nodes[node]['pos']) - np.array(G.nodes[neighbour_2]['pos'])
    be = bending_energy(to_node, from_node, alpha=1.0)
    
    return be , same_line_same_direction(to_node, from_node)

def add_bending_energy_to_node(G: nx.Graph,node : int,neighbour_1: int ,neighbour_2: int) -> nx.Graph:
    
    be = calc_be(G,node, neighbour_1,neighbour_2)
    if be > 2:
        G.nodes[node].setdefault('bending_energy', [])
        return G , be
    
    G.nodes[node].setdefault('bending_energy', []).append(((neighbour_1,neighbour_2),be))
    return G , be

def knn_graph_from_bending_energy_2(
        G: nx.Graph,
        G_full: nx.Graph,
        k: int = 2,
        be_limit: float = 2.0,
        radius: float = 6.0,
) -> nx.Graph:
    """
    Grow a graph by repeatedly connecting pairs of zero‑degree nodes,
    choosing neighbours with the lowest bending energy below ``be_limit``.
    """

    if k < 1:
        raise ValueError("k must be ≥ 1")

    # ------------------------------------------------------------------
    # 1.  KD‑tree on the *static* node positions
    # ------------------------------------------------------------------
    full_pos_dict = nx.get_node_attributes(G_full, "pos")
    full_poses = list(full_pos_dict.values()) # (N, 2)
    
    node_pos = nx.get_node_attributes(G, "pos")
    nodes    = np.fromiter(node_pos.keys(), dtype=int)
    coords   = np.array([node_pos[n] for n in nodes], dtype=float)
    tree     = cKDTree(coords)

    paths     = []
    new_path  = True          # toggle: True ⇒ need to start a fresh path
    prev_node = current_node = None
        # 1.  radius‑graph as sparse COO
    # radius = 6.0
    S = tree.sparse_distance_matrix(tree, radius*3, output_type="coo_matrix")

    # 2.  degree = how many neighbours each node has inside the radius
    idx_all = np.hstack((S.row, S.col))             # row ∪ col  →  undirected degree
    deg     = np.bincount(idx_all, minlength=len(nodes))

    # 3.  the nodes with the *fewest* neighbours
    n = 10                                            # how many to keep
    idx = np.argpartition(deg, n)[:n]                 # O(N), no full sort
    idx = idx[np.argsort(deg[idx])]                   # now in ascending order
    best_n_ids = list(nodes[idx])                           # the actual node IDs
    # print(best_n_ids, deg[idx])
    
    path       = []
    visited     = set()
    alive = set(G.nodes)
    while True:
        # --------------------------------------------------------------
        # 2.  Update the zero‑degree pool and quit if nothing to do
        # --------------------------------------------------------------
        zero_deg = [n for n in G.nodes if G.degree[n] == 0]
        if len(zero_deg) <= 1:          # 0 or 1 isolated node left
            break

        # --------------------------------------------------------------
        # 3.  Start a new path (take the first isolated node)
        # --------------------------------------------------------------
        if new_path:
            
            # if len(path) > 0:
            #     print("Path: ", path)
            #     print("Length of path: ", len(path))
            #     pos = nx.get_node_attributes(G, 'pos')
            #     nx.draw(G, pos=pos, node_size=5, edge_color='red', with_labels=True)
                
            #     plt.pause(0.01)
            
            # print("\n--------------------- New Path ----------------------------\n")

            path       = []
            # print("Zero degree nodes: ", zero_deg)
            # print("Length of zero degree nodes: ", len(zero_deg))
            # print("Best nodes: ", best_n_ids)
            if len(best_n_ids) == 0:
                # print("No more best nodes")
                prev_node = zero_deg[0]
                # print("Best node: ", prev_node)
            while len(best_n_ids) > 0:
    
                prev_node = best_n_ids.pop(0)
                if prev_node not in zero_deg:
                    prev_node = zero_deg[0]
                else:
                    # print("Best node: ", prev_node)
                    break


            
            path.append(prev_node)
            visited.add(prev_node)

            # find the nearest *other* zero‑degree node
            qpos   = node_pos[prev_node]
            dist, idx = tree.query(qpos, k=1 + 1)
            cand = [ nodes[n] for i,n in enumerate(idx) if nodes[n] not in visited 
                    and dist[i] < radius*2
                    
                    ]

            if not cand:          # no way to start – leave the loop
                # print("No candidate found to start a new path for node. removing node.", prev_node)
                G.remove_node(prev_node)
                alive.discard(prev_node)   
                continue

            current_node = cand[0]
            path.append(current_node)
            visited.add(current_node)
            G = add_edges(G, prev_node, current_node)
            new_path = False      # we are now *inside* a path

        # --------------------------------------------------------------
        # 4.  Extend the current path
        # --------------------------------------------------------------
        qpos   = node_pos[current_node]
        dist, idx = tree.query(qpos, k=k + 1)
        cand = [ nodes[n] for i,n in enumerate(idx) if nodes[n] not in visited 
                and dist[i] < radius*2
                ]

        if not cand:              # dead end → start a fresh path
            paths.append(path)
            new_path = True
            # print("No candidate found to extend the path")
            continue

        # pick the candidate with the **smallest bending energy**
        be_vals = {}
        for n in cand:
            be, slsd = calc_be(G,
                               node=current_node,
                               neighbour_1=prev_node,
                               neighbour_2=n)
            if not slsd and be <= be_limit:
                be_vals[n] = be

        if not be_vals:           # all too “bendy” → start anew
            paths.append(path)
            new_path = True
            # print("No candidate found to big bending energy")
            continue

        best = min(be_vals, key=be_vals.get)
        best_list = sorted(be_vals, key=be_vals.get)
        while len(best_list) > 0:
            best = best_list.pop(0)

            # if the new edge is not passing through the graph dont add it
            current_node_pose = node_pos[current_node]
            best_candidate_pose = node_pos[best]
            rows,cols = line(current_node_pose[0], current_node_pose[1],best_candidate_pose[0], best_candidate_pose[1])
            line_poses = [tuple((rows[i],cols[i])) for i in range(len(rows))]
            if set(line_poses).issubset(full_poses):
                G = add_edges(G, current_node, best)
                G.nodes[current_node].setdefault("bending_energy", []).append(
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


    # ------------------------------------------------------------------
    # 5.  Remove any leftover isolated nodes
    # ------------------------------------------------------------------
    G.remove_nodes_from([n for n in G.nodes if G.degree[n] == 0])
    return G

def knn_graph_from_bending_energy(
        G: nx.Graph,
        k: int = 2,
        be_limit: float = 2.0,
    ) -> nx.Graph:
    """
    Build a k‑nearest‑neighbour graph from an existing graph *G*
    whose nodes have a `"pos"` attribute = (x, y).

    Parameters
    ----------
    G     : nx.Graph or nx.DiGraph
        Source graph with node attribute ``"pos": (x, y)``.
    k        : int, default 2
        Each node is connected to its k nearest neighbours.
    in_place : bool, default False
        If True, add the k‑NN edges to *G* and return it.
        If False, create and return a fresh nx.Graph that
        contains only the k‑NN edges (node attributes copied).

    Returns
    -------
    G : nx.Graph
        Undirected k‑NN graph with edge attribute ``weight`` =
        Euclidean distance and node attribute ``pos``.
    """
    if k < 1:
        raise ValueError("k must be ≥ 1")

    # ---- 1. Extract node IDs and their positions ---------------------------
    node_dict = nx.get_node_attributes(G, 'pos')
    print("number of nodes {} ".format(len(node_dict)))
    nodes = np.fromiter(node_dict.keys(), dtype=int)
    # coords = np.fromiter(node_dict.values(), dtype=tuple)
    coords = np.array([node_dict[n] for n in nodes], dtype=int)
    # coords = list(node_dict.values())
    # ---- 2. Build KD‑tree for O(N log N) neighbour look‑ups ---------------
    tree = cKDTree(coords)
    
    # Sparse COO containing only pairs whose Euclidean distance < radius
    # 1.  radius‑graph as sparse COO
    radius = 6.0
    S = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")

    # 2.  degree = how many neighbours each node has inside the radius
    idx_all = np.hstack((S.row, S.col))             # row ∪ col  →  undirected degree
    deg     = np.bincount(idx_all, minlength=len(nodes))

    # 3.  the nodes with the *fewest* neighbours
    # min_deg = deg.min()
    # least_crowded_ids = nodes[deg == min_deg]       # “real” node IDs
    # print("min‑deg =", min_deg, "  winners:", least_crowded_ids)
    n = 10                                            # how many to keep
    idx = np.argpartition(deg, n)[:n]                 # O(N), no full sort
    idx = idx[np.argsort(deg[idx])]                   # now in ascending order
    best_n_ids = nodes[idx]                           # the actual node IDs
    print(best_n_ids, deg[idx])
    # traverse the graph on one od the branches
    new_path = True

    paths = []
    visited = set()
    # make a list of nodes sorted by how many nodes there are in a radius n
    
    while len(visited) < len(nodes):
        if new_path:
            print("\n--------------------- New Path ----------------------------\n")
            path = []
            # pick an unvisited node
            unvisited_nodes = [
                    n for n in nodes
                    if n not in visited           # not processed yet
                    ]
            if not unvisited_nodes:
                break
            prev_node = unvisited_nodes[0]
            path.append(prev_node)
            visited.add(prev_node)
            prev_node_pos = node_dict[prev_node]

            _, nbr_idx = tree.query(prev_node_pos, k=1 + 1)  # self + k
            candidate_nodes = nodes[nbr_idx]
            candidate_nodes = [n for n in candidate_nodes
                   if n not in visited
                   and G.degree[n] == 0]   # <── NEW: still deg 0

            if candidate_nodes and candidate_nodes[0] not in visited:
                current_node = candidate_nodes[0]
                path.append(current_node)
                visited.add(current_node)
                G = add_edges(G, prev_node, current_node)
                new_path = False
            else:
                print("No new candidate to start with")
                continue

        current_node_pos = node_dict[current_node]
        _, nbr_idx = tree.query(current_node_pos, k=k + 1)
        candidate_nodes = nodes[nbr_idx]
        candidate_nodes = [n for n in candidate_nodes if n != current_node]

        # Only select candidates that are not visited yet
        candidate_nodes = [n for n in candidate_nodes if n not in visited]

        if not candidate_nodes:
            # No candidate nodes, start a new path
            new_path = True
            paths.append(path)
            print("Path: {} Len: {}".format(path, len(path)))
            continue

        # get bending energy values
        candidate_nodes_be = {}
        for candidate_node in candidate_nodes:
            be, slsd = calc_be(G, node=current_node, neighbour_1=prev_node, neighbour_2=candidate_node)
            if be > be_limit or slsd:
                continue
            candidate_nodes_be[candidate_node] = be

        if not candidate_nodes_be:
            new_path = True
            paths.append(path)
            print("Path: {} Len: {}".format(path, len(path)))
            continue

        best_candidate = min(candidate_nodes_be, key=candidate_nodes_be.get)
        best_candidate_be = candidate_nodes_be[best_candidate]

        G = add_edges(G, current_node, best_candidate)
        G.nodes[current_node].setdefault('bending_energy', []).append(((prev_node, best_candidate), best_candidate_be))

        path.append(best_candidate)
        visited.add(best_candidate)
        prev_node = current_node
        current_node = best_candidate

    # remove 0 degree nodes
    zero_degree_nodes = [n for n in G.nodes if G.degree[n] == 0]
    for node in zero_degree_nodes:
        G.remove_node(node)
    # # reomve neighboring nodes with degree 1
    # one_degree_nodes = [n for n in G.nodes if G.degree[n] == 1]
    # for node in one_degree_nodes:
    #     # if the neighbor also has degree 1, remove the node
    #     neighbors = list(G.neighbors(node))
    #     one_degree_neighbors = [n for n in neighbors if G.degree[n] == 1]
    #     for n in one_degree_neighbors:
    #         # remove the node and the edge
    #         G.remove_node(node)
    #         break
    return G

def connect_leaf_nodes(G: nx.Graph, be_limit=2.0) -> nx.Graph:
    leaf_nodes = [n for n in G.nodes if G.degree[n] == 1]
    if len(leaf_nodes) < 2:
        return G
    # subG = G.subgraph(leaf_nodes)
    # ---- 2. Build KD‑tree for O(N log N) neighbour look‑ups only of one degree nodes---------------
    # pos_dict = nx.get_node_attributes(subG, "pos")
    # nodes = np.fromiter(pos_dict.keys(), dtype=int)
    # coords = np.array([pos_dict[n] for n in nodes], dtype=float)
    
    # tree = cKDTree(coords)
    visited = set()
    # find closest leaf nodes
    for node_1 in leaf_nodes:
        node_1_pos = G.nodes[node_1]["pos"]
        bes = []
        if node_1 not in visited:
            visited.add(node_1)
        elif node_1 in visited:
            continue
        for node_2 in leaf_nodes:
            if node_2 in visited:
                continue
            # Skip self or already connected nodes
            if G.has_edge(node_1, node_2):
                continue
            
            node_2_pos = G.nodes[node_2]["pos"]

            vector = np.array(node_2_pos) - np.array(node_1_pos)
            dist = np.linalg.norm(vector)
            if dist > 70:
                continue
            nb_1 = list(G.neighbors(node_1))[0]
            be ,same_line = calc_be(G,node_1, nb_1, node_2)
            if same_line:
                continue
            bes.append((node_2, be))
        
        # Sort the candidates by bending energy
        bes.sort(key=lambda x: x[1])
        # Get the best candidate
        if not bes:
            continue
        best_candidate = bes[0][0]
        best_candidate_be = bes[0][1]
        if best_candidate_be > be_limit:
            continue
        visited.add(best_candidate)
        # calc the bending energy for the best candidate
        # best candidate neighbors
        bc_nb_1 = list(G.neighbors(best_candidate))[0]
        be ,same_line = calc_be(G,best_candidate, node_1, bc_nb_1)
        if same_line or be > be_limit:
            continue
        G.add_edge(node_1, best_candidate, weight=float(dist))
        G.nodes[node_1].setdefault('bending_energy', []).append(((nb_1, best_candidate), best_candidate_be))
        G.nodes[best_candidate].setdefault('bending_energy', []).append(((node_1, bc_nb_1), be))
    return G

def prune_short_branches(G: nx.Graph, min_length: int = 5) -> nx.Graph:
    
    """
    Remove branches shorter than `min_length` from the graph `G`.
    """
    
    # get all leaf nodes
    leaf_nodes = [n for n in G.nodes if G.degree[n] == 1]
    
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
                path = nx.shortest_path(G, source=node_1, target=node_2)
            except nx.NetworkXNoPath:
                continue # no path between the two nodes
        # print("Path{}".format(path))
        visited.add(path[-1])
        if len(path) <= min_length:
            # remove the path
            for node in path:
                if G.has_node(node):
                    G.remove_node(node)
                
            continue
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
    poses = list(pos_dict.values()) # (N, 2)
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

    # --- build representative (centroid) nodes -----------------------------
    rep_pos = {}                  # rep_node_id -> (x, y)

    rep_id_gen = itertools.count(start=0)

    for cluster in clusters:
        rows = [node_index[n] for n in cluster]
        cluster_coords = coords[rows]
        centroid = tuple(cluster_coords.mean(axis=0, dtype=int))
        if centroid in poses:
            rep_node = next(rep_id_gen)           # fresh node ID
            rep_pos[rep_node] = centroid

    # --- create simplified graph -------------------------------------------
    simplified_G = nx.Graph()
    simplified_G.add_nodes_from(
        [(n, {"pos": rep_pos[n]}) for n in rep_pos]
    )

    return simplified_G
def dlo_width_from_mask(
    poses: np.ndarray,          # (N, 2)  (y, x) pixel coords, centre‑line
    mask_bin: np.ndarray,       # H×W uint8, 1 = object
    statistic: Literal["median", "mean","min","max"] = "median",
) -> float:
    """
    Width (diameter) from distance‑transform sampled at centre‑line points.
    """
    if mask_bin.dtype != np.uint8:
        mask_bin = (mask_bin > 0).astype(np.uint8)

    # radius map
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)  # float32 radius

    # sample radii at centre‑line coordinates
    xs ,ys = poses[:, 0].astype(int), poses[:, 1].astype(int)
    valid = (ys >= 0) & (ys < mask_bin.shape[0]) & (xs >= 0) & (xs < mask_bin.shape[1])
    radii = dist[ys[valid], xs[valid]]    # radius at each sampled point
    diameters = radii * 2                 # convert to diameter

    if diameters.size == 0:
        raise ValueError("No valid centre‑line points inside mask.")
    if statistic == "median":
        return int(np.median(diameters))
    elif statistic == "mean":
        return int(np.mean(diameters))
    elif statistic == "min":
        return int(np.min(diameters))
    elif statistic == "max":
        return int(np.max(diameters))
    
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

def same_line_same_direction(edge_1, edge_2, tol=1e-9):
    """
    Return True if v and w are colinear and point in the same direction.
    v, w are length-2 numpy arrays (or lists / tuples).
    """
    v = np.asarray(edge_1, dtype=float)
    w = np.asarray(edge_2, dtype=float)

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