
import math
import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx

def extract_skeleton(mask):
    skel = skeletonize(mask > 0).astype(np.uint8)
    ys, xs = np.where(skel)
    G = nx.Graph()
    pix_id = { (y, x): idx for idx, (y, x) in enumerate(zip(ys, xs)) }

    # populate nodes
    for idx, (y, x) in enumerate(zip(ys, xs)):
        G.add_node(idx, pos=(y, x))

    # 8‑neighbour connectivity
    neigh = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    for y, x in zip(ys, xs):
        for dy, dx in neigh:
            n = (y + dy, x + dx)
            if n in pix_id:
                G.add_edge(pix_id[(y, x)], pix_id[n])

    # endpoints = degree 1
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    if len(endpoints) < 2:
        raise RuntimeError("Failed to find unique endpoints – check segmentation or use multi‑component stitching.")

    # choose the longest simple path between any pair of endpoints
    longest = []
    for s in endpoints:
        for t in endpoints:
            if s >= t:
                continue
            try:
                path = nx.shortest_path(G, source=s, target=t)  # simple path in skeleton graph
                if len(path) > len(longest):
                    longest = path
            except nx.NetworkXNoPath:
                continue
    path_pixels = [G.nodes[n]['pix'][::-1] for n in longest]  # (x,y)
    return np.array(path_pixels, dtype=np.float32),G