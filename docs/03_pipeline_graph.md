# 03 — Graph pipeline (`dloseg.graph`)

The core of the repo: turn a binary DLO mask into one smooth B-spline per wire, resolving crossings. Orchestrated by `get_spline(mask, config)` in `src/dloseg/graph/pipeline.py`; all stages are methods of `DLOGraph` (`src/dloseg/graph/dlo_graph.py`).

## Stages

### 1. `load_from_mask` — mask → graph
- Pad the mask (`padding_size`), dilate then erode (`dilate_iterations` / `erode_iterations`, 3×3 rect kernel) to close small gaps.
- Thin to a 1-px skeleton with **Guo–Hall** (`cv2.ximgproc.thinning`).
- Build a k-NN graph (k=3, `scipy.spatial.KDTree`) over skeleton pixels, edges capped at `max_dist_to_connect_nodes`.
- Reduce to a **minimum spanning tree** (Prim) — removes small cycles the skeleton creates.

### 2. `prune_short_branches_and_delete_junctions` — clean topology
- `_connect_leaf_nodes`: joins leaf nodes to nearby nodes (≤ 3 px) to close skeletonization gaps — including across disconnected components.
- `_prune_short_branches`: removes dead-end branches shorter than `max_prune_length` nodes (skeletonization "hairs").
- `_prune_path_between_3_degree_nodes` + junction removal: deletes all degree>2 nodes, so only **simple paths** remain. Wire crossings are deliberately broken here and repaired in stage 4.

### 3. `fit_spline_to_branches` — smooth each path
Each remaining path is resampled through a parametric B-spline (`bspline_fitting.smooth_2d_branch_splprep`, scipy `splprep`), controlled by `spline.smoothing` and `spline.max_num_points`. The graph is rebuilt from the smoothed points.

### 4. `reconstruct_dlo_2` — resolve crossings
The junction removal in stage 2 left 4 loose endpoints at every crossing. These are clustered by proximity (`max_dist_to_connect_leafs`), and within each cluster the pairing that minimizes the **total turning angle** (tangent continuity) is connected. This is what decides "which wire continues where" at a crossing.

### 5. `fit_bspline_to_graph` — final output
For every pair of remaining leaf nodes with a connecting path, fit one B-spline (`spline.k`, `spline.n_points`). Results land in `graph.full_bsplines` — a list of `(n_points, 2)` arrays, one per wire.

## Config reference

| Key | Typical | Meaning |
|-----|---------|---------|
| `padding_size` | 0 | border padding before morphology (px) |
| `dilate_iterations` / `erode_iterations` | 1 / 1 | mask closing strength |
| `max_dist_to_connect_nodes` | 5 | k-NN edge cap when building the graph (px) |
| `max_prune_length` | 5–10 | dead-end branches shorter than this are deleted (nodes) |
| `max_dist_to_connect_leafs` | 30–60 | crossing-repair cluster radius (px) — the most sensitive knob; too small leaves wires split, too big connects different wires |
| `spline.smoothing` | 20 | scipy `splprep` s-parameter for branch smoothing |
| `spline.max_num_points` | 50–500 | resample count per branch |
| `spline.k` | 3 | final B-spline degree |
| `spline.n_points` | 200–500 | samples of the final spline |
| `show_*` flags | False | pop a matplotlib window after each stage (debugging) |

## Performance

~50 FPS on 256×256 masks, CPU only (measured over the 300-image SBHC set, 0 failures). Masks are resized to 256×256 before processing and splines rescaled back — see `scripts/benchmark_sbhc.py` for the scaling pattern.

## Gotchas

- The pipeline assumes wires **terminate at mask boundaries or free ends** — closed loops have no leaf nodes and produce no final spline.
- `full_bsplines` coordinates are in the padded, resized frame; subtract `2*padding_size` and rescale to map back to the original image (see `scripts/extract_spline.py`).
- Alternative smoothing backends (Savitzky–Golay, per-axis UnivariateSpline) exist in `bspline_fitting.py` but only `splprep` is wired in.

Next: [04 — Segmentation](04_segmentation.md)
