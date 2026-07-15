"""The binary-mask -> 2D B-spline pipeline entry function.

`get_spline` orchestrates the full DLOGraph flow:
load -> prune -> per-branch spline fit -> DLO reconstruction -> B-spline fit.
Used by the entry points in scripts/.

Stereo/ZED helpers (calibration, rectification) live in dloseg.recon3d.stereo;
3D triangulation lives in dloseg.recon3d.bspline_3d_recon.
"""

import time

from dloseg.graph.dlo_graph import DLOGraph


def get_spline(mask, config, verbose=False):
    """
    Run the full DLO pipeline on a binary mask:
    load -> prune -> per-branch spline fit -> DLO reconstruction -> B-spline fit.

    Args:
        mask: Binary mask (np.ndarray) of the DLO(s).
        config: Configuration dictionary containing parameters.
        verbose: If True, print per-stage timing.

    Returns:
        DLOGraph: The processed graph with fitted splines in `full_bsplines`.
    """
    graph = DLOGraph(config=config)

    load_time = time.time()
    graph.load_from_mask(mask=mask, config=config)
    if verbose:
        print(f"Time to load mask: {time.time() - load_time:.3f} seconds")

    # Visualize initial graph
    if config["show_initial_graph"]:
        graph.visualize(
            node_size=config["node_size_large"], with_labels=False, title="Initial Tree Graph"
        )

    prune_time = time.time()
    graph.prune_short_branches_and_delete_junctions(max_length=config["max_prune_length"])
    if verbose:
        print(f"Time to prune branches: {time.time() - prune_time:.3f} seconds")

    if config["show_pruned_graph"]:
        graph.visualize(
            node_size=config["node_size_small"], with_labels=False, title="Pruned Graph"
        )

    start_fit = time.time()
    # Fit spline to branches
    graph.fit_spline_to_branches(
        smoothing=config["spline"]["smoothing"], max_num_points=config["spline"]["max_num_points"]
    )
    if verbose:
        print(f"Time to fit spline to branches: {time.time() - start_fit:.3f} seconds")

    if config["show_spline_graph"]:
        graph.visualize(
            node_size=config["node_size_small"],
            with_labels=False,
            title="Graph After Spline Fitting",
        )

    start_dlo = time.time()
    graph.reconstruct_dlo_2()
    if verbose:
        print(f"Time to reconstruct DLO: {time.time() - start_dlo:.3f} seconds")
    if config["show_dlo_graph"]:
        graph.visualize(
            node_size=config["node_size_small"],
            with_labels=False,
            title="Graph After DLO Reconstruction",
        )

    # Fit B-spline to the full graph
    graph.fit_bspline_to_graph()
    return graph
