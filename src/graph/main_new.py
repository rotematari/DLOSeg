import cv2
import matplotlib.pyplot as plt
import time
from graph.dlo_graph import DLOGraph  # Assuming you place the class in graph/dlo_graph.py
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,)
logger.info("Logger initialized for DLOGraph module.")

if __name__ == '__main__':
    # Start timing
    start_total = time.time()
    
    # Load the mask
    mask_path = '/home/admina/segmetation/DLOSeg/outputs/grounded_sam2_local_demo/groundingdino_mask_0.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(mask, cmap='gray')
    
    # Create graph instance and load mask
    graph = DLOGraph(logger=logger)
    graph.load_from_mask(
                    mask,
                        downsample=1,
                        statistic="mean",
                        min_cluster_factor=2.50,
                        ds_factor=1.0,
                    )

    # Visualize initial graph
    # graph.visualize(node_size=1, with_labels=True, title="Initial Graph")
    graph.visualize_on_mask(node_size=1, with_labels=False, title="Initial Graph on Mask")
    
    # Connect with bending energy
    graph.connect_with_bending_energy(k=10, 
                                      be_limit=0.5, 
                                      radius_factor=1.5,
                                      show_log=False)

    graph.visualize(node_size=5, with_labels=False, title="Graph After Bending Energy")

    # Connect leaf nodes
    # graph.connect_leaf_nodes(be_limit=5.0,max_dist_factor=3.0)
    graph.connect_nodes_by_topology(be_limit=1.50,
                                    max_dist_factor=3.0,
                                    show_log=True)

    graph.visualize(node_size=5, with_labels=False, title="Graph After Node Connection")

    # Prune short branches
    graph.prune_short_branches(min_length=10)
    
    # Visualize final graph
    graph.visualize(node_size=5, with_labels=False, title="Final Graph")

    # Find optimal path
    # path = graph.optimize_path()
    
    # # Fit spline
    # if path:
    #     x_spline, y_spline = graph.fit_spline(path, smoothing=10)
    #     plt.plot(x_spline, y_spline, 'g-', lw=2)
    #     plt.gca().invert_yaxis()
    #     plt.pause(0.01)
    
    print(f"Total processing time: {time.time() - start_total:.3f} seconds")
    print("Done")