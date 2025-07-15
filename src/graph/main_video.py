import cv2
import matplotlib.pyplot as plt
import time
from graph.dlo_graph import DLOGraph  # Assuming you place the class in graph/dlo_graph.py
import logging
import numpy as np
import yaml

def load_yaml(yaml_path):
    """
    Load a YAML file and return its contents.
    
    Args:
        yaml_path (str): Path to the YAML file.
        
    Returns:
        dict: The contents of the YAML file.
    """
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        logging.error(f"YAML file not found: {yaml_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        return {}

def get_spline(mask, config):
    """
    Fit a spline to the branches of the graph.
    
    Args:
        mask: The mask to fit the spline to.
        config: Configuration dictionary containing parameters.
        
    Returns:
        DLOGraph: The processed graph with fitted splines.
    """
    
    graph = DLOGraph(config=config)
    
    load_time = time.time()
    graph.load_from_mask(
                    mask=mask,
                    config=config
                    )
    # print(f"Time to load mask: {time.time() - load_time:.3f} seconds")

    # Visualize initial graph
    if config['show_initial_graph']:
        graph.visualize(node_size=config['node_size_large'], with_labels=False, title="Initial Tree Graph")
    prune_time = time.time()
    graph.prune_short_branches_and_delete_junctions(max_length=config['max_prune_length'])
    # print(f"Time to prune branches: {time.time() - prune_time:.3f} seconds")
    
    if config['show_pruned_graph']:
        graph.visualize(node_size=config['node_size_small'], with_labels=True, title="Pruned Graph")
    start_fit = time.time()
    # Fit spline to branches
    graph.fit_spline_to_branches(smoothing=config['spline']['smoothing'], max_num_points=config['spline']['max_num_points'])
    # print(f"Time to fit spline to branches: {time.time() - start_fit:.3f} seconds")

    
    if config['show_spline_graph']:
        graph.visualize(node_size=config['node_size_small'], with_labels=False, title="Graph After Spline Fitting")
    
    start_dlo = time.time()
    graph.reconstruct_dlo_2()
    # print(f"Time to reconstruct DLO: {time.time() - start_dlo:.3f} seconds")
    if config['show_dlo_graph']:
        graph.visualize(node_size=config['node_size_small'], with_labels=False, title="Graph After DLO Reconstruction",)
    # Fit B-spline to the full graph
    graph.fit_bspline_to_graph()
    # print("Done")
    return graph


if __name__ == '__main__':

    config = {
        # File paths
        'video_path': '/home/admina/segmetation/DLOSeg/src/graph/video/spline_video.webm',

        # Graph processing parameters
        'padding_size': 0,
        'max_prune_length': 10,
        'dialate_iterations': 1,  # Number of iterations for dilation
        'erode_iterations': 1,  # Number of iterations for erosion
        'max_dist_to_connect_leafs': 30,  # Maximum distance to connect leaf nodes
        'max_dist_to_connect_nodes': 5,  # Maximum distance to connect internal nodes

        # Spline fitting parameters
        'spline': {
            'smoothing': 20,
            'max_num_points': 50
        },
        
        # Visualization settings
        'on_mask': False,  # Whether to draw the mask as background
        'show_initial_graph': False,
        'show_pruned_graph': False,
        'show_spline_graph': False,
        'show_dlo_graph': False,
        'show_rectification': False,
        'show_final_plots': False,
        'node_size_small': 1,
        'node_size_large': 5,
        'figure_size': (12, 10),
        
        
        # Processing settings
        'processing': {
            'process_right_image': False  # Set to True to process right image as well
        }
    }
    cap = cv2.VideoCapture(config['video_path'])
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w, target_h = 256, 256
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        # 1) extract & preprocess mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        if orig_h > target_h or orig_w > target_w:
            mask_proc = cv2.resize(mask, (target_w, target_h),
                                interpolation=cv2.INTER_NEAREST)
        else:
            mask_proc = mask

        # 2) run your pipeline
        try:
            G = get_spline(mask_proc, config)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
        # 3) pull out spline pts (Nx2 array) and rescale to orig size
        if hasattr(G, 'full_bspline') and G.full_bspline is not None:
            pts = G.full_bspline
        else:
            pts = np.array([G.G.nodes[n]['pos'] for n in G.G.nodes()])
            print("count ", count, "no spline found")
            

        scale = np.array([orig_w/target_w, orig_h/target_h])
        pad   = np.array([G.padding_size*2, G.padding_size*2])
        pts = (pts * scale - pad).round().astype(int)

        # 4) draw on a blank canvas instead of the frame
        canvas = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        # use polylines for a single draw call
        cv2.polylines(canvas, [pts], isClosed=False, color=(0,255,0), thickness=1)
        # cv2.namedWindow('Spline Only', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Spline Only', 1280, 720)
        # 5) display only the spline
        cv2.imshow('Spline Only', canvas)
        # print(f"Processed frame {count} with {len(pts)} spline points")
        # time.sleep(0.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break