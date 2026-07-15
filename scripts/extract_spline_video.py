"""Real-time DLO spline extraction on a video stream.

Reads a video file frame by frame, thresholds each frame into a binary mask,
resizes to 256x256, runs the DLOGraph pipeline (dloseg/graph/pipeline.py:get_spline),
rescales the resulting spline points to the original resolution and draws
them on a blank canvas with cv2.polylines, displayed live in an OpenCV
window (press 'q' to quit).

Usage: set config['video_path'] in __main__, then `python scripts/extract_spline_video.py`.
"""
import cv2
import numpy as np

from dloseg.graph.pipeline import get_spline

if __name__ == '__main__':

    config = {
        # File paths
        'video_path': 'PATH/TO/YOUR/VIDEO.webm',  # set me: input video of the DLO

        # Graph processing parameters
        'padding_size': 0,
        'max_prune_length': 10,
        'dilate_iterations': 1,  # Number of iterations for dilation
        'erode_iterations': 1,  # Number of iterations for erosion
        'max_dist_to_connect_leafs': 30,  # Maximum distance to connect leaf nodes
        'max_dist_to_connect_nodes': 5,  # Maximum distance to connect internal nodes

        # Spline fitting parameters
        'spline': {
            'k': 3,  # B-spline degree
            'smoothing': 20,
            'n_points': 200,
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
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {config['video_path']}")
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

        # 2) run the pipeline
        try:
            G = get_spline(mask_proc, config)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        # 3) pull out spline pts and rescale to orig size
        if G.full_bsplines:
            splines = G.full_bsplines
        else:
            splines = [np.array([G.G.nodes[n]['pos'] for n in G.G.nodes()])]
            print("count ", count, "no spline found")

        scale = np.array([orig_w/target_w, orig_h/target_h])
        pad   = np.array([G.padding_size*2, G.padding_size*2])

        # 4) draw on a blank canvas instead of the frame
        canvas = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for pts in splines:
            pts = (pts * scale - pad).round().astype(int)
            # use polylines for a single draw call
            cv2.polylines(canvas, [pts], isClosed=False, color=(0,255,0), thickness=1)

        # 5) display only the spline
        cv2.imshow('Spline Only', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
