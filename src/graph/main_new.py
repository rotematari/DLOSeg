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

def get_spline(mask):
    """
    Fit a spline to the branches of the graph.
    
    Args:
        graph (DLOGraph): The graph to fit the spline to.
        smoothing (int): Smoothing factor for the spline fitting.
        max_num_points (int, optional): Maximum number of points in the spline.
        
    Returns:
        None
    """
    graph = DLOGraph()
    print(f"Time to setup graph: {time.time() - setup_time:.3f} seconds")
    load_time = time.time()
    graph.load_from_mask(
                    mask=mask,
                        statistic="mean",
                        min_cluster_factor=0.50,
                        padding_size=40,
                    )
    print(f"Time to load mask: {time.time() - load_time:.3f} seconds")

    # Visualize initial graph
    graph.visualize(node_size=5, with_labels=False, title="Initial Tree Graph")
    prune_time = time.time()
    graph.prune_short_branches_and_delete_junctions(max_length=6)
    print(f"Time to prune branches: {time.time() - prune_time:.3f} seconds")
    

    
    graph.visualize(node_size=1, with_labels=False, title="Pruned Graph")
    start_fit = time.time()
    # Fit spline to branches
    graph.fit_spline_to_branches(smoothing=20, max_num_points=10)
    print(f"Time to fit spline to branches: {time.time() - start_fit:.3f} seconds")
    time_to_prune = time.time()
    
    graph.visualize(node_size=1, with_labels=False, title="Graph After Spline Fitting")
    
    start_dlo = time.time()
    graph.reconstruct_dlo()
    print(f"Time to reconstruct DLO: {time.time() - start_dlo:.3f} seconds")
    graph.visualize(node_size=1, with_labels=False, title="Graph After DLO Reconstruction")

    print(f"Total processing time: {time.time() - start_total:.3f} seconds")
    print("Done")
    
    graph.fit_bspline_to_graph()
    return graph



def rectify_stereo_pair(image_left, image_right, calib_data, alpha=0):
    """
    Performs stereo rectification on a pair of images.

    Args:
        image_left (np.ndarray): The raw (distorted) left image.
        image_right (np.ndarray): The raw (distorted) right image.
        calib_data (dict): Dictionary with camera calibration data containing:
                           K1, D1, K2, D2, R, T.
        alpha (float): Free scaling parameter. alpha=0 means the rectified images
                       are zoomed in to show only valid pixels. alpha=1 means all
                       source image pixels are retained, possibly with black borders.

    Returns:
        tuple: (rectified_left, rectified_right)
               A tuple of the two rectified images.
    """
    print("Starting stereo rectification process...")
    
    # Extract calibration parameters
    K1, D1 = calib_data['K1'], calib_data['D1']
    K2, D2 = calib_data['K2'], calib_data['D2']
    R, T = calib_data['R'], calib_data['T']
    
    # Get image size
    height, width = image_left.shape
    image_size = (width, height)
    
    # --- Step 1: Compute Rectification Transforms ---
    print(f"1. Computing rectification transforms with cv2.stereoRectify (alpha={alpha})...")
    # This function computes the rotation matrices (R1, R2), new projection
    # matrices (P1, P2), and a disparity-to-depth mapping matrix (Q).
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=alpha
    )

    print("\n--- Calculated Rectification Data ---")
    print("New Left Projection Matrix (P1):\n", P1)
    print("\nNew Right Projection Matrix (P2):\n", P2)
    print("Note: These new P1 and P2 matrices should be used for triangulation with the rectified images.")
    print("-----------------------------------")


    # --- Step 2: Compute Undistortion and Rectification Maps ---
    print("\n2. Computing rectification maps with cv2.initUndistortRectifyMap...")
    # This creates a lookup table for where each pixel in the new rectified
    # image comes from in the original distorted image.
    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    
    # --- Step 3: Apply the Maps to the Images ---
    print("3. Applying maps to images with cv2.remap...")
    rectified_left = cv2.remap(image_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
    rectified_right = cv2.remap(image_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)
    
    print("\nRectification complete.")
    return rectified_left, rectified_right

def get_zed_calibration(zed_calib_path, res='720p'):
    """
    Get the ZED calibration parameters for a given resolution.

    Args:
        zed_calib_path (str): Path to the ZED calibration YAML file.
        res (str): Resolution string, one of '2K', '1080p' (FHD), '720p' (HD), or 'VGA' (case‐insensitive).

    Returns:
        dict: {
            'K1': 3×3 left intrinsic,
            'D1': (k1,k2,p1,p2,k3) left distortion,
            'K2': 3×3 right intrinsic,
            'D2': (k1,k2,p1,p2,k3) right distortion,
            'R': 3×3 rotation from left→right,
            'T': 3×1 translation from left→right (meters),
            'P1': 3×4 left projection,
            'P2': 3×4 right projection,
            'baseline': float (meters),
            'resolution': str (suffix used)
        }
    """
    # load_yaml should return the parsed YAML as a dict
    calib = load_yaml(zed_calib_path)

    # map common res tags to your YAML suffixes
    res_map = {
        '2k':    '2K',
        '1080p': 'FHD',
        'fhd':   'FHD',
        '720p':  'HD',
        'hd':    'HD',
        'vga':   'VGA'
    }
    suffix = res_map.get(res.lower(), res)

    # pick the matching intrinsics + distortion
    left  = calib[f'LEFT_CAM_{suffix}']
    right = calib[f'RIGHT_CAM_{suffix}']

    K1 = np.array([
        [left['fx'],    0.0,        left['cx']],
        [0.0,           left['fy'], left['cy']],
        [0.0,           0.0,        1.0       ]
    ])
    D1 = np.array([ left['k1'], left['k2'], left['p1'], left['p2'], left['k3'] ])

    K2 = np.array([
        [right['fx'],   0.0,         right['cx']],
        [0.0,           right['fy'], right['cy']],
        [0.0,           0.0,         1.0        ]
    ])
    D2 = np.array([ right['k1'], right['k2'], right['p1'], right['p2'], right['k3'] ])

    # stereo extrinsics
    stereo = calib['STEREO']
    # baseline in meters (YAML is in mm)
    baseline = stereo['Baseline'] / 1000.0
    rx = stereo[f'RX_{suffix}']
    ry = stereo[f'CV_{suffix}']
    rz = stereo[f'RZ_{suffix}']
    ty = stereo['TY']
    tz = stereo['TZ']

    # helper: Euler angles (X, Y, Z) → rotation matrix
    def euler_to_R(rx, ry, rz):
        Rx = cv2.Rodrigues(np.array([rx, 0.0, 0.0]))[0]
        Ry = cv2.Rodrigues(np.array([0.0, ry, 0.0]))[0]
        Rz = cv2.Rodrigues(np.array([0.0, 0.0, rz]))[0]
        return Rz @ Ry @ Rx

    R = euler_to_R(rx, ry, rz)
    T = np.array([baseline, ty, tz], dtype=np.float64).reshape(3, 1)

    # build projection matrices
    P1 = K1 @ np.hstack([ np.eye(3), np.zeros((3,1)) ])
    P2 = K2 @ np.hstack([ R, T ])

    return {
        'K1': K1, 'D1': D1,
        'K2': K2, 'D2': D2,
        'R': R,   'T': T,
        'P1': P1, 'P2': P2,
        'baseline': baseline,
        'resolution': suffix
    }
def visualize_rectification(raw_left, raw_right, rectified_left, rectified_right):
    """Displays the original and rectified images side-by-side with epipolar lines."""
    
    # Combine images for comparison
    raw_combined = np.concatenate((raw_left, raw_right), axis=1)
    rectified_combined = np.concatenate((rectified_left, rectified_right), axis=1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.imshow(cv2.cvtColor(raw_combined, cv2.COLOR_BGR2RGB))
    ax1.set_title('Before Rectification (Raw Images)')
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(rectified_combined, cv2.COLOR_BGR2RGB))
    ax2.set_title('After Rectification (Epipolar lines are horizontal)')
    ax2.axis('off')

if __name__ == '__main__':

    zed_calib_path = '/home/admina/segmetation/DLOSeg/src/zed/calibration_data/zed_2i_cal_data.yaml'
    zed_calibration = get_zed_calibration(zed_calib_path)
    # Start timing
    start_total = time.time()
    read_time = time.time()
    # Load the mask
    mask_l_path = '/home/admina/segmetation/DLOSeg/outputs/seg_data_720_15fps/img_06_left_mask_0.png'
    mask_l = cv2.imread(mask_l_path, cv2.IMREAD_GRAYSCALE)
    mask_r_path = '/home/admina/segmetation/DLOSeg/outputs/seg_data_720_15fps/img_06_right_mask_0.png'
    mask_r = cv2.imread(mask_r_path, cv2.IMREAD_GRAYSCALE)
    img_real_path ='/home/admina/segmetation/DLOSeg/src/graph/data_720_15fps/img_06.png'
    img_real = cv2.imread(img_real_path, cv2.IMREAD_GRAYSCALE)
    # split to left right
    img_real_l, img_real_r = img_real[:, :img_real.shape[1] // 2], img_real[:, img_real.shape[1] // 2:]

    
    print(f"Time to read images: {time.time() - read_time:.3f} seconds")
    # Normalize mask to 0-1 range
    setup_time = time.time()
    img_real_l = img_real_l * mask_l  # Normalize mask to 0-1 range
    img_real_l[img_real_l < img_real_l[img_real_l > 0].mean()] = 0
    
    img_real_r = img_real_r * mask_r  # Normalize mask to 0-1 range
    img_real_r[img_real_r < img_real_r[img_real_r > 0].mean()] = 0
    # plt.imshow(img_real_r, cmap='gray')
    # plt.show()
    # Create graph instance and load mask

    rect_left, rect_right = rectify_stereo_pair(img_real_l, img_real_r, zed_calibration, alpha=0)
    
    # visualize_rectification(img_real_l, img_real_r, rect_left, rect_right)
    
    print("-----------left-------------\n")
    G_left = get_spline(rect_left)
    print("-----------right-------------\n")
    G_right = get_spline(rect_right)


    
    # plot full_bspline of left and right
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(G_left.full_bspline[:, 0], G_left.full_bspline[:, 1], 'r-', label='Left Full B-spline')
    ax1.set_title('Left Full B-spline')
    ax1.axis('equal')
    ax1.legend()
    ax2.plot(G_right.full_bspline[:, 0], G_right.full_bspline[:, 1], 'b-', label='Right Full B-spline')
    ax2.set_title('Right Full B-spline')
    ax2.axis('equal')
    ax2.legend()
    plt.show()
    print(f"Left graph nodes: {len(G_left.nodes())}, edges: {len(G_left.edges())}")


    # spline matching