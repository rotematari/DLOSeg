"""Stereo camera helpers for the 3D reconstruction work.

ZED calibration parsing and stereo rectification — the front half of the
3D pipeline (rectified left/right masks -> 2D splines -> triangulated 3D
spline, see recon3d/bspline_3d_recon.py):

- load_yaml:                 safe YAML loader.
- get_zed_calibration:       parse a ZED calibration YAML into K/D/R/T/P.
- rectify_stereo_pair:       cv2.stereoRectify-based rectification.
- get_images_from_stereo_pair: load left/right mask images from disk.
- visualize_rectification:   before/after rectification comparison plot.
"""
import logging

import cv2
import matplotlib.pyplot as plt
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


def rectify_stereo_pair(image_left, image_right, calib_data, alpha=0,
                        interpolation=cv2.INTER_LINEAR, verbose=True):
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
        interpolation: cv2 interpolation flag for the remap. Use
                       cv2.INTER_NEAREST for binary masks so they stay binary.
        verbose (bool): Print the computed rectification data.

    Returns:
        tuple: (rectified_left, rectified_right, P1, P2)
               The two rectified images plus the NEW projection matrices to
               use for triangulation in the rectified frame.
    """
    if verbose:
        print("Starting stereo rectification process...")

    # Extract calibration parameters
    K1, D1 = calib_data['K1'], calib_data['D1']
    K2, D2 = calib_data['K2'], calib_data['D2']
    R, T = calib_data['R'], calib_data['T']

    # Get image size
    height, width = image_left.shape
    image_size = (width, height)

    # --- Step 1: Compute Rectification Transforms ---
    # This function computes the rotation matrices (R1, R2), new projection
    # matrices (P1, P2), and a disparity-to-depth mapping matrix (Q).
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=alpha
    )

    if verbose:
        print("\n--- Calculated Rectification Data ---")
        print("New Left Projection Matrix (P1):\n", P1)
        print("\nNew Right Projection Matrix (P2):\n", P2)
        print("Note: These new P1 and P2 matrices should be used for triangulation with the rectified images.")
        print("-----------------------------------")

    # --- Step 2: Compute Undistortion and Rectification Maps ---
    # This creates a lookup table for where each pixel in the new rectified
    # image comes from in the original distorted image.
    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    # --- Step 3: Apply the Maps to the Images ---
    rectified_left = cv2.remap(image_left, map1_left, map2_left, interpolation=interpolation)
    rectified_right = cv2.remap(image_right, map1_right, map2_right, interpolation=interpolation)

    return rectified_left, rectified_right, P1, P2


def get_zed_calibration(zed_calib_path, res='720p'):
    """
    Get the ZED calibration parameters for a given resolution.

    Args:
        zed_calib_path (str): Path to the ZED calibration YAML file.
        res (str): Resolution string, one of '2K', '1080p' (FHD), '720p' (HD), or 'VGA' (case-insensitive).

    Returns:
        dict: {
            'K1': 3x3 left intrinsic,
            'D1': (k1,k2,p1,p2,k3) left distortion,
            'K2': 3x3 right intrinsic,
            'D2': (k1,k2,p1,p2,k3) right distortion,
            'R': 3x3 rotation from left->right,
            'T': 3x1 translation from left->right (meters),
            'P1': 3x4 left projection,
            'P2': 3x4 right projection,
            'baseline': float (meters),
            'resolution': str (suffix used)
        }
    """
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

    # helper: Euler angles (X, Y, Z) -> rotation matrix
    def euler_to_R(rx, ry, rz):
        Rx = cv2.Rodrigues(np.array([rx, 0.0, 0.0]))[0]
        Ry = cv2.Rodrigues(np.array([0.0, ry, 0.0]))[0]
        Rz = cv2.Rodrigues(np.array([0.0, 0.0, rz]))[0]
        return Rz @ Ry @ Rx

    R = euler_to_R(rx, ry, rz)
    # OpenCV convention: T maps left-camera coords into the right-camera frame,
    # so a right camera physically at +B along x gives t_x = -B.
    # TY/TZ are in millimeters in the ZED YAML, like Baseline.
    T = np.array([-baseline, ty / 1000.0, tz / 1000.0], dtype=np.float64).reshape(3, 1)

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


def get_images_from_stereo_pair(image_left_path, image_right_path):
    """Load the left/right mask images (grayscale) from disk."""
    img_l = cv2.imread(image_left_path, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(image_right_path, cv2.IMREAD_GRAYSCALE)
    if img_l is None:
        raise FileNotFoundError(f"Could not read left image: {image_left_path}")
    if img_r is None:
        raise FileNotFoundError(f"Could not read right image: {image_right_path}")
    return img_l, img_r


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
