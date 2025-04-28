import numpy as np
import matplotlib.pyplot as plt
from zed.utils.utils import load_yaml  # Assumes load_yaml is defined here
import open3d as o3d
from scipy.signal import savgol_filter
import cv2
def depth_to_point_cloud(depth, fx, fy, cx, cy, y_up=False, flatten=True):
    """
    Convert depth map (metric, camera frame) to 3‑D point cloud.
    Parameters
    ----------
    depth : (H,W) array_like
        Depth in metres along +Z.
    fx, fy, cx, cy : float
        Pinhole intrinsics.
    y_up : bool, optional
        Flip Y to point upwards.
    flatten : bool, optional
        If True, return (N,3); else (H,W,3).
    """
    depth = depth.astype(np.float32)
    H, W = depth.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    if y_up:
        Y = -Y
    Z = depth

    if flatten:
        pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        valid = np.isfinite(pts[:, 2]) & (pts[:, 2] > 0)
        return pts[valid]
    else:
        pts = np.stack((X, Y, Z), axis=-1)
        pts[~np.isfinite(pts[..., 2])] = np.nan
        return pts

def visualize_point_cloud_plt(points, sample=1):
    """
    Visualize a point cloud using matplotlib's 3D scatter plot.
    
    Parameters:
    - points: A (H, W, 3) numpy array of 3D coordinates.
    - sample: Factor to subsample the points (default is 1, i.e. no subsampling).
    """
    # Optionally subsample the point cloud for visualization performance
    if sample > 1:
        points = points[::sample, ::sample, :]
    
    # Reshape the array to a list of 3D points (N x 3)
    points = points.reshape(-1, 3)
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, c=points[:, 2], cmap='viridis')
    
    ax.set_title("Point Cloud Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.colorbar(scatter, ax=ax, label="Depth")
    plt.show()


def visualize_point_cloud(points, sample=1):
    """
    Visualize a point cloud using Open3D.

    Parameters:
    - points: A (H, W, 3) numpy array of 3D coordinates.
    - sample: Factor to subsample the points (default is 1, i.e., no subsampling).
    """
    # Optionally subsample for better visualization performance
    if sample > 1:
        points = points[::sample, :]

    # Flatten the array to (N, 3)
    points = points.reshape(-1, 3)

    # Remove zero points (points with no depth)
    points = points[np.any(points != 0, axis=1)]

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize using Open3D
    o3d.visualization.draw_geometries([pcd])


def sg_filter(depth_map,path_coords):
    # Extract depth values along the path
    path_values = np.array([depth_map[y, x] for (y, x) in path_coords])

    # Apply Savitzky-Golay filter
    window_length = 200  # Choose an odd integer >= polyorder+2
    polyorder = 5
    smoothed_values = savgol_filter(path_values, window_length, polyorder)
    # Write smoothed values back to depth map (optional)
    for (y, x), val in zip(path_coords, smoothed_values):
        depth_map[y, x] = val
    return depth_map

if __name__ == "__main__":
    # Load depth map from file
    depth_file = "/home/admina/segmetation/DLOSeg/src/FoundationStereo/output/depth_meter.npy"
    depth = np.load(depth_file)
    img_path = "/home/admina/segmetation/DLOSeg/outputs/grounded_sam2_local_demo/groundingdino_mask_0.png"
    # Load mask image
    # mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    path_1 = np.load("/home/admina/segmetation/DLOSeg/path_img_1.npy")
    mask_new = [(x-1,y-1) for x,y in path_1]
    mask = np.zeros_like(depth)
    for x,y in mask_new:
        mask[x,y] = 1
    depth = depth * mask
    
    # depth = sg_filter(depth,mask_new)
    
    # Load YAML calibration data using load_yaml from zed.utils.utils
    yaml_file = "/home/admina/segmetation/DLOSeg/src/zed/calibration_data/cal_data.yaml"
    cal_data = load_yaml(yaml_file)
    
    # For this example, we'll use the 'LEFT_CAM_2K' intrinsic parameters
    cam_params = cal_data.get("LEFT_CAM_2K")
    if cam_params is None:
        raise ValueError("LEFT_CAM_2K parameters not found in calibration data")
    
    fx = cam_params.get("fx")
    fy = cam_params.get("fy")
    cx = cam_params.get("cx")
    cy = cam_params.get("cy")
    
    if None in [fx, fy, cx, cy]:
        raise ValueError("Missing intrinsic parameters in the calibration data")
    
    # Convert the depth map to a point cloud
    point_cloud = depth_to_point_cloud(depth, fx, fy, cx, cy)
    
    np.save("/home/admina/segmetation/DLOSeg/src/FoundationStereo/output/point_cloud.npy", point_cloud)
    
    # Visualize the point cloud
    visualize_point_cloud(point_cloud, sample=2)  # Adjust sample factor as needed for performance
