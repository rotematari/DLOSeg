import numpy as np
import matplotlib.pyplot as plt
from zed.utils.utils import load_yaml  # Assumes load_yaml is defined here

def depth_to_point_cloud(depth, fx, fy, cx, cy):
    """
    Convert a depth map to a 3D point cloud.
    
    Parameters:
    - depth: 2D numpy array with depth values.
    - fx, fy: Focal lengths.
    - cx, cy: Principal point coordinates.
    
    Returns:
    - points: A (H, W, 3) array containing the 3D coordinates.
    """
    height, width = depth.shape
    # Create a grid of pixel coordinates (u,v)
    u = np.tile(np.arange(width), (height, 1))
    v = np.tile(np.arange(height), (width, 1)).T

    # Compute 3D coordinates
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    
    # Stack into an array of 3D points
    points = np.stack((X, Y, Z), axis=-1)
    return points

def visualize_point_cloud(points, sample=1):
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

if __name__ == "__main__":
    # Load depth map from file
    depth_file = "/home/admina/segmetation/DLOSeg/src/FoundationStereo/output/depth_meter.npy"
    depth = np.load(depth_file)
    
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
    visualize_point_cloud(point_cloud, sample=5)  # Adjust sample factor as needed for performance
