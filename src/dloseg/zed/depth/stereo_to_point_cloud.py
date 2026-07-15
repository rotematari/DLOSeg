import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import pyzed.sl as sl  # ZED SDK Python bindings

class ZedPointCloudProcessor:
    def __init__(self):
        """Initialize ZED camera processor"""
        # Create a ZED camera object
        self.zed = sl.Camera()
        
        # Set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # ULTRA for max quality
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.3  # Min depth in meters
        
    def open_camera(self):
        """Open the ZED camera"""
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")
        print("ZED camera opened successfully")
        
    def close_camera(self):
        """Close the ZED camera"""
        self.zed.close()
        print("ZED camera closed")
        
    def process_frame(self):
        """Capture and process a single frame"""
        # Create image and point cloud objects
        image = sl.Mat()
        point_cloud = sl.Mat()
        depth = sl.Mat()
        
        # Set runtime parameters
        runtime_params = sl.RuntimeParameters()
        runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD
        
        # Grab a new frame
        if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            
            # Retrieve depth map
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            
            # Retrieve colored point cloud
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            # Convert to numpy arrays
            image_np = image.get_data()
            depth_np = depth.get_data()
            point_cloud_np = point_cloud.get_data()
            
            return image_np, depth_np, point_cloud_np
        else:
            return None, None, None
            
    def process_images(self, left_img_path, right_img_path):
        """
        Process stereo images using ZED SDK's depth estimation
        
        Note: This is less accurate than using the camera directly, 
        as ZED SDK is optimized for their specific camera hardware
        """
        # Load the images
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        if left_img is None or right_img is None:
            raise ValueError("Could not load stereo images")
            

        
        
        # Create an artificial SVO file from the images
        # Note: This is a simplified approach and may not work perfectly
        # For best results, use the ZED camera directly
        
        # For demonstration - this would need more work for real implementation
        print("Note: For best results, capture directly from ZED camera")
        print("Processing images with standard OpenCV fallback...")
        
        # Fall back to OpenCV implementation when not using camera directly
        return self.process_stereo_opencv(left_img, right_img)
    
    def process_stereo_opencv(self, left_img, right_img):
        """Fallback to OpenCV when direct ZED hardware not available"""
        # Similar to your existing implementation
        # This is just a simplified version
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        
        return 
    
    def save_point_cloud(self, point_cloud, output_file):
        """Save point cloud to file"""
        pcd = o3d.geometry.PointCloud()
        
        # Handle point clouds with or without color
        if point_cloud.shape[1] >= 6:
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255.0)
        else:
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            
        # Save to file
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"Point cloud saved to {output_file}")
        
    def visualize_point_cloud(self, point_cloud):
        """Visualize point cloud using Open3D"""
        pcd = o3d.geometry.PointCloud()
        
        # Handle point clouds with or without color
        if isinstance(point_cloud, np.ndarray):
            if point_cloud.shape[1] >= 6:
                pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255.0)
            else:
                pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        # Create coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd, coord_frame])


def main():
    # Initialize processor
    processor = ZedPointCloudProcessor()
    
    # Two modes of operation:
    
    # 1. Direct camera capture (preferred)
    use_camera = False  # Set to True when you have a ZED camera connected
    
    if use_camera:
        try:
            processor.open_camera()
            print("Capturing frame from ZED camera...")
            image, depth, point_cloud = processor.process_frame()
            output_file = "/home/admina/segmetation/DLOSeg/src/zed/output/zed_point_cloud.ply"
            processor.save_point_cloud(point_cloud, output_file)
            processor.visualize_point_cloud(point_cloud)
        finally:
            processor.close_camera()
    
    # 2. Process existing stereo images
    else:
        print("Processing existing stereo images...")
        left_img_path = "src/zed/record/recordings/png/left/left000300.png"
        right_img_path = "src/zed/record/recordings/png/right/right000300.png"
        
        image, depth, point_cloud = processor.process_images(left_img_path, right_img_path)
        output_file = "/home/admina/segmetation/DLOSeg/src/zed/output/point_cloud.ply"
        processor.save_point_cloud(point_cloud, output_file)
        processor.visualize_point_cloud(point_cloud)


if __name__ == "__main__":
    main()