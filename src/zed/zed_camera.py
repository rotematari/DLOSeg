import pyzed.sl as sl
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from threading import Thread, Event
import time
import Grounded_SAM_2.grounding_dino.groundingdino.datasets.transforms as T
from typing import Tuple, List
import torch
from graph_3d.segmentor import Segmentor
from PIL import Image
import supervision as sv


class ZedCamera:
    """
    ZED camera handler with real-time visualization capabilities
    """
    def __init__(self,
                 resolution=sl.RESOLUTION.HD2K,  # or HD1080, HD720, etc.
                 fps=30,
                 depth_mode=sl.DEPTH_MODE.NEURAL_PLUS,  # or NEURAL, NEURAL_PLUS
                 coordinate_units=sl.UNIT.METER,
                 depth_min_distance=0.4,
                 depth_max_distance=10.0,
                 use_cuda=True):
        """
        Initialize ZED camera with specified parameters
        
        Parameters:
        -----------
        resolution : sl.RESOLUTION
            Camera resolution (HD2K, HD1080, HD720, etc.)
        fps : int
            Camera frame rate
        depth_mode : sl.DEPTH_MODE
            Depth sensing mode (NEURAL_PLUS for highest quality)
        coordinate_units : sl.UNIT
            Units for measurements (METER, MILLIMETER, etc.)
        depth_min_distance : float
            Minimum depth distance
        depth_max_distance : float
            Maximum depth distance
        use_cuda : bool
            Whether to use CUDA acceleration
        """
        # Create camera instance
        self.camera = sl.Camera()
        self.segmentor = Segmentor
        # Initialize parameters
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = depth_mode
        self.init_params.coordinate_units = coordinate_units
        self.init_params.depth_minimum_distance = depth_min_distance
        self.init_params.depth_maximum_distance = depth_max_distance
        
        # Enable CUDA if specified
        if not use_cuda:
            self.init_params.disable_cuda = True
            
        # Runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        # self.runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD
        self.runtime_params.confidence_threshold = 100
        self.runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
        
        
        # Camera state
        self.is_open = False
        self.is_streaming = False
        self.stop_event = Event()
        
        # Data containers
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()
        
        # Visualization components
        self.image_vis_window = "ZED Camera - RGB"
        self.depth_vis_window = "ZED Camera - Depth"
        self.pc_vis = None  # Open3D visualizer
        
        # segmentation masks should be a tuple of (time, mask)
        self.segmented_mask = None
        self.count = 0
        self.tot_time = 0
    def open(self):
        """
        Open the ZED camera with configured parameters
        
        Returns:
        --------
        bool : Success status
        """
        if self.is_open:
            print("Camera is already open")
            return True
            
        status = self.camera.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open ZED camera: {status}")
            return False
            
        print("ZED camera opened successfully")
        self.is_open = True
        # Register key callbacks for visualization controls
        self.register_key_callbacks()
        return True
        
    def close(self):
        """Close the ZED camera"""
        if self.is_streaming:
            self.stop_streaming()
        
        if self.is_open:
            self.camera.close()
            self.is_open = False
            print("ZED camera closed")
            
        # Close visualization windows
        cv2.destroyAllWindows()
        if self.pc_vis is not None:
            self.pc_vis.destroy_window()
    
    def get_camera_info(self):
        """Get and print camera information"""
        if not self.is_open:
            print("Camera not open")
            return None
            
        info = self.camera.get_camera_information()
        print(f"Camera Model: {info.camera_model}")
        print(f"Serial Number: {info.serial_number}")
        print(f"Camera Firmware: {info.camera_configuration.firmware_version}")
        print(f"Resolution: {info.camera_configuration.resolution.width}x{info.camera_configuration.resolution.height}")
        print(f"FPS: {info.camera_configuration.fps}")
        
        return info
        
    def grab_frame(self):
        """
        Capture a single frame from the camera
        
        Returns:
        --------
        tuple : (image_np, depth_np, point_cloud_np) or (None, None, None) on error
        """
        if not self.is_open:
            print("Camera not open")
            return None, None, None
            
        # Grab a new frame
        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            self.camera.retrieve_image(self.image, sl.VIEW.LEFT)
            
            # Retrieve depth map
            self.camera.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            
            # Retrieve colored point cloud
            self.camera.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            
            # Convert to numpy arrays
            image_np = self.image.get_data()
            depth_np = self.depth.get_data()
            point_cloud_np = self.point_cloud.get_data()
            
            return image_np, depth_np, point_cloud_np
        else:
            print("Error capturing frame")
            return None, None, None
    
    def visualize_frame(self, image=True, depth=True, point_cloud=False, segment=True):
        """
        Visualize a single frame (image, depth, and/or point cloud)
        
        Parameters:
        -----------
        image : bool
            Whether to display RGB image
        depth : bool
            Whether to display depth map
        point_cloud : bool
            Whether to display point cloud
        """
        image_np, depth_np, point_cloud_np = self.grab_frame()

        if image_np is None:
            return
        try:
            seg_time = time.time()
            # Segment the image
            source_image, sam_input_boxes,sam_mask = self.segment_image(image_np)
            self.count += 1
            self.tot_time += time.time() - seg_time

            if self.count > 100:
                print(f"Average segmentation time: {self.tot_time / self.count:.2f} seconds")
                self.count = 0
                self.tot_time = 0

            if len(sam_mask.shape) == 3:
                sam_mask = sam_mask.squeeze() 
        except Exception as e:
            print(f"Error during segmentation: {e}")
            sam_mask = None
        # Display RGB image
        if image:
            
            cv2.namedWindow("Segmented Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Segmented Image", 1500, 840)
            cv2.imshow("Segmented Image", sam_mask)
            
        # Display depth map
        if depth:
            # Normalize depth for visualization
            depth_vis = np.zeros_like(depth_np)
            valid_mask = depth_np > 0
            if valid_mask.any():
                depth_vis[valid_mask] = cv2.normalize(
                    depth_np[valid_mask], 
                    None, 
                    0, 255, 
                    cv2.NORM_MINMAX, 
                    dtype=cv2.CV_8U
                )
            # Apply colormap
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow(self.depth_vis_window, depth_vis)
            
        # Display point cloud
        if point_cloud:
            # segment the point cloud
            if sam_mask is not None:
                # Apply the mask to the point cloud
                try:
                    sam_mask = sam_mask.astype(np.uint8)
                    # point_cloud_np = point_cloud_np[sam_mask == 1]
                    self._visualize_point_cloud(point_cloud_np,sam_mask)
                except Exception as e:
                    print(f"Error applying mask to point cloud: {e}")
                    


        # Process UI events
        cv2.waitKey(1)
    def center_point_cloud(self):
        """Center the view on the current point cloud"""
        if self.pc_vis is not None and hasattr(self, 'pcd_geometry'):
            # Reset view to center on the point cloud
            self.pc_vis.reset_view_point(True)
            print("Point cloud view centered")
    def set_view_angle(self, view_preset):
        """
        Change the camera viewpoint to a preset angle
        
        Parameters:
        -----------
        view_preset : str
            One of 'front', 'top', 'side', 'isometric'
        """
        if self.pc_vis is None:
            return
            
        # Get view control
        view_control = self.pc_vis.get_view_control()
        
        # Get the bounding box to determine appropriate camera distance
        if not hasattr(self, 'pcd_geometry') or len(self.pcd_geometry.points) == 0:
            return
            
        bbox = self.pcd_geometry.get_axis_aligned_bounding_box()
        bbox_center = bbox.get_center()
        bbox_size = bbox.get_extent()
        
        # Camera distance based on bounding box size
        dist = max(bbox_size) * 2.5
        
        # Set parameters based on view preset
        if view_preset == 'front':
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, -1, 0])
            view_control.set_lookat(bbox_center)
            view_control.set_zoom(0.7)
        elif view_preset == 'top':
            view_control.set_front([0, -1, 0])
            view_control.set_up([0, 0, -1])
            view_control.set_lookat(bbox_center)
            view_control.set_zoom(0.7)
        elif view_preset == 'side':
            view_control.set_front([1, 0, 0])
            view_control.set_up([0, -1, 0])
            view_control.set_lookat(bbox_center)
            view_control.set_zoom(0.7)
        elif view_preset == 'isometric':
            view_control.set_front([1, 1, -1])
            view_control.set_up([0, -1, 0])
            view_control.set_lookat(bbox_center)
            view_control.set_zoom(0.7)
        
        # Update renderer
        self.pc_vis.poll_events()
        self.pc_vis.update_renderer()
        print(f"View changed to {view_preset}")
    def register_key_callbacks(self):
        """Register keyboard callbacks for controlling the view"""
        if self.pc_vis is None:
            return
            
        # Key mappings for different views
        self.pc_vis.register_key_callback(ord('1'), lambda vis: self.set_view_angle('front'))
        self.pc_vis.register_key_callback(ord('2'), lambda vis: self.set_view_angle('top'))
        self.pc_vis.register_key_callback(ord('3'), lambda vis: self.set_view_angle('side'))
        self.pc_vis.register_key_callback(ord('4'), lambda vis: self.set_view_angle('isometric'))
        self.pc_vis.register_key_callback(ord('C'), lambda vis: self.center_point_cloud())
        
        print("Key controls registered:")
        print("  1 - Front view")
        print("  2 - Top view")
        print("  3 - Side view")
        print("  4 - Isometric view")
        print("  C - Center view")
        print("  Mouse: Left-drag to rotate, Right-drag to pan, Scroll to zoom")
        
        
    def _visualize_point_cloud(self, point_cloud_np, sam_mask=None):
        """Visualize point cloud using Open3D"""
        # Create a new point cloud object
        new_pcd = o3d.geometry.PointCloud()
        
        # If point cloud is 3D array (HxWx4), flatten it properly
        if len(point_cloud_np.shape) == 3:
            # Process only valid points (with depth)
            mask = np.isfinite(point_cloud_np[..., 2])
            
            # If we have a segmentation mask, combine it with depth mask
            if sam_mask is not None:
                # Ensure same shape as depth mask
                resized_mask = cv2.resize(sam_mask.astype(np.float32), 
                                        (point_cloud_np.shape[1], point_cloud_np.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST) > 0.5
                mask = mask & resized_mask
            
            # Extract XYZ and RGBA using the mask
            xyz = point_cloud_np[..., :3][mask]
            rgba = point_cloud_np[..., 3][mask].view(np.uint32)
        else:
            # Already flattened point cloud
            mask = np.isfinite(point_cloud_np[:, 2])
            xyz = point_cloud_np[:, :3][mask]
            rgba = point_cloud_np[:, 3][mask].view(np.uint32)
        
        # Extract RGB from packed RGBA
        r = (rgba >> 0) & 0xFF
        g = (rgba >> 8) & 0xFF
        b = (rgba >> 16) & 0xFF
        rgb = np.vstack([r, g, b]).T / 255.0
        
        # Set data for new point cloud
        new_pcd.points = o3d.utility.Vector3dVector(xyz)
        new_pcd.colors = o3d.utility.Vector3dVector(rgb)
        
        # Store reference to original geometry
        if not hasattr(self, 'pcd_geometry'):
            self.pcd_geometry = new_pcd
        
        # # Initialize or update visualizer
        # if self.pc_vis is None:
        #     self.pc_vis = o3d.visualization.Visualizer()
        #     self.pc_vis.create_window("ZED Point Cloud", width=800, height=600)
        #     self.pc_vis.add_geometry(new_pcd)
        #     self.pcd_geometry = new_pcd  # Store reference
            
        #     # Add coordinate frame
        #     self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        #     self.pc_vis.add_geometry(self.frame)
        # In your _visualize_point_cloud method, modify the visualizer initialization:
        if self.pc_vis is None:
            self.pc_vis = o3d.visualization.Visualizer()
            self.pc_vis.create_window("ZED Point Cloud", width=800, height=600)
            self.pc_vis.add_geometry(new_pcd)
            self.pcd_geometry = new_pcd  # Store reference
            
            # # Add coordinate frame
            # self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            # self.pc_vis.add_geometry(self.frame)
            
            # Set initial view to isometric
            view_control = self.pc_vis.get_view_control()
            view_control.set_zoom(0.7)  # Adjust zoom level
            
            # Set rendering options for better visualization
            render_option = self.pc_vis.get_render_option()
            render_option.point_size = 1.0  # Larger points
            render_option.background_color = np.array([255, 255, 255])  # White background
            render_option.show_coordinate_frame = True  # Show coordinate frame
        else:
            # Copy data to existing point cloud
            self.pcd_geometry.points = new_pcd.points
            self.pcd_geometry.colors = new_pcd.colors
            
            # Tell Open3D to update the existing object
            self.pc_vis.update_geometry(self.pcd_geometry)
        
        # Update camera view
        self.pc_vis.poll_events()
        self.pc_vis.update_renderer()

    def start_streaming(self, show_image=True, show_depth=True, show_point_cloud=False, fps=30):
        """
        Start continuous streaming with visualization
        
        Parameters:
        -----------
        show_image : bool
            Whether to display RGB image
        show_depth : bool
            Whether to display depth map
        show_point_cloud : bool
            Whether to display point cloud
        fps : int
            Target visualization frame rate
        """
        if not self.is_open:
            if not self.open():
                return False
        
        if self.is_streaming:
            print("Already streaming")
            return True
            
        # Reset stop event
        self.stop_event.clear()
        self.is_streaming = True
        
        # Start streaming thread
        stream_thread = Thread(
            target=self._streaming_loop,
            args=(show_image, show_depth, show_point_cloud, fps),
            daemon=True
        )
        stream_thread.start()
        
        return True
    
    def _streaming_loop(self, show_image, show_depth, show_point_cloud, fps):
        """
        Streaming loop for continuous visualization
        
        Parameters:
        -----------
        show_image : bool
            Whether to display RGB image
        show_depth : bool
            Whether to display depth map
        show_point_cloud : bool
            Whether to display point cloud
        fps : int
            Target visualization frame rate
        """
        delay = 1.0 / fps
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            self.visualize_frame(
                image=show_image,
                depth=show_depth,
                point_cloud=show_point_cloud
            )
            
            # Maintain consistent frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def stop_streaming(self):
        """Stop the streaming process"""
        if self.is_streaming:
            self.stop_event.set()
            time.sleep(0.5)  # Wait for thread to exit
            self.is_streaming = False
    def set_up_segmentor(self, segmentor):
        
        """
        Set up the segmentor with the specified models
        
        Parameters:
        -----------
        dino_model : str
            Path to the GroundingDINO model
        sam_model : str
            Path to the Segment Anything Model (optional)
        """
        self.segmentor = segmentor
    def segment_image(self, image_np):
        """
        Segment the image using the segmentor
        
        Parameters:
        -----------
        image_np : np.array
            The input image as a NumPy array in RGB format
        boxes : List[float]
            List of bounding boxes for segmentation
        labels : List[str]
            List of labels for each bounding box
        scores : List[float]fsdffsdf
            List of confidence scores for each bounding box
        
        Returns:
        --------
        Tuple[np.array, np.array]
            The segmented image and the corresponding masks
        """
        # Preprocess the image for the segmentor\
        
        source_image, dino_ready_img = self.preprocess_image_for_segmentor(image_np[:, :, :3])
        self.segmentor.img_source = source_image
        self.segmentor.sam_predictor.set_image(source_image)
        
        boxes, confidences, labels = self.segmentor.gdino_predictor(
            model=self.segmentor.grounding_model,
            image= dino_ready_img,
            caption=self.segmentor.text_prompt,
            box_threshold=self.segmentor.box_threshold,
            text_threshold=self.segmentor.text_threshold,
        )
        sam_input_boxes = self.segmentor.sam_preprocess(boxes)
        try:
            sam_masks, scores, logits = self.segmentor.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=sam_input_boxes,
                multimask_output=False,
            )
            best_mask_indx = np.argmax(scores)
            best_sam_mask = sam_masks[best_mask_indx]
        except Exception as e:
            print(f"Error during SAM prediction: {e}")
            print("using last mask")
            best_sam_mask = self.segmented_mask.copy()
            
        self.segmented_mask = best_sam_mask.copy()
        
        class_names = labels
        # Convert class names to class IDs
        class_ids = np.array(list(range(len(class_names))))
        # Implement visualization steps here
        detections = sv.Detections(
                    xyxy=sam_input_boxes,  # (n, 4)
                    
                    # mask=masks.astype(bool),  # (n, h, w)
                    class_id=class_ids
                )
        annotated_frame = self.segmentor.postprocess(detections)
        return source_image, annotated_frame , best_sam_mask
    def preprocess_image_for_segmentor(self, image_np)-> Tuple[np.array, torch.Tensor]:
        """
        Preprocess the image for the segmentor

        Parameters:
        -----------
        image_np : np.array
            The input image as a NumPy array in RGB format

        Returns:
        --------
        Tuple[np.array, torch.Tensor]
            The preprocessed image and its corresponding tensor
        """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pil = Image.fromarray(image_np.astype('uint8'), 'RGB')
        # image = np.asarray(image_np)
        image_transformed, _ = transform(image_pil, None)
        return image_np, image_transformed
    
    def save_point_cloud(self, filename):
        """
        Save the current point cloud to a file
        
        Parameters:
        -----------
        filename : str
            Output filename (.ply format)
        """
        _, _, point_cloud_np = self.grab_frame()
        if point_cloud_np is None:
            return False
        
        # Process point cloud data
        mask = np.isfinite(point_cloud_np[..., 2])
        xyz = point_cloud_np[..., :3][mask]
        rgba = point_cloud_np[..., 3][mask].view(np.uint32)
        
        # Extract RGB
        r = (rgba >> 0) & 0xFF
        g = (rgba >> 8) & 0xFF
        b = (rgba >> 16) & 0xFF
        rgb = np.vstack([r, g, b]).T / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        
        # Save to file
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Point cloud saved to {filename}")
        return True

    def save_depth_image(self, filename):
        """
        Save the current depth map to a file
        
        Parameters:
        -----------
        filename : str
            Output filename (.png format)
        """
        _, depth_np, _ = self.grab_frame()
        if depth_np is None:
            return False
        
        # Save depth as 16-bit PNG
        cv2.imwrite(filename, depth_np.astype(np.uint16))
        print(f"Depth map saved to {filename}")
        return True
        
    def set_depth_mode(self, depth_mode):
        """
        Change the depth mode
        
        Parameters:
        -----------
        depth_mode : sl.DEPTH_MODE
            New depth mode
        """
        was_open = self.is_open
        was_streaming = self.is_streaming
        
        # Need to close and reopen camera to change depth mode
        if was_streaming:
            self.stop_streaming()
        if was_open:
            self.close()
        
        self.init_params.depth_mode = depth_mode
        
        # Restore previous state
        if was_open:
            self.open()
        if was_streaming:
            self.start_streaming()

if __name__ == "__main__":
    configs ={
        
        "sam2":{
            "checkpoint": "/home/admina/segmetation/DLOSeg/src/segment_anything_2_real_time/checkpoints/sam2.1_hiera_large.pt",
            "config": "configs/sam2.1/sam2.1_hiera_l.yaml"
        },
        "grounding_dino":{
            "config":"src/Grounded_SAM_2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "checkpoint":"src/Grounded_SAM_2/gdino_checkpoints/groundingdino_swint_ogc.pth",
            "box_threshold" : 0.35,
            "text_threshold" : 0.25,
            "text_prompt": "cable."

    }
    }
    
    # set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Example usage
    segmentor = Segmentor(models=["sam2_image"], device=device, configs=configs)
    time.sleep(0.5)
    # Example usage
    zed = ZedCamera(
        resolution=sl.RESOLUTION.HD720,
        depth_mode=sl.DEPTH_MODE.NEURAL_PLUS,
        fps=60
    )

    zed.set_up_segmentor(segmentor)
    try:
        # Open camera and start streaming
        zed.open()
        zed.get_camera_info()
        
        # Start streaming with all visualizations
        zed.start_streaming(
            show_image=True, 
            show_depth=False,
            show_point_cloud=True
        )
        
        # Let it run for a while
        print("Press Ctrl+C to stop streaming")
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Clean up
        zed.close()