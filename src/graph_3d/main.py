from graph_3d.segmentor import Segmentor
from zed.zed_camera import ZedCamera
from pyzed.sl import sl
import time
import torch



if __name__ == "__main__":
    # Initialize the ZED camera
    zed = ZedCamera(
        resolution=sl.RESOLUTION.HD720,
        depth_mode=sl.DEPTH_MODE.ULTRA,
        fps=30
    )
    # dino 
    configs ={
        
        "sam2":{
            "checkpoint": "/home/admina/segmetation/DLOSeg/src/segment_anything_2_real_time/checkpoints/sam2.1_hiera_small.pt",
            "config": "configs/sam2.1/sam2.1_hiera_s.yaml"
        },
        "grounding_dino":{
            "config":"src/Grounded_SAM_2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "checkpoint":"src/Grounded_SAM_2/gdino_checkpoints/groundingdino_swint_ogc.pth"

    }
    }
    
    # set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Example usage
    segmentor = Segmentor(models=["sam2_camera"], device=device, configs=configs)
    
    IMAGE_PATH = "/home/admina/segmetation/DLOSeg/src/zed/record/recordings/png/left/left000300.png"
    TEXT_PROMPT = "cable."
    BOX_THRESHOLD = 0.35 # confidence threshold for box prediction
    TEXT_THRESHOLD = 0.25 # confidence threshold for text prediction

    segmentor = Segmentor(models=["sam2_camera"], device=device, configs=configs)

    try:
        # Open camera and start streaming
        zed.open()
        zed.get_camera_info()
        print("Camera opened successfully.")
        
        # Start streaming with all visualizations
        zed.start_streaming(
            show_image=True, 
            show_depth=False,
            show_point_cloud=False
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