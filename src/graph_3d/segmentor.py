import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything_2_real_time.sam2.build_sam import build_sam2_camera_predictor, build_sam2_video_predictor , build_sam2
from segment_anything_2_real_time.sam2.sam2_image_predictor import SAM2ImagePredictor
from Grounded_SAM_2.grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import time 
import supervision as sv
import cv2
from torchvision.ops import box_convert
import numpy as np

class Segmentor:
    def __init__(self, models, device='cpu', configs=None):
        self.models = models
        self.device = device
        self.configs = configs if configs else {}
        self.sam_predictor, self.gdino_predictor = self.load_model()
        
        self.text_prompt = self.configs["grounding_dino"].get("text_prompt", "cable.")
        self.box_threshold = self.configs["grounding_dino"].get("box_threshold", 0.35)
        self.text_threshold = self.configs["grounding_dino"].get("text_threshold", 0.25)
        
        self.img_source = None

    def load_model(self):
        
        
        # Load the SAM2 model
        
        sam_cfg = self.configs["sam2"]["config"]
        sam_checkpoint = self.configs["sam2"]["checkpoint"]

        if "sam2_camera" in self.models:
            sam_predictor = build_sam2_camera_predictor(sam_cfg, sam_checkpoint, device=self.device)
        elif "sam2_video" in self.models:
            sam_predictor = build_sam2_video_predictor(sam_cfg, sam_checkpoint, device=self.device)
        elif "sam2_image" in self.models:
            # Load the SAM2 model
            sam2_model = build_sam2(sam_cfg, sam_checkpoint, device=self.device)

            sam_predictor = SAM2ImagePredictor(sam2_model)
        # Load the Grounding DINO model
        gdino_cfg = self.configs["grounding_dino"]["config"]
        gdino_checkpoint = self.configs["grounding_dino"]["checkpoint"]

        self.grounding_model = load_model(
            model_config_path=gdino_cfg, 
            model_checkpoint_path=gdino_checkpoint,
            device=self.device
        )
        gdino_predictor = predict
        
        
        print("Loaded models successfully")
        return sam_predictor, gdino_predictor

    
    
    def load_image(self, image_path):

        self.img_source, self.dino_ready_img = load_image(image_path)
        # self.sam_predictor.set_image(self.img_source)
        
    def sam_preprocess(self,boxes):
        
        # process the box prompt for SAM 2
        if self.img_source is not None:
            h, w, _ = self.img_source.shape

        boxes = boxes * torch.Tensor([w, h, w, h])
        sam_input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        return sam_input_boxes
    
    def postprocess(self, box_output):
        # Implement postprocessing steps here
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=self.img_source.copy(), detections=box_output)
        return annotated_frame

    def visualize(self, boxes, labels):
        
        class_names = labels

        class_ids = np.array(list(range(len(class_names))))
        # Implement visualization steps here
        detections = sv.Detections(
                    xyxy=boxes,  # (n, 4)
                    
                    # mask=masks.astype(bool),  # (n, h, w)
                    class_id=class_ids
                )
        annotated_frame = self.postprocess(detections)
        cv2.imshow("Segmented Image", annotated_frame)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Clean up
        return annotated_frame



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
    
    IMAGE_PATH = "/home/admina/segmetation/DLOSeg/src/zed/record/recordings/png/left/left000300.png"
    # TEXT_PROMPT = "cable."
    # BOX_THRESHOLD = 0.35 # confidence threshold for box prediction
    # TEXT_THRESHOLD = 0.25 # confidence threshold for text prediction
    # Load the image
    start_gino = time.time()
    segmentor.load_image(IMAGE_PATH)
    # Set the image for the SAM2 predictor
    
    boxes, confidences, labels = segmentor.gdino_predictor(
        model=segmentor.grounding_model,
        image=segmentor.dino_ready_img,
        caption=segmentor.text_prompt,
        box_threshold=segmentor.box_threshold,
        text_threshold=segmentor.text_threshold,
    )
    print("Time taken to load image and predict:", time.time() - start_gino)
    sam_input_boxes = segmentor.sam_preprocess(boxes)
    segmentor.visualize(sam_input_boxes, labels)

    print(sam_input_boxes)