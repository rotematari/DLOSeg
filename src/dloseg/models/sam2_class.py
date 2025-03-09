import torch
import numpy as np
from sam2_realtime.sam2.build_sam import build_sam2
from sam2_realtime.sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
from matplotlib import pyplot as plt
import argparse

class SAM2Wrapper:
    """
    Wrapper class for SAM2 model that provides easy access to encoder, 
    Gaussian matrix encoder, and decoder components.
    """
    def __init__(self, checkpoint_path, model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml", device=None):
        """
        Initialize the SAM2 model and expose its key components.
        
        Args:
            checkpoint_path (str): Path to the SAM2 model checkpoint
            model_cfg (str): Path to the model configuration file
            device (torch.device): Device to run the model on (CPU/CUDA/MPS)
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        
        self.device = device
        self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
    @property
    def encoder(self):
        """Get the image encoder"""
        return self.sam2_model.image_encoder
    
    @property
    def prompt_encoder(self):
        """Get the prompt encoder"""
        return self.sam2_model.sam_prompt_encoder
    
    @property
    def decoder(self):
        """Get the mask decoder"""
        return self.sam2_model.mask_decoder
    
    @property
    def positional_encoding_gaussian_matrix(self):
        """Get the Gaussian positional encoding matrix"""
        return self.sam2_model.sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix
    
    def set_image(self, image):
        """Process an image and generate image embeddings"""
        return self.predictor.set_image(image)
    
    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True):
        """Run prediction using the image predictor"""
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output
        )
    
    def get_dense_pe(self):
        """Get the dense positional encoding"""
        return self.prompt_encoder.get_dense_pe()
    
    def encode_prompts(self, points=None, boxes=None, masks=None):
        """Encode prompts using the prompt encoder"""
        return self.prompt_encoder(points=points, boxes=boxes, masks=masks)
    
    def get_image_embeddings(self):
        """Get the current image embeddings from the predictor"""
        if self.predictor._features is None:
            raise ValueError("No image has been set. Call set_image() first.")
        return self.predictor._features["image_embed"]

    def decode_masks(self, image_embeddings, sparse_embeddings, dense_embeddings):
        """Use the decoder to generate masks from embeddings"""
        return self.decoder(
            image_embeddings=image_embeddings,
            image_pe=dense_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=True,
        )
        


def test_sam2_wrapper(image_path, checkpoint_path, model_cfg=None):
            """
            Test the SAM2Wrapper class with an example image
            
            Args:
                image_path (str): Path to test image
                checkpoint_path (str): Path to SAM2 model checkpoint
                model_cfg (str): Path to model configuration file
            """
            
            # Initialize the wrapper
            if model_cfg:
                wrapper = SAM2Wrapper(checkpoint_path, model_cfg)
            else:
                wrapper = SAM2Wrapper(checkpoint_path)
            
            # Load and process an image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Set the image
            wrapper.set_image(image)
            
            # Try prediction with a point prompt
            # Example: click at the center of the image
            h, w = image.shape[:2]
            point_coords = np.array([[w//2, h//2]])
            point_labels = np.array([1])  # 1 indicates foreground
            
            masks, scores, logits = wrapper.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
            # Display results
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.scatter(point_coords[:, 0], point_coords[:, 1], color='red', marker='*', s=200)
            plt.title("Input Image with Point Prompt")
            
            plt.subplot(1, 2, 2)
            plt.imshow(image)
            plt.imshow(masks[0], alpha=0.5)  # Show best mask
            plt.title(f"Segmentation Mask (score: {scores[0]:.3f})")
            plt.show()
            
            print(f"Generated {len(masks)} masks")
            print(f"Best mask score: {scores[0]:.3f}")
            
            return wrapper, masks, scores, logits


if __name__ == "__main__":
            
    parser = argparse.ArgumentParser(description="Test SAM2 Wrapper")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml", 
                                help="Path to model configuration")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    
    args = parser.parse_args()
            
    test_sam2_wrapper(args.image, args.checkpoint, args.config)