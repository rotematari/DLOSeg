import torch
import cv2
import numpy as np
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
# from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import matplotlib.pyplot as plt

# Load CLIPSeg Model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model.eval()

def get_clipseg_mask(image_path, text_prompt):
    """Generate a segmentation mask from CLIPSeg given an image and text prompt."""
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=text_prompt, images=[image] * len(text_prompt), return_tensors="pt")
    inputs =inputs.to(device)


    with torch.no_grad():
        """ Get the embeddings of the image and text.
            the meaning is that the image and text are passed through the text and vision models and after that the pooled output pass throu the projection
            layer to get the embeddings.
        """
        outputs = clipseg_model(**inputs)
        heatmap_embds 
    return heatmap_embds

# Load SAM Model
sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l.pth").to(device)
sam_predictor = SamPredictor(sam)


def segment_with_sam(image_path, clipseg_mask):
    """Use SAM to refine CLIPSeg-based segmentation."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image)

    # Convert CLIPSeg mask into SAM's point prompt format
    y, x = np.where(clipseg_mask > 0.5)  # Extract foreground points
    input_points = np.array([[xi, yi] for xi, yi in zip(x, y)])

    if len(input_points) == 0:
        return None  # No detected object

    with torch.no_grad():
        masks, _, _ = sam_predictor.predict(point_coords=input_points, point_labels=np.ones(len(input_points)))

    return masks[0]  # Return best segmentation mask


# Define Adapter Model
class AdapterNetwork(torch.nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.projection = torch.nn.Linear(64, embedding_dim)  # Map CLIPSeg to SAM space
        self.self_attention = torch.nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.filtering_mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, embedding_dim)
        )
        self.classifier = torch.nn.Linear(embedding_dim, 1)  # Binary classification

    def forward(self, clipseg_embedding):
        x = self.projection(clipseg_embedding)
        x, _ = self.self_attention(x, x, x)
        x = self.filtering_mlp(x)
        x = self.classifier(x)
        return torch.sigmoid(x)  # Probability of keeping the mask

# Instantiate Adapter
adapter = AdapterNetwork().to(device)
adapter.eval()


def full_pipeline(image_path, text_prompt):
    """Full pipeline for CLIPSeg + SAM + Adapter"""
    clipseg_mask = get_clipseg_mask(image_path, text_prompt)
    sam_mask = segment_with_sam(image_path, clipseg_mask)

    if sam_mask is None:
        print("No object detected.")
        return None

    sam_mask_tensor = torch.tensor(sam_mask).float().unsqueeze(0).to(device)
    classification_score = adapter(sam_mask_tensor)

    if classification_score.item() > 0.5:
        return sam_mask
    else:
        print("Filtered out low-confidence mask.")
        return None


if __name__ == "__main__":
    image_path = "test.jpg"
    text_prompt = ["acable"]

    segmentation_result = full_pipeline(image_path, text_prompt)
    
    if segmentation_result is not None:
        plt.imshow(segmentation_result, cmap="gray")
        plt.title("Final Segmentation Result")
        plt.show()
