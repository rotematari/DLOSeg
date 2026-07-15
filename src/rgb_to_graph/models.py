"""Model definitions for RGB -> DLO heatmap/mask prediction.

DloSegFormer: thin wrapper around a segmentation_models_pytorch Segformer
(mit_b5 encoder, ImageNet weights, 1 output class) that maps a 3-channel
RGB image to a single-channel spline heatmap/mask. Includes an unNormalize
helper for visualizing ImageNet-normalized inputs.

Running this file directly builds the model and benchmarks average inference
time on a random 256x256 input.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp


class DloSegFormer(nn.Module):
    def __init__(self,config:Dict = None):
        super().__init__()
        
        backbone = smp.Segformer(
            encoder_name="mit_b5", # best "mit_b5" fast and accurate ""
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

        self.backbone = backbone

    def forward(self, x):
        
        feats = self.backbone(x)        # returns feature map of shape (B, in_ch, H, W)
        
        return feats
    def unNormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Unnormalize a tensor image."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    
    
if __name__ == "__main__":
    # Example usage
    # model = EncoderHeatmapHead()
    config = {
        'num_pts': 100,  # Number of points per spline
        'img_size': (256, 256),  # Image size
        'num_classes': 1  # Number of classes (e.g., start and end points)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the model
    model = DloSegFormer(config=config).to(device)
    print(model)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)  # Batch size of 1, 3 channels, 256x256 image
    # Load one image

    from PIL import Image
    import time
    import torchvision.transforms as transforms

    # Load and preprocess grayscale image
    # image = Image.open(path).convert("L")  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # input_tensor = transform(input_tensor).unsqueeze(0)
    model.eval()  # Set the model to evaluation mode
    total_time = 0.0
    num_iterations = 100  # Number of iterations for averaging inference time
    with torch.no_grad():  # Disable gradient calculation
        for i in range(num_iterations):

            start_time = time.time()
            output = model(input_tensor)
            total_time += time.time() - start_time
    # output = model(input_tensor)
    print("Output shape:", output.shape)
    print(f"Inference time: {total_time/num_iterations:.4f} seconds")

    # # plot the heatmap
    # import matplotlib.pyplot as plt
    # # Convert output to numpy for plotting
    # heatmap = output[0].detach().numpy()  # Shape: (2, 256, 256)

    # # Create figure with subplots
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # # Plot first channel
    # im1 = axes[0].imshow(heatmap[0], cmap='hot', interpolation='nearest')
    # axes[0].set_title('Heatmap Channel 1')
    # axes[0].axis('off')
    # plt.colorbar(im1, ax=axes[0])

    # # Plot second channel
    # im2 = axes[1].imshow(heatmap[1], cmap='hot', interpolation='nearest')
    # axes[1].set_title('Heatmap Channel 2')
    # axes[1].axis('off')
    # plt.colorbar(im2, ax=axes[1])

    # plt.tight_layout()
    # plt.show()