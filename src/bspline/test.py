import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet50,ResNet50_Weights ,resnet152 
from torchvision.models.segmentation import deeplabv3_resnet50 ,DeepLabV3_ResNet50_Weights
import torch
import torch.nn as nn
import torchvision.models as models
import random

# ─── Dataset (same as before) ────────────────────────────────────────────────
class SplinePointDataset(Dataset):
    def __init__(self, root, img_size=(256,256), num_pts=200):
        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.ids = sorted(os.listdir(self.img_dir))
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        self.num_pts = num_pts

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx].split(".")[0]
        mask = Image.open(os.path.join(self.img_dir, f"{id_}.png")).convert("L")
        mask = self.transform(mask)                  # (1,H,W)
        lbl = np.load(os.path.join(self.lbl_dir, f"{id_}.npz"))
        spline = lbl["spline"]                       # (num_pts,3)
        return mask, torch.from_numpy(spline).float()

# ─── Model (same as before) ─────────────────────────────────────────────────
class SplineNet(nn.Module):
    """
    An Encoder-Decoder model that takes a binary mask image and generates a sequence
    of 2D control points for a B-spline.

    The architecture consists of:
    1. Encoder: A pre-trained ResNet-34, modified for single-channel input,
       which acts as a feature extractor.
    2. Decoder: An LSTM network that takes the image features and autoregressively
       generates the control points one by one.
    """
    def __init__(self, num_control_points: int, hidden_size: int = 512, embedding_size: int = 256):
        """
        Initializes the SplineNet model.

        Args:
            num_control_points (int): The number of control points the model should output.
            hidden_size (int): The size of the LSTM's hidden state.
            embedding_size (int): The size of the embedding for the coordinate points
                                  fed into the LSTM.
        """
        super(SplineNet, self).__init__()

        self.num_control_points = num_control_points
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        # --- 1. ENCODER ---
        # Load a pre-trained ResNet-34 model
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Modify the first convolutional layer to accept single-channel (grayscale) images
        # instead of the standard 3-channel RGB images.
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # We use the ResNet model up to its second-to-last layer (before the final pooling and fc layers)
        # as our feature extractor.
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # The output of the ResNet-34's final convolutional block has 512 channels.
        self.encoder_feature_size = 512 

        # Add an adaptive average pooling layer. This ensures that the output feature map
        # has a fixed size (1x1) regardless of the input image size, which gives us a
        # fixed-length feature vector.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))


        # --- 2. DECODER ---
        # A linear layer to project the encoder's feature vector to the initial LSTM hidden state.
        self.init_hidden = nn.Linear(self.encoder_feature_size, self.hidden_size)
        
        # A linear layer to project the encoder's feature vector to the initial LSTM cell state.
        self.init_cell = nn.Linear(self.encoder_feature_size, self.hidden_size)

        # A linear layer to embed the 2D input coordinates into a higher-dimensional space.
        # This embedded vector will be the input to the LSTM cell at each timestep.
        self.coord_embedding = nn.Linear(2, self.embedding_size) # Input is an (x, y) point

        # The core of our decoder: an LSTM cell.
        self.lstm_cell = nn.LSTMCell(self.embedding_size, self.hidden_size)

        # A final fully connected layer to map the LSTM's hidden state to a 2D control point.
        self.fc_out = nn.Linear(self.hidden_size, 2) # Output is an (x, y) point
        
        # A learnable parameter to act as the "start-of-sequence" (SOS) token.
        # This will be the initial input to the LSTM decoder.
        self.sos_token = nn.Parameter(torch.randn(1, 2))


    def forward(self, img_mask: torch.Tensor, ground_truth_points: torch.Tensor = None, teacher_forcing_ratio: float = 0.5):
        """
        Defines the forward pass of the model.

        Args:
            img_mask (torch.Tensor): A batch of input mask images.
                                     Shape: (batch_size, 1, height, width)
            ground_truth_points (torch.Tensor, optional): The ground-truth control points.
                                     Used for teacher forcing during training.
                                     Shape: (batch_size, num_control_points, 2)
            teacher_forcing_ratio (float): The probability of using the ground-truth point
                                           as the next input instead of the model's own prediction.
                                           Defaults to 0.5.

        Returns:
            torch.Tensor: The sequence of generated control points.
                          Shape: (batch_size, num_control_points, 2)
        """
        batch_size = img_mask.size(0)

        # --- Encoding Step ---
        # Pass the image through the encoder to get features.
        # Shape: (batch_size, 512, H/32, W/32)
        features = self.encoder(img_mask)
        
        # Apply adaptive pooling to get a fixed-size feature vector.
        # Shape: (batch_size, 512, 1, 1)
        features = self.adaptive_pool(features)
        
        # Flatten the features for the decoder.
        # Shape: (batch_size, 512)
        features = features.view(batch_size, -1)

        # --- Decoding Step ---
        # Initialize the LSTM hidden and cell states from the image features.
        # Shape of hidden & cell: (batch_size, hidden_size)
        hidden = self.init_hidden(features)
        cell = self.init_cell(features)

        # Prepare a list to store the output points at each timestep.
        outputs = []

        # The initial input to the decoder is the start-of-sequence (SOS) token,
        # repeated for each item in the batch.
        # Shape: (batch_size, 2)
        current_point = self.sos_token.repeat(batch_size, 1)

        # Autoregressively generate the sequence of control points.
        for i in range(self.num_control_points):
            # Embed the current input point.
            # Shape: (batch_size, embedding_size)
            embedded_point = self.coord_embedding(current_point)

            # Pass the embedded point and the current states through the LSTM cell.
            # hidden and cell are updated in-place.
            hidden, cell = self.lstm_cell(embedded_point, (hidden, cell))

            # Generate the output point from the hidden state.
            # Shape: (batch_size, 2)
            output_point = self.fc_out(hidden)
            
            # Append the predicted point to our list of outputs.
            outputs.append(output_point)

            # --- Teacher Forcing ---
            # During training, we can randomly choose to feed the ground-truth point
            # as the input for the next timestep. This stabilizes training.
            use_teacher_forcing = (ground_truth_points is not None) and (random.random() < teacher_forcing_ratio)
            
            if use_teacher_forcing:
                current_point = ground_truth_points[:, i]
            else:
                # If not using teacher forcing, use the model's own prediction as the next input.
                current_point = output_point.detach() # Use .detach() to prevent gradients from flowing back through here

        # Stack the list of output points into a single tensor.
        # The list has `num_control_points` tensors of shape (batch_size, 2).
        # `torch.stack` with dim=1 will create a tensor of shape (batch_size, num_control_points, 2).
        outputs = torch.stack(outputs, dim=1)

        return outputs
class Mask2Points(nn.Module):
    def __init__(self, num_pts=200, out_dim=3, pretrained=True, img_size=(256, 256)):
        super().__init__()
        self.num_pts = num_pts
        self.out_dim  = out_dim

        # 1) Build DeepLabV3 and replace its final conv to give 1-channel output
        weights = (DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                   if pretrained else None)
        seg = deeplabv3_resnet50(weights=weights, progress=True)
        seg.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        self.encoder = seg

        # 2) Temporarily eval to avoid BN errors on a single‐sample forward
        self.encoder.eval()
        with torch.no_grad():
            H, W = img_size
            dummy = torch.zeros(1, 3, H, W)
            out_map = self.encoder(dummy)['out']  # e.g. (1,1,H',W')
        # restore training mode
        self.encoder.train()

        # 3) Compute flattened feature size
        B, C, Hp, Wp = out_map.shape
        feat_dim = C * Hp * Wp

        # 4) Regression head
        self.fc = nn.Linear(feat_dim, num_pts * out_dim)

    def forward(self, mask: torch.Tensor):
        """
        mask: (B,1,H,W) binary input
        → expand to 3 channels → run through DeepLabV3 → flatten → FC
        → (B, num_pts, out_dim)
        """
        B = mask.size(0)
        x = mask.expand(-1, 3, -1, -1)          # (B,3,H,W)
        out_map = self.encoder(x)['out']         # (B,1,H',W')
        feat = out_map.view(B, -1)               # (B, feat_dim)
        coords = self.fc(feat)                   # (B, num_pts*out_dim)
        pts = coords.view(B, self.num_pts, self.out_dim)
        # fit a spline to the points
        
        return pts , # tck
# class Mask2Points(nn.Module):
#     def __init__(self, num_pts=200,out_dim=3):
#         super().__init__()
#         self.out_dim = out_dim
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 256→128
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 3, stride=2, padding=1), # 128→64
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64→32
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),                   # → (B,64,1,1)
#         )
#         self.fc = nn.Linear(64, num_pts * out_dim)  # 64 features to 3*M points
#         self.num_pts = num_pts

#     def forward(self, mask):
#         B = mask.size(0)
#         f = self.encoder(mask).view(B, -1)            # (B,64)
#         out = self.fc(f)                              # (B, 3*M)
#         pts = out.view(B, self.num_pts, self.out_dim)            # (B,M,3)
#         return pts

# ─── Train & Test loops ─────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device, train=True,out_dim=3):
    mse = nn.MSELoss()
    total_loss = 0.0
    if train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        for mask, spline_gt in loader:
            mask, spline_gt = mask.to(device), spline_gt.to(device)
            
            if train:
                optimizer.zero_grad()
            pts_pred = model(mask)
            # print(f"pts_pred shape: {pts_pred.size()}, spline_gt shape: {spline_gt.size()}")
            loss = mse(pts_pred, spline_gt[:,:, :out_dim])  # Compare only the first out_dim dimensions
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * mask.size(0)

    return total_loss / len(loader.dataset)

# ─── Visualization ─────────────────────────────────────────────────────────
def visualize_examples(model, dataset, device, n_examples=4, out_dim=3):
    model.eval()
    fig = plt.figure(figsize=(12, 3*n_examples))
    for i in range(n_examples):
        mask, spline_gt = dataset[i]
        with torch.no_grad():
            pred = model(mask.unsqueeze(0).to(device))[0].cpu().numpy()
        gt = spline_gt.numpy()
        if out_dim == 3:
            ax = fig.add_subplot(n_examples, 2, 2*i+1, projection='3d')
            ax.plot(gt[:,0], gt[:,1], gt[:,2], 'g', label='GT')
            ax.plot(pred[:,0], pred[:,1], pred[:,2], 'r--', label='Pred')
            ax.set_title(f"3D Curve #{i}")
            ax.legend()

        ax2 = fig.add_subplot(n_examples, 2, 2*i+2)
        ax2.plot(gt[:,0], gt[:,1], 'g', label='GT XY')
        ax2.plot(pred[:,0], pred[:,1], 'r--', label='Pred XY')
        ax2.set_aspect('equal', 'box')
        ax2.set_title(f"2D Projection #{i}")
        ax2.legend()
    plt.tight_layout()
    plt.show()

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    # Hyperparams
    root    = "/home/admina/segmetation/DLOSeg/dataset"
    img_sz  = (256,256)
    num_pts = 200
    batch   = 8
    epochs  = 50
    lr      = 1e-4
    out_dim = 2  # Number of dimensions for each point (x, y, z)
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & splits
    full_ds = SplinePointDataset(root, img_size=img_sz, num_pts=num_pts)
    val_size = int(0.1 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [len(full_ds)-val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=2)

    # Model & optimizer
    model = Mask2Points(num_pts=num_pts,pretrained=False,out_dim=out_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    # Training loop
    for ep in range(1, epochs+1):
        train_loss = run_epoch(model, train_loader, opt, device, train=True, out_dim=out_dim)
        val_loss   = run_epoch(model, val_loader, None, device, train=False, out_dim=out_dim)
        print(f"Epoch {ep:02d} • Train L_shape={train_loss:.4f} • Val L_shape={val_loss:.4f}")
        # Save model checkpoint
        # best val loss
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save(model.state_dict(), "best_model.pth")

    # # Final visualization
    # print("Training complete. Visualizing examples...")
    # best_model = Mask2Points(num_pts=num_pts, pretrained=False, out_dim=out_dim).to(device)
    # best_model.load_state_dict(torch.load("best_model.pth"))
    visualize_examples(model, full_ds, device, n_examples=4,out_dim=out_dim)
    print("Done!")

if __name__ == "__main__":
    main()
