import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import random

# ─── Dataset ────────────────────────────────────────────────────────────────
class SplinePointDataset(Dataset):
    """
    Custom PyTorch Dataset for loading mask images and their corresponding B-spline control points.
    """
    def __init__(self, root, img_size=(256, 256), num_pts=200):
        """
        Args:
            root (str): Root directory of the dataset, containing 'images' and 'labels' subfolders.
            img_size (tuple): The target size to resize images to.
            num_pts (int): The number of points defining the spline.
        """
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
        """
        Retrieves a single data sample (image and spline points).
        """
        id_ = self.ids[idx].split(".")[0]
        mask = Image.open(os.path.join(self.img_dir, f"{id_}.png")).convert("L")
        mask = self.transform(mask)  # Shape: (1, H, W)
        lbl = np.load(os.path.join(self.lbl_dir, f"{id_}.npz"))
        spline = lbl["spline"]  # Shape: (num_pts, 3)
        return mask, torch.from_numpy(spline).float()

# ─── Model Definitions ──────────────────────────────────────────────────────
class SplineNet(nn.Module):
    """
    An Encoder-Decoder model that takes a binary mask image and generates a sequence
    of 2D control points for a B-spline using an RNN (LSTM) decoder.
    """
    def __init__(self, num_control_points: int, hidden_size: int = 512, embedding_size: int = 256):
        super(SplineNet, self).__init__()
        self.num_control_points = num_control_points
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        # Encoder: Pre-trained ResNet-34
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.encoder_feature_size = 512 
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Decoder: LSTM
        self.init_hidden = nn.Linear(self.encoder_feature_size, self.hidden_size)
        self.init_cell = nn.Linear(self.encoder_feature_size, self.hidden_size)
        self.coord_embedding = nn.Linear(2, self.embedding_size)
        self.lstm_cell = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, 2)
        self.sos_token = nn.Parameter(torch.randn(1, 2))

    def forward(self, img_mask: torch.Tensor, ground_truth_points: torch.Tensor = None, teacher_forcing_ratio: float = 0.5):
        batch_size = img_mask.size(0)
        features = self.encoder(img_mask)
        features = self.adaptive_pool(features).view(batch_size, -1)
        
        hidden = self.init_hidden(features)
        cell = self.init_cell(features)
        
        outputs = []
        current_point = self.sos_token.repeat(batch_size, 1)

        for i in range(self.num_control_points):
            embedded_point = self.coord_embedding(current_point)
            hidden, cell = self.lstm_cell(embedded_point, (hidden, cell))
            output_point = self.fc_out(hidden)
            outputs.append(output_point)
            
            use_teacher_forcing = (self.training and ground_truth_points is not None and random.random() < teacher_forcing_ratio)
            if use_teacher_forcing:
                current_point = ground_truth_points[:, i, :2]
            else:
                current_point = output_point.detach()

        return torch.stack(outputs, dim=1)

class Mask2Points(nn.Module):
    """
    An alternative model that uses a segmentation architecture (DeepLabV3) as an encoder
    and a simple fully connected layer as a decoder to predict all points at once.
    """
    def __init__(self, num_pts=200, out_dim=3, pretrained=True, img_size=(256, 256)):
        super().__init__()
        self.num_pts = num_pts
        self.out_dim = out_dim

        weights = (DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None)
        seg = deeplabv3_resnet50(weights=weights, progress=True)
        seg.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        self.encoder = seg

        # Temporarily set to eval mode to avoid BN errors on a single‐sample forward pass
        self.encoder.eval()
        with torch.no_grad():
            H, W = img_size
            dummy = torch.zeros(1, 3, H, W)
            out_map = self.encoder(dummy)['out']
        self.encoder.train()

        feat_dim = out_map.shape[1] * out_map.shape[2] * out_map.shape[3]
        self.fc = nn.Linear(feat_dim, num_pts * out_dim)

    def forward(self, mask: torch.Tensor):
        B = mask.size(0)
        x = mask.expand(-1, 3, -1, -1)
        out_map = self.encoder(x)['out']
        feat = out_map.view(B, -1)
        coords = self.fc(feat)
        pts = coords.view(B, self.num_pts, self.out_dim)
        return pts

# ─── Train & Test loops ─────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device, train=True, out_dim=3, teacher_forcing_ratio=0.1):
    mse = nn.MSELoss()
    total_loss = 0.0
    model.train() if train else model.eval()

    with torch.set_grad_enabled(train):
        for mask, spline_gt in loader:
            mask, spline_gt = mask.to(device), spline_gt.to(device)
            
            if train:
                optimizer.zero_grad()
                
            # --- MODIFICATION FOR TEACHER FORCING ---
            # Check if the model is SplineNet and if we are in training mode
            # If so, pass the ground truth points to enable teacher forcing.
            if isinstance(model, SplineNet) and train:
                pts_pred = model(mask, ground_truth_points=spline_gt, teacher_forcing_ratio=teacher_forcing_ratio)
            else:
                # For Mask2Points or for evaluation, call without ground_truth_points
                pts_pred = model(mask)
                
            loss = mse(pts_pred, spline_gt[:, :, :out_dim])
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * mask.size(0)

    return total_loss / len(loader.dataset)

# ─── Visualization ─────────────────────────────────────────────────────────
def visualize_examples(model, dataset, device, n_examples=4, out_dim=3):
    model.eval()
    fig = plt.figure(figsize=(12, 3 * n_examples))
    for i in range(n_examples):
        mask, spline_gt = dataset[i]
        with torch.no_grad():
            # For inference, we never pass ground truth points
            if isinstance(model, SplineNet):
                pred = model(mask.unsqueeze(0).to(device), ground_truth_points=None)[0].cpu().numpy()
            else:
                pred = model(mask.unsqueeze(0).to(device))[0].cpu().numpy()
        
        gt = spline_gt.numpy()
        
        if out_dim == 3:
            ax = fig.add_subplot(n_examples, 2, 2 * i + 1, projection='3d')
            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'g', label='GT')
            ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'r--', label='Pred')
            ax.set_title(f"3D Curve #{i}")
            ax.legend()

        ax2 = fig.add_subplot(n_examples, 2, 2 * i + 2)
        ax2.plot(gt[:, 0], gt[:, 1], 'g', label='GT XY')
        ax2.plot(pred[:, 0], pred[:, 1], 'r--', label='Pred XY')
        ax2.set_aspect('equal', 'box')
        ax2.set_title(f"2D Projection #{i}")
        ax2.legend()
        
    plt.tight_layout()
    plt.show()

# ─── Main Execution ─────────────────────────────────────────────────────────
def main():
    # Hyperparameters
    root = "/home/admina/segmetation/DLOSeg/dataset" # IMPORTANT: Update this path
    img_sz = (256, 256)
    num_pts = 200
    batch = 8
    epochs = 15
    lr = 1e-4
    out_dim = 2  # Set to 2 for 2D points (x, y) or 3 for 3D (x, y, z)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoaders
    full_ds = SplinePointDataset(root, img_size=img_sz, num_pts=num_pts)
    val_size = int(0.1 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)

    # Model, Optimizer
    # NOTE: This script is set to use Mask2Points. You can switch to SplineNet if desired.
    model = SplineNet(num_control_points=num_pts).to(device)
    # model = Mask2Points(num_pts=num_pts, pretrained=False, out_dim=out_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    # Training Loop
    print("Starting training...")
    for ep in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, opt, device, train=True, out_dim=out_dim)
        val_loss = run_epoch(model, val_loader, None, device, train=False, out_dim=out_dim)
        print(f"Epoch {ep:02d} • Train Loss={train_loss:.4f} • Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), "best_model.pth")
            # print(f"  -> Saved new best model with validation loss: {best_val_loss:.4f}")

    # Final Visualization
    print("\nTraining complete. Visualizing examples...")
    # To load the best model for visualization:
    # model.load_state_dict(torch.load("best_model.pth"))
    visualize_examples(model, full_ds, device, n_examples=4, out_dim=out_dim)
    print("Done!")

if __name__ == "__main__":
    # Check if the dataset path exists before running
    if not os.path.exists("/home/admina/segmetation/DLOSeg/dataset"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Dataset root path does not exist.           !!!")
        print("!!! Please update the 'root' variable in the main()    !!!")
        print("!!! function to point to your dataset directory.       !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main()
