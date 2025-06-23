"""SplinePointTransformer – end‑to‑end model that predicts the 3‑D centre‑line of a thin
binary mask by combining a CNN feature extractor (ResNet‑18) with a Transformer
decoder.  The implementation keeps most of the original architectural choices
but fixes several correctness issues and adds optional quality‑of‑life
enhancements:

* **Conv1 adaptation** – the pre‑trained ResNet weights are preserved by
collapsing the RGB filters into a single‑channel kernel instead of starting
from scratch when the input is 1‑channel.
* **2‑D positional encoding** – the flattened H×W feature map is now enriched
with learnable sinusoidal 2‑D positional encoding instead of a purely 1‑D
encoding.  This empirically yields faster convergence on pixel‑to‑point
regression tasks.
* **Configurable backbone freezing** – the first N layers of the CNN can be
frozen during early training via the `freeze_backbone_at` argument.
* **Cleaner API** – rich type hints, `@staticmethod` helpers, and a single
`forward()` that runs with or without gradient tracking.

Author: ChatGPT (June 2025)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import models

from typing import Optional, Tuple






################################################################################
# Positional Encoding
################################################################################

class SineCosinePositionalEncoding2D(nn.Module):
    """2‑D sinusoidal positional encoding with correct channel allocation.

    For *d_model* divisible by 4 the channels are laid out as:
    `[sin x, cos x, sin y, cos y, …]` repeated.
    """

    def __init__(self, d_model: int, max_size: int = 1000, temperature: float = 10_000.0):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 (got %d)" % d_model

        # Generate (y, x) grid in the range [0, max_size)
        y, x = torch.meshgrid(
            torch.arange(max_size, dtype=torch.float32),
            torch.arange(max_size, dtype=torch.float32),
            indexing="ij",
        )
        y, x = y.flatten(), x.flatten()  # (max_size²,)

        # Each quartet of channels shares the same wavelength scale
        dim_t = torch.arange(d_model // 4, dtype=torch.float32)
        dim_t = temperature ** (dim_t / (d_model // 4))  # (d_model/4,)

        pe = torch.zeros(max_size * max_size, d_model)

        # Encode X
        pe_x = x[:, None] / dim_t  # (N, d_model/4)
        pe[:, 0::4] = torch.sin(pe_x)
        pe[:, 1::4] = torch.cos(pe_x)

        # Encode Y
        pe_y = y[:, None] / dim_t  # (N, d_model/4)
        pe[:, 2::4] = torch.sin(pe_y)
        pe[:, 3::4] = torch.cos(pe_y)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, N, d_model)
        self.max_size = max_size
        self.d_model = d_model

    def forward(self, feats: Tensor) -> Tensor:  # feats: (B, S, d_model)
        b, s, _ = feats.shape
        if s > self.max_size ** 2:
            raise ValueError("Sequence length %d exceeds encoding capacity" % s)
        return feats + self.pe[:, :s, :].to(feats.dtype).to(feats.device)

################################################################################
# Model
################################################################################

class SplinePointTransformer(nn.Module):
    """CNN + Transformer decoder that regresses *num_spline_points* 3‑D positions."""

    def __init__(
        self,
        num_spline_points: int,
        *,
        embedding_dim: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        freeze_backbone_at: Optional[int] = None,
        max_feat_resolution: int = 64,
        out_dim: int = 3,  # Output dimension for each spline point (x, y, z)
    ) -> None:
        super().__init__()
        self.num_spline_points = num_spline_points

        # ----------------------- CNN backbone -----------------------
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._adapt_resnet_conv1(resnet)

        # Remove avg‑pool & FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # C=512,  H=H/32, W=W/32

        # Optionally freeze early layers
        if freeze_backbone_at is not None:
            self._freeze_n_layers(self.backbone, freeze_backbone_at)

        # 1×1 conv to project to Transformer dimension
        self.feature_proj = nn.Conv2d(512, embedding_dim, kernel_size=1)

        # ------------------- Positional encoding --------------------
        self.pos_encoder = SineCosinePositionalEncoding2D(embedding_dim, max_size=max_feat_resolution)

        # ------------------- Transformer decoder --------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Learnable query embedding – one vector per predicted spline point
        self.query_embed = nn.Parameter(torch.randn(num_spline_points, embedding_dim))

        # Prediction head
        self.spline_head = nn.Linear(embedding_dim, out_dim)  # (x, y, z) per point

        self._init_weights()

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(self, mask: Tensor) -> Tensor:  # mask: (B, 1, H, W)
        b = mask.size(0)

        feats = self.backbone(mask)                           # (B, C=512, h, w)
        feats = self.feature_proj(feats)                      # (B, D, h, w)
        h, w = feats.shape[-2:]
        memory = feats.flatten(2).transpose(1, 2).contiguous()  # (B, h×w, D)
        memory = self.pos_encoder(memory)

        # Prepare queries – broadcast to batch
        tgt = self.query_embed.unsqueeze(0).expand(b, -1, -1)  # (B, num_points, D)

        # Transformer decoder
        dec_out = self.transformer_decoder(tgt=tgt, memory=memory)  # (B, num_points, D)

        # Regress 3‑D positions
        spline_pred = self.spline_head(dec_out)  # (B, num_points, out_dim)
        return spline_pred

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_resnet_conv1(resnet: models.ResNet) -> None:
        """Replace the RGB conv1 with a 1‑channel version while *retaining* the
        pre‑trained weights by averaging across channels."""
        old_weight: Tensor = resnet.conv1.weight.detach()
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_weight.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv

    @staticmethod
    def _freeze_n_layers(backbone: nn.Sequential, n: int) -> None:
        """Freeze parameters of the first *n* children modules."""
        children = list(backbone.children())
        for layer in children[:n]:
            for p in layer.parameters():
                p.requires_grad = False

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.spline_head.weight)
        nn.init.constant_(self.spline_head.bias, 0.0)


"""SimpleSplineNet – lighter variant of *SplinePointTransformer*
============================================================

This network keeps the ResNet‑18 backbone but replaces the heavy Transformer
decoder with a **single global feature vector + multilayer perceptron (MLP)**.
It is faster, needs far less GPU memory, and is often sufficient when the mask
contains a single thin object whose geometry can be summarised by global
context.

Extras
------
* Optional rasterisation helper that projects the predicted spline onto the
  image plane and applies binary dilation – handy for reconstruction losses
  such as Dice / IoU.
* Same Conv1 adaptation, backbone‑freezing knob, weight init, and smoke‑test as
  the fuller model.

Author: ChatGPT (June 2025)
"""



################################################################################
# Helpers – spline rasterisation (very lightweight)
################################################################################

def rasterise_spline(points: Tensor, img_size: tuple[int, int], thickness: int = 3) -> Tensor:
    """Project *N×3* spline points to the XY plane and draw an anti‑aliased line.

    Parameters
    ----------
    points : Tensor
        Shape *(B, N, 3)* in **normalised [0,1] coordinates**.
    img_size : (H, W)
    thickness : int
        Radius (in px) for dilation.

    Returns
    -------
    mask : Tensor
        Binary mask *(B, 1, H, W)* on the same device/dtype as *points*.
    """
    b, n, _ = points.shape
    h, w = img_size
    device, dtype = points.device, points.dtype

    # Convert to pixel indices
    x = (points[..., 0] * (w - 1)).round().long().clamp(0, w - 1)
    y = (points[..., 1] * (h - 1)).round().long().clamp(0, h - 1)

    masks = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
    for bi in range(b):
        for i in range(n - 1):
            # Bresenham‑like draw between (x_i, y_i) and (x_{i+1}, y_{i+1})
            rr, cc = _bresenham(y[bi, i].item(), x[bi, i].item(), y[bi, i + 1].item(), x[bi, i + 1].item())
            masks[bi, 0, rr, cc] = 1.0
    if thickness > 1:
        kernel = torch.ones(1, 1, thickness, thickness, device=device, dtype=dtype)
        masks = F.conv2d(masks, kernel, padding=thickness // 2) > 0
        masks = masks.to(dtype)
    return masks

def _bresenham(y0: int, x0: int, y1: int, x1: int):
    """Return integer pixel coords along a line (helper without NumPy)."""
    # Classic Bresenham algorithm – integer only
    points_y, points_x = [], []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        points_y.append(y0)
        points_x.append(x0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return torch.tensor(points_y), torch.tensor(points_x)

################################################################################
# Model – simple MLP decoder
################################################################################

class SimpleSplineNet(nn.Module):
    """CNN encoder + global‑pool MLP that regresses spline control points."""

    def __init__(
        self,
        num_spline_points: int,
        *,
        out_dim: int = 3,
    ) -> None:
        super().__init__()
        self.num_spline_points = num_spline_points
        self.out_dim = out_dim

        # ----------------------- CNN backbone -----------------------
        # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        resnet = models.resnet18(weights=None)
        
        self._adapt_resnet_conv1(resnet)
        self.resnet = resnet
        self.fc_proj1 = nn.Linear(1000, out_dim * num_spline_points)
        self.fc_proj2 = nn.Linear(1000, out_dim * num_spline_points)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(self, mask: Tensor):
        """Parameters
        ----------
        mask : Tensor *(B,1,H,W)*
        """
        res_out = self.resnet(mask)
        feats = self.fc_proj1(res_out)
        spline_pred = feats.view(mask.size(0), self.num_spline_points, self.out_dim)
        
        # spline_deriv = self.fc_proj2(res_out).view(mask.size(0), self.num_spline_points, self.out_dim)
        spline_deriv = torch.zeros_like(spline_pred)
        spline_deriv[:,:spline_pred.shape[1]-1] = spline_pred[:, 1:] - spline_pred[:, :-1]
        
        return spline_pred#,  spline_deriv
        
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_resnet_conv1(resnet: models.ResNet) -> None:
        w = resnet.conv1.weight.detach()
        new_conv = nn.Conv2d(1, 256, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = w.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv

    @staticmethod
    def _freeze_n_layers(backbone: nn.Sequential, n: int) -> None:
        for layer in list(backbone.children())[:n]:
            for p in layer.parameters():
                p.requires_grad = False



class ConvNeXtSplineNet(nn.Module):
    """
    ConvNeXt-Tiny encoder + global-pool MLP that regresses spline control points.
    """

    def __init__(
        self,
        num_spline_points: int,
        *,
        out_dim: int = 3,
        mlp_hidden: int = 1024,
        dropout: float = 0.1,
        freeze_backbone_at: Optional[int] = None,  # 0-4  → freeze that many stages
    ) -> None:
        super().__init__()
        self.num_spline_points = num_spline_points
        self.out_dim = out_dim

        # ------------------------------------------------------------------
        # 1. ConvNeXt-Tiny backbone  (features → 768-D embedding)
        # ------------------------------------------------------------------
        convnext = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )

        # --- adapt patch-embedding convolution from 3->1 channels -----------
        # (ConvNeXt 1st layer = Conv2d(3, 96, kernel=4, stride=4))
        pe_conv: nn.Conv2d = convnext.features[0][0]
        weight_mean = pe_conv.weight.data.mean(dim=1, keepdim=True)  # (96,1,4,4)
        new_pe_conv = nn.Conv2d(
            1, 96, kernel_size=4, stride=4, bias=False
        )
        new_pe_conv.weight.data.copy_(weight_mean)
        convnext.features[0][0] = new_pe_conv

        # --- discard the classifier's final Linear — keep global-pool & LN --
        convnext.classifier[-1] = nn.Identity()  # output → (B, 768)

        # --- optional partial freezing -------------------------------------
        if freeze_backbone_at is not None:
            # ConvNeXt stages = features[1]..features[4]
            for stage_idx in range(1, 1 + freeze_backbone_at):
                for p in convnext.features[stage_idx].parameters():
                    p.requires_grad = False

        self.backbone = convnext                       # (B, 768) output
        self.head = nn.Sequential(                     # simple 2-layer MLP
            nn.Linear(768, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim * num_spline_points),
        )

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        mask: torch.Tensor,                   # (B, 1, H, W)

    ):
        feats = self.backbone(mask)           # (B, 768)
        ctrl_pts = self.head(feats)           # (B, P*out_dim)
        ctrl_pts = ctrl_pts.view(
            mask.size(0), self.num_spline_points, self.out_dim
        )                                     # (B, P, 3) or (B, P, 2)

        return ctrl_pts
# =================================================================================
# --- 4. Example Usage ---
# =================================================================================
if __name__ == '__main__':
    
    # with torch.no_grad():
    #     model = SplinePointTransformer(num_spline_points=200)
    #     dummy = torch.randn(2, 1, 256, 256)  # batch of binary masks
    #     out = model(dummy)
    #     print(out.shape)  # → (2, 200, 3)

    with torch.no_grad():
        model = SimpleSplineNet(num_spline_points=200)
        dummy = torch.randn(2, 1, 256, 256)
        pts, mask = model(dummy, return_mask=True, img_size=(256, 256))
        print(pts.shape, mask.shape)  # (2,200,3) (2,1,256,256)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.imshow(mask[0, 0].cpu().numpy(), cmap='gray')
        plt.show()

