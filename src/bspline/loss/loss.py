import torch
import torch.nn.functional as F
import torch.nn as nn
from chamferdist import ChamferDistance

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        """
        Computes the Root Mean Square Error (RMSE) between predicted and target tensors.
        Args:
            pred (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Target tensor.
        Returns:
            torch.Tensor: RMSE loss value.
        """
        return torch.sqrt(F.mse_loss(pred, target))
# def chamfer_loss(pred_pts: torch.Tensor,
#                  gt_pts:   torch.Tensor,
#                  squared: bool = True) -> torch.Tensor:
#     """
#     pred_pts, gt_pts : (B, N, D) and (B, M, D)  — point clouds sampled from each spline
#     squared          : use L2^2 (default) or L2

#     Returns: scalar mean Chamfer distance over the batch
#     """
#     # Pair-wise distances — uses CUDA kernels in PyTorch ≥1.10
#     dists = torch.cdist(pred_pts, gt_pts, p=2)              # (B, N, M)
#     if squared:
#         dists = dists ** 2

#     # For every point take the closest point in the opposite cloud
#     pred2gt = dists.min(dim=2)[0]                           # (B, N)
#     gt2pred = dists.min(dim=1)[0]                           # (B, M)

#     loss = pred2gt.mean(dim=1) + gt2pred.mean(dim=1)        # (B,)
#     return loss.mean()


def total_arclength_loss(pred_pts: torch.Tensor,
                         gt_pts:   torch.Tensor) -> torch.Tensor:
    """
    Computes || L_pred  – L_gt ||^2
    pred_pts, gt_pts : (B, N, D)
    """
    def arc_len(x):
        seg = x[:, 1:] - x[:, :-1]          # (B, N-1, D)
        return seg.norm(dim=-1).sum(dim=-1) # (B,)

    loss = (arc_len(pred_pts) - arc_len(gt_pts)) ** 2
    return loss.mean()


# ---------- resample ground-truth spline densely ----------
def resample(curve, T=1000):
    """curve: (B, N, 2)  →  dense (B, T, 2) via linear interp"""
    B, N, _ = curve.shape
    t_src = torch.linspace(0, 1, N, device=curve.device)
    t_tgt = torch.linspace(0, 1, T, device=curve.device)
    idx = torch.searchsorted(t_src, t_tgt, right=True) - 1
    idx = idx.clamp(0, N-2)

    c0, c1 = curve[:, idx], curve[:, idx+1]           # (B, T, 2)
    w = (t_tgt - t_src[idx]) / (t_src[idx+1] - t_src[idx] + 1e-10)
    return c0*(1-w.unsqueeze(-1)) + c1*w.unsqueeze(-1)

class ChamferLoss(nn.Module):
    def __init__(self, npoints=None):
        super().__init__() 
        self.cd = ChamferDistance()
        # self.npoints = npoints

    def earth_mover_distance(self, y_true, y_pred):
        return torch.mean(torch.square(
            torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1).mean()
    
    def forward(self, y_true, y_pred):
        # if y_true.ndim != 3 and self.npoints is not None:
        #     self.batch = y_true.shape[0] // self.npoints
        #     y_true = y_true.view(self.batch, self.npoints, 2)
        
        # if y_pred.ndim != 3 and self.npoints is not None:
        #     self.batch = y_true.shape[0] // self.npoints
        #     y_pred = y_pred.view(self.batch, self.npoints, 2)
    
        return  self.cd(y_true, y_pred, bidirectional=True) #+ self.earth_mover_distance(y_true, y_pred)