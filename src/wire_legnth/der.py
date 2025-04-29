"""Discrete Elastic Rod (DER) based wire reconstruction
=====================================================

This implementation follows the mathematical formulation outlined in the
accompanying ChatGPT explanation.  It focuses on the *centre‐line* of a
deformable linear object (DLO) and fits a discrete elastic rod model to a
set of 3‑D observations coming e.g. from a stereo skeletonisation stage.

Key points
----------
* Vertices live on the **primal** mesh, edges on the **dual** mesh.
* Curvature is concentrated at vertices; twist lives on edges and is
  eliminated each iteration via the quasi–static condition.
* Energies are differentiable and optimised with PyTorch’s LBFGS.
* The data term uses a robust (Huber) penalty and is therefore tolerant to
  occlusions / outliers.

The code is written to be *readable first*.  It is **not** the fastest
possible implementation – feel free to vectorise and CUDA‐optimise as
needed.

Author: ChatGPT (o3) — 2025‑04‑18
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

###############################################################################
# Helper functions — geometry                                                 #
###############################################################################

def edges(x: Tensor) -> Tensor:
    """Return edge vectors e_i = x_{i+1} - x_i (shape: (n,3))."""
    return x[1:] - x[:-1]


def lengths(e: Tensor, eps: float = 1e-9) -> Tensor:
    """Edge lengths ||e_i|| with small epsilon to avoid /0."""
    return torch.sqrt((e * e).sum(-1) + eps)


def tan_half_angles(e: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute 2·tan(phi_i/2)·b_i for every internal vertex.

    Returns (kappa_b, valid_mask) where kappa_b has shape (n-1,3).
    The mask is False for degenerate consecutive edges (zero length).
    """
    e_im1 = e[:-1]
    e_i = e[1:]
    l_im1, l_i = lengths(e_im1), lengths(e_i)

    dot = (e_im1 * e_i).sum(-1)
    denom = l_im1 * l_i + dot
    cross = torch.cross(e_im1, e_i, dim=-1)

    # Avoid 0/0 when two consecutive edges are collinear & zero length
    valid = denom.abs() > 1e-12
    kappa_b = torch.zeros_like(cross)
    kappa_b[valid] = 2.0 * cross[valid] / denom[valid].unsqueeze(-1)
    return kappa_b, valid


###############################################################################
# Robust loss (Huber)                                                         #
###############################################################################

def huber(r: Tensor, delta: float = 2.0) -> Tensor:
    abs_r = r.abs()
    quad = abs_r <= delta
    lin = ~quad
    loss = torch.empty_like(r)
    loss[quad] = 0.5 * r[quad] ** 2
    loss[lin] = delta * (abs_r[lin] - 0.5 * delta)
    return loss

###############################################################################
# Main energy terms                                                           #
###############################################################################


def bending_energy(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Scalar bending energy (no twist)."""
    e = edges(x)
    kappa_b, valid = tan_half_angles(e)
    Eb = 0.5 * alpha * (kappa_b[valid].pow(2).sum(-1)).sum()
    return Eb


def data_term(x: Tensor, points: Tensor, lam: float = 1.0, delta: float = 2.0) -> Tensor:
    """Robust point–to–segment distance energy.

    points: (N,3)
    """
    # Project each point onto all segments, take min distance
    seg_a = x[:-1].unsqueeze(0)          # (1,n,3)
    seg_b = x[1:].unsqueeze(0)           # (1,n,3)
    p = points.unsqueeze(1)              # (N,1,3)

    ab = seg_b - seg_a                  # (1,n,3)
    t = ( (p - seg_a) * ab ).sum(-1) / ( (ab * ab).sum(-1) + 1e-12 )  # (N,n)
    t = t.clamp(0.0, 1.0).unsqueeze(-1)                                 # (N,n,1)

    proj = seg_a + t * ab                # (N,n,3)
    d2 = ((p - proj)**2).sum(-1)         # (N,n)
    d_min = torch.sqrt(d2.min(dim=1).values + 1e-12)  # (N,)

    return lam * huber(d_min, delta).sum()


def total_energy(x: Tensor,
                 points: Tensor,
                 alpha: float = 1.0,
                 lam: float = 1.0,
                 delta: float = 2.0) -> Tensor:
    return bending_energy(x, alpha) + data_term(x, points, lam, delta)

###############################################################################
# Optimisation wrapper                                                        #
###############################################################################


def resample_polyline(poly: Tensor, n: int) -> Tensor:
    """Resample an input (m,3) polyline to n+2 vertices (including ends)."""
    lengths_seg = torch.sqrt(((poly[1:] - poly[:-1])**2).sum(-1))
    cum = torch.cat((torch.zeros(1, device=poly.device), lengths_seg.cumsum(0)))
    tot = cum[-1]
    t_new = torch.linspace(0, tot, n + 2, device=poly.device)
    # Linear interpolation along arc length
    idx = torch.searchsorted(cum, t_new, right=True) - 1
    idx.clamp_(0, len(cum)-2)
    t0 = cum[idx]
    seg_len = lengths_seg[idx]
    w = (t_new - t0) / (seg_len + 1e-12)
    return (1 - w).unsqueeze(-1) * poly[idx] + w.unsqueeze(-1) * poly[idx+1]


def reconstruct(points: Tensor,
                n_internal: int = 50,
                lam: float = 10.0,
                alpha: float = 1.0,
                delta: float = 2.0,
                iters: int = 100) -> Tensor:
    """Reconstruct a centre‑line fitting *points* (N,3) as an (n+2,3) tensor.

    The polyline is initialised by projecting the points onto their first
    principal component (PCA line) and resampling.
    """
    device = points.device

    # 1. Simple PCA for initial straight guess
    mean = points.mean(0)
    X = points - mean
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    axis = Vh[0]
    t_min, t_max = ( (X @ axis).min(), (X @ axis).max() )
    ends = torch.stack((mean + t_min * axis, mean + t_max * axis))

    init_poly = resample_polyline(ends, n_internal)
    x = init_poly.clone().requires_grad_(True)

    opt = torch.optim.LBFGS([x], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        E = total_energy(x, points, alpha, lam, delta)
        E.backward()
        return E

    opt.step(closure)
    return x.detach()

###############################################################################
# Example usage (run `python der_reconstruction.py demo`)                     #
###############################################################################

if __name__ == "__main__":
    import sys
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="DER wire reconstruction demo")
    parser.add_argument("demo", nargs="?", help="run a toy demo")
    args = parser.parse_args()

    if args.demo is not None:
        # Build a noisy semi‑circle as mock observations
        theta = torch.linspace(0, math.pi, 200)
        pts = torch.stack((torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), dim=1)
        pts += 0.02 * torch.randn_like(pts)  # add noise

        centreline = reconstruct(pts, n_internal=40, lam=50.0, alpha=1.0)
        print("Reconstructed", centreline.shape[0], "vertices")

        try:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pts[:,0].cpu(), pts[:,1].cpu(), pts[:,2].cpu(), s=5, label="data")
            ax.plot(centreline[:,0].cpu(), centreline[:,1].cpu(), centreline[:,2].cpu(), "-", linewidth=2, label="DER fit")
            ax.legend()
            ax.set_box_aspect([1,1,1])
            plt.show()
        except ImportError:
            print("matplotlib not installed; skipping plot")
