import torch
import torch.nn as nn

import torch
import torch.nn as nn

class BSplineFittingLayer(nn.Module):
    """
    Differentiable least-squares B-spline fitting layer.

    Implements control-point estimation by solving the normal equations:
      C = (N^T N + λI)^{-1} N^T P
    where N is the B-spline basis matrix evaluated at sample parameters u,
    P are target points, and λ is a Tikhonov regularization weight.

    References:
      - Piegl & Tiller, The NURBS Book (1997), §9.2 (least‐squares fitting).
      - P. Dierckx, Curve and Surface Fitting with Splines (1993) (smoothing splines).
    """
    def __init__(self, degree: int, num_control_points: int, num_samples: int, regularization_lambda: float = 0.0):
        super().__init__()
        self.K = degree
        self.M = num_control_points
        self.S = num_samples
        self.L = self.M + self.K + 1  # total knot count
        # **Store** the regularization weight
        self.regularization_lambda = regularization_lambda

        # parameter samples in [0,1]
        u = torch.linspace(0.0, 1.0, steps=self.S)
        self.register_buffer('u', u)

    def forward(self, knots: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Fit control points to target points via least squares.

        Args:
          knots: Tensor of shape (batch, L) or (L,) - knot vector
          points: Tensor of shape (batch, d, S) or (d, S) - target points P

        Returns:
          control_points: Tensor of shape (batch, d, M) or (d, M)
        """
        # Ensure batch dimensions
        if knots.dim() == 1:
            knots = knots.unsqueeze(0)  # (1, L)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # (1, d, S)

        batch, L = knots.shape
        _, d, S = points.shape

        # Build B-spline basis matrix N (batch, M, S)
        # We reuse the evaluator’s basis computation (Cox–de Boor)
        # Level-0: N0 shape (batch, L-1, S)
        num_funcs = L - 1
        u0 = self.u.view(1, 1, S).expand(batch, num_funcs, S)
        t_unsq = knots.unsqueeze(-1)  # (batch, L, 1)
        N0 = ((u0 >= t_unsq[:, :-1]) & (u0 < t_unsq[:, 1:])).float()
        basis = [N0]

        for k in range(1, self.K + 1):
            N_prev = basis[-1]
            num_funcs = L - k - 1
            u_k = self.u.view(1, 1, S).expand(batch, num_funcs, S)

            t_i = t_unsq[:, :num_funcs]
            t_ik = t_unsq[:, k:k+num_funcs]
            denom1 = (t_ik - t_i).clamp_min(1e-6)
            term1 = ((u_k - t_i) / denom1) * N_prev[:, :num_funcs, :]

            t_i1 = t_unsq[:, 1:1+num_funcs]
            t_ik1 = t_unsq[:, k+1:k+1+num_funcs]
            denom2 = (t_ik1 - t_i1).clamp_min(1e-6)
            term2 = ((t_ik1 - u_k) / denom2) * N_prev[:, 1:1+num_funcs, :]

            basis.append(term1 + term2)

        # Final basis N_final: (batch, M, S)
        N_final = basis[-1]

        # Reshape for normal equations
        # N   : (batch, S, M)
        # P   : (batch, S, d)
        N_mat = N_final.permute(0, 2, 1)
        P_mat = points.permute(0, 2, 1)

        # Compute (N^T N + λI) [M×M] and N^T P [M×d]
        NT_N = torch.bmm(N_mat.transpose(1, 2), N_mat)  # (batch, M, M)
        if self.regularization_lambda > 0:
            I = torch.eye(self.M, device=NT_N.device).unsqueeze(0)
            NT_N = NT_N + self.regularization_lambda * I

        NT_P = torch.bmm(N_mat.transpose(1, 2), P_mat)  # (batch, M, d)

        # Solve for control points: (batch, M, d)
        C_mat = torch.linalg.solve(NT_N, NT_P)

        # Return shape (batch, d, M)
        C = C_mat.permute(0, 2, 1)
        if C.size(0) == 1:
            return C.squeeze(0)  # (d, M)
        return C

# Example usage
if __name__ == "__main__":
    K = 3
    M = 8
    d = 3
    S = 100
    L = M + K + 1

    knots = torch.sort(torch.rand(L))[0]             # (L,)
    points = torch.rand(d, S)                        # (d, S)
    fitting_layer = BSplineFittingLayer(K, M, S, regularization_lambda=1e-3)
    control_pts = fitting_layer(knots, points)       # (d, M)
    print("Fitted control points shape:", control_pts.shape)

    
    points_np = points.detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    
    plt.plot(points_np[0], points_np[1], label='Spline Curve', color='blue')
    plt.show()