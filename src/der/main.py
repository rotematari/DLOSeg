from der.utils import *
import matplotlib.pyplot as plt
import cv2
import numpy as np

import torch.nn.functional as F
import torch
"""
this code is an implementation for the 
Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects
from the paper: 
https://github.com/roahmlab/DEFORM
https://arxiv.org/abs/2406.05931

Algorithm 1 One Step Prediction with DEFORM
Inputs: ˆ Xt, ˆ Vt, ut 
1: θ∗ t(Xt) = argminθt P(Xt,θt,α) ▷ Section.4.1 
2: ˆ Vt+1 = ˆ Vt −∆tM−1∂P(Xt,θ∗ t(Xt,α),α) ∂Xt 
3: ˆ Xt+1 = ˆ Xt +∆t ˆ Vt+1 +DNN ▷ (3) ▷ Section.4.2 
4: Inextensibility Enforcement on ˆ Xt+1 ▷ Section.4.3 
5: ˆ Vt+1 = (ˆ Xt+1 − ˆ Xt)/∆t ▷ Velocity Update return ˆ Xt+1, ˆ Vt+1

"""
def get_bishop_frame():
    
    """
    Get the initial bishop frame for the first edge.
    Returns:
        u_0 (np.ndarray): The bishop frame vector in frame {t0,u0,v0} at edge 0.
    """
    pass

def get_rest_configuration():
    """
    Get the rest configuration of the centerline.
    Returns:
        x_hat (np.ndarray): The position of the centerline in the rest state.
    """
    pass

def get_initial_state():
    """
    Get the initial state of the centerline.
    Returns:
        x (np.ndarray): The initial position of the centerline.
        v (np.ndarray): The initial velocity of the centerline.
    """
    pass


def get_boundary_conditions():
    """
    Get the boundary conditions for the simulation.
    Returns:
        boundary_conditions (str): The type of boundary conditions (e.g., 'free', 'clamped', 'body-coupled').
    """
    pass
def compute_u0(e0, init_direct):
    """
    Initialize first edge's trigonometric bishop frame

    Parameters:
    e0: first edge
    init_direct: a randomly chosen 3D vector (not aligned with e0)

    return:
    m_u0: first edge's bishop frame's u axis that is perpendicular to e0
    """
    batch = e0.size()[0]
    N_0 = torch.cross(e0, init_direct.view(batch, -1))
    u0 = F.normalize(torch.cross(N_0, e0), dim=1)
    return u0
    
def computeEdges(vertices):
    # edge-link vectors
    # vertices dim = batch x n_vert x 3
    edges = vertices[:, 1:] - vertices[:, :-1]
    # assert torch.isfinite(edges.all()), "infinite problem: computeEdges"
    # assert not torch.isnan(edges).any(), "nan problem: computeEdges"
    # output dim = batch x (n_vert - 1) = n_edge x 3
    return edges

def computeLengths(edges):
    # input dim = batch x (n_vert - 1) = n_edge x 3
    # compute the length of each link: EdgeL
    # compute the sum of two adjacent links: RegionL, used in energy computation
    batch = edges.size()[0]
    EdgeL = torch.norm(edges, dim=2)
    RegionL = torch.zeros(batch, 1, device=edges.device)
    RegionL = torch.cat((RegionL, (EdgeL[:, 1:] + EdgeL[:, :-1])), dim=1)
    # output dim = batch x (n_vert - 1) = n_edge x 1 for botha
    return EdgeL, RegionL

def computeKB(edges, m_restEdgeL):
    """
    discrete curvature binormal: DER paper eq 1

    we use the rest edges lengths m_restEdgeL to inforce inextensibility ?
    """
    kb = torch.zeros_like(edges)
    kb[:, 1:] = torch.clamp(2 * torch.cross(edges[:, :-1], edges[:, 1:], dim=2)  
                                / (m_restEdgeL[:, :-1] * m_restEdgeL[:, 1:] \
                                + (edges[:, :-1] * edges[:, 1:]).sum(dim=2)).unsqueeze(dim=2), min=-20, max=20)
    
    return kb

def computeW(kb, m1, m2):
    o_wij = torch.cat(((kb * m2).sum(dim=2).unsqueeze(dim=2), -(kb * m1).sum(dim=2).unsqueeze(dim=2)), dim=2)
    return o_wij

def computeMaterialCurvature(self, kb, m1, m2):
    """
    Compute material curvature in material frame. DER paper eq 2

    Parameters:
    kb: discrete curvature binormal
    m1, m2: material frame

    returns: material curvature
    
    why do we  need 1 and 2 ?
    The bending energy of the rod needs to average or combine the curvatures as measured in the frames on both sides of the edge (see the discrete energy sum in the DER paper).

    This ensures continuity and correct measurement of bending, especially near places where material properties or frames might change.

    For general (not isotropic) rods, using both is essential for accuracy.
    
    The function builds m_W1 and m_W2:

    m_W1 is material curvature using the left endpoint’s material frame for each edge.

    m_W2 is material curvature using the right endpoint’s material frame for each edge.

    Note: the first element (index 0) is left zero because the first edge doesn’t have a left neighbor.
    """
    batch, n_edge = kb.size()[0], kb.size()[1]
    m_W1 = torch.zeros(batch, n_edge, 2).to(self.device)
    m_W2 = torch.zeros(batch, n_edge, 2).to(self.device)

    m_W1[:, 1:] = computeW(kb[:, 1:], m1[:, :-1], m2[:, :-1])
    m_W2[:, 1:] = computeW(kb[:, 1:], m1[:, 1:], m2[:, 1:])
    return m_W1, m_W2


def sqrt_safe(value):
    # safe square root for rotation angle calculation
    return torch.sqrt(torch.clamp(value, 1e-10))
def extractSinandCos(magnitude):
    """
    Extract sine and cosine values from the magnitude of kb.
    """
    # extract phi: the turning angle between two consecutive edges
    constant = 4.0
    o_sinPhi = sqrt_safe(magnitude/(constant+magnitude))
    o_cosPhi = sqrt_safe(constant/(constant+magnitude))
    return o_sinPhi, o_cosPhi
def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    q1 and q2 are tensors of shape (batch_size, 4).
    """
    w1 = q1[:, 0]
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]
    w2 = q2[:, 0]
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), -1)

def quaternion_p(theta, kb):
    """
    form quaternion coordinates for rotation
    """
    return torch.cat((theta, kb), dim=1)

def quaternion_conjugate(q):
    """
    Compute the conjugate of a quaternion.
    q is a tensor of shape (..., 4).
    """
    q_conj = q.clone()  # make a copy of the input tensor
    q_conj[..., 1:] *= -1  # negate the vector part of the quaternion
    return q_conj

def quaternion_rotation(o_u, edges, q, i):
    batch = o_u.size()[0]
    p = quaternion_p(torch.zeros(batch, 1).to(o_u.device), o_u[:, i - 1])
    quat_p = quaternion_multiply(quaternion_multiply(q[:, i], p), quaternion_conjugate(q[:, i]))
    u = F.normalize(quat_p[:, 1:4], dim=1)
    v = F.normalize(torch.cross(edges[:, i], u), dim=1)
    return u.unsqueeze(dim=1), v.unsqueeze(dim=1)
def quaternion_q(theta, kb):
    # form quaternion coordinates for rotation
    # output dim = batch x selected_edge x 3
    # w, x, y, z
    return torch.cat((theta.unsqueeze(dim=2), kb), dim=2)

def computeBishopFrame(self, u0, edges, restEdgeL):
    """
    compute the rest of bishop frame as reference frame based on the first bishop reference frame
    Bishop: twist-minimized reference frame. I am editing a document to clarify its derivation.

    Parameters:
    e0: first edge
    edges: edge vectors of wire segmentation
    restEdgeL: edge length of undeformed wire

    returns:
    b_u: Bishop frame u
    b_v: Bishop frame v
    kb: discrete curvature binormal
    """
    batch = edges.size()[0]
    # all the kbs
    kb = computeKB(edges, restEdgeL) # DER paper: discrete curvature binormal: DER paper eq 1
    
    b_u = u0.unsqueeze(dim=1)
    # direction for the first edge's bishop frame v
    b_v = F.normalize(torch.cross(edges[:, 0], b_u[:, 0], dim=1)).unsqueeze(dim=1)
    # ||kb||^2 = k_x^2 + k_y^2 + k_z^2
    magnitude = (kb * kb).sum(dim=2)
    
    sinPhi, cosPhi = extractSinandCos(magnitude)
    # q = cos(phi/2) + sin(phi/2) * (k_x, k_y, k_z)
    q = quaternion_q(cosPhi, 
                        sinPhi.unsqueeze(dim=2) * F.normalize(kb, dim=2) 
                    )
    
    for i in range(1, self.n_edge):
        
        b_u = torch.cat((b_u,
        torch.where(torch.ones(batch, 1).to(self.device) - cosPhi[:, i].unsqueeze(dim=1) <= self.err * torch.ones(batch, 1).to(self.device), 
        b_u[:, i - 1], 
        quaternion_rotation(b_u, edges, q, i)[0][:, 0, :]).unsqueeze(dim=1)), dim=1)
        b_v = torch.cat((b_v, torch.where(torch.ones(batch, 1).to(self.device) - cosPhi[:, i].unsqueeze(dim=1) <= self.err * torch.ones(batch, 1).to(self.device), b_v[:, i - 1], quaternion_rotation(b_u, edges, q, i)[1][:, 0, :]).unsqueeze(dim=1)), dim=1)
    return b_u, b_v, kb

def main():

    # Required inputs for the DER simulation


    rest_vertices = get_rest_configuration()  # Position of centerline in rest state

    x, v = get_initial_state()  # Initial state of the centerline

    boundary_conditions = get_boundary_conditions()  # free, clamped or body-coupled ends

    rest_edges = computeEdges(rest_vertices)  # Compute rest edges from the rest configuration

    rest_edges_l, rest_region_l = computeLengths(rest_edges)  # Compute rest edge lengths and region lengths
    
    init_direct = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # Initial direction for the first edge's bishop frame
    u_0 = compute_u0()  # Bishop frame vector in frame {t0,u0,v0} at edge 0
    
    

    
    
    # precompute rest material curvature
if __name__ == "__main__":
    main()