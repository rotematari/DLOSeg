"""Simple demonstration of the DEFORM simulation."""
import torch
import matplotlib.pyplot as plt
from der.dder.DEFORM_sim import DEFORM_sim
from der.dder.util import computeEdges

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate simulator with the default rest configuration
N_VERT = 13
N_EDGE = N_VERT - 1
sim = DEFORM_sim(N_VERT, N_EDGE, pbd_iter=5, device=DEVICE).to(DEVICE)

current_vert = sim.rest_vert
current_vel = torch.zeros_like(current_vert, device=DEVICE)
init_direction = torch.tensor([[[0.0, 0.0, 1.0]]], device=DEVICE)

clamped_index = torch.tensor([True] + [False]*(N_VERT-2) + [True], dtype=torch.bool, device=DEVICE)
clamped_selection = torch.tensor([0, 1, N_VERT-2, N_VERT-1], dtype=torch.long, device=DEVICE)
theta_full = torch.zeros(1, N_EDGE + 1, device=DEVICE)

m_u0 = sim.DEFORM_func.compute_u0(computeEdges(current_vert)[:,0], init_direction[:,0])

trajectories = [current_vert.squeeze().cpu()]
for _ in range(50):
    input_pos = current_vert[:, clamped_selection]
    current_vert, current_vel, theta_full = sim(
        current_vert, current_vel, init_direction, clamped_index,
        m_u0, input_pos, clamped_selection, theta_full, mode="evaluation"
    )
    trajectories.append(current_vert.squeeze().cpu())

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for verts in trajectories[::10]:
    ax.plot(verts[:,0], verts[:,1], verts[:,2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
