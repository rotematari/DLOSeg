#!/usr/bin/env python3
"""
Minimal **non‑learning** simulation & visualisation of a Deformable Linear
Object (DLO) using a Discrete‑Elastic‑Rod‑style model.

Updates (2025‑06‑04)
~~~~~~~~~~~~~~~~~~~~
* **Explicit inextensibility enforcement** – velocities are re‑projected so
  that the rod remains perfectly unstretchable both in position *and* in the
  velocity (time‑derivative) sense.  After constraint projection we now set
  `vᵢ ← (xᵢⁿ⁺¹ − xᵢⁿ)/Δt`, which is the classic PBD‑XPBD trick to keep
  constraints satisfied on the next step as well.
* Minor refactor of the integrator loop for clarity.

Run with:
    python simple_dlo_sim.py

Rotem – play with `N_SEGMENTS`, `BEND_STIFF`, and `DT` to see stability
limits; you should now be able to push the timestep a bit higher because the
stretch constraint is doubly enforced.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ──────────────────────────────────────────────────────────────────────────────
# Global simulation parameters ─ tweak freely
# ──────────────────────────────────────────────────────────────────────────────
N_SEGMENTS: int = 200          # number of edges  (N_VERTS = N_SEGMENTS+1)
L_REST:    float = 0.005       # rest length of each segment [m]
BEND_STIFF:float = 0.008       # bending stiffness coefficient [N·m]
GRAVITY          = np.array([0, 0, -9.81], dtype=np.float64)  # grav accel
DT:         float = 1e-2      # simulation timestep [s]
N_STEPS:    int   = 1_000     # total steps to simulate
N_ITERS:    int   = 50         # PBD iterations per step (constraint solver)
DAMPING:    float = 0.999     # numerical velocity damping factor
EPS:        float = 1e-12     # tiny value to avoid 0 ÷ 0
MAX_CORR:   float = 0.5 * L_REST  # clip per‑iteration correction magnitude

# ──────────────────────────────────────────────────────────────────────────────
# Initial geometry & state
# ──────────────────────────────────────────────────────────────────────────────
num_verts: int = N_SEGMENTS + 1
vertices  = np.zeros((num_verts, 3), dtype=np.float64)
vertices[:, 0] = np.linspace(0, N_SEGMENTS * L_REST, num_verts)

velocities = np.zeros_like(vertices)

# Choose which vertices are pinned (True = fixed in space)
clamped_mask = np.zeros(num_verts, dtype=bool)
clamped_mask[0]  = True   # pin start
clamped_mask[-1] = True   # uncomment to pin end as well

# Cache helpers
inv_masses = np.ones(num_verts, dtype=np.float64)  # inverse masses (1/kg)
inv_masses[clamped_mask] = 0.0

# Pre‑compute rest lengths (all equal here but kept general)
rest_lengths = np.full(N_SEGMENTS, L_REST, dtype=np.float64)

# Save initial pinned positions (used every step to hard‑clamp)
vertices_init = vertices.copy()

# ──────────────────────────────────────────────────────────────────────────────
# Constraint solvers
# ──────────────────────────────────────────────────────────────────────────────

def project_lengths(verts: np.ndarray) -> None:
    """Length constraints: make every segment exactly *rest_lengths[i]* long.

    Modified Position‑Based Dynamics projector with *N_ITERS* Gauss‑Seidel
    sweeps.  Each sweep applies the analytical correction for one distance
    constraint and distributes it according to inverse masses.
    """
    for _ in range(N_ITERS):
        for i in range(N_SEGMENTS):
            p, q = verts[i], verts[i + 1]
            d    = q - p
            dist = np.linalg.norm(d)
            if dist < EPS:
                continue  # collapsed segment: cannot resolve direction

            # Lagrange multiplier (scalar)
            denom = dist * (inv_masses[i] + inv_masses[i + 1] + EPS)
            C = (dist - rest_lengths[i]) / denom
            C = np.clip(C, -MAX_CORR, MAX_CORR)  # safety clip
            corr = C * (d / dist)

            # Apply symmetric correction
            verts[i]     +=  inv_masses[i]     * corr
            verts[i + 1] -=  inv_masses[i + 1] * corr

# ──────────────────────────────────────────────────────────────────────────────
# Forces
# ──────────────────────────────────────────────────────────────────────────────

def compute_forces(verts: np.ndarray) -> np.ndarray:
    """Return per‑vertex forces (N×3 array)."""
    F = np.zeros_like(verts)
    F += GRAVITY  # gravity on every vertex (mass assumed 1 kg)

    # Simple Laplacian bending: F_i += k (v_{i-1} - 2 v_i + v_{i+1}) / L^2
    k = BEND_STIFF / (L_REST ** 2)
    lap = verts[:-2] - 2.0 * verts[1:-1] + verts[2:]
    F[1:-1] += k * lap
    return F

# ──────────────────────────────────────────────────────────────────────────────
# Simulation step (XPBD‑style velocity update)
# ──────────────────────────────────────────────────────────────────────────────

def step() -> None:
    global vertices, velocities

    # Snapshot positions at start of step for velocity recomputation later
    verts_prev = vertices.copy()

    # External + bending forces ---------------------------------------------
    forces = compute_forces(vertices)

    # Semi‑implicit Euler (velocity first) -----------------------------------
    velocities += DT * forces * inv_masses[:, None]
    velocities *= DAMPING  # optional numerical damping

    # Predict positions ------------------------------------------------------
    vertices += DT * velocities

    # Constraint projection --------------------------------------------------
    project_lengths(vertices)            # inextensibility (positions)
    vertices[clamped_mask] = vertices_init[clamped_mask]  # hard clamp pins

    # **Velocity projection** – ensures constraints hold in velocity domain
    velocities = (vertices - verts_prev) / DT
    velocities[clamped_mask] = 0.0
    velocities *= DAMPING  # second damping pass keeps things tame

    # Sanity check -----------------------------------------------------------
    if not np.isfinite(vertices).all():
        raise FloatingPointError("Non‑finite values detected – simulation unstable.")

# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6, 4))
ax  = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 0.2, 0.5])
line, = ax.plot([], [], [], "o-", lw=2)
ax.set_xlim( -L_REST, N_SEGMENTS * L_REST + L_REST)
ax.set_ylim( -L_REST,  L_REST)
ax.set_zlim(-N_SEGMENTS * L_REST*50, L_REST)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Simple DER‑style DLO (inextensible)")


def init_anim():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,


def update_anim(frame):
    step()
    line.set_data(vertices[:, 0], vertices[:, 1])
    line.set_3d_properties(vertices[:, 2])
    return line,

ani = FuncAnimation(fig, update_anim, frames=N_STEPS, init_func=init_anim,
                    interval=10, blit=True, repeat=False)
plt.show()
