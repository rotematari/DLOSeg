"""
Discrete Elastic Rod – Section 4.2 reference implementation
-----------------------------------------------------------
Bergou et al., “Discrete Elastic Rods”, TOG 2008, §4.2
"""

import numpy as np
from numpy.linalg import norm


def _safe_normalize(v, eps=1e-12):
    n = norm(v)
    return v / n if n > eps else v


def _rodrigues_rotate(v, axis, angle):
    """Rotate vector v about 'axis' by 'angle' (Rodrigues’ formula)."""
    axis = _safe_normalize(axis)
    return (v * np.cos(angle)
            + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1 - np.cos(angle)))


class DiscreteRod:
    """
    Minimal data-structure for the discrete Kirchhoff rod model (§4.2).

    Parameters
    ----------
    x : (n+2,3) array
        Vertex positions  x_0 … x_{n+1}.
    theta : (n+1,) array or None
        Material-frame angles θ_i (edge variables).  If None, zeros are assumed.
    alpha, beta : float
        Isotropic bending / twisting moduli (α, β).
    B : (n+1,2,2) array or None
        Optional anisotropic bending matrices B^j (one per edge).
        If None, isotropic B = α·Id is used.
    """

    def __init__(self, x, theta=None, alpha=1.0, beta=1.0, B=None):
        self.x = np.asarray(x, float)
        self.n = self.x.shape[0] - 2
        self.e = self.x[1:] - self.x[:-1]                      # edges e_i
        self.l = norm(self.e, axis=1)                         # |e_i|
        self.t = _safe_normalize(self.e)                      # tangents t_i
        # Voronoi lengths  ℓ_i = |e_{i-1}| + |e_i|
        self.li = self.l[:-1] + self.l[1:]

        self.theta = np.zeros(self.n + 1) if theta is None else np.asarray(theta, float)
        self.alpha, self.beta = float(alpha), float(beta)

        if B is None:
            self.B = np.array([alpha * np.eye(2)] * (self.n + 1))
        else:
            self.B = np.asarray(B, float)
        # --- geometric invariants ---------------------------
        self._compute_curvature_binormals()                   # κ_b^i
        self._build_bishop_frames()                           # u_i , v_i
        self._build_material_frames()                         # m1_i , m2_i

    # ------------------------------------------------------------------
    # 4.2.1  Discrete curvature & curvature binormal  (Eq. 1)
    # ------------------------------------------------------------------
    def _compute_curvature_binormals(self):
        e_prev, e_next = self.e[:-1], self.e[1:]
        numer = 2.0 * np.cross(e_prev, e_next)
        denom = (self.l[:-1] * self.l[1:] +
                 np.einsum('ij,ij->i', e_prev, e_next))
        self.kappa_b = numer / denom[:, None]                 # (κ_b)_i

    # ------------------------------------------------------------------
    # 4.2.2  Parallel transport & Bishop frame
    # ------------------------------------------------------------------
    def _build_bishop_frames(self):
        # choose u_0 ⟂ t_0
        z = np.array([0.0, 0.0, 1.0])
        u0 = _safe_normalize(np.cross(self.t[0], z)
                             if abs(np.dot(self.t[0], z)) < 0.99
                             else np.cross(self.t[0], [1, 0, 0]))
        self.u = np.empty_like(self.t)
        self.v = np.empty_like(self.t)
        self.u[0] = u0
        self.v[0] = np.cross(self.t[0], self.u[0])

        for i in range(1, self.n + 1):
            axis = np.cross(self.t[i - 1], self.t[i])
            sin_phi = norm(axis)
            if sin_phi < 1e-12:                 # t_{i-1} parallel t_i
                self.u[i] = self.u[i - 1]
            else:
                cos_phi = np.dot(self.t[i - 1], self.t[i])
                phi = np.arctan2(sin_phi, cos_phi)
                self.u[i] = _rodrigues_rotate(self.u[i - 1], axis, phi)
            self.v[i] = np.cross(self.t[i], self.u[i])

    # ------------------------------------------------------------------
    # Material frame  m₁, m₂  via θ_i
    # ------------------------------------------------------------------
    def _build_material_frames(self):
        c, s = np.cos(self.theta), np.sin(self.theta)
        self.m1 = c[:, None] * self.u + s[:, None] * self.v
        self.m2 = -s[:, None] * self.u + c[:, None] * self.v

    # ------------------------------------------------------------------
    # Bending energy  Eq. (3)  or isotropic simplification
    # ------------------------------------------------------------------
    def bending_energy(self):
        if np.allclose(self.B, self.alpha * np.eye(2)):        # isotropic
            en = self.alpha * (norm(self.kappa_b, axis=1) ** 2) * self.li
            return en.sum()                                    # ∑ α|κ_b|² ℓ_i  :contentReference[oaicite:0]{index=0}
        # --- anisotropic / curved -------------------------------------
        omega = np.stack([(self.kappa_b @ self.m2[:-1].T).diagonal(),
                          -(self.kappa_b @ self.m1[:-1].T).diagonal()], axis=1)
        omega_next = np.stack([(self.kappa_b @ self.m2[1:].T).diagonal(),
                               -(self.kappa_b @ self.m1[1:].T).diagonal()], axis=1)
        W = np.empty(self.n - 1)
        for i in range(1, self.n):
            for j in (i - 1, i):
                diff = (omega[i] if j == i else omega_next[i - 1])
                W[i - 1] = 0.5 / self.li[i - 1] * diff @ self.B[j] @ diff
        return W.sum()                                         # Eq. (3) :contentReference[oaicite:1]{index=1}

    # ------------------------------------------------------------------
    # Twisting energy  Eq. (4.2.2)
    # ------------------------------------------------------------------
    def twisting_energy(self):
        m_i = np.diff(self.theta)                              # m_i = θ_i − θ_{i-1}
        return (self.beta * (m_i ** 2) / self.li).sum()        # ∑ β m_i² / ℓ_i  :contentReference[oaicite:2]{index=2}

    def total_energy(self):
        return self.bending_energy() + self.twisting_energy()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3‑D projection)


def plot_rod(
    rod,
    ax=None,
    show_material_frames=False,
    frame_step=2,
    frame_len=0.05,
    **plot_kwargs,
):
    """
    Plot the centreline of a discrete elastic rod (DLO) in 3‑D.

    Parameters
    ----------
    rod : DiscreteRod instance or (N,3) array‑like
        The rod to plot.  If a DiscreteRod is given we use `rod.x`;
        otherwise the argument itself is interpreted as an (N,3) array
        of vertex positions.
    ax : matplotlib 3‑D axis, optional
        Existing axis to draw on.  If omitted, a new figure is created.
    show_material_frames : bool, default False
        If True and `rod` is a DiscreteRod, draw its material‑frame
        basis vectors (m1, m2) every `frame_step` vertices.
    frame_step : int, default 2
        Spacing (in vertices) between displayed frames.
    frame_len : float, default 0.05
        Length of each frame vector.
    **plot_kwargs :
        Extra keyword arguments forwarded to `ax.plot`, e.g. linewidth.

    Returns
    -------
    ax : matplotlib 3‑D axis
        The axis the rod was drawn on.
    """
    # ------------------------------------------------------------------
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        standalone = True
    else:
        standalone = False

    verts = np.asarray(rod.x) if hasattr(rod, "x") else np.asarray(rod)
    ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], **plot_kwargs)

    # Optionally plot material frames (m1, m2)
    if show_material_frames and hasattr(rod, "m1"):
        for i in range(0, len(rod.m1), frame_step):
            o = verts[i]
            m1 = rod.m1[i] * frame_len
            m2 = rod.m2[i] * frame_len
            ax.quiver(
                o[0],
                o[1],
                o[2],
                m1[0],
                m1[1],
                m1[2],
                linewidth=1,
            )
            ax.quiver(
                o[0],
                o[1],
                o[2],
                m2[0],
                m2[1],
                m2[2],
                linewidth=1,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])

    if standalone:
        plt.show()

    return ax


# ----------------------------------------------------------------------
# Demo with a gently‑sine‑wave rod:
t = np.linspace(0, 1, 40)
demo_vertices = np.column_stack([t, 0.1 * np.sin(2 * np.pi * t), np.zeros_like(t)])

plot_rod(demo_vertices, linewidth=2)


if __name__ == "__main__":
    
    # vertices of a gently bent rod (10 edges)
    t = np.linspace(0, 1, 12)
    x = np.column_stack([t, 0.1*np.sin(2*np.pi*t), np.zeros_like(t)])

    rod = DiscreteRod(x, alpha=0.8, beta=2.0)   # straight, isotropic
    print("E_bend  =", rod.bending_energy())
    print("E_twist =", rod.twisting_energy())    # zero because θ_i ≡ 0
    
    plot_rod(rod, linewidth=2, color='blue')