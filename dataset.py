
"""
dataset.py — ground-truth trajectories for multi‑vortex Lamb–Oseen flow

Generates training/eval data as particle trajectories r(t) advected by the
*analytic* Lamb–Oseen multi‑vortex velocity field. Use this to train a true
Neural ODE that learns the vector field f_θ(x, y, t) ≈ v(x, y, t).

Requirements: numpy, torch
(torch is only used to return tensors compatible with DataLoader)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

Array = np.ndarray

@dataclass
class Vortex:
    Gamma: float          # circulation
    x0: float             # center x
    y0: float             # center y

@dataclass
class FlowConfig:
    nu: float = 1e-2                 # kinematic viscosity
    vortices: Sequence[Vortex] = (
        Vortex( 1.0, -0.5,  0.0),
        Vortex(-0.8,  0.5,  0.5),
        Vortex( 0.6,  0.0, -0.5),
    )

def lamb_oseen_velocity(x: Array, y: Array, t: float, cfg: FlowConfig) -> Tuple[Array, Array]:
    """
    Analytic velocity field of a superposition of Lamb–Oseen vortices in 2D.

    v_i(x,t) = Γ_i / (2π r_i^2) * (1 - exp(-r_i^2/(4νt))) * e_z × (x - x_i)
    Returns (u_x, u_y) with broadcasting over x,y.
    t must be > 0 (for t=0 you can clamp to small epsilon).
    """
    nu = cfg.nu
    eps_t = max(t, 1e-6)
    ux = np.zeros_like(x, dtype=np.float64)
    uy = np.zeros_like(y, dtype=np.float64)
    for v in cfg.vortices:
        dx = x - v.x0
        dy = y - v.y0
        r2 = dx*dx + dy*dy + 1e-12  # avoid divide by zero
        factor = (1.0 - np.exp(-r2 / (4.0*nu*eps_t))) / (2.0*math.pi*r2)
        ux += -v.Gamma * dy * factor
        uy +=  v.Gamma * dx * factor
    return ux, uy

def rk4_step(pos: Array, t: float, dt: float, cfg: FlowConfig) -> Array:
    """
    One RK4 step for r'(t) = v(r,t). pos shape (..., 2)
    """
    x, y = pos[..., 0], pos[..., 1]

    def f(p, tt):
        u, v = lamb_oseen_velocity(p[...,0], p[...,1], tt, cfg)
        return np.stack([u, v], axis=-1)

    k1 = f(pos, t)
    k2 = f(pos + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(pos + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(pos + dt*k3,     t + dt)
    return pos + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate_trajectories(
    n_traj: int = 1024,
    t0: float = 0.05,
    t1: float = 1.05,
    steps: int = 51,
    domain: Tuple[float,float,float,float] = (-2.0, 2.0, -2.0, 2.0),
    seed: Optional[int] = 0,
    cfg: Optional[FlowConfig] = None,
) -> Tuple[Array, Array]:  # (trajectories [N,T,2], times [T])
    """
    Simulate N particle trajectories advected by the analytic field.
    Returns:
      r_traj: shape (n_traj, steps, 2)
      t_span: shape (steps,)
    """
    if cfg is None:
        cfg = FlowConfig()
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    x_min, x_max, y_min, y_max = domain
    r0 = np.stack([
        rng.uniform(x_min, x_max, size=n_traj),
        rng.uniform(y_min, y_max, size=n_traj)
    ], axis=-1).astype(np.float64)

    t_span = np.linspace(t0, t1, steps, dtype=np.float64)
    dt = (t1 - t0) / (steps - 1)

    traj = np.empty((n_traj, steps, 2), dtype=np.float64)
    traj[:, 0, :] = r0
    pos = r0.copy()
    t = t0
    for k in range(1, steps):
        pos = rk4_step(pos, t, dt, cfg)
        traj[:, k, :] = pos
        t += dt
    return traj, t_span

class TrajectoryDataset(Dataset):
    """
    Each item returns: (r0 [2], t_span [T], r_traj [T,2])
    """
    def __init__(self, n_traj: int = 4096, steps: int = 51, seed: int = 0, cfg: Optional[FlowConfig] = None):
        self.traj, self.t = simulate_trajectories(n_traj=n_traj, steps=steps, seed=seed, cfg=cfg)
        self.traj = torch.from_numpy(self.traj).float()
        self.t = torch.from_numpy(self.t).float()

    def __len__(self) -> int:
        return self.traj.shape[0]

    def __getitem__(self, idx: int):
        r_traj = self.traj[idx]         # [T,2]
        r0 = r_traj[0]                  # [2]
        return r0, self.t, r_traj

if __name__ == "__main__":
    # quick smoke test & save
    traj, t = simulate_trajectories(n_traj=8, steps=21, seed=42)
    np.savez("/mnt/data/lamb_oseen_trajs.npz", traj=traj, t=t)
    print("Saved example to /mnt/data/lamb_oseen_trajs.npz", traj.shape, t.shape)
