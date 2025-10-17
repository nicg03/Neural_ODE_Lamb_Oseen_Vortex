# neural_ode.py — true Neural ODE for 2D advection in a multi-vortex field.
from __future__ import annotations
import os, math
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torchdiffeq import odeint_adjoint as odeint  # memory-efficient
except Exception:
    from torchdiffeq import odeint                   # fallback

import numpy as np
import matplotlib.pyplot as plt
from dataset import TrajectoryDataset

class ODEFunc(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        t_feat = t.expand(B, 1)
        inp = torch.cat([state, t_feat], dim=-1)  # [B,3]
        return self.net(inp)

class NeuralODE(nn.Module):
    def __init__(self, hidden: int = 128, rtol: float = 1e-5, atol: float = 1e-6, method: str = "dopri5"):
        super().__init__()
        self.func = ODEFunc(hidden=hidden)
        self.rtol = rtol
        self.atol = atol
        self.method = method

    @torch.no_grad()
    def rollout(self, r0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        traj = odeint(self.func, r0, t, rtol=self.rtol, atol=self.atol, method=self.method)  # [T,B,2]
        return traj.transpose(0,1)  # [B,T,2]

    def forward(self, r0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        traj = odeint(self.func, r0, t, rtol=self.rtol, atol=self.atol, method=self.method)
        return traj.transpose(0,1)

def trajectory_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target)**2)

def plot_training_curves(train_losses: List[float], learning_rates: List[float], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Loss
    plt.figure(figsize=(8,4.5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); # plt.title('Curva di apprendimento')
    plt.yscale('log'); plt.grid(True); plt.legend()
    loss_path = os.path.join(out_dir, 'training_loss.pdf')
    plt.tight_layout(); plt.savefig(loss_path, dpi=150); plt.close()

    # LR
    plt.figure(figsize=(8,4.5))
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Schedule')
    plt.grid(True); plt.legend()
    lr_path = os.path.join(out_dir, 'learning_rate.pdf')
    plt.tight_layout(); plt.savefig(lr_path, dpi=150); plt.close()
    print(f"Saved plots:\n  - {loss_path}\n  - {lr_path}")

def train_neural_ode(
    epochs: int = 50,
    batch_size: int = 64,
    steps: int = 51,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    out_dir: str = "data"
) -> Tuple[NeuralODE, Dict[str, np.ndarray]]:
    ds = TrajectoryDataset(n_traj=8192, steps=steps, seed=0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = NeuralODE(hidden=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

    t = ds.t.to(device)  # [T]

    train_losses: List[float] = []
    learning_rates: List[float] = []

    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for r0, t_span, r_traj in dl:
            r0 = r0.to(device)                  # [B,2]
            r_traj = r_traj.to(device)          # [B,T,2]
            pred = model(r0, t)                 # [B,T,2]
            loss = trajectory_loss(pred, r_traj)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        avg = total / len(dl)
        sch.step(avg)

        # log
        train_losses.append(avg)
        learning_rates.append(opt.param_groups[0]['lr'])

        if ep % 5 == 0 or ep == 1:
            print(f"[{ep:03d}] train MSE = {avg:.6f} | lr={opt.param_groups[0]['lr']:.2e}")

    # salva metriche
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics_neural_ode.npz")
    np.savez(metrics_path, train_losses=np.array(train_losses), learning_rates=np.array(learning_rates))
    print(f"Saved metrics to {metrics_path}")

    return model, {"train_losses": np.array(train_losses), "learning_rates": np.array(learning_rates)}

@torch.no_grad()
def eval_endpoint_rmse(model: NeuralODE, n_eval: int = 512, steps: int = 51, device: str = "cpu") -> float:
    ds = TrajectoryDataset(n_traj=n_eval, steps=steps, seed=123)
    r0 = ds.traj[:,0,:].to(device)        # [N,2]
    t = ds.t.to(device)                   # [T]
    target = ds.traj.to(device)           # [N,T,2]
    pred = model.rollout(r0, t)           # [N,T,2]
    rmse = torch.sqrt(torch.mean((pred[:,-1,:] - target[:,-1,:])**2)).item()
    return rmse

if __name__ == "__main__":
    model, hist = train_neural_ode(epochs=100, batch_size=128, steps=51, lr=1e-5, out_dir="data")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rmse = eval_endpoint_rmse(model, n_eval=256, steps=51, device=device)

    # salva pesi (solo state_dict, come prima)
    ckpt_path = "data/neural_ode_lamb_oseen.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path} | endpoint RMSE = {rmse:.4e}")

    # genera i grafici dalle metriche appena salvate
    plot_training_curves(hist["train_losses"], hist["learning_rates"], out_dir="data")

    ########################

    metrics_path = "data/metrics_neural_ode.npz"
    out_dir = "data"

    m = np.load(metrics_path)
    train_losses = m["train_losses"]
    learning_rates = m["learning_rates"]

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8,4.5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); # plt.title('Curva di apprendimento')
    plt.yscale('log'); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'training_loss.pdf'), dpi=150); plt.close()

    plt.figure(figsize=(8,4.5))
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Schedule')
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'learning_rate.pdf'), dpi=150); plt.close()

    print("Saved plots to:", os.path.abspath(out_dir))
