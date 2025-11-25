from __future__ import annotations
import os, math
from typing import Tuple, Dict, List, Literal, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torchdiffeq import odeint_adjoint as odeint 
except Exception:
    from torchdiffeq import odeint                 

import numpy as np
import matplotlib.pyplot as plt
from dataset import TrajectoryDataset


class ODEFuncAug(nn.Module):
    def __init__(self, hidden: int, aug_dim: int, aug_dynamics: Literal["static","learned"] = "static"):
        super().__init__()
        in_dim = 2 + aug_dim + 1  # x,y + a + t
        self.aug_dim = aug_dim
        self.aug_dynamics = aug_dynamics

        self.net_xy = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )

        if aug_dynamics == "learned":
            self.net_a = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, aug_dim)
            )
        else:
            self.net_a = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        state: [B, 2+aug_dim]; returns [B, 2+aug_dim]
        """
        B = state.shape[0]
        t_feat = t.expand(B, 1)
        inp = torch.cat([state, t_feat], dim=-1)  # [B, 3+aug_dim]

        v_xy = self.net_xy(inp)                   # [B,2]
        if self.net_a is None:
            v_a = torch.zeros(B, self.aug_dim, device=state.device, dtype=state.dtype)
        else:
            v_a = self.net_a(inp)                 # [B,aug_dim]
        return torch.cat([v_xy, v_a], dim=-1)     # [B, 2+aug_dim]


class AugmentedNeuralODE(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        aug_dim: int = 2,
        aug_init: Literal["zeros","mlp"] = "zeros",
        aug_dynamics: Literal["static","learned"] = "static",
        rtol: float = 1e-5,
        atol: float = 1e-6,
        method: str = "dopri5"
    ):
        super().__init__()
        self.func = ODEFuncAug(hidden=hidden, aug_dim=aug_dim, aug_dynamics=aug_dynamics)
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.aug_dim = aug_dim
        self.aug_init = aug_init

        if aug_init == "mlp":
            self.enc = nn.Sequential(
                nn.Linear(2, hidden), nn.Tanh(),
                nn.Linear(hidden, aug_dim)
            )
            for m in self.enc:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                    nn.init.zeros_(m.bias)
        else:
            self.enc = None

    def _concat_aug(self, r0: torch.Tensor) -> torch.Tensor:
        B = r0.shape[0]
        if self.enc is None:
            a0 = torch.zeros(B, self.aug_dim, device=r0.device, dtype=r0.dtype)
        else:
            a0 = self.enc(r0)
        return torch.cat([r0, a0], dim=-1)  # [B, 2+aug_dim]

    @torch.no_grad()
    def rollout(self, r0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s0 = self._concat_aug(r0)                                 # [B,2+aug]
        traj = odeint(self.func, s0, t, rtol=self.rtol, atol=self.atol, method=self.method)  # [T,B,2+aug]
        traj = traj.transpose(0,1)                                # [B,T,2+aug]
        return traj[...,:2]                                       # [B,T,2]

    def forward(self, r0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s0 = self._concat_aug(r0)
        traj = odeint(self.func, s0, t, rtol=self.rtol, atol=self.atol, method=self.method)  # [T,B,2+aug]
        traj = traj.transpose(0,1)
        return traj[...,:2]


def trajectory_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target)**2)


def plot_training_curves(train_losses: List[float], learning_rates: List[float], out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    # Loss
    plt.figure(figsize=(8,4.5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.yscale('log'); plt.grid(True); plt.legend()
    loss_path = os.path.join(out_dir, f'training_loss_{tag}.pdf')
    plt.tight_layout(); plt.savefig(loss_path, dpi=150); plt.close()

    # LR
    plt.figure(figsize=(8,4.5))
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Schedule')
    plt.grid(True); plt.legend()
    lr_path = os.path.join(out_dir, f'learning_rate_{tag}.pdf')
    plt.tight_layout(); plt.savefig(lr_path, dpi=150); plt.close()
    print(f"Saved plots:\n  - {loss_path}\n  - {lr_path}")


def train_anode(
    epochs: int = 50,
    batch_size: int = 64,
    steps: int = 51,
    lr: float = 1e-3,
    hidden: int = 128,
    aug_dim: int = 2,
    aug_init: Literal["zeros","mlp"] = "zeros",
    aug_dynamics: Literal["static","learned"] = "static",
    lambda_a0: float = 0.0,  
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    out_dir: str = "data"
) -> Tuple[AugmentedNeuralODE, Dict[str, np.ndarray]]:
    ds = TrajectoryDataset(n_traj=8192, steps=steps, seed=0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = AugmentedNeuralODE(
        hidden=hidden, aug_dim=aug_dim, aug_init=aug_init, aug_dynamics=aug_dynamics
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

    t = ds.t.to(device)  # [T]

    train_losses: List[float] = []
    learning_rates: List[float] = []

    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for r0, _, r_traj in dl:
            r0 = r0.to(device)           # [B,2]
            r_traj = r_traj.to(device)   # [B,T,2]

            pred = model(r0, t)          # [B,T,2]
            loss = trajectory_loss(pred, r_traj)

            if (model.enc is not None) and (lambda_a0 > 0.0):
                a0 = model.enc(r0)       # [B,aug_dim]
                loss = loss + lambda_a0 * torch.mean(a0**2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        avg = total / len(dl)
        sch.step(avg)

        train_losses.append(avg)
        learning_rates.append(opt.param_groups[0]['lr'])

        if ep % 5 == 0 or ep == 1:
            print(f"[{ep:03d}] train MSE = {avg:.6f} | lr={opt.param_groups[0]['lr']:.2e}")

    # save metrics
    os.makedirs(out_dir, exist_ok=True)
    tag = f"anode_aug{aug_dim}_{aug_init}_{aug_dynamics}"
    metrics_path = os.path.join(out_dir, f"metrics_{tag}.npz")
    np.savez(metrics_path, train_losses=np.array(train_losses), learning_rates=np.array(learning_rates))
    print(f"Saved metrics to {metrics_path}")

    return model, {"train_losses": np.array(train_losses), "learning_rates": np.array(learning_rates), "tag": tag}


@torch.no_grad()
def eval_endpoint_rmse(model: AugmentedNeuralODE, n_eval: int = 512, steps: int = 51, device: str = "cpu") -> float:
    ds = TrajectoryDataset(n_traj=n_eval, steps=steps, seed=123)
    r0 = ds.traj[:,0,:].to(device)        # [N,2]
    t = ds.t.to(device)                   # [T]
    target = ds.traj.to(device)           # [N,T,2]
    pred = model.rollout(r0, t)           # [N,T,2]
    rmse = torch.sqrt(torch.mean((pred[:,-1,:] - target[:,-1,:])**2)).item()
    return rmse


if __name__ == "__main__":
    model, hist = train_anode(
        epochs=100, batch_size=128, steps=51, lr=1e-4,
        hidden=128, aug_dim=2, aug_init="mlp", aug_dynamics="learned",
        lambda_a0=1e-4, out_dir="data"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rmse = eval_endpoint_rmse(model, n_eval=256, steps=51, device=device)

    tag = hist.get("tag", "anode")
    ckpt_path = f"data/{tag}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path} | endpoint RMSE = {rmse:.4e}")

    plot_training_curves(hist["train_losses"], hist["learning_rates"], out_dir="data", tag=tag)

    metrics_path = f"data/metrics_{tag}.npz"
    out_dir = "data"
    m = np.load(metrics_path)
    train_losses = m["train_losses"]
    learning_rates = m["learning_rates"]

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,4.5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.yscale('log'); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'training_loss_{tag}.pdf'), dpi=150); plt.close()

    plt.figure(figsize=(8,4.5))
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Schedule')
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'learning_rate_{tag}.pdf'), dpi=150); plt.close()

    print("Saved plots to:", os.path.abspath(out_dir))
