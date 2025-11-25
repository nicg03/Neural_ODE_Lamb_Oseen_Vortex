from __future__ import annotations
import os, math, argparse, time, sys
from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    from torchdiffeq import odeint
except Exception:
    odeint = None
try:
    from config import TRAINING_CONFIG, MODEL_CONFIG, PATHS, DATASET_CONFIG, VORTEX_CONFIG
except Exception as e:
    TRAINING_CONFIG = {"learning_rate":1e-3}
    MODEL_CONFIG = {}
    PATHS = {"figures":"figs", "model":"data/ff_surrogate.pt", "dataset":"data/dataset.pt"}
    DATASET_CONFIG = {"domain_size":2.0, "grid_resolution":128, "time":1.0, "viscosity":0.01}
    VORTEX_CONFIG = {
        "vortex1":{"gamma":1.0, "x":-0.6, "y":0.0},
        "vortex2":{"gamma":-0.8, "x":0.6, "y":0.0},
        "vortex3":{"gamma":0.6, "x":0.0, "y":0.7},
    }

try:
    from dataset import TrajectoryDataset
except Exception as e:
    TrajectoryDataset = None

# ----------------------------
#  Lamb–Oseen utilities
# ----------------------------
def lamb_oseen_velocity(x, y, gamma, t, nu, x0=0.0, y0=0.0):
    x = x - x0
    y = y - y0
    r2 = x*x + y*y
    r2 = np.where(r2 == 0.0, 1e-18, r2)
    factor = gamma / (2.0 * np.pi * r2) * (1.0 - np.exp(-r2 / (4.0 * nu * t)))
    u_x = -y * factor
    u_y =  x * factor
    return u_x, u_y

def classic_multi_vortex(X, Y, vortices, t, nu):
    ux = np.zeros_like(X)
    uy = np.zeros_like(Y)
    for (gamma, x0, y0) in vortices:
        uxi, uyi = lamb_oseen_velocity(X, Y, gamma, t, nu, x0, y0)
        ux += uxi; uy += uyi
    return ux, uy

def deterministic_vortices_from_config(vcfg: Dict) -> List[Tuple[float,float,float]]:
    out = []
    for k in sorted(vcfg.keys()):
        d = vcfg[k]
        out.append((float(d["gamma"]), float(d["x"]), float(d["y"])))
    return out

class FFModel(nn.Module):
    """Surrogate (x,y,t)->(vx,vy) con MLP 3->2"""
    def __init__(self, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net(xyt)

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

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h,     y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

@torch.no_grad()
def rollout_with_ff(ff: FFModel, r0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Integra r' = ff(r,t) usando RK4 esplicito. r0: [B,2], t: [T]."""
    device = r0.device
    B = r0.shape[0]
    traj = torch.zeros(B, t.numel(), 2, device=device, dtype=r0.dtype)
    traj[:,0,:] = r0
    for i in range(t.numel()-1):
        ti = t[i].item(); h = (t[i+1]-t[i]).item()
        def f_scalar(ts, y):
            inp = torch.cat([y, torch.full((B,1), ts, device=device, dtype=r0.dtype)], dim=-1)
            return ff(inp)
        y = traj[:,i,:]
        y_next = rk4_step(f_scalar, ti, y, h)
        traj[:,i+1,:] = y_next
    return traj


def finite_differences_div_vort(ux: np.ndarray, uy: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    dx = float(grid[1]-grid[0])
    dy = dx
    dux_dx = np.gradient(ux, dx, axis=1)  # d(ux)/dx
    duy_dy = np.gradient(uy, dy, axis=0)  # d(uy)/dy
    dvort = np.gradient(uy, dx, axis=1) - np.gradient(ux, dy, axis=0)  # dvy/dx - dux/dy
    div = dux_dx + duy_dy
    return div, dvort

def circulation_on_circle(ux: np.ndarray, uy: np.ndarray, grid: np.ndarray, cx: float, cy: float, radius: float, npts: int = 512) -> float:
    Xg, Yg = np.meshgrid(grid, grid, indexing='xy')
    # Parametrizzazione del cerchio
    theta = np.linspace(0, 2*np.pi, npts, endpoint=False)
    xs = cx + radius*np.cos(theta)
    ys = cy + radius*np.sin(theta)
    # Indici per interpolazione
    def interp(F, x, y):
        ix = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid)-2)
        iy = np.clip(np.searchsorted(grid, y) - 1, 0, len(grid)-2)
        x1, x2 = grid[ix], grid[ix+1]
        y1, y2 = grid[iy], grid[iy+1]
        tx = np.where(x2!=x1, (x - x1)/(x2 - x1), 0.0)
        ty = np.where(y2!=y1, (y - y1)/(y2 - y1), 0.0)
        f11 = F[iy, ix]; f21 = F[iy, ix+1]
        f12 = F[iy+1, ix]; f22 = F[iy+1, ix+1]
        return (1-tx)*(1-ty)*f11 + tx*(1-ty)*f21 + (1-tx)*ty*f12 + tx*ty*f22

    uxs = interp(ux, xs, ys)
    uys = interp(uy, xs, ys)
    dx = -radius*np.sin(theta)*(2*np.pi/npts)
    dy =  radius*np.cos(theta)*(2*np.pi/npts)
    gamma = np.sum(uxs*dx + uys*dy)
    return float(gamma)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_heatmap_err(err: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(5.2,4.6))
    im = plt.imshow(err, origin="lower", extent=[-1,1,-1,1])
    plt.xlabel('x'); plt.ylabel('y'); # plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show(); plt.savefig(out_path, dpi=200); plt.close()

def plot_overlay_trajs(trajs: List[np.ndarray], labels: List[str], title: str, out_path: str):
    plt.figure(figsize=(6,5))
    for arr, lab in zip(trajs, labels):
        # arr: [B,T,2]; disegna qualche traiettoria
        sel = min(10, arr.shape[0])
        for i in range(sel):
            plt.plot(arr[i,:,0], arr[i,:,1], label=lab if i==0 else None)
    plt.xlabel('x'); plt.ylabel('y'); # plt.title(title)
    plt.legend()
    plt.tight_layout(); plt.show(); plt.savefig(out_path, dpi=200); plt.close()

def plot_box(values: List[np.ndarray], labels: List[str], title: str, out_path: str):
    plt.figure(figsize=(6,4.5))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel('EPE end-point'); # plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_curve(x: List[float], y: List[float], xlabel: str, ylabel: str, title: str, out_path: str):
    plt.figure(figsize=(6,4.5))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel); plt.ylabel(ylabel); # plt.title(title)
    plt.grid(True)
    plt.tight_layout(); plt.show(); plt.savefig(out_path, dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff_ckpt", type=str, default=PATHS.get("model","data/ff_surrogate.pt"), help="checkpoint FF (dict con 'model_state_dict')")
    ap.add_argument("--ode_ckpt", type=str, required=False, default="data/neural_ode_lamb_oseen.pt", help="state_dict Neural ODE")
    ap.add_argument("--out_dir", type=str, default=PATHS.get("figures","figs"))
    ap.add_argument("--grid_res", type=int, default=DATASET_CONFIG.get("grid_resolution", 128))
    ap.add_argument("--domain", type=float, default=DATASET_CONFIG.get("domain_size", 2.0))
    ap.add_argument("--time_snap", type=float, default=DATASET_CONFIG.get("time", 1.0))
    ap.add_argument("--viscosity", type=float, default=DATASET_CONFIG.get("viscosity", 0.01))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=51)
    ap.add_argument("--traj_eval", type=int, default=256)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    grid = np.linspace(-args.domain, args.domain, args.grid_res, dtype=np.float64)
    X, Y = np.meshgrid(grid, grid, indexing='xy')
    vortices = deterministic_vortices_from_config(VORTEX_CONFIG)

    ux_gt, uy_gt = classic_multi_vortex(X, Y, vortices, t=args.time_snap, nu=args.viscosity)

    ff = FFModel(hidden=128, depth=3).to(args.device)
    if os.path.exists(args.ff_ckpt):
        ckpt = torch.load(args.ff_ckpt, map_location=args.device)
        state = ckpt.get('model_state_dict', ckpt)
        ff.load_state_dict(state, strict=False)
        ff.eval()
        print("[OK] Caricato FF da", args.ff_ckpt)
    else:
        print("[WARN] FF checkpoint non trovato:", args.ff_ckpt)

    ode_func = ODEFunc(hidden=128).to(args.device)
    if args.ode_ckpt and os.path.exists(args.ode_ckpt):
        sd = torch.load(args.ode_ckpt, map_location=args.device)
        try:
            ode_func.load_state_dict(sd, strict=True)
            print("[OK] Caricato Neural ODE state_dict in ODEFunc")
        except Exception as e:
            new_sd = {}
            for k,v in sd.items():
                if k.startswith("func."):
                    new_sd[k[len("func."):]] = v
            ode_func.load_state_dict(new_sd, strict=False)
            print("[OK] Caricato (best-effort) Neural ODE in ODEFunc:", e)
    else:
        print("[WARN] ODE checkpoint non trovato:", args.ode_ckpt)

    with torch.no_grad():
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, args.time_snap, dtype=np.float32)], axis=1)
        pts_t = torch.from_numpy(pts.astype(np.float32)).to(args.device)
        v_ff = ff(pts_t).detach().cpu().numpy()
        ux_ff = v_ff[:,0].reshape(X.shape)
        uy_ff = v_ff[:,1].reshape(Y.shape)

        if ode_func is not None:
            r = torch.from_numpy(np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)).to(args.device)
            t_tensor = torch.tensor(args.time_snap, dtype=torch.float32, device=args.device)
            v_ode = ode_func(t_tensor, r).detach().cpu().numpy()
            ux_ode = v_ode[:,0].reshape(X.shape)
            uy_ode = v_ode[:,1].reshape(Y.shape)
        else:
            ux_ode = np.zeros_like(ux_ff); uy_ode = np.zeros_like(uy_ff)

    e_ff = np.sqrt((ux_ff-ux_gt)**2 + (uy_ff-uy_gt)**2)
    e_ode = np.sqrt((ux_ode-ux_gt)**2 + (uy_ode-uy_gt)**2)

    plot_heatmap_err(e_ff, "FF — errore sul campo (||Δv||)", os.path.join(args.out_dir, "F1_ff_field_error.pdf"))
    plot_heatmap_err(e_ode, "Neural ODE — errore sul campo (||Δv||)", os.path.join(args.out_dir, "F1_ode_field_error.pdf"))
    print("[F1] Salvate heatmap errori campo.")


    if TrajectoryDataset is not None:
        ds = TrajectoryDataset(n_traj=max(args.traj_eval, 512), steps=args.steps, seed=123)
        t_vec = ds.t.numpy()  # [T]
        T = t_vec.shape[0]
        traj_gt = ds.traj.numpy()  # [N,T,2]
        r0 = ds.traj[:,0,:].to(args.device)  # [N,2]
        t_torch = ds.t.to(args.device)

        with torch.no_grad():
            traj_ff = rollout_with_ff(ff, r0, t_torch).cpu().numpy()
            if odeint is not None:
                # Integra con ODE (odeint su ODEFunc)
                def _ode_roll(r0_batch: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                    traj = odeint(ode_func, r0_batch, t)  # [T,B,2]
                    return traj.transpose(0,1)
                traj_ode = _ode_roll(r0, t_torch).cpu().numpy()
            else:
                traj_ode = traj_ff.copy()  # fallback

        epe_ff = np.linalg.norm(traj_ff[:,-1,:] - traj_gt[:,-1,:], axis=1)
        epe_ode = np.linalg.norm(traj_ode[:,-1,:] - traj_gt[:,-1,:], axis=1)

        plot_overlay_trajs([traj_gt, traj_ff, traj_ode], ["GT", "FF", "ODE"], "Traiettorie (10 esempi)", os.path.join(args.out_dir, "F2_trajs_overlay.pdf"))
        plot_box([epe_ff, epe_ode], ["FF","ODE"], "End-Point Error (GT vs pred.)", os.path.join(args.out_dir, "F2_epe_box.pdf"))
        print(f"[F2] EPE FF (mean±std): {epe_ff.mean():.4e} ± {epe_ff.std():.4e} | ODE: {epe_ode.mean():.4e} ± {epe_ode.std():.4e}")
    else:
        print("[F2] TrajectoryDataset assente: salto confronto traiettorie.")


    if TrajectoryDataset is not None and odeint is not None:
        dts = [1e-3, 5e-3, 1e-2]
        epe_dt_ff: List[float] = []
        epe_dt_ode: List[float] = []
        ds = TrajectoryDataset(n_traj=128, steps=args.steps, seed=7)
        target = ds.traj.numpy()
        r0 = ds.traj[:,0,:].to(args.device)
        for dt in dts:
            T = int(max(2, (ds.t[-1]-ds.t[0]).item()/dt)) + 1
            t_custom = torch.linspace(ds.t[0].item(), ds.t[-1].item(), T, device=args.device)
            with torch.no_grad():
                traj_ff = rollout_with_ff(ff, r0, t_custom).cpu().numpy()
                traj_ode = odeint(ode_func, r0, t_custom).transpose(0,1).cpu().numpy()
            epe_ff = np.linalg.norm(traj_ff[:,-1,:] - target[:,-1,:], axis=1).mean()
            epe_ode = np.linalg.norm(traj_ode[:,-1,:] - target[:,-1,:], axis=1).mean()
            epe_dt_ff.append(epe_ff); epe_dt_ode.append(epe_ode)
        plot_curve(dts, epe_dt_ff, "Δt", "EPE end-point", "Step-invariance FF", os.path.join(args.out_dir, "F3_step_inv_ff.pdf"))
        plot_curve(dts, epe_dt_ode, "Δt", "EPE end-point", "Step-invariance ODE", os.path.join(args.out_dir, "F3_step_inv_ode.pdf"))
        print("[F3] Step-invariance valutata.")
    else:
        print("[F3] Skip step-invariance (richiede TrajectoryDataset e torchdiffeq).")

    div_ff, vort_ff = finite_differences_div_vort(ux_ff, uy_ff, grid)
    div_ode, vort_ode = finite_differences_div_vort(ux_ode, uy_ode, grid)
    plot_heatmap_err(np.abs(div_ff), "FF — |div v|", os.path.join(args.out_dir, "F4_ff_div.pdf"))
    plot_heatmap_err(np.abs(div_ode), "ODE — |div v|", os.path.join(args.out_dir, "F4_ode_div.pdf"))
    plot_heatmap_err(np.abs(vort_ff), "FF — |ω|", os.path.join(args.out_dir, "F4_ff_vorticity.pdf"))
    plot_heatmap_err(np.abs(vort_ode), "ODE — |ω|", os.path.join(args.out_dir, "F4_ode_vorticity.pdf"))
    print("[F4] Divergenza/Vorticità salvate.")


    loops_r = [0.25*args.domain, 0.5*args.domain, 0.8*args.domain]
    gam_gt = []; gam_ff = []; gam_ode = []
    for rr in loops_r:
        gam_gt.append(circulation_on_circle(ux_gt, uy_gt, grid, cx=0.0, cy=0.0, radius=rr))
        gam_ff.append(circulation_on_circle(ux_ff, uy_ff, grid, cx=0.0, cy=0.0, radius=rr))
        gam_ode.append(circulation_on_circle(ux_ode, uy_ode, grid, cx=0.0, cy=0.0, radius=rr))
    def pct_err(pred, ref):
        pred = np.array(pred); ref = np.array(ref)
        return list(100.0*np.abs(pred-ref)/(np.maximum(1e-12, np.abs(ref))))
    err_ff = pct_err(gam_ff, gam_gt)
    err_ode = pct_err(gam_ode, gam_gt)

    plt.figure(figsize=(6,4.5))
    plt.plot(loops_r, err_ff, marker='o', label='FF')
    plt.plot(loops_r, err_ode, marker='s', label='ODE')
    plt.xlabel('r loop'); plt.ylabel('Errore circolazione [%]');  # plt.title('Consistenza di circolazione')
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "F5_circulation_error.pdf"), dpi=200); plt.close()
    print("[F5] Circolazione su loop completata.")

    print(f"[DONE] Figure salvate in: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
