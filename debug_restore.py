#!/usr/bin/env python3
"""
debug_restore.py — Compare _save_samples inference vs Restorer pipeline on
the same input frame. If both produce identical output, the Restorer code is
correct and any quality gap is purely a model/data issue.

Usage:
    python debug_restore.py \
        --data data/restoration_pairs_480x360 \
        --config configs/gpu/restoration.yaml \
        --out debug_compare.png
"""
import argparse, sys, math, random
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

def _to_np(t):
    return (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def run_save_samples_style(model, ab, deg_window_list, centre_idx, seed,
                            t_start, ddim_steps, device):
    """
    Exact copy of _save_samples DDIM loop from train_restoration.py.
    deg_window_list: list of T tensors each (3,H,W) in [0,1] on device
    Returns (3,H,W) in [0,1].
    """
    T = len(deg_window_list)
    half = T // 2
    # unsqueeze to (1,3,H,W) matching _save_samples
    cond_frames = [f.unsqueeze(0) for f in deg_window_list]
    centre_deg  = deg_window_list[centre_idx]  # (3,H,W) in [0,1]

    gen   = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(1, 3, *centre_deg.shape[-2:], device=device, generator=gen)
    ab_s  = ab[t_start]
    x     = ab_s.sqrt() * (centre_deg.unsqueeze(0) * 2 - 1) + (1 - ab_s).sqrt() * noise

    step_idx = torch.linspace(t_start, 0, ddim_steps + 1).long()
    for i in range(ddim_steps):
        t_cur  = int(step_idx[i].item())
        t_next = int(step_idx[i + 1].item())
        t_b    = torch.full((T,), t_cur, device=device, dtype=torch.long)
        inp_list = [torch.cat([x, (cond_frames[ti] * 2 - 1)], dim=1)
                    for ti in range(T)]
        model_in = torch.cat(inp_list, dim=0)   # (T, 6, H, W)
        with torch.amp.autocast("cuda"):
            eps   = model(model_in, t_b)
        eps_c = eps[half:half+1].float()
        ab_t  = ab[t_cur]; ab_p = ab[t_next]
        x0_p  = (x - (1 - ab_t).sqrt() * eps_c) / ab_t.sqrt()
        x0_p  = x0_p.clamp(-1, 2)
        x     = ab_p.sqrt() * x0_p + (1 - ab_p).sqrt() * eps_c

    return ((x.squeeze(0) + 1.0) * 0.5).clamp(0, 1)


def run_restorer_style(model, ab, deg_window_list, centre_idx, seed,
                       t_start, ddim_steps, device):
    """
    Exact copy of Restorer._ddim_loop logic (single window, B=1).
    deg_window_list: list of T tensors each (3,H,W) in [0,1] on device
    Returns (3,H,W) in [0,1].
    """
    T    = len(deg_window_list)
    half = T // 2
    windows_norm = [torch.stack(deg_window_list, dim=0) * 2.0 - 1.0]  # [(T,3,H,W)]
    centre_norm  = windows_norm[0][half]  # (3,H,W)

    gen   = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(3, *centre_norm.shape[-2:], device=device, generator=gen)
    ab_s  = ab[t_start]
    x     = (ab_s.sqrt() * centre_norm + (1 - ab_s).sqrt() * noise).unsqueeze(0)  # (1,3,H,W)

    step_idx = torch.linspace(t_start, 0, ddim_steps + 1).long()
    for si in range(ddim_steps):
        t_cur  = int(step_idx[si].item())
        t_next = int(step_idx[si + 1].item())
        rows   = []
        for ti in range(T):
            rows.append(torch.cat([x, windows_norm[0][ti:ti+1]], dim=1))
        model_in = torch.cat(rows, dim=0)   # (T, 6, H, W)
        t_b      = torch.full((T,), t_cur, device=device, dtype=torch.long)
        with torch.amp.autocast("cuda"):
            eps_all = model(model_in, t_b)
        eps_all = eps_all.float().view(1, T, 3, *x.shape[-2:])
        eps_c   = eps_all[0, half:half+1]   # (1, 3, H, W)
        ab_t    = ab[t_cur]; ab_p = ab[t_next]
        x0_p    = (x - (1 - ab_t).sqrt() * eps_c) / ab_t.sqrt()
        x0_p    = x0_p.clamp(-1.0, 2.0)
        x       = ab_p.sqrt() * x0_p + (1 - ab_p).sqrt() * eps_c

    return ((x[0] + 1.0) * 0.5).clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    required=True)
    p.add_argument("--config",  default="configs/gpu/restoration.yaml")
    p.add_argument("--out",     default="debug_compare.png")
    p.add_argument("--sample",  type=int, default=0, help="Dataset sample index")
    p.add_argument("--device",  default="cuda")
    args = p.parse_args()

    import yaml
    from omegaconf import OmegaConf
    cfg_dict = yaml.safe_load(open(args.config)) or {}
    cfg      = OmegaConf.create(cfg_dict)
    device   = torch.device(args.device)

    # Load model
    from src.restoration._network import RestoreUNet
    from src.restoration._diffusion import GaussianDiffusion

    ckpt = torch.load(cfg.model.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    mcfg  = ckpt.get("model_cfg", {}) if "model_cfg" in ckpt else {}
    if any(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}

    T_ckpt = mcfg.get("temporal_window", cfg.model.temporal_window)
    model  = RestoreUNet(
        base_channels    = mcfg.get("base_channels",    cfg.model.base_channels),
        encoder_channels = tuple(mcfg.get("encoder_channels", cfg.model.encoder_channels)),
        num_res_blocks   = mcfg.get("num_res_blocks",   cfg.model.num_res_blocks),
        T                = T_ckpt,
        n_heads          = mcfg.get("n_heads",          cfg.model.n_heads),
    )
    model.load_state_dict(state)
    model.to(device).eval()

    diff = GaussianDiffusion(T=1000)
    ab   = diff.alpha_bar.to(device)

    t_start   = cfg.inference.t_start
    ddim_steps = cfg.inference.ddim_steps
    T_win = T_ckpt
    half  = T_win // 2

    # Load a sample from the training dataset
    data_dir  = Path(args.data)
    orig_root = data_dir / "original"
    deg_root  = data_dir / "degraded"

    video_dirs = sorted(d for d in orig_root.iterdir() if d.is_dir())
    if not video_dirs:
        print("No videos found in dataset"); return

    vid_dir  = video_dirs[0]
    deg_dir  = deg_root / vid_dir.name
    names    = sorted(p.name for p in vid_dir.glob("*.png"))
    centre_i = half + args.sample * 5  # pick a frame
    centre_i = min(centre_i, len(names) - half - 1)

    print(f"Video: {vid_dir.name}  frame: {names[centre_i]}")

    deg_frames = []
    for offset in range(-half, half + 1):
        fi = min(max(centre_i + offset, 0), len(names) - 1)
        bgr = cv2.imread(str(deg_dir / names[fi]))
        deg_frames.append(_to_tensor(bgr).to(device))
    orig_bgr = cv2.imread(str(vid_dir / names[centre_i]))
    orig_t   = _to_tensor(orig_bgr).to(device)

    seed = centre_i

    with torch.no_grad():
        out_ss  = run_save_samples_style(model, ab, deg_frames, half, seed,
                                         t_start, ddim_steps, device)
        out_res = run_restorer_style(    model, ab, deg_frames, half, seed,
                                         t_start, ddim_steps, device)

    # Pixel difference between the two paths
    diff_t = (out_ss - out_res).abs()
    print(f"Max pixel diff (ss vs restorer): {diff_t.max().item():.6f}")
    print(f"Mean pixel diff:                 {diff_t.mean().item():.6f}")

    deg_np  = _to_np(deg_frames[half])
    ss_np   = _to_np(out_ss)
    res_np  = _to_np(out_res)
    orig_np = _to_np(orig_t)

    grid = np.concatenate([deg_np, ss_np, res_np, orig_np], axis=1)
    out_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, out_bgr)
    print(f"Saved: {args.out}")
    print("Columns: degraded | _save_samples | restorer | original")


if __name__ == "__main__":
    main()
