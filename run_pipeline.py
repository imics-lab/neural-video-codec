#!/usr/bin/env python3
"""
run_pipeline.py — Full end-to-end pipeline entry point.

Chains: compress → decompress → restore → upscale

Usage:
    python run_pipeline.py --video INPUT.mp4
                           [--config configs/gpu/pipeline.yaml]
                           [--output OUTPUT.mp4]
                           [--downscale 0.5]   # eval mode: compare with original
                           [--skip-restore]
                           [--skip-upscale]
                           [--save-intermediate]
                           [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
import time
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("codec.pipeline")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)
    import os; os.environ["YOLO_VERBOSE"] = "true" if verbose else "false"


def _status(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def _load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if yaml is not None:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    with open(p) as f:
        import json
        return json.load(f)


def _merge_sub_config(pipeline_cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Load a sub-stage config, merging pipeline overrides on top."""
    sub_cfg_path = (pipeline_cfg.get(key, {}) or {}).get("config", "")
    if sub_cfg_path and Path(sub_cfg_path).exists():
        base = _load_config(sub_cfg_path)
    else:
        base = {}
    overrides = pipeline_cfg.get(key, {}) or {}
    return {**base, **{k: v for k, v in overrides.items() if k != "config"}}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full pipeline: compress → decompress → restore → upscale."
    )
    p.add_argument("--video",           required=True)
    p.add_argument("--config",          default="configs/gpu/pipeline.yaml")
    p.add_argument("--output",          default=None)
    p.add_argument("--downscale",       type=float, default=None,
                   help="Downscale input by this factor (eval mode, e.g. 0.5)")
    p.add_argument("--skip-restore",    action="store_true")
    p.add_argument("--skip-upscale",    action="store_true")
    p.add_argument("--save-intermediate", action="store_true")
    p.add_argument("--verbose",         action="store_true")
    return p.parse_args()


def main() -> int:
    args   = _parse_args()
    _setup_logging(args.verbose)

    pipeline_cfg = _load_config(args.config)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        return 1

    downscale = args.downscale or float(
        (pipeline_cfg.get("input", {}) or {}).get("downscale_input", 1.0)
    )
    save_intermediate = args.save_intermediate or bool(
        (pipeline_cfg.get("output", {}) or {}).get("save_intermediate", False)
    )
    out_dir = Path(
        (pipeline_cfg.get("output", {}) or {}).get("out_dir", "outputs/pipeline")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    if args.output:
        final_out = Path(args.output)
    else:
        final_out = out_dir / f"{stem}_pipeline.mp4"

    # ── Step 1: Compress ──────────────────────────────────────────────────────
    _status("Step 1/4 — Compressing ...")
    t0 = time.perf_counter()

    # Apply downscale for eval mode
    actual_video = video_path
    if abs(downscale - 1.0) > 1e-4:
        _status(f"  Downscaling input by {downscale}×")
        import tempfile, cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  * downscale)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * downscale)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        wtr = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        while True:
            ok, frm = cap.read()
            if not ok: break
            wtr.write(cv2.resize(frm, (W, H), interpolation=cv2.INTER_AREA))
        cap.release(); wtr.release()
        actual_video = Path(tmp.name)

    comp_cfg = _merge_sub_config(pipeline_cfg, "compression")
    from src.compression.phase_compress import compress_video
    archive_bytes = compress_video(str(actual_video), comp_cfg)

    if save_intermediate:
        arc_path = out_dir / f"{stem}.zip"
        with open(arc_path, "wb") as f:
            f.write(archive_bytes)
        _status(f"  Archive saved → {arc_path}")

    _status(f"  Compression done in {time.perf_counter()-t0:.1f}s "
            f"({len(archive_bytes)/1e6:.2f} MB)")

    # ── Step 2: Decompress ────────────────────────────────────────────────────
    _status("Step 2/4 — Decompressing ...")
    t1 = time.perf_counter()

    decomp_cfg = _merge_sub_config(pipeline_cfg, "decompression")
    from src.decompression.phase_decompress import decompress_archive
    decomp_result = decompress_archive(archive_bytes, decomp_cfg)
    frames = decomp_result["frames"]
    fps    = decomp_result["fps"]

    if save_intermediate:
        from src.postprocessing.video_assembler import assemble_video
        decomp_path = out_dir / f"{stem}_decompressed.mp4"
        assemble_video(iter(frames), decomp_path, fps=fps)
        _status(f"  Decompressed video → {decomp_path}")

    _status(f"  Decompression done in {time.perf_counter()-t1:.1f}s "
            f"({len(frames)} frames)")

    # ── Step 3: Restore (optional) ────────────────────────────────────────────
    restore_enabled = (
        not args.skip_restore
        and bool((pipeline_cfg.get("restoration", {}) or {}).get("enable", True))
    )

    if restore_enabled:
        _status("Step 3/4 — Restoring (Model R) ...")
        t2 = time.perf_counter()
        restore_cfg = _merge_sub_config(pipeline_cfg, "restoration")
        from src.restoration.phase_restore import restore_frames
        frames = restore_frames(frames, restore_cfg)

        if save_intermediate:
            from src.postprocessing.video_assembler import assemble_video
            rest_path = out_dir / f"{stem}_restored.mp4"
            assemble_video(iter(frames), rest_path, fps=fps)
            _status(f"  Restored video → {rest_path}")

        _status(f"  Restoration done in {time.perf_counter()-t2:.1f}s")
    else:
        _status("Step 3/4 — Restoration SKIPPED")

    # ── Step 4: Upscale (optional) ────────────────────────────────────────────
    upscale_enabled = (
        not args.skip_upscale
        and bool((pipeline_cfg.get("upscaling", {}) or {}).get("enable", True))
    )

    if upscale_enabled:
        _status("Step 4/4 — Upscaling (S3Diff) ...")
        t3 = time.perf_counter()
        upscale_cfg = pipeline_cfg.get("upscaling", {}) or {}
        scale       = int(upscale_cfg.get("scale", 4))
        half        = bool(upscale_cfg.get("half", False))
        seed        = int(upscale_cfg.get("seed", 42))

        import sys as _sys, random as _random
        import numpy as _np
        import torch as _torch
        import torch.nn.functional as _F
        from torchvision import transforms as _transforms

        S3DIFF_DIR  = ROOT / "S3Diff"
        weights_dir = Path(upscale_cfg.get("weights_dir", "weights"))
        DE_NET_PATH = weights_dir / "de_net.pth"
        S3DIFF_PATH = weights_dir / "s3diff.pkl"
        DE_NET_URL  = "https://huggingface.co/zhangap/S3Diff/resolve/main/de_net.pth"
        S3DIFF_URL  = "https://huggingface.co/zhangap/S3Diff/resolve/main/s3diff.pkl"

        import urllib.request as _urlreq
        import math as _math
        import subprocess as _subprocess

        def _dl(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            _status(f"  Downloading {dest.name} ...")
            _urlreq.urlretrieve(url, dest)

        if not S3DIFF_DIR.exists():
            _status("  Cloning S3Diff ...")
            _subprocess.run(
                ["git", "clone", "https://github.com/ArcticHare105/S3Diff.git", str(S3DIFF_DIR)],
                check=True,
            )
        if not DE_NET_PATH.exists():
            _dl(DE_NET_URL, DE_NET_PATH)
        if not S3DIFF_PATH.exists():
            _dl(S3DIFF_URL, S3DIFF_PATH)

        s3diff_src = S3DIFF_DIR / "src"
        for _p in [str(s3diff_src), str(S3DIFF_DIR)]:
            if _p not in _sys.path:
                _sys.path.insert(0, _p)

        from huggingface_hub import snapshot_download as _snap_dl
        from s3diff import S3Diff as _S3Diff
        from de_net import DEResNet as _DEResNet

        _status("  Downloading stabilityai/sd-turbo (cached after first run) ...")
        _sd_path = _snap_dl(repo_id="stabilityai/sd-turbo")

        _status("  Loading S3Diff ...")
        _upscale_device = _torch.device(pipeline_cfg.get("device", "cuda"))
        _net_sr = _S3Diff(lora_rank_unet=32, lora_rank_vae=16,
                          sd_path=_sd_path, pretrained_path=str(S3DIFF_PATH))
        _net_sr.set_eval()
        if half:
            _net_sr.half()

        _net_de = _DEResNet(num_in_ch=3, num_degradation=2)
        _de_ckpt = _torch.load(str(DE_NET_PATH), map_location="cpu")
        _net_de.load_state_dict(_de_ckpt.get("state_dict", _de_ckpt))
        _net_de.to(_upscale_device).eval()

        def _set_seed(s):
            _torch.manual_seed(s); _torch.cuda.manual_seed_all(s)
            _np.random.seed(s); _random.seed(s)

        def _upscale_frame(frame_bgr):
            import cv2 as _cv2
            frame_rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
            im_lr = _transforms.ToTensor()(frame_rgb).unsqueeze(0).to(_upscale_device)
            if half:
                im_lr = im_lr.half()
            ori_h, ori_w = im_lr.shape[2:]
            im_up = _F.interpolate(im_lr, size=(ori_h * scale, ori_w * scale),
                                   mode="bilinear", align_corners=False).contiguous()
            im_norm = (im_up * 2.0 - 1.0).clamp(-1.0, 1.0)
            res_h, res_w = im_norm.shape[2:]
            pad_h = _math.ceil(res_h / 64) * 64 - res_h
            pad_w = _math.ceil(res_w / 64) * 64 - res_w
            im_pad = _F.pad(im_norm, (0, pad_w, 0, pad_h), mode="reflect")
            with _torch.no_grad():
                deg = _net_de(im_lr).to(im_pad.device)
                out = _net_sr(im_pad, deg, prompt="a clear and high quality image")
            out = out[:, :, :res_h, :res_w]
            out_t = (out * 0.5 + 0.5).clamp(0, 1).cpu().float()
            out_np = (out_t[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(_np.uint8)
            return _cv2.cvtColor(out_np, _cv2.COLOR_RGB2BGR)

        upscaled = []
        for i, frame in enumerate(frames):
            _set_seed(seed)
            upscaled.append(_upscale_frame(frame))
            if (i + 1) % 10 == 0 or (i + 1) == len(frames):
                elapsed = time.perf_counter() - t3
                fps_up = (i + 1) / elapsed
                eta = (len(frames) - i - 1) / fps_up if fps_up > 0 else 0
                _status(f"  [upscale] {i+1}/{len(frames)} frames  {fps_up:.2f} fps  ETA {eta:.0f}s")
        frames = upscaled
        _status(f"  Upscaling done in {time.perf_counter()-t3:.1f}s")
    else:
        _status("Step 4/4 — Upscaling SKIPPED")

    # ── Write final output ────────────────────────────────────────────────────
    from src.postprocessing.video_assembler import assemble_video
    assemble_video(iter(frames), final_out, fps=fps)

    total = time.perf_counter() - t0
    _status(f"Pipeline complete in {total:.1f}s → {final_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
