#!/usr/bin/env python3
"""
run_pipeline.py — Modular pipeline: mix and match stages freely.

Available stages (run in the order listed):
    compress          compress --input video with DCVC → archive in memory
    decompress        decompress archive → BGR frame list
    restore           apply RestoreUNet diffusion model
    upscale-bicubic   Lanczos 4× bicubic resize to out_w × out_h
    upscale-s3diff    S3Diff one-step diffusion SR
    upscale-osediff   OSEDiff one-step diffusion SR (SD2.1-based)
    upscale-wan       Wan2.1 I2V video upscaling
    upscale-cogvideo  CogVideoX V2V video enhancement

Usage examples:

  # Full pipeline (default stages)
  python run_pipeline.py --video bird1.mp4 --config configs/gpu/compression.yaml

  # Skip restore, compare Wan2.1 upscale straight from decompressed
  python run_pipeline.py --video bird1.mp4 --config configs/gpu/compression.yaml \\
      --stages compress decompress upscale-wan

  # Start from an already-decompressed video (skip compress+decompress)
  python run_pipeline.py --input bird1_decompressed.mp4 --config configs/gpu/compression.yaml \\
      --stages upscale-wan

  # Restore only (no upscale)
  python run_pipeline.py --input bird1_decompressed.mp4 --config configs/gpu/compression.yaml \\
      --stages restore
"""
from __future__ import annotations

import argparse
import logging
import sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("codec.pipeline")

VALID_STAGES = [
    "compress",
    "decompress",
    "restore",
    "upscale-bicubic",
    "upscale-s3diff",
    "upscale-osediff",
    "upscale-wan",
    "upscale-cogvideo",
]
DEFAULT_STAGES = ["compress", "decompress", "restore", "upscale-bicubic"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _status(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)
    import os; os.environ["YOLO_VERBOSE"] = "true" if verbose else "false"


def _load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        return (yaml.safe_load(f) if yaml else __import__("json").load(f)) or {}


def _merge_sub_config(pipeline_cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Load a sub-stage config file if referenced, then merge pipeline overrides."""
    sub_cfg_path = (pipeline_cfg.get(key, {}) or {}).get("config", "")
    base     = _load_config(sub_cfg_path) if sub_cfg_path and Path(sub_cfg_path).exists() else {}
    overrides = {k: v for k, v in (pipeline_cfg.get(key, {}) or {}).items() if k != "config"}
    return {**base, **overrides}


def _load_video_frames(path: Path):
    """Read all BGR frames from a video file. Returns (frames, fps)."""
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        frames.append(frm)
    cap.release()
    return frames, fps


def _save_video(frames: List[np.ndarray], path: Path, fps: float) -> None:
    from src.postprocessing.video_assembler import assemble_video
    path.parent.mkdir(parents=True, exist_ok=True)
    assemble_video(iter(frames), path, fps=fps)


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Modular pipeline: pick any combination of stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input — either raw video (for compress) or pre-decompressed video
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", "--input", dest="input",
                     help="Input video (raw .mp4 for compress, or decompressed .mp4 otherwise)")
    p.add_argument("--config",  default="configs/gpu/compression.yaml",
                   help="Main config YAML (default: configs/gpu/compression.yaml)")
    p.add_argument("--output",  default=None,
                   help="Output video path (auto-named if omitted)")
    p.add_argument("--stages",  nargs="+", default=DEFAULT_STAGES,
                   metavar="STAGE",
                   help=(f"Ordered list of stages to run. "
                         f"Valid: {', '.join(VALID_STAGES)}. "
                         f"Default: {' '.join(DEFAULT_STAGES)}"))
    p.add_argument("--detections", default=None,
                   help="ZIP archive or JSON file containing detections for ROI overlay")
    p.add_argument("--save-intermediate", action="store_true",
                   help="Save video after each stage to output dir")
    p.add_argument("--out-w",      type=int, default=None, help="Upscale output width")
    p.add_argument("--out-h",      type=int, default=None, help="Upscale output height")
    p.add_argument("--ddim-steps", type=int, default=None,
                   help="Override restoration ddim_steps (e.g. 1 3 6 10 20)")
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


# ── Stage implementations ─────────────────────────────────────────────────────

def stage_compress(input_video: Path, pipeline_cfg: dict):
    """Returns archive bytes + detection dict."""
    _status("compress — DCVC encode ...")
    from src.compression.phase_compress import compress_video
    archive_bytes = compress_video(str(input_video), pipeline_cfg)
    _status(f"  archive size: {len(archive_bytes)/1e6:.2f} MB")
    return archive_bytes


def stage_decompress(archive_bytes: bytes, pipeline_cfg: dict):
    """Returns (frames, fps, width, height, detections)."""
    _status("decompress — DCVC decode ...")
    import zipfile, io, json as _json
    from src.decompression.phase_decompress import decompress_archive
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as arc:
        detections = _json.loads(arc.read("detections.json"))
    decomp_cfg = _merge_sub_config(pipeline_cfg, "decompression")
    result = decompress_archive(archive_bytes, decomp_cfg)
    frames = result["frames"]
    fps    = result["fps"]
    width  = result.get("width", frames[0].shape[1] if frames else 0)
    height = result.get("height", frames[0].shape[0] if frames else 0)
    _status(f"  {len(frames)} frames @ {fps:.1f} fps  {width}×{height}")
    # DCVC sets CUDA_VISIBLE_DEVICES — clear it so later CUDA ops work
    import os; os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    return frames, fps, width, height, detections


def stage_restore(frames, fps, width, height, detections, pipeline_cfg: dict,
                  ddim_steps: int = None):
    """Returns restored frames list."""
    _status("restore — RestoreUNet diffusion ...")
    restore_cfg = _merge_sub_config(pipeline_cfg, "restoration")
    if ddim_steps is not None:
        restore_cfg.setdefault("inference", {})["ddim_steps"] = ddim_steps
        _status(f"  ddim_steps overridden → {ddim_steps}")
    from src.restoration.phase_restore import restore_frames
    frames = restore_frames(frames, restore_cfg,
                            detections=detections, width=width, height=height)
    _status(f"  restored {len(frames)} frames")
    return frames


def stage_upscale_bicubic(frames, out_w: int, out_h: int):
    _status(f"upscale-bicubic — Lanczos → {out_w}×{out_h} ...")
    return [cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4) for f in frames]


def stage_upscale_s3diff(frames, out_w: int, out_h: int, pipeline_cfg: dict):
    _status(f"upscale-s3diff — S3Diff → {out_w}×{out_h} ...")
    import math, random, torch, torch.nn.functional as F, subprocess
    from torchvision import transforms
    from huggingface_hub import snapshot_download

    S3DIFF_DIR  = ROOT / "S3Diff"
    weights_dir = Path(pipeline_cfg.get("upscaling", {}).get("weights_dir", "weights"))
    DE_NET_PATH = weights_dir / "de_net.pth"
    S3DIFF_PATH = weights_dir / "s3diff.pkl"

    def _dl(url, dest):
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)

    if not S3DIFF_DIR.exists():
        subprocess.run(["git", "clone",
            "https://github.com/ArcticHare105/S3Diff.git", str(S3DIFF_DIR)], check=True)
    if not DE_NET_PATH.exists():
        _dl("https://huggingface.co/zhangap/S3Diff/resolve/main/de_net.pth", DE_NET_PATH)
    if not S3DIFF_PATH.exists():
        _dl("https://huggingface.co/zhangap/S3Diff/resolve/main/s3diff.pkl", S3DIFF_PATH)

    for _p in [str(S3DIFF_DIR / "src"), str(S3DIFF_DIR)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from s3diff import S3Diff
    from de_net import DEResNet

    device  = torch.device(pipeline_cfg.get("device", "cuda"))
    sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    net_sr  = S3Diff(lora_rank_unet=32, lora_rank_vae=16,
                     sd_path=sd_path, pretrained_path=str(S3DIFF_PATH))
    net_sr.set_eval().cuda()
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    ckpt = torch.load(str(DE_NET_PATH), map_location="cpu")
    net_de.load_state_dict(ckpt.get("state_dict", ckpt))
    net_de.to(device).half().eval()

    seed = int((pipeline_cfg.get("upscaling", {}) or {}).get("seed", 42))

    def _upscale(bgr):
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im_lr  = transforms.ToTensor()(rgb).unsqueeze(0).to(device)
        im_up  = F.interpolate(im_lr, size=(out_h, out_w), mode="bilinear", align_corners=False)
        im_norm = (im_up * 2.0 - 1.0).clamp(-1.0, 1.0)
        ph = math.ceil(out_h / 64) * 64 - out_h
        pw = math.ceil(out_w / 64) * 64 - out_w
        im_pad = F.pad(im_norm, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad(), torch.amp.autocast("cuda"):
            deg = net_de(im_lr.half()).to(device=im_pad.device)
            out = net_sr(im_pad, deg.float(), prompt="a clear and high quality image")
        out = out[:, :, :out_h, :out_w]
        out_np = (out * 0.5 + 0.5).clamp(0, 1).cpu().float()
        out_np = (out_np[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    result = []
    for i, f in enumerate(frames):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        result.append(_upscale(f))
        if (i + 1) % 10 == 0 or (i + 1) == len(frames):
            el = time.perf_counter() - t0
            _status(f"  [{i+1}/{len(frames)}] {(i+1)/el:.1f} fps")
    return result


def stage_upscale_osediff(frames, out_w: int, out_h: int, pipeline_cfg: dict,
                          detections: dict = None):
    _status(f"upscale-osediff — OSEDiff → {out_w}×{out_h} ...")
    from src.upscaling.osediff_upscaler import OSEDiffUpscaler
    ose_cfg = dict(pipeline_cfg.get("osediff_upscaling", {}) or {})
    ose_cfg.setdefault("out_w", out_w)
    ose_cfg.setdefault("out_h", out_h)
    upscaler = OSEDiffUpscaler(ose_cfg)
    return upscaler.upscale_sequence(frames, detections=detections)


def stage_upscale_cogvideo(frames, out_w: int, out_h: int, pipeline_cfg: dict,
                           detections: dict = None):
    _status(f"upscale-cogvideo — CogVideoX V2V → {out_w}×{out_h} ...")
    from src.upscaling.cogvideo_upscaler import CogVideoUpscaler
    cog_cfg = _merge_sub_config(pipeline_cfg, "cogvideo_upscaling")
    cog_cfg.setdefault("out_w", out_w)
    cog_cfg.setdefault("out_h", out_h)
    cog_cfg.setdefault("device", pipeline_cfg.get("device", "cuda"))
    upscaler = CogVideoUpscaler(cog_cfg)
    return upscaler.upscale_sequence(frames, detections=detections)


def stage_upscale_wan(frames, out_w: int, out_h: int, pipeline_cfg: dict):
    _status(f"upscale-wan — Wan2.1 I2V → {out_w}×{out_h} ...")
    from src.upscaling.wan_upscaler import WanUpscaler
    wan_cfg = _merge_sub_config(pipeline_cfg, "wan_upscaling")
    wan_cfg.setdefault("out_w", out_w)
    wan_cfg.setdefault("out_h", out_h)
    wan_cfg.setdefault("device", pipeline_cfg.get("device", "cuda"))
    upscaler = WanUpscaler(wan_cfg)
    return upscaler.upscale_sequence(frames)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    stages = [s.lower() for s in args.stages]
    for s in stages:
        if s not in VALID_STAGES:
            print(f"ERROR: unknown stage '{s}'. Valid: {', '.join(VALID_STAGES)}", file=sys.stderr)
            return 1

    # Validate that upscale stages don't conflict
    upscale_stages = [s for s in stages if s.startswith("upscale-")]
    if len(upscale_stages) > 1:
        print(f"ERROR: only one upscale stage at a time (got: {upscale_stages})", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 1

    pipeline_cfg = _load_config(args.config)

    # Pre-warm all CUDA contexts before DCVC takes over
    if "compress" in stages or "decompress" in stages:
        import torch as _torch
        for _i in range(_torch.cuda.device_count()):
            _torch.zeros(1, device=f"cuda:{_i}")

    # Output directory and naming
    out_dir = Path((pipeline_cfg.get("output", {}) or {}).get("out_dir", "outputs/pipeline"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    final_out = Path(args.output) if args.output else (
        out_dir / f"{stem}_{'_'.join(stages)}.mp4"
    )

    # ── Determine starting state ──────────────────────────────────────────────
    archive_bytes = None
    frames: List[np.ndarray] = []
    fps    = 30.0
    width  = height = 0
    detections: dict = {}

    # Load detections from --detections arg (ZIP archive or JSON file)
    if args.detections:
        det_path = Path(args.detections)
        if det_path.suffix.lower() == ".zip":
            import zipfile as _zf, io as _io
            with _zf.ZipFile(det_path) as _arc:
                detections = __import__("json").loads(_arc.read("detections.json"))
        else:
            with open(det_path) as _f:
                detections = __import__("json").load(_f)
        _status(f"Loaded detections from {det_path}: {len(detections)} frames")

    # Out dimensions for upscale stages
    _input_res = (pipeline_cfg.get("input", {}) or {}).get("input_resolution", "") or ""
    if _input_res:
        _rw, _rh = (int(x) for x in _input_res.lower().split("x"))
    else:
        _rh, _rw = (0, 0)

    out_w = args.out_w or (pipeline_cfg.get("upscaling", {}) or {}).get("out_w", _rw * 2 or 960)
    out_h = args.out_h or (pipeline_cfg.get("upscaling", {}) or {}).get("out_h", _rh * 2 or 720)

    t0 = time.perf_counter()

    # If compress is NOT in stages, load the input video as already-decoded frames
    if "compress" not in stages:
        _status(f"Loading decompressed video: {input_path}")
        frames, fps = _load_video_frames(input_path)
        h0, w0 = frames[0].shape[:2]
        width, height = w0, h0
        # Infer out dimensions from input if not set
        if out_w == 0: out_w = w0 * 2
        if out_h == 0: out_h = h0 * 2
        _status(f"  {len(frames)} frames @ {fps:.1f} fps  {w0}×{h0}")


    # ── Run stages ────────────────────────────────────────────────────────────
    for stage in stages:
        t_s = time.perf_counter()

        if stage == "compress":
            # Optional: resize input before compression
            actual_input = input_path
            if _input_res:
                import tempfile
                _tw, _th = (int(x) for x in _input_res.lower().split("x"))
                cap = cv2.VideoCapture(str(input_path))
                fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                wtr = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (_tw, _th))
                while True:
                    ok, frm = cap.read()
                    if not ok: break
                    wtr.write(cv2.resize(frm, (_tw, _th), interpolation=cv2.INTER_AREA))
                cap.release(); wtr.release()
                actual_input = Path(tmp.name)
            archive_bytes = stage_compress(actual_input, pipeline_cfg)
            if actual_input != input_path:
                import os; os.unlink(actual_input)
            if args.save_intermediate:
                arc_path = out_dir / f"{stem}.zip"
                arc_path.write_bytes(archive_bytes)
                _status(f"  archive → {arc_path}")

        elif stage == "decompress":
            frames, fps, width, height, detections = stage_decompress(archive_bytes, pipeline_cfg)
            if out_w == 0: out_w = width * 2
            if out_h == 0: out_h = height * 2
            if args.save_intermediate:
                p = out_dir / f"{stem}_decompressed.mp4"
                _save_video(frames, p, fps)
                _status(f"  decompressed → {p}")

        elif stage == "restore":
            frames = stage_restore(frames, fps, width, height, detections, pipeline_cfg,
                                   ddim_steps=args.ddim_steps)
            if args.save_intermediate:
                p = out_dir / f"{stem}_restored.mp4"
                _save_video(frames, p, fps)
                _status(f"  restored → {p}")

        elif stage == "upscale-bicubic":
            frames = stage_upscale_bicubic(frames, out_w, out_h)

        elif stage == "upscale-s3diff":
            frames = stage_upscale_s3diff(frames, out_w, out_h, pipeline_cfg)

        elif stage == "upscale-osediff":
            frames = stage_upscale_osediff(frames, out_w, out_h, pipeline_cfg,
                                           detections=detections)

        elif stage == "upscale-wan":
            frames = stage_upscale_wan(frames, out_w, out_h, pipeline_cfg)

        elif stage == "upscale-cogvideo":
            frames = stage_upscale_cogvideo(frames, out_w, out_h, pipeline_cfg,
                                            detections=detections)

        _status(f"  {stage} done in {time.perf_counter()-t_s:.1f}s")

    # ── Write final output ────────────────────────────────────────────────────
    _save_video(frames, final_out, fps)
    _status(f"Pipeline complete in {time.perf_counter()-t0:.1f}s → {final_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
