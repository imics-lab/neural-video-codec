#!/usr/bin/env python3
import argparse
import io
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import add_config_arg, apply_config_defaults, load_config
from src.layers.cuda_inference import replicate_pad
from src.models.int16_reference import DMCIInt16Reference, DMCInt16Reference
from src.utils.stream_helper import SPSHelper, write_ip, write_sps
from src.utils.transforms import ycbcr420_to_444_np
from src.utils.equivalence import (
    collect_entrypoint_metadata,
    collect_git_metadata,
    collect_model_bundle_metadata,
    collect_relevant_env,
    collect_torch_device_metadata,
    sha256_bytes,
)
from src.utils.video_reader import YUV420Reader


INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Standalone DCVC-RT INT16 encoder: mp4 -> .bin"
    )
    add_config_arg(parser)
    parser.add_argument(
        "--input_mp4",
        type=str,
        default="test.mp4",
        help="Input MP4 path relative to this folder or absolute path.",
    )
    parser.add_argument(
        "--bundle_path",
        type=str,
        default="models/int16_bundle_v1.0.0.pt",
        help="Path to the local int16 reference bundle.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for the generated .bin and JSON metrics.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=-1,
        help="Number of frames to encode. Use -1 to encode the full video.",
    )
    parser.add_argument("--qp_i", type=int, default=32, help="I-frame QP.")
    parser.add_argument(
        "--qp_p",
        type=int,
        default=32,
        help="Base P-frame QP before module-bank shift.",
    )
    parser.add_argument(
        "--reset_interval",
        type=int,
        default=32,
        help="Feature-adaptor reset interval validated for the standalone path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--warmup_skip",
        type=int,
        default=10,
        help="Frames skipped when reporting average steady-state encode time.",
    )
    parser.add_argument(
        "--keep_yuv",
        action="store_true",
        help="Keep the temporary raw YUV file used for encoding.",
    )
    parser.add_argument(
        "--log_frame_stats",
        action="store_true",
        help="Enable verbose int16 tensor diagnostics inside the runtime.",
    )
    parser.add_argument(
        "--enable_pframe_graphs",
        action="store_true",
        help="Enable the optional P-frame CUDA graph path for stable NN submodules.",
    )
    parser.add_argument(
        "--disable_encode_only",
        action="store_true",
        help="Force the full encoder+decoder sync path instead of the faster encode-only mode.",
    )
    parser.add_argument(
        "--profile_pframe_stages",
        action="store_true",
        help="Enable the existing P-frame stage profiler and write a structured JSON artifact.",
    )
    parser.add_argument(
        "--profile_output_json",
        type=str,
        default=None,
        help="Optional JSON path for stage-profile output. Relative paths resolve under --output_dir.",
    )
    parser.add_argument(
        "--enable_async_entropy_prep",
        action="store_true",
        help=(
            "Opt-in experimental P-frame entropy prep overlap for encode-only mode. "
            "Dynamic entropy packing still stays outside whole-frame CUDA graph capture."
        ),
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Validate MP4 metadata, runtime flags, and bundle presence without encoding.",
    )
    parser.add_argument(
        "--print_json",
        action="store_true",
        help="Print full JSON payload to stdout (default prints concise summary).",
    )

    known, _ = parser.parse_known_args(argv)
    cfg = load_config(known.config)
    apply_config_defaults(parser, cfg, "encode")

    return parser.parse_args(argv)


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def run_command(command):
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def probe_video(input_mp4):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(input_mp4),
    ]
    completed = run_command(command)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) == 0:
        raise RuntimeError(f"No video stream found in {input_mp4}")

    stream = streams[0]
    fps_expr = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    try:
        fps = float(Fraction(fps_expr))
    except Exception:
        fps = 30.0

    nb_frames = stream.get("nb_frames")
    if nb_frames in (None, "", "N/A"):
        duration = stream.get("duration")
        if duration not in (None, "", "N/A"):
            nb_frames = int(round(float(duration) * fps))
        else:
            nb_frames = None
    else:
        nb_frames = int(nb_frames)

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "nb_frames": nb_frames,
    }


def resolve_frame_count(requested_frames, available_frames):
    if requested_frames is not None and requested_frames > 0:
        if available_frames is None:
            return int(requested_frames)
        return int(min(requested_frames, available_frames))
    if available_frames is None:
        raise RuntimeError(
            "Unable to infer frame count from ffprobe. Pass --frames explicitly."
        )
    return int(available_frames)


def extract_mp4_to_yuv(input_mp4, output_yuv, frames):
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_mp4),
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        "yuv420p",
    ]
    if frames > 0:
        command.extend(["-frames:v", str(frames)])
    command.extend(["-f", "rawvideo", str(output_yuv)])
    run_command(command)


def np_image_to_tensor(img, device):
    image = torch.from_numpy(img).to(device=device).to(dtype=torch.float32) / 255.0
    return image.unsqueeze(0)


def get_padding_size(height, width, p=16):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return new_w - width, new_h - height


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device=device)


def fps_label(fps):
    rounded = round(fps)
    if abs(fps - rounded) < 1e-6:
        return str(int(rounded))
    return f"{fps:.3f}".replace(".", "p")


def resolve_output_path(path_str, output_dir):
    path = Path(path_str)
    if not path.is_absolute():
        path = (output_dir / path).resolve()
    return path


def configure_runtime_flags(args):
    encode_only = not args.disable_encode_only
    if args.enable_pframe_graphs:
        os.environ["DCVC_ENABLE_INT16_PFRAME_GRAPHS"] = "1"
    else:
        os.environ.pop("DCVC_ENABLE_INT16_PFRAME_GRAPHS", None)
    if args.profile_pframe_stages:
        os.environ["DCVC_PROFILE_INT16_PIPELINE"] = "1"
    else:
        os.environ.pop("DCVC_PROFILE_INT16_PIPELINE", None)
    if args.enable_async_entropy_prep:
        os.environ["DCVC_INT16_ASYNC_ENTROPY_PREP"] = "1"
    else:
        os.environ.pop("DCVC_INT16_ASYNC_ENTROPY_PREP", None)
    if encode_only:
        os.environ["DCVC_INT16_ENCODE_ONLY"] = "1"
    else:
        os.environ.pop("DCVC_INT16_ENCODE_ONLY", None)
    return {
        "encode_only": bool(encode_only),
        "enable_pframe_graphs": bool(args.enable_pframe_graphs),
        "profile_pframe_stages": bool(args.profile_pframe_stages),
        "enable_async_entropy_prep": bool(args.enable_async_entropy_prep),
        "log_frame_stats": bool(args.log_frame_stats),
    }


def build_preflight_report(args, input_mp4, bundle_path, output_dir, video_info, frame_count):
    requested_flags = configure_runtime_flags(args)
    effective_flags = {
        "encode_only": bool(requested_flags["encode_only"]),
        "enable_pframe_graphs": bool(requested_flags["enable_pframe_graphs"]),
        "profile_pframe_stages": bool(requested_flags["profile_pframe_stages"]),
        "async_entropy_prep": False,
        "log_frame_stats": bool(requested_flags["log_frame_stats"]),
    }
    device = torch.device(args.device)
    profile_path = (
        resolve_output_path(args.profile_output_json, output_dir)
        if args.profile_output_json
        else output_dir / f"{input_mp4.stem}_{video_info['width']}x{video_info['height']}_{fps_label(video_info['fps'])}_{frame_count}f_q{args.qp_i}_pframe_profile.json"
    )
    return {
        "codec": "dcvc_rt_int16",
        "mode": "encode_preflight",
        "input_mp4": str(input_mp4),
        "bundle_path": str(bundle_path),
        "bundle_exists": bool(bundle_path.exists()),
        "output_dir": str(output_dir),
        "profile_output_json": str(profile_path),
        "frames_requested": int(args.frames),
        "frames_resolved": int(frame_count),
        "width": int(video_info["width"]),
        "height": int(video_info["height"]),
        "fps": float(video_info["fps"]),
        "source_frame_count": (
            int(video_info["nb_frames"]) if video_info["nb_frames"] is not None else None
        ),
        "qp_i": int(args.qp_i),
        "qp_p": int(args.qp_p),
        "reset_interval": int(args.reset_interval),
        "device": str(device),
        "requested_flags": requested_flags,
        "effective_flags": effective_flags,
        "equivalence_class": build_equivalence_class(
            __file__,
            args,
            device,
            bundle_path,
            video_info["width"],
            video_info["height"],
            video_info["fps"],
            frame_count,
            0,
            requested_flags,
            effective_flags,
            [
                "Preflight only; no bitstream was produced.",
                "Async entropy prep is reverted and currently inactive.",
            ],
        ),
    }


def summarize_stage_profiles(profile_records, warmup_skip):
    effective_skip = min(max(int(warmup_skip), 0), max(len(profile_records) - 1, 0))
    steady_records = profile_records[effective_skip:] if profile_records else []
    buckets = {}
    for record in steady_records:
        for stage_name, stage_metrics in record["stages"].items():
            bucket = buckets.setdefault(stage_name, {"gpu_ms": [], "cpu_ms": []})
            bucket["gpu_ms"].append(float(stage_metrics["gpu_ms"]))
            bucket["cpu_ms"].append(float(stage_metrics["cpu_ms"]))

    stage_summary = {}
    avg_gpu_ms_by_stage = {}
    avg_cpu_ms_by_stage = {}
    for stage_name, bucket in buckets.items():
        gpu_values = bucket["gpu_ms"]
        cpu_values = bucket["cpu_ms"]
        if len(gpu_values) == 0:
            continue
        avg_gpu = sum(gpu_values) / len(gpu_values)
        avg_cpu = sum(cpu_values) / len(cpu_values)
        stage_summary[stage_name] = {
            "avg_gpu_ms": float(avg_gpu),
            "avg_cpu_ms": float(avg_cpu),
            "min_gpu_ms": float(min(gpu_values)),
            "max_gpu_ms": float(max(gpu_values)),
            "min_cpu_ms": float(min(cpu_values)),
            "max_cpu_ms": float(max(cpu_values)),
            "samples": int(len(gpu_values)),
        }
        avg_gpu_ms_by_stage[stage_name] = float(avg_gpu)
        avg_cpu_ms_by_stage[stage_name] = float(avg_cpu)

    return {
        "warmup_skip_frames": int(effective_skip),
        "frames_profiled": int(len(steady_records)),
        "avg_gpu_ms_by_stage": avg_gpu_ms_by_stage,
        "avg_cpu_ms_by_stage": avg_cpu_ms_by_stage,
        "stage_summary": stage_summary,
        "per_frame": profile_records,
    }


def build_equivalence_class(
    script_path,
    args,
    device,
    bundle_path,
    width,
    height,
    fps,
    frame_count,
    encoded_frame_count,
    requested_flags,
    effective_flags,
    notes,
):
    return {
        "git": collect_git_metadata(ROOT),
        "runtime": collect_torch_device_metadata(device),
        "env": collect_relevant_env(),
        "model_bundle": collect_model_bundle_metadata(bundle_path),
        "entrypoint": collect_entrypoint_metadata(script_path, sys.argv[1:]),
        "video": {
            "width": int(width),
            "height": int(height),
            "resolution": f"{int(width)}x{int(height)}",
            "fps": float(fps),
            "frames_requested": int(frame_count),
            "frames_encoded": int(encoded_frame_count),
        },
        "qps": {
            "qp_i": int(args.qp_i),
            "qp_p_base": int(args.qp_p),
            "reset_interval": int(args.reset_interval),
        },
        "requested_flags": requested_flags,
        "effective_flags": effective_flags,
        "notes": notes,
    }


def load_models(bundle_path, device, padded_height, padded_width, log_frame_stats=False):
    bundle_blob = torch.load(bundle_path, map_location="cpu", weights_only=False)
    bundle_models = bundle_blob["models"] if "models" in bundle_blob else bundle_blob

    i_frame_net = DMCIInt16Reference(bundle_models["i_frame_net"]).to(device).eval()
    p_frame_net = DMCInt16Reference(bundle_models["p_frame_net"]).to(device).eval()

    if hasattr(i_frame_net, "init_cuda_graph"):
        i_frame_net.init_cuda_graph(padded_height, padded_width)
    if hasattr(p_frame_net, "set_log_frame_stats"):
        p_frame_net.set_log_frame_stats(log_frame_stats)

    return i_frame_net, p_frame_net


def print_preflight_summary(report):
    print("[preflight] OK")
    print(f"input:  {report['input_mp4']}")
    print(f"bundle: {report['bundle_path']}")
    print(
        f"video:  {report['width']}x{report['height']} @ {report['fps']:.3f} fps, "
        f"frames={report['frames_requested']}"
    )
    print(f"flags:  encode_only={report['requested_flags']['encode_only']}, "
          f"graphs={report['requested_flags']['enable_pframe_graphs']}, "
          f"profile={report['requested_flags']['profile_pframe_stages']}")


def print_encode_summary(metrics, metrics_path):
    print("[encode] done")
    print(f"bitstream: {metrics['bitstream_path']}")
    print(f"metrics:   {metrics_path}")
    print(
        f"frames:    {metrics['frames_encoded']} | "
        f"time: {metrics['total_encode_time_sec']:.2f}s | "
        f"avg: {metrics['avg_frame_encode_time_ms']:.2f} ms/frame"
    )
    print(
        f"size:      {metrics['total_bytes']} bytes | "
        f"bitrate: {metrics['bitrate_kbps']:.2f} kbps | "
        f"sha256: {metrics['bitstream_sha256']}"
    )


def encode_video(args):
    input_mp4 = resolve_path(args.input_mp4)
    if not input_mp4.exists():
        raise FileNotFoundError(f"Input MP4 not found: {input_mp4}")

    bundle_path = resolve_path(args.bundle_path)
    output_dir = resolve_path(args.output_dir)

    video_info = probe_video(input_mp4)
    width = video_info["width"]
    height = video_info["height"]
    fps = video_info["fps"] if video_info["fps"] > 0 else 30.0
    frame_count = resolve_frame_count(args.frames, video_info["nb_frames"])

    if args.check_only:
        preflight = build_preflight_report(
            args, input_mp4, bundle_path, output_dir, video_info, frame_count
        )
        if args.print_json:
            print(json.dumps(preflight, indent=2))
        else:
            print_preflight_summary(preflight)
        return

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Int16 bundle not found: {bundle_path}. "
            "Run scripts/download_models.* or pass --check_only for MP4/runtime preflight."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_mp4.stem
    fps_name = fps_label(fps)
    temp_yuv = output_dir / f"{stem}_{width}x{height}_{fps_name}_{frame_count}f.yuv"
    bitstream_path = output_dir / f"{stem}_{width}x{height}_{fps_name}_{frame_count}f_q{args.qp_i}.bin"
    metrics_path = output_dir / f"{stem}_{width}x{height}_{fps_name}_{frame_count}f_q{args.qp_i}.json"
    profile_path = (
        resolve_output_path(args.profile_output_json, output_dir)
        if args.profile_output_json
        else output_dir / f"{stem}_{width}x{height}_{fps_name}_{frame_count}f_q{args.qp_i}_pframe_profile.json"
    )

    extract_t0 = time.perf_counter()
    extract_mp4_to_yuv(input_mp4, temp_yuv, frame_count)
    extract_time = time.perf_counter() - extract_t0

    device = torch.device(args.device)
    configure_runtime_flags(args)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
        torch.cuda.set_device(device)

    padding_r, padding_b = get_padding_size(height, width, 16)
    padded_height = height + padding_b
    padded_width = width + padding_r

    i_frame_net, p_frame_net = load_models(
        bundle_path,
        device,
        padded_height,
        padded_width,
        log_frame_stats=args.log_frame_stats,
    )
    if hasattr(p_frame_net, "init_cuda_graph_pframe"):
        p_frame_net.init_cuda_graph_pframe(padded_height, padded_width)

    use_two_entropy_coders = height * width > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

    reader = YUV420Reader(str(temp_yuv), width, height)
    output_buff = io.BytesIO()
    sps_helper = SPSHelper()

    encoded_frame_count = 0
    frame_types = []
    bits = []
    encoding_times = []
    last_qp = 0
    encode_only = not args.disable_encode_only
    stage_profile_records = []

    p_frame_net.set_curr_poc(0)
    pframe_graphs_enabled = bool(getattr(p_frame_net, "_cuda_graph_enabled", False))
    profiling_enabled = os.environ.get("DCVC_PROFILE_INT16_PIPELINE", "0") == "1"
    async_entropy_prep_enabled = False
    if hasattr(p_frame_net, "is_async_entropy_prep_active"):
        async_entropy_prep_enabled = bool(
            p_frame_net.is_async_entropy_prep_active(encode_only=encode_only)
        )
    requested_flags = {
        "encode_only": bool(encode_only),
        "enable_pframe_graphs": bool(args.enable_pframe_graphs),
        "profile_pframe_stages": bool(args.profile_pframe_stages),
        "enable_async_entropy_prep": bool(args.enable_async_entropy_prep),
        "log_frame_stats": bool(args.log_frame_stats),
    }
    effective_flags = {
        "encode_only": bool(encode_only),
        "enable_pframe_graphs": bool(pframe_graphs_enabled),
        "profile_pframe_stages": bool(profiling_enabled),
        "async_entropy_prep": bool(async_entropy_prep_enabled),
        "log_frame_stats": bool(args.log_frame_stats),
    }
    equivalence_notes = []
    if pframe_graphs_enabled:
        equivalence_notes.append(
            "P-frame CUDA graphs currently cover only stable NN submodules; "
            "dynamic entropy packing and CPU rANS handoff remain outside capture."
        )
    if (
        (args.enable_async_entropy_prep or os.environ.get("DCVC_INT16_ASYNC_ENTROPY_PREP") == "1")
        and not async_entropy_prep_enabled
    ):
        equivalence_notes.append(
            "Async entropy prep is reverted and currently inactive; the implementation "
            "is retained only as reference code."
        )

    encode_t0 = time.perf_counter()
    try:
        with torch.no_grad():
            for frame_idx in range(frame_count):
                y, uv = reader.read_one_frame()
                if y is None or uv is None:
                    break

                x = np_image_to_tensor(ycbcr420_to_444_np(y, uv), device).to(torch.float32)
                x_padded = replicate_pad(x, padding_b, padding_r)

                synchronize(device)
                frame_t0 = time.perf_counter()

                if frame_idx == 0:
                    curr_qp = args.qp_i
                    is_i_frame = True
                    sps = {
                        "sps_id": -1,
                        "height": height,
                        "width": width,
                        "ec_part": 1 if use_two_entropy_coders else 0,
                        "use_ada_i": 0,
                    }
                    encoded = i_frame_net.compress(x_padded, curr_qp)
                    p_frame_net.clear_dpb()
                    p_frame_net.add_ref_frame(None, encoded["x_hat"])
                    frame_types.append("I")
                else:
                    fa_idx = INDEX_MAP[frame_idx % 8]
                    use_ada_i = 1 if (args.reset_interval > 0 and frame_idx % args.reset_interval == 1) else 0
                    if use_ada_i:
                        p_frame_net.prepare_feature_adaptor_i(last_qp)

                    curr_qp = p_frame_net.shift_qp(args.qp_p, fa_idx)
                    is_i_frame = False
                    sps = {
                        "sps_id": -1,
                        "height": height,
                        "width": width,
                        "ec_part": 1 if use_two_entropy_coders else 0,
                        "use_ada_i": use_ada_i,
                    }
                    encoded = p_frame_net.compress(x_padded, curr_qp, encode_only=encode_only)
                    if hasattr(p_frame_net, "consume_last_profile_result"):
                        profile_result = p_frame_net.consume_last_profile_result()
                        if profile_result:
                            stage_profile_records.append(
                                {
                                    "frame_idx": int(frame_idx),
                                    "qp": int(curr_qp),
                                    "stages": profile_result,
                                }
                            )
                    last_qp = curr_qp
                    frame_types.append("P")

                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                sps_bytes = 0
                if sps_new:
                    sps_bytes = write_sps(output_buff, sps)
                stream_bytes = write_ip(output_buff, is_i_frame, sps_id, curr_qp, encoded["bit_stream"])
                bits.append((stream_bytes + sps_bytes) * 8)

                synchronize(device)
                encoding_times.append(time.perf_counter() - frame_t0)
                encoded_frame_count += 1
    finally:
        reader.close()

    with open(bitstream_path, "wb") as fp:
        fp.write(output_buff.getbuffer())

    total_encode_time = time.perf_counter() - encode_t0
    bitstream_sha256 = sha256_bytes(output_buff.getbuffer())
    total_bits = int(sum(bits))
    total_bytes = len(output_buff.getbuffer())
    effective_skip = min(max(args.warmup_skip, 0), max(len(encoding_times) - 1, 0))
    steady_times = encoding_times[effective_skip:] if encoding_times else []
    avg_encoding_time = (
        sum(steady_times) / len(steady_times) if len(steady_times) > 0 else 0.0
    )
    bpp = (
        total_bits / float(encoded_frame_count * width * height)
        if encoded_frame_count > 0
        else 0.0
    )
    kbps = total_bits / (encoded_frame_count / fps) / 1000.0 if encoded_frame_count > 0 else 0.0
    equivalence_class = build_equivalence_class(
        __file__,
        args,
        device,
        bundle_path,
        width,
        height,
        fps,
        frame_count,
        encoded_frame_count,
        requested_flags,
        effective_flags,
        equivalence_notes,
    )

    metrics = {
        "codec": "dcvc_rt_int16",
        "mode": "encode",
        "input_mp4": str(input_mp4),
        "bundle_path": str(bundle_path),
        "temp_yuv": str(temp_yuv),
        "bitstream_path": str(bitstream_path),
        "frames_requested": int(frame_count),
        "frames_encoded": int(encoded_frame_count),
        "source_frame_count": (
            int(video_info["nb_frames"]) if video_info["nb_frames"] is not None else None
        ),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "qp_i": int(args.qp_i),
        "qp_p": int(args.qp_p),
        "reset_interval": int(args.reset_interval),
        "frame_types": frame_types,
        "total_bits": total_bits,
        "total_bytes": total_bytes,
        "bitstream_sha256": bitstream_sha256,
        "all_frame_bpp": float(bpp),
        "bitrate_kbps": float(kbps),
        "extract_time_sec": float(extract_time),
        "total_encode_time_sec": float(total_encode_time),
        "total_encode_time_min": float(total_encode_time / 60.0),
        "avg_frame_encode_time_sec": float(avg_encoding_time),
        "avg_frame_encode_time_ms": float(avg_encoding_time * 1000.0),
        "warmup_skip_frames": int(effective_skip),
        "use_two_entropy_coders": bool(use_two_entropy_coders),
        "device": str(device),
        "encode_only": bool(encode_only),
        "enable_pframe_graphs": bool(pframe_graphs_enabled),
        "profile_pframe_stages": bool(profiling_enabled),
        "async_entropy_prep": bool(async_entropy_prep_enabled),
        "log_frame_stats": bool(args.log_frame_stats),
        "requested_flags": requested_flags,
        "effective_flags": effective_flags,
        "equivalence_class": equivalence_class,
    }

    if profiling_enabled:
        profile_summary = summarize_stage_profiles(stage_profile_records, args.warmup_skip)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_artifact = {
            "codec": "dcvc_rt_int16",
            "mode": "encode_pframe_profile",
            "input_mp4": str(input_mp4),
            "bitstream_path": str(bitstream_path),
            "metrics_path": str(metrics_path),
            "profile_path": str(profile_path),
            "frames_requested": int(frame_count),
            "frames_encoded": int(encoded_frame_count),
            "profiled_pframes": int(len(stage_profile_records)),
            "requested_flags": requested_flags,
            "effective_flags": effective_flags,
            "equivalence_class": equivalence_class,
            "summary": profile_summary,
        }
        with open(profile_path, "w", encoding="utf-8") as fp:
            json.dump(profile_artifact, fp, indent=2)
        metrics["pframe_profile_path"] = str(profile_path)
        metrics["pframe_profiled_frames"] = int(profile_summary["frames_profiled"])
    else:
        metrics["pframe_profile_path"] = None
        metrics["pframe_profiled_frames"] = 0

    if not args.keep_yuv and temp_yuv.exists():
        temp_yuv.unlink()
        metrics["temp_yuv_deleted"] = True
    else:
        metrics["temp_yuv_deleted"] = False

    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    if args.print_json:
        print(json.dumps(metrics, indent=2))
    else:
        print_encode_summary(metrics, metrics_path)


def main():
    encode_video(parse_args())


if __name__ == "__main__":
    main()
