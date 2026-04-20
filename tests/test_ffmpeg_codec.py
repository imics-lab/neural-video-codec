from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compression.ffmpeg_codec import VideoInfo, _stream_codec_config, encode_ffmpeg_frames_to_bytes  # noqa: E402


class _FakePipe:
    def __init__(self, *, break_on_write: bool = False) -> None:
        self.break_on_write = bool(break_on_write)
        self.closed = False

    def write(self, data: bytes) -> int:
        if self.break_on_write:
            raise BrokenPipeError(32, "Broken pipe")
        return len(data)

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    def __init__(self, *, break_on_write: bool = False, returncode: int = 1, stderr_text: str = "") -> None:
        self.stdin = _FakePipe(break_on_write=break_on_write)
        self.stdout = None
        self.stderr = io.BytesIO(stderr_text.encode("utf-8"))
        self._returncode = int(returncode)

    def wait(self, timeout: int | None = None) -> int:
        return self._returncode


class TestFfmpegCodec(unittest.TestCase):
    def test_auto_encoder_prefers_first_available_av1_encoder(self) -> None:
        cfg = {
            "codec": {
                "ffmpeg_bin": "ffmpeg",
                "roi": {
                    "codec": "av1",
                    "encoder": "auto",
                    "encoder_candidates": ["libsvtav1", "libaom-av1", "librav1e"],
                    "container": "mkv",
                    "preset": "slowest",
                    "qp": 10,
                },
            }
        }
        with patch("src.compression.ffmpeg_codec._available_ffmpeg_encoders", return_value={"libaom-av1", "libx265"}):
            stream_cfg = _stream_codec_config(cfg, "roi")

        self.assertEqual(stream_cfg["encoder"], "libaom-av1")
        self.assertEqual(stream_cfg["preset"], "slowest")

    def test_missing_explicit_encoder_raises_clear_runtime_error(self) -> None:
        cfg = {
            "codec": {
                "ffmpeg_bin": "ffmpeg",
                "roi": {
                    "codec": "av1",
                    "encoder": "libsvtav1",
                    "container": "mkv",
                    "preset": "slowest",
                    "qp": 10,
                },
            }
        }
        with patch("src.compression.ffmpeg_codec._available_ffmpeg_encoders", return_value={"libaom-av1"}):
            with self.assertRaises(RuntimeError) as ctx:
                _stream_codec_config(cfg, "roi")

        self.assertIn("not available", str(ctx.exception))
        self.assertIn("encoder=auto", str(ctx.exception))

    def test_encode_wraps_broken_pipe_with_ffmpeg_stderr(self) -> None:
        frames = [(0, np.zeros((4, 6, 3), dtype=np.uint8))]
        cfg = {
            "codec": {
                "ffmpeg_bin": "ffmpeg",
                "roi": {
                    "codec": "av1",
                    "encoder": "auto",
                    "encoder_candidates": ["libsvtav1"],
                    "container": "mkv",
                    "preset": "slowest",
                    "qp": 10,
                },
            }
        }
        with tempfile.TemporaryDirectory() as td:
            with patch("src.compression.ffmpeg_codec._available_ffmpeg_encoders", return_value={"libsvtav1"}):
                with patch(
                    "src.compression.ffmpeg_codec.subprocess.Popen",
                    return_value=_FakeProc(
                        break_on_write=True,
                        returncode=1,
                        stderr_text="Unknown encoder 'libsvtav1'",
                    ),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        encode_ffmpeg_frames_to_bytes(
                            frames,
                            info=VideoInfo(width=6, height=4, fps=30.0, frames=1),
                            cfg=cfg,
                            stream_name="roi",
                            video_path=str(Path(td) / "video.mp4"),
                        )

        self.assertIn("terminated early", str(ctx.exception))
        self.assertIn("Unknown encoder", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
