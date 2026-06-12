from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compression.phase4_dcvc import compress_keep_streams_dcvc  # noqa: E402
from src.compression.dcvc_encoder import VideoInfo  # noqa: E402


class TestPhase4Metadata(unittest.TestCase):
    def test_phase4_uses_encoder_frame_index_map_authoritatively(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = tmp / "video.mp4"
            model_i = tmp / "i.pth.tar"
            model_p = tmp / "p.pth.tar"
            repo_dir = tmp / "DCVC"

            video_path.write_bytes(b"video")
            model_i.write_bytes(b"i")
            model_p.write_bytes(b"p")
            repo_dir.mkdir(parents=True, exist_ok=True)

            fake_roi = {
                "bitstream_bytes": b"roi",
                "meta": {"frames_encoded": 2, "frame_index_map": [2, 6], "device": "cuda"},
            }
            fake_bg = {
                "bitstream_bytes": b"bg",
                "meta": {"frames_encoded": 1, "frame_index_map": [4], "device": "cuda"},
            }
            fake_cache = {
                "roi_store": None,
                "roi_indices": [2, 5],
                "roi_count": 2,
                "bg_store": None,
                "bg_indices": [0],
                "bg_count": 1,
            }

            with patch("src.compression.phase4_dcvc.probe_video", return_value=VideoInfo(32, 24, 30.0, 10)):
                with patch("src.compression.phase4_dcvc._capture_rendered_kept_frames_single_pass", return_value=fake_cache):
                    with patch(
                        "src.compression.phase4_dcvc.encode_dcvc_frames_to_bytes",
                        side_effect=[fake_roi, fake_bg],
                    ):
                        out = compress_keep_streams_dcvc(
                            source_video_path=video_path,
                            roi_bbox_map={},
                            frame_drop_result={"roi_kept_frames": [2, 5], "bg_kept_frames": [0]},
                            compression_cfg={
                                "dcvc": {
                                    "repo_dir": str(repo_dir),
                                    "model_i": str(model_i),
                                    "model_p": str(model_p),
                                    "use_cuda": True,
                                },
                                "quality": {"roi_qp_i": 1, "roi_qp_p": 1, "bg_qp_i": 1, "bg_qp_p": 1},
                                "roi": {"min_conf": 0.5, "dilate_px": 6},
                            },
                            root_dir=ROOT,
                        )

        self.assertEqual(out["meta"]["streams"]["roi"]["frame_index_map"], [2, 6])
        self.assertEqual(out["meta"]["streams"]["bg"]["frame_index_map"], [4])
        self.assertEqual(out["meta"]["roi"]["min_conf"], 0.5)
        self.assertEqual(out["meta"]["roi"]["dilate_px"], 0)
        self.assertEqual(out["meta"]["roi"]["visible_dilate_px"], 0)
        self.assertEqual(out["meta"]["roi"]["requested_dilate_px"], 6)

    def test_phase4_rejects_inconsistent_encoder_frame_index_map(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = tmp / "video.mp4"
            model_i = tmp / "i.pth.tar"
            model_p = tmp / "p.pth.tar"
            repo_dir = tmp / "DCVC"

            video_path.write_bytes(b"video")
            model_i.write_bytes(b"i")
            model_p.write_bytes(b"p")
            repo_dir.mkdir(parents=True, exist_ok=True)

            fake_roi = {
                "bitstream_bytes": b"roi",
                "meta": {"frames_encoded": 2, "frame_index_map": [2], "device": "cuda"},
            }
            fake_bg = {
                "bitstream_bytes": b"bg",
                "meta": {"frames_encoded": 1, "frame_index_map": [4], "device": "cuda"},
            }
            fake_cache = {
                "roi_store": None,
                "roi_indices": [2, 5],
                "roi_count": 2,
                "bg_store": None,
                "bg_indices": [0],
                "bg_count": 1,
            }

            with patch("src.compression.phase4_dcvc.probe_video", return_value=VideoInfo(32, 24, 30.0, 10)):
                with patch("src.compression.phase4_dcvc._capture_rendered_kept_frames_single_pass", return_value=fake_cache):
                    with patch(
                        "src.compression.phase4_dcvc.encode_dcvc_frames_to_bytes",
                        side_effect=[fake_roi, fake_bg],
                    ):
                        with self.assertRaises(RuntimeError):
                            compress_keep_streams_dcvc(
                                source_video_path=video_path,
                                roi_bbox_map={},
                                frame_drop_result={"roi_kept_frames": [2, 5], "bg_kept_frames": [0]},
                                compression_cfg={
                                    "dcvc": {
                                        "repo_dir": str(repo_dir),
                                        "model_i": str(model_i),
                                        "model_p": str(model_p),
                                        "use_cuda": True,
                                    },
                                    "quality": {"roi_qp_i": 1, "roi_qp_p": 1, "bg_qp_i": 1, "bg_qp_p": 1},
                                    "roi": {"min_conf": 0.5},
                                },
                                root_dir=ROOT,
                            )

    def test_phase4_reads_source_video_once_for_roi_and_bg_streams(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = tmp / "video.mp4"
            model_i = tmp / "i.pth.tar"
            model_p = tmp / "p.pth.tar"
            repo_dir = tmp / "DCVC"

            video_path.write_bytes(b"video")
            model_i.write_bytes(b"i")
            model_p.write_bytes(b"p")
            repo_dir.mkdir(parents=True, exist_ok=True)

            frames = [
                np.full((4, 6, 3), 10, dtype=np.uint8),
                np.full((4, 6, 3), 20, dtype=np.uint8),
                np.full((4, 6, 3), 30, dtype=np.uint8),
            ]
            capture_opens = []

            class _FakeCapture:
                def __init__(self, frames_in):
                    self._frames = [f.copy() for f in frames_in]
                    self._pos = 0
                    capture_opens.append(1)

                def isOpened(self):
                    return True

                def read(self):
                    if self._pos >= len(self._frames):
                        return False, None
                    frame = self._frames[self._pos]
                    self._pos += 1
                    return True, frame.copy()

                def release(self):
                    return None

            def _fake_encode(frame_iter, *, info, cfg, video_path):
                items = list(frame_iter)
                return {
                    "bitstream_bytes": video_path.encode("utf-8"),
                    "meta": {
                        "frames_encoded": len(items),
                        "frame_index_map": [idx for idx, _ in items],
                        "device": "cuda:0",
                    },
                }

            with patch("src.compression.phase4_dcvc.probe_video", return_value=VideoInfo(6, 4, 30.0, 3)):
                with patch("src.compression.phase4_dcvc.cv2.VideoCapture", side_effect=lambda *_: _FakeCapture(frames)):
                    with patch("src.compression.phase4_dcvc.encode_dcvc_frames_to_bytes", side_effect=_fake_encode):
                        out = compress_keep_streams_dcvc(
                            source_video_path=video_path,
                            roi_bbox_map={},
                            frame_drop_result={"roi_kept_frames": [0, 2], "bg_kept_frames": [1, 2]},
                            compression_cfg={
                                "dcvc": {
                                    "repo_dir": str(repo_dir),
                                    "model_i": str(model_i),
                                    "model_p": str(model_p),
                                    "use_cuda": True,
                                },
                                "quality": {"roi_qp_i": 1, "roi_qp_p": 1, "bg_qp_i": 1, "bg_qp_p": 1},
                                "roi": {"min_conf": 0.0},
                            },
                            root_dir=ROOT,
                        )

        self.assertEqual(len(capture_opens), 1)
        self.assertEqual(out["meta"]["streams"]["roi"]["frame_index_map"], [0, 2])
        self.assertEqual(out["meta"]["streams"]["bg"]["frame_index_map"], [1, 2])


if __name__ == "__main__":
    unittest.main()
