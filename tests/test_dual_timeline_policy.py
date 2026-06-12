from __future__ import annotations

import unittest
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frame_removal import apply_dual_timeline_policy, build_dual_timeline_metadata  # noqa: E402


class TestDualTimelinePolicy(unittest.TestCase):
    def test_fixed_policy_resets_background_cadence_when_detection_regime_changes(self) -> None:
        frame_result = {
            "per_frame": {
                "0": {"state": "STILL", "bbox_missing": True, "roi_count": 0, "track_births": 0},
                "1": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "track_births": 1},
                "2": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "track_births": 0},
                "3": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "track_births": 0},
                "4": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "track_births": 0},
                "5": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "track_births": 0},
                "6": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "track_births": 0},
                "7": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "track_births": 0},
                "8": {"state": "STILL", "bbox_missing": True, "roi_count": 0, "track_births": 0},
                "9": {"state": "STILL", "bbox_missing": True, "roi_count": 0, "track_births": 0},
                "10": {"state": "STILL", "bbox_missing": True, "roi_count": 0, "track_births": 0},
                "11": {"state": "MOTION", "bbox_missing": False, "roi_count": 1, "track_births": 1},
                "12": {"state": "MOTION", "bbox_missing": False, "roi_count": 1, "track_births": 0},
                "13": {"state": "MOTION", "bbox_missing": False, "roi_count": 1, "track_births": 0},
            },
            "stats": {"num_frames_read": 14, "width": 1280, "height": 720},
        }
        frame_cfg = {
            "dual_timeline": {
                "enable": True,
                "roi_motion_interval": 2,
                "roi_still_interval": 3,
                "bg_interval": 6,
                "bg_idle_interval": 4,
            }
        }

        dual = build_dual_timeline_metadata(frame_result, frame_cfg)

        self.assertEqual(dual["policy"]["roi_mode"], "fixed_state_interval")
        self.assertEqual(dual["policy"]["bg_mode"], "fixed_segment_interval")
        self.assertEqual(dual["roi_kept_frames"], [1, 4, 7, 11, 13])
        self.assertEqual(dual["bg_kept_frames"], [0, 1, 7, 8, 11])

    def test_apply_dual_timeline_policy_updates_top_level_lists_and_per_frame_flags(self) -> None:
        frame_result = {
            "kept_frames": [1, 4, 7, 11, 13],
            "dropped_frames": [0, 2, 3, 5, 6, 8, 9, 10, 12],
            "per_frame": {
                "0": {"state": "STILL", "bbox_missing": True, "roi_count": 0, "keep": False},
                "1": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "keep": True},
                "2": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "keep": False},
                "3": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "keep": False},
                "4": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "keep": True},
                "5": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "keep": False},
                "6": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "keep": False},
                "7": {"state": "STILL", "bbox_missing": False, "roi_count": 1, "keep": True},
                "8": {"state": "STILL", "bbox_missing": True, "roi_count": 0, "keep": False},
                "9": {"state": "STILL", "bbox_missing": True, "roi_count": 0, "keep": False},
                "10": {"state": "STILL", "bbox_missing": True, "roi_count": 0, "keep": False},
                "11": {"state": "MOTION", "bbox_missing": False, "roi_count": 1, "keep": True},
                "12": {"state": "MOTION", "bbox_missing": False, "roi_count": 1, "keep": False},
                "13": {"state": "MOTION", "bbox_missing": False, "roi_count": 1, "keep": True},
            },
            "stats": {"num_frames_read": 14},
        }
        frame_cfg = {
            "dual_timeline": {
                "enable": True,
                "roi_motion_interval": 2,
                "roi_still_interval": 3,
                "bg_interval": 6,
                "bg_idle_interval": 4,
            }
        }

        out = apply_dual_timeline_policy(frame_result, frame_cfg)

        self.assertEqual(out["policy_mode"], "dual_only")
        self.assertEqual(out["roi_kept_frames"], [1, 4, 7, 11, 13])
        self.assertEqual(out["bg_kept_frames"], [0, 1, 7, 8, 11])
        self.assertFalse(out["per_frame"]["0"]["keep"])
        self.assertTrue(out["per_frame"]["0"]["bg_keep"])
        self.assertTrue(out["per_frame"]["1"]["keep"])
        self.assertTrue(out["per_frame"]["1"]["bg_keep"])
        self.assertFalse(out["per_frame"]["8"]["keep"])
        self.assertTrue(out["per_frame"]["8"]["bg_keep"])


if __name__ == "__main__":
    unittest.main()
