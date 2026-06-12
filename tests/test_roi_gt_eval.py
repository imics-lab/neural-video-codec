from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "roi_detection"))

import run_gt_eval as gte  # noqa: E402


class TestRoiGtEval(unittest.TestCase):
    def test_parse_cvat_video_xml_interpolates_track_boxes(self) -> None:
        xml_text = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <task>
      <size>5</size>
    </task>
  </meta>
  <track id="0" label="animal" source="manual">
    <box frame="0" xtl="0" ytl="0" xbr="10" ybr="10" outside="0" occluded="0" keyframe="1" />
    <box frame="2" xtl="10" ytl="0" xbr="20" ybr="10" outside="0" occluded="0" keyframe="1" />
    <box frame="4" xtl="20" ytl="0" xbr="30" ybr="10" outside="1" occluded="0" keyframe="1" />
  </track>
</annotations>
"""
        with tempfile.TemporaryDirectory() as td:
            gt_path = Path(td) / "video.xml"
            gt_path.write_text(xml_text, encoding="utf-8")
            parsed = gte._parse_cvat_xml(gt_path, labels_filter=None, fallback_total_frames=None)

        self.assertEqual(parsed.source_format, "cvat_xml_video")
        self.assertEqual(parsed.total_frames, 5)
        self.assertEqual(parsed.evaluated_frames, [0, 1, 2, 3, 4])
        self.assertEqual(sorted(parsed.frame_map.keys()), [0, 1, 2, 3])
        self.assertAlmostEqual(parsed.frame_map[0][0].x1, 0.0)
        self.assertAlmostEqual(parsed.frame_map[1][0].x1, 5.0)
        self.assertAlmostEqual(parsed.frame_map[2][0].x1, 10.0)
        self.assertAlmostEqual(parsed.frame_map[3][0].x1, 15.0)

    def test_compare_pred_vs_gt_perfect_overlap_scores_full_credit(self) -> None:
        gt_data = gte.GroundTruthData(
            source_path=Path("dummy.xml"),
            source_format="cvat_xml_video",
            frame_map={0: [gte.Box(0.0, 0.0, 10.0, 10.0, "animal")]},
            evaluated_frames=[0],
            total_frames=1,
            labels_seen=["animal"],
        )
        pred_frame_map = {0: [gte.Box(0.0, 0.0, 10.0, 10.0, "animal")]}

        metrics = gte._compare_pred_vs_gt(gt_data, pred_frame_map, width=32, height=32, fallback_total_frames=1)

        self.assertEqual(metrics["gt_presence_frames"], 1.0)
        self.assertEqual(metrics["pred_presence_frames"], 1.0)
        self.assertEqual(metrics["presence_recall_pct"], 100.0)
        self.assertEqual(metrics["presence_precision_pct"], 100.0)
        self.assertEqual(metrics["mean_gt_coverage_pct"], 100.0)
        self.assertEqual(metrics["coverage_ge_90_pct"], 100.0)
        self.assertEqual(metrics["mean_iou_on_gt_frames"], 1.0)
        self.assertEqual(metrics["mean_roi_efficiency_pct"], 100.0)
        self.assertEqual(metrics["mean_instance_coverage_pct"], 100.0)
        self.assertEqual(metrics["instance_coverage_ge_90_pct"], 100.0)
        self.assertEqual(metrics["mean_best_iou_per_instance"], 1.0)
        self.assertEqual(metrics["frames_all_instances_covered_ge_90_pct"], 100.0)

    def test_compare_pred_vs_gt_multi_instance_metrics_penalize_missing_second_animal(self) -> None:
        gt_data = gte.GroundTruthData(
            source_path=Path("dummy.xml"),
            source_format="cvat_xml_video",
            frame_map={
                0: [
                    gte.Box(0.0, 0.0, 10.0, 10.0, "animal"),
                    gte.Box(20.0, 0.0, 30.0, 10.0, "animal"),
                ]
            },
            evaluated_frames=[0],
            total_frames=1,
            labels_seen=["animal"],
        )
        pred_frame_map = {0: [gte.Box(0.0, 0.0, 10.0, 10.0, "animal")]}

        metrics = gte._compare_pred_vs_gt(gt_data, pred_frame_map, width=64, height=32, fallback_total_frames=1)

        self.assertEqual(metrics["presence_recall_pct"], 100.0)
        self.assertEqual(metrics["mean_gt_coverage_pct"], 50.0)
        self.assertEqual(metrics["gt_instance_count"], 2.0)
        self.assertEqual(metrics["gt_instance_miss_count"], 1.0)
        self.assertEqual(metrics["mean_instance_coverage_pct"], 50.0)
        self.assertEqual(metrics["instance_coverage_ge_50_pct"], 50.0)
        self.assertEqual(metrics["instance_coverage_ge_90_pct"], 50.0)
        self.assertEqual(metrics["mean_best_iou_per_instance"], 0.5)
        self.assertEqual(metrics["instance_best_iou_ge_50_pct"], 50.0)
        self.assertEqual(metrics["frames_all_instances_covered_ge_90_pct"], 0.0)

    def test_parse_yolo_labeled_frames_uses_train_split_and_normalized_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "video_gt"
            labels_dir = root / "labels" / "Train"
            labels_dir.mkdir(parents=True)
            (root / "data.yaml").write_text(
                "Train: Train.txt\nnames:\n  0: animal\npath: .\n",
                encoding="utf-8",
            )
            (root / "Train.txt").write_text(
                "data/images/Train/frame_000001.png\n"
                "data/images/Train/frame_000003.png\n",
                encoding="utf-8",
            )
            (labels_dir / "frame_000001.txt").write_text("0 0.500000 0.500000 0.200000 0.400000\n", encoding="utf-8")

            parsed = gte._load_ground_truth(root, labels_filter=None, fallback_total_frames=10, image_width=100, image_height=50)

        self.assertEqual(parsed.source_format, "yolo_txt_frames")
        self.assertEqual(parsed.total_frames, 10)
        self.assertEqual(parsed.evaluated_frames, [1, 3])
        self.assertEqual(parsed.labels_seen, ["animal"])
        self.assertEqual(sorted(parsed.frame_map.keys()), [1])
        self.assertAlmostEqual(parsed.frame_map[1][0].x1, 40.0)
        self.assertAlmostEqual(parsed.frame_map[1][0].y1, 15.0)
        self.assertAlmostEqual(parsed.frame_map[1][0].x2, 60.0)
        self.assertAlmostEqual(parsed.frame_map[1][0].y2, 35.0)


if __name__ == "__main__":
    unittest.main()
