from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.roi_masking import compose_soft, mask_to_alpha  # noqa: E402


class TestRoiMasking(unittest.TestCase):
    def test_mask_to_alpha_feathers_boundary(self) -> None:
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 255

        alpha = mask_to_alpha(mask, feather_px=2)

        self.assertEqual(float(alpha[0, 0]), 0.0)
        self.assertEqual(float(alpha[3, 3]), 1.0)
        self.assertGreater(float(alpha[1, 3]), 0.0)
        self.assertLess(float(alpha[1, 3]), 1.0)

    def test_compose_soft_blends_only_on_roi_edge(self) -> None:
        bg = np.full((7, 7, 3), 20, dtype=np.uint8)
        roi = np.full((7, 7, 3), 220, dtype=np.uint8)
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 255
        alpha = mask_to_alpha(mask, feather_px=2)

        out = compose_soft(roi, bg, alpha)

        np.testing.assert_array_equal(out[0, 0], bg[0, 0])
        np.testing.assert_array_equal(out[3, 3], roi[3, 3])
        self.assertTrue(np.all(out[1, 3] > bg[1, 3]))
        self.assertTrue(np.all(out[1, 3] < roi[1, 3]))


if __name__ == "__main__":
    unittest.main()
