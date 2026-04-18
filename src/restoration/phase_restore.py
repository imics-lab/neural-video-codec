"""
Restoration phase orchestrator — loads config, instantiates Restorer, runs sequence.

Public API:
    restore_frames(frames, cfg) -> List[np.ndarray]
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf


def restore_frames(
    frames: List[np.ndarray],
    cfg: Dict[str, Any],
    progress_cb: Optional[Callable[[int], None]] = None,
    detections: Optional[Dict[str, Any]] = None,
    width: int = 0,
    height: int = 0,
) -> List[np.ndarray]:
    """
    Apply Model R restoration, then blend ROI and BG with different strengths.

    ROI pixels: strength * restored + (1 - strength) * decompressed
    BG pixels:  fully restored

    Args:
        frames:      List of BGR uint8 arrays (from DCVC decompression).
        cfg:         Restoration config dict.
        detections:  ROI detection dict from archive {frame_idx -> [boxes]}.
        width/height: Frame dimensions for mask building.
    """
    from .restorer import Restorer

    cfg_node = OmegaConf.create(cfg)
    restorer = Restorer(cfg_node)
    restored = restorer.restore_sequence(frames)

    # Blend: ROI gets partial restoration, BG gets full restoration
    if detections and width > 0 and height > 0:
        from ..roi_masking.roi_masking import build_frame_mask, mask_to_alpha
        roi_strength = float(cfg.get("inference", {}).get("roi_restore_strength", 0.3))
        result = []
        for fi, (decomp_f, rest_f) in enumerate(zip(frames, restored)):
            mask  = build_frame_mask(frame_idx=fi, width=width, height=height,
                                     roi_map=detections, min_conf=0.0, dilate_px=0)
            alpha = mask_to_alpha(mask, feather_px=8)[..., None]  # (H, W, 1)
            # alpha=1 in ROI, 0 in BG
            # ROI: lerp(decompressed, restored, roi_strength)
            # BG:  fully restored
            blend_weight = alpha * roi_strength + (1.0 - alpha) * 1.0
            out = np.clip(
                decomp_f.astype(np.float32) * (1.0 - blend_weight)
                + rest_f.astype(np.float32) * blend_weight,
                0, 255
            ).astype(np.uint8)
            result.append(out)
    else:
        result = restored

    if progress_cb is not None:
        for _ in result:
            progress_cb(1)

    return result
