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
) -> List[np.ndarray]:
    """
    Apply Model R (temporal-attention diffusion restoration) to a sequence.

    Args:
        frames:      List of BGR uint8 arrays (degraded, from DCVC decompression).
        cfg:         Restoration config dict.
        progress_cb: Called with 1 after each restored frame.

    Returns:
        List of restored BGR uint8 arrays, same length.
    """
    import cv2

    from .restorer import Restorer

    cfg_node = OmegaConf.create(cfg)
    restorer = Restorer(cfg_node)

    input_scale = float(cfg.get("input_scale", 1.0))
    orig_size = None
    if input_scale < 1.0 and frames:
        h, w = frames[0].shape[:2]
        orig_size = (w, h)
        th, tw = int(h * input_scale), int(w * input_scale)
        frames = [cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA) for f in frames]

    restored = restorer.restore_sequence(frames)

    if orig_size is not None:
        restored = [cv2.resize(f, orig_size, interpolation=cv2.INTER_LINEAR) for f in restored]

    if progress_cb is not None:
        for _ in restored:
            progress_cb(1)

    return restored
