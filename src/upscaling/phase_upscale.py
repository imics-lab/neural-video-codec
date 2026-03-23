"""
Upscaling phase orchestrator — loads config, instantiates Upscaler, runs sequence.

Public API:
    upscale_frames(frames, cfg) -> List[np.ndarray]
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf


def upscale_frames(
    frames: List[np.ndarray],
    cfg: Dict[str, Any],
    progress_cb: Optional[Callable[[int], None]] = None,
) -> List[np.ndarray]:
    """
    Apply Model S (diffusion 2× super-resolution) to a sequence.

    Args:
        frames:      List of BGR uint8 arrays (restored, from Model R output).
        cfg:         Upscaling config dict.
        progress_cb: Called with 1 after each upscaled frame.

    Returns:
        List of 2× upscaled BGR uint8 arrays, same length.
    """
    from .upscaler import Upscaler

    cfg_node = OmegaConf.create(cfg)
    upscaler = Upscaler(cfg_node)

    result = []
    for frame in frames:
        result.append(upscaler.upscale_frame(frame))
        if progress_cb is not None:
            progress_cb(1)

    return result
