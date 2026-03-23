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
    from .restorer import Restorer

    cfg_node = OmegaConf.create(cfg)
    restorer = Restorer(cfg_node)

    restored = restorer.restore_sequence(frames)

    if progress_cb is not None:
        for _ in restored:
            progress_cb(1)

    return restored
