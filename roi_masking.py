"""
Root-level re-export shim. The canonical implementation lives in src/roi_masking/.
Scripts that insert src/ into sys.path import from there directly; this file exists
for scripts that run from the project root without the sys.path insertion.
"""
from src.roi_masking.roi_masking import (  # noqa: F401
    boxes_for_frame,
    build_boxes_mask,
    build_frame_mask,
    mask_to_alpha,
    compose_soft,
)

__all__ = [
    "boxes_for_frame",
    "build_boxes_mask",
    "build_frame_mask",
    "mask_to_alpha",
    "compose_soft",
]
