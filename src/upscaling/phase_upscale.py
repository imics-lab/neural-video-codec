"""
UpscalingPhase — orchestrates the full video upscaling pipeline phase.

Iterates frames via FrameExtractor, calls Upscaler per frame,
assembles output with VideoAssembler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from .upscaler import Upscaler
from preprocessing.frame_extractor import FrameExtractor, load_roi_data, create_weight_mask
from postprocessing.video_assembler import VideoAssembler


class UpscalingPhase:
    """
    Runs the upscaling phase end-to-end.

    Args:
        config: UpscalingConfig dataclass (from src.pipeline.config_schema).
    """

    def __init__(self, config) -> None:
        self.config = config
        self.upscaler = Upscaler(config)

    def run(
        self,
        input_video: str,
        roi_json: Optional[str],
        output_video: str,
    ) -> None:
        """
        Upscale every frame of input_video and write to output_video.

        Args:
            input_video:  Path to the LR input video.
            roi_json:     Optional path to ROI detections JSON. If None, a
                          uniform weight mask is used for all frames.
            output_video: Path to write the upscaled output video.
        """
        cfg = self.config

        roi_data: dict = {}
        if roi_json is not None:
            roi_data = load_roi_data(roi_json)
            logger.info(f"Loaded ROI data: {len(roi_data)} keyed frames from {roi_json}")
        else:
            logger.info("No ROI JSON provided — using uniform weight mask for all frames.")

        Path(output_video).parent.mkdir(parents=True, exist_ok=True)

        with FrameExtractor(input_video) as extractor:
            out_w = extractor.width  * cfg.scale
            out_h = extractor.height * cfg.scale
            total = extractor.num_frames

            logger.info(
                f"Input:  {extractor.width}x{extractor.height} @ {extractor.fps:.2f} fps "
                f"({total} frames)"
            )
            logger.info(f"Output: {out_w}x{out_h} -> {output_video}")

            with VideoAssembler(output_video, extractor.fps, out_w, out_h) as assembler:
                for frame_idx, frame_bgr in extractor:
                    h, w = frame_bgr.shape[:2]
                    weight_mask = create_weight_mask(
                        frame_idx=frame_idx,
                        roi_data=roi_data,
                        h=h,
                        w=w,
                        default_weight=cfg.roi.default_weight,
                        padding=cfg.roi.padding,
                    )
                    upscaled = self.upscaler.upscale_frame(frame_bgr, weight_mask)
                    assembler.write(upscaled)

                    if (frame_idx + 1) % 25 == 0 or frame_idx == 0:
                        logger.info(f"  {frame_idx + 1}/{total} frames processed")

        logger.info(f"Upscaling complete. Saved: {output_video}")
