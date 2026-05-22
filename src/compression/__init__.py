from __future__ import annotations

from .phase4_dcvc import compress_keep_streams_dcvc
from .ffmpeg_encoder import encode_roi_bg_ffmpeg


def compress_keep_streams(
    *,
    source_video_path,
    roi_bbox_map,
    frame_drop_result,
    compression_cfg,
    root_dir,
):
    """Dispatch to DCVC or ffmpeg encoder based on compression_cfg['codec']."""
    codec = str(compression_cfg.get("codec", "dcvc")).lower()
    if codec == "dcvc":
        return compress_keep_streams_dcvc(
            source_video_path=source_video_path,
            roi_bbox_map=roi_bbox_map,
            frame_drop_result=frame_drop_result,
            compression_cfg=compression_cfg,
            root_dir=root_dir,
        )
    return encode_roi_bg_ffmpeg(
        source_video_path=source_video_path,
        roi_bbox_map=roi_bbox_map,
        frame_drop_result=frame_drop_result,
        compression_cfg=compression_cfg,
        root_dir=root_dir,
    )


__all__ = ["compress_keep_streams", "compress_keep_streams_dcvc", "encode_roi_bg_ffmpeg"]
