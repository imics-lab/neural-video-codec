# Model Distribution

Do not rely on committing large model binaries directly to Git history.

This project uses `models/models.manifest.json` as the source of truth for model files:

- `MDV6-yolov9-c.pt`
- `cvpr2025_image.pth.tar`
- `cvpr2025_video.pth.tar`

## Consumer workflow

1. Run installer (`install.bat` on Windows or `install.sh` on Linux).
2. Place model files in this `models/` folder using filenames from `models.manifest.json`.
3. Verify SHA256 checksums against `models.manifest.json`.

## Maintainer workflow

1. Upload model artifacts to your host of choice:
   - GitHub Release assets, Hugging Face, S3, GCS, etc.
2. Update `models/models.manifest.json`:
   - set `base_url` and/or per-model `url`
   - keep correct `sha256`
   - keep correct `size_bytes`
3. Commit manifest changes.

This keeps repository clone size small while preserving deterministic model bootstrap.
