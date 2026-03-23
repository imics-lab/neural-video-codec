"""
GaussianDiffusion for the restoration model — identical maths to upscaling/_diffusion.py
but kept separate so the two models can diverge independently.
"""
from __future__ import annotations

# Re-export from the upscaling module to avoid duplication.
# If the restoration model needs different schedule parameters in future,
# copy the implementation here and customise.
from ..upscaling._diffusion import GaussianDiffusion  # noqa: F401

__all__ = ["GaussianDiffusion"]
