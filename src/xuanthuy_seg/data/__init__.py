"""Spatial splitting primitives; heavy raster loaders live in ``data.cp2``."""

from .splits import GUARD, TRAIN, VALIDATION, build_split_mask, split_summary

__all__ = [
    "GUARD",
    "TRAIN",
    "VALIDATION",
    "build_split_mask",
    "split_summary",
]
