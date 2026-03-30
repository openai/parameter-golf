from __future__ import annotations

from dataclasses import dataclass

from .config import SegmenterConfig


@dataclass(frozen=True)
class SegmentStats:
    total_steps: int
    total_patches: int
    mean_patch_size: float
    compression_ratio: float


class AdaptiveSegmenter:
    def __init__(self, config: SegmenterConfig):
        self.config = config

    def should_commit(self, patch_length: int, novelty: float) -> bool:
        if patch_length >= self.config.max_patch_size:
            return True
        if patch_length < self.config.min_patch_size:
            return False
        if self.config.mode == "fixed":
            return patch_length >= self.config.patch_size
        return patch_length >= self.config.patch_size or novelty >= self.config.novelty_threshold

    @staticmethod
    def summarize(total_steps: int, total_patches: int) -> SegmentStats:
        patches = max(total_patches, 1)
        return SegmentStats(
            total_steps=total_steps,
            total_patches=total_patches,
            mean_patch_size=total_steps / patches,
            compression_ratio=total_steps / patches,
        )

