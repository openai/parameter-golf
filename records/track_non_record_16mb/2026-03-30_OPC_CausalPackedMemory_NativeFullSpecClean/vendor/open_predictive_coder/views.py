from __future__ import annotations

import numpy as np

from .config import LatentConfig
from .latents import LatentObservation


class ByteLatentFeatureView:
    def __init__(self, max_patch_size: int):
        if max_patch_size < 1:
            raise ValueError("max_patch_size must be >= 1")
        self.max_patch_size = max_patch_size

    @staticmethod
    def feature_dim(config: LatentConfig) -> int:
        return (
            config.reservoir_features
            + config.reservoir_features
            + config.global_dim
            + config.latent_dim
            + 3
        )

    def encode(self, observation: LatentObservation) -> np.ndarray:
        return np.concatenate(
            [
                observation.prediction_error,
                observation.patch_summary,
                observation.global_state,
                observation.latent,
                np.array(
                    [
                        observation.novelty,
                        observation.patch_length / self.max_patch_size,
                        1.0 if observation.boundary else 0.0,
                    ],
                    dtype=np.float64,
                ),
            ]
        )


__all__ = [
    "ByteLatentFeatureView",
]
