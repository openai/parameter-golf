from __future__ import annotations

from .config import (
    DelayLineConfig,
    HierarchicalSubstrateConfig,
    LatentConfig,
    MixedMemoryConfig,
    OpenPredictiveCoderConfig,
    ReservoirConfig,
    SegmenterConfig,
)


def echo_state_small() -> OpenPredictiveCoderConfig:
    return OpenPredictiveCoderConfig(
        substrate_kind="echo_state",
        segmenter=SegmenterConfig(mode="adaptive", patch_size=4, min_patch_size=2, max_patch_size=8, novelty_threshold=0.08),
        reservoir=ReservoirConfig(size=96, connectivity=0.12, spectral_radius=0.9, leak=0.35, seed=11),
        latent=LatentConfig(latent_dim=24, global_dim=24, reservoir_features=24, readout_l2=1e-5),
    )


def delay_small() -> OpenPredictiveCoderConfig:
    return OpenPredictiveCoderConfig(
        substrate_kind="delay",
        segmenter=SegmenterConfig(mode="adaptive", patch_size=4, min_patch_size=2, max_patch_size=8, novelty_threshold=0.08),
        delay=DelayLineConfig(history_length=12, embedding_dim=16, vocabulary_size=256, input_scale=0.2, decay=0.95, seed=11),
        latent=LatentConfig(latent_dim=12, global_dim=12, reservoir_features=12, readout_l2=1e-5),
    )


def mixed_memory_small() -> OpenPredictiveCoderConfig:
    return OpenPredictiveCoderConfig(
        substrate_kind="mixed_memory",
        segmenter=SegmenterConfig(mode="adaptive", patch_size=4, min_patch_size=2, max_patch_size=8, novelty_threshold=0.08),
        mixed_memory=MixedMemoryConfig(
            reservoir=ReservoirConfig(size=64, connectivity=0.15, spectral_radius=0.9, leak=0.3, seed=13),
            delay=DelayLineConfig(history_length=8, embedding_dim=8, vocabulary_size=256, input_scale=0.2, decay=0.95, seed=13),
        ),
        latent=LatentConfig(latent_dim=16, global_dim=16, reservoir_features=16, readout_l2=1e-5),
    )


def hierarchical_small() -> OpenPredictiveCoderConfig:
    return OpenPredictiveCoderConfig(
        substrate_kind="hierarchical",
        segmenter=SegmenterConfig(mode="adaptive", patch_size=4, min_patch_size=2, max_patch_size=8, novelty_threshold=0.08),
        hierarchical=HierarchicalSubstrateConfig(
            fast_size=24,
            mid_size=32,
            slow_size=40,
            vocabulary_size=256,
            fast_connectivity=0.2,
            mid_connectivity=0.12,
            slow_connectivity=0.08,
            fast_spectral_radius=0.8,
            mid_spectral_radius=0.9,
            slow_spectral_radius=0.95,
            fast_leak=0.4,
            mid_leak=0.3,
            slow_leak=0.2,
            input_scale=0.15,
            upward_scale=0.08,
            slow_update_stride=2,
            seed=17,
        ),
        latent=LatentConfig(latent_dim=24, global_dim=24, reservoir_features=24, readout_l2=1e-5),
    )


__all__ = [
    "delay_small",
    "echo_state_small",
    "hierarchical_small",
    "mixed_memory_small",
]
