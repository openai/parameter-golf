from __future__ import annotations

from dataclasses import replace

from .config import (
    DelayLineConfig,
    HierarchicalSubstrateConfig,
    MixedMemoryConfig,
    OpenPredictiveCoderConfig,
    OscillatoryMemoryConfig,
    ReservoirConfig,
    SubstrateKind,
)
from .delay import DelayLineSubstrate
from .hierarchical import HierarchicalSubstrate
from .mixed_memory import MixedMemorySubstrate
from .oscillatory_memory import OscillatoryMemorySubstrate
from .reservoir import EchoStateReservoir
from .substrates import TokenSubstrate


def create_echo_state_substrate(
    config: ReservoirConfig | None = None,
    *,
    vocabulary_size: int = 256,
) -> EchoStateReservoir:
    return EchoStateReservoir(
        config=config or ReservoirConfig(),
        vocabulary_size=vocabulary_size,
    )


def create_delay_line_substrate(config: DelayLineConfig | None = None) -> DelayLineSubstrate:
    return DelayLineSubstrate(config=config)


def create_oscillatory_memory_substrate(
    config: OscillatoryMemoryConfig | None = None,
) -> OscillatoryMemorySubstrate:
    return OscillatoryMemorySubstrate(config=config)


def create_mixed_memory_substrate(config: MixedMemoryConfig | None = None) -> MixedMemorySubstrate:
    return MixedMemorySubstrate(config=config)


def create_hierarchical_substrate(
    config: HierarchicalSubstrateConfig | None = None,
) -> HierarchicalSubstrate:
    return HierarchicalSubstrate(config=config)


def create_substrate_for_model(config: OpenPredictiveCoderConfig) -> TokenSubstrate:
    if config.substrate_kind == "echo_state":
        return create_echo_state_substrate(
            config.reservoir,
            vocabulary_size=config.vocabulary_size,
        )
    if config.substrate_kind == "delay":
        return create_delay_line_substrate(
            replace(config.delay, vocabulary_size=config.vocabulary_size),
        )
    if config.substrate_kind == "oscillatory":
        return create_oscillatory_memory_substrate(
            replace(config.oscillatory, vocabulary_size=config.vocabulary_size),
        )
    if config.substrate_kind == "mixed_memory":
        return create_mixed_memory_substrate(
            replace(
                config.mixed_memory,
                delay=replace(config.mixed_memory.delay, vocabulary_size=config.vocabulary_size),
            )
        )
    if config.substrate_kind == "hierarchical":
        return create_hierarchical_substrate(
            replace(config.hierarchical, vocabulary_size=config.vocabulary_size),
        )
    raise ValueError(f"Unknown substrate_kind: {config.substrate_kind}")


def create_substrate(config: object | None = None) -> TokenSubstrate:
    if config is None or isinstance(config, ReservoirConfig):
        return create_echo_state_substrate(config)
    if isinstance(config, DelayLineConfig):
        return create_delay_line_substrate(config)
    if isinstance(config, OscillatoryMemoryConfig):
        return create_oscillatory_memory_substrate(config)
    if isinstance(config, MixedMemoryConfig):
        return create_mixed_memory_substrate(config)
    if isinstance(config, HierarchicalSubstrateConfig):
        return create_hierarchical_substrate(config)
    if isinstance(config, OpenPredictiveCoderConfig):
        return create_substrate_for_model(config)
    raise TypeError(f"Unsupported substrate config: {type(config)!r}")


__all__ = [
    "SubstrateKind",
    "create_delay_line_substrate",
    "create_echo_state_substrate",
    "create_hierarchical_substrate",
    "create_mixed_memory_substrate",
    "create_oscillatory_memory_substrate",
    "create_substrate",
    "create_substrate_for_model",
]
