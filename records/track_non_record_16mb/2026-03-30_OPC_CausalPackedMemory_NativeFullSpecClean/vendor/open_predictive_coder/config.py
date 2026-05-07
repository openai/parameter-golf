from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SegmenterMode = Literal["fixed", "adaptive"]
ReservoirTopology = Literal["erdos_renyi", "small_world"]
MemoryMergeMode = Literal["concatenate"]
SubstrateKind = Literal["echo_state", "delay", "mixed_memory", "hierarchical", "oscillatory"]


@dataclass(frozen=True)
class SegmenterConfig:
    mode: SegmenterMode = "adaptive"
    patch_size: int = 8
    min_patch_size: int = 4
    max_patch_size: int = 24
    novelty_threshold: float = 0.14

    def __post_init__(self) -> None:
        if self.patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if self.min_patch_size < 1:
            raise ValueError("min_patch_size must be >= 1")
        if self.max_patch_size < self.min_patch_size:
            raise ValueError("max_patch_size must be >= min_patch_size")
        if self.patch_size < self.min_patch_size or self.patch_size > self.max_patch_size:
            raise ValueError("patch_size must lie within [min_patch_size, max_patch_size]")
        if self.novelty_threshold < 0.0:
            raise ValueError("novelty_threshold must be >= 0")


@dataclass(frozen=True)
class ReservoirConfig:
    size: int = 512
    connectivity: float = 0.05
    spectral_radius: float = 0.95
    leak: float = 0.25
    input_scale: float = 0.2
    topology: ReservoirTopology = "erdos_renyi"
    rewire_prob: float = 0.1
    seed: int = 7

    def __post_init__(self) -> None:
        if self.size < 8:
            raise ValueError("reservoir size must be >= 8")
        if not 0.0 < self.connectivity <= 1.0:
            raise ValueError("connectivity must lie in (0, 1]")
        if self.spectral_radius <= 0.0:
            raise ValueError("spectral_radius must be > 0")
        if not 0.0 < self.leak <= 1.0:
            raise ValueError("leak must lie in (0, 1]")
        if self.input_scale <= 0.0:
            raise ValueError("input_scale must be > 0")
        if not 0.0 <= self.rewire_prob <= 1.0:
            raise ValueError("rewire_prob must lie in [0, 1]")


@dataclass(frozen=True)
class DelayLineConfig:
    history_length: int = 16
    embedding_dim: int = 16
    vocabulary_size: int = 256
    input_scale: float = 0.2
    decay: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.input_scale <= 0.0:
            raise ValueError("input_scale must be > 0")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must lie in (0, 1]")

    @property
    def state_dim(self) -> int:
        return self.history_length * self.embedding_dim


@dataclass(frozen=True)
class LinearMemoryConfig:
    embedding_dim: int = 16
    vocabulary_size: int = 256
    decays: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.97)
    input_scale: float = 0.2
    seed: int = 7

    def __post_init__(self) -> None:
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if not self.decays:
            raise ValueError("decays must contain at least one bank")
        if any(decay <= 0.0 or decay >= 1.0 for decay in self.decays):
            raise ValueError("all decays must lie in (0, 1)")
        if self.input_scale <= 0.0:
            raise ValueError("input_scale must be > 0")

    @property
    def state_dim(self) -> int:
        return len(self.decays) * self.embedding_dim


@dataclass(frozen=True)
class OscillatoryMemoryConfig:
    vocabulary_size: int = 256
    embedding_dim: int = 16
    decay_rates: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)
    oscillatory_modes: int = 4
    oscillatory_damping_range: tuple[float, float] = (0.85, 0.98)
    oscillatory_period_range: tuple[float, float] = (4.0, 32.0)
    input_scale: float = 0.2
    seed: int = 7

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if not self.decay_rates:
            raise ValueError("decay_rates must contain at least one bank")
        if any(rate <= 0.0 or rate >= 1.0 for rate in self.decay_rates):
            raise ValueError("all decay_rates must lie in (0, 1)")
        if self.oscillatory_modes < 1:
            raise ValueError("oscillatory_modes must be >= 1")
        if len(self.oscillatory_damping_range) != 2:
            raise ValueError("oscillatory_damping_range must contain exactly two values")
        if len(self.oscillatory_period_range) != 2:
            raise ValueError("oscillatory_period_range must contain exactly two values")
        low_damping, high_damping = self.oscillatory_damping_range
        low_period, high_period = self.oscillatory_period_range
        if not 0.0 < low_damping < high_damping < 1.0:
            raise ValueError("oscillatory_damping_range must lie inside (0, 1)")
        if not 0.0 < low_period < high_period:
            raise ValueError("oscillatory_period_range must lie in positive increasing order")
        if self.input_scale <= 0.0:
            raise ValueError("input_scale must be > 0")

    @property
    def decay_bank_count(self) -> int:
        return len(self.decay_rates)

    @property
    def oscillatory_bank_count(self) -> int:
        return self.oscillatory_modes

    @property
    def state_dim(self) -> int:
        return (self.decay_bank_count * self.embedding_dim) + (2 * self.oscillatory_bank_count * self.embedding_dim)


@dataclass(frozen=True)
class MixedMemoryConfig:
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    delay: DelayLineConfig = field(default_factory=DelayLineConfig)
    merge_mode: MemoryMergeMode = "concatenate"

    def __post_init__(self) -> None:
        if self.merge_mode != "concatenate":
            raise ValueError("merge_mode must be 'concatenate'")

    @property
    def state_dim(self) -> int:
        return self.reservoir.size + self.delay.state_dim


@dataclass(frozen=True)
class HierarchicalSubstrateConfig:
    fast_size: int = 128
    vocabulary_size: int = 256
    fast_connectivity: float = 0.15
    fast_spectral_radius: float = 0.7
    fast_topology: ReservoirTopology = "erdos_renyi"
    fast_rewire_prob: float = 0.1
    fast_leak: float = 0.35
    mid_size: int = 256
    mid_connectivity: float = 0.08
    mid_spectral_radius: float = 0.9
    mid_topology: ReservoirTopology = "erdos_renyi"
    mid_rewire_prob: float = 0.1
    mid_leak: float = 0.25
    slow_size: int = 384
    slow_connectivity: float = 0.04
    slow_spectral_radius: float = 0.98
    slow_topology: ReservoirTopology = "erdos_renyi"
    slow_rewire_prob: float = 0.1
    slow_leak: float = 0.15
    input_scale: float = 0.2
    upward_scale: float = 0.1
    slow_update_stride: int = 1
    seed: int = 7

    def __post_init__(self) -> None:
        if self.fast_size < 4:
            raise ValueError("fast_size must be >= 4")
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.mid_size < 4:
            raise ValueError("mid_size must be >= 4")
        if self.slow_size < 4:
            raise ValueError("slow_size must be >= 4")
        if not 0.0 < self.fast_connectivity <= 1.0:
            raise ValueError("fast_connectivity must lie in (0, 1]")
        if not 0.0 < self.mid_connectivity <= 1.0:
            raise ValueError("mid_connectivity must lie in (0, 1]")
        if not 0.0 < self.slow_connectivity <= 1.0:
            raise ValueError("slow_connectivity must lie in (0, 1]")
        if self.fast_spectral_radius <= 0.0:
            raise ValueError("fast_spectral_radius must be > 0")
        if self.mid_spectral_radius <= 0.0:
            raise ValueError("mid_spectral_radius must be > 0")
        if self.slow_spectral_radius <= 0.0:
            raise ValueError("slow_spectral_radius must be > 0")
        if not 0.0 < self.fast_leak <= 1.0:
            raise ValueError("fast_leak must lie in (0, 1]")
        if not 0.0 < self.mid_leak <= 1.0:
            raise ValueError("mid_leak must lie in (0, 1]")
        if not 0.0 < self.slow_leak <= 1.0:
            raise ValueError("slow_leak must lie in (0, 1]")
        if self.input_scale <= 0.0:
            raise ValueError("input_scale must be > 0")
        if self.upward_scale <= 0.0:
            raise ValueError("upward_scale must be > 0")
        if self.slow_update_stride < 1:
            raise ValueError("slow_update_stride must be >= 1")

    @property
    def state_dim(self) -> int:
        return self.fast_size + self.mid_size + self.slow_size


@dataclass(frozen=True)
class SampledReadoutBandConfig:
    name: str
    start: int
    stop: int
    sample_count: int | None = None
    sample_indices: tuple[int, ...] = ()
    include_mean: bool = True
    include_energy: bool = True
    include_drift: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_indices", tuple(int(index) for index in self.sample_indices))
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.start < 0:
            raise ValueError("start must be >= 0")
        if self.stop <= self.start:
            raise ValueError("stop must be > start")
        if self.sample_count is not None and self.sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        if self.sample_indices and self.sample_count is not None:
            raise ValueError("sample_count and sample_indices are mutually exclusive")
        if len(set(self.sample_indices)) != len(self.sample_indices):
            raise ValueError("sample_indices must be unique")
        width = self.width
        if self.sample_count is not None and self.sample_count > width:
            raise ValueError("sample_count must be <= band width")
        if any(index < 0 or index >= width for index in self.sample_indices):
            raise ValueError("sample_indices must lie within the band width")

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def resolved_sample_count(self) -> int:
        if self.sample_indices:
            return len(self.sample_indices)
        if self.sample_count is not None:
            return self.sample_count
        return self.width

    @property
    def feature_dim(self) -> int:
        count = self.resolved_sample_count
        return count + int(self.include_mean) + int(self.include_energy) + int(self.include_drift)


@dataclass(frozen=True)
class SampledReadoutConfig:
    state_dim: int
    bands: tuple[SampledReadoutBandConfig, ...]
    seed: int = 7

    def __post_init__(self) -> None:
        object.__setattr__(self, "bands", tuple(self.bands))
        if self.state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if not self.bands:
            raise ValueError("bands must contain at least one band")
        for band in self.bands:
            if band.stop > self.state_dim:
                raise ValueError("band stop must be <= state_dim")

    @property
    def feature_dim(self) -> int:
        return sum(band.feature_dim for band in self.bands)


@dataclass(frozen=True)
class LatentConfig:
    latent_dim: int = 96
    global_dim: int = 96
    reservoir_features: int = 96
    bridge_scale: float = 0.25
    global_update_scale: float = 0.3
    readout_l2: float = 1e-3

    def __post_init__(self) -> None:
        if self.latent_dim < 4:
            raise ValueError("latent_dim must be >= 4")
        if self.global_dim < 4:
            raise ValueError("global_dim must be >= 4")
        if self.reservoir_features < 4:
            raise ValueError("reservoir_features must be >= 4")
        if self.bridge_scale <= 0.0:
            raise ValueError("bridge_scale must be > 0")
        if self.global_update_scale <= 0.0:
            raise ValueError("global_update_scale must be > 0")
        if self.readout_l2 < 0.0:
            raise ValueError("readout_l2 must be >= 0")


@dataclass(frozen=True)
class OpenPredictiveCoderConfig:
    vocabulary_size: int = 256
    substrate_kind: SubstrateKind = "echo_state"
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    delay: DelayLineConfig = field(default_factory=DelayLineConfig)
    oscillatory: OscillatoryMemoryConfig = field(default_factory=OscillatoryMemoryConfig)
    mixed_memory: MixedMemoryConfig = field(default_factory=MixedMemoryConfig)
    hierarchical: HierarchicalSubstrateConfig = field(default_factory=HierarchicalSubstrateConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.substrate_kind == "echo_state":
            substrate_state_dim = self.reservoir.size
        elif self.substrate_kind == "delay":
            substrate_state_dim = self.delay.state_dim
            if self.delay.vocabulary_size < self.vocabulary_size:
                raise ValueError("delay.vocabulary_size must be >= vocabulary_size")
        elif self.substrate_kind == "oscillatory":
            substrate_state_dim = self.oscillatory.state_dim
            if self.oscillatory.vocabulary_size < self.vocabulary_size:
                raise ValueError("oscillatory.vocabulary_size must be >= vocabulary_size")
        elif self.substrate_kind == "mixed_memory":
            substrate_state_dim = self.mixed_memory.state_dim
            if self.mixed_memory.delay.vocabulary_size < self.vocabulary_size:
                raise ValueError("mixed_memory.delay.vocabulary_size must be >= vocabulary_size")
        elif self.substrate_kind == "hierarchical":
            substrate_state_dim = self.hierarchical.state_dim
            if self.hierarchical.vocabulary_size < self.vocabulary_size:
                raise ValueError("hierarchical.vocabulary_size must be >= vocabulary_size")
        else:
            raise ValueError(f"Unknown substrate_kind: {self.substrate_kind}")

        if self.latent.reservoir_features > substrate_state_dim:
            raise ValueError("latent.reservoir_features must be <= chosen substrate state_dim")

    @property
    def feature_dim(self) -> int:
        return (
            self.latent.reservoir_features
            + self.latent.reservoir_features
            + self.latent.global_dim
            + self.latent.latent_dim
            + 3
        )


LatentControllerConfig = LatentConfig
ByteLatentPredictiveCoderConfig = OpenPredictiveCoderConfig
