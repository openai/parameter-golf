from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SampledReadoutBandConfig, SampledReadoutConfig


@dataclass(frozen=True)
class SampledBandSummary:
    name: str
    indices: np.ndarray
    sampled: np.ndarray
    mean: float
    energy: float
    drift: float


class SampledMultiscaleReadout:
    def __init__(self, config: SampledReadoutConfig):
        self.config = config
        self._band_slices = tuple(slice(band.start, band.stop) for band in config.bands)
        self._band_indices = tuple(self._resolve_band_indices(index, band) for index, band in enumerate(config.bands))

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    @property
    def band_slices(self) -> tuple[slice, ...]:
        return self._band_slices

    @property
    def band_indices(self) -> tuple[np.ndarray, ...]:
        return self._band_indices

    def _coerce_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 1:
            raise ValueError("state must be rank-1")
        if state.shape[0] != self.config.state_dim:
            raise ValueError("state does not match configured state_dim")
        return state

    def _resolve_band_indices(self, band_index: int, band: SampledReadoutBandConfig) -> np.ndarray:
        if band.sample_indices:
            indices = np.asarray(band.sample_indices, dtype=np.int64)
        elif band.sample_count is None or band.sample_count == band.width:
            indices = np.arange(band.width, dtype=np.int64)
        else:
            rng = np.random.default_rng(self.config.seed + band_index)
            indices = np.sort(rng.choice(band.width, size=band.sample_count, replace=False)).astype(np.int64)
        return indices + band.start

    def split(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
        state = self._coerce_state(state)
        return tuple(state[band_slice] for band_slice in self._band_slices)

    def summaries(
        self,
        state: np.ndarray,
        previous_state: np.ndarray | None = None,
    ) -> tuple[SampledBandSummary, ...]:
        state = self._coerce_state(state)
        previous = None if previous_state is None else self._coerce_state(previous_state)
        if previous is not None and previous.shape != state.shape:
            raise ValueError("previous_state does not match state shape")

        summaries: list[SampledBandSummary] = []
        for band, band_slice, band_indices in zip(self.config.bands, self._band_slices, self._band_indices):
            band_state = state[band_slice]
            sampled = band_state[band_indices - band.start]
            mean = float(np.mean(band_state))
            energy = float(np.mean(np.square(band_state)))
            if band.include_drift:
                if previous is None:
                    drift = 0.0
                else:
                    drift = float(np.mean(np.abs(band_state - previous[band_slice])))
            else:
                drift = 0.0
            summaries.append(
                SampledBandSummary(
                    name=band.name,
                    indices=band_indices.copy(),
                    sampled=sampled.copy(),
                    mean=mean,
                    energy=energy,
                    drift=drift,
                )
            )
        return tuple(summaries)

    def encode(
        self,
        state: np.ndarray,
        previous_state: np.ndarray | None = None,
    ) -> np.ndarray:
        features: list[np.ndarray] = []
        for band, summary in zip(self.config.bands, self.summaries(state, previous_state=previous_state)):
            features.append(summary.sampled.astype(np.float64, copy=False))
            if band.include_mean:
                features.append(np.asarray([summary.mean], dtype=np.float64))
            if band.include_energy:
                features.append(np.asarray([summary.energy], dtype=np.float64))
            if band.include_drift:
                features.append(np.asarray([summary.drift], dtype=np.float64))
        if not features:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(features)


__all__ = [
    "SampledBandSummary",
    "SampledMultiscaleReadout",
]
