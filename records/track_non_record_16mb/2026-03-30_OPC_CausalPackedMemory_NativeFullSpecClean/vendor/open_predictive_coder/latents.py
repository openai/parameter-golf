from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LatentConfig
from .patching import AdaptiveSegmenter
from .reservoir import spectral_radius


def _normalized_matrix(rng: np.random.Generator, rows: int, cols: int, scale: float = 1.0) -> np.ndarray:
    matrix = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
    return (matrix / np.sqrt(max(cols, 1))) * scale


def _scaled_square(rng: np.random.Generator, size: int, radius: float) -> np.ndarray:
    matrix = rng.normal(loc=0.0, scale=1.0, size=(size, size))
    matrix = matrix / np.sqrt(max(size, 1))
    current = spectral_radius(matrix)
    if current == 0.0:
        return matrix
    return matrix * (radius / current)


@dataclass
class LatentState:
    global_state: np.ndarray
    previous_view: np.ndarray | None
    patch_sum: np.ndarray
    patch_length: int
    last_latent: np.ndarray
    steps: int
    patches: int


@dataclass(frozen=True)
class LatentObservation:
    local_view: np.ndarray
    predicted_view: np.ndarray
    prediction_error: np.ndarray
    patch_summary: np.ndarray
    global_state: np.ndarray
    latent: np.ndarray
    novelty: float
    patch_length: int
    boundary: bool


class LatentCommitter:
    def __init__(self, config: LatentConfig, substrate_size: int, seed: int):
        if config.reservoir_features > substrate_size:
            raise ValueError("config.reservoir_features must be <= substrate_size")

        self.config = config
        rng = np.random.default_rng(seed)
        self.sample_indices = np.sort(
            rng.choice(
                substrate_size,
                size=config.reservoir_features,
                replace=False,
            )
        )
        self.commit_projection = _normalized_matrix(
            rng,
            rows=config.latent_dim,
            cols=config.reservoir_features,
            scale=config.bridge_scale,
        )
        self.global_recurrent = _scaled_square(rng, config.global_dim, radius=0.9)
        self.global_input = _normalized_matrix(
            rng,
            rows=config.global_dim,
            cols=config.latent_dim,
            scale=config.global_update_scale,
        )
        self.local_predictor = _normalized_matrix(
            rng,
            rows=config.reservoir_features,
            cols=config.global_dim,
            scale=1.0,
        )

    def initial_state(self) -> LatentState:
        return LatentState(
            global_state=np.zeros(self.config.global_dim, dtype=np.float64),
            previous_view=None,
            patch_sum=np.zeros(self.config.reservoir_features, dtype=np.float64),
            patch_length=0,
            last_latent=np.zeros(self.config.latent_dim, dtype=np.float64),
            steps=0,
            patches=0,
        )

    def sample(self, substrate_state: np.ndarray) -> np.ndarray:
        substrate_state = np.asarray(substrate_state, dtype=np.float64)
        if substrate_state.ndim != 1:
            raise ValueError("substrate_state must be rank-1")
        return substrate_state[self.sample_indices]

    def step(
        self,
        state: LatentState,
        local_view: np.ndarray,
        segmenter: AdaptiveSegmenter,
    ) -> LatentObservation:
        local_view = np.asarray(local_view, dtype=np.float64)
        if local_view.shape != (self.config.reservoir_features,):
            raise ValueError("local_view does not match configured reservoir_features")

        if state.previous_view is None:
            novelty = 0.0
        else:
            novelty = float(np.mean(np.abs(local_view - state.previous_view)))

        state.patch_sum = state.patch_sum + local_view
        state.patch_length += 1
        patch_summary = state.patch_sum / state.patch_length
        boundary = segmenter.should_commit(state.patch_length, novelty)

        if boundary:
            latent = np.tanh(self.commit_projection @ patch_summary)
            state.global_state = np.tanh((self.global_recurrent @ state.global_state) + (self.global_input @ latent))
            state.last_latent = latent
            state.patches += 1

        predicted_view = np.tanh(self.local_predictor @ state.global_state)
        prediction_error = local_view - predicted_view
        observation = LatentObservation(
            local_view=local_view.copy(),
            predicted_view=predicted_view.copy(),
            prediction_error=prediction_error.copy(),
            patch_summary=patch_summary.copy(),
            global_state=state.global_state.copy(),
            latent=state.last_latent.copy(),
            novelty=novelty,
            patch_length=state.patch_length,
            boundary=boundary,
        )

        if boundary:
            state.patch_sum = np.zeros_like(state.patch_sum)
            state.patch_length = 0

        state.previous_view = local_view.copy()
        state.steps += 1
        return observation


__all__ = [
    "LatentCommitter",
    "LatentObservation",
    "LatentState",
]
