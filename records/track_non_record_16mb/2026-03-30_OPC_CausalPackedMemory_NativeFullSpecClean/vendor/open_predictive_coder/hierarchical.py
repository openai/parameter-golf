from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HierarchicalSubstrateConfig


def _spectral_radius(matrix: np.ndarray) -> float:
    values = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(values)))


def _make_recurrent_matrix(
    rng: np.random.Generator,
    size: int,
    connectivity: float,
    target_radius: float,
) -> np.ndarray:
    weights = rng.standard_normal((size, size))
    mask = (rng.random((size, size)) < connectivity).astype(np.float64)
    np.fill_diagonal(mask, 0.0)
    weights = weights * mask
    current = _spectral_radius(weights)
    if current == 0.0:
        return weights
    return weights * (target_radius / current)


def _make_input_matrix(rng: np.random.Generator, rows: int, cols: int, scale: float) -> np.ndarray:
    weights = rng.normal(loc=0.0, scale=scale, size=(rows, cols))
    return weights / np.sqrt(max(cols, 1))


@dataclass(frozen=True)
class HierarchicalStateSlices:
    fast: slice
    mid: slice
    slow: slice


class HierarchicalSubstrate:
    def __init__(self, config: HierarchicalSubstrateConfig | None = None):
        self.config = config or HierarchicalSubstrateConfig()
        self._step_index = 0
        rng = np.random.default_rng(self.config.seed)

        self.fast_recurrent = _make_recurrent_matrix(
            rng,
            size=self.config.fast_size,
            connectivity=self.config.fast_connectivity,
            target_radius=self.config.fast_spectral_radius,
        )
        self.mid_recurrent = _make_recurrent_matrix(
            rng,
            size=self.config.mid_size,
            connectivity=self.config.mid_connectivity,
            target_radius=self.config.mid_spectral_radius,
        )
        self.slow_recurrent = _make_recurrent_matrix(
            rng,
            size=self.config.slow_size,
            connectivity=self.config.slow_connectivity,
            target_radius=self.config.slow_spectral_radius,
        )

        self.fast_input = _make_input_matrix(
            rng,
            rows=self.config.fast_size,
            cols=self.config.vocabulary_size,
            scale=self.config.input_scale,
        )
        self.mid_input = _make_input_matrix(
            rng,
            rows=self.config.mid_size,
            cols=self.config.vocabulary_size,
            scale=self.config.input_scale,
        )
        self.slow_input = _make_input_matrix(
            rng,
            rows=self.config.slow_size,
            cols=self.config.vocabulary_size,
            scale=self.config.input_scale,
        )

        self.fast_up = _make_input_matrix(
            rng,
            rows=self.config.mid_size,
            cols=self.config.fast_size,
            scale=self.config.upward_scale,
        )
        self.mid_up = _make_input_matrix(
            rng,
            rows=self.config.slow_size,
            cols=self.config.mid_size,
            scale=self.config.upward_scale,
        )

        self._fast_slice = slice(0, self.config.fast_size)
        self._mid_slice = slice(self.config.fast_size, self.config.fast_size + self.config.mid_size)
        self._slow_slice = slice(
            self.config.fast_size + self.config.mid_size,
            self.config.fast_size + self.config.mid_size + self.config.slow_size,
        )

    @property
    def state_dim(self) -> int:
        return self.config.fast_size + self.config.mid_size + self.config.slow_size

    @property
    def state_slices(self) -> HierarchicalStateSlices:
        return HierarchicalStateSlices(fast=self._fast_slice, mid=self._mid_slice, slow=self._slow_slice)

    def initial_state(self) -> np.ndarray:
        self._step_index = 0
        return np.zeros(self.state_dim, dtype=np.float64)

    def _split_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 1:
            raise ValueError("state must be rank-1")
        if state.shape[0] != self.state_dim:
            raise ValueError("state does not match configured state_dim")
        return state[self._fast_slice], state[self._mid_slice], state[self._slow_slice]

    def _coerce_token(self, token: int) -> int:
        token_id = int(token)
        if token_id < 0 or token_id >= self.config.vocabulary_size:
            raise ValueError("token is out of range for the configured vocabulary_size")
        return token_id

    def step(self, state: np.ndarray, token: int) -> np.ndarray:
        fast_state, mid_state, slow_state = self._split_state(state)
        token_id = self._coerce_token(token)

        fast_drive = self.fast_recurrent @ fast_state + self.fast_input[:, token_id]
        next_fast = (1.0 - self.config.fast_leak) * fast_state + self.config.fast_leak * np.tanh(fast_drive)

        mid_drive = self.mid_recurrent @ mid_state + self.mid_input[:, token_id] + self.fast_up @ next_fast
        next_mid = (1.0 - self.config.mid_leak) * mid_state + self.config.mid_leak * np.tanh(mid_drive)

        slow_active = self.config.slow_update_stride == 1 or (
            (self._step_index + 1) % self.config.slow_update_stride == 0
        )

        if slow_active:
            slow_drive = self.slow_recurrent @ slow_state + self.slow_input[:, token_id] + self.mid_up @ next_mid
            next_slow = (1.0 - self.config.slow_leak) * slow_state + self.config.slow_leak * np.tanh(slow_drive)
        else:
            next_slow = slow_state.copy()

        self._step_index += 1
        return np.concatenate([next_fast, next_mid, next_slow])


__all__ = [
    "HierarchicalStateSlices",
    "HierarchicalSubstrate",
    "HierarchicalSubstrateConfig",
]
