from __future__ import annotations

import numpy as np

from .config import MixedMemoryConfig
from .delay import DelayLineSubstrate
from .reservoir import EchoStateReservoir


from dataclasses import dataclass


@dataclass(frozen=True)
class MixedMemoryStateSlices:
    reservoir: slice
    delay: slice


class MixedMemorySubstrate:
    def __init__(self, config: MixedMemoryConfig | None = None):
        self.config = config or MixedMemoryConfig()
        self.reservoir = EchoStateReservoir(
            config=self.config.reservoir,
            vocabulary_size=self.config.delay.vocabulary_size,
        )
        self.delay = DelayLineSubstrate(self.config.delay)
        self._reservoir_slice = slice(0, self.reservoir.state_dim)
        self._delay_slice = slice(self.reservoir.state_dim, self.reservoir.state_dim + self.config.delay.state_dim)

    @property
    def reservoir_dim(self) -> int:
        return self.reservoir.state_dim

    @property
    def delay_dim(self) -> int:
        return self.config.delay.state_dim

    @property
    def state_dim(self) -> int:
        return self.reservoir_dim + self.delay_dim

    @property
    def state_slices(self) -> MixedMemoryStateSlices:
        return MixedMemoryStateSlices(reservoir=self._reservoir_slice, delay=self._delay_slice)

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.state_dim, dtype=np.float64)

    def reservoir_view(self, state: np.ndarray) -> np.ndarray:
        return self._split_state(state)[0]

    def delay_view(self, state: np.ndarray) -> np.ndarray:
        return self._split_state(state)[1]

    def _split_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 1:
            raise ValueError("state must be rank-1")
        if state.shape[0] != self.state_dim:
            raise ValueError("state does not match configured state_dim")
        return state[self._reservoir_slice], state[self._delay_slice]

    def step(self, state: np.ndarray, token: int) -> np.ndarray:
        reservoir_state, delay_state = self._split_state(state)
        token_id = int(token)
        if token_id < 0 or token_id >= self.config.delay.vocabulary_size:
            raise ValueError("token is out of range for the configured vocabulary_size")

        next_reservoir = self.reservoir.step(reservoir_state, token_id)
        next_delay = self.delay.step(delay_state, token_id)
        return np.concatenate([next_reservoir, next_delay])


__all__ = [
    "MixedMemoryConfig",
    "MixedMemoryStateSlices",
    "MixedMemorySubstrate",
]
