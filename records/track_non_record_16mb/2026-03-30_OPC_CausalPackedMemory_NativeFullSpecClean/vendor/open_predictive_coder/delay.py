from __future__ import annotations

import numpy as np

from .config import DelayLineConfig


class DelayLineSubstrate:
    def __init__(self, config: DelayLineConfig | None = None):
        self.config = config or DelayLineConfig()
        rng = np.random.default_rng(self.config.seed)
        self._token_embeddings = rng.normal(
            loc=0.0,
            scale=self.config.input_scale,
            size=(self.config.vocabulary_size, self.config.embedding_dim),
        ).astype(np.float64)
        self._token_embeddings /= np.sqrt(max(self.config.embedding_dim, 1))

    @property
    def state_dim(self) -> int:
        return self.config.history_length * self.config.embedding_dim

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.state_dim, dtype=np.float64)

    def _coerce_token(self, token: int) -> int:
        index = int(token)
        if index < 0 or index >= self.config.vocabulary_size:
            raise ValueError(
                f"token {index} is out of range for vocabulary_size={self.config.vocabulary_size}"
            )
        return index

    def history_view(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.state_dim,):
            raise ValueError("state has unexpected shape")
        return state.reshape(self.config.history_length, self.config.embedding_dim)

    def step(self, state: np.ndarray, token: int) -> np.ndarray:
        history = self.history_view(state)
        token_index = self._coerce_token(token)
        next_history = np.empty_like(history)
        next_history[0] = self._token_embeddings[token_index]
        if self.config.history_length > 1:
            next_history[1:] = history[:-1] * self.config.decay
        return next_history.reshape(-1)


__all__ = [
    "DelayLineConfig",
    "DelayLineSubstrate",
]
