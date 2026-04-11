from __future__ import annotations

import numpy as np

from .config import LinearMemoryConfig


class LinearMemorySubstrate:
    def __init__(self, config: LinearMemoryConfig | None = None):
        self.config = config or LinearMemoryConfig()
        rng = np.random.default_rng(self.config.seed)
        self._token_embeddings = rng.normal(
            loc=0.0,
            scale=self.config.input_scale,
            size=(self.config.vocabulary_size, self.config.embedding_dim),
        ).astype(np.float64)
        self._token_embeddings /= np.sqrt(max(self.config.embedding_dim, 1))

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    @property
    def bank_count(self) -> int:
        return len(self.config.decays)

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.state_dim, dtype=np.float64)

    def state_view(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.state_dim,):
            raise ValueError("state has unexpected shape")
        return state.reshape(self.bank_count, self.config.embedding_dim)

    def _coerce_token(self, token: int) -> int:
        index = int(token)
        if index < 0 or index >= self.config.vocabulary_size:
            raise ValueError(
                f"token {index} is out of range for vocabulary_size={self.config.vocabulary_size}"
            )
        return index

    def step(self, state: np.ndarray, token: int) -> np.ndarray:
        banks = self.state_view(state)
        token_index = self._coerce_token(token)
        token_embedding = self._token_embeddings[token_index]
        next_banks = np.empty_like(banks)
        for index, decay in enumerate(self.config.decays):
            next_banks[index] = (decay * banks[index]) + token_embedding
        return next_banks.reshape(-1)


__all__ = [
    "LinearMemoryConfig",
    "LinearMemorySubstrate",
]
