from __future__ import annotations

import numpy as np

from .config import OscillatoryMemoryConfig


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _orthogonal_matrix(rng: np.random.Generator, size: int) -> np.ndarray:
    matrix = rng.normal(loc=0.0, scale=1.0, size=(size, size))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs
    return q.astype(np.float64, copy=False)


class OscillatoryMemorySubstrate:
    def __init__(self, config: OscillatoryMemoryConfig | None = None):
        self.config = config or OscillatoryMemoryConfig()
        rng = np.random.default_rng(self.config.seed)

        token_embeddings = rng.normal(
            loc=0.0,
            scale=self.config.input_scale,
            size=(self.config.vocabulary_size, self.config.embedding_dim),
        ).astype(np.float64)
        self._token_embeddings = _normalize_rows(token_embeddings)

        self._decay_input = np.stack(
            [_orthogonal_matrix(rng, self.config.embedding_dim) for _ in range(self.config.decay_bank_count)],
            axis=0,
        )

        self._osc_input_cos = np.stack(
            [_orthogonal_matrix(rng, self.config.embedding_dim) for _ in range(self.config.oscillatory_bank_count)],
            axis=0,
        )
        self._osc_input_sin = np.stack(
            [_orthogonal_matrix(rng, self.config.embedding_dim) for _ in range(self.config.oscillatory_bank_count)],
            axis=0,
        )

        low_damping, high_damping = self.config.oscillatory_damping_range
        low_period, high_period = self.config.oscillatory_period_range
        if self.config.oscillatory_bank_count == 1:
            damping = np.asarray([(low_damping + high_damping) * 0.5], dtype=np.float64)
            periods = np.asarray([(low_period + high_period) * 0.5], dtype=np.float64)
        else:
            damping = np.linspace(low_damping, high_damping, num=self.config.oscillatory_bank_count, dtype=np.float64)
            periods = np.geomspace(low_period, high_period, num=self.config.oscillatory_bank_count, dtype=np.float64)
        self._osc_damping = damping
        self._osc_cos = damping * np.cos((2.0 * np.pi) / periods)
        self._osc_sin = damping * np.sin((2.0 * np.pi) / periods)

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.state_dim, dtype=np.float64)

    def _coerce_token(self, token: int) -> int:
        index = int(token)
        if index < 0 or index >= self.config.vocabulary_size:
            raise ValueError(
                f"token {index} is out of range for vocabulary_size={self.config.vocabulary_size}"
            )
        return index

    def _split_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.state_dim,):
            raise ValueError("state has unexpected shape")
        decay_width = self.config.decay_bank_count * self.config.embedding_dim
        decay_state = state[:decay_width].reshape(self.config.decay_bank_count, self.config.embedding_dim)
        oscillatory_state = state[decay_width:].reshape(
            self.config.oscillatory_bank_count,
            2,
            self.config.embedding_dim,
        )
        return decay_state, oscillatory_state

    def step(self, state: np.ndarray, token: int) -> np.ndarray:
        decay_state, oscillatory_state = self._split_state(state)
        token_index = self._coerce_token(token)
        token_vector = self._token_embeddings[token_index]

        next_decay = np.empty_like(decay_state)
        for index, rate in enumerate(self.config.decay_rates):
            drive = self._decay_input[index] @ token_vector
            next_decay[index] = (rate * decay_state[index]) + drive

        next_oscillatory = np.empty_like(oscillatory_state)
        for index in range(self.config.oscillatory_bank_count):
            cos_state = oscillatory_state[index, 0]
            sin_state = oscillatory_state[index, 1]
            drive_cos = self._osc_input_cos[index] @ token_vector
            drive_sin = self._osc_input_sin[index] @ token_vector

            next_oscillatory[index, 0] = (
                self._osc_cos[index] * cos_state
                - self._osc_sin[index] * sin_state
                + drive_cos
            )
            next_oscillatory[index, 1] = (
                self._osc_sin[index] * cos_state
                + self._osc_cos[index] * sin_state
                + drive_sin
            )

        return np.concatenate([next_decay.reshape(-1), next_oscillatory.reshape(-1)])


__all__ = [
    "OscillatoryMemoryConfig",
    "OscillatoryMemorySubstrate",
]
