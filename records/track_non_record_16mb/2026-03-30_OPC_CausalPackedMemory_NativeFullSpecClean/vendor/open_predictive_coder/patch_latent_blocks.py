from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _as_float_array(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _coerce_tokens(
    tokens: str | bytes | bytearray | memoryview | np.ndarray | list[int] | tuple[int, ...],
) -> np.ndarray:
    if isinstance(tokens, str):
        return np.frombuffer(tokens.encode("utf-8"), dtype=np.uint8)
    if isinstance(tokens, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(tokens), dtype=np.uint8)
    array = np.asarray(tokens)
    if array.ndim != 1:
        raise ValueError("tokens must be rank-1")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("tokens must contain integers")
    return array.astype(np.uint8, copy=False)


def _coerce_matrix(values: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("matrix values must be rank-2")
    return array


def _scaled_matrix(rng: np.random.Generator, rows: int, cols: int, scale: float) -> np.ndarray:
    if rows < 1 or cols < 1:
        raise ValueError("matrix dimensions must be >= 1")
    matrix = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
    return (matrix / np.sqrt(max(cols, 1))) * scale


def _coerce_2d(values: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim != 2:
        raise ValueError("values must be rank-1 or rank-2")
    return array


@dataclass(frozen=True)
class LocalByteEncoderConfig:
    vocabulary_size: int = 256
    local_dim: int = 32
    state_dim: int = 32
    output_dim: int | None = None
    embedding_scale: float = 1.0
    input_scale: float = 1.0
    recurrent_scale: float = 0.7
    output_scale: float = 1.0
    output_l2: float = 1e-4
    seed: int = 7

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.local_dim < 1:
            raise ValueError("local_dim must be >= 1")
        if self.state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if self.output_dim is None:
            object.__setattr__(self, "output_dim", self.local_dim)
        if self.output_dim is None or self.output_dim < 1:
            raise ValueError("output_dim must be >= 1")
        if self.embedding_scale <= 0.0:
            raise ValueError("embedding_scale must be > 0")
        if self.input_scale <= 0.0:
            raise ValueError("input_scale must be > 0")
        if self.recurrent_scale <= 0.0:
            raise ValueError("recurrent_scale must be > 0")
        if self.output_scale <= 0.0:
            raise ValueError("output_scale must be > 0")
        if self.output_l2 < 0.0:
            raise ValueError("output_l2 must be >= 0")

    @property
    def feature_dim(self) -> int:
        return int(self.output_dim)


@dataclass(frozen=True)
class PatchPoolerConfig:
    mode: Literal["mean", "last", "mix"] = "mean"
    mix_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in {"mean", "last", "mix"}:
            raise ValueError("mode must be one of mean, last, mix")
        if not 0.0 <= self.mix_weight <= 1.0:
            raise ValueError("mix_weight must be between 0 and 1")


@dataclass(frozen=True)
class GlobalLocalBridgeConfig:
    global_dim: int
    latent_dim: int
    local_dim: int
    learning_rate: float = 0.05
    l2: float = 1e-4
    seed: int = 11
    use_bias: bool = True

    def __post_init__(self) -> None:
        if self.global_dim < 0:
            raise ValueError("global_dim must be >= 0")
        if self.latent_dim < 0:
            raise ValueError("latent_dim must be >= 0")
        if self.local_dim < 1:
            raise ValueError("local_dim must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.l2 < 0.0:
            raise ValueError("l2 must be >= 0")

    @property
    def input_dim(self) -> int:
        return self.global_dim + self.latent_dim


class LocalByteEncoder:
    def __init__(self, config: LocalByteEncoderConfig | None = None):
        self.config = config or LocalByteEncoderConfig()
        rng = np.random.default_rng(self.config.seed)
        self.embedding = _scaled_matrix(
            rng,
            rows=self.config.vocabulary_size,
            cols=self.config.local_dim,
            scale=self.config.embedding_scale,
        )
        self.input_weights = _scaled_matrix(
            rng,
            rows=self.config.state_dim,
            cols=self.config.local_dim,
            scale=self.config.input_scale,
        )
        self.recurrent_weights = _scaled_matrix(
            rng,
            rows=self.config.state_dim,
            cols=self.config.state_dim,
            scale=self.config.recurrent_scale,
        )
        self.output_weights = _scaled_matrix(
            rng,
            rows=self.config.output_dim,
            cols=self.config.state_dim,
            scale=self.config.output_scale,
        )
        self.state_bias = np.zeros(self.config.state_dim, dtype=np.float64)
        self.output_bias = np.zeros(self.config.output_dim, dtype=np.float64)

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.config.state_dim, dtype=np.float64)

    def _step_hidden(self, token: int, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        token = int(token)
        if token < 0 or token >= self.config.vocabulary_size:
            raise ValueError("token must lie inside the vocabulary")
        current_state = self.initial_state() if state is None else np.asarray(state, dtype=np.float64)
        if current_state.shape != (self.config.state_dim,):
            raise ValueError("state does not match configured state_dim")
        token_vector = self.embedding[token]
        next_state = np.tanh(
            self.recurrent_weights @ current_state
            + self.input_weights @ token_vector
            + self.state_bias
        )
        return next_state, current_state

    def step(self, token: int, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        next_state, _ = self._step_hidden(token, state)
        local_features = np.tanh(self.output_weights @ next_state + self.output_bias)
        return local_features, next_state

    def hidden_states(
        self,
        tokens: str | bytes | bytearray | memoryview | np.ndarray | list[int] | tuple[int, ...],
        *,
        initial_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        token_array = _coerce_tokens(tokens)
        state = self.initial_state() if initial_state is None else np.asarray(initial_state, dtype=np.float64)
        if state.shape != (self.config.state_dim,):
            raise ValueError("initial_state does not match configured state_dim")
        hidden: list[np.ndarray] = []
        for token in token_array:
            state, _ = self._step_hidden(int(token), state)
            hidden.append(state)
        if not hidden:
            return np.zeros((0, self.state_dim), dtype=np.float64), state.copy()
        return np.vstack(hidden), state.copy()

    def encode(
        self,
        tokens: str | bytes | bytearray | memoryview | np.ndarray | list[int] | tuple[int, ...],
        *,
        initial_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        hidden, state = self.hidden_states(tokens, initial_state=initial_state)
        if hidden.shape[0] == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float64), state.copy()
        features = np.tanh(hidden @ self.output_weights.T + self.output_bias)
        return features, state.copy()

    def output_error(
        self,
        hidden_states: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
        targets: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    ) -> float:
        hidden = _coerce_2d(hidden_states)
        target_array = _coerce_2d(targets)
        if hidden.shape[1] != self.state_dim:
            raise ValueError("hidden_states do not match configured state_dim")
        if target_array.shape != (hidden.shape[0], self.feature_dim):
            raise ValueError("targets do not match hidden_states or feature_dim")
        predicted = np.tanh(hidden @ self.output_weights.T + self.output_bias)
        return float(np.mean(np.square(predicted - target_array)))

    def fit_output(
        self,
        hidden_states: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
        targets: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    ) -> float:
        hidden = _coerce_2d(hidden_states)
        target_array = _coerce_2d(targets)
        if hidden.shape[1] != self.state_dim:
            raise ValueError("hidden_states do not match configured state_dim")
        if target_array.shape != (hidden.shape[0], self.feature_dim):
            raise ValueError("targets do not match hidden_states or feature_dim")
        clipped = np.clip(target_array, -0.999999, 0.999999)
        preactivation = np.arctanh(clipped)
        design = np.concatenate([hidden, np.ones((hidden.shape[0], 1), dtype=np.float64)], axis=1)
        penalty = self.config.output_l2 * np.eye(design.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        solution = np.linalg.solve(design.T @ design + penalty, design.T @ preactivation)
        self.output_weights = solution[:-1].T
        self.output_bias = solution[-1]
        return self.output_error(hidden, target_array)


class PatchPooler:
    def __init__(self, config: PatchPoolerConfig | None = None):
        self.config = config or PatchPoolerConfig()

    def pool(self, block: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...]) -> np.ndarray:
        array = _coerce_matrix(block)
        if array.shape[0] == 0:
            raise ValueError("block must contain at least one row")
        if self.config.mode == "mean":
            return np.mean(array, axis=0)
        if self.config.mode == "last":
            return array[-1].copy()
        mean = np.mean(array, axis=0)
        last = array[-1]
        return ((1.0 - self.config.mix_weight) * mean) + (self.config.mix_weight * last)


class GlobalLocalBridge:
    def __init__(self, config: GlobalLocalBridgeConfig | None = None):
        self.config = config or GlobalLocalBridgeConfig(global_dim=16, latent_dim=16, local_dim=16)
        rng = np.random.default_rng(self.config.seed)
        self.weights = _scaled_matrix(
            rng,
            rows=self.config.input_dim,
            cols=self.config.local_dim,
            scale=0.1,
        )
        self.bias = np.zeros(self.config.local_dim, dtype=np.float64)

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    @property
    def output_dim(self) -> int:
        return self.config.local_dim

    def stack_state(
        self,
        global_state: np.ndarray | list[float] | tuple[float, ...],
        latent_state: np.ndarray | list[float] | tuple[float, ...],
    ) -> np.ndarray:
        global_array = _as_float_array(global_state)
        latent_array = _as_float_array(latent_state)
        if global_array.shape != (self.config.global_dim,):
            raise ValueError("global_state does not match configured global_dim")
        if latent_array.shape != (self.config.latent_dim,):
            raise ValueError("latent_state does not match configured latent_dim")
        return np.concatenate([global_array, latent_array])

    def predict_batch(self, inputs: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...]) -> np.ndarray:
        array = _coerce_2d(inputs)
        if array.shape[1] != self.config.input_dim:
            raise ValueError("inputs do not match configured input_dim")
        return array @ self.weights + self.bias

    def predict(
        self,
        global_state: np.ndarray | list[float] | tuple[float, ...],
        latent_state: np.ndarray | list[float] | tuple[float, ...],
    ) -> np.ndarray:
        return self.predict_batch(self.stack_state(global_state, latent_state)[None, :])[0]

    def reconstruction_error(
        self,
        inputs: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
        targets: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    ) -> float:
        predicted = self.predict_batch(inputs)
        target_array = _coerce_2d(targets)
        if target_array.shape != predicted.shape:
            raise ValueError("targets must match predicted shape")
        return float(np.mean(np.square(predicted - target_array)))

    def fit(
        self,
        inputs: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
        targets: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    ) -> float:
        x = _coerce_2d(inputs)
        y = _coerce_2d(targets)
        if x.shape[1] != self.config.input_dim:
            raise ValueError("inputs do not match configured input_dim")
        if y.shape[1] != self.config.local_dim:
            raise ValueError("targets do not match configured local_dim")
        if x.shape[0] != y.shape[0]:
            raise ValueError("inputs and targets must have the same number of rows")

        if self.config.use_bias:
            design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
            penalty = self.config.l2 * np.eye(design.shape[1], dtype=np.float64)
            penalty[-1, -1] = 0.0
            solution = np.linalg.solve(design.T @ design + penalty, design.T @ y)
            self.weights = solution[:-1]
            self.bias = solution[-1]
        else:
            penalty = self.config.l2 * np.eye(x.shape[1], dtype=np.float64)
            self.weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
            self.bias = np.zeros(self.config.local_dim, dtype=np.float64)
        return self.reconstruction_error(x, y)

    def update(
        self,
        inputs: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
        targets: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
        *,
        steps: int = 1,
    ) -> float:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        x = _coerce_2d(inputs)
        y = _coerce_2d(targets)
        if x.shape != (y.shape[0], self.config.input_dim):
            raise ValueError("inputs do not match configured input_dim")
        if y.shape[1] != self.config.local_dim:
            raise ValueError("targets do not match configured local_dim")
        if x.shape[0] != y.shape[0]:
            raise ValueError("inputs and targets must have the same number of rows")

        rate = self.config.learning_rate
        for _ in range(steps):
            predicted = x @ self.weights + self.bias
            error = predicted - y
            self.weights -= rate * ((x.T @ error) / x.shape[0] + (self.config.l2 * self.weights))
            if self.config.use_bias:
                self.bias -= rate * np.mean(error, axis=0)
        return self.reconstruction_error(x, y)


__all__ = [
    "GlobalLocalBridge",
    "GlobalLocalBridgeConfig",
    "LocalByteEncoder",
    "LocalByteEncoderConfig",
    "PatchPooler",
    "PatchPoolerConfig",
]
