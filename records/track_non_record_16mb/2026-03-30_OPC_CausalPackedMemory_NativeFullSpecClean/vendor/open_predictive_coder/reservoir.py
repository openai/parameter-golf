from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ReservoirConfig, ReservoirTopology


def spectral_radius(matrix: np.ndarray) -> float:
    values = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(values)))


def _scale_spectral_radius(matrix: np.ndarray, target: float) -> np.ndarray:
    radius = spectral_radius(matrix)
    if radius == 0.0:
        return matrix
    return matrix * (target / radius)


def _small_world_degree(size: int, connectivity: float) -> int:
    degree = max(2, int(round(connectivity * (size - 1))))
    degree = min(degree, size - 1)
    if degree % 2 == 1:
        degree = degree - 1 if degree > 2 else degree + 1
    return max(2, degree)


def _small_world_mask(size: int, connectivity: float, rewire_prob: float, rng: np.random.Generator) -> np.ndarray:
    degree = _small_world_degree(size, connectivity)
    adjacency = np.zeros((size, size), dtype=np.float64)
    half = degree // 2

    for node in range(size):
        for offset in range(1, half + 1):
            neighbor = (node + offset) % size
            adjacency[node, neighbor] = 1.0
            adjacency[neighbor, node] = 1.0

    for node in range(size):
        for offset in range(1, half + 1):
            neighbor = (node + offset) % size
            if adjacency[node, neighbor] == 0.0 or rng.random() >= rewire_prob:
                continue
            adjacency[node, neighbor] = 0.0
            adjacency[neighbor, node] = 0.0

            candidates = np.flatnonzero(adjacency[node] == 0.0)
            candidates = candidates[candidates != node]
            if candidates.size == 0:
                adjacency[node, neighbor] = 1.0
                adjacency[neighbor, node] = 1.0
                continue

            replacement = int(rng.choice(candidates))
            adjacency[node, replacement] = 1.0
            adjacency[replacement, node] = 1.0

    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def build_recurrent_matrix(config: ReservoirConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    matrix = rng.standard_normal((config.size, config.size))
    if config.topology == "erdos_renyi":
        mask = (rng.random((config.size, config.size)) < config.connectivity).astype(np.float64)
    elif config.topology == "small_world":
        mask = _small_world_mask(config.size, config.connectivity, config.rewire_prob, rng)
    else:
        raise ValueError(f"Unknown reservoir topology: {config.topology}")
    matrix = matrix * mask
    np.fill_diagonal(matrix, 0.0)
    return _scale_spectral_radius(matrix, config.spectral_radius)


@dataclass(frozen=True)
class EchoStateReservoir:
    config: ReservoirConfig
    vocabulary_size: int = 256

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.config.seed + 17)
        object.__setattr__(self, "recurrent", build_recurrent_matrix(self.config))
        object.__setattr__(
            self,
            "input_weights",
            rng.normal(
                loc=0.0,
                scale=self.config.input_scale,
                size=(self.config.size, self.vocabulary_size),
            ),
        )

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.config.size, dtype=np.float64)

    @property
    def state_dim(self) -> int:
        return self.config.size

    def step(self, state: np.ndarray, token: int) -> np.ndarray:
        drive = self.recurrent @ state + self.input_weights[:, int(token)]
        proposed = np.tanh(drive)
        return ((1.0 - self.config.leak) * state) + (self.config.leak * proposed)
