from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import softmax


@dataclass
class RidgeReadout:
    input_dim: int
    output_dim: int
    alpha: float = 1e-3
    use_bias: bool = True
    weights: np.ndarray | None = field(default=None, init=False)

    def _augment(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if not self.use_bias:
            return features
        bias = np.ones((features.shape[0], 1), dtype=np.float64)
        return np.concatenate([features, bias], axis=1)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.ndim != 2:
            raise ValueError("features must be rank-2")
        if targets.ndim != 1:
            raise ValueError("targets must be rank-1")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have the same number of rows")

        x = self._augment(features)
        y = np.eye(self.output_dim, dtype=np.float64)[targets.astype(np.int64)]
        regularizer = np.eye(x.shape[1], dtype=np.float64) * self.alpha
        if self.use_bias:
            regularizer[-1, -1] = 0.0
        gram = x.T @ x + regularizer
        rhs = x.T @ y
        self.weights = np.linalg.solve(gram, rhs)

    def _require_weights(self) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("The readout has not been fit yet.")
        return self.weights

    def logits(self, features: np.ndarray) -> np.ndarray:
        return self._augment(features) @ self._require_weights()

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        return softmax(self.logits(features), axis=-1)

