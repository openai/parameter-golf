from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _coerce_feature_vector(features: BoundaryFeatures | Sequence[float] | np.ndarray, *, feature_dim: int) -> np.ndarray:
    if isinstance(features, BoundaryFeatures):
        vector = features.as_array()
    else:
        vector = np.asarray(features, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("features must be a 1D vector")
    if vector.shape[0] != feature_dim:
        raise ValueError(f"features must have shape ({feature_dim},)")
    return vector


def _coerce_feature_matrix(
    features: Sequence[BoundaryFeatures | Sequence[float] | np.ndarray] | np.ndarray,
    *,
    feature_dim: int,
) -> np.ndarray:
    if isinstance(features, np.ndarray):
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
    else:
        rows = [_coerce_feature_vector(row, feature_dim=feature_dim) for row in features]
        matrix = np.vstack(rows)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D matrix or a sequence of vectors")
    if matrix.shape[1] != feature_dim:
        raise ValueError(f"features must have shape (n, {feature_dim})")
    return matrix


def _resolve_target_rate(config: BoundaryScorerConfig, override: float | None = None) -> float | None:
    if override is not None:
        return override
    if config.target_boundary_rate is not None:
        return config.target_boundary_rate
    if config.target_patch_size is not None:
        return 1.0 / float(config.target_patch_size)
    return None


def _update_mean(current_mean: float, count: int, value: float) -> float:
    return current_mean + ((value - current_mean) / float(count))


@dataclass(frozen=True)
class BoundaryScorerConfig:
    feature_dim: int = 5
    learning_rate: float = 0.1
    l2: float = 1e-3
    threshold: float = 0.5
    min_patch_size: int = 2
    max_patch_size: int = 8
    target_patch_size: float | None = None
    target_boundary_rate: float | None = None
    target_regularization: float = 0.25
    initial_bias: float = -1.0

    def __post_init__(self) -> None:
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.l2 < 0.0:
            raise ValueError("l2 must be >= 0")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.min_patch_size < 1:
            raise ValueError("min_patch_size must be >= 1")
        if self.max_patch_size < self.min_patch_size:
            raise ValueError("max_patch_size must be >= min_patch_size")
        if self.target_patch_size is not None and self.target_patch_size <= 0.0:
            raise ValueError("target_patch_size must be > 0")
        if self.target_boundary_rate is not None and not 0.0 < self.target_boundary_rate <= 1.0:
            raise ValueError("target_boundary_rate must be in (0, 1]")
        if self.target_regularization < 0.0:
            raise ValueError("target_regularization must be >= 0")

    @property
    def commit_threshold(self) -> float:
        return self.threshold


@dataclass
class BoundaryScorerState:
    weights: np.ndarray
    bias: float
    steps_seen: int = 0
    boundaries_seen: int = 0
    patches_seen: int = 0
    current_patch_length: int = 0
    mean_probability: float = 0.0
    mean_patch_length: float = 0.0
    mean_target_rate: float = 0.0
    last_probability: float = 0.0
    last_logit: float = 0.0

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1D vector")
        if not np.isfinite(self.bias):
            raise ValueError("bias must be finite")


@dataclass(frozen=True)
class BoundaryFeatures:
    bias: float = 1.0
    novelty: float = 0.0
    drift: float = 0.0
    patch_progress: float = 0.0
    patch_utilization: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.bias),
                float(self.novelty),
                float(self.drift),
                float(self.patch_progress),
                float(self.patch_utilization),
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class BoundaryDecision:
    boundary: bool
    probability: float
    logit: float
    patch_length: int
    next_patch_length: int
    features: BoundaryFeatures


class LearnedBoundaryScorer:
    def __init__(self, config: BoundaryScorerConfig | None = None, state: BoundaryScorerState | None = None):
        self.config = config or BoundaryScorerConfig()
        if state is None:
            state = BoundaryScorerState(
                weights=np.zeros(self.config.feature_dim, dtype=np.float64),
                bias=self.config.initial_bias,
            )
        if state.weights.shape != (self.config.feature_dim,):
            raise ValueError(f"state.weights must have shape ({self.config.feature_dim},)")
        self.state = state

    def logit(self, features: BoundaryFeatures | Sequence[float] | np.ndarray) -> float:
        vector = _coerce_feature_vector(features, feature_dim=self.config.feature_dim)
        return float(np.dot(self.state.weights, vector) + self.state.bias)

    def probability(self, features: BoundaryFeatures | Sequence[float] | np.ndarray) -> float:
        return float(_sigmoid(np.asarray([self.logit(features)], dtype=np.float64))[0])

    def update(
        self,
        features: BoundaryFeatures | Sequence[float] | np.ndarray,
        target: bool | float,
        *,
        target_rate: float | None = None,
    ) -> BoundaryScorerState:
        vector = _coerce_feature_vector(features, feature_dim=self.config.feature_dim)
        target_value = float(target)
        if not 0.0 <= target_value <= 1.0:
            raise ValueError("target must be in [0, 1]")

        logit = float(np.dot(self.state.weights, vector) + self.state.bias)
        probability = float(_sigmoid(np.asarray([logit], dtype=np.float64))[0])

        error = probability - target_value
        grad_weights = (error * vector) + (self.config.l2 * self.state.weights)
        grad_bias = error

        effective_target_rate = _resolve_target_rate(self.config, target_rate)
        if effective_target_rate is not None:
            grad_bias += self.config.target_regularization * (probability - effective_target_rate)

        self.state.weights = self.state.weights - (self.config.learning_rate * grad_weights)
        self.state.bias = float(self.state.bias - (self.config.learning_rate * grad_bias))

        self.state.steps_seen += 1
        self.state.mean_probability = _update_mean(self.state.mean_probability, self.state.steps_seen, probability)
        self.state.last_probability = probability
        self.state.last_logit = logit
        if target_rate is not None or self.config.target_boundary_rate is not None or self.config.target_patch_size is not None:
            if effective_target_rate is not None:
                self.state.mean_target_rate = _update_mean(
                    self.state.mean_target_rate,
                    self.state.steps_seen,
                    effective_target_rate,
                )
        return self.state

    def fit(
        self,
        features: Sequence[BoundaryFeatures | Sequence[float] | np.ndarray] | np.ndarray,
        targets: Sequence[bool | float] | np.ndarray,
        *,
        epochs: int = 1,
        target_rate: float | None = None,
    ) -> BoundaryScorerState:
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        matrix = _coerce_feature_matrix(features, feature_dim=self.config.feature_dim)
        labels = np.asarray(targets, dtype=np.float64)
        if labels.ndim != 1:
            raise ValueError("targets must be a 1D vector")
        if matrix.shape[0] != labels.shape[0]:
            raise ValueError("features and targets must contain the same number of rows")

        for _ in range(epochs):
            for row, target in zip(matrix, labels):
                self.update(row, bool(target), target_rate=target_rate)
        return self.state


class LearnedSegmenter:
    def __init__(self, config: BoundaryScorerConfig | None = None, scorer: LearnedBoundaryScorer | None = None):
        self.config = config or BoundaryScorerConfig()
        self.scorer = scorer or LearnedBoundaryScorer(self.config)
        if self.scorer.config != self.config:
            raise ValueError("scorer config must match segmenter config")

    @property
    def state(self) -> BoundaryScorerState:
        return self.scorer.state

    def reset(self) -> None:
        self.scorer.state.current_patch_length = 0

    def _feature_builder(
        self,
        *,
        novelty: float = 0.0,
        drift: float = 0.0,
        patch_length: int | None = None,
    ) -> BoundaryFeatures:
        current_length = self.state.current_patch_length + 1 if patch_length is None else int(patch_length)
        scale = self.config.target_patch_size or float(self.config.max_patch_size)
        scale = max(scale, 1.0)
        return BoundaryFeatures(
            novelty=float(novelty),
            drift=float(drift),
            patch_progress=current_length / scale,
            patch_utilization=current_length / float(self.config.max_patch_size),
        )

    def step(
        self,
        features: BoundaryFeatures | Sequence[float] | np.ndarray | None = None,
        *,
        novelty: float = 0.0,
        drift: float = 0.0,
        target: bool | None = None,
        learn: bool = False,
        target_rate: float | None = None,
    ) -> BoundaryDecision:
        candidate_length = self.state.current_patch_length + 1
        feature_row = features if features is not None else self._feature_builder(
            novelty=novelty,
            drift=drift,
            patch_length=candidate_length,
        )

        probability = self.scorer.probability(feature_row)
        logit = self.scorer.logit(feature_row)

        if candidate_length >= self.config.max_patch_size:
            boundary = True
        elif candidate_length < self.config.min_patch_size:
            boundary = False
        else:
            boundary = probability >= self.config.threshold

        if learn:
            if target is None:
                raise ValueError("learn=True requires target")
            self.scorer.update(feature_row, target, target_rate=target_rate)

        self.state.steps_seen += 1
        self.state.mean_probability = _update_mean(self.state.mean_probability, self.state.steps_seen, probability)
        self.state.last_probability = probability
        self.state.last_logit = logit

        if boundary:
            self.state.boundaries_seen += 1
            self.state.patches_seen += 1
            self.state.mean_patch_length = _update_mean(
                self.state.mean_patch_length,
                self.state.patches_seen,
                float(candidate_length),
            )
            self.state.current_patch_length = 0
        else:
            self.state.current_patch_length = candidate_length

        if target is not None:
            self.state.mean_target_rate = _update_mean(
                self.state.mean_target_rate,
                self.state.steps_seen,
                1.0 if target else 0.0,
            )

        return BoundaryDecision(
            boundary=boundary,
            probability=probability,
            logit=logit,
            patch_length=candidate_length,
            next_patch_length=self.state.current_patch_length,
            features=feature_row if isinstance(feature_row, BoundaryFeatures) else BoundaryFeatures(
                novelty=float(novelty),
                drift=float(drift),
                patch_progress=(candidate_length / float(self.config.target_patch_size or self.config.max_patch_size)),
                patch_utilization=candidate_length / float(self.config.max_patch_size),
            ),
        )

    def fit(
        self,
        features: Sequence[BoundaryFeatures | Sequence[float] | np.ndarray] | np.ndarray,
        targets: Sequence[bool | float] | np.ndarray,
        *,
        epochs: int = 1,
        target_rate: float | None = None,
    ) -> BoundaryScorerState:
        self.scorer.fit(features, targets, epochs=epochs, target_rate=target_rate)
        return self.state

    def decide(
        self,
        patch_length: int,
        novelty: float,
        surprise: float,
        *,
        train: bool = False,
        update_steps: int = 1,
    ) -> BoundaryDecision:
        if patch_length < 1:
            raise ValueError("patch_length must be >= 1")
        features = self._feature_builder(
            novelty=novelty,
            drift=surprise,
            patch_length=patch_length,
        )
        probability = self.scorer.probability(features)
        logit = self.scorer.logit(features)

        if patch_length >= self.config.max_patch_size:
            boundary = True
            target = True
        elif patch_length < self.config.min_patch_size:
            boundary = False
            target = False
        else:
            boundary = probability >= self.config.threshold
            if self.config.target_patch_size is None:
                target = boundary
            else:
                target = patch_length >= int(round(self.config.target_patch_size))

        if train:
            for _ in range(update_steps):
                self.scorer.update(
                    features,
                    target,
                    target_rate=_resolve_target_rate(self.config),
                )
            probability = self.scorer.probability(features)
            logit = self.scorer.logit(features)
            if patch_length >= self.config.max_patch_size:
                boundary = True
            elif patch_length < self.config.min_patch_size:
                boundary = False
            else:
                boundary = probability >= self.config.threshold

        return BoundaryDecision(
            boundary=boundary,
            probability=probability,
            logit=logit,
            patch_length=patch_length,
            next_patch_length=0 if boundary else patch_length,
            features=features,
        )


__all__ = [
    "BoundaryDecision",
    "BoundaryFeatures",
    "BoundaryScorerConfig",
    "BoundaryScorerState",
    "LearnedBoundaryScorer",
    "LearnedSegmenterConfig",
    "LearnedSegmenter",
]

LearnedSegmenterConfig = BoundaryScorerConfig
