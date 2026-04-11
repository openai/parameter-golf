from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, WhiteKernel

from search.config import ParamSpec


EPSILON = 1e-6


class Space:
    def __init__(self, spec: ParamSpec, *, integer: bool = False, pow2: bool = False):
        self.spec = spec
        self.min = spec.min
        self.max = spec.max
        self.scale = 0.5 if spec.scale == "auto" else float(spec.scale)
        self.integer = integer
        self.pow2 = pow2
        self.round_to = spec.round_to
        self.norm_min = self.normalize(self.min)
        self.norm_max = self.normalize(self.max)

    def clamp(self, value: float) -> float:
        return min(max(value, self.min), self.max)

    def _round(self, value: float) -> float:
        if self.pow2:
            value = 2 ** round(math.log2(max(value, self.min)))
        if self.round_to:
            value = round(value / self.round_to) * self.round_to
        if self.integer:
            value = round(value)
        return self.clamp(value)


class LinearSpace(Space):
    def normalize(self, value: float) -> float:
        zero_one = (value - self.min) / (self.max - self.min)
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> float:
        zero_one = (value + 1) / 2
        return self._round(zero_one * (self.max - self.min) + self.min)


class LogSpace(Space):
    def normalize(self, value: float) -> float:
        zero_one = (math.log10(value) - math.log10(self.min)) / (math.log10(self.max) - math.log10(self.min))
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> float:
        zero_one = (value + 1) / 2
        exponent = zero_one * (math.log10(self.max) - math.log10(self.min)) + math.log10(self.min)
        return self._round(10 ** exponent)


def make_space(spec: ParamSpec) -> Space:
    if spec.distribution == "uniform":
        return LinearSpace(spec)
    if spec.distribution == "int_uniform":
        return LinearSpace(spec, integer=True)
    if spec.distribution == "uniform_pow2":
        return LogSpace(spec, integer=True, pow2=True)
    if spec.distribution in {"log_uniform", "log_normal"}:
        return LogSpace(spec)
    raise ValueError(f"Unsupported distribution: {spec.distribution}")


class SearchSpace:
    def __init__(self, specs: dict[str, ParamSpec]):
        self.names = list(specs.keys())
        self.spaces = [make_space(specs[name]) for name in self.names]
        self.dimension = len(self.names)
        self.min_bounds = np.array([space.norm_min for space in self.spaces], dtype=np.float64)
        self.max_bounds = np.array([space.norm_max for space in self.spaces], dtype=np.float64)
        self.scales = np.array([space.scale for space in self.spaces], dtype=np.float64)
        self.search_center = np.zeros(self.dimension, dtype=np.float64)

    def sample(self, n: int, rng: np.random.Generator, centers: np.ndarray | None = None) -> np.ndarray:
        centers = self.search_center[None, :] if centers is None else np.atleast_2d(centers)
        center_indices = rng.integers(0, len(centers), size=n)
        deltas = (2 * rng.random((n, self.dimension)) - 1) * self.scales
        samples = centers[center_indices] + deltas
        return np.clip(samples, self.min_bounds, self.max_bounds)

    def to_mapping(self, sample: np.ndarray) -> dict[str, float | int]:
        values: dict[str, float | int] = {}
        for name, space, norm_value in zip(self.names, self.spaces, sample, strict=True):
            value = space.unnormalize(float(norm_value))
            if float(value).is_integer():
                values[name] = int(value)
            else:
                values[name] = float(value)
        return values

    def from_mapping(self, values: dict[str, float | int]) -> np.ndarray:
        norm = []
        for name, space in zip(self.names, self.spaces, strict=True):
            if name not in values:
                raise KeyError(f"Missing hyperparameter {name}")
            norm.append(space.normalize(float(values[name])))
        return np.array(norm, dtype=np.float64)


@dataclass
class Observation:
    input: np.ndarray
    output: float
    cost: float
    metadata: dict[str, object] | None = None


def pareto_points(observations: list[Observation]) -> list[Observation]:
    if not observations:
        return []
    scores = np.array([obs.output for obs in observations], dtype=np.float64)
    costs = np.array([obs.cost for obs in observations], dtype=np.float64)
    sorted_indices = np.argsort(costs)
    pareto: list[Observation] = []
    max_score_so_far = -math.inf
    for idx in sorted_indices:
        if scores[idx] > max_score_so_far + EPSILON:
            pareto.append(observations[idx])
            max_score_so_far = scores[idx]
    return pareto


def prune_pareto_front(pareto: list[Observation], *, efficiency_threshold: float = 0.5, pruning_stop_score_fraction: float = 0.98) -> list[Observation]:
    if len(pareto) < 2:
        return pareto
    ordered = sorted(pareto, key=lambda obs: obs.cost)
    scores = np.array([obs.output for obs in ordered], dtype=np.float64)
    costs = np.array([obs.cost for obs in ordered], dtype=np.float64)
    score_range = max(scores.max() - scores.min(), EPSILON)
    cost_range = max(costs.max() - costs.min(), EPSILON)
    max_score = scores[-1]
    keep = ordered[:]
    for idx in range(len(ordered) - 1, 1, -1):
        if scores[idx - 1] < pruning_stop_score_fraction * max_score:
            break
        score_gain = (scores[idx] - scores[idx - 1]) / score_range
        cost_gain = (costs[idx] - costs[idx - 1]) / cost_range
        efficiency = score_gain / (cost_gain + EPSILON)
        if efficiency < efficiency_threshold:
            keep.pop(idx)
        else:
            break
    return keep


class ProteinLite:
    def __init__(
        self,
        specs: dict[str, ParamSpec],
        *,
        seed: int,
        warm_start_suggestions: int,
        candidate_samples: int,
        max_observations: int,
        suggestions_per_center: int,
        prune_pareto: bool,
        gp_alpha: float,
        target_cost_ratios: tuple[float, ...],
    ):
        self.space = SearchSpace(specs)
        self.seed = seed
        self.warm_start_suggestions = warm_start_suggestions
        self.candidate_samples = candidate_samples
        self.max_observations = max_observations
        self.suggestions_per_center = suggestions_per_center
        self.prune_enabled = prune_pareto
        self.gp_alpha = gp_alpha
        self.target_cost_ratios = target_cost_ratios
        self.observations: list[Observation] = []
        self.top_observations: list[Observation] = []

    def _rng_for(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + 17_431 * (index + 1))

    def _sobol_sample(self, index: int) -> np.ndarray:
        sobol = Sobol(d=self.space.dimension, scramble=True, seed=self.seed)
        if index > 0:
            sobol.fast_forward(index)
        return 2 * sobol.random(1)[0] - 1

    def _sample_observations(self) -> list[Observation]:
        if len(self.observations) <= self.max_observations:
            return list(self.observations)
        recent_count = self.max_observations // 2
        older = self.observations[:-recent_count]
        recent = self.observations[-recent_count:]
        rng = np.random.default_rng(self.seed + 9_973 * len(self.observations))
        selected = rng.choice(len(older), size=self.max_observations - recent_count, replace=False)
        sampled_older = [older[idx] for idx in np.sort(selected)]
        return sampled_older + recent

    def _train_gps(self, observations: list[Observation]) -> tuple[GaussianProcessRegressor | None, GaussianProcessRegressor | None, dict[str, float]]:
        if len(observations) < 3:
            return None, None, {}
        x = np.stack([obs.input for obs in observations], axis=0)
        scores = np.array([obs.output for obs in observations], dtype=np.float64)
        costs = np.array([obs.cost for obs in observations], dtype=np.float64)

        score_min, score_max = scores.min(), scores.max()
        log_costs = np.log(np.maximum(costs, EPSILON))
        cost_min, cost_max = log_costs.min(), log_costs.max()
        score_norm = (scores - score_min) / (abs(score_max - score_min) + EPSILON)
        cost_norm = (log_costs - cost_min) / (abs(cost_max - cost_min) + EPSILON)

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * (
            DotProduct() + Matern(length_scale=np.ones(self.space.dimension), nu=1.5)
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score_gp = GaussianProcessRegressor(kernel=kernel, alpha=self.gp_alpha, normalize_y=False, random_state=self.seed)
            cost_gp = GaussianProcessRegressor(kernel=kernel, alpha=self.gp_alpha, normalize_y=False, random_state=self.seed + 1)
            score_gp.fit(x, score_norm)
            cost_gp.fit(x, cost_norm)

        stats = {
            "score_min": float(score_min),
            "score_max": float(score_max),
            "log_cost_min": float(cost_min),
            "log_cost_max": float(cost_max),
        }
        return score_gp, cost_gp, stats

    def _candidate_centers(self) -> np.ndarray:
        pareto = pareto_points(self.observations)
        if self.prune_enabled:
            pareto = prune_pareto_front(pareto)
        centers = [obs.input for obs in pareto] or [self.space.search_center]
        if self.top_observations:
            centers.extend(obs.input for obs in self.top_observations)
        return np.stack(centers, axis=0)

    def suggest(self, run_index: int) -> tuple[dict[str, float | int], dict[str, float]]:
        if run_index < self.warm_start_suggestions or len(self.observations) < 3:
            return self.space.to_mapping(self._sobol_sample(run_index)), {}

        rng = self._rng_for(run_index)
        sampled_observations = self._sample_observations()
        score_gp, cost_gp, stats = self._train_gps(sampled_observations)
        centers = self._candidate_centers()
        candidate_count = max(self.candidate_samples, len(centers) * self.suggestions_per_center)
        candidates = self.space.sample(candidate_count, rng, centers)

        if score_gp is None or cost_gp is None:
            best = candidates[0]
            return self.space.to_mapping(best), {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_score_norm = score_gp.predict(candidates)
            pred_cost_norm = cost_gp.predict(candidates)

        pred_scores = pred_score_norm * (stats["score_max"] - stats["score_min"]) + stats["score_min"]
        pred_log_cost = pred_cost_norm * (stats["log_cost_max"] - stats["log_cost_min"]) + stats["log_cost_min"]
        pred_costs = np.exp(pred_log_cost)
        target_ratio = self.target_cost_ratios[run_index % len(self.target_cost_ratios)]
        target_weight = np.clip(1.0 - np.abs(target_ratio - pred_cost_norm), 0.0, 1.0)
        ratings = pred_score_norm * target_weight
        best_idx = int(np.argmax(ratings))
        info = {
            "predicted_score": float(pred_scores[best_idx]),
            "predicted_cost": float(pred_costs[best_idx]),
            "predicted_rating": float(ratings[best_idx]),
        }
        return self.space.to_mapping(candidates[best_idx]), info

    def observe(self, params: dict[str, float | int], *, score: float, cost: float, metadata: dict[str, object] | None = None) -> None:
        observation = Observation(
            input=self.space.from_mapping(params),
            output=float(score),
            cost=float(cost),
            metadata=metadata,
        )
        self.observations.append(observation)
        self.top_observations.append(observation)
        self.top_observations.sort(key=lambda obs: obs.output, reverse=True)
        self.top_observations = self.top_observations[:5]

