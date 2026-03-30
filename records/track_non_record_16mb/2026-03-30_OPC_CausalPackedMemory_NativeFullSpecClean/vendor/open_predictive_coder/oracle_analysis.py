from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .artifacts import (
    ArtifactAccounting,
    ArtifactMetadata,
    make_artifact_accounting,
    make_replay_span,
)
from .codecs import ensure_tokens
from .config import HierarchicalSubstrateConfig, OpenPredictiveCoderConfig, SampledReadoutBandConfig, SampledReadoutConfig
from .bidirectional_context import BidirectionalContextConfig, BidirectionalContextProbe, BidirectionalContextStats
from .control import ControllerSummary
from .hierarchical import HierarchicalSubstrate
from .hierarchical_views import HierarchicalFeatureView
from .metrics import bits_per_byte_from_probabilities
from .presets import hierarchical_small
from .routing import RoutingConfig, SummaryRouter
from .sampled_readout import SampledMultiscaleReadout
from .train_modes import TrainModeConfig


def _resolve_hierarchical_config(model: OpenPredictiveCoderConfig) -> HierarchicalSubstrateConfig:
    if model.substrate_kind != "hierarchical":
        raise ValueError("oracle analysis requires a hierarchical model config")
    return model.hierarchical


def _alignment_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    width = min(left.size, right.size)
    if width == 0:
        return 0.0, 0.0, 0.0, 0.0

    left = left[:width]
    right = right[:width]
    diff = left - right
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))

    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    denom = left_norm * right_norm
    if denom == 0.0:
        pearson = 1.0 if np.allclose(left, right) else 0.0
    else:
        pearson = float(np.clip(float(np.dot(left_centered, right_centered) / denom), -1.0, 1.0))

    cosine_denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if cosine_denom == 0.0:
        cosine = 1.0 if np.allclose(left, right) else 0.0
    else:
        cosine = float(np.clip(float(np.dot(left, right) / cosine_denom), -1.0, 1.0))

    return pearson, cosine, mae, rmse


@dataclass(frozen=True)
class OracleAnalysisConfig:
    model: OpenPredictiveCoderConfig = field(default_factory=hierarchical_small)
    train_mode: TrainModeConfig = field(
        default_factory=lambda: TrainModeConfig(
            state_mode="through_state",
            slow_update_stride=3,
            rollout_checkpoints=(8, 16, 24),
            rollout_checkpoint_stride=12,
        )
    )
    fast_sample_size: int = 8
    mid_sample_size: int = 8
    slow_sample_size: int = 12
    route_oracle_bias: float = 0.05
    route_temperature: float = 1.0
    bidirectional_context: BidirectionalContextConfig | None = None

    def __post_init__(self) -> None:
        hierarchical = _resolve_hierarchical_config(self.model)
        if self.fast_sample_size < 1 or self.fast_sample_size > hierarchical.fast_size:
            raise ValueError("fast_sample_size must lie within the fast bank size")
        if self.mid_sample_size < 1 or self.mid_sample_size > hierarchical.mid_size:
            raise ValueError("mid_sample_size must lie within the mid bank size")
        if self.slow_sample_size < 1 or self.slow_sample_size > hierarchical.slow_size:
            raise ValueError("slow_sample_size must lie within the slow bank size")
        if self.route_temperature <= 0.0:
            raise ValueError("route_temperature must be > 0")
        if self.bidirectional_context is not None:
            if self.bidirectional_context.left_order < 0:
                raise ValueError("bidirectional_context.left_order must be >= 0")
            if self.bidirectional_context.right_order < 0:
                raise ValueError("bidirectional_context.right_order must be >= 0")


@dataclass(frozen=True)
class OracleAnalysisPoint:
    checkpoint: int
    slow_update_active: bool
    route_names: tuple[str, ...]
    route_weights: np.ndarray
    selected_route: str
    alignment_pearson: float
    alignment_cosine: float
    alignment_mae: float
    alignment_rmse: float
    route_bits_per_byte: float

    def __post_init__(self) -> None:
        route_weights = np.asarray(self.route_weights, dtype=np.float64).reshape(-1)
        if route_weights.size < 1:
            raise ValueError("OracleAnalysisPoint requires route weights")
        object.__setattr__(self, "route_weights", route_weights)


@dataclass(frozen=True)
class OracleAnalysisReport:
    tokens: int
    checkpoints: tuple[int, ...]
    points: tuple[OracleAnalysisPoint, ...]
    mean_alignment_pearson: float
    mean_alignment_cosine: float
    mean_alignment_mae: float
    mean_alignment_rmse: float
    mean_route_bits_per_byte: float
    oracle_preference_rate: float
    accounting: ArtifactAccounting
    bidirectional_context: BidirectionalContextStats | None = None

    @property
    def bits_per_byte(self) -> float:
        return self.mean_route_bits_per_byte


@dataclass(frozen=True)
class OracleAnalysisFitReport:
    sequences: int
    tokens: int
    train_bits_per_byte: float
    mean_alignment_pearson: float
    mean_alignment_cosine: float
    mean_alignment_mae: float
    mean_alignment_rmse: float
    oracle_preference_rate: float
    accounting: ArtifactAccounting
    bidirectional_context: BidirectionalContextStats | None = None

    @property
    def bits_per_byte(self) -> float:
        return self.train_bits_per_byte


class OracleAnalysisAdapter:
    def __init__(
        self,
        config: OracleAnalysisConfig | None = None,
        *,
        artifact_name: str = "oracle_analysis",
        metadata: ArtifactMetadata | None = None,
    ):
        self.config = config or OracleAnalysisConfig()
        hierarchical = self.config.model.hierarchical
        self.train_mode = self.config.train_mode
        self.bidirectional_probe = (
            BidirectionalContextProbe(self.config.bidirectional_context)
            if self.config.bidirectional_context is not None
            else None
        )
        self.substrate = HierarchicalSubstrate(hierarchical)
        self.feature_view = HierarchicalFeatureView(hierarchical)
        self.sampled_readout = SampledMultiscaleReadout(
            SampledReadoutConfig(
                state_dim=hierarchical.state_dim,
                seed=hierarchical.seed + 31,
                bands=(
                    SampledReadoutBandConfig(
                        name="fast",
                        start=0,
                        stop=hierarchical.fast_size,
                        sample_count=self.config.fast_sample_size,
                        include_mean=True,
                        include_energy=True,
                        include_drift=True,
                    ),
                    SampledReadoutBandConfig(
                        name="mid",
                        start=hierarchical.fast_size,
                        stop=hierarchical.fast_size + hierarchical.mid_size,
                        sample_count=self.config.mid_sample_size,
                        include_mean=True,
                        include_energy=True,
                        include_drift=True,
                    ),
                    SampledReadoutBandConfig(
                        name="slow",
                        start=hierarchical.fast_size + hierarchical.mid_size,
                        stop=hierarchical.state_dim,
                        sample_count=self.config.slow_sample_size,
                        include_mean=True,
                        include_energy=True,
                        include_drift=True,
                    ),
                ),
            )
        )
        feature_dim = self.feature_view.feature_dim + self.sampled_readout.feature_dim
        projection_weights = np.linspace(1.0, 0.25, num=feature_dim, dtype=np.float64)
        self.router = SummaryRouter(
            RoutingConfig(
                mode="projection",
                projection_weights=tuple(float(value) for value in projection_weights),
                route_biases=(0.0, self.config.route_oracle_bias),
                temperature=self.config.route_temperature,
            )
        )
        self.artifact_name = artifact_name
        self.metadata = metadata or ArtifactMetadata()
        self._last_fit_accounting = make_artifact_accounting(
            self.artifact_name,
            0,
            0,
            metadata=self.metadata,
            tokens=0,
            comparisons=0,
            oracle_selected=0,
        )

    @staticmethod
    def _coerce_sequences(
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> tuple[np.ndarray, ...]:
        if isinstance(data, (str, bytes, bytearray, memoryview, np.ndarray)):
            return (ensure_tokens(data),)
        if isinstance(data, Sequence) and data and all(isinstance(item, int) for item in data):
            return (ensure_tokens(data),)
        if isinstance(data, Sequence):
            return tuple(ensure_tokens(item) for item in data)
        return (ensure_tokens(data),)

    def _scan_states(self, tokens: np.ndarray) -> list[np.ndarray]:
        state = self.substrate.initial_state()
        states = [state.copy()]
        for token in tokens:
            state = self.substrate.step(state, int(token))
            states.append(state.copy())
        return states

    def _encode_state(self, state: np.ndarray, previous_state: np.ndarray | None) -> np.ndarray:
        previous = previous_state if self.train_mode.uses_through_state else None
        return np.concatenate(
            [
                self.feature_view.encode(state, previous_state=previous),
                self.sampled_readout.encode(state, previous_state=previous),
            ]
        )

    @staticmethod
    def _combine_bidirectional_stats(stats: Sequence[BidirectionalContextStats]) -> BidirectionalContextStats | None:
        if not stats:
            return None
        candidate_sizes = tuple(size for stat in stats for size in stat.candidate_sizes)
        neighborhoods = tuple(neighborhood for stat in stats for neighborhood in stat.neighborhoods)
        sequence_length = sum(stat.sequence_length for stat in stats)
        neighborhood_count = sum(stat.neighborhood_count for stat in stats)
        left_context_count = sum(stat.left_context_count for stat in stats)
        right_context_count = sum(stat.right_context_count for stat in stats)
        pair_context_count = sum(stat.pair_context_count for stat in stats)
        return BidirectionalContextStats(
            sequence_length=sequence_length,
            neighborhood_count=neighborhood_count,
            left_context_count=left_context_count,
            right_context_count=right_context_count,
            pair_context_count=pair_context_count,
            deterministic_fraction=float(np.mean([stat.deterministic_fraction for stat in stats])),
            candidate_le_2_rate=float(np.mean([stat.candidate_le_2_rate for stat in stats])),
            candidate_le_4_rate=float(np.mean([stat.candidate_le_4_rate for stat in stats])),
            candidate_le_8_rate=float(np.mean([stat.candidate_le_8_rate for stat in stats])),
            mean_candidate_size=float(np.mean([stat.mean_candidate_size for stat in stats])),
            median_candidate_size=float(np.median(candidate_sizes)) if candidate_sizes else 0.0,
            max_candidate_size=max((stat.max_candidate_size for stat in stats), default=0),
            mean_left_support=float(np.mean([stat.mean_left_support for stat in stats])),
            mean_right_support=float(np.mean([stat.mean_right_support for stat in stats])),
            mean_pair_support=float(np.mean([stat.mean_pair_support for stat in stats])),
            candidate_sizes=candidate_sizes,
            neighborhoods=neighborhoods,
        )

    def _make_accounting(
        self,
        tokens: np.ndarray,
        points: Sequence[OracleAnalysisPoint],
        *,
        checkpoint_values: Sequence[int],
    ) -> ArtifactAccounting:
        oracle_spans = [
            make_replay_span(
                checkpoint - 1,
                checkpoint,
                label="oracle",
                checkpoint=int(checkpoint),
                route_bits_per_byte=float(point.route_bits_per_byte),
                selected_route=point.selected_route,
            )
            for checkpoint, point in zip(checkpoint_values, points)
            if point.selected_route == "oracle"
        ]
        oracle_selected = sum(int(point.selected_route == "oracle") for point in points)
        return make_artifact_accounting(
            self.artifact_name,
            int(tokens.size),
            oracle_selected,
            replay_spans=tuple(oracle_spans),
            metadata=self.metadata,
            tokens=int(tokens.size),
            comparisons=len(points),
            oracle_selected=oracle_selected,
        )

    def compare(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> OracleAnalysisReport:
        tokens = ensure_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        total_steps = int(tokens.size)
        checkpoints = self.train_mode.resolve_rollout_checkpoints(total_steps)
        forward_states = self._scan_states(tokens)
        reverse_states = self._scan_states(tokens[::-1])
        bidirectional_context = self.bidirectional_probe.scan(tokens) if self.bidirectional_probe is not None else None

        points: list[OracleAnalysisPoint] = []
        pearsons: list[float] = []
        cosines: list[float] = []
        maes: list[float] = []
        rmses: list[float] = []
        route_bits: list[float] = []
        oracle_selected = 0

        for checkpoint in checkpoints:
            suffix_len = total_steps - checkpoint
            causal_state = forward_states[checkpoint]
            causal_prev = forward_states[checkpoint - 1] if checkpoint > 0 else None
            oracle_state = reverse_states[suffix_len]
            oracle_prev = reverse_states[suffix_len - 1] if suffix_len > 0 else None

            causal_feature = self._encode_state(causal_state, causal_prev)
            oracle_feature = self._encode_state(oracle_state, oracle_prev)
            decision = self.router.route(
                (
                    ControllerSummary(causal_feature, name="causal"),
                    ControllerSummary(oracle_feature, name="oracle"),
                ),
                names=("causal", "oracle"),
            )
            alignment_pearson, alignment_cosine, alignment_mae, alignment_rmse = _alignment_metrics(
                causal_feature,
                oracle_feature,
            )
            route_bits_per_byte = bits_per_byte_from_probabilities(
                decision.weights[None, :],
                np.asarray([decision.selected_index], dtype=np.int64),
            )
            selected_route = decision.route_names[decision.selected_index]
            oracle_selected += int(selected_route == "oracle")

            points.append(
                OracleAnalysisPoint(
                    checkpoint=checkpoint,
                    slow_update_active=self.train_mode.should_update_slow(max(checkpoint - 1, 0)),
                    route_names=decision.route_names,
                    route_weights=decision.weights.copy(),
                    selected_route=selected_route,
                    alignment_pearson=alignment_pearson,
                    alignment_cosine=alignment_cosine,
                    alignment_mae=alignment_mae,
                    alignment_rmse=alignment_rmse,
                    route_bits_per_byte=route_bits_per_byte,
                )
            )
            pearsons.append(alignment_pearson)
            cosines.append(alignment_cosine)
            maes.append(alignment_mae)
            rmses.append(alignment_rmse)
            route_bits.append(route_bits_per_byte)

        return OracleAnalysisReport(
            tokens=total_steps,
            checkpoints=checkpoints,
            points=tuple(points),
            mean_alignment_pearson=float(np.mean(pearsons)),
            mean_alignment_cosine=float(np.mean(cosines)),
            mean_alignment_mae=float(np.mean(maes)),
            mean_alignment_rmse=float(np.mean(rmses)),
            mean_route_bits_per_byte=float(np.mean(route_bits)),
            oracle_preference_rate=float(oracle_selected / max(len(points), 1)),
            bidirectional_context=bidirectional_context,
            accounting=self._make_accounting(tokens, points, checkpoint_values=checkpoints),
        )

    analyze = compare

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> OracleAnalysisReport:
        return self.compare(sequence)

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> OracleAnalysisFitReport:
        sequences = self._coerce_sequences(data)
        total_tokens = 0
        total_points = 0
        weighted_bits = 0.0
        alignment_pearsons: list[float] = []
        alignment_cosines: list[float] = []
        alignment_maes: list[float] = []
        alignment_rmses: list[float] = []
        oracle_selected = 0
        bidirectional_contexts: list[BidirectionalContextStats] = []
        artifact_bytes = 0
        replay_bytes = 0
        replay_spans = []
        offset = 0

        for sequence in sequences:
            tokens = ensure_tokens(sequence)
            report = self.compare(tokens)
            total_tokens += int(tokens.size)
            total_points += len(report.points)
            weighted_bits += report.bits_per_byte * len(report.points)
            alignment_pearsons.append(report.mean_alignment_pearson)
            alignment_cosines.append(report.mean_alignment_cosine)
            alignment_maes.append(report.mean_alignment_mae)
            alignment_rmses.append(report.mean_alignment_rmse)
            oracle_selected += sum(int(point.selected_route == "oracle") for point in report.points)
            if report.bidirectional_context is not None:
                bidirectional_contexts.append(report.bidirectional_context)

            accounting = report.accounting
            artifact_bytes += accounting.artifact_bytes
            replay_bytes += accounting.replay_bytes
            replay_spans.extend(
                make_replay_span(
                    span.start + offset,
                    span.stop + offset,
                    label=span.label,
                    metadata=span.metadata,
                )
                for span in accounting.replay_spans
            )
            offset += int(tokens.size)

        mean_bits = 0.0 if total_points == 0 else weighted_bits / float(total_points)
        fit_accounting = make_artifact_accounting(
            self.artifact_name,
            artifact_bytes,
            replay_bytes,
            replay_spans=tuple(replay_spans),
            metadata=self.metadata,
            tokens=total_tokens,
            comparisons=total_points,
            oracle_selected=oracle_selected,
        )
        self._last_fit_accounting = fit_accounting
        return OracleAnalysisFitReport(
            sequences=len(sequences),
            tokens=total_tokens,
            train_bits_per_byte=mean_bits,
            mean_alignment_pearson=float(np.mean(alignment_pearsons)) if alignment_pearsons else 0.0,
            mean_alignment_cosine=float(np.mean(alignment_cosines)) if alignment_cosines else 0.0,
            mean_alignment_mae=float(np.mean(alignment_maes)) if alignment_maes else 0.0,
            mean_alignment_rmse=float(np.mean(alignment_rmses)) if alignment_rmses else 0.0,
            oracle_preference_rate=0.0 if total_points == 0 else oracle_selected / float(total_points),
            bidirectional_context=self._combine_bidirectional_stats(bidirectional_contexts),
            accounting=fit_accounting,
        )

    def accounting(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | None = None,
    ) -> ArtifactAccounting:
        if sequence is None:
            return self._last_fit_accounting
        tokens = ensure_tokens(sequence)
        if tokens.size == 0:
            return make_artifact_accounting(
                self.artifact_name,
                0,
                0,
                metadata=self.metadata,
                tokens=0,
                comparisons=0,
                oracle_selected=0,
            )
        return self.compare(tokens).accounting


__all__ = [
    "OracleAnalysisAdapter",
    "OracleAnalysisConfig",
    "OracleAnalysisFitReport",
    "OracleAnalysisPoint",
    "OracleAnalysisReport",
]
