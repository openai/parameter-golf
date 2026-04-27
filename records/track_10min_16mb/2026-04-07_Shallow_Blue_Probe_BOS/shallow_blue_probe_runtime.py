from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROBE_FEATURE_NAMES = (
    "backbone_agrees_with_ngram",
    "ngram_log_support",
    "ngram_prediction_prob",
    "ngram_top2_prediction_prob",
    "ngram_top1_top2_margin",
    "ngram_log_unique_continuations",
    "ngram_continuation_entropy_bits",
    "lz_triggered",
    "lz_log_match_len",
    "lz_log_support",
    "lz_prediction_prob",
    "backbone_entropy_bits",
    "backbone_top1_prob",
    "prefix_log_tokens",
    "regime_repeat_fraction",
)


@dataclass(frozen=True)
class ProbeRuntimeArtifact:
    feature_names: tuple[str, ...]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    hidden_dim: int
    fc1_weight: np.ndarray
    fc1_bias: np.ndarray
    fc2_weight: np.ndarray
    fc2_bias: float
    selected_threshold: float
    ngram_backoff_eps: float
    policy_kind: str = "hard_gate"
    deployment_alpha_lo: float | None = None
    deployment_alpha_hi: float | None = None
    selected_target_accept_share: float | None = None
    calibration_min_seq_pos_in_chunk: int | None = None
    source_summary_path: str | None = None

    def __post_init__(self) -> None:
        scaled_weight = self.fc1_weight / self.feature_stds[np.newaxis, :]
        adjusted_bias = self.fc1_bias - (
            self.fc1_weight @ (self.feature_means / self.feature_stds)
        )
        object.__setattr__(self, "_fc1_scaled_weight", scaled_weight)
        object.__setattr__(self, "_fc1_adjusted_bias", adjusted_bias)

    @property
    def safe_alpha_scale(self) -> float:
        return max(0.0, 1.0 - float(self.ngram_backoff_eps))

    def predict_score(self, feature_vector: np.ndarray) -> float:
        if isinstance(feature_vector, np.ndarray) and feature_vector.dtype == np.float64:
            x = feature_vector
        else:
            x = np.asarray(feature_vector, dtype=np.float64)
        if x.shape != self.feature_means.shape:
            raise ValueError(
                f"feature vector shape {x.shape} does not match expected "
                f"{self.feature_means.shape}"
            )
        hidden = self._fc1_scaled_weight @ x
        hidden += self._fc1_adjusted_bias
        np.maximum(hidden, 0.0, out=hidden)
        return float(self.fc2_weight @ hidden + self.fc2_bias)

    def accepts(self, feature_vector: np.ndarray) -> bool:
        return self.predict_score(feature_vector) >= float(self.selected_threshold)

    @property
    def uses_two_level_uplift(self) -> bool:
        return str(self.policy_kind) == "two_level_uplift"


@dataclass
class ProbePrefixState:
    """Track cartography-aligned prefix features over scored document rows."""

    scored_prefix_count: int = 0
    target_token_counts: Counter[int] = field(default_factory=Counter)

    def reset(self) -> None:
        self.scored_prefix_count = 0
        self.target_token_counts.clear()

    def prefix_nonbos_tokens(self) -> int:
        return int(self.scored_prefix_count)

    def regime_repeat_fraction(self) -> float:
        prefix_nonbos_tokens = int(self.scored_prefix_count)
        if prefix_nonbos_tokens <= 0:
            return 0.0
        unique_prefix_tokens = len(self.target_token_counts)
        return 1.0 - (float(unique_prefix_tokens) / float(prefix_nonbos_tokens))

    def update_with_target(self, target_token_id: int) -> None:
        self.target_token_counts[int(target_token_id)] += 1
        self.scored_prefix_count += 1


def _as_float_array(values: list[float] | tuple[float, ...], *, ndim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"expected {ndim}D array, got shape {arr.shape}")
    return arr


def load_probe_runtime_artifact(path: str | Path) -> ProbeRuntimeArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProbeRuntimeArtifact(
        feature_names=tuple(str(name) for name in payload["feature_names"]),
        feature_means=_as_float_array(payload["feature_means"], ndim=1),
        feature_stds=_as_float_array(payload["feature_stds"], ndim=1),
        hidden_dim=int(payload["hidden_dim"]),
        fc1_weight=_as_float_array(payload["fc1_weight"], ndim=2),
        fc1_bias=_as_float_array(payload["fc1_bias"], ndim=1),
        fc2_weight=_as_float_array(payload["fc2_weight"], ndim=1),
        fc2_bias=float(payload["fc2_bias"]),
        selected_threshold=float(payload["selected_threshold"]),
        ngram_backoff_eps=float(payload.get("ngram_backoff_eps", 0.0)),
        policy_kind=str(payload.get("policy_kind", "hard_gate")),
        deployment_alpha_lo=(
            None
            if payload.get("deployment_alpha_lo") is None
            else float(payload["deployment_alpha_lo"])
        ),
        deployment_alpha_hi=(
            None
            if payload.get("deployment_alpha_hi") is None
            else float(payload["deployment_alpha_hi"])
        ),
        selected_target_accept_share=(
            None
            if payload.get("selected_target_accept_share") is None
            else float(payload["selected_target_accept_share"])
        ),
        calibration_min_seq_pos_in_chunk=(
            None
            if payload.get("calibration_min_seq_pos_in_chunk") is None
            else int(payload["calibration_min_seq_pos_in_chunk"])
        ),
        source_summary_path=payload.get("source_summary_path"),
    )


def save_probe_runtime_artifact(path: str | Path, artifact: ProbeRuntimeArtifact) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "shallow_blue_probe_tiny_mlp_v1",
        "feature_names": list(artifact.feature_names),
        "feature_means": artifact.feature_means.tolist(),
        "feature_stds": artifact.feature_stds.tolist(),
        "hidden_dim": int(artifact.hidden_dim),
        "fc1_weight": artifact.fc1_weight.tolist(),
        "fc1_bias": artifact.fc1_bias.tolist(),
        "fc2_weight": artifact.fc2_weight.tolist(),
        "fc2_bias": float(artifact.fc2_bias),
        "selected_threshold": float(artifact.selected_threshold),
        "ngram_backoff_eps": float(artifact.ngram_backoff_eps),
        "policy_kind": str(artifact.policy_kind),
        "deployment_alpha_lo": (
            None
            if artifact.deployment_alpha_lo is None
            else float(artifact.deployment_alpha_lo)
        ),
        "deployment_alpha_hi": (
            None
            if artifact.deployment_alpha_hi is None
            else float(artifact.deployment_alpha_hi)
        ),
        "selected_target_accept_share": (
            None
            if artifact.selected_target_accept_share is None
            else float(artifact.selected_target_accept_share)
        ),
        "calibration_min_seq_pos_in_chunk": (
            None
            if artifact.calibration_min_seq_pos_in_chunk is None
            else int(artifact.calibration_min_seq_pos_in_chunk)
        ),
        "source_summary_path": artifact.source_summary_path,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_primary_ngram_backoff_eps(path: str | Path) -> float | None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    safe_expert = payload.get("safe_expert")
    if not isinstance(safe_expert, dict):
        return None
    value = safe_expert.get("primary_backoff_eps")
    if value is None:
        return None
    return float(value)


def build_live_probe_feature_vector(
    *,
    ngram_prediction_token_id: int,
    ngram_support: int,
    ngram_prediction_prob: float,
    ngram_top2_prediction_prob: float,
    ngram_unique_continuations: int,
    ngram_continuation_entropy_bits: float,
    lz_triggered: int,
    lz_match_len: int,
    lz_support: int,
    lz_prediction_prob: float,
    backbone_top1_token_id: int,
    backbone_entropy_bits: float,
    backbone_top1_prob: float,
    prefix_nonbos_tokens: int,
    regime_repeat_fraction: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    top1_top2_margin = max(
        float(ngram_prediction_prob) - float(ngram_top2_prediction_prob),
        0.0,
    )
    if out is None:
        out = np.empty(len(PROBE_FEATURE_NAMES), dtype=np.float64)
    elif out.shape != (len(PROBE_FEATURE_NAMES),):
        raise ValueError(
            f"feature output buffer shape {out.shape} does not match "
            f"expected {(len(PROBE_FEATURE_NAMES),)}"
        )

    out[0] = 1.0 if int(backbone_top1_token_id) == int(ngram_prediction_token_id) else 0.0
    out[1] = math.log1p(float(ngram_support))
    out[2] = float(ngram_prediction_prob)
    out[3] = float(ngram_top2_prediction_prob)
    out[4] = float(top1_top2_margin)
    out[5] = math.log1p(float(ngram_unique_continuations))
    out[6] = float(ngram_continuation_entropy_bits)
    out[7] = float(lz_triggered)
    out[8] = math.log1p(float(lz_match_len))
    out[9] = math.log1p(float(lz_support))
    out[10] = float(lz_prediction_prob)
    out[11] = float(backbone_entropy_bits)
    out[12] = float(backbone_top1_prob)
    out[13] = math.log1p(float(prefix_nonbos_tokens))
    out[14] = float(regime_repeat_fraction)
    return out
