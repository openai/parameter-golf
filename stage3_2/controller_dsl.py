from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


FEATURES = ("progress", "step_avg_ms", "train_loss_slope", "warmdown_frac")
OPS = (">", "<")
PHASES = ("early", "mid", "late")
SNAPSHOT_SCORES = ("deployed", "raw")
SNAPSHOT_MODES = ("ema", "last", "best_deployed_last_k", "best_raw_last_k")
PULSE_MODES = ("export_surrogate", "late_qat")

ACTION_BOUNDS: dict[str, tuple[float, float]] = {
    "ema_decay": (0.995, 0.9999),
    "qat_alpha": (0.0, 1.0),
    "export_surrogate_weight": (0.0, 0.002),
    "checkpoint_capture_rate": (25.0, 250.0),
    "token_lr_mult": (0.0, 1.5),
    "matrix_lr_mult": (0.5, 1.5),
    "scalar_lr_mult": (0.0, 1.5),
    "head_lr_mult": (0.0, 1.5),
    "freeze_token": (0.0, 1.0),
    "freeze_head": (0.0, 1.0),
    "checkpoint_selection_mode": (0.0, 0.0),
}

ACTION_KEYS = tuple(ACTION_BOUNDS.keys())
ACTION_TYPES: dict[str, str] = {
    "ema_decay": "float",
    "qat_alpha": "float",
    "export_surrogate_weight": "float",
    "checkpoint_capture_rate": "int",
    "token_lr_mult": "float",
    "matrix_lr_mult": "float",
    "scalar_lr_mult": "float",
    "head_lr_mult": "float",
    "freeze_token": "bool",
    "freeze_head": "bool",
    "checkpoint_selection_mode": "mode",
}

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "progress": (0.0, 1.0),
    "warmdown_frac": (0.0, 1.0),
    "step_avg_ms": (300.0, 1400.0),
    "train_loss_slope": (-0.1, 0.1),
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_action_value(action: str, value: Any) -> Any:
    if action == "checkpoint_selection_mode":
        if value not in SNAPSHOT_MODES:
            return "ema"
        return value
    kind = ACTION_TYPES[action]
    lo, hi = ACTION_BOUNDS[action]
    if kind == "bool":
        return int(bool(int(value)))
    if kind == "int":
        return int(round(clamp(float(value), lo, hi)))
    return float(clamp(float(value), lo, hi))


@dataclass(frozen=True)
class GateSpec:
    feature: str
    op: str
    threshold: float
    action: str
    value: Any

    def __post_init__(self) -> None:
        if self.feature not in FEATURES:
            raise ValueError(f"unknown gate feature: {self.feature}")
        if self.op not in OPS:
            raise ValueError(f"unknown gate op: {self.op}")
        if self.action not in ACTION_KEYS:
            raise ValueError(f"unknown gate action: {self.action}")
        lo, hi = FEATURE_BOUNDS[self.feature]
        object.__setattr__(self, "threshold", clamp(float(self.threshold), lo, hi))
        object.__setattr__(self, "value", normalize_action_value(self.action, self.value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "op": self.op,
            "threshold": self.threshold,
            "action": self.action,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GateSpec":
        return cls(
            feature=str(data["feature"]),
            op=str(data["op"]),
            threshold=float(data["threshold"]),
            action=str(data["action"]),
            value=data["value"],
        )


@dataclass(frozen=True)
class SnapshotSpec:
    every: int = 100
    start_frac: float = 0.75
    last_k: int = 6
    score: str = "deployed"
    mode: str = "ema"

    def __post_init__(self) -> None:
        if self.score not in SNAPSHOT_SCORES:
            raise ValueError(f"unknown snapshot score: {self.score}")
        if self.mode not in SNAPSHOT_MODES:
            raise ValueError(f"unknown snapshot mode: {self.mode}")
        object.__setattr__(self, "every", int(clamp(float(self.every), 25, 250)))
        object.__setattr__(self, "start_frac", clamp(float(self.start_frac), 0.5, 0.95))
        object.__setattr__(self, "last_k", int(clamp(float(self.last_k), 2, 12)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "every": self.every,
            "start_frac": self.start_frac,
            "last_k": self.last_k,
            "score": self.score,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotSpec":
        return cls(
            every=int(data.get("every", 100)),
            start_frac=float(data.get("start_frac", 0.75)),
            last_k=int(data.get("last_k", 6)),
            score=str(data.get("score", "deployed")),
            mode=str(data.get("mode", "ema")),
        )


@dataclass(frozen=True)
class PulseSpec:
    every: int = 8
    late_start: float = 0.72
    mode: str = "export_surrogate"
    weight: float = 0.0007

    def __post_init__(self) -> None:
        if self.mode not in PULSE_MODES:
            raise ValueError(f"unknown pulse mode: {self.mode}")
        object.__setattr__(self, "every", int(clamp(float(self.every), 2, 32)))
        object.__setattr__(self, "late_start", clamp(float(self.late_start), 0.5, 0.95))
        object.__setattr__(self, "weight", clamp(float(self.weight), 0.0, 0.002))

    def to_dict(self) -> dict[str, Any]:
        return {
            "every": self.every,
            "late_start": self.late_start,
            "mode": self.mode,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PulseSpec":
        return cls(
            every=int(data.get("every", 8)),
            late_start=float(data.get("late_start", 0.72)),
            mode=str(data.get("mode", "export_surrogate")),
            weight=float(data.get("weight", 0.0007)),
        )


@dataclass(frozen=True)
class ControllerSpec:
    phase_boundaries: tuple[float, float] = (0.6, 0.82)
    phase_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    gates: tuple[GateSpec, ...] = ()
    snapshot: SnapshotSpec | None = None
    pulse: PulseSpec | None = None

    def __post_init__(self) -> None:
        b1, b2 = self.phase_boundaries
        b1 = clamp(float(b1), 0.4, 0.9)
        b2 = clamp(float(b2), b1 + 0.05, 0.95)
        object.__setattr__(self, "phase_boundaries", (b1, b2))
        cleaned: dict[str, dict[str, Any]] = {}
        for phase in PHASES:
            raw = self.phase_defaults.get(phase, {})
            phase_dict: dict[str, Any] = {}
            for action, value in raw.items():
                if action not in ACTION_KEYS:
                    continue
                phase_dict[action] = normalize_action_value(action, value)
            cleaned[phase] = phase_dict
        object.__setattr__(self, "phase_defaults", cleaned)
        if len(self.gates) > 2:
            raise ValueError("controller supports at most 2 gates in the initial search")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "phase_boundaries": list(self.phase_boundaries),
            "phase_defaults": self.phase_defaults,
        }
        if self.gates:
            data["gates"] = [gate.to_dict() for gate in self.gates]
        if self.snapshot is not None:
            data["snapshot"] = self.snapshot.to_dict()
        if self.pulse is not None:
            data["pulse"] = self.pulse.to_dict()
        return data

    def to_env_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def canonical_json(self) -> str:
        canonical = self.to_dict()
        if "gates" in canonical:
            canonical["gates"] = sorted(
                canonical["gates"],
                key=lambda gate: (
                    gate["feature"],
                    gate["op"],
                    float(gate["threshold"]),
                    gate["action"],
                    str(gate["value"]),
                ),
            )
        return json.dumps(canonical, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControllerSpec":
        gates = tuple(GateSpec.from_dict(g) for g in data.get("gates", []))
        snapshot = SnapshotSpec.from_dict(data["snapshot"]) if "snapshot" in data else None
        pulse = PulseSpec.from_dict(data["pulse"]) if "pulse" in data else None
        return cls(
            phase_boundaries=(float(data["phase_boundaries"][0]), float(data["phase_boundaries"][1])),
            phase_defaults={phase: dict(values) for phase, values in data.get("phase_defaults", {}).items()},
            gates=gates,
            snapshot=snapshot,
            pulse=pulse,
        )
