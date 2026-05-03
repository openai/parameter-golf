from __future__ import annotations

import copy
import random
from dataclasses import replace
from typing import Any

from controller_dsl import (
    ACTION_KEYS,
    ACTION_TYPES,
    FEATURES,
    FEATURE_BOUNDS,
    OPS,
    PULSE_MODES,
    SNAPSHOT_MODES,
    SNAPSHOT_SCORES,
    ControllerSpec,
    GateSpec,
    PulseSpec,
    SnapshotSpec,
    clamp,
)


def _mutate_numeric_value(action: str, value: Any, rng: random.Random) -> Any:
    kind = ACTION_TYPES[action]
    if kind == "mode":
        choices = [mode for mode in SNAPSHOT_MODES if mode != value]
        return rng.choice(choices) if choices else value
    if kind == "bool":
        return 0 if int(value) else 1
    base = float(value)
    if action == "ema_decay":
        return round(clamp(base + rng.uniform(-0.0008, 0.0008), 0.995, 0.9999), 6)
    if action == "qat_alpha":
        return round(clamp(base + rng.uniform(-0.18, 0.18), 0.0, 1.0), 3)
    if action == "export_surrogate_weight":
        return round(clamp(base + rng.uniform(-0.00035, 0.00035), 0.0, 0.002), 6)
    if action == "checkpoint_capture_rate":
        return int(clamp(round(base + rng.choice([-50, -25, 25, 50])), 25, 250))
    if action.endswith("_lr_mult"):
        return round(clamp(base + rng.uniform(-0.2, 0.2), 0.0, 1.5), 3)
    return value


def _copy_phase_defaults(spec: ControllerSpec) -> dict[str, dict[str, Any]]:
    return {phase: dict(values) for phase, values in spec.phase_defaults.items()}


def _bounded_threshold(feature: str, value: float) -> float:
    lo, hi = FEATURE_BOUNDS[feature]
    return round(clamp(value, lo, hi), 4)


def mutate_numeric(spec: ControllerSpec, rng: random.Random) -> tuple[ControllerSpec, dict[str, Any]]:
    options: list[tuple[str, str, str]] = []
    for phase, phase_actions in spec.phase_defaults.items():
        for action in phase_actions:
            options.append(("phase", phase, action))
    for gate_idx, gate in enumerate(spec.gates):
        options.append(("gate_threshold", str(gate_idx), gate.feature))
        options.append(("gate_value", str(gate_idx), gate.action))
    if spec.snapshot is not None:
        options.extend([("snapshot", "every", ""), ("snapshot", "start_frac", ""), ("snapshot", "last_k", "")])
    if spec.pulse is not None:
        options.extend([("pulse", "every", ""), ("pulse", "late_start", ""), ("pulse", "weight", "")])
    options.extend([("boundary", "0", ""), ("boundary", "1", "")])
    kind, index, field = rng.choice(options)

    if kind == "phase":
        phase_defaults = _copy_phase_defaults(spec)
        old = phase_defaults[index][field]
        phase_defaults[index][field] = _mutate_numeric_value(field, old, rng)
        new_spec = replace(spec, phase_defaults=phase_defaults)
        return new_spec, {"operator": "numeric", "description": f"{index}.{field} {old} -> {phase_defaults[index][field]}"}

    if kind == "gate_threshold":
        gates = list(spec.gates)
        gate = gates[int(index)]
        old = gate.threshold
        new_threshold = _bounded_threshold(gate.feature, old + rng.uniform(-0.12, 0.12) * (FEATURE_BOUNDS[gate.feature][1] - FEATURE_BOUNDS[gate.feature][0]))
        gates[int(index)] = replace(gate, threshold=new_threshold)
        new_spec = replace(spec, gates=tuple(gates))
        return new_spec, {"operator": "numeric", "description": f"gate[{index}] threshold {old} -> {new_threshold}"}

    if kind == "gate_value":
        gates = list(spec.gates)
        gate = gates[int(index)]
        old = gate.value
        new_value = _mutate_numeric_value(gate.action, old, rng)
        gates[int(index)] = replace(gate, value=new_value)
        new_spec = replace(spec, gates=tuple(gates))
        return new_spec, {"operator": "numeric", "description": f"gate[{index}] value {old} -> {new_value}"}

    if kind == "snapshot" and spec.snapshot is not None:
        snapshot = spec.snapshot
        if index == "every":
            new = replace(snapshot, every=int(clamp(snapshot.every + rng.choice([-25, 25, 50]), 25, 250)))
        elif index == "start_frac":
            new = replace(snapshot, start_frac=round(clamp(snapshot.start_frac + rng.uniform(-0.08, 0.08), 0.5, 0.95), 3))
        else:
            new = replace(snapshot, last_k=int(clamp(snapshot.last_k + rng.choice([-2, -1, 1, 2]), 2, 12)))
        return replace(spec, snapshot=new), {"operator": "numeric", "description": f"snapshot.{index} tuned"}

    if kind == "pulse" and spec.pulse is not None:
        pulse = spec.pulse
        if index == "every":
            new = replace(pulse, every=int(clamp(pulse.every + rng.choice([-4, -2, 2, 4]), 2, 32)))
        elif index == "late_start":
            new = replace(pulse, late_start=round(clamp(pulse.late_start + rng.uniform(-0.08, 0.08), 0.5, 0.95), 3))
        else:
            new = replace(pulse, weight=round(clamp(pulse.weight + rng.uniform(-0.0003, 0.0003), 0.0, 0.002), 6))
        return replace(spec, pulse=new), {"operator": "numeric", "description": f"pulse.{index} tuned"}

    b1, b2 = spec.phase_boundaries
    if index == "0":
        new_b1 = round(clamp(b1 + rng.uniform(-0.06, 0.06), 0.4, b2 - 0.05), 3)
        return replace(spec, phase_boundaries=(new_b1, b2)), {"operator": "numeric", "description": f"boundary0 {b1} -> {new_b1}"}
    new_b2 = round(clamp(b2 + rng.uniform(-0.06, 0.06), b1 + 0.05, 0.95), 3)
    return replace(spec, phase_boundaries=(b1, new_b2)), {"operator": "numeric", "description": f"boundary1 {b2} -> {new_b2}"}


def mutate_wiring(spec: ControllerSpec, rng: random.Random) -> tuple[ControllerSpec, dict[str, Any]]:
    choices = ["phase_action_move"]
    if spec.gates:
        choices.append("gate_feature")
        choices.append("gate_action")
        choices.append("gate_op")
    if spec.snapshot is not None:
        choices.append("snapshot_mode")
    if spec.pulse is not None:
        choices.append("pulse_mode")
    op = rng.choice(choices)

    if op == "phase_action_move":
        phase_defaults = _copy_phase_defaults(spec)
        populated = [phase for phase in phase_defaults if phase_defaults[phase]]
        if not populated:
            return mutate_numeric(spec, rng)
        source_phase = rng.choice(populated)
        action = rng.choice(list(phase_defaults[source_phase].keys()))
        target_phase = rng.choice([phase for phase in phase_defaults if phase != source_phase])
        value = phase_defaults[source_phase].pop(action)
        phase_defaults[target_phase][action] = value
        return replace(spec, phase_defaults=phase_defaults), {
            "operator": "wiring",
            "description": f"move {action} from {source_phase} to {target_phase}",
        }

    if op == "gate_feature":
        gates = list(spec.gates)
        idx = rng.randrange(len(gates))
        gate = gates[idx]
        feature = rng.choice([feature for feature in FEATURES if feature != gate.feature])
        lo, hi = FEATURE_BOUNDS[feature]
        threshold = clamp(gate.threshold, lo, hi)
        gates[idx] = replace(gate, feature=feature, threshold=threshold)
        return replace(spec, gates=tuple(gates)), {"operator": "wiring", "description": f"gate[{idx}] feature -> {feature}"}

    if op == "gate_action":
        gates = list(spec.gates)
        idx = rng.randrange(len(gates))
        gate = gates[idx]
        action = rng.choice([candidate for candidate in ACTION_KEYS if candidate != gate.action and candidate != "checkpoint_selection_mode"])
        value = 0.0 if action in {"export_surrogate_weight", "qat_alpha"} else 1.0
        gates[idx] = GateSpec(feature=gate.feature, op=gate.op, threshold=gate.threshold, action=action, value=value)
        return replace(spec, gates=tuple(gates)), {"operator": "wiring", "description": f"gate[{idx}] action -> {action}"}

    if op == "gate_op":
        gates = list(spec.gates)
        idx = rng.randrange(len(gates))
        gate = gates[idx]
        new_op = "<" if gate.op == ">" else ">"
        gates[idx] = replace(gate, op=new_op)
        return replace(spec, gates=tuple(gates)), {"operator": "wiring", "description": f"gate[{idx}] op {gate.op} -> {new_op}"}

    if op == "snapshot_mode" and spec.snapshot is not None:
        modes = [mode for mode in SNAPSHOT_MODES if mode != spec.snapshot.mode]
        mode = rng.choice(modes)
        score = spec.snapshot.score
        if mode == "best_raw_last_k":
            score = "raw"
        elif mode == "best_deployed_last_k":
            score = "deployed"
        return replace(spec, snapshot=replace(spec.snapshot, mode=mode, score=score)), {
            "operator": "wiring",
            "description": f"snapshot mode -> {mode}",
        }

    if op == "pulse_mode" and spec.pulse is not None:
        mode = rng.choice([candidate for candidate in PULSE_MODES if candidate != spec.pulse.mode])
        return replace(spec, pulse=replace(spec.pulse, mode=mode)), {"operator": "wiring", "description": f"pulse mode -> {mode}"}

    return mutate_numeric(spec, rng)


def mutate_structural(spec: ControllerSpec, rng: random.Random) -> tuple[ControllerSpec, dict[str, Any]]:
    choices = ["toggle_pulse", "toggle_gate", "toggle_phase_action", "snapshot_mode_jump"]
    op = rng.choice(choices)

    if op == "toggle_pulse":
        if spec.pulse is None:
            pulse = PulseSpec(every=8, late_start=0.72, mode="export_surrogate", weight=0.0006)
            return replace(spec, pulse=pulse), {"operator": "structural", "description": "add pulse block"}
        return replace(spec, pulse=None), {"operator": "structural", "description": "remove pulse block"}

    if op == "toggle_gate":
        gates = list(spec.gates)
        if gates and (len(gates) == 2 or rng.random() < 0.5):
            idx = rng.randrange(len(gates))
            removed = gates.pop(idx)
            return replace(spec, gates=tuple(gates)), {"operator": "structural", "description": f"remove gate on {removed.feature}->{removed.action}"}
        action = rng.choice(["export_surrogate_weight", "qat_alpha", "checkpoint_capture_rate", "matrix_lr_mult", "head_lr_mult"])
        feature = rng.choice(FEATURES)
        lo, hi = FEATURE_BOUNDS[feature]
        threshold = (lo + hi) / 2.0
        value: Any = 0.0 if action == "export_surrogate_weight" else 0.15 if action == "qat_alpha" else 75 if action == "checkpoint_capture_rate" else 1.1
        gates.append(GateSpec(feature=feature, op=rng.choice(OPS), threshold=threshold, action=action, value=value))
        return replace(spec, gates=tuple(gates[:2])), {"operator": "structural", "description": f"add gate {feature}->{action}"}

    if op == "toggle_phase_action":
        phase_defaults = _copy_phase_defaults(spec)
        phase = rng.choice(list(phase_defaults.keys()))
        candidate_actions = ["export_surrogate_weight", "qat_alpha", "token_lr_mult", "head_lr_mult", "checkpoint_capture_rate"]
        existing = set(phase_defaults[phase].keys())
        if existing and rng.random() < 0.5:
            action = rng.choice(list(existing))
            del phase_defaults[phase][action]
            return replace(spec, phase_defaults=phase_defaults), {"operator": "structural", "description": f"remove {action} from {phase}"}
        action = rng.choice([action for action in candidate_actions if action not in existing])
        default_value: Any = {
            "export_surrogate_weight": 0.0004,
            "qat_alpha": 0.25,
            "token_lr_mult": 0.0,
            "head_lr_mult": 0.5,
            "checkpoint_capture_rate": 100,
        }[action]
        phase_defaults[phase][action] = default_value
        return replace(spec, phase_defaults=phase_defaults), {"operator": "structural", "description": f"add {action} to {phase}"}

    if spec.snapshot is not None:
        mode = rng.choice([mode for mode in SNAPSHOT_MODES if mode != spec.snapshot.mode])
        score = "raw" if mode == "best_raw_last_k" else "deployed"
        if mode in {"ema", "last"}:
            score = spec.snapshot.score
        return replace(spec, snapshot=replace(spec.snapshot, mode=mode, score=score)), {
            "operator": "structural",
            "description": f"jump snapshot mode -> {mode}",
        }

    return mutate_numeric(spec, rng)


def mutate(spec: ControllerSpec, rng: random.Random) -> tuple[ControllerSpec, dict[str, Any]]:
    roll = rng.random()
    if roll < 0.6:
        return mutate_numeric(spec, rng)
    if roll < 0.9:
        return mutate_wiring(spec, rng)
    return mutate_structural(spec, rng)
