from __future__ import annotations

from typing import Any


TRACK_STAGE = {
    "local-smoke": "scout",
    "local-cuda-smoke": "confirm",
    "nonrecord-16mb": "candidate",
    "record-10min-16mb": "record_rehearsal",
}

NEXT_STAGE = {
    "scout": "confirm",
    "confirm": "candidate",
    "candidate": "record_rehearsal",
    "record_rehearsal": "record_rehearsal",
}


FAMILY_RULES: dict[str, dict[str, str]] = {
    "baseline_capture": {
        "expected_upside": "establish a trusted local anchor for later comparisons",
        "risk_level": "low",
        "kill_criteria": "invalidate only if the run fails to produce planner-eligible metrics",
        "promotion_rule": "promote only after a completed run establishes a clean local baseline",
    },
    "manual_override": {
        "expected_upside": "test a requested idea directly without planner interference",
        "risk_level": "medium",
        "kill_criteria": "discard if the override fails to improve the intended metric or breaks run validity",
        "promotion_rule": "retest once on the same profile, then promote only if planner-eligible and clearly better",
    },
    "quant_gap_reduction": {
        "expected_upside": "improve final roundtrip quality",
        "risk_level": "medium",
        "kill_criteria": "discard if final roundtrip does not improve or bytes grow with no compensating gain",
        "promotion_rule": "Promote if roundtrip quality improves and the final roundtrip improves while quant gap shrinks on the same profile",
    },
    "byte_reduction": {
        "expected_upside": "reduce total artifact bytes while preserving most of the final roundtrip score",
        "risk_level": "medium",
        "kill_criteria": "discard if bytes do not improve meaningfully or quality regresses sharply",
        "promotion_rule": "promote if bytes fall while roundtrip quality stays within a small regression band",
    },
    "export_policy": {
        "expected_upside": "trade a few bytes for cleaner roundtrip reconstruction",
        "risk_level": "medium",
        "kill_criteria": "discard if byte growth is not repaid by better final roundtrip quality",
        "promotion_rule": "promote if roundtrip gains clearly outweigh byte growth on repeatable runs",
    },
    "stability_under_quantization": {
        "expected_upside": "stabilize weights or activations so export damage is smaller",
        "risk_level": "low",
        "kill_criteria": "discard if stability probes reduce learning without improving roundtrip quality",
        "promotion_rule": "promote if final roundtrip improves with flat or better bytes",
    },
    "short_run_learning_speed": {
        "expected_upside": "increase progress per unit wallclock inside short scout budgets",
        "risk_level": "medium",
        "kill_criteria": "discard if faster updates only add noise or hurt the endpoint",
        "promotion_rule": "promote if the endpoint improves and the run stays stable enough for confirm-stage follow-up",
    },
    "signal_probe": {
        "expected_upside": "separate undertraining from genuinely bad settings",
        "risk_level": "low",
        "kill_criteria": "discard if the extra probe adds no new ranking signal",
        "promotion_rule": "promote the parent line only if the probe reveals clear remaining headroom",
    },
}


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower()).strip("_") or "unknown"


def _stage_for_profile(profile_name: str) -> str:
    if profile_name.startswith("torch_record_"):
        return TRACK_STAGE["record-10min-16mb"]
    if profile_name.startswith("torch_nonrecord_"):
        return TRACK_STAGE["nonrecord-16mb"]
    if profile_name.startswith("torch_single_gpu_"):
        return TRACK_STAGE["local-cuda-smoke"]
    return TRACK_STAGE["local-smoke"]


def funnel_stage_for_profile(profile_name: str) -> str:
    return _stage_for_profile(profile_name)


def next_funnel_stage(stage: str) -> str:
    return NEXT_STAGE.get(stage, stage)


def _family_from_inputs(
    *,
    mutation_name: str,
    mutation_kind: str,
    objective: str,
    code_mutation: dict[str, Any] | None,
) -> str:
    code_mutation_name = (code_mutation or {}).get("name")
    if mutation_kind == "baseline":
        return "baseline_capture"
    if code_mutation_name in {"quant_clip_tighter", "quant_keep_float_bigger"}:
        return "quant_gap_reduction"
    if code_mutation_name == "quant_keep_float_smaller":
        return "byte_reduction"
    if code_mutation_name == "quant_scale_fp32":
        return "export_policy"
    if mutation_name == "fallback" or "signal probe" in objective:
        return "signal_probe"
    if mutation_kind == "code" and code_mutation is None:
        return "manual_override"
    if objective == "reduce submission bytes":
        return "byte_reduction"
    if objective == "reduce quantization gap":
        if code_mutation and code_mutation.get("name") == "quant_scale_fp32":
            return "export_policy"
        if code_mutation:
            return "quant_gap_reduction"
        return "stability_under_quantization"
    if objective == "stability probe":
        return "stability_under_quantization"
    if objective in {"optimizer tuning", "schedule tuning"}:
        return "short_run_learning_speed"
    return "manual_override" if mutation_kind == "code" else "signal_probe"


def build_hypothesis_metadata(
    *,
    profile_name: str,
    mutation_name: str,
    mutation_kind: str,
    objective: str,
    hypothesis: str,
    code_mutation: dict[str, Any] | None,
    parent_hypothesis_id: str | None,
    parent_lineage_id: str | None,
) -> dict[str, Any]:
    family = _family_from_inputs(
        mutation_name=mutation_name,
        mutation_kind=mutation_kind,
        objective=objective,
        code_mutation=code_mutation,
    )
    rules = FAMILY_RULES[family]
    base_id = f"hyp::{profile_name}::{family}::{_slug(mutation_name)}"
    lineage_id = parent_lineage_id if parent_lineage_id and family != "baseline_capture" else base_id
    if family == "baseline_capture":
        lineage_id = base_id
    hypothesis_id = f"{lineage_id}::{_slug(mutation_name)}" if lineage_id != base_id else base_id
    return {
        "hypothesis_id": hypothesis_id,
        "hypothesis_family": family,
        "parent_hypothesis_id": parent_hypothesis_id,
        "lineage_id": lineage_id,
        "hypothesis_stage": _stage_for_profile(profile_name),
        "funnel_stage": _stage_for_profile(profile_name),
        "promotion_target_stage": next_funnel_stage(_stage_for_profile(profile_name)),
        "expected_upside": rules["expected_upside"],
        "risk_level": rules["risk_level"],
        "kill_criteria": rules["kill_criteria"],
        "promotion_rule": rules["promotion_rule"],
        "thesis": hypothesis,
    }
