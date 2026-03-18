from __future__ import annotations

from copy import deepcopy
from typing import Any

from .code_mutations import planned_code_mutation
from .common import canonical_code_mutation, canonical_env, timestamp_slug, utc_now_iso
from .history import planner_eligible_rows
from .hypotheses import build_hypothesis_metadata, funnel_stage_for_profile, next_funnel_stage
from .profiles import resolve_profile


FLOAT_KEYS = {"TIED_EMBED_LR", "MATRIX_LR", "SCALAR_LR", "QK_GAIN_INIT", "GRAD_CLIP_NORM"}
INT_KEYS = {"ITERATIONS", "WARMDOWN_ITERS", "WARMUP_STEPS", "TRAIN_BATCH_TOKENS"}
QUALITY_WIN_THRESHOLD = 0.002
GAP_WIN_THRESHOLD = 0.003
BYTE_WIN_THRESHOLD = 32_768
QUALITY_REGRESSION_TOLERANCE = 0.001


def _format_value(key: str, value: float | int) -> str:
    if key in INT_KEYS:
        return str(int(round(float(value))))
    if key in FLOAT_KEYS:
        return f"{float(value):.6g}"
    return str(value)


def _scale(env: dict[str, str], key: str, factor: float, *, floor: float | None = None, ceil: float | None = None) -> dict[str, str]:
    new_env = deepcopy(env)
    current = float(new_env[key])
    value = current * factor
    if floor is not None:
        value = max(value, floor)
    if ceil is not None:
        value = min(value, ceil)
    new_env[key] = _format_value(key, value)
    return new_env


def _set(env: dict[str, str], key: str, value: float | int | str) -> dict[str, str]:
    new_env = deepcopy(env)
    new_env[key] = _format_value(key, value) if not isinstance(value, str) else value
    return new_env


def _candidate_signature(env: dict[str, str], code_mutation: dict[str, Any] | None) -> str:
    return f"{canonical_env(env)}::{canonical_code_mutation(code_mutation)}"


def _comparable_history(history: list[dict[str, Any]], profile_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in planner_eligible_rows(history, profile_name):
        spec = row.get("spec", {})
        metrics = row.get("metrics", {})
        if metrics.get("final_roundtrip_val_bpb") is None:
            continue
        rows.append(row)
    return rows


def _profile_history(history: list[dict[str, Any]], profile_name: str) -> list[dict[str, Any]]:
    return [row for row in history if row.get("spec", {}).get("profile") == profile_name]


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: row.get("metrics", {}).get("final_roundtrip_val_bpb", float("inf")))


def _find_row_by_experiment_id(rows: list[dict[str, Any]], experiment_id: str | None) -> dict[str, Any] | None:
    if not experiment_id:
        return None
    for row in rows:
        if row.get("spec", {}).get("experiment_id") == experiment_id:
            return row
    return None


def _lineage_rows(rows: list[dict[str, Any]], lineage_id: str | None) -> list[dict[str, Any]]:
    if not lineage_id:
        return []
    return [row for row in rows if row.get("spec", {}).get("lineage_id") == lineage_id]


def _winner_reference(row: dict[str, Any], eligible_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    spec = row.get("spec", {})
    parent = _find_row_by_experiment_id(eligible_rows, spec.get("parent_experiment_id"))
    if parent is not None:
        return parent
    lineage_id = spec.get("lineage_id")
    lineage = [candidate for candidate in _lineage_rows(eligible_rows, lineage_id) if candidate is not row]
    if lineage:
        return _best_row(lineage)
    earlier = [candidate for candidate in eligible_rows if candidate is not row]
    return _best_row(earlier)


def _is_winner(row: dict[str, Any], eligible_rows: list[dict[str, Any]]) -> bool:
    reference = _winner_reference(row, eligible_rows)
    if reference is None:
        return False
    current_metrics = row.get("metrics", {})
    reference_metrics = reference.get("metrics", {})
    current_final = current_metrics.get("final_roundtrip_val_bpb")
    reference_final = reference_metrics.get("final_roundtrip_val_bpb")
    current_gap = current_metrics.get("quant_gap_bpb")
    reference_gap = reference_metrics.get("quant_gap_bpb")
    current_bytes = current_metrics.get("total_submission_size_int8_zlib_bytes")
    reference_bytes = reference_metrics.get("total_submission_size_int8_zlib_bytes")
    quality_improvement = (
        reference_final is not None
        and current_final is not None
        and (reference_final - current_final) >= QUALITY_WIN_THRESHOLD
    )
    gap_improvement = (
        reference_gap is not None
        and current_gap is not None
        and (reference_gap - current_gap) >= GAP_WIN_THRESHOLD
        and (
            reference_final is None
            or current_final is None
            or current_final <= (reference_final + QUALITY_REGRESSION_TOLERANCE)
        )
    )
    bytes_improvement = (
        reference_bytes is not None
        and current_bytes is not None
        and (reference_bytes - current_bytes) >= BYTE_WIN_THRESHOLD
        and (
            reference_final is None
            or current_final is None
            or current_final <= (reference_final + QUALITY_REGRESSION_TOLERANCE)
        )
    )
    return bool(quality_improvement or gap_improvement or bytes_improvement)


def _neighbor_probe_env(env: dict[str, str], mutation_name: str) -> dict[str, str] | None:
    if mutation_name == "tied_embed_lr_down":
        return _scale(env, "TIED_EMBED_LR", 0.9, floor=0.005)
    if mutation_name == "tied_embed_lr_up":
        return _scale(env, "TIED_EMBED_LR", 1.1, ceil=0.5)
    if mutation_name == "matrix_lr_down":
        return _scale(env, "MATRIX_LR", 0.85, floor=0.005)
    if mutation_name == "matrix_lr_up":
        return _scale(env, "MATRIX_LR", 1.1, ceil=0.2)
    if mutation_name == "scalar_lr_down":
        return _scale(env, "SCALAR_LR", 0.85, floor=0.005)
    if mutation_name == "qk_gain_down":
        return _scale(env, "QK_GAIN_INIT", 0.85, floor=0.75)
    if mutation_name == "warmdown_shorter":
        return _set(env, "WARMDOWN_ITERS", max(int(float(env.get("WARMDOWN_ITERS", "1200"))) // 2, 200))
    if mutation_name == "iterations_up":
        return _set(env, "ITERATIONS", min(int(float(env.get("ITERATIONS", "200"))) + 100, 1200))
    if mutation_name == "grad_clip_on":
        return _set(env, "GRAD_CLIP_NORM", 0.75)
    return None


def _confirm_env(env: dict[str, str], profile_name: str) -> dict[str, str]:
    if funnel_stage_for_profile(profile_name) == "scout":
        return _set(env, "ITERATIONS", min(int(float(env.get("ITERATIONS", "200"))) + 50, 1200))
    return deepcopy(env)


def _choose_neighbor_probe(
    *,
    latest_row: dict[str, Any],
    profile: dict[str, Any],
    tried_signatures: set[str],
) -> tuple[dict[str, str], dict[str, Any] | None, str, str, str] | None:
    latest_spec = latest_row.get("spec", {})
    latest_env = deepcopy(latest_spec.get("env", {}))
    latest_code_mutation = deepcopy(latest_spec.get("code_mutation"))
    explicit_neighbor = _neighbor_probe_env(latest_env, latest_spec.get("mutation_name", ""))
    if explicit_neighbor is not None:
        signature = _candidate_signature(explicit_neighbor, latest_code_mutation)
        if signature not in tried_signatures:
            return (
                explicit_neighbor,
                latest_code_mutation,
                latest_spec.get("mutation_name", "neighbor_probe"),
                latest_spec.get("mutation_kind", "env"),
                "Probe one nearby variant of the confirmed winning line before branching elsewhere.",
            )
    for candidate in _candidate_pool(latest_env, latest_row, profile["script"]):
        if candidate.get("code_mutation") is not None:
            continue
        signature = _candidate_signature(candidate["env"], latest_code_mutation)
        if signature in tried_signatures:
            continue
        return (
            candidate["env"],
            latest_code_mutation,
            candidate["name"],
            "env",
            candidate["hypothesis"],
        )
    return None


def _candidate_pool(base_env: dict[str, str], latest_row: dict[str, Any] | None, script: str) -> list[dict[str, Any]]:
    quant_gap = None
    total_bytes = None
    if latest_row is not None:
        quant_gap = latest_row.get("metrics", {}).get("quant_gap_bpb")
        total_bytes = latest_row.get("metrics", {}).get("total_submission_size_int8_zlib_bytes")

    candidates: list[dict[str, Any]] = []
    if total_bytes is not None and total_bytes >= 15_900_000:
        code = planned_code_mutation(script, "quant_keep_float_smaller")
        candidates.append(
            {
                "name": code["name"],
                "env": deepcopy(base_env),
                "code_mutation": code,
                "objective": "reduce submission bytes",
                "hypothesis": "Lowering the float-passthrough threshold may buy back artifact bytes when the code+model package is too close to the limit.",
                "rationale": f"Latest submission size was {int(total_bytes):,} bytes, so the planner tries a byte-saving quantization policy first.",
            }
        )
    if quant_gap is not None and quant_gap > 0.02:
        candidates.extend(
            [
                {
                    "name": "quant_clip_tighter",
                    "env": deepcopy(base_env),
                    "code_mutation": planned_code_mutation(script, "quant_clip_tighter"),
                    "objective": "reduce quantization gap",
                    "hypothesis": "A slightly tighter int8 clipping percentile may improve the roundtrip score without touching the model topology.",
                    "rationale": f"Latest quantization gap was {quant_gap:.4f} BPB, so the planner tries a direct quantization-policy mutation first.",
                },
                {
                    "name": "quant_scale_fp32",
                    "env": deepcopy(base_env),
                    "code_mutation": planned_code_mutation(script, "quant_scale_fp32"),
                    "objective": "reduce quantization gap",
                    "hypothesis": "Keeping per-row scales in fp32 may preserve more fidelity through the int8 roundtrip at some byte cost.",
                    "rationale": f"Latest quantization gap was {quant_gap:.4f} BPB, so the planner allocates some bytes to cleaner dequantization metadata.",
                },
                {
                    "name": "quant_keep_float_bigger",
                    "env": deepcopy(base_env),
                    "code_mutation": planned_code_mutation(script, "quant_keep_float_bigger"),
                    "objective": "reduce quantization gap",
                    "hypothesis": "Protecting a larger class of small tensors from int8 export may help the final roundtrip score.",
                    "rationale": f"Latest quantization gap was {quant_gap:.4f} BPB, so the planner raises the keep-float threshold before more aggressive training changes.",
                },
                {
                    "name": "qk_gain_down",
                    "env": _scale(base_env, "QK_GAIN_INIT", 0.85, floor=0.75),
                    "code_mutation": None,
                    "objective": "reduce quantization gap",
                    "hypothesis": "A slightly lower QK gain should calm late activations and preserve more quality after int8 roundtrip.",
                    "rationale": f"Latest quantization gap was {quant_gap:.4f} BPB, so the planner prioritizes stability before raw-capacity tweaks.",
                },
                {
                    "name": "matrix_lr_down",
                    "env": _scale(base_env, "MATRIX_LR", 0.85, floor=0.005),
                    "code_mutation": None,
                    "objective": "reduce quantization gap",
                    "hypothesis": "A lower matrix LR should reduce weight volatility and help the quantized artifact track the raw model more closely.",
                    "rationale": f"Latest quantization gap was {quant_gap:.4f} BPB, so the planner tries a softer Muon matrix update.",
                },
                {
                    "name": "grad_clip_on",
                    "env": _set(base_env, "GRAD_CLIP_NORM", 0.5),
                    "code_mutation": None,
                    "objective": "stability probe",
                    "hypothesis": "Mild gradient clipping may cut bad spikes without materially slowing learning on the smoke profile.",
                    "rationale": f"Latest quantization gap was {quant_gap:.4f} BPB, so the planner adds a conservative stability guardrail.",
                },
            ]
        )

    candidates.extend(
        [
            {
                "name": "tied_embed_lr_down",
                "env": _scale(base_env, "TIED_EMBED_LR", 0.9, floor=0.005),
                "code_mutation": None,
                "objective": "optimizer tuning",
                "hypothesis": "Slightly reducing the tied embedding LR may improve early stability on the compact vocabulary model.",
                "rationale": "Embedding weights are a large fraction of the tiny model budget, so their LR is a high-leverage first knob.",
            },
            {
                "name": "tied_embed_lr_up",
                "env": _scale(base_env, "TIED_EMBED_LR", 1.1, ceil=0.5),
                "code_mutation": None,
                "objective": "optimizer tuning",
                "hypothesis": "A slightly higher tied embedding LR may improve token compression faster in the short smoke regime.",
                "rationale": "The smoke profile is short, so the planner also checks whether the embedding path wants to move faster, not only slower.",
            },
            {
                "name": "scalar_lr_down",
                "env": _scale(base_env, "SCALAR_LR", 0.85, floor=0.005),
                "code_mutation": None,
                "objective": "optimizer tuning",
                "hypothesis": "Lower scalar LR may stabilize control tensors and reduce divergence between raw and quantized performance.",
                "rationale": "Control scalars influence attention/MLP mixing directly, so a calmer update can be disproportionately valuable.",
            },
            {
                "name": "matrix_lr_up",
                "env": _scale(base_env, "MATRIX_LR", 1.1, ceil=0.2),
                "code_mutation": None,
                "objective": "optimizer tuning",
                "hypothesis": "If the current run underfits the short smoke horizon, a slightly higher matrix LR could buy faster progress.",
                "rationale": "The planner balances stability probes with one explicit faster-learning probe.",
            },
            {
                "name": "warmdown_shorter",
                "env": _set(base_env, "WARMDOWN_ITERS", max(int(float(base_env.get("WARMDOWN_ITERS", "1200"))) // 2, 200)),
                "code_mutation": None,
                "objective": "schedule tuning",
                "hypothesis": "A shorter warmdown may keep learning rates higher for more of the short smoke run.",
                "rationale": "The current smoke profile is brief enough that the default warmdown may become active earlier than ideal.",
            },
            {
                "name": "iterations_up",
                "env": _set(base_env, "ITERATIONS", min(int(float(base_env.get("ITERATIONS", "200"))) + 100, 1000)),
                "code_mutation": None,
                "objective": "signal probe",
                "hypothesis": "A somewhat longer smoke run can reveal whether current changes are still improving at the endpoint.",
                "rationale": "The planner wants occasional longer probes so it can distinguish undertraining from poor settings.",
            },
        ]
    )
    return candidates


def plan_next_experiment(
    history: list[dict[str, Any]],
    profile_name: str | None,
    manual_overrides: dict[str, str] | None = None,
    manual_code_mutation: str | None = None,
) -> dict[str, Any]:
    resolved_name, profile = resolve_profile(profile_name)
    profile_stage = funnel_stage_for_profile(resolved_name)
    eligible_profile_rows = planner_eligible_rows(history, resolved_name)
    comparable = _comparable_history(history, resolved_name)
    best_row = _best_row(comparable)
    latest_row = comparable[-1] if comparable else None
    base_env = deepcopy(profile["base_env"])
    parent_id = None
    inherited_code_mutation = None
    if best_row is not None:
        base_env.update(best_row.get("spec", {}).get("env", {}))
        parent_id = best_row.get("spec", {}).get("experiment_id")
        inherited_code_mutation = deepcopy(best_row.get("spec", {}).get("code_mutation"))
    parent_hypothesis_id = best_row.get("spec", {}).get("hypothesis_id") if best_row is not None else None
    parent_lineage_id = best_row.get("spec", {}).get("lineage_id") if best_row is not None else None

    tried_signatures = {
        _candidate_signature(row.get("spec", {}).get("env", {}), row.get("spec", {}).get("code_mutation"))
        for row in eligible_profile_rows
        if row.get("spec", {}).get("env") is not None
    }
    followup_of_experiment_id = None
    followup_reason = None
    funnel_stage = profile_stage
    promotion_target_stage = next_funnel_stage(profile_stage)

    if latest_row is not None and _is_winner(latest_row, eligible_profile_rows):
        latest_spec = latest_row.get("spec", {})
        lineage = _lineage_rows(eligible_profile_rows, latest_spec.get("lineage_id"))
        has_confirm = any(row.get("spec", {}).get("followup_reason") == "confirm_winner" for row in lineage)
        has_neighbor = any(row.get("spec", {}).get("followup_reason") == "neighbor_probe" for row in lineage)
        if not has_confirm:
            chosen_env = _confirm_env(deepcopy(latest_spec.get("env", {})), resolved_name)
            chosen_code_mutation = deepcopy(latest_spec.get("code_mutation"))
            inherits_parent_code_mutation = chosen_code_mutation is not None
            mutation_name = latest_spec.get("mutation_name", "confirm_winner")
            mutation_kind = latest_spec.get("mutation_kind", "env")
            objective = latest_spec.get("objective", "confirm promising line")
            hypothesis = (
                f"Confirm promising line `{latest_spec.get('lineage_id')}` with one repeat on the same profile before exploring further."
            )
            rationale = "The latest planner-eligible run looks like a winner, so the harness confirms it once before branching into new ideas."
            parent_id = latest_spec.get("experiment_id")
            parent_hypothesis_id = latest_spec.get("hypothesis_id")
            parent_lineage_id = latest_spec.get("lineage_id")
            followup_of_experiment_id = latest_spec.get("experiment_id")
            followup_reason = "confirm_winner"
            funnel_stage = "confirm"
            promotion_target_stage = next_funnel_stage(funnel_stage)
        elif not has_neighbor and latest_spec.get("followup_reason") == "confirm_winner":
            neighbor = _choose_neighbor_probe(
                latest_row=latest_row,
                profile=profile,
                tried_signatures=tried_signatures,
            )
            if neighbor is not None:
                chosen_env, chosen_code_mutation, mutation_name, mutation_kind, hypothesis = neighbor
                inherits_parent_code_mutation = (
                    chosen_code_mutation is not None and latest_spec.get("code_mutation") is not None
                )
                objective = latest_spec.get("objective", "probe winning neighborhood")
                rationale = "The winning line has one confirmation, so the harness now probes one nearby variant before moving on."
                parent_id = latest_spec.get("experiment_id")
                parent_hypothesis_id = latest_spec.get("hypothesis_id")
                parent_lineage_id = latest_spec.get("lineage_id")
                followup_of_experiment_id = latest_spec.get("experiment_id")
                followup_reason = "neighbor_probe"
                funnel_stage = "confirm"
                promotion_target_stage = next_funnel_stage(funnel_stage)
            else:
                latest_row = None

    if not comparable:
        chosen_env = deepcopy(base_env)
        chosen_code_mutation = None
        inherits_parent_code_mutation = False
        mutation_name = "baseline"
        mutation_kind = "baseline"
        objective = "establish comparable baseline"
        hypothesis = "Run the profile baseline end-to-end so the harness has a clean local anchor before it starts mutating anything."
        rationale = "There is no comparable completed history for this profile yet, so the planner starts with a baseline capture."
    elif followup_reason is None:
        chosen_env = None
        chosen_code_mutation = deepcopy(inherited_code_mutation)
        inherits_parent_code_mutation = chosen_code_mutation is not None
        mutation_name = "fallback"
        mutation_kind = "env"
        objective = "fallback probe"
        hypothesis = "Probe a nearby variant of the current best run."
        rationale = "All primary mutations were already tried, so the planner falls back to a nearby schedule change."
        for candidate in _candidate_pool(base_env, latest_row, profile["script"]):
            explicit_code_mutation = deepcopy(candidate.get("code_mutation")) if candidate.get("code_mutation") else None
            candidate_code_mutation = explicit_code_mutation or deepcopy(inherited_code_mutation)
            signature = _candidate_signature(candidate["env"], candidate_code_mutation)
            if signature in tried_signatures:
                continue
            chosen_env = candidate["env"]
            if explicit_code_mutation is not None:
                chosen_code_mutation = explicit_code_mutation
                inherits_parent_code_mutation = False
            mutation_name = candidate["name"]
            mutation_kind = "code" if explicit_code_mutation else "env"
            objective = candidate["objective"]
            hypothesis = candidate["hypothesis"]
            rationale = candidate["rationale"]
            break
        if chosen_env is None:
            chosen_env = _set(base_env, "ITERATIONS", min(int(float(base_env.get("ITERATIONS", "200"))) + 50, 1200))

    if manual_overrides:
        chosen_env.update({k: str(v) for k, v in manual_overrides.items()})
    if manual_code_mutation:
        chosen_code_mutation = planned_code_mutation(profile["script"], manual_code_mutation)
        inherits_parent_code_mutation = False
        mutation_name = manual_code_mutation
        mutation_kind = "code"
        objective = f"manual code mutation: {manual_code_mutation}"
        hypothesis = f"Materialize the requested code mutation `{manual_code_mutation}` on top of the current profile baseline."
        rationale = "Manual code-mutation override requested by the operator."
        followup_of_experiment_id = None
        followup_reason = None
        funnel_stage = profile_stage
        promotion_target_stage = next_funnel_stage(profile_stage)

    run_index = len(history) + 1
    experiment_id = f"{timestamp_slug()}_{resolved_name}_{run_index:04d}_{mutation_name}"
    tags = list(dict.fromkeys([resolved_name, mutation_name] + ([chosen_code_mutation["name"]] if chosen_code_mutation else [])))
    hypothesis_meta = build_hypothesis_metadata(
        profile_name=resolved_name,
        mutation_name=mutation_name,
        mutation_kind=mutation_kind,
        objective=objective,
        hypothesis=hypothesis,
        code_mutation=chosen_code_mutation,
        parent_hypothesis_id=parent_hypothesis_id,
        parent_lineage_id=parent_lineage_id,
    )
    return {
        "experiment_id": experiment_id,
        "created_at": utc_now_iso(),
        "source": "planner",
        "profile": resolved_name,
        "track": profile["track"],
        "launcher": profile["launcher"],
        "script": profile["script"],
        "nproc_per_node": profile.get("nproc_per_node"),
        "run_timeout_seconds": profile.get("run_timeout_seconds"),
        "idle_timeout_seconds": profile.get("idle_timeout_seconds"),
        "required_modules": list(profile.get("required_modules", [])),
        "required_gpus": profile.get("required_gpus"),
        "supports_autoloop": bool(profile.get("supports_autoloop", True)),
        "require_challenge_ready": bool(profile.get("require_challenge_ready", False)),
        "objective": objective,
        "hypothesis": hypothesis,
        "rationale": rationale,
        "mutation_name": mutation_name,
        "mutation_kind": mutation_kind,
        "parent_experiment_id": parent_id,
        "followup_of_experiment_id": followup_of_experiment_id,
        "followup_reason": followup_reason,
        "env": chosen_env,
        "code_mutation": chosen_code_mutation,
        "inherits_parent_code_mutation": inherits_parent_code_mutation,
        "tags": tags,
        **hypothesis_meta,
        "funnel_stage": funnel_stage,
        "promotion_target_stage": promotion_target_stage,
    }
