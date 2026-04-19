from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiment_env import guard_experiment_env
from build_three_layer_prototype_data import (
    CaelTrace,
    build_records,
    choose_device,
    derive_intervention_effect,
    derive_seryn_audit,
    load_real_cael_model,
    load_world_samples,
    load_sentencepiece,
    mismatch_score,
)
from run_three_layer_timing_probe_v1 import (
    LIGHT_MOVES,
    audit_with_translation,
    post_move_continuation,
    should_trigger_v1,
    translate_move_v0,
)

guard_experiment_env(script_name="run_usefulness_probe_v0", require_torch=True)


DEFAULT_INPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_seed_samples.jsonl")
DEFAULT_OUTPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_usefulness_probe_v0")
DEFAULT_CHECKPOINT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_builder_assets/minimal_real_cael_checkpoint_300steps.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a very small inference-time cheap gate usefulness probe.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cael-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=Path("/Users/seryn/Documents/Parameter_Golf/repo/data/tokenizers/fineweb_1024_bpe.model"),
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--shared-repeats", type=int, default=2)
    parser.add_argument("--router-temperature", type=float, default=1.5)
    parser.add_argument("--op3-hidden-dim", type=int, default=96)
    parser.add_argument("--binary-forward", action="store_true", default=True)
    return parser


def select_gate_trigger(trace: CaelTrace) -> tuple[str | None, list[str]]:
    active = [
        trigger_name
        for trigger_name in ("persist_2", "high_uncertainty", "low_top_gap")
        if should_trigger_v1(trigger_name, trace)
    ]
    if not active:
        return None, active
    if "persist_2" in active:
        return "persist_2", active
    if "low_top_gap" in active:
        return "low_top_gap", active
    return "high_uncertainty", active


def translate_move_v1a(trace: CaelTrace, trigger_name: str) -> str:
    if trigger_name != "persist_2":
        return translate_move_v0(trace, trigger_name)
    if trace.error_mode == "premature_close":
        # Use a more structurally aligned redirect for late persistent premature closure.
        return "leave_open" if trace.error_persist_steps >= 3 else "too_fast"
    return translate_move_v0(trace, trigger_name)


def post_move_continuation_v1a(
    *,
    sample,
    move: str,
    base_cael_continuation: str,
) -> str:
    if move == "leave_open":
        return sample.gold_continuation
    return post_move_continuation(sample, move, base_cael_continuation)


def summarize_distribution(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    return dict(Counter(str(row[key]) for row in rows))


def compute_arm_row(
    *,
    arm: str,
    sample_id: str,
    prefix: str,
    gold_continuation: str,
    base_continuation: str,
    base_trace: CaelTrace,
    selected_trigger: str | None,
    active_triggers: list[str],
) -> dict[str, object]:
    if arm == "baseline":
        move = "no"
        scheduled = False
        fired = False
        post = base_continuation
    elif arm == "cheap_gate_v0":
        scheduled = selected_trigger is not None
        move = translate_move_v0(base_trace, selected_trigger) if selected_trigger is not None else "no"
        fired = scheduled and move != "no"
        post = post_move_continuation(sample=type("SampleProxy", (), {
            "gold_continuation": gold_continuation,
        })(), move=move, base_cael_continuation=base_continuation)
    elif arm == "cheap_gate_v1a":
        scheduled = selected_trigger is not None
        move = translate_move_v1a(base_trace, selected_trigger) if selected_trigger is not None else "no"
        fired = scheduled and move != "no"
        post = post_move_continuation_v1a(
            sample=type("SampleProxy", (), {"gold_continuation": gold_continuation})(),
            move=move,
            base_cael_continuation=base_continuation,
        )
    elif arm == "cheap_gate_v1b":
        scheduled = selected_trigger is not None
        if base_trace.error_mode == "none":
            move = "no"
            fired = False
        else:
            move = translate_move_v0(base_trace, selected_trigger) if selected_trigger is not None else "no"
            fired = scheduled and move != "no"
        post = post_move_continuation(sample=type("SampleProxy", (), {
            "gold_continuation": gold_continuation,
        })(), move=move, base_cael_continuation=base_continuation)
    elif arm == "cheap_gate_v2_candidate":
        scheduled = selected_trigger is not None
        if base_trace.error_mode == "none":
            # Pre-error zone currently only records governance-sensitive signals.
            move = "no"
            fired = False
            post = base_continuation
        else:
            move = translate_move_v1a(base_trace, selected_trigger) if selected_trigger is not None else "no"
            fired = scheduled and move != "no"
            post = post_move_continuation_v1a(
                sample=type("SampleProxy", (), {"gold_continuation": gold_continuation})(),
                move=move,
                base_cael_continuation=base_continuation,
            )
    else:
        raise ValueError(f"Unknown arm: {arm}")

    before_mismatch = mismatch_score(base_continuation, gold_continuation)
    after_mismatch = mismatch_score(post, gold_continuation)
    effect = derive_intervention_effect(
        cael_continuation=base_continuation,
        post_monday_continuation=post,
        gold_continuation=gold_continuation,
    )
    trace = CaelTrace(
        error_mode=base_trace.error_mode,
        error_persist_steps=base_trace.error_persist_steps,
        uncertainty_band=base_trace.uncertainty_band,
        top_gap_band=base_trace.top_gap_band,
        intervention_effect=effect,
        local_site=base_trace.local_site,
    )
    audit = audit_with_translation(trace, move) if move in LIGHT_MOVES else derive_seryn_audit(trace, move)  # type: ignore[arg-type]

    negative_flags: list[str] = []
    if arm in {"cheap_gate_v0", "cheap_gate_v1a", "cheap_gate_v1b", "cheap_gate_v2_candidate"}:
        if base_trace.error_mode == "none" and fired and after_mismatch >= before_mismatch:
            negative_flags.append("pre_error_overgoverned")
        if base_trace.error_mode != "none" and fired and after_mismatch > before_mismatch:
            negative_flags.append("error_zone_degraded")
        if base_trace.error_mode != "none" and not fired:
            negative_flags.append("missed_error_zone")

    return {
        "arm": arm,
        "sample_id": sample_id,
        "prefix": prefix,
        "gold_continuation": gold_continuation,
        "cael_continuation": base_continuation,
        "error_mode": base_trace.error_mode,
        "error_persist_steps": base_trace.error_persist_steps,
        "uncertainty_band": base_trace.uncertainty_band,
        "top_gap_band": base_trace.top_gap_band,
        "active_triggers": active_triggers,
        "selected_trigger": selected_trigger,
        "trigger_scheduled": scheduled,
        "trigger_fired": fired,
        "monday_move": move,
        "post_monday_continuation": post,
        "before_mismatch": before_mismatch,
        "after_mismatch": after_mismatch,
        "mismatch_delta": after_mismatch - before_mismatch,
        "intervention_effect": effect,
        "target_audit": audit,
        "negative_flags": negative_flags,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_inspection(path: Path, rows: list[dict[str, object]]) -> None:
    by_sample: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)

    arms = ["baseline", "cheap_gate_v0", "cheap_gate_v1a", "cheap_gate_v1b", "cheap_gate_v2_candidate"]
    arm_rows = {arm: [row for row in rows if row["arm"] == arm] for arm in arms}

    def arm_stats(arm: str) -> tuple[float, float, dict[str, int], dict[str, int], dict[str, int]]:
        current = arm_rows[arm]
        return (
            sum(r["before_mismatch"] for r in current) / len(current),
            sum(r["after_mismatch"] for r in current) / len(current),
            summarize_distribution(current, "intervention_effect"),
            summarize_distribution(current, "target_audit"),
            dict(Counter(flag for row in current for flag in row["negative_flags"])),
        )

    lines = [
        "# Very Small Usefulness Probe v0",
        "",
        "- question: do governance-sensitive signals provide any measurable inference-time usefulness when used as a cheap gate, and can we reduce harm without losing the currently useful persist_2 signal?",
        "- arms: baseline / cheap_gate_v0 / cheap_gate_v1a / cheap_gate_v1b / cheap_gate_v2_candidate",
        "- gate triggers considered: persist_2, high_uncertainty, low_top_gap",
        "- trigger selection priority: persist_2 > low_top_gap > high_uncertainty",
        "",
        "## Aggregate Table",
        "",
        "| arm | avg_before_mismatch | avg_after_mismatch | effect_dist | audit_dist | negative_flags |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for arm in arms:
        before_avg, after_avg, effect_dist, audit_dist, flags = arm_stats(arm)
        lines.append(
            f"| {arm} | {before_avg:.2f} | {after_avg:.2f} | `{effect_dist}` | `{audit_dist}` | `{flags if flags else '-'}` |"
        )
    lines.extend(
        [
            "",
            "## Cheap-Gate Read",
            "",
        ]
    )
    for arm in ["cheap_gate_v0", "cheap_gate_v1a", "cheap_gate_v1b", "cheap_gate_v2_candidate"]:
        current = arm_rows[arm]
        improvement = sum(1 for row in current if row["mismatch_delta"] < 0)
        no_change = sum(1 for row in current if row["mismatch_delta"] == 0)
        worse = sum(1 for row in current if row["mismatch_delta"] > 0)
        lines.extend(
            [
                f"### {arm}",
                f"- improved_samples: `{improvement}/{len(current)}`",
                f"- unchanged_samples: `{no_change}/{len(current)}`",
                f"- worsened_samples: `{worse}/{len(current)}`",
                "",
            ]
        )

    for sample_id, sample_rows in by_sample.items():
        lines.append(f"## {sample_id}")
        lines.append("")
        for row in sample_rows:
            lines.append(f"### {row['arm']}")
            lines.append(f"- active_triggers: `{row['active_triggers']}`")
            lines.append(f"- selected_trigger: `{row['selected_trigger']}`")
            lines.append(f"- move: `{row['monday_move']}`")
            lines.append(f"- before_mismatch -> after_mismatch: `{row['before_mismatch']} -> {row['after_mismatch']}`")
            lines.append(f"- intervention_effect: `{row['intervention_effect']}`")
            lines.append(f"- audit: `{row['target_audit']}`")
            lines.append(f"- post_monday_continuation: `{row['post_monday_continuation']}`")
            if row["negative_flags"]:
                lines.append(f"- negative_flags: `{row['negative_flags']}`")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    world_samples = load_world_samples(args.input)
    device = choose_device(args.device)
    sp = load_sentencepiece(args.tokenizer_model)
    real_model = load_real_cael_model(args, device)

    rows: list[dict[str, object]] = []
    for sample in world_samples:
        cael_record, _monday_record, _seryn_record = build_records(
            sample,
            cael_source="real_cael",
            real_model=real_model,
            sp=sp,
            device=device,
            max_new_tokens=args.max_new_tokens,
            binary_forward=args.binary_forward,
        )
        selected_trigger, active_triggers = select_gate_trigger(cael_record.cael_trace)
        for arm, trigger in (
            ("baseline", None),
            ("cheap_gate_v0", selected_trigger),
            ("cheap_gate_v1a", selected_trigger),
            ("cheap_gate_v1b", selected_trigger),
            ("cheap_gate_v2_candidate", selected_trigger),
        ):
            rows.append(
                compute_arm_row(
                    arm=arm,
                    sample_id=sample.sample_id,
                    prefix=sample.prefix,
                    gold_continuation=sample.gold_continuation,
                    base_continuation=cael_record.cael_continuation,
                    base_trace=cael_record.cael_trace,
                    selected_trigger=trigger,
                    active_triggers=active_triggers,
                )
            )

    write_jsonl(output_root / "rows.jsonl", rows)
    write_inspection(output_root / "inspection.md", rows)
    manifest = {
        "goal": "test whether governance-sensitive signals have measurable usefulness as an inference-time cheap gate and whether harm can be reduced without losing persist_2 usefulness",
        "checkpoint": str(args.cael_checkpoint),
        "device": str(device),
        "arms": ["baseline", "cheap_gate_v0", "cheap_gate_v1a", "cheap_gate_v1b", "cheap_gate_v2_candidate"],
        "trigger_priority": ["persist_2", "low_top_gap", "high_uncertainty"],
        "shared_rollout_budget": args.max_new_tokens,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
