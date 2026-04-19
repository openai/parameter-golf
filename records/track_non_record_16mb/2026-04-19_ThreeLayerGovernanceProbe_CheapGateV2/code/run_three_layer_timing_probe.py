from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from experiment_env import guard_experiment_env
from build_three_layer_prototype_data import (
    CaelTrace,
    WorldSample,
    build_records,
    choose_device,
    derive_intervention_effect,
    derive_monday_move,
    derive_seryn_audit,
    load_real_cael_model,
    load_world_samples,
    load_sentencepiece,
    mock_post_monday_continuation,
)

guard_experiment_env(script_name="run_three_layer_timing_probe", require_torch=True)


DEFAULT_INPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_seed_samples.jsonl")
DEFAULT_OUTPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_timing_probe_run1")
DEFAULT_CHECKPOINT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_builder_assets/minimal_real_cael_checkpoint_300steps.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run intervention-timing perturbation v0 on the three-layer builder chain.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cael-source", choices=("mock_v0", "real_cael"), default="real_cael")
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


TRIGGERS = (
    "no_intervention",
    "first_drift",
    "persist_2",
    "high_uncertainty",
    "low_top_gap",
)


def should_trigger(trigger_name: str, trace: CaelTrace) -> bool:
    if trigger_name == "no_intervention":
        return False
    if trigger_name == "first_drift":
        return trace.error_mode != "none"
    if trigger_name == "persist_2":
        return trace.error_mode != "none" and trace.error_persist_steps >= 2
    if trigger_name == "high_uncertainty":
        return trace.uncertainty_band == "high"
    if trigger_name == "low_top_gap":
        return trace.top_gap_band == "narrow"
    raise ValueError(f"Unknown trigger: {trigger_name}")


def apply_triggered_governance(
    sample: WorldSample,
    *,
    base_cael_continuation: str,
    base_trace: CaelTrace,
    trigger_name: str,
) -> dict[str, object]:
    scheduled = should_trigger(trigger_name, base_trace)
    monday_move = derive_monday_move(base_trace) if scheduled else "no"
    fired = scheduled and monday_move != "no"
    if fired:
        post_monday_continuation = mock_post_monday_continuation(sample, monday_move)
    else:
        post_monday_continuation = base_cael_continuation
    filled_trace = CaelTrace(
        error_mode=base_trace.error_mode,
        error_persist_steps=base_trace.error_persist_steps,
        uncertainty_band=base_trace.uncertainty_band,
        top_gap_band=base_trace.top_gap_band,
        intervention_effect=derive_intervention_effect(
            cael_continuation=base_cael_continuation,
            post_monday_continuation=post_monday_continuation,
            gold_continuation=sample.gold_continuation,
        ),
        local_site=base_trace.local_site,
    )
    target_audit = derive_seryn_audit(filled_trace, monday_move)
    return {
        "trigger_name": trigger_name,
        "trigger_scheduled": scheduled,
        "trigger_fired": fired,
        "cael_continuation": base_cael_continuation,
        "cael_trace": asdict(filled_trace),
        "target_move": monday_move,
        "post_monday_continuation": post_monday_continuation,
        "target_audit": target_audit,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_inspection(path: Path, rows_by_sample: dict[str, list[dict[str, object]]]) -> None:
    lines = [
        "# Three-layer timing probe inspection",
        "",
        "- mechanism: single-backbone, intervention-timing perturbation",
        "- comparison_focus: continuation / trace / monday_move / seryn_audit",
        "- trigger_set: no_intervention, first_drift, persist_2, high_uncertainty, low_top_gap",
        "",
    ]
    for sample_id, rows in rows_by_sample.items():
        lines.append(f"## {sample_id}")
        lines.append("")
        for row in rows:
            lines.append(f"### {row['variant']}")
            lines.append(f"- prefix: `{row['prefix']}`")
            lines.append(f"- gold: `{row['gold_continuation']}`")
            lines.append(f"- trigger_fired: `{row['trigger_fired']}`")
            lines.append(f"- cael_continuation: `{row['cael_continuation']}`")
            lines.append(f"- cael_trace: `{json.dumps(row['cael_trace'], ensure_ascii=False)}`")
            lines.append(f"- monday_target_move: `{row['target_move']}`")
            lines.append(f"- post_monday_continuation: `{row['post_monday_continuation']}`")
            lines.append(f"- seryn_target_audit: `{row['target_audit']}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    world_samples = load_world_samples(args.input)
    device = choose_device(args.device)
    sp = None
    real_model = None
    if args.cael_source == "real_cael":
        sp = load_sentencepiece(args.tokenizer_model)
        real_model = load_real_cael_model(args, device)

    rows: list[dict[str, object]] = []
    rows_by_sample: dict[str, list[dict[str, object]]] = {}
    for sample in world_samples:
        cael_record, _monday_record, _seryn_record = build_records(
            sample,
            cael_source=args.cael_source,
            real_model=real_model,
            sp=sp,
            device=device,
            max_new_tokens=args.max_new_tokens,
            binary_forward=args.binary_forward,
        )
        sample_rows: list[dict[str, object]] = []
        for trigger_name in TRIGGERS:
            triggered = apply_triggered_governance(
                sample,
                base_cael_continuation=cael_record.cael_continuation,
                base_trace=cael_record.cael_trace,
                trigger_name=trigger_name,
            )
            row = {
                "variant": trigger_name,
                "sample_id": sample.sample_id,
                "prefix": sample.prefix,
                "gold_continuation": sample.gold_continuation,
                **triggered,
            }
            rows.append(row)
            sample_rows.append(row)
        rows_by_sample[sample.sample_id] = sample_rows

    write_jsonl(output_root / "timing_probe_rows.jsonl", rows)
    write_inspection(output_root / "inspection.md", rows_by_sample)
    manifest = {
        "cael_source": args.cael_source,
        "cael_checkpoint": str(args.cael_checkpoint) if args.cael_checkpoint else None,
        "device": str(device),
        "trigger_set": list(TRIGGERS),
        "goal": "compare policy/audit branching under fixed continuation and timing perturbation",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
