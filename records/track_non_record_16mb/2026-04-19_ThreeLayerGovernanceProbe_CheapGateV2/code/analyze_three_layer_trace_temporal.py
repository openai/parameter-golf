from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_env import guard_experiment_env
from build_three_layer_prototype_data import (
    WorldSample,
    choose_device,
    entropy_band,
    gap_band,
    infer_error_mode,
    infer_local_site,
    load_real_cael_model,
    load_sentencepiece,
    load_world_samples,
)
from run_three_layer_timing_probe_v1 import should_trigger_v1, translate_move_v0

guard_experiment_env(script_name="analyze_three_layer_trace_temporal", require_torch=True)

import torch


DEFAULT_INPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_seed_samples.jsonl")
DEFAULT_CHECKPOINT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_builder_assets/minimal_real_cael_checkpoint_300steps.pt")
DEFAULT_OUTPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_timing_probe_run2/trace_temporal_audit.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit earliest visible governance phase in real Cael rollouts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cael-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=Path("/Users/seryn/Documents/Parameter_Golf/repo/data/tokenizers/fineweb_1024_bpe.model"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def phase_from_persist(error_mode: str, persist_steps: int) -> str:
    if error_mode == "none":
        return "none"
    if persist_steps == 1:
        return "first_drift"
    return "persist_2"


def decode_prefix_timeline(
    sample: WorldSample,
    *,
    model,
    sp,
    device: torch.device,
    max_new_tokens: int,
    binary_forward: bool,
) -> list[dict[str, object]]:
    token_ids = [int(sp.bos_id()), *sp.encode(sample.prefix, out_type=int)]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    timeline: list[dict[str, object]] = []
    persist_steps = 0
    with torch.inference_mode():
        for step_idx in range(1, max_new_tokens + 1):
            context = input_ids[:, -256:]
            logits, _stats = model(context, target_ids=None, binary_forward=binary_forward)
            next_logits = logits[:, -1, :].float()
            probs = torch.softmax(next_logits, dim=-1)
            entropy = float((-(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)).item())
            top2 = torch.topk(next_logits, k=2, dim=-1).values
            gap = float((top2[:, 0] - top2[:, 1]).item())
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            full_text = sp.decode(input_ids[0].tolist())
            continuation = full_text[len(sample.prefix) :].strip() if full_text.startswith(sample.prefix) else full_text.strip()
            continuation = continuation or sample.gold_continuation.strip()
            error_mode = infer_error_mode(sample, continuation)
            if error_mode == "none":
                persist_steps = 0
            else:
                persist_steps += 1

            timeline.append(
                {
                    "step": step_idx,
                    "continuation": continuation,
                    "error_mode": error_mode,
                    "error_persist_steps": persist_steps,
                    "phase": phase_from_persist(error_mode, persist_steps),
                    "uncertainty_value": round(entropy, 4),
                    "uncertainty_band": entropy_band(entropy),
                    "top_gap_value": round(gap, 4),
                    "top_gap_band": gap_band(gap),
                    "local_site": infer_local_site(sample, error_mode),
                    "trigger_hits_v1": [],
                    "candidate_moves_v1_translation": [],
                }
            )
            current = timeline[-1]
            trigger_hits = [
                trigger_name
                for trigger_name in ("first_drift", "persist_2", "high_uncertainty", "low_top_gap")
                if should_trigger_v1(trigger_name, current_trace := type("TraceProxy", (), {
                    "error_mode": current["error_mode"],
                    "error_persist_steps": current["error_persist_steps"],
                    "uncertainty_band": current["uncertainty_band"],
                    "top_gap_band": current["top_gap_band"],
                    "local_site": current["local_site"],
                })())
            ]
            current["trigger_hits_v1"] = trigger_hits
            current["candidate_moves_v1_translation"] = [
                f"{trigger_name}:{translate_move_v0(current_trace, trigger_name)}"
                for trigger_name in trigger_hits
            ]
    return timeline


def summarize_timeline(timeline: list[dict[str, object]]) -> dict[str, object]:
    earliest_non_none = next((row for row in timeline if row["error_mode"] != "none"), None)
    earliest_first_drift = next((row for row in timeline if row["phase"] == "first_drift"), None)
    earliest_persist = next((row for row in timeline if row["phase"] == "persist_2"), None)
    has_transition = any(
        timeline[idx - 1]["phase"] == "first_drift" and timeline[idx]["phase"] == "persist_2"
        for idx in range(1, len(timeline))
    )
    return {
        "earliest_non_none_step": None if earliest_non_none is None else earliest_non_none["step"],
        "earliest_first_drift_step": None if earliest_first_drift is None else earliest_first_drift["step"],
        "earliest_persist_2_step": None if earliest_persist is None else earliest_persist["step"],
        "has_none_to_first_to_persist_transition": has_transition,
    }


def write_report(
    path: Path,
    *,
    checkpoint: Path,
    samples: list[WorldSample],
    timelines: dict[str, list[dict[str, object]]],
) -> None:
    lines = [
        "# Trace Temporal Audit",
        "",
        f"- checkpoint: `{checkpoint}`",
        "- purpose: determine whether `first_drift` is absent because the trigger is wrong or because the earliest visible phase is already late in the current observation window.",
        "",
    ]

    transition_count = 0
    first_visible_at_step_1 = 0
    for sample in samples:
        timeline = timelines[sample.sample_id]
        summary = summarize_timeline(timeline)
        if summary["has_none_to_first_to_persist_transition"]:
            transition_count += 1
        if summary["earliest_non_none_step"] == 1:
            first_visible_at_step_1 += 1

        lines.extend(
            [
                f"## {sample.sample_id}",
                "",
                f"- prefix: `{sample.prefix}`",
                f"- gold: `{sample.gold_continuation}`",
                f"- earliest_non_none_step: `{summary['earliest_non_none_step']}`",
                f"- earliest_first_drift_step: `{summary['earliest_first_drift_step']}`",
                f"- earliest_persist_2_step: `{summary['earliest_persist_2_step']}`",
                f"- has_none_to_first_to_persist_transition: `{summary['has_none_to_first_to_persist_transition']}`",
                "",
                "| step | phase | error_mode | persist | uncertainty | gap | local_site | trigger_hits_v1 | candidate_moves_v1_translation | continuation |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in timeline:
            lines.append(
                "| "
                f"{row['step']} | {row['phase']} | {row['error_mode']} | {row['error_persist_steps']} | "
                f"{row['uncertainty_band']} ({row['uncertainty_value']}) | {row['top_gap_band']} ({row['top_gap_value']}) | "
                f"{row['local_site']} | {','.join(row['trigger_hits_v1']) or '-'} | "
                f"{','.join(row['candidate_moves_v1_translation']) or '-'} | "
                f"`{str(row['continuation']).replace('|', '/')}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Aggregate Read",
            "",
            f"- samples_with_visible_first_to_persist_transition: `{transition_count}/{len(samples)}`",
            f"- samples_where_earliest_non_none_is_step_1: `{first_visible_at_step_1}/{len(samples)}`",
            "- interpretation rule:",
            "  - if most samples first become non-none at step 1 and immediately move into persist_2, then current observation is already late for first_drift in this trace schema.",
            "  - if some samples show a clean `none -> first_drift -> persist_2` path, then first_drift exists in the current window and the problem is more likely trigger/policy design.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    samples = load_world_samples(args.input)
    sp = load_sentencepiece(args.tokenizer_model)
    model = load_real_cael_model(args, device)

    timelines = {
        sample.sample_id: decode_prefix_timeline(
            sample,
            model=model,
            sp=sp,
            device=device,
            max_new_tokens=args.max_new_tokens,
            binary_forward=args.binary_forward,
        )
        for sample in samples
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, checkpoint=args.cael_checkpoint, samples=samples, timelines=timelines)


if __name__ == "__main__":
    main()
