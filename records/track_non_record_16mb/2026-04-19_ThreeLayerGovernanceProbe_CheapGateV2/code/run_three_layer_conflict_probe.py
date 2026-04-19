from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from experiment_env import guard_experiment_env
from build_three_layer_prototype_data import (
    WorldSample,
    build_records,
    choose_device,
    load_real_cael_model,
    load_world_samples,
    load_sentencepiece,
)

guard_experiment_env(script_name="run_three_layer_conflict_probe", require_torch=True)


DEFAULT_INPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_seed_samples.jsonl")
DEFAULT_OUTPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_conflict_probe_run1")
DEFAULT_CHECKPOINT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_builder_assets/minimal_real_cael_checkpoint.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a very small three-layer cheap conflict probe.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cael-source", choices=("mock_v0", "real_cael"), default="real_cael")
    parser.add_argument("--cael-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=Path("/Users/seryn/Documents/Parameter_Golf/repo/data/tokenizers/fineweb_1024_bpe.model"),
    )
    parser.add_argument("--device", type=str, default="cpu")
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


def short_history_prefix(text: str) -> str:
    stripped = text.strip()
    if "." in stripped:
        parts = [part.strip() for part in stripped.split(".") if part.strip()]
        if parts:
            return parts[-1] + "."
    words = stripped.split()
    if len(words) <= 4:
        return stripped
    return " ".join(words[-4:])


def framed_prefix(text: str) -> str:
    return f"Continue carefully. {text.strip()}"


def make_variant_samples(base_samples: list[WorldSample]) -> dict[str, list[WorldSample]]:
    variants: dict[str, list[WorldSample]] = {
        "real_baseline": base_samples,
        "context_short_history": [],
        "context_framed_prefix": [],
    }
    for sample in base_samples:
        variants["context_short_history"].append(
            WorldSample(
                sample_id=sample.sample_id,
                prefix=short_history_prefix(sample.prefix),
                gold_continuation=sample.gold_continuation,
            )
        )
        variants["context_framed_prefix"].append(
            WorldSample(
                sample_id=sample.sample_id,
                prefix=framed_prefix(sample.prefix),
                gold_continuation=sample.gold_continuation,
            )
        )
    return variants


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_inspection(path: Path, variant_rows: dict[str, list[dict[str, object]]]) -> None:
    lines = [
        "# Three-layer cheap conflict probe inspection",
        "",
        "- mechanism: single-backbone, multi-view rollout",
        "- conflict_source_v0: context/history perturbation",
        "- comparison_focus: continuation / trace / monday_move / seryn_audit",
        "",
    ]
    sample_ids = [row["sample_id"] for row in variant_rows["real_baseline"]]
    for sample_id in sample_ids:
        lines.append(f"## {sample_id}")
        lines.append("")
        for variant_name, rows in variant_rows.items():
            row = next(item for item in rows if item["sample_id"] == sample_id)
            lines.append(f"### {variant_name}")
            lines.append(f"- prefix: `{row['prefix']}`")
            lines.append(f"- gold: `{row['gold_continuation']}`")
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
    variant_samples = make_variant_samples(world_samples)
    device = choose_device(args.device)
    sp = None
    real_model = None
    if args.cael_source == "real_cael":
        sp = load_sentencepiece(args.tokenizer_model)
        real_model = load_real_cael_model(args, device)

    all_rows: dict[str, list[dict[str, object]]] = {}
    summary_rows: list[dict[str, object]] = []
    for variant_name, samples in variant_samples.items():
        rows: list[dict[str, object]] = []
        for sample in samples:
            cael, monday, seryn = build_records(
                sample,
                cael_source=args.cael_source,
                real_model=real_model,
                sp=sp,
                device=device,
                max_new_tokens=args.max_new_tokens,
                binary_forward=args.binary_forward,
            )
            row = {
                "variant": variant_name,
                "sample_id": sample.sample_id,
                "prefix": sample.prefix,
                "gold_continuation": sample.gold_continuation,
                "cael_continuation": cael.cael_continuation,
                "cael_trace": asdict(cael.cael_trace),
                "target_move": monday.target_move,
                "post_monday_continuation": seryn.post_monday_continuation,
                "target_audit": seryn.target_audit,
            }
            rows.append(row)
            summary_rows.append(
                {
                    "variant": variant_name,
                    "sample_id": sample.sample_id,
                    "error_mode": cael.cael_trace.error_mode,
                    "uncertainty_band": cael.cael_trace.uncertainty_band,
                    "top_gap_band": cael.cael_trace.top_gap_band,
                    "intervention_effect": cael.cael_trace.intervention_effect,
                    "local_site": cael.cael_trace.local_site,
                    "target_move": monday.target_move,
                    "target_audit": seryn.target_audit,
                }
            )
        all_rows[variant_name] = rows
        write_jsonl(output_root / f"{variant_name}.jsonl", rows)

    write_jsonl(output_root / "summary_rows.jsonl", summary_rows)
    write_inspection(output_root / "inspection.md", all_rows)
    manifest = {
        "cael_source": args.cael_source,
        "cael_checkpoint": str(args.cael_checkpoint) if args.cael_checkpoint else None,
        "device": str(device),
        "variants": list(variant_samples.keys()),
        "conflict_source": "context_history_perturbation_v0",
        "planned_axes": {
            "real_baseline": "original_prefix",
            "context_short_history": "shortened_history_view",
            "context_framed_prefix": "light_framing_header_view",
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
