from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from experiment_env import guard_experiment_env

guard_experiment_env(script_name="run_aux_supervision_probe_v0", require_torch=True)

import torch
from torch import nn
from torch.nn import functional as F

from build_three_layer_prototype_data import (
    CaelTrace,
    WorldSample,
    build_records,
    choose_device,
    load_real_cael_model,
    load_sentencepiece,
    load_world_samples,
    mismatch_score,
)
from run_understanding_gate_probe_eval import make_config
from run_usefulness_probe_v0 import compute_arm_row, select_gate_trigger


DEFAULT_INPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_seed_samples.jsonl")
DEFAULT_OUTPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_aux_supervision_probe_v0")
DEFAULT_CHECKPOINT = Path(
    "/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_builder_assets/minimal_real_cael_checkpoint_300steps.pt"
)
DEFAULT_DECISION_SUPPORT = Path("/Users/seryn/Documents/Parameter_Golf/repo/decision_shape_v2_support_samples.jsonl")


@dataclass(frozen=True)
class AuxTrainExample:
    sample_id: str
    text: str
    zone_label: int
    zone_name: str
    decision_label: int
    decision_name: str


ZONE_NAMES = ("pre_error_zone", "error_zone")
DECISION_NAMES = ("no_action", "mark_only", "redirect")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a very small auxiliary supervision probe on top of the current real-Cael checkpoint.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cael-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=Path("/Users/seryn/Documents/Parameter_Golf/repo/data/tokenizers/fineweb_1024_bpe.model"),
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux-alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--shared-repeats", type=int, default=2)
    parser.add_argument("--router-temperature", type=float, default=1.5)
    parser.add_argument("--op3-hidden-dim", type=int, default=96)
    parser.add_argument("--binary-forward", action="store_true", default=True)
    parser.add_argument("--decision-support-path", type=Path, default=DEFAULT_DECISION_SUPPORT)
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_zone_label(trace: CaelTrace) -> tuple[int, str]:
    if trace.error_mode != "none":
        return 1, "error_zone"
    return 0, "pre_error_zone"


def derive_decision_label(trace: CaelTrace) -> tuple[int, str]:
    if trace.error_mode != "none":
        return 2, "redirect"
    if trace.uncertainty_band == "high" or trace.top_gap_band == "narrow":
        return 1, "mark_only"
    return 0, "no_action"


def build_aux_examples(
    *,
    world_samples: list[WorldSample],
    base_model,
    sp,
    device: torch.device,
    max_new_tokens: int,
    binary_forward: bool,
) -> list[AuxTrainExample]:
    rows: list[AuxTrainExample] = []
    for sample in world_samples:
        cael_record, _monday_record, _seryn_record = build_records(
            sample,
            cael_source="real_cael",
            real_model=base_model,
            sp=sp,
            device=device,
            max_new_tokens=max_new_tokens,
            binary_forward=binary_forward,
        )
        zone_label, zone_name = derive_zone_label(cael_record.cael_trace)
        decision_label, decision_name = derive_decision_label(cael_record.cael_trace)
        rows.append(
            AuxTrainExample(
                sample_id=sample.sample_id,
                text=f"{sample.prefix} {sample.gold_continuation}".strip(),
                zone_label=zone_label,
                zone_name=zone_name,
                decision_label=decision_label,
                decision_name=decision_name,
            )
        )
    return rows


def load_support_examples(path: Path) -> list[AuxTrainExample]:
    rows: list[AuxTrainExample] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(
                AuxTrainExample(
                    sample_id=str(payload["sample_id"]),
                    text=str(payload["text"]),
                    zone_label=int(payload["zone_label"]),
                    zone_name=str(payload["zone_name"]),
                    decision_label=int(payload["decision_label"]),
                    decision_name=str(payload["decision_name"]),
                )
            )
    return rows


def encode_example(sp, text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = [int(sp.bos_id()), *sp.encode(text, out_type=int)]
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
    return tokens[:-1].unsqueeze(0), tokens[1:].unsqueeze(0)


def init_model(args: argparse.Namespace, device: torch.device):
    model = load_real_cael_model(args, device)
    model.train()
    return model


def train_variant(
    *,
    variant_name: str,
    args: argparse.Namespace,
    device: torch.device,
    sp,
    train_examples: list[AuxTrainExample],
    target_family: str,
) -> tuple[Path, dict[str, object]]:
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = init_model(args, device)
    if target_family == "none":
        aux_num_labels = 0
        aux_head = None
    elif target_family == "zone_classification_v0":
        aux_num_labels = len(ZONE_NAMES)
        aux_head = nn.Linear(args.d_model, aux_num_labels).to(device)
    elif target_family == "decision_shape_v1":
        aux_num_labels = len(DECISION_NAMES)
        aux_head = nn.Linear(args.d_model, aux_num_labels).to(device)
    else:
        raise ValueError(f"Unknown target_family: {target_family}")
    parameters = list(model.parameters()) + (list(aux_head.parameters()) if aux_head is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    per_step_rows: list[dict[str, object]] = []

    for step in range(1, args.steps + 1):
        example = train_examples[(step - 1) % len(train_examples)]
        inputs, targets = encode_example(sp, example.text, device)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = model(
            inputs,
            targets,
            binary_forward=args.binary_forward,
            entropy_reg_weight=0.0,
            bridge_entropy_reg_weight=0.0,
        )
        lm_loss = float(stats["lm_loss"].detach().cpu())
        if aux_head is not None:
            aux_logits = aux_head(stats["pooled_hidden_live"])
            if target_family == "zone_classification_v0":
                aux_targets = torch.tensor([example.zone_label], dtype=torch.long, device=device)
                aux_target_name = example.zone_name
            else:
                aux_targets = torch.tensor([example.decision_label], dtype=torch.long, device=device)
                aux_target_name = example.decision_name
            aux_loss = F.cross_entropy(aux_logits, aux_targets, reduction="mean")
            loss = loss + args.aux_alpha * aux_loss
            aux_loss_value = float(aux_loss.detach().cpu())
        else:
            aux_loss_value = 0.0
            aux_target_name = "none"
        loss.backward()
        optimizer.step()
        per_step_rows.append(
            {
                "step": step,
                "sample_id": example.sample_id,
                "zone_name": example.zone_name,
                "decision_name": example.decision_name,
                "target_family": target_family,
                "aux_target_name": aux_target_name,
                "loss": float(loss.detach().cpu()),
                "lm_loss": lm_loss,
                "aux_loss": aux_loss_value,
            }
        )

    checkpoint_path = args.output_root / f"{variant_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    summary = {
        "variant": variant_name,
        "steps": args.steps,
        "target_family": target_family,
        "use_aux": aux_head is not None,
        "aux_alpha": args.aux_alpha if aux_head is not None else 0.0,
        "final_train_loss": per_step_rows[-1]["loss"],
        "final_train_lm_loss": per_step_rows[-1]["lm_loss"],
        "final_train_aux_loss": per_step_rows[-1]["aux_loss"],
        "checkpoint_path": str(checkpoint_path),
        "per_step_rows": per_step_rows,
    }
    return checkpoint_path, summary


def evaluate_variant(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    world_samples: list[WorldSample],
    sp,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    eval_args = argparse.Namespace(**vars(args))
    eval_args.cael_checkpoint = checkpoint_path
    model = load_real_cael_model(eval_args, device)
    rows: list[dict[str, object]] = []
    for sample in world_samples:
        cael_record, _monday_record, _seryn_record = build_records(
            sample,
            cael_source="real_cael",
            real_model=model,
            sp=sp,
            device=device,
            max_new_tokens=args.max_new_tokens,
            binary_forward=args.binary_forward,
        )
        selected_trigger, active_triggers = select_gate_trigger(cael_record.cael_trace)
        for arm, trigger in (("baseline", None), ("cheap_gate_v2_candidate", selected_trigger)):
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

    def summarize(arm: str) -> dict[str, object]:
        arm_rows = [row for row in rows if row["arm"] == arm]
        return {
            "avg_before_mismatch": sum(float(row["before_mismatch"]) for row in arm_rows) / len(arm_rows),
            "avg_after_mismatch": sum(float(row["after_mismatch"]) for row in arm_rows) / len(arm_rows),
            "effect_dist": dict(Counter(str(row["intervention_effect"]) for row in arm_rows)),
            "audit_dist": dict(Counter(str(row["target_audit"]) for row in arm_rows)),
            "negative_flags": dict(Counter(flag for row in arm_rows for flag in row["negative_flags"])),
        }

    summary = {
        "baseline": summarize("baseline"),
        "cheap_gate_v2_candidate": summarize("cheap_gate_v2_candidate"),
    }
    return rows, summary


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(
    *,
    output_root: Path,
    train_examples: list[AuxTrainExample],
    support_examples: list[AuxTrainExample],
    train_summaries: list[dict[str, object]],
    eval_summaries: dict[str, dict[str, object]],
    eval_rows: dict[str, list[dict[str, object]]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    label_rows = [
        {
            "sample_id": row.sample_id,
            "text": row.text,
            "zone_label": row.zone_label,
            "zone_name": row.zone_name,
            "decision_label": row.decision_label,
            "decision_name": row.decision_name,
        }
        for row in train_examples
    ]
    write_jsonl(output_root / "train_examples.jsonl", label_rows)
    write_jsonl(
        output_root / "decision_shape_v2_support_examples.jsonl",
        [
            {
                "sample_id": row.sample_id,
                "text": row.text,
                "zone_label": row.zone_label,
                "zone_name": row.zone_name,
                "decision_label": row.decision_label,
                "decision_name": row.decision_name,
            }
            for row in support_examples
        ],
    )

    for summary in train_summaries:
        variant = str(summary["variant"])
        write_jsonl(output_root / f"{variant}_train_curve.jsonl", summary["per_step_rows"])

    for variant, rows in eval_rows.items():
        write_jsonl(output_root / f"{variant}_usefulness_rows.jsonl", rows)

    summary_csv = output_root / "aux_supervision_probe_results.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "target_family",
                "steps",
                "aux_alpha",
                "final_train_lm_loss",
                "final_train_aux_loss",
                "baseline_avg_before_mismatch",
                "baseline_avg_after_mismatch",
                "gate_v2_avg_before_mismatch",
                "gate_v2_avg_after_mismatch",
                "gate_v2_effect_dist",
                "gate_v2_audit_dist",
                "gate_v2_negative_flags",
            ],
        )
        writer.writeheader()
        for train_summary in train_summaries:
            variant = str(train_summary["variant"])
            eval_summary = eval_summaries[variant]
            writer.writerow(
                {
                    "variant": variant,
                    "target_family": train_summary["target_family"],
                    "steps": train_summary["steps"],
                    "aux_alpha": train_summary["aux_alpha"],
                    "final_train_lm_loss": train_summary["final_train_lm_loss"],
                    "final_train_aux_loss": train_summary["final_train_aux_loss"],
                    "baseline_avg_before_mismatch": eval_summary["baseline"]["avg_before_mismatch"],
                    "baseline_avg_after_mismatch": eval_summary["baseline"]["avg_after_mismatch"],
                    "gate_v2_avg_before_mismatch": eval_summary["cheap_gate_v2_candidate"]["avg_before_mismatch"],
                    "gate_v2_avg_after_mismatch": eval_summary["cheap_gate_v2_candidate"]["avg_after_mismatch"],
                    "gate_v2_effect_dist": json.dumps(eval_summary["cheap_gate_v2_candidate"]["effect_dist"], ensure_ascii=False),
                    "gate_v2_audit_dist": json.dumps(eval_summary["cheap_gate_v2_candidate"]["audit_dist"], ensure_ascii=False),
                    "gate_v2_negative_flags": json.dumps(eval_summary["cheap_gate_v2_candidate"]["negative_flags"], ensure_ascii=False),
                }
            )

    summary_json = {
        "target_families": ["zone_classification_v0", "decision_shape_v1", "decision_shape_v2"],
        "observed_zone_names": list(sorted({row.zone_name for row in train_examples})),
        "observed_decision_names": list(sorted({row.decision_name for row in train_examples})),
        "support_example_count": len(support_examples),
        "train_summaries": train_summaries,
        "eval_summaries": eval_summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Auxiliary Supervision Probe v2",
        "",
        "- goal: test whether supplementing class support lets the decision-shape target move closer to the current useful gate boundary",
        "- target_families: `zone_classification_v0` / `decision_shape_v1` / `decision_shape_v2`",
        f"- observed_zone_names: `{sorted({row.zone_name for row in train_examples})}`",
        f"- observed_decision_names: `{sorted({row.decision_name for row in train_examples})}`",
        f"- decision_shape_v2 support examples: `{len(support_examples)}`",
        "- note: base seed set exposes only `mark_only` / `redirect`; `decision_shape_v2` adds explicit `no_action` support and additional `mark_only` support",
        "",
        "## Variants",
        "",
    ]
    for train_summary in train_summaries:
        variant = str(train_summary["variant"])
        eval_summary = eval_summaries[variant]
        lines.extend(
            [
                f"### {variant}",
                f"- target_family: `{train_summary['target_family']}`",
                f"- final_train_lm_loss: `{train_summary['final_train_lm_loss']:.6f}`",
                f"- final_train_aux_loss: `{train_summary['final_train_aux_loss']:.6f}`",
                f"- baseline avg_after_mismatch: `{eval_summary['baseline']['avg_after_mismatch']:.2f}`",
                f"- cheap_gate_v2 avg_after_mismatch: `{eval_summary['cheap_gate_v2_candidate']['avg_after_mismatch']:.2f}`",
                f"- cheap_gate_v2 effect_dist: `{eval_summary['cheap_gate_v2_candidate']['effect_dist']}`",
                f"- cheap_gate_v2 audit_dist: `{eval_summary['cheap_gate_v2_candidate']['audit_dist']}`",
                f"- cheap_gate_v2 negative_flags: `{eval_summary['cheap_gate_v2_candidate']['negative_flags'] or '-'}`",
                "",
            ]
        )
    (output_root / "inspection.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_root = args.output_root.resolve()
    device = choose_device(args.device)
    sp = load_sentencepiece(args.tokenizer_model)
    world_samples = load_world_samples(args.input)
    base_model = load_real_cael_model(args, device)
    base_train_examples = build_aux_examples(
        world_samples=world_samples,
        base_model=base_model,
        sp=sp,
        device=device,
        max_new_tokens=args.max_new_tokens,
        binary_forward=args.binary_forward,
    )
    support_examples = load_support_examples(args.decision_support_path.resolve())

    train_summaries: list[dict[str, object]] = []
    eval_rows: dict[str, list[dict[str, object]]] = {}
    eval_summaries: dict[str, dict[str, object]] = {}

    for variant_name, target_family in (
        ("train_baseline_minimal", "none"),
        ("train_aux_probe_v0", "zone_classification_v0"),
        ("train_aux_probe_v1", "decision_shape_v1"),
        ("train_aux_probe_v2", "decision_shape_v1"),
    ):
        current_train_examples = base_train_examples if variant_name != "train_aux_probe_v2" else [*base_train_examples, *support_examples]
        checkpoint_path, train_summary = train_variant(
            variant_name=variant_name,
            args=args,
            device=device,
            sp=sp,
            train_examples=current_train_examples,
            target_family=target_family,
        )
        train_summary["train_example_count"] = len(current_train_examples)
        rows, eval_summary = evaluate_variant(
            args=args,
            checkpoint_path=checkpoint_path,
            world_samples=world_samples,
            sp=sp,
            device=device,
        )
        train_summaries.append(train_summary)
        eval_rows[variant_name] = rows
        eval_summaries[variant_name] = eval_summary

    write_outputs(
        output_root=args.output_root,
        train_examples=base_train_examples,
        support_examples=support_examples,
        train_summaries=train_summaries,
        eval_summaries=eval_summaries,
        eval_rows=eval_rows,
    )
    print(json.dumps({"output_root": str(args.output_root), "variants": [row["variant"] for row in train_summaries]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
