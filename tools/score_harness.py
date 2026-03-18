#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SEPARATOR = "=" * 100
VALIDATION_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+) train_time:(?P<train_time_ms>\d+)ms "
    r"step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms(?: tta_updates:(?P<tta_update_steps>\d+))?"
)
ROUNDTRIP_RE = re.compile(
    r"final_int8_zlib_roundtrip val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+) eval_time:(?P<eval_time_ms>\d+)ms"
    r"(?: tta_updates:(?P<tta_update_steps>\d+))?"
)
ROUNDTRIP_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
PEAK_MEMORY_RE = re.compile(
    r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB"
)
RAW_MODEL_RE = re.compile(r"Serialized model: (?P<bytes>\d+) bytes")
CODE_SIZE_RE = re.compile(r"Code size: (?P<bytes>\d+) bytes")
RAW_TOTAL_RE = re.compile(r"Total submission size: (?P<bytes>\d+) bytes")
QUANT_MODEL_RE = re.compile(
    r"Serialized model int8\+zlib: (?P<model_bytes>\d+) bytes "
    r"\(payload:(?P<payload_bytes>\d+) raw_torch:(?P<raw_torch_bytes>\d+) "
    r"payload_ratio:(?P<payload_ratio>[-+0-9.eE]+)x\)"
)
QUANT_TOTAL_RE = re.compile(r"Total submission size int8\+zlib: (?P<bytes>\d+) bytes")
SELECTED_EXPORT_RE = re.compile(r"Selected export candidate: (?P<name>\S+)")
SHARED_DEPTH_RE = re.compile(
    r"shared_depth:enabled:(?P<enabled>True|False) physical_blocks:(?P<physical>\d+) "
    r"logical_layers:(?P<logical>\d+) map:(?P<map>\S+)"
)
EVAL_TTA_RE = re.compile(
    r"eval_tta:enabled:(?P<enabled>True|False) lr:(?P<lr>[-+0-9.eE]+) "
    r"steps_per_batch:(?P<steps>\d+) params:(?P<params>\d+) "
    r"include:(?P<include>\S+) exclude:(?P<exclude>\S+)"
)
TRAINING_END_RE = re.compile(
    r"stopping_early: wallclock_cap train_time:(?P<train_time_ms>\d+)ms step:(?P<step>\d+)/(?P<iterations>\d+)"
)


def cut_preamble(text: str) -> str:
    parts = text.split(SEPARATOR)
    if len(parts) >= 3:
        return SEPARATOR.join(parts[2:])
    return text


def parse_score_jsonl(path: Path) -> dict[str, object]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events:
        raise ValueError(f"No score events found in {path}")
    validations = [event for event in events if event.get("event") == "validation"]
    roundtrip = next((event for event in reversed(events) if event.get("event") == "roundtrip_eval"), None)
    artifact_quantized = next((event for event in reversed(events) if event.get("event") == "artifact_quantized"), None)
    artifact_raw = next((event for event in reversed(events) if event.get("event") == "artifact_raw"), None)
    training_end = next((event for event in reversed(events) if event.get("event") == "training_end"), None)
    run_config = next((event for event in reversed(events) if event.get("event") == "run_config"), None)
    pre_quant = validations[-1] if validations else None

    summary = {
        "artifact_bytes": artifact_quantized.get("total_submission_bytes") if artifact_quantized else None,
        "candidate_name": artifact_quantized.get("candidate_name") if artifact_quantized else None,
        "code_bytes": run_config.get("code_bytes") if run_config else None,
        "eval_time_ms": roundtrip.get("eval_time_ms") if roundtrip else None,
        "final_step": training_end.get("final_step") if training_end else None,
        "iterations": training_end.get("iterations") if training_end else None,
        "peak_memory_allocated_mib": training_end.get("peak_memory_allocated_mib") if training_end else None,
        "peak_memory_reserved_mib": training_end.get("peak_memory_reserved_mib") if training_end else None,
        "physical_block_count": run_config.get("physical_block_count") if run_config else None,
        "post_quant_bpb": roundtrip.get("val_bpb") if roundtrip else None,
        "post_quant_val_loss": roundtrip.get("val_loss") if roundtrip else None,
        "pre_quant_bpb": pre_quant.get("val_bpb") if pre_quant else None,
        "pre_quant_step": pre_quant.get("step") if pre_quant else None,
        "pre_quant_train_time_ms": pre_quant.get("train_time_ms") if pre_quant else None,
        "pre_quant_val_loss": pre_quant.get("val_loss") if pre_quant else None,
        "qat_enabled": run_config.get("qat_enabled") if run_config else None,
        "qat_last_steps": run_config.get("qat_last_steps") if run_config else None,
        "qat_param_count": run_config.get("qat_param_count") if run_config else None,
        "qat_quant_scheme": run_config.get("qat_quant_scheme") if run_config else None,
        "quant_penalty_bpb": None,
        "quant_penalty_val_loss": None,
        "quantized_model_bytes": artifact_quantized.get("quantized_model_bytes") if artifact_quantized else None,
        "raw_model_bytes": artifact_raw.get("model_bytes") if artifact_raw else None,
        "run_id": events[-1].get("run_id"),
        "shared_depth_blocks": run_config.get("shared_depth_blocks") if run_config else None,
        "shared_depth_enabled": run_config.get("shared_depth_enabled") if run_config else None,
        "shared_depth_map": run_config.get("shared_depth_map") if run_config else None,
        "tta_enabled": run_config.get("tta_enabled") if run_config else None,
        "tta_exclude_patterns": run_config.get("tta_exclude_patterns") if run_config else None,
        "tta_include_patterns": run_config.get("tta_include_patterns") if run_config else None,
        "tta_lr": run_config.get("tta_lr") if run_config else None,
        "tta_param_count": run_config.get("tta_param_count") if run_config else None,
        "tta_steps_per_batch": run_config.get("tta_steps_per_batch") if run_config else None,
        "tta_update_steps": roundtrip.get("tta_update_steps") if roundtrip else None,
        "train_time_ms": training_end.get("train_time_ms") if training_end else None,
    }
    if pre_quant and roundtrip:
        summary["quant_penalty_bpb"] = roundtrip["val_bpb"] - pre_quant["val_bpb"]
        summary["quant_penalty_val_loss"] = roundtrip["val_loss"] - pre_quant["val_loss"]
    return summary


def parse_text_log(path: Path) -> dict[str, object]:
    text = cut_preamble(path.read_text(encoding="utf-8"))
    validations: list[dict[str, object]] = []
    roundtrip: dict[str, object] | None = None
    raw_model_bytes = None
    code_bytes = None
    raw_total_bytes = None
    quantized_model_bytes = None
    quant_payload_bytes = None
    quant_raw_torch_bytes = None
    quant_payload_ratio = None
    quant_total_bytes = None
    candidate_name = None
    peak_memory_allocated_mib = None
    peak_memory_reserved_mib = None
    stopped_train_time_ms = None
    stopped_step = None
    stopped_iterations = None
    tta_enabled = None
    tta_lr = None
    tta_steps_per_batch = None
    tta_param_count = None
    tta_include_patterns = None
    tta_exclude_patterns = None
    shared_depth_enabled = None
    shared_depth_blocks = None
    shared_depth_map = None
    physical_block_count = None

    for line in text.splitlines():
        if match := VALIDATION_RE.search(line):
            validations.append(
                {
                    "iterations": int(match.group("iterations")),
                    "step": int(match.group("step")),
                    "step_avg_ms": float(match.group("step_avg_ms")),
                    "tta_update_steps": int(match.group("tta_update_steps")) if match.group("tta_update_steps") else None,
                    "train_time_ms": int(match.group("train_time_ms")),
                    "val_bpb": float(match.group("val_bpb")),
                    "val_loss": float(match.group("val_loss")),
                }
            )
            continue
        if match := ROUNDTRIP_RE.search(line):
            roundtrip = {
                "eval_time_ms": int(match.group("eval_time_ms")),
                "tta_update_steps": int(match.group("tta_update_steps")) if match.group("tta_update_steps") else None,
                "val_bpb": float(match.group("val_bpb")),
                "val_loss": float(match.group("val_loss")),
            }
            continue
        if match := ROUNDTRIP_EXACT_RE.search(line):
            if roundtrip is None:
                roundtrip = {}
            roundtrip["val_bpb"] = float(match.group("val_bpb"))
            roundtrip["val_loss"] = float(match.group("val_loss"))
            continue
        if match := PEAK_MEMORY_RE.search(line):
            peak_memory_allocated_mib = int(match.group("allocated"))
            peak_memory_reserved_mib = int(match.group("reserved"))
            continue
        if match := RAW_MODEL_RE.search(line):
            raw_model_bytes = int(match.group("bytes"))
            continue
        if match := CODE_SIZE_RE.search(line):
            code_bytes = int(match.group("bytes"))
            continue
        if match := RAW_TOTAL_RE.search(line):
            raw_total_bytes = int(match.group("bytes"))
            continue
        if match := QUANT_MODEL_RE.search(line):
            quantized_model_bytes = int(match.group("model_bytes"))
            quant_payload_bytes = int(match.group("payload_bytes"))
            quant_raw_torch_bytes = int(match.group("raw_torch_bytes"))
            quant_payload_ratio = float(match.group("payload_ratio"))
            continue
        if match := QUANT_TOTAL_RE.search(line):
            quant_total_bytes = int(match.group("bytes"))
            continue
        if match := SELECTED_EXPORT_RE.search(line):
            candidate_name = match.group("name")
            continue
        if match := SHARED_DEPTH_RE.search(line):
            shared_depth_enabled = match.group("enabled") == "True"
            physical_block_count = int(match.group("physical"))
            logical_layers = int(match.group("logical"))
            shared_depth_map = [int(item) for item in match.group("map").split(",")]
            shared_depth_blocks = physical_block_count if shared_depth_enabled else logical_layers
            continue
        if match := EVAL_TTA_RE.search(line):
            tta_enabled = match.group("enabled") == "True"
            tta_lr = float(match.group("lr"))
            tta_steps_per_batch = int(match.group("steps"))
            tta_param_count = int(match.group("params"))
            tta_include_patterns = [] if match.group("include") == "-" else match.group("include").split(",")
            tta_exclude_patterns = [] if match.group("exclude") == "-" else match.group("exclude").split(",")
            continue
        if match := TRAINING_END_RE.search(line):
            stopped_train_time_ms = int(match.group("train_time_ms"))
            stopped_step = int(match.group("step"))
            stopped_iterations = int(match.group("iterations"))

    pre_quant = validations[-1] if validations else None
    run_id = path.parent.name if path.name == "train.log" else path.stem
    summary = {
        "artifact_bytes": quant_total_bytes,
        "candidate_name": candidate_name,
        "code_bytes": code_bytes,
        "eval_time_ms": roundtrip.get("eval_time_ms") if roundtrip else None,
        "final_step": stopped_step if stopped_step is not None else (pre_quant.get("step") if pre_quant else None),
        "iterations": stopped_iterations if stopped_iterations is not None else (pre_quant.get("iterations") if pre_quant else None),
        "peak_memory_allocated_mib": peak_memory_allocated_mib,
        "peak_memory_reserved_mib": peak_memory_reserved_mib,
        "physical_block_count": physical_block_count,
        "post_quant_bpb": roundtrip.get("val_bpb") if roundtrip else None,
        "post_quant_val_loss": roundtrip.get("val_loss") if roundtrip else None,
        "pre_quant_bpb": pre_quant.get("val_bpb") if pre_quant else None,
        "pre_quant_step": pre_quant.get("step") if pre_quant else None,
        "pre_quant_train_time_ms": pre_quant.get("train_time_ms") if pre_quant else None,
        "pre_quant_val_loss": pre_quant.get("val_loss") if pre_quant else None,
        "quant_payload_bytes": quant_payload_bytes,
        "quant_payload_ratio": quant_payload_ratio,
        "quant_penalty_bpb": None,
        "quant_penalty_val_loss": None,
        "quant_raw_torch_bytes": quant_raw_torch_bytes,
        "quantized_model_bytes": quantized_model_bytes,
        "raw_model_bytes": raw_model_bytes,
        "raw_total_bytes": raw_total_bytes,
        "run_id": run_id,
        "shared_depth_blocks": shared_depth_blocks,
        "shared_depth_enabled": shared_depth_enabled,
        "shared_depth_map": shared_depth_map,
        "tta_enabled": tta_enabled,
        "tta_exclude_patterns": tta_exclude_patterns,
        "tta_include_patterns": tta_include_patterns,
        "tta_lr": tta_lr,
        "tta_param_count": tta_param_count,
        "tta_steps_per_batch": tta_steps_per_batch,
        "tta_update_steps": roundtrip.get("tta_update_steps") if roundtrip else None,
        "train_time_ms": stopped_train_time_ms if stopped_train_time_ms is not None else (pre_quant.get("train_time_ms") if pre_quant else None),
    }
    if pre_quant and roundtrip:
        summary["quant_penalty_bpb"] = roundtrip["val_bpb"] - pre_quant["val_bpb"]
        summary["quant_penalty_val_loss"] = roundtrip["val_loss"] - pre_quant["val_loss"]
    return summary


def load_summary(path: Path) -> dict[str, object]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".jsonl":
        return parse_score_jsonl(path)
    return parse_text_log(path)


def render_text(summary: dict[str, object]) -> str:
    def format_value(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.8f}".rstrip("0").rstrip(".")
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        return str(value)

    ordered_fields = [
        "run_id",
        "candidate_name",
        "qat_enabled",
        "qat_last_steps",
        "qat_quant_scheme",
        "qat_param_count",
        "shared_depth_enabled",
        "shared_depth_blocks",
        "physical_block_count",
        "shared_depth_map",
        "tta_enabled",
        "tta_lr",
        "tta_steps_per_batch",
        "tta_param_count",
        "tta_update_steps",
        "tta_include_patterns",
        "tta_exclude_patterns",
        "pre_quant_bpb",
        "post_quant_bpb",
        "quant_penalty_bpb",
        "pre_quant_val_loss",
        "post_quant_val_loss",
        "quant_penalty_val_loss",
        "pre_quant_step",
        "train_time_ms",
        "eval_time_ms",
        "artifact_bytes",
        "quantized_model_bytes",
        "raw_model_bytes",
        "code_bytes",
        "peak_memory_allocated_mib",
        "peak_memory_reserved_mib",
    ]
    return "\n".join(
        f"{field}: {format_value(summary[field])}"
        for field in ordered_fields
        if field in summary and summary[field] is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Parameter Golf score logs.")
    parser.add_argument("path", type=Path, help="Path to a text log, score JSONL, or score summary JSON.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    args = parser.parse_args()

    summary = load_summary(args.path)
    if args.format == "json":
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    print(render_text(summary))


if __name__ == "__main__":
    main()
