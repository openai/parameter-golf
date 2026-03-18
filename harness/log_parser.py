from __future__ import annotations

import re
from pathlib import Path
from typing import Any


TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+train_loss:(?P<train_loss>[-+0-9.eE]+)\s+"
    r"train_time:(?P<train_time_ms>[-+0-9.eE]+)ms\s+step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms"
    r"(?:\s+tok_s:(?P<tok_s>[-+0-9.eE]+))?"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+val_loss:(?P<val_loss>[-+0-9.eE]+)\s+"
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)\s+train_time:(?P<train_time_ms>[-+0-9.eE]+)ms\s+"
    r"step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms"
)
FINAL_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact\s+val_loss:(?P<val_loss>[-+0-9.eE]+)\s+val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
SERIALIZED_TORCH_RE = re.compile(r"Serialized model int8\+zlib:\s+(?P<bytes>\d+)\s+bytes")
SERIALIZED_MLX_RE = re.compile(r"serialized_model_int8_zlib:(?P<bytes>\d+)\s+bytes")
TOTAL_SIZE_RE = re.compile(r"Total submission size int8\+zlib:\s+(?P<bytes>\d+)\s+bytes")
RAW_MODEL_TORCH_RE = re.compile(r"Serialized model:\s+(?P<bytes>\d+)\s+bytes")
RAW_MODEL_MLX_RE = re.compile(r"saved_model:(?P<path>.+?)\s+bytes:(?P<bytes>\d+)")
STOP_RE = re.compile(
    r"stopping_early:\s+wallclock_cap\s+train_time:(?P<train_time_ms>[-+0-9.eE]+)ms\s+step:(?P<step>\d+)/(?P<iterations>\d+)"
)
PEAK_MEM_RE = re.compile(
    r"peak memory allocated:\s+(?P<allocated_mib>\d+)\s+MiB\s+reserved:\s+(?P<reserved_mib>\d+)\s+MiB"
)
MODEL_PARAMS_RE = re.compile(r"model_params:(?P<model_params>\d+)")
DATASET_RE = re.compile(r"train_loader:dataset:(?P<dataset>[^\s]+)\s+train_shards:(?P<actual>\d+)(?:/(?P<expected>\d+))?")
SUBSET_RE = re.compile(
    r"WARNING:\s+train_loader:subset\s+dataset:(?P<dataset>[^\s]+)\s+train_shards:(?P<actual>\d+)/(?P<expected>\d+)"
)
VAL_SHARDS_RE = re.compile(r"val_loader:shards\s+pattern=(?P<pattern>.+?)\s+tokens:(?P<tokens>\d+)")


def _float(value: str) -> float:
    return float(value)


def _int(value: str) -> int:
    return int(float(value))


def parse_log(log_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "log_path": str(log_path),
        "train_history": [],
        "val_history": [],
        "best_val_bpb": None,
        "best_val_loss": None,
        "last_pre_quant_val_bpb": None,
        "last_pre_quant_val_loss": None,
        "final_roundtrip_val_bpb": None,
        "final_roundtrip_val_loss": None,
        "quant_gap_bpb": None,
        "serialized_model_int8_zlib_bytes": None,
        "total_submission_size_int8_zlib_bytes": None,
        "raw_model_bytes": None,
        "stopped_early_wallclock": False,
        "stop_step": None,
        "iterations": None,
        "last_train_loss": None,
        "last_train_time_ms": None,
        "peak_memory_allocated_mib": None,
        "peak_memory_reserved_mib": None,
        "model_params": None,
        "dataset": None,
        "train_shards_actual": None,
        "train_shards_expected": None,
        "subset_warning": False,
        "val_tokens": None,
        "metrics_valid": False,
        "parse_warnings": [],
        "has_training_signal": False,
        "has_validation_signal": False,
        "has_final_roundtrip_signal": False,
        "has_dataset_signal": False,
        "log_signature": None,
    }

    if not log_path.is_file():
        metrics["parse_warnings"].append("log_file_missing")
        return metrics

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if match := TRAIN_RE.search(line):
                row = {
                    "step": _int(match.group("step")),
                    "iterations": _int(match.group("iterations")),
                    "train_loss": _float(match.group("train_loss")),
                    "train_time_ms": _float(match.group("train_time_ms")),
                    "step_avg_ms": _float(match.group("step_avg_ms")),
                    "tok_s": _float(match.group("tok_s")) if match.group("tok_s") else None,
                }
                metrics["train_history"].append(row)
                metrics["has_training_signal"] = True
                metrics["iterations"] = row["iterations"]
                metrics["last_train_loss"] = row["train_loss"]
                metrics["last_train_time_ms"] = row["train_time_ms"]
                continue

            if match := VAL_RE.search(line):
                row = {
                    "step": _int(match.group("step")),
                    "iterations": _int(match.group("iterations")),
                    "val_loss": _float(match.group("val_loss")),
                    "val_bpb": _float(match.group("val_bpb")),
                    "train_time_ms": _float(match.group("train_time_ms")),
                    "step_avg_ms": _float(match.group("step_avg_ms")),
                }
                metrics["val_history"].append(row)
                metrics["has_training_signal"] = True
                metrics["has_validation_signal"] = True
                metrics["iterations"] = row["iterations"]
                metrics["last_pre_quant_val_loss"] = row["val_loss"]
                metrics["last_pre_quant_val_bpb"] = row["val_bpb"]
                if metrics["best_val_bpb"] is None or row["val_bpb"] < metrics["best_val_bpb"]:
                    metrics["best_val_bpb"] = row["val_bpb"]
                    metrics["best_val_loss"] = row["val_loss"]
                continue

            if match := FINAL_RE.search(line):
                metrics["final_roundtrip_val_loss"] = _float(match.group("val_loss"))
                metrics["final_roundtrip_val_bpb"] = _float(match.group("val_bpb"))
                metrics["has_final_roundtrip_signal"] = True
                continue

            if match := SERIALIZED_TORCH_RE.search(line):
                metrics["serialized_model_int8_zlib_bytes"] = _int(match.group("bytes"))
                continue

            if match := SERIALIZED_MLX_RE.search(line):
                metrics["serialized_model_int8_zlib_bytes"] = _int(match.group("bytes"))
                continue

            if match := TOTAL_SIZE_RE.search(line):
                metrics["total_submission_size_int8_zlib_bytes"] = _int(match.group("bytes"))
                continue

            if match := RAW_MODEL_TORCH_RE.search(line):
                metrics["raw_model_bytes"] = _int(match.group("bytes"))
                continue

            if match := RAW_MODEL_MLX_RE.search(line):
                metrics["raw_model_bytes"] = _int(match.group("bytes"))
                continue

            if match := STOP_RE.search(line):
                metrics["stopped_early_wallclock"] = True
                metrics["has_training_signal"] = True
                metrics["stop_step"] = _int(match.group("step"))
                metrics["iterations"] = _int(match.group("iterations"))
                metrics["last_train_time_ms"] = _float(match.group("train_time_ms"))
                continue

            if match := PEAK_MEM_RE.search(line):
                metrics["peak_memory_allocated_mib"] = _int(match.group("allocated_mib"))
                metrics["peak_memory_reserved_mib"] = _int(match.group("reserved_mib"))
                continue

            if match := MODEL_PARAMS_RE.search(line):
                metrics["model_params"] = _int(match.group("model_params"))
                continue

            if match := SUBSET_RE.search(line):
                metrics["dataset"] = match.group("dataset")
                metrics["train_shards_actual"] = _int(match.group("actual"))
                metrics["train_shards_expected"] = _int(match.group("expected"))
                metrics["subset_warning"] = True
                metrics["has_dataset_signal"] = True
                continue

            if match := DATASET_RE.search(line):
                metrics["dataset"] = match.group("dataset")
                metrics["train_shards_actual"] = _int(match.group("actual"))
                metrics["train_shards_expected"] = _int(match.group("expected")) if match.group("expected") else None
                metrics["has_dataset_signal"] = True
                continue

            if match := VAL_SHARDS_RE.search(line):
                metrics["val_tokens"] = _int(match.group("tokens"))
                metrics["has_dataset_signal"] = True
                continue

    if (
        metrics["final_roundtrip_val_bpb"] is not None
        and metrics["last_pre_quant_val_bpb"] is not None
    ):
        metrics["quant_gap_bpb"] = metrics["final_roundtrip_val_bpb"] - metrics["last_pre_quant_val_bpb"]

    if metrics["stop_step"] is None:
        if metrics["val_history"]:
            metrics["stop_step"] = metrics["val_history"][-1]["step"]
        elif metrics["train_history"]:
            metrics["stop_step"] = metrics["train_history"][-1]["step"]

    if metrics["subset_warning"]:
        metrics["parse_warnings"].append("subset_dataset_detected")
    if not metrics["has_dataset_signal"]:
        metrics["parse_warnings"].append("dataset_signal_missing")
    if not metrics["has_final_roundtrip_signal"]:
        metrics["parse_warnings"].append("no_final_roundtrip_metric_detected")

    strong_signal = bool(
        metrics["has_training_signal"]
        or metrics["has_validation_signal"]
        or metrics["has_final_roundtrip_signal"]
        or metrics["stopped_early_wallclock"]
        or metrics["serialized_model_int8_zlib_bytes"] is not None
        or metrics["raw_model_bytes"] is not None
    )
    valid_signal = bool(
        metrics["has_training_signal"]
        or metrics["has_validation_signal"]
        or metrics["has_final_roundtrip_signal"]
    )
    if not valid_signal:
        metrics["parse_warnings"].append("no_train_or_val_lines_detected")
    if not strong_signal:
        metrics["parse_warnings"].append("trainer_output_signature_weak")
    metrics["log_signature"] = "trainer_log" if strong_signal else None
    metrics["metrics_valid"] = valid_signal

    return metrics
