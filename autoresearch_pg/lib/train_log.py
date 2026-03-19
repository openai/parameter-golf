from __future__ import annotations

import re
from pathlib import Path
from typing import Any


TRAIN_VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) val_loss:(?P<val_loss>[0-9.]+) "
    r"val_bpb:(?P<val_bpb>[0-9.]+) train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>[0-9.]+)ms"
)
TRAIN_LOSS_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) train_loss:(?P<train_loss>[0-9.]+) "
    r"train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>[0-9.]+)ms"
)
FINAL_RE = re.compile(
    r"final_int8_zlib_roundtrip val_loss:(?P<val_loss>[0-9.]+) "
    r"val_bpb:(?P<val_bpb>[0-9.]+) eval_time:(?P<eval_time_ms>\d+)ms"
)
FINAL_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"
)
STOP_RE = re.compile(
    r"stopping_early: (?P<reason>\S+) train_time:(?P<train_time_ms>\d+)ms step:(?P<step>\d+)/(?P<iterations>\d+)"
)
MODEL_PARAMS_RE = re.compile(r"model_params:(?P<model_params>\d+)")
TRAIN_SHARDS_RE = re.compile(r"train_loader:dataset:(?P<dataset>\S+) train_shards:(?P<train_shards>\d+)")
VAL_TOKENS_RE = re.compile(r"val_loader:shards pattern=(?P<pattern>.+) tokens:(?P<val_tokens>\d+)")
PEAK_MEM_RE = re.compile(
    r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB"
)
MODEL_INT8_RE = re.compile(r"Serialized model int8\+zlib: (?P<bytes>\d+) bytes")
MODEL_RAW_RE = re.compile(r"Serialized model: (?P<bytes>\d+) bytes")
CODE_SIZE_RE = re.compile(r"Code size: (?P<bytes>\d+) bytes")
TOTAL_RE = re.compile(r"Total submission size int8\+zlib: (?P<bytes>\d+) bytes")


def _as_int(match: re.Match[str], name: str) -> int:
    return int(match.group(name))


def _as_float(match: re.Match[str], name: str) -> float:
    return float(match.group(name))


def parse_train_log(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path.resolve()),
        "train_vals": [],
        "train_losses": [],
        "artifact": {},
        "timing": {},
    }

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if match := TRAIN_VAL_RE.search(line):
            payload["train_vals"].append(
                {
                    "step": _as_int(match, "step"),
                    "iterations": _as_int(match, "iterations"),
                    "val_loss": _as_float(match, "val_loss"),
                    "val_bpb": _as_float(match, "val_bpb"),
                    "train_time_ms": _as_int(match, "train_time_ms"),
                    "step_avg_ms": _as_float(match, "step_avg_ms"),
                }
            )
            payload["timing"]["train_time_ms"] = _as_int(match, "train_time_ms")
            payload["timing"]["step_avg_ms"] = _as_float(match, "step_avg_ms")
            continue

        if match := TRAIN_LOSS_RE.search(line):
            payload["train_losses"].append(
                {
                    "step": _as_int(match, "step"),
                    "iterations": _as_int(match, "iterations"),
                    "train_loss": _as_float(match, "train_loss"),
                    "train_time_ms": _as_int(match, "train_time_ms"),
                    "step_avg_ms": _as_float(match, "step_avg_ms"),
                }
            )
            payload["timing"]["train_time_ms"] = _as_int(match, "train_time_ms")
            payload["timing"]["step_avg_ms"] = _as_float(match, "step_avg_ms")
            continue

        if match := FINAL_RE.search(line):
            payload["final_int8"] = {
                "val_loss": _as_float(match, "val_loss"),
                "val_bpb": _as_float(match, "val_bpb"),
                "eval_time_ms": _as_int(match, "eval_time_ms"),
            }
            continue

        if match := FINAL_EXACT_RE.search(line):
            payload["final_int8_exact"] = {
                "val_loss": _as_float(match, "val_loss"),
                "val_bpb": _as_float(match, "val_bpb"),
            }
            continue

        if match := STOP_RE.search(line):
            payload["stopping"] = {
                "reason": match.group("reason"),
                "train_time_ms": _as_int(match, "train_time_ms"),
                "step": _as_int(match, "step"),
                "iterations": _as_int(match, "iterations"),
            }
            payload["timing"]["train_time_ms"] = _as_int(match, "train_time_ms")
            continue

        if match := MODEL_PARAMS_RE.search(line):
            payload["model_params"] = _as_int(match, "model_params")
            continue

        if match := TRAIN_SHARDS_RE.search(line):
            payload["dataset"] = {
                "name": match.group("dataset"),
                "train_shards": _as_int(match, "train_shards"),
            }
            continue

        if match := VAL_TOKENS_RE.search(line):
            payload["dataset"] = payload.get("dataset", {})
            payload["dataset"]["val_pattern"] = match.group("pattern")
            payload["dataset"]["val_tokens"] = _as_int(match, "val_tokens")
            continue

        if match := PEAK_MEM_RE.search(line):
            payload["peak_memory"] = {
                "allocated_mib": _as_int(match, "allocated"),
                "reserved_mib": _as_int(match, "reserved"),
            }
            continue

        if match := MODEL_INT8_RE.search(line):
            payload["artifact"]["model_int8_zlib_bytes"] = _as_int(match, "bytes")
            continue

        if match := MODEL_RAW_RE.search(line):
            payload["artifact"]["model_raw_bytes"] = _as_int(match, "bytes")
            continue

        if match := CODE_SIZE_RE.search(line):
            payload["artifact"]["code_bytes"] = _as_int(match, "bytes")
            continue

        if match := TOTAL_RE.search(line):
            payload["artifact"]["bytes_total"] = _as_int(match, "bytes")
            continue

    if payload["train_vals"]:
        payload["last_train_val"] = payload["train_vals"][-1]
    if payload["train_losses"]:
        payload["last_train_loss"] = payload["train_losses"][-1]
    return payload
