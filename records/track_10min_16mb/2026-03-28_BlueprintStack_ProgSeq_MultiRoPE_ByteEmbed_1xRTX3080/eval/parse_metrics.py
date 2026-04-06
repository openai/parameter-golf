#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FINAL_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"
)
TOTAL_BYTES_RE = re.compile(r"Total submission size(?: (?P<label>[^:]+))?: (?P<bytes>\d+) bytes")
TRAIN_TIME_RE = re.compile(r"train_time:(?P<ms>\d+)ms")
EVAL_TIME_RE = re.compile(r"eval_time:(?P<ms>\d+)ms")
COMPRESSOR_RE = re.compile(r"Serialized model (?P<label>[^:]+): (?P<bytes>\d+) bytes")
FINAL_MODE_RE = re.compile(r"final_eval_mode:(?P<mode>\w+)(?: stride:(?P<stride>\d+) batch_seqs:(?P<batch>\d+))?")
STOP_STEP_RE = re.compile(r"step:(?P<step>\d+)/(?P<iters>\d+)")


def last_match(pattern: re.Pattern[str], lines: list[str]) -> re.Match[str] | None:
    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            return match
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse parameter-golf train logs into structured metrics.")
    parser.add_argument("log_path")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    lines = log_path.read_text(encoding="utf-8").splitlines()

    final_exact = last_match(FINAL_EXACT_RE, lines)
    total_bytes = last_match(TOTAL_BYTES_RE, lines)
    train_time = last_match(TRAIN_TIME_RE, lines)
    eval_time = last_match(EVAL_TIME_RE, lines)
    compressor = last_match(COMPRESSOR_RE, lines)
    final_mode = last_match(FINAL_MODE_RE, lines)
    stop_step = last_match(STOP_STEP_RE, lines)

    result = {
        "log_path": str(log_path),
        "val_loss_exact": float(final_exact.group("val_loss")) if final_exact else None,
        "val_bpb_exact": float(final_exact.group("val_bpb")) if final_exact else None,
        "total_submission_bytes": int(total_bytes.group("bytes")) if total_bytes else None,
        "train_time_ms": int(train_time.group("ms")) if train_time else None,
        "eval_time_ms": int(eval_time.group("ms")) if eval_time else None,
        "compression_label": compressor.group("label") if compressor else None,
        "compressor": (
            compressor.group("label").split("+", 1)[1]
            if compressor and "+" in compressor.group("label")
            else None
        ),
        "artifact_bytes": int(compressor.group("bytes")) if compressor else None,
        "final_eval_mode": final_mode.group("mode") if final_mode else None,
        "eval_stride": int(final_mode.group("stride")) if final_mode and final_mode.group("stride") else None,
        "eval_batch_seqs": int(final_mode.group("batch")) if final_mode and final_mode.group("batch") else None,
        "final_step": int(stop_step.group("step")) if stop_step else None,
        "planned_iterations": int(stop_step.group("iters")) if stop_step else None,
    }

    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"log_path={result['log_path']}")
        print(f"val_loss_exact={result['val_loss_exact']}")
        print(f"val_bpb_exact={result['val_bpb_exact']}")
        print(f"total_submission_bytes={result['total_submission_bytes']}")
        print(f"train_time_ms={result['train_time_ms']}")
        print(f"eval_time_ms={result['eval_time_ms']}")
        print(f"compression_label={result['compression_label']}")
        print(f"compressor={result['compressor']}")
        print(f"artifact_bytes={result['artifact_bytes']}")
        print(f"final_eval_mode={result['final_eval_mode']}")
        print(f"eval_stride={result['eval_stride']}")
        print(f"eval_batch_seqs={result['eval_batch_seqs']}")
        print(f"final_step={result['final_step']}")
        print(f"planned_iterations={result['planned_iterations']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
