#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FINAL_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-+0-9.eE]+) val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+) train_time:(?P<train_time_ms>[-+0-9.eE]+)ms "
    r"step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms"
)
PLAIN_STEP_RE = re.compile(
    r"(?P<step>\d+)/(?P<iterations>\d+) val_loss:\s*(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:\s*(?P<val_bpb>[-+0-9.eE]+)"
)
TIMED_EVAL_RE = re.compile(
    r"(?P<label>pre-quantization post-ema|quantized|quantized_sliding_window) "
    r"val_loss:(?P<val_loss>[-+0-9.eE]+) val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
    r"(?: eval_time:(?P<eval_time_ms>[-+0-9.eE]+)ms)?"
)
TRAIN_PROGRESS_RE = re.compile(
    r"(?P<step>\d+)/(?P<iterations>\d+) train_loss:\s*(?P<train_loss>[-+0-9.eE]+) "
    r"train_time:\s*(?P<train_time_min>[-+0-9.eE]+)m tok/s:\s*(?P<tok_per_sec>[-+0-9.eE]+)"
)
SIZE_RE = re.compile(r"Total submission size (?:int8\+zlib|quantized\+\w+): (?P<size>\d+) bytes")
STOP_RE = re.compile(
    r"stopping_early: wallclock_cap train_time:\s*(?P<train_time_ms>[-+0-9.eE]+)ms "
    r"step:\s*(?P<step>\d+)/(?P<iterations>\d+)"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def maybe_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def parse_train_log(path: Path) -> tuple[dict[str, Any], list[str]]:
    metrics: dict[str, Any] = {}
    issues: list[str] = []
    final_match: dict[str, Any] | None = None
    last_step_match: dict[str, Any] | None = None
    plain_step_match: dict[str, Any] | None = None
    timed_evals: dict[str, dict[str, Any]] = {}
    train_progress_match: dict[str, Any] | None = None
    stop_match: dict[str, Any] | None = None
    submission_size_bytes: int | None = None

    if not path.exists():
        return metrics, ["train.log missing"]

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        final = FINAL_RE.search(line)
        if final:
            final_match = {
                "val_loss": float(final.group("val_loss")),
                "val_bpb": float(final.group("val_bpb")),
            }
        step = STEP_RE.search(line)
        if step:
            last_step_match = {
                "step": int(step.group("step")),
                "iterations": int(step.group("iterations")),
                "val_loss": float(step.group("val_loss")),
                "val_bpb": float(step.group("val_bpb")),
                "train_time_ms": float(step.group("train_time_ms")),
                "step_avg_ms": float(step.group("step_avg_ms")),
            }
        plain_step = PLAIN_STEP_RE.search(line)
        if plain_step:
            plain_step_match = {
                "step": int(plain_step.group("step")),
                "iterations": int(plain_step.group("iterations")),
                "val_loss": float(plain_step.group("val_loss")),
                "val_bpb": float(plain_step.group("val_bpb")),
            }
        timed_eval = TIMED_EVAL_RE.search(line)
        if timed_eval:
            item: dict[str, Any] = {
                "val_loss": float(timed_eval.group("val_loss")),
                "val_bpb": float(timed_eval.group("val_bpb")),
            }
            eval_time_ms = timed_eval.group("eval_time_ms")
            if eval_time_ms is not None:
                item["eval_time_ms"] = float(eval_time_ms)
            timed_evals[timed_eval.group("label")] = item
        train_progress = TRAIN_PROGRESS_RE.search(line)
        if train_progress:
            train_progress_match = {
                "step": int(train_progress.group("step")),
                "iterations": int(train_progress.group("iterations")),
                "train_loss": float(train_progress.group("train_loss")),
                "train_time_ms": 60000.0 * float(train_progress.group("train_time_min")),
                "tok_per_sec": float(train_progress.group("tok_per_sec")),
            }
        stop = STOP_RE.search(line)
        if stop:
            stop_match = {
                "step": int(stop.group("step")),
                "iterations": int(stop.group("iterations")),
                "train_time_ms": float(stop.group("train_time_ms")),
            }
        size = SIZE_RE.search(line)
        if size:
            submission_size_bytes = int(size.group("size"))

    if last_step_match:
        metrics["pre_quant_bpb"] = last_step_match["val_bpb"]
        metrics["pre_quant_val_loss"] = last_step_match["val_loss"]
        metrics["step"] = last_step_match["step"]
        metrics["iterations"] = last_step_match["iterations"]
        metrics["train_time_ms"] = last_step_match["train_time_ms"]
        metrics["step_avg_ms"] = last_step_match["step_avg_ms"]
    elif plain_step_match:
        metrics["pre_quant_bpb"] = plain_step_match["val_bpb"]
        metrics["pre_quant_val_loss"] = plain_step_match["val_loss"]
        metrics["step"] = plain_step_match["step"]
        metrics["iterations"] = plain_step_match["iterations"]
    else:
        issues.append("no validation step found in train.log")

    pre_quant_timed = timed_evals.get("pre-quantization post-ema")
    quantized = timed_evals.get("quantized")
    sliding = timed_evals.get("quantized_sliding_window")
    if pre_quant_timed:
        metrics["pre_quant_bpb"] = pre_quant_timed["val_bpb"]
        metrics["pre_quant_val_loss"] = pre_quant_timed["val_loss"]
        if "eval_time_ms" in pre_quant_timed:
            metrics["pre_quant_eval_time_ms"] = pre_quant_timed["eval_time_ms"]
    if quantized:
        metrics["post_quant_bpb"] = quantized["val_bpb"]
        metrics["post_quant_val_loss"] = quantized["val_loss"]
        if "eval_time_ms" in quantized:
            metrics["post_quant_eval_time_ms"] = quantized["eval_time_ms"]
    if sliding:
        metrics["sliding_bpb"] = sliding["val_bpb"]
        metrics["sliding_val_loss"] = sliding["val_loss"]
        if "eval_time_ms" in sliding:
            metrics["sliding_eval_time_ms"] = sliding["eval_time_ms"]

    if sliding:
        metrics["score_bpb"] = sliding["val_bpb"]
        metrics["post_quant_bpb"] = sliding["val_bpb"]
        metrics["post_quant_val_loss"] = sliding["val_loss"]
    elif final_match:
        metrics["score_bpb"] = final_match["val_bpb"]
        metrics["post_quant_bpb"] = final_match["val_bpb"]
        metrics["post_quant_val_loss"] = final_match["val_loss"]
    elif quantized:
        metrics["score_bpb"] = quantized["val_bpb"]
        issues.append("no quantized_sliding_window line found; using quantized val_bpb")
    elif last_step_match:
        metrics["score_bpb"] = last_step_match["val_bpb"]
        issues.append("no final_int8_zlib_roundtrip_exact line found; using last pre-quant val_bpb")
    elif plain_step_match:
        metrics["score_bpb"] = plain_step_match["val_bpb"]
        issues.append("no post-quant metric found; using last validation val_bpb")
    else:
        issues.append("no metric could be extracted from train.log")

    if submission_size_bytes is not None:
        metrics["submission_size_bytes"] = submission_size_bytes
    if train_progress_match:
        metrics.setdefault("step", train_progress_match["step"])
        metrics.setdefault("iterations", train_progress_match["iterations"])
        metrics.setdefault("train_time_ms", train_progress_match["train_time_ms"])
        metrics["train_loss"] = train_progress_match["train_loss"]
        metrics["tok_per_sec"] = train_progress_match["tok_per_sec"]
    if stop_match and "train_time_ms" not in metrics:
        metrics["train_time_ms"] = stop_match["train_time_ms"]
        metrics["step"] = stop_match["step"]
        metrics["iterations"] = stop_match["iterations"]

    return metrics, issues


def extract_generation(generation_dir: Path, output_path: Path) -> None:
    bundle_dir = generation_dir / "bundle"
    bundle_manifest = load_json(bundle_dir / "bundle_manifest.json")
    slots_payload: dict[str, Any] = {}

    for slot in bundle_manifest["slots"]:
        results_dir = bundle_dir / slot["results_path"]
        item: dict[str, Any] = {
            "role": slot["role"],
            "family": slot["family"],
            "command": slot["command"],
            "results_dir": str(results_dir.relative_to(bundle_dir)),
        }
        for name in ("runner_status", "executor_status", "selection", "summary", "failure"):
            payload = maybe_load(results_dir / f"{name}.json")
            if payload is not None:
                item[name] = payload

        train_log = results_dir / "train.log"
        if train_log.exists():
            item["train_log"] = str(train_log.relative_to(bundle_dir))
            metrics, issues = parse_train_log(train_log)
            if metrics:
                item["metrics"] = metrics
            if issues:
                item.setdefault("notes", issues)

        runner_status = item.get("runner_status")
        if isinstance(runner_status, dict) and int(runner_status.get("returncode", 0)) != 0:
            item.setdefault(
                "failure",
                {
                    "reason": "runner returned non-zero exit code",
                    "returncode": int(runner_status["returncode"]),
                },
            )

        slots_payload[slot["slot"]] = item

    write_json(
        output_path,
        {
            "cycle_id": bundle_manifest["cycle_id"],
            "generation": bundle_manifest["generation"],
            "slots": slots_payload,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pgolf metrics from harness train logs.")
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_generation(Path(args.generation_dir).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()
