#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_result(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def find_result_paths(args_paths: list[str]) -> list[Path]:
    results: list[Path] = []
    for raw in args_paths:
        path = Path(raw)
        if path.is_dir():
            candidate = path / "result.json"
            if candidate.exists():
                results.append(candidate)
        elif path.name == "result.json" and path.exists():
            results.append(path)
    return sorted(results)


def mode_value(result: dict[str, object], mode: str, key: str) -> str:
    eval_modes = result.get("eval_modes", {})
    if not isinstance(eval_modes, dict):
        return "-"
    metrics = eval_modes.get(mode)
    if not isinstance(metrics, dict):
        return "-"
    value = metrics.get(key)
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V1b result.json files")
    parser.add_argument("paths", nargs="+", help="Run directories or result.json files")
    args = parser.parse_args()

    paths = find_result_paths(args.paths)
    if not paths:
        raise SystemExit("no V1b result.json files found")

    header = (
        "run",
        "controller_bpb",
        "raw_bpb",
        "raw_delta",
        "refined_bpb",
        "refined_delta",
        "artifact_mb",
        "train_tok_s",
        "raw_mem_mb",
        "summary_mem_mb",
    )
    widths = [38, 14, 12, 11, 14, 14, 11, 11, 10, 12]
    print(" ".join(label.ljust(width) for label, width in zip(header, widths)))
    print(" ".join("-" * width for width in widths))
    for path in paths:
        result = load_result(path)
        run_name = path.parent.name
        artifact_mb = "-"
        artifact_bytes = result.get("model_artifact_bytes")
        if isinstance(artifact_bytes, (int, float)):
            artifact_mb = f"{artifact_bytes / 1e6:.3f}"
        train_tok_s = result.get("train_tokens_per_s")
        train_tok_s_text = f"{float(train_tok_s):.1f}" if isinstance(train_tok_s, (int, float)) else "-"
        refined_memory = result.get("eval_modes", {}).get("refined", {}).get("memory", {})
        raw_mem_mb = refined_memory.get("raw_memory_mb_estimate")
        summary_mem_mb = refined_memory.get("summary_memory_mb_estimate")
        raw_mem_text = f"{float(raw_mem_mb):.3f}" if isinstance(raw_mem_mb, (int, float)) else "-"
        summary_mem_text = f"{float(summary_mem_mb):.3f}" if isinstance(summary_mem_mb, (int, float)) else "-"
        row = (
            run_name[: widths[0] - 1],
            mode_value(result, "controller", "val_bpb"),
            mode_value(result, "raw", "val_bpb"),
            mode_value(result, "raw", "delta_vs_controller_bpb"),
            mode_value(result, "refined", "val_bpb"),
            mode_value(result, "refined", "delta_vs_controller_bpb"),
            artifact_mb,
            train_tok_s_text,
            raw_mem_text,
            summary_mem_text,
        )
        print(" ".join(str(value).ljust(width) for value, width in zip(row, widths)))


if __name__ == "__main__":
    main()
