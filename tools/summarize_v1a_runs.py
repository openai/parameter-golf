#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize V1a run results from result.json files")
    parser.add_argument("paths", nargs="+", help="One or more result.json files or run directories")
    return parser.parse_args()


def resolve_result_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            candidate = path / "result.json"
            if candidate.exists():
                paths.append(candidate)
        elif path.name == "result.json" and path.exists():
            paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    result_paths = resolve_result_paths(args.paths)
    rows: list[dict[str, str]] = []
    for path in result_paths:
        with path.open() as handle:
            payload = json.load(handle)
        cfg = payload.get("config", {})
        eval_metrics = payload.get("eval", {})
        size_est = payload.get("size_estimate", {})
        rows.append(
            {
                "run": str(path.parent.name),
                "val_bpb": f"{eval_metrics.get('val_bpb', float('nan')):.6f}",
                "loss": f"{eval_metrics.get('loss', float('nan')):.6f}",
                "tok/s": f"{eval_metrics.get('tokens_per_s', float('nan')):.1f}",
                "artifact_mb": f"{(payload.get('model_artifact_bytes') or 0.0) / 1_000_000.0:.3f}",
                "compact_est_mb": f"{size_est.get('compact_model_mb_estimate', float('nan')):.3f}",
                "expanded_sem_mb": f"{size_est.get('expanded_semantic_mb_estimate', float('nan')):.3f}",
                "semantic_layers": str(cfg.get("semantic_layers", "")),
                "use_semantic": str(cfg.get("use_semantic_memory", "")),
            }
        )
    if not rows:
        raise SystemExit("no result.json files found")

    headers = list(rows[0].keys())
    widths = {key: max(len(key), max(len(row[key]) for row in rows)) for key in headers}
    print(" ".join(key.ljust(widths[key]) for key in headers))
    print(" ".join("-" * widths[key] for key in headers))
    for row in rows:
        print(" ".join(row[key].ljust(widths[key]) for key in headers))


if __name__ == "__main__":
    main()
