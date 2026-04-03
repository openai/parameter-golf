#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


RECORD_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_JSON = RECORD_DIR / "train_result.json"
DEFAULT_OUTPUT_JSON = RECORD_DIR / "submission.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DeepFloor submission.json from a train result")
    parser.add_argument("--result-json", default=str(DEFAULT_RESULT_JSON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--name", default="DeepFloor")
    parser.add_argument("--author", default="Kenneth Malloy")
    parser.add_argument("--github-id", default="KenMalloy")
    parser.add_argument("--track", default="non_record_16mb")
    parser.add_argument(
        "--blurb",
        default=(
            "DeepFloor recurrent multi-view submission with adaptive floor or fused cross-token recurrence, "
            "high-precision runtime state, and stability-controlled low-bit artifact accounting."
        ),
    )
    parser.add_argument(
        "--code-files",
        nargs="*",
        default=[
            str(RECORD_DIR / "train_gpt.py"),
            str(RECORD_DIR / "deepfloor_snapshot.py"),
        ],
        help="Files whose bytes should count toward the honest code budget for this snapshot.",
    )
    return parser.parse_args()


def code_bytes(paths: list[str]) -> int:
    total = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (RECORD_DIR / path).resolve()
        total += path.stat().st_size
    return total


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_json)
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    artifact = payload.get("artifact", {})
    train = payload.get("train", {})
    val = payload.get("val", {})
    config = payload.get("config", {})
    bytes_model = int(artifact.get("estimated_bytes", 0))
    bytes_code = code_bytes(list(args.code_files))
    submission = {
        "author": args.author,
        "github_id": args.github_id,
        "name": args.name,
        "blurb": args.blurb,
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "track": args.track,
        "val_loss": float(val.get("loss")) if "loss" in val else None,
        "val_bpb": float(val.get("bpb")) if "bpb" in val else None,
        "bytes_total": int(bytes_model + bytes_code),
        "bytes_model_estimated": bytes_model,
        "bytes_code": bytes_code,
        "wallclock_seconds": float(train.get("elapsed_seconds", 0.0)),
        "device": config.get("device"),
    }
    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    main()
