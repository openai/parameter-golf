from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_SUBMISSION_DIR = Path(
    "records/track_10min_16mb/2026-04-07_Shallow_Blue_Probe_BOS"
)
SIZE_LIMIT_BYTES = 16_000_000


def newest_log_path(folder: Path) -> Path | None:
    latest_log = folder / "latest_run.log"
    if latest_log.is_file():
        return latest_log
    logs_dir = folder / "logs"
    if not logs_dir.is_dir():
        return None
    candidates = sorted(logs_dir.glob("*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_optional(pattern: str, text: str, cast):
    match = re.search(pattern, text)
    if not match:
        return None
    return cast(match.group(1))


def build_summary(folder: Path) -> dict[str, object]:
    log_path = newest_log_path(folder)
    log_text = log_path.read_text(encoding="utf-8") if log_path else ""
    artifact_path = folder / "final_model.int8.ptz"

    artifact_bytes = artifact_path.stat().st_size if artifact_path.is_file() else None
    code_bytes = parse_optional(r"Submission code size: (\d+) bytes", log_text, int)
    probe_bytes = parse_optional(r"Shallow Blue probe artifact size: (\d+) bytes", log_text, int)
    submission_total_bytes = parse_optional(
        r"Total submission size int8\+zlib: (\d+) bytes",
        log_text,
        int,
    )
    roundtrip_val_loss = parse_optional(
        r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+)",
        log_text,
        float,
    )
    roundtrip_val_bpb = parse_optional(
        r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)",
        log_text,
        float,
    )
    shallow_blue_val_bpb = parse_optional(
        r"final_shallow_blue_probe_exact val_bpb:([0-9.]+)",
        log_text,
        float,
    )
    shallow_blue_delta_bpb = parse_optional(
        r"final_shallow_blue_probe_exact val_bpb:[0-9.]+ delta_bpb:([+-]?[0-9.]+)",
        log_text,
        float,
    )
    shallow_blue_elapsed_seconds = parse_optional(
        r"final_shallow_blue_probe_exact val_bpb:[0-9.]+ delta_bpb:[+-]?[0-9.]+ elapsed_seconds:([0-9.]+)",
        log_text,
        float,
    )
    wallclock_ms = parse_optional(
        r"stopping_early: wallclock_cap train_time:(\d+)ms",
        log_text,
        int,
    )
    step_stop = parse_optional(
        r"stopping_early: wallclock_cap train_time:\d+ms step:(\d+)/",
        log_text,
        int,
    )

    return {
        "folder": str(folder),
        "log": str(log_path) if log_path else None,
        "artifact_path": str(artifact_path),
        "artifact_bytes": artifact_bytes,
        "artifact_under_16mb": artifact_bytes is not None and artifact_bytes < SIZE_LIMIT_BYTES,
        "submission_code_bytes": code_bytes,
        "probe_artifact_bytes": probe_bytes,
        "submission_total_bytes_from_log": submission_total_bytes,
        "submission_under_16mb": submission_total_bytes is not None
        and submission_total_bytes < SIZE_LIMIT_BYTES,
        "roundtrip_val_loss": roundtrip_val_loss,
        "roundtrip_val_bpb": roundtrip_val_bpb,
        "shallow_blue_val_bpb": shallow_blue_val_bpb,
        "shallow_blue_delta_bpb": shallow_blue_delta_bpb,
        "shallow_blue_elapsed_seconds": shallow_blue_elapsed_seconds,
        "wallclock_stop_ms": wallclock_ms,
        "step_stop": step_stop,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the Shallow Blue record-track submission run.")
    parser.add_argument("folder", nargs="?", default=str(DEFAULT_SUBMISSION_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = build_summary(Path(args.folder))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print("Shallow Blue record submission summary")
    for key in [
        "folder",
        "log",
        "artifact_path",
        "artifact_bytes",
        "artifact_under_16mb",
        "submission_code_bytes",
        "probe_artifact_bytes",
        "submission_total_bytes_from_log",
        "submission_under_16mb",
        "roundtrip_val_loss",
        "roundtrip_val_bpb",
        "shallow_blue_val_bpb",
        "shallow_blue_delta_bpb",
        "shallow_blue_elapsed_seconds",
        "wallclock_stop_ms",
        "step_stop",
    ]:
        print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()
