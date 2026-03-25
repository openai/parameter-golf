#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path
from typing import Iterable

FIELDNAMES = [
    "run_id",
    "git_commit",
    "branch",
    "timestamp",
    "seed",
    "gpu_name",
    "data_variant",
    "train_shards",
    "train_time_ms",
    "step_avg_ms",
    "val_loss",
    "val_bpb",
    "compressed_model_bytes",
    "artifact_bytes",
    "log_path",
]

RUN_ID_PATH_RE = re.compile(r"^logs/([A-Za-z0-9_.\-]+)\.txt$")
SEED_RE = re.compile(r"\bseed:(\d+)\b")
TRAIN_LOADER_RE = re.compile(r"train_loader:dataset:([^\s]+)\s+train_shards:(\d+)")
STEP_RE = re.compile(r"step:\d+/\d+.*?train_time:(\d+)ms.*?step_avg:([0-9.]+)ms")
STOPPING_RE = re.compile(r"stopping_early:.*?train_time:(\d+)ms")
ROUNDTRIP_RE = re.compile(r"final_int8_zlib_roundtrip(?:_exact)? val_loss:([0-9.]+) val_bpb:([0-9.]+)")
VAL_STEP_RE = re.compile(r"step:\d+/\d+ val_loss:([0-9.]+) val_bpb:([0-9.]+)")
COMPRESSED_RE = re.compile(r"Serialized model int8\+zlib:\s*(\d+)\s+bytes")
ARTIFACT_RE = re.compile(r"Total submission size int8\+zlib:\s*(\d+)\s+bytes")
TIMESTAMP_BRACKET_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
HARNESS_META_RE = re.compile(r"HARNESS_META\s+([A-Za-z0-9_]+)=(.*)")


def _last_group(text: str, pattern: re.Pattern[str], idx: int = 1) -> str:
    matches = pattern.findall(text)
    if not matches:
        return ""
    last = matches[-1]
    if isinstance(last, tuple):
        return str(last[idx - 1])
    return str(last)


def _read_metadata_for_log(log_path: Path) -> dict[str, str]:
    meta_path = log_path.parent / "run_metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[str(k)] = "" if v is None else str(v)
    return out


def _read_harness_meta(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = HARNESS_META_RE.search(line.strip())
        if not m:
            continue
        out[m.group(1)] = m.group(2).strip()
    return out


def _normalize_variant(dataset_name: str) -> str:
    if dataset_name.startswith("fineweb10B_"):
        return dataset_name.removeprefix("fineweb10B_")
    return dataset_name


def _fallback_timestamp(log_path: Path, text: str) -> str:
    m = TIMESTAMP_BRACKET_RE.search(text)
    if m:
        try:
            parsed = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            return parsed.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    return dt.datetime.fromtimestamp(log_path.stat().st_mtime, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_log(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metadata = _read_metadata_for_log(log_path)
    harness_meta = _read_harness_meta(text)

    run_id = metadata.get("run_id") or harness_meta.get("run_id")
    if not run_id:
        for line in text.splitlines():
            m = RUN_ID_PATH_RE.match(line.strip())
            if m:
                run_id = m.group(1)
                break
    if not run_id:
        run_id = log_path.stem

    dataset_name = ""
    train_shards = metadata.get("train_shards") or harness_meta.get("train_shards", "")
    train_loader_matches = TRAIN_LOADER_RE.findall(text)
    if train_loader_matches:
        dataset_name, parsed_train_shards = train_loader_matches[-1]
        if not train_shards:
            train_shards = parsed_train_shards

    data_variant = metadata.get("data_variant") or harness_meta.get("data_variant", "")
    if not data_variant and dataset_name:
        data_variant = _normalize_variant(dataset_name)

    seed = metadata.get("seed") or harness_meta.get("seed", "")
    if not seed:
        seed = _last_group(text, SEED_RE)

    step_matches = STEP_RE.findall(text)
    train_time_ms = ""
    step_avg_ms = ""
    if step_matches:
        train_time_ms, step_avg_ms = step_matches[-1]
    stop_time = _last_group(text, STOPPING_RE)
    if stop_time:
        train_time_ms = stop_time

    roundtrip_matches = ROUNDTRIP_RE.findall(text)
    val_loss = ""
    val_bpb = ""
    if roundtrip_matches:
        val_loss, val_bpb = roundtrip_matches[-1]
    else:
        val_step_matches = VAL_STEP_RE.findall(text)
        if val_step_matches:
            val_loss, val_bpb = val_step_matches[-1]

    compressed_model_bytes = _last_group(text, COMPRESSED_RE)
    artifact_bytes = _last_group(text, ARTIFACT_RE)
    if not artifact_bytes:
        artifact_bytes = compressed_model_bytes

    timestamp = (
        metadata.get("timestamp_utc")
        or harness_meta.get("timestamp_utc")
        or _fallback_timestamp(log_path, text)
    )

    row = {
        "run_id": run_id,
        "git_commit": metadata.get("git_commit") or harness_meta.get("git_commit", ""),
        "branch": metadata.get("branch") or harness_meta.get("branch", ""),
        "timestamp": timestamp,
        "seed": seed,
        "gpu_name": metadata.get("gpu_name") or harness_meta.get("gpu_name", ""),
        "data_variant": data_variant,
        "train_shards": train_shards,
        "train_time_ms": train_time_ms,
        "step_avg_ms": step_avg_ms,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "compressed_model_bytes": compressed_model_bytes,
        "artifact_bytes": artifact_bytes,
        "log_path": str(log_path.resolve()),
    }
    return row


def gather_logs(explicit_logs: list[str], patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in explicit_logs:
        p = Path(raw).expanduser()
        if p.is_file():
            paths.append(p.resolve())
    for pattern in patterns:
        for match in glob.glob(pattern):
            p = Path(match)
            if p.is_file():
                paths.append(p.resolve())
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in sorted(paths):
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def existing_log_paths(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    out: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("log_path", "")
            if value:
                out.add(value)
    return out


def append_rows(csv_path: Path, rows: Iterable[dict[str, str]]) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    already = existing_log_paths(csv_path)
    rows_to_write = [row for row in rows if row["log_path"] not in already]
    if not rows_to_write:
        return 0
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows_to_write:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    return len(rows_to_write)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse train logs into structured rows.")
    parser.add_argument("--log", action="append", default=[], help="Path to a log file. Repeat for multiple logs.")
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern to find logs. Repeat for multiple patterns.",
    )
    parser.add_argument("--output", default="results/runs.csv", help="Output CSV path. Default: results/runs.csv")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output.")
    args = parser.parse_args()

    patterns = args.glob or ["logs/runs/*/console.log", "logs/*.txt"]
    logs = gather_logs(args.log, patterns)
    if not logs:
        if not args.quiet:
            print("No logs found. Nothing to parse.")
        return 0

    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for log_path in logs:
        try:
            rows.append(parse_log(log_path))
        except Exception as exc:
            failures.append(f"{log_path}: {exc}")

    written = append_rows(Path(args.output), rows)
    if not args.quiet:
        print(f"parsed_logs: {len(rows)}")
        print(f"rows_appended: {written}")
        print(f"output_csv: {Path(args.output).resolve()}")
    if failures:
        for msg in failures:
            print(f"WARN: {msg}", file=sys.stderr)
    return 1 if failures and not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
