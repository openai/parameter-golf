#!/usr/bin/env python3
"""One-shot CSV schema migration: 14-col -> 24-col results.csv.

Preserves every existing row and backfills known metadata from TSV sweep files.
Safe to run multiple times (idempotent: detects new header and exits).
"""
import csv
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "logs" / "sweep" / "results.csv"
SWEEPS_DIR = REPO / "scripts" / "sweeps"
SCRIPT_NAME = "train_gpt_sota_decoded.py"

NEW_HEADER = [
    "label", "status", "exit_code", "wall_s",
    "train_loss", "pre_quant_bpb", "quant_bpb", "sliding_bpb", "ttt_bpb", "delta_bpb",
    "tok_s", "peak_mem_gb", "failure_class", "timestamp",
    "hostname", "script", "script_sha8", "seed", "iterations",
    "sliding_enabled", "ttt_enabled", "fast_smoke", "overrides", "notes",
]


def load_overrides_map() -> dict[str, str]:
    """Walk scripts/sweeps/*.tsv and build label -> overrides string."""
    m: dict[str, str] = {}
    for tsv in sorted(SWEEPS_DIR.glob("*.tsv")):
        for raw in tsv.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            label, overrides = parts[0].strip(), parts[1].strip()
            m[label] = overrides
    # p1_baseline had no overrides
    m.setdefault("p1_baseline", "")
    return m


def script_sha8() -> str:
    p = REPO / SCRIPT_NAME
    if not p.exists():
        return ""
    r = subprocess.run(["sha256sum", str(p)], capture_output=True, text=True)
    if r.returncode != 0:
        return ""
    return r.stdout.split()[0][:8]


def seed_for(label: str, overrides: str) -> str:
    # Explicit SEED= in overrides wins
    for tok in overrides.split():
        if tok.startswith("SEED="):
            return tok.split("=", 1)[1]
    # s2_var_seed42 / s2_var_seed314 naming convention
    if "seed42" in label:
        return "42"
    if "seed314" in label:
        return "314"
    # Default used by run_experiment.sh
    return "1337"


def main() -> int:
    if not CSV_PATH.exists():
        print(f"no csv at {CSV_PATH}", file=sys.stderr)
        return 1

    rows = list(csv.reader(CSV_PATH.open()))
    if not rows:
        print("empty csv", file=sys.stderr)
        return 1

    header = rows[0]
    overrides_map = load_overrides_map()
    host = socket.gethostname() or "unknown"
    sha = script_sha8()

    # Mode A: full header migration (14 -> 24 cols).
    if header != NEW_HEADER:
        if len(header) != 14:
            print(f"unexpected header length {len(header)}; aborting", file=sys.stderr)
            return 2
        out_rows = [NEW_HEADER]
        for row in rows[1:]:
            if len(row) < 14:
                row = row + [""] * (14 - len(row))
            label = row[0]
            overrides = overrides_map.get(label, "")
            seed = seed_for(label, overrides)
            out_rows.append(row[:14] + [
                host, SCRIPT_NAME, sha, seed, "150",
                "0", "0", "1", overrides, "",
            ])
        tmp = CSV_PATH.with_suffix(".csv.migrated")
        with tmp.open("w", newline="") as f:
            csv.writer(f).writerows(out_rows)
        tmp.replace(CSV_PATH)
        print(f"migrated header + {len(rows)-1} rows -> {CSV_PATH}")
        return 0

    # Mode B: header already v2, but some rows may be short (in-flight runs
    # that wrote with an older in-memory harness). Pad them with backfilled
    # metadata to reach 24 cols. Idempotent.
    fixed = 0
    out_rows = [header]
    for row in rows[1:]:
        if len(row) == 24:
            out_rows.append(row)
            continue
        if len(row) < 14:
            row = row + [""] * (14 - len(row))
        label = row[0]
        overrides = overrides_map.get(label, "")
        seed = seed_for(label, overrides)
        padded = row[:14] + [
            host, SCRIPT_NAME, sha, seed, "150",
            "0", "0", "1", overrides, "",
        ]
        out_rows.append(padded)
        fixed += 1
    if fixed == 0:
        print("already migrated; all rows clean")
        return 0
    tmp = CSV_PATH.with_suffix(".csv.migrated")
    with tmp.open("w", newline="") as f:
        csv.writer(f).writerows(out_rows)
    tmp.replace(CSV_PATH)
    print(f"padded {fixed} short row(s) -> {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
