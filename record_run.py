"""Parse a train_gpt.py run log into one row of the Sprint 002 ablation record.

Schema (matches hero.md Section 4 `output_per_run`):
    {
      "row":            "B0" | "A1" | ... | "C2",
      "run_id":         str,                # from RUN_ID env at training time
      "seed":           int,
      "config":         {KEY: VAL, ...},    # env-var toggles ACTIVE for this run
      "config_hash":    str,                # sha1 of sorted-tuple form of `config`
      "bpb":            float,              # final int8/raw-zlib-roundtrip val_bpb
      "val_loss":       float,              # paired val_loss for the same line
      "train_s":        float,              # last train_time:XXXms in the log, /1000
      "eval_s":         float,              # eval_time:XXXms from the roundtrip line, /1000
      "artifact_bytes": int,                # Total submission size (code + zlib payload)
      "world_size":     int,                # parsed from world_size:N grad_accum_steps:M
      "stopped_early":  bool,               # True if 'stopping_early: wallclock_cap' present
      "git_commit":     str,                # `git rev-parse HEAD` at record-time
      "log_path":       str,                # source log filename
      "recorded_at":    str,                # ISO8601 UTC at record-time
    }

Append-only writes to `results/runs.jsonl`. Aggregation is downstream's problem
(grep, jq, pandas — pick your poison).

Usage:
    python record_run.py logs/b0_seed1.txt --row B0 --seed 1337
    python record_run.py logs/a1_seed1.txt --row A1 --seed 1337 --config "QUANTIZE_WEIGHTS=none"
    python record_run.py logs/c1_seed3.txt --row C1 --seed 31337 \
        --config "QUANTIZE_WEIGHTS=none NUM_KV_HEADS=8"

If --seed is omitted the script tries to extract it from the run_id (e.g. "b0_seed1337").
If --config is omitted it defaults to {} (B0 baseline).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPO / "results" / "runs.jsonl"

# All toggles in the Sprint 002 ablation matrix. Anything else passed via --config
# is rejected so we don't silently miss a typo'd env-var name.
KNOWN_TOGGLES = {
    "QUANTIZE_WEIGHTS",
    "QUANT_SCHEME",
    "SDPA_BACKEND",
    "OPTIMIZER",
    "NUM_KV_HEADS",
    "TIE_EMBEDDINGS",
}


def parse_config_string(s: str) -> dict[str, str]:
    """Parse 'KEY=VAL KEY=VAL ...' or comma-separated. Empty -> {}."""
    out: dict[str, str] = {}
    if not s.strip():
        return out
    tokens = re.split(r"[\s,]+", s.strip())
    for tok in tokens:
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"--config token {tok!r} missing '=' (expected KEY=VAL)")
        k, v = tok.split("=", 1)
        if k not in KNOWN_TOGGLES:
            raise ValueError(
                f"--config key {k!r} not in known toggles {sorted(KNOWN_TOGGLES)}"
            )
        out[k] = v
    return out


def config_hash(config: dict[str, str]) -> str:
    """sha1 over sorted (key, value) tuples. Empty config -> hash of '[]'."""
    canonical = json.dumps(sorted(config.items()), separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def first(pat: str, text: str, group: int = 1) -> Optional[str]:
    m = re.search(pat, text)
    return m.group(group) if m else None


def last(pat: str, text: str, group: int = 1) -> Optional[str]:
    matches = list(re.finditer(pat, text))
    return matches[-1].group(group) if matches else None


def parse_log(log_text: str) -> dict[str, object]:
    """Extract bpb / train_s / eval_s / artifact_bytes / world_size / stopped_early."""
    out: dict[str, object] = {}

    # Final roundtrip line — emitted regardless of QUANTIZE_WEIGHTS branch.
    # train_gpt.py logs: final_int8_zlib_roundtrip ... or final_raw_zlib_roundtrip ...
    rt = re.search(
        r"final_(int8|raw)_zlib_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+) eval_time:(\d+)ms",
        log_text,
    )
    if not rt:
        raise RuntimeError(
            "log missing 'final_*_zlib_roundtrip val_loss:.. val_bpb:.. eval_time:..ms' line; "
            "did training complete?"
        )
    out["val_loss"] = float(rt.group(2))
    out["bpb"] = float(rt.group(3))
    out["eval_s"] = float(rt.group(4)) / 1000.0

    # Total submission size — last occurrence (covers both fp32 and zlib lines).
    size_line = last(
        r"Total submission size (?:int8\+zlib|raw\+zlib): (\d+) bytes", log_text
    )
    if size_line is None:
        # Fallback: pre-quant 'Total submission size: X bytes' (older paths).
        size_line = last(r"Total submission size: (\d+) bytes", log_text)
    if size_line is None:
        raise RuntimeError("log missing 'Total submission size ...: N bytes' line")
    out["artifact_bytes"] = int(size_line)

    # Last train_time:XXXms in the log → total training wallclock.
    train_ms = last(r"train_time:(\d+)ms", log_text)
    if train_ms is None:
        raise RuntimeError("log missing 'train_time:..ms' lines")
    out["train_s"] = float(train_ms) / 1000.0

    ws = first(r"world_size:(\d+) grad_accum_steps:\d+", log_text)
    out["world_size"] = int(ws) if ws else None

    out["stopped_early"] = "stopping_early: wallclock_cap" in log_text
    return out


def extract_run_id(log_path: Path, log_text: str) -> str:
    """Run id is the log filename stem by convention (`logs/{RUN_ID}.txt`)."""
    return log_path.stem


def extract_seed(run_id: str, log_text: str, seed_arg: Optional[int]) -> int:
    if seed_arg is not None:
        return seed_arg
    # Try the run_id pattern first, e.g. "b0_seed1337" or "a1_seed3".
    m = re.search(r"seed[_-]?(\d+)", run_id, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    raise RuntimeError(
        f"could not infer --seed from run_id={run_id!r}; pass --seed explicitly"
    )


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({type(e).__name__})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="path to logs/{RUN_ID}.txt")
    ap.add_argument("--row", required=True, choices=["B0", "A1", "A2", "A3", "A4", "A5", "A6", "C1", "C2"])
    ap.add_argument("--seed", type=int, default=None, help="seed used for the run; inferred from run_id if omitted")
    ap.add_argument("--config", default="", help='active env-var toggles, e.g. "QUANTIZE_WEIGHTS=none NUM_KV_HEADS=8"')
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"jsonl path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--dry-run", action="store_true", help="print the row instead of appending")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"FATAL: log file not found: {args.log}", file=sys.stderr)
        return 2

    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log(log_text)
    config = parse_config_string(args.config)
    run_id = extract_run_id(args.log, log_text)
    seed = extract_seed(run_id, log_text, args.seed)

    row = {
        "row": args.row,
        "run_id": run_id,
        "seed": seed,
        "config": config,
        "config_hash": config_hash(config),
        "bpb": parsed["bpb"],
        "val_loss": parsed["val_loss"],
        "train_s": parsed["train_s"],
        "eval_s": parsed["eval_s"],
        "artifact_bytes": parsed["artifact_bytes"],
        "world_size": parsed["world_size"],
        "stopped_early": parsed["stopped_early"],
        "git_commit": git_head(),
        "log_path": str(args.log),
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    line = json.dumps(row, separators=(",", ":"), sort_keys=True)
    if args.dry_run:
        print(line)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"appended to {args.output}: row={args.row} seed={seed} bpb={row['bpb']:.4f} "
          f"size={row['artifact_bytes']:,} train_s={row['train_s']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
