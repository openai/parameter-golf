#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


LOSS_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iters>\d+)\s+loss:(?P<loss>[0-9.]+).*?avg:(?P<avg_ms>[0-9.]+)ms"
)
GATE_RE = re.compile(r"eng_gate step=(?P<step>\d+)\s+mean=(?P<mean>[-+0-9.eE]+)\s+std=(?P<std>[-+0-9.eE]+)")
BATCH_RE = re.compile(r"train_batch_tokens=(?P<tokens>\d+)")
NAN_RE = re.compile(r"\bnan\b", re.IGNORECASE)


def parse_last_train_line(log_path: Path) -> tuple[float | None, float | None]:
    last_loss = None
    last_avg_ms = None
    try:
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = LOSS_RE.search(line)
            if m:
                last_loss = float(m.group("loss"))
                last_avg_ms = float(m.group("avg_ms"))
    except FileNotFoundError:
        return (None, None)
    return (last_loss, last_avg_ms)


def parse_probe_csv(log_dir: Path, run_id: str) -> dict[str, float | None]:
    probe_path = log_dir / run_id / "probe_summary.csv"
    if not probe_path.exists():
        probe_path = Path("logs") / f"skc_matrix_{run_id}" / "probe_summary.csv"
    if not probe_path.exists():
        return {
            "skc_zero_delta": None,
            "eng_zero_delta": None,
            "amp_skc_amp_res": None,
        }
    with probe_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "skc_zero_delta": None,
            "eng_zero_delta": None,
            "amp_skc_amp_res": None,
        }
    last = rows[-1]
    amp_vals = []
    for (k, v) in last.items():
        if k.startswith("amp_skc_L"):
            try:
                amp_vals.append(float(v))
            except (TypeError, ValueError):
                pass
    amp_mean = (sum(amp_vals) / len(amp_vals)) if amp_vals else None
    out = {
        "skc_zero_delta": _to_float(last.get("skc_zero_delta")),
        "eng_zero_delta": _to_float(last.get("eng_zero_delta")),
        "amp_skc_amp_res": amp_mean,
    }
    return out


def parse_gate_stats(log_path: Path) -> dict[str, float | None]:
    last_mean = None
    last_std = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GATE_RE.search(line)
            if m:
                last_mean = _to_float(m.group("mean"))
                last_std = _to_float(m.group("std"))
    return {"gate_mean": last_mean, "gate_std": last_std}


def parse_batch_tokens(log_path: Path) -> int | None:
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = BATCH_RE.search(line)
            if m:
                try:
                    return int(m.group("tokens"))
                except ValueError:
                    return None
    return None


def has_nan(log_path: Path) -> bool:
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if NAN_RE.search(line):
                return True
    return False


def _to_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", required=True, type=Path)
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--run-id", default="", type=str)
    p.add_argument("--assert-gate-trained", action="store_true")
    p.add_argument("--assert-no-nan", action="store_true")
    p.add_argument("--assert-batch-tokens", type=int, default=0)
    args = p.parse_args()

    rows: list[dict[str, object]] = []
    txt_files = sorted(args.log_dir.glob("*.txt"))
    if args.run_id:
        txt_files = [txt for txt in txt_files if txt.stem == args.run_id]
    for txt in txt_files:
        run_id = txt.stem
        (loss_final, step_time_avg_ms) = parse_last_train_line(txt)
        probe = parse_probe_csv(args.log_dir, run_id)
        gate = parse_gate_stats(txt)
        batch_tokens = parse_batch_tokens(txt)
        nan_found = has_nan(txt)
        rows.append(
            {
                "run_id": run_id,
                "loss_final": loss_final,
                "step_time_avg_ms": step_time_avg_ms,
                "skc_zero_delta": probe["skc_zero_delta"],
                "eng_zero_delta": probe["eng_zero_delta"],
                "amp_skc/amp_res": probe["amp_skc_amp_res"],
                "eng_gate_mean": gate["gate_mean"],
                "eng_gate_std": gate["gate_std"],
                "train_batch_tokens": batch_tokens,
                "nan_found": nan_found,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "loss_final",
                "step_time_avg_ms",
                "skc_zero_delta",
                "eng_zero_delta",
                "amp_skc/amp_res",
                "eng_gate_mean",
                "eng_gate_std",
                "train_batch_tokens",
                "nan_found",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"WROTE_CSV={args.out_csv}")
    print(f"WROTE_JSON={args.out_json}")
    print(f"RUNS={len(rows)}")
    failed = False
    for row in rows:
        run_id = str(row["run_id"])
        gate_mean = _to_float(str(row["eng_gate_mean"])) if row["eng_gate_mean"] is not None else None
        gate_std = _to_float(str(row["eng_gate_std"])) if row["eng_gate_std"] is not None else None
        batch_tokens = row["train_batch_tokens"]
        nan_found = bool(row["nan_found"])
        if args.assert_no_nan and nan_found:
            print(f"ASSERT_FAIL run={run_id} reason=nan_found")
            failed = True
        if args.assert_batch_tokens > 0 and batch_tokens != args.assert_batch_tokens:
            print(f"ASSERT_FAIL run={run_id} reason=batch_tokens expected={args.assert_batch_tokens} actual={batch_tokens}")
            failed = True
        if args.assert_gate_trained:
            if gate_mean is None or gate_std is None:
                print(f"ASSERT_FAIL run={run_id} reason=missing_gate_stats")
                failed = True
            elif abs(gate_mean - 0.5) < 0.02 and gate_std < 0.01:
                print(f"ASSERT_FAIL run={run_id} reason=gate_untrained mean={gate_mean:.6f} std={gate_std:.6f}")
                failed = True
    if failed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
