#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path

from parse_run_logs import parse_log_text


def load_summary(path):
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return parse_log_text(p.read_text(encoding="utf-8", errors="replace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="train_seed*.log or parser summary JSON files")
    ap.add_argument("--submission", default="submission.json")
    args = ap.parse_args()

    summaries = [load_summary(p) for p in args.inputs]
    by_seed = {}
    for s in summaries:
        seed = s.get("seed")
        val_bpb = s.get("val_bpb")
        if seed is None:
            raise SystemExit(f"missing seed in {s}")
        if val_bpb is None:
            raise SystemExit(f"missing val_bpb for seed {seed}")
        by_seed[str(seed)] = {
            "val_bpb": val_bpb,
            "artifact_bytes": s.get("artifact_bytes"),
            "train_seconds": s.get("train_seconds"),
            "eval_seconds": s.get("eval_seconds"),
            "sliding_bpb": s.get("sliding_bpb"),
            "ttt_bpb": s.get("ttt_bpb"),
        }

    vals = [by_seed[k]["val_bpb"] for k in sorted(by_seed, key=lambda x: int(x))]
    sub_path = Path(args.submission)
    sub = json.loads(sub_path.read_text(encoding="utf-8"))
    sub["val_bpb"] = sum(vals) / len(vals)
    sub["val_bpb_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    sub["seed_results"] = by_seed
    sub["seeds"] = [int(k) for k in sorted(by_seed, key=lambda x: int(x))]
    sub["compliance"]["train_under_600s"] = all((s.get("train_seconds") or 9999) <= 600.5 for s in summaries)
    sub["compliance"]["artifact_under_16mb"] = all((s.get("artifact_bytes") or 99999999) <= 16_000_000 for s in summaries)
    sub["compliance"]["eval_under_600s"] = all((s.get("eval_seconds") or 9999) <= 600.5 for s in summaries)
    sub["compliance"]["three_seeds"] = len(by_seed) >= 3
    sub_path.write_text(json.dumps(sub, indent=2) + "\n", encoding="utf-8")
    print(f"updated {sub_path} from {len(by_seed)} seed result(s)")


if __name__ == "__main__":
    main()
