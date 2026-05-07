#!/usr/bin/env python3
import json
import math
import re
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_PREFIX = "shortchunk16_t2048_local0875_seed"


PATTERNS = {
    "train_stop": re.compile(r"stopping_early: wallclock_cap train_time: ([0-9.]+)ms step: ([0-9]+)/([0-9]+)"),
    "prequant": re.compile(r"diagnostic pre-quantization post-ema val_loss:([0-9.]+) val_bpb:([0-9.]+) eval_time:([0-9.]+)ms"),
    "quantized": re.compile(r"diagnostic quantized val_loss:([0-9.]+) val_bpb:([0-9.]+) eval_time:([0-9.]+)ms"),
    "ttt": re.compile(r"quantized_ttt_phased val_loss:([0-9.]+) val_bpb:([0-9.]+) eval_time:([0-9.]+)ms"),
    "total_eval": re.compile(r"total_eval_time:([0-9.]+)s"),
    "artifact": re.compile(r"Serialized model quantized\+pergroup: ([0-9]+) bytes"),
    "total_size": re.compile(r"Total submission size quantized\+pergroup: ([0-9]+) bytes"),
    "source_uncompressed": re.compile(r"Code size \(uncompressed\): ([0-9]+) bytes"),
    "source_compressed": re.compile(r"Code size \(compressed\): ([0-9]+) bytes"),
}


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def last_match(text, name):
    matches = list(PATTERNS[name].finditer(text))
    return matches[-1] if matches else None


def parse_seed(seed):
    run_dir = ROOT / f"{RUN_PREFIX}{seed}"
    train_text = read_text(run_dir / "train.log")
    eval_text = read_text(run_dir / "eval.log")
    out = {"seed": int(seed), "run_dir": str(run_dir.relative_to(ROOT))}

    m = last_match(train_text, "train_stop")
    if m:
        out["train_time_ms"] = int(float(m.group(1)))
        out["steps"] = int(m.group(2))
        out["iterations"] = int(m.group(3))

    for key in ("prequant", "quantized"):
        m = last_match(train_text, key)
        if m:
            out[f"{key}_val_loss"] = float(m.group(1))
            out[f"{key}_val_bpb"] = float(m.group(2))
            out[f"{key}_eval_time_ms"] = int(float(m.group(3)))

    for key in ("artifact", "total_size", "source_uncompressed", "source_compressed"):
        m = last_match(train_text, key)
        if m:
            out[key] = int(m.group(1))

    m = last_match(eval_text, "ttt")
    if m:
        out["ttt_val_loss"] = float(m.group(1))
        out["ttt_val_bpb"] = float(m.group(2))
        out["ttt_eval_time_ms"] = int(float(m.group(3)))
    m = last_match(eval_text, "total_eval")
    if m:
        out["total_eval_time_s"] = float(m.group(1))

    return out


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main(argv):
    seeds = argv or ["42", "0", "1234"]
    rows = [parse_seed(seed) for seed in seeds]
    bpbs = [row["ttt_val_bpb"] for row in rows if "ttt_val_bpb" in row]
    losses = [row["ttt_val_loss"] for row in rows if "ttt_val_loss" in row]
    mean_bpb, std_bpb = mean_std(bpbs)
    mean_loss, std_loss = mean_std(losses)

    print("# shortchunk16_t2048_local0875 multiseed summary")
    print()
    print("| seed | steps | train_time_s | quant_bpb | ttt_bpb | ttt_eval_s | total_size | artifact |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {seed} | {steps} | {train_time_s} | {quant_bpb} | {ttt_bpb} | {ttt_eval_s} | {total_size} | {artifact} |".format(
                seed=row["seed"],
                steps=row.get("steps", ""),
                train_time_s=f"{row['train_time_ms'] / 1000:.1f}" if "train_time_ms" in row else "",
                quant_bpb=f"{row['quantized_val_bpb']:.8f}" if "quantized_val_bpb" in row else "",
                ttt_bpb=f"{row['ttt_val_bpb']:.8f}" if "ttt_val_bpb" in row else "",
                ttt_eval_s=f"{row['total_eval_time_s']:.1f}" if "total_eval_time_s" in row else "",
                total_size=row.get("total_size", ""),
                artifact=row.get("artifact", ""),
            )
        )

    print()
    if mean_bpb is not None:
        print(f"mean_ttt_val_bpb: {mean_bpb:.8f}")
        print(f"std_ttt_val_bpb: {std_bpb:.8f}")
    if mean_loss is not None:
        print(f"mean_ttt_val_loss: {mean_loss:.8f}")
        print(f"std_ttt_val_loss: {std_loss:.8f}")

    max_eval = max((row.get("total_eval_time_s", math.nan) for row in rows), default=math.nan)
    max_train = max((row.get("train_time_ms", math.nan) / 1000 for row in rows), default=math.nan)
    max_size = max((row.get("total_size", 0) for row in rows), default=0)
    print(f"max_train_time_s: {max_train:.1f}" if not math.isnan(max_train) else "max_train_time_s:")
    print(f"max_eval_time_s: {max_eval:.1f}" if not math.isnan(max_eval) else "max_eval_time_s:")
    print(f"max_total_submission_size: {max_size}")

    draft = {
        "name": "SP8192 LQER SparseGate BOSFix shortchunk16 TTT",
        "track": "10min_16mb",
        "seeds": [row["seed"] for row in rows],
        "val_loss": mean_loss,
        "val_bpb": mean_bpb,
        "val_loss_std": std_loss,
        "val_bpb_std": std_bpb,
        "seed_results": rows,
        "artifact_bytes_max": max((row.get("artifact", 0) for row in rows), default=0),
        "bytes_total": max_size,
        "eval_time_s_max": None if math.isnan(max_eval) else max_eval,
        "train_time_s_max": None if math.isnan(max_train) else max_train,
        "technique_summary": "Old compliant training snapshot plus eval-only short-doc TTT chunk-size 16 for docs shorter than 2048, local TTT LR multiplier 0.875, eval_seq_len=2560.",
    }
    (ROOT / "submission_draft.json").write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
