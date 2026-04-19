#!/usr/bin/env python3
"""Parse a spec-006 train.log into tidy CSVs.

Expected log lines (from current train_gpt_sota.py plus the grad-norm diff on exp/dense-ckpts-grad-logging):
  "{step}/{iterations} train_loss: {loss:.4f} train_time: {min:.1f}m tok/s: {tok:.0f}"
  "{step} grad_norms: total={total:.4f} per_layer=[{n0:.4f},{n1:.4f},...]"  (new, exact format TBD)
  "val_loss: {val:.4f} step {step}"   (format-check against actual log)
  "warmdown_start_step{step}"
  "layer_loop:enabled step:{step}"

Regexes below are tentative; validate + tweak when real spec 006 log lands.
"""
import argparse, csv, re, sys
from pathlib import Path

RE_TRAIN = re.compile(r"^(\d+)/\d+\s+train_loss:\s+([\d.]+)\s+train_time:\s+([\d.]+)m\s+tok/s:\s+(\d+)")
RE_VAL   = re.compile(r"val_loss[: ]+([\d.]+).*?step\s+(\d+)", re.IGNORECASE)
# Grad-norm format is dependent on diff shape; parse both "total=" and "per_layer=[...]" variants.
RE_GRAD_TOTAL = re.compile(r"(?:^|\s)(\d+)\s+grad_norms?:\s+total=([\d.eE+-]+)")
RE_GRAD_LAYERS = re.compile(r"per_layer=\[([^\]]+)\]")
RE_WARMDOWN = re.compile(r"warmdown_start_step(\d+)")
RE_RECUR    = re.compile(r"layer_loop:enabled step:(\d+)")

def parse(path: Path):
    train_rows, val_rows, grad_rows, events = [], [], [], {}
    with path.open() as f:
        for line in f:
            if m := RE_TRAIN.match(line):
                step, loss, tmin, toks = int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))
                train_rows.append({"step": step, "train_loss": loss, "train_time_min": tmin, "tok_per_s": toks})
            if m := RE_VAL.search(line):
                val_rows.append({"step": int(m.group(2)), "val_loss": float(m.group(1))})
            if m := RE_GRAD_TOTAL.search(line):
                step, total = int(m.group(1)), float(m.group(2))
                per_layer = []
                if m2 := RE_GRAD_LAYERS.search(line):
                    per_layer = [float(x) for x in m2.group(1).split(",")]
                grad_rows.append({"step": step, "grad_total": total, "per_layer": per_layer})
            if m := RE_WARMDOWN.search(line):
                events["warmdown_start"] = int(m.group(1))
            if m := RE_RECUR.search(line):
                events["recurrence_activate"] = int(m.group(1))
    return train_rows, val_rows, grad_rows, events

def write_csv(rows, cols, path: Path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    train, val, grad, events = parse(args.log)

    write_csv(train, ["step", "train_loss", "train_time_min", "tok_per_s"], args.outdir / "train_loss.csv")
    write_csv(val,   ["step", "val_loss"],                                 args.outdir / "val_loss.csv")

    # Grad norms: flatten per_layer into columns if present.
    max_layers = max((len(r["per_layer"]) for r in grad), default=0)
    cols = ["step", "grad_total"] + [f"layer_{i}" for i in range(max_layers)]
    flat = []
    for r in grad:
        row = {"step": r["step"], "grad_total": r["grad_total"]}
        for i, v in enumerate(r["per_layer"]):
            row[f"layer_{i}"] = v
        flat.append(row)
    write_csv(flat, cols, args.outdir / "grad_norms.csv")

    with (args.outdir / "events.txt").open("w") as f:
        for k, v in events.items():
            f.write(f"{k}: {v}\n")

    print(f"parsed {len(train)} train rows, {len(val)} val rows, {len(grad)} grad-norm rows")
    print(f"events: {events}")

if __name__ == "__main__":
    main()
