#!/usr/bin/env python3
"""Print the merged article-ready table from matcha JSONLs + parameter-golf .txt logs.

Usage:
    python analyze.py

Reads from ../data/matcha/ and ../data/pg_logs/ (relative to this script).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"

CONFIGS = [
    ("lb_baseline",      "1. Baseline",        "2026-03-17_NaiveBaseline"),
    ("lb_slide64",       "2. SlidingWindow",   "2026-03-19_SlidingWindowEval"),
    ("lb_lora_ttt",      "3. LoRA TTT",        "2026-03-17_LoRA_TTT"),
    ("lb_11l_ema_gptq",  "4. 11L EMA+GPTQ",    "2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233"),
    ("lb_par_resid_dr",  "5. ParResid+MiniDR", "2026-03-31_ParallelResiduals_MiniDepthRecurrence"),
]

# train_batch_tokens × steps for each run (parsed from logs, hardcoded here)
TOKENS_PER_STEP = {
    "lb_baseline":     524288,
    "lb_slide64":      524288,
    "lb_lora_ttt":     524288,
    "lb_11l_ema_gptq": 786432,
    "lb_par_resid_dr": 786432,
}

SCORE_PATTERNS = [
    re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"),
    re.compile(r"final_int8_zlib_roundtrip\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"),
    re.compile(r"final_int8_ttt_lora\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"),
    re.compile(r"final_int6_sliding_window_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"),
    re.compile(r"quantized_ttt_phased\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"),
]
STEP_RE = re.compile(r"step:\s*(\d+)/\d+")
PARAMS_RE = re.compile(r"^model_params:(\d+)")


def parse_log(txt_path):
    val_bpb, last_step, params = None, 0, None
    if not txt_path.exists():
        return val_bpb, last_step, params
    for line in txt_path.read_text().splitlines():
        for pat in SCORE_PATTERNS:
            m = pat.search(line)
            if m:
                val_bpb = float(m.group(2))
        m = STEP_RE.search(line)
        if m:
            last_step = max(last_step, int(m.group(1)))
        m = PARAMS_RE.search(line)
        if m and params is None:
            params = int(m.group(1))
    return val_bpb, last_step, params


def main():
    rows = []
    for run_id, label, _sub in CONFIGS:
        jsonl = DATA_DIR / "matcha" / f"{run_id}.jsonl"
        txt = DATA_DIR / "pg_logs" / f"{run_id}.txt"

        records = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
        se = next(r for r in records if r.get("type") == "session_end")
        steps_jsonl = [r for r in records if r.get("type") == "step"]
        train_step_energy_wh = sum(r["energy_j"] for r in steps_jsonl) / 3600

        val_bpb, last_step, params = parse_log(txt)
        tokens = last_step * TOKENS_PER_STEP[run_id] if last_step else None
        per_gpu = sorted(se["gpus"], key=lambda g: g["idx"])
        energies = [g["energy_j"] for g in per_gpu]
        median = sorted(energies)[len(energies) // 2]
        spread_pct = (max(energies) - min(energies)) / median * 100

        rows.append({
            "label": label, "run_id": run_id,
            "params": params, "steps": last_step,
            "tokens_b": tokens / 1e9 if tokens else None,
            "val_bpb": val_bpb,
            "energy_kwh": se["energy_wh"] / 1000,
            "duration_s": se["duration_s"],
            "avg_kw": se["avg_power_w"] / 1000,
            "peak_kw": se["peak_power_w"] / 1000,
            "wh_per_mtok": se["energy_wh"] / (tokens / 1e6) if tokens else None,
            "spread_pct": spread_pct,
            "train_phase_wh": train_step_energy_wh,
            "post_train_wh": se["energy_wh"] - train_step_energy_wh,
            "post_train_pct": (se["energy_wh"] - train_step_energy_wh) / se["energy_wh"] * 100,
        })

    base = rows[0]
    for r in rows[1:]:
        r["delta_bpb"] = r["val_bpb"] - base["val_bpb"]
        r["delta_wh"] = r["energy_kwh"] * 1000 - base["energy_kwh"] * 1000

    # Print headline table
    print()
    h = f"{'config':<22} {'params':>8} {'steps':>6} {'val_bpb':>8} {'Δbpb':>9} {'kWh':>6} {'avg kW':>7} {'dur s':>6} {'Wh/Mtok':>9} {'GPU spr':>8}"
    print(h); print("-" * len(h))
    for r in rows:
        db = r.get("delta_bpb")
        db_str = f"{db:+.4f}" if db is not None else "anchor"
        params_str = f"{r['params']/1e6:.1f}M" if r["params"] else "?"
        wh_per_mtok = f"{r['wh_per_mtok']:.4f}" if r["wh_per_mtok"] else "?"
        print(f"{r['label']:<22} {params_str:>8} {r['steps']:>6,d} {r['val_bpb']:>8.4f} {db_str:>9} "
              f"{r['energy_kwh']:>6.3f} {r['avg_kw']:>7.2f} {r['duration_s']:>6.0f} {wh_per_mtok:>9} {r['spread_pct']:>7.2f}%")

    # Train vs post-train split
    print(f"\n{'config':<22} {'train Wh':>9} {'post-train Wh':>14} {'post-train %':>14}")
    print("-" * 66)
    for r in rows:
        print(f"{r['label']:<22} {r['train_phase_wh']:>9.0f} {r['post_train_wh']:>14.0f} {r['post_train_pct']:>13.0f}%")

    # Energy efficiency vs baseline
    print(f"\nEnergy efficiency (Wh per 0.001 BPB-drop vs baseline):")
    for r in rows[1:]:
        if r.get("delta_bpb") and abs(r["delta_bpb"]) > 1e-6:
            ratio = -r["delta_wh"] / r["delta_bpb"]
            sign_word = "saved per 0.001 BPB drop" if ratio < 0 else "spent per 0.001 BPB drop"
            print(f"  {r['label']:<22} Δbpb {r['delta_bpb']:+.4f}  Δenergy {r['delta_wh']:+7.1f} Wh  →  {abs(ratio):>5.0f} Wh {sign_word}")


if __name__ == "__main__":
    main()
