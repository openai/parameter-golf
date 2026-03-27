#!/usr/bin/env bash
# Overnight ablation: 5 configs × 500 steps
# Usage: bash run_ablation.sh  (run in tmux or nohup)
set -euo pipefail
cd ~/parameter-golf
source ~/pg_env/bin/activate

RESULTS=~/pg_overnight_results.txt

echo "=== Ablation suite started $(date) ===" | tee -a "$RESULTS"
echo "" >> "$RESULTS"

# ─── helper ──────────────────────────────────────────────────────────────────
# run_one <label> [KEY=VAL ...]
# Common env: 500 iters, 8192 batch, no intermediate val, 100-step warmdown.
# MAX_WALLCLOCK_SECONDS=0 disables the wallclock cap so all 500 steps run.
run_one() {
    local name="$1"; shift
    local run_id="ablation_$(echo "$name" | tr '[:upper:]' '[:lower:]')"
    echo "=== $name started at $(date) ===" | tee -a "$RESULTS"

    env \
        RUN_ID="$run_id" \
        ITERATIONS=500 \
        TRAIN_BATCH_TOKENS=8192 \
        VAL_LOSS_EVERY=0 \
        WARMUP_STEPS=5 \
        VAL_BATCH_SIZE=524288 \
        MAX_WALLCLOCK_SECONDS=0 \
        WARMDOWN_ITERS=100 \
        "$@" \
        python3 train_gpt_mlx_kl.py

    local logfile="logs/${run_id}.txt"
    echo "--- $name results ---" >> "$RESULTS"
    grep "final_int8_zlib_roundtrip_exact" "$logfile" | tail -1 >> "$RESULTS"
    grep "serialized_int8_zlib:" "$logfile" | tail -1 >> "$RESULTS"
    grep "model_params:" "$logfile" | head -1 >> "$RESULTS"
    echo "=== $name done at $(date) ===" | tee -a "$RESULTS"
    echo "" >> "$RESULTS"
}

# ─── runs ─────────────────────────────────────────────────────────────────────
run_one RunA
run_one RunB  BIGRAM_HASH_SIZE=0
run_one RunC  QAT_START_FRAC=1.0
run_one RunD  USE_ORTHO_INIT=0
run_one RunE  NUM_LAYERS=9 MLP_MULT=2 BIGRAM_HASH_SIZE=0 USE_ORTHO_INIT=0 QAT_START_FRAC=1.0 EMA_START_FRAC=1.0

echo "=== All ablations done at $(date) ===" | tee -a "$RESULTS"

# ─── generate summary ─────────────────────────────────────────────────────────
python3 - "$RESULTS" <<'PYEOF'
import re, sys, pathlib

results_path = pathlib.Path(sys.argv[1])
text = results_path.read_text()

runs = ["RunA", "RunB", "RunC", "RunD", "RunE"]
labels = {
    "RunA": "Full KL stack (baseline for diff)",
    "RunB": "No BigramHash",
    "RunC": "No QAT",
    "RunD": "No OrthoInit",
    "RunE": "Baseline config (9L 2xMLP)",
}

data = {}
for run in runs:
    pat_bpb   = rf"--- {run} results ---.*?val_bpb:(\S+)"
    pat_bytes = rf"--- {run} results ---.*?serialized_int8_zlib:(\d+)"
    m_bpb   = re.search(pat_bpb,   text, re.DOTALL)
    m_bytes = re.search(pat_bytes, text, re.DOTALL)
    data[run] = {
        "val_bpb":       float(m_bpb.group(1))   if m_bpb   else None,
        "artifact_bytes": int(m_bytes.group(1))  if m_bytes else None,
    }

ref_bpb = data["RunA"]["val_bpb"]

lines = []
lines.append("# Parameter Golf — Overnight Ablation Summary\n")
lines.append(f"500 iterations · 8192 batch tokens · WARMDOWN_ITERS=100\n")
lines.append("")
lines.append("## Results table\n")
lines.append("| Run | Config | val_bpb | Δ vs RunA | Artifact bytes |")
lines.append("|-----|--------|---------|-----------|----------------|")
for run in runs:
    d = data[run]
    bpb   = f"{d['val_bpb']:.6f}"   if d["val_bpb"]       else "N/A"
    delta = ""
    if ref_bpb and d["val_bpb"]:
        diff = d["val_bpb"] - ref_bpb
        delta = f"+{diff:.6f}" if diff >= 0 else f"{diff:.6f}"
    abytes = f"{d['artifact_bytes']:,}" if d["artifact_bytes"] else "N/A"
    lines.append(f"| {run} | {labels[run]} | {bpb} | {delta} | {abytes} |")

lines.append("")
lines.append("## Innovation analysis\n")

ablations = {
    "BigramHash":  ("RunB", "Removing BigramHash"),
    "QAT (int6)":  ("RunC", "Removing QAT"),
    "OrthoInit":   ("RunD", "Removing OrthoInit"),
}

impacts = {}
for name, (run, desc) in ablations.items():
    if ref_bpb and data[run]["val_bpb"]:
        impact = data[run]["val_bpb"] - ref_bpb  # positive = hurts (higher bpb is worse)
        impacts[name] = impact
        lines.append(f"- **{name}**: {desc} → Δval_bpb = {impact:+.6f}  "
                     f"({'helps' if impact > 0 else 'neutral/hurts'})")

if impacts:
    best  = max(impacts, key=lambda k: impacts[k])
    worst = min(impacts, key=lambda k: impacts[k])
    lines.append("")
    lines.append(f"**Most impactful innovation**: {best} (Δ={impacts[best]:+.6f} — removing it hurts most)")
    lines.append(f"**Least impactful innovation**: {worst} (Δ={impacts[worst]:+.6f})")

lines.append("")
lines.append("## Recommended H100 config\n")
if impacts:
    useful = [k for k, v in impacts.items() if v > 5e-5]
    neutral = [k for k, v in impacts.items() if abs(v) <= 5e-5]
    lines.append("Based on 500-step ablations on M1 (indicative only):\n")
    lines.append(f"- **Keep (proved helpful)**: {', '.join(useful) if useful else 'all innovations showed benefit'}")
    if neutral:
        lines.append(f"- **Borderline (retest at full 20k steps)**: {', '.join(neutral)}")
lines.append("- Architecture: NUM_LAYERS=11, MLP_MULT=3 (always on — not ablated here)")
lines.append("- For first H100 run use **Run A config** (full KL stack) as the reference,")
lines.append("  then drop the least-impactful innovation if artifact is too large.")

lines.append("")
lines.append("## train_gpt_kl.py\n")
import subprocess, os
kl_path = pathlib.Path("~/parameter-golf/train_gpt_kl.py").expanduser()
if kl_path.exists():
    result = subprocess.run(["python3", "-m", "py_compile", str(kl_path)], capture_output=True)
    if result.returncode == 0:
        size = kl_path.stat().st_size
        lines.append(f"✓ Created and syntactically valid ({size:,} bytes)")
    else:
        lines.append(f"✗ Syntax error: {result.stderr.decode()}")
else:
    lines.append("✗ File not found — creation may have failed")

summary = pathlib.Path("~/pg_overnight_summary.md").expanduser()
summary.write_text("\n".join(lines) + "\n")
print(f"Summary written to {summary}")
PYEOF

echo "Done. See ~/pg_overnight_results.txt and ~/pg_overnight_summary.md"
