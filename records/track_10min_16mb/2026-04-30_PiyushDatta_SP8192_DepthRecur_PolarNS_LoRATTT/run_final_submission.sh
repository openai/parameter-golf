#!/bin/bash
# Usage:
#   bash run_final_submission.sh                  # auto-detect GPUs, 3 seeds
#   bash run_final_submission.sh --nproc 4        # force 4 GPUs
#   bash run_final_submission.sh --seeds 42       # single seed test
#   bash run_final_submission.sh --seeds 42 314   # two seeds
#
# Ctrl-C stops cleanly. Completed seed logs are preserved.

DATA_DIR="${DATA_DIR:-./data/}"
NPROC=""
SEEDS=(42 314 999)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc) NPROC="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --seeds) shift; SEEDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$NPROC" ]; then
    echo "ERROR: --nproc is required"
    echo "Usage: $0 --nproc N [--seeds 42 314 999] [--data-dir DIR]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
cd "$REPO_ROOT"

echo "GPUs: $NPROC | Seeds: ${SEEDS[*]} | Data: $DATA_DIR"

# Check data
if [ ! -d "$DATA_DIR/datasets/fineweb10B_sp8192" ]; then
    echo "Downloading SP8192 dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
fi

mkdir -p "$SCRIPT_DIR/logs"
COMPLETED_SEEDS=()

cleanup() {
    echo ""
    echo "Interrupted. Completed seeds: ${COMPLETED_SEEDS[*]:-none}"
    [ ${#COMPLETED_SEEDS[@]} -gt 0 ] && parse_results
    exit 130
}
trap cleanup INT TERM

parse_results() {
    python3 << PYEOF
import json, re, os
seeds = [${SEEDS[*]}]
script_dir = "$SCRIPT_DIR"
results = {}
sliding_vals, ttt_vals = [], []
for seed in seeds:
    lp = os.path.join(script_dir, "logs", f"seed_{seed}.log")
    if not os.path.exists(lp): continue
    with open(lp) as f: text = f.read()
    r = {"seed": seed}
    for pat, key in [
        (r"stopping_early.*step:\s*(\d+)/", "training_steps"),
        (r"swa:applying SWA weights \((\d+) checkpoints\)", "swa_checkpoints"),
    ]:
        m = re.search(pat, text)
        if m: r[key] = int(m.group(1))
    for pat, key in [
        (r"pre-quantization post-ema val_loss:([\d.]+) val_bpb:([\d.]+)", "pre_quant_val_bpb"),
        (r"quantized_sliding_window val_loss:([\d.]+) val_bpb:([\d.]+)", "sliding_val_bpb"),
        (r"quantized_ttt_phased val_loss:([\d.]+) val_bpb:([\d.]+)", "ttt_val_bpb"),
    ]:
        m = re.search(pat, text)
        if m: r[key] = float(m.group(2))
    m = re.search(r"Total submission size quantized\+\w+:\s*(\d+)\s*bytes", text)
    if m: r["artifact_bytes"] = int(m.group(1))
    m = re.search(r"stopping_early.*train_time:\s*(\d+)ms", text)
    if m: r["train_time_s"] = int(m.group(1)) / 1000
    m = re.search(r"TOTAL_EVAL_TIME:\s*([\d.]+)s", text)
    if m: r["eval_time_s"] = float(m.group(1))
    if "sliding_val_bpb" in r: sliding_vals.append(r["sliding_val_bpb"])
    if "ttt_val_bpb" in r: ttt_vals.append(r["ttt_val_bpb"])
    results[str(seed)] = r

print(f"\n{'Seed':>6} | {'Sliding':>10} | {'TTT':>10} | {'Size':>12} | {'Steps':>6} | {'Train':>7} | {'Eval':>7}")
print("-" * 80)
for seed in seeds:
    r = results.get(str(seed), {})
    ts = f"{r['train_time_s']:.0f}s" if 'train_time_s' in r else 'N/A'
    es = f"{r['eval_time_s']:.0f}s" if 'eval_time_s' in r else 'N/A'
    print(f"{seed:>6} | {r.get('sliding_val_bpb', 'N/A'):>10} | {r.get('ttt_val_bpb', 'N/A'):>10} | {r.get('artifact_bytes', 'N/A'):>12} | {r.get('training_steps', 'N/A'):>6} | {ts:>7} | {es:>7}")

for label, vals in [("Sliding", sliding_vals), ("TTT", ttt_vals)]:
    if vals:
        m = sum(vals)/len(vals)
        s = (sum((x-m)**2 for x in vals)/len(vals))**.5
        print(f"\n{label} mean: {m:.6f} (std: {s:.6f})")

final_vals = ttt_vals if ttt_vals else sliding_vals
final_key = "ttt_val_bpb" if ttt_vals else "sliding_val_bpb"
sub_path = os.path.join(script_dir, "submission.json")
with open(sub_path) as f: sub = json.load(f)
if final_vals:
    sub["val_bpb"] = round(sum(final_vals)/len(final_vals), 5)
    sub["val_bpb_std"] = round((sum((x-sub["val_bpb"])**2 for x in final_vals)/len(final_vals))**.5, 6)
for seed in seeds:
    r = results.get(str(seed), {})
    sub["seed_results"][str(seed)] = {k: r.get(k) for k in ["sliding_val_bpb","ttt_val_bpb","artifact_bytes","training_steps","pre_quant_val_bpb"]}
with open(sub_path, "w") as f: json.dump(sub, f, indent=2)
print(f"\nUpdated {sub_path}")
if final_vals: print(f"val_bpb: {sub['val_bpb']}")
PYEOF
}

for SEED in "${SEEDS[@]}"; do
    LOG="$SCRIPT_DIR/logs/seed_${SEED}.log"
    echo ""
    echo "=== SEED=$SEED starting at $(date) ==="

    timeout 1200 bash -c "SEED=$SEED DATA_DIR=\"$DATA_DIR\" \
    python -m torch.distributed.run --standalone --nproc_per_node=$NPROC \
        \"$SCRIPT_DIR/train_gpt.py\" 2>&1" | tee "$LOG"
    RC=${PIPESTATUS[0]}

    if [ $RC -eq 124 ]; then
        echo "KILLED: seed $SEED exceeded 20 min total wall clock"
    elif [ $RC -eq 0 ]; then
        COMPLETED_SEEDS+=("$SEED")
    else
        echo "WARNING: seed $SEED failed (exit $RC)"
    fi

    grep -E "stopping_early|TOTAL_EVAL_TIME|quantized_sliding|quantized_ttt_phased|Total submission" "$LOG" 2>/dev/null | tail -5
    echo ""
done

echo "=== ALL DONE ==="
parse_results
echo ""
echo "git add records/track_10min_16mb/2026-04-30_PiyushDatta_SP8192_DepthRecur_PolarNS_LoRATTT/"
echo "git commit -m 'Final submission: SP8192 DepthRecur PolarNS LoRATTT'"
echo "git push"
