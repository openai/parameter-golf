# Overnight Run Launch — Manual Playbook

Run from SSH into the RunPod box, in the `/workspace/parameter-golf` directory.
Time required: ~20 minutes (10 min smoke tests, 5 min launch, 5 min verify).

## Step 1: Pull the latest code

```bash
cd /workspace/parameter-golf
git fetch origin
git checkout sp8192-rebase
git pull origin sp8192-rebase
git log --oneline -10  # confirm you see: rebase + 3 lever commits
```

Expected commits (top 4):
```
Port per-layer QK-Gain init schedule to SP8192 base (QK_GAIN_INIT_SCHEDULE)
Port mixed-regime GPTQ calibration to SP8192 base (CALIB_SPLIT_BY_MODULE)
Port QUANT_ONLY_CHECKPOINT mode to SP8192 base
Gitignore logs dir, *.pt, *.ptz checkpoint files
```

## Step 2: Environment sanity checks

```bash
nvidia-smi                              # confirm H100, no other users
df -h /workspace                        # confirm >50GB free
python --version                        # should be Python 3.10+
pip show torch | grep Version           # should be 2.11.0+cu130 or compatible
pip show flash-attn 2>/dev/null | grep Version   # FA3 must be installed
pip show sentencepiece 2>/dev/null | grep Version  # required for SP8192
pip show brotli 2>/dev/null | grep Version         # required for compression
```

If flash-attn or brotli is missing: `pip install flash_attn_3-3.0.0 brotli sentencepiece` (check requirements.txt first).

## Step 3: Start tmux

```bash
tmux new -s pg_overnight
```

## Step 4: GPU smoke test — base SP8192 (100 iters, ~3 min)

Run the clean SP8192 base to confirm the stack is functional:

```bash
mkdir -p logs
MAX_WALLCLOCK_SECONDS=180 \
  DATA_DIR=/workspace/parameter-golf/data \
  python records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt_human.py \
  2>&1 | tee logs/sp8192_base_smoke_$(date +%Y%m%d_%H%M).log
```

Pass criteria:
- Loss decreasing for 50+ steps
- No CUDA OOM
- No import errors
- Exits cleanly (no NaN/Inf)

If this fails, DO NOT proceed. Debug first.

## Step 5: GPU smoke test — ported file with QK-Gain schedule (50 iters, ~3 min)

```bash
QK_GAIN_INIT_SCHEDULE="2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5" \
  MAX_WALLCLOCK_SECONDS=180 \
  DATA_DIR=/workspace/parameter-golf/data \
  python train_gpt_sp8192_opt.py \
  2>&1 | tee logs/sp8192_opt_qkgain_smoke_$(date +%Y%m%d_%H%M).log
```

Pass criteria:
- Loss decreasing (no explosion from aggressive gain values)
- No NaN/Inf in first 50 steps

If loss explodes, use the conservative fallback schedule instead:
```bash
QK_GAIN_INIT_SCHEDULE="1.5,1.7,2.0,2.2,2.5,2.5,2.3,2.0,1.8,1.6,1.5"
```

## Step 6: Verify QUANT_ONLY_CHECKPOINT mode (optional, ~15 min)

If you have a checkpoint from a prior run (`final_model.pt`):

```bash
QUANT_ONLY_CHECKPOINT=/workspace/parameter-golf/final_model.pt \
  DATA_DIR=/workspace/parameter-golf/data \
  python train_gpt_sp8192_opt.py \
  2>&1 | tee logs/sp8192_opt_quant_only_test_$(date +%Y%m%d_%H%M).log
```

Expected: skips training, prints `[QUANT_ONLY] Loading checkpoint`, runs GPTQ, prints BPB.

## Step 7: Launch overnight run (nohup, backgrounded)

```bash
QK_GAIN_INIT_SCHEDULE="2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5" \
  DATA_DIR=/workspace/parameter-golf/data \
  SEED=42 \
  nohup python train_gpt_sp8192_opt.py \
  > logs/sp8192_opt_overnight_$(date +%Y%m%d_%H%M).log 2>&1 &

echo $! > logs/sp8192_opt_overnight.pid
echo "PID: $(cat logs/sp8192_opt_overnight.pid)"
```

## Step 8: Verify it's running (watch for 2 minutes)

```bash
sleep 120
tail -n 30 logs/sp8192_opt_overnight_*.log | tail -30
ps -p $(cat logs/sp8192_opt_overnight.pid)
nvidia-smi  # GPU should be >80% utilized
```

## Step 9: Note pod details

```bash
mkdir -p logs
cat >> logs/OVERNIGHT_RUN_NOTES.md << EOF
Pod ID: ${RUNPOD_POD_ID:-unknown}
Hostname: $(hostname)
Started: $(date)
PID: $(cat logs/sp8192_opt_overnight.pid 2>/dev/null)
Log: $(ls logs/sp8192_opt_overnight_*.log 2>/dev/null | tail -1)
QK_GAIN_INIT_SCHEDULE: 2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5
EOF
cat logs/OVERNIGHT_RUN_NOTES.md
```

## Step 10: Detach tmux and sleep

Detach: press **Ctrl-B**, then **D**

The run will continue in the background. Budget: ~$24-30 for 8-10hr on 1×H100 at $3/hr.

---

## Fallback: QUANT_ONLY sweep on old checkpoint (if Step 5 failed)

Run a calibration sweep on the existing Mar 25 checkpoint (produces sweep data, not SP8192 data):

```bash
QUANT_ONLY_CHECKPOINT=/workspace/parameter-golf/final_model.pt \
  DATA_DIR=/workspace/parameter-golf/data \
  CALIB_SPLIT_BY_MODULE=1 \
  CALIB_ATTN_BATCHES=128 \
  CALIB_MLP_BATCHES=64 \
  nohup python train_gpt_sp8192_opt.py \
  > logs/quant_only_calib_sweep_$(date +%Y%m%d_%H%M).log 2>&1 &

echo $! > logs/quant_only_calib_sweep.pid
```

---

## 3-seed record run (after overnight produces a promising BPB)

If the overnight run shows BPB improvement vs SP8192 base (1.08563), run the 3-seed record attempt:

```bash
# Edit run_record_3seed.sh to point at train_gpt_sp8192_opt.py first
# Then:
SCHEDULE="2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5"
for SEED in 42 314 999; do
  QK_GAIN_INIT_SCHEDULE="${SCHEDULE}" \
    SEED=${SEED} \
    DATA_DIR=/workspace/parameter-golf/data \
    python train_gpt_sp8192_opt.py \
    2>&1 | tee logs/sp8192_opt_seed${SEED}_$(date +%Y%m%d_%H%M).log
done
```

Statistical bar: must beat 1.08563 by ≥0.005 BPB at p<0.01 via 3-seed Welch t-test to qualify as a record.
