# Parameter Golf — Quickstart

## Step 1: Get Compute

### Apply for free credits (do this FIRST)
- Compute grant: https://modelcraft.runpod.io/
- Participant form: https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf

### Spin up a pod (RunPod)

**For dev iteration (1 GPU, ~$1.50-2.00/hr):**
1. Go to https://console.runpod.io/deploy?template=y5cejece4j
2. Select: 1x NVIDIA A100 80GB SXM (or 1x H100 PCIe)
3. Deploy, wait ~2 min, SSH in

**For final submission (8 GPU, ~$21.50/hr):**
1. Same template: https://console.runpod.io/deploy?template=y5cejece4j
2. Select: 8x NVIDIA H100 80GB HBM3
3. Deploy, SSH in

## Step 2: Setup (on pod)

```bash
cd /workspace
git clone https://github.com/keshav55/parameter-golf.git
cd parameter-golf

# Download dataset (full — all 80 shards + validation)
python3 data/cached_challenge_fineweb.py --variant sp1024
```

## Step 3: Run

### Quick dev test (1 GPU, ~2 min)
```bash
bash atris/scripts/run_v1_dev.sh
```

### Architecture sweep (1 GPU, ~30 min total)
```bash
bash atris/scripts/run_v2_sweep.sh
```

### Full submission run (8 GPU, ~10 min)
```bash
bash atris/scripts/run_v1.sh
```

### Custom experiment
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=my_experiment \
NUM_LAYERS=10 \
MODEL_DIM=576 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Step 4: Submit

When you have a result that beats 1.2244 BPB:

```bash
# Update submission files with real metrics
cd records/track_10min_16mb/2026-03-19_AtrisLabs/
# Edit submission.json with actual val_bpb, bytes_total, etc.
# Copy your train_gpt.py and train.log here

# Push to fork
git add .
git commit -m "Atris v1: BPB submission"
git push fork main

# Open PR against openai/parameter-golf
gh pr create --repo openai/parameter-golf \
  --title "Atris Labs: [BPB SCORE] — [approach summary]" \
  --body "See records/track_10min_16mb/2026-03-19_AtrisLabs/README.md"
```

## Key Metrics to Watch

- `final_int8_zlib_roundtrip val_bpb:X.XXXX` — THIS is the official score
- `Total submission size int8+zlib: XXXXX bytes` — must be < 16,000,000
- `model_params:XXXXX` — total parameter count

## Current Targets

| Version | Config | Expected BPB | Status |
|---------|--------|-------------|--------|
| v1 | 10L, LR=0.02 | ~1.21-1.22 | Ready to run |
| v2 | Best from sweep | ~1.20-1.21 | Sweep first |
| v3 | Weight sharing + QAT + INT6 | ~1.18-1.20 | Code changes needed |
