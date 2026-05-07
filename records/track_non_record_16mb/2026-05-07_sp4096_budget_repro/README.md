# SP4096 Budget Reproduction Candidate

This is our first serious Parameter Golf candidate: a clean reproduction target based on Kevin Clark's accepted **4096-Vocab + Larger Model + High WD + Simplifications** run.

This folder is intentionally conservative. The `train_gpt.py` starts from:

`records/track_10min_16mb/2026-04-01_Vocab4096_MLPMult4_WD085/train_gpt.py`

and adds one reproducibility guard: if the quantized artifact is slightly over the decimal `16,000,000` byte cap, it coarsens selected quantized tensors until the artifact fits. The RunPod result used only `ve_shared.embed.weight`.

Why start here:

- Proven score: `val_bpb ~= 1.09785` on 8xH100.
- Proven artifact size: about `15.9 MB`, under the decimal `16,000,000` byte cap.
- Easier data path than the stronger CaseOps submissions.
- Good enough to be a real baseline before spending RunPod money on experiments.

This is a non-record reproduction/iteration folder, not a claim of a novel leaderboard result.

## Current Valid Result

Best RunPod 1xH100 budget run, seed 42, 86 train shards, `QK_GAIN_INIT=4.5`:

```text
steps: 4271
train_time_ms: 3590178
pre_quant_post_ema val_bpb: 1.11595986
valid_roundtrip val_bpb: 1.12578632
valid_sliding val_bpb: 1.10743376
valid_blob_bytes: 15,916,816
code_bytes: 70,379
total_bytes: 15,987,195
```

See `runpod_results/RESULTS.md` for the full run note.

## Data Setup

Run from the repository root:

```bash
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp4096 --train-shards 86
```

The verified budget run uses 86 SP4096 train shards, about `17 GB` materialized. Use a pod volume with enough room for temporary Hugging Face cache files, logs, and artifacts.

The script expects:

```text
data/datasets/fineweb10B_sp4096/fineweb_train_*.bin
data/datasets/fineweb10B_sp4096/fineweb_val_*.bin
data/tokenizers/fineweb_4096_bpe.model
```

## Smoke Run

Use this on a 1xH100/4090-class pod to verify dependencies, data, CUDA, FlashAttention, logging, serialization, quantization, and eval:

```bash
cd records/track_non_record_16mb/2026-05-07_sp4096_budget_repro
DATA_DIR=../../../data \
RUN_ID=sp4096_smoke \
SEED=42 \
ITERATIONS=8 \
MAX_WALLCLOCK_SECONDS=0 \
WARMUP_STEPS=1 \
TRAIN_LOG_EVERY=1 \
VAL_LOSS_EVERY=0 \
GPTQ_CALIBRATION_BATCHES=1 \
SLIDING_WINDOW_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Budget 1xH100 Run

This is not leaderboard-comparable because the official record budget is 10 minutes on 8xH100, but it is useful for a $20 RunPod budget:

```bash
cd records/track_non_record_16mb/2026-05-07_sp4096_budget_repro
DATA_DIR=../../../data \
RUN_ID=sp4096_1xh100_seed42_86shards_qk45 \
SEED=42 \
QK_GAIN_INIT=4.5 \
MAX_WALLCLOCK_SECONDS=3600 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

On 1xH100, this approximates the GPU-hours of the original 8xH100 10-minute budget, but overheads and scaling are not identical.

If storage is tight, the verified budget result used `86` SP4096 train shards:

```bash
rm -f data/manifest.json
HF_HOME=/workspace/hf-cache \
HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub \
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp4096 --train-shards 86
```

## Official-Style Reproduction

For a comparable reproduction, use 8xH100 SXM:

```bash
cd records/track_non_record_16mb/2026-05-07_sp4096_budget_repro
DATA_DIR=../../../data \
RUN_ID=sp4096_repro_seed42 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Expected accepted-record ballpark from the source run:

```text
pre-quantization post-ema val_bpb ~= 1.1041
final_int6_roundtrip val_bpb ~= 1.1159
final_int6_sliding_window val_bpb ~= 1.0974
Total submission size ~= 15.9 MB
```

## Next Experiment Knobs

Once the reproduction works, the first cheap sweeps I would try are:

- `QK_GAIN_INIT=5.0`
- `MUON_WD=0.09` and `EMBED_WD=0.09`
- `MATRIX_LR=0.018`
- `GPTQ_CALIBRATION_BATCHES=32` for faster iteration, then restore `64`

Treat one seed as signal-finding only. A real claim needs multiple seeds.
