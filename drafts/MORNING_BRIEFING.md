# Deployment Plan — READY TO EXECUTE

## SOTA: PR #315 at 1.1248 bpb

## TTT is DEAD — PR #303 and #317 confirm it hurts with XSA+EMA. Don't use it.

## Our code: PR #315 base + OLB (our invention). 1621 lines, syntax OK.

## On the pod

```bash
cd /workspace
git clone https://github.com/bopmite/parameter-golf.git && cd parameter-golf
git checkout bopmite/qat-ttt-valueembed
python3 data/cached_challenge_fineweb.py --variant sp1024
cp records/track_10min_16mb/2026-03-20_QAT_TTT_ValueEmbed/train_gpt.py .
```

### Run 1: Baseline (no OLB)
```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
OLB_LR=0 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Run 2: With OLB
Same but `OLB_LR=0.1 OLB_MOMENTUM=0.9`

### Runs 3-5: Best config with SEED=42, SEED=2025

## Decision gates
- Run 1 > 1.130 = broken, debug
- Run 2 < Run 1 = OLB works, run seeds
- Run 2 >= Run 1 = try OLB_LR=0.05, or submit baseline
