# Non-Record Submission: Discriminative TTT + SLOT-24 (3-seed)

**SLOT-24 val_bpb: 0.7093 ± 0.0025** (mean over 3 seeds, 8xH100 SXM)

This is a non-record, near-frontier submission combining two existing open PRs:
- **Discriminative TTT** (PR #1351, resouer): per-block adaptive learning rate during pre-quantization test-time training
- **SLOT-24** (PR #1376, stukenov): per-sample eval-time delta optimization with a hidden-state delta plus logit bias, 24 AdamW steps

Neither technique is new. The contribution is (1) running both together, (2) verifying the combination replicates across 3 seeds with low variance, and (3) a finding about SLOT saturation.

This submission follows earlier work in PR #975 (compression-aware training thesis, negative result).

---

## Results

| Seed | Post-EMA BPB | Int6 Roundtrip | Sliding Window | **SLOT-24 BPB** | Sub size |
|------|-------------|----------------|----------------|-----------------|----------|
| 1337 | 1.1121 | 1.1141 | 1.0934 | **0.70865** | 15.85 MB |
| 42   | 1.1123 | 1.1143 | 1.0936 | **0.71199** | 15.84 MB |
| 2025 | 1.1105 | 1.1126 | 1.0918 | **0.70714** | 15.88 MB |
| **mean** | **1.1116** | **1.1137** | **1.0929** | **0.70926** | |
| **std** | 0.0009 | 0.0009 | 0.0010 | **0.0025** | |

All three runs used identical config, script, and hardware — only `SEED` varied.

**vs PR #1376 (stukenov, SLOT-24 with flat TTT):** 0.7093 vs 0.7094 — essentially tied. The discriminative TTT contribution (~0.002 BPB) is real but small.

---

## What Changed vs PR #1376

One addition: discriminative TTT replaces flat TTT.

| Phase | PR #1376 | This submission |
|-------|----------|-----------------|
| Pre-quant TTT | Flat AdamW, 6 epochs, freeze_blocks=2 | Discriminative: per-block LR scaled 0.3x→1.0x early→late layers, freeze_blocks=0, 10 epochs |
| SLOT eval | SLOT-24, stride=96 | Identical |
| Everything else | — | Identical |

Discriminative TTT (from PR #1351) assigns lower LR to early blocks (which are more general) and higher LR to late blocks (which are more task-specific). This is a standard insight from fine-tuning literature applied to TTT.

---

## Key Finding: SLOT Saturation

Discriminative TTT improves pre-SLOT BPB by **~0.009** (1.121 → 1.112 post-EMA). But after SLOT-24, the improvement narrows to **~0.002 BPB**. SLOT is so powerful that it compensates for base model quality differences, leaving diminishing returns for pre-SLOT improvements.

Implication: pushing below 0.70 BPB requires improving SLOT itself (more steps, tighter stride, better delta parameterization), not the base model. The ceiling is the SLOT eval budget, not the float model quality.

---

## Run Commands

All runs on 8xH100 SXM Secure Cloud, 600s wallclock, template `y5cejece4j`.

```bash
# Seed 1337
PYTHONUNBUFFERED=1 RUN_ID=run1_disc_ttt_slot24_seed1337 SEED=1337 \
DEEP_SLOT_ENABLED=0 SLOT_ENABLED=1 TTT_ENABLED=1 TTT_DISCRIMINATIVE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Seed 42
PYTHONUNBUFFERED=1 RUN_ID=run2_disc_ttt_slot24_seed42 SEED=42 \
DEEP_SLOT_ENABLED=0 SLOT_ENABLED=1 TTT_ENABLED=1 TTT_DISCRIMINATIVE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Seed 2025
PYTHONUNBUFFERED=1 RUN_ID=run3_disc_ttt_slot24_seed2025 SEED=2025 \
DEEP_SLOT_ENABLED=0 SLOT_ENABLED=1 TTT_ENABLED=1 TTT_DISCRIMINATIVE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Configuration

```
NUM_LAYERS=11  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4  MLP_MULT=3.0
VOCAB_SIZE=1024  TRAIN_BATCH_TOKENS=786432  TRAIN_SEQ_LEN=2048
MAX_WALLCLOCK_SECONDS=600  WARMUP_STEPS=20
MATRIX_LR=0.025  SCALAR_LR=0.025  EMBED_LR=0.035
ROPE_DIMS=16  XSA_LAST_N=11  LN_SCALE=1
BIGRAM_VOCAB_SIZE=1536  BIGRAM_DIM=128
TTT_LR=0.0005  TTT_EPOCHS=6  TTT_FREEZE_BLOCKS=2  (flat)
TTT_DISCRIMINATIVE=1  (overrides: 10 epochs, freeze_blocks=0, per-block LR 0.3x→1.0x)
SLOT_STEPS=24  SLOT_LR=0.024  SLOT_LR_MIN=0.001  SLOT_STRIDE=96
GPTQ_CALIB_BATCHES=32  (int6 + LZMA preset=9)
```

- Model params: 26,928,220
- Step time: ~89.3ms/step on 8xH100 SXM
- TTT time: ~189s (discriminative, 10 epochs)
- GPTQ time: ~5s
- SLOT-24 eval time: ~254s

---

## Included Files

- `train_gpt.py` — full training script; discriminative TTT activated by `TTT_DISCRIMINATIVE=1`
- `run1_seed1337.log` — seed 1337, SLOT-24 BPB 0.70865
- `run2_seed42.log` — seed 42, SLOT-24 BPB 0.71199
- `run3_seed2025.log` — seed 2025, SLOT-24 BPB 0.70714
- `submission.json` — leaderboard metadata
