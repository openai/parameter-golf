# SP8192 + Pre-quant TTT + Parallel Residuals + QK5 + EMA

**Run:** 011  
**Track:** 10min_16mb  
**Author:** Joshua Martinez  
**Date:** 2026-04-09  
**Status:** QUEUED  

## Hypothesis

Porting our pre-quant TTT technique (1.07389 BPB on SP1024) to SP8192 tokenizer will:
1. Isolate the tokenizer effect (SP8192 dominates leaderboard with 4/5 top submissions)
2. Match or beat SOTA 1.0810 BPB
3. Prove our pre-quant TTT generalizes across tokenizers

**Expected:** 1.070-1.078 BPB

## Techniques

Same as PR #1489, but with SP8192 tokenizer:

1. **SP8192 Tokenizer** — Dominant on leaderboard (4/5 top submissions)
2. **Pre-quant AdamW TTT** — 6 epochs, lr=0.0005, freeze first 2 blocks
3. **Parallel Residuals (L7+)** — GPT-J style
4. **QK-Gain 5.0** — Higher than PR #1019's 1.5
5. **EMA 0.9965** — Weight averaging before quantization
6. **GPTQ int6 + brotli** — Standard compression stack
7. **Sliding Window Eval** — Stride 64
8. **ETLB** — 5-step logit bias optimization

## Configuration

```
VOCAB_SIZE=8192
NUM_LAYERS=11
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=4.0
QK_GAIN_INIT=5.0
PREQUANT_TTT_ENABLED=1
PREQUANT_TTT_LR=0.0005
PREQUANT_TTT_EPOCHS=6
PREQUANT_TTT_FREEZE_BLOCKS=2
EMA_DECAY=0.9965
GPTQ_ENABLED=1
SLIDING_WINDOW_ENABLED=1
ETLB_ENABLED=1
TRAIN_SEQ_LEN=2048
MAX_WALLCLOCK_SECONDS=588
SEEDS=42,314,999
```

## Results

**RUNNING** — Check logs/run011.log for progress

| Seed | val_bpb | Status |
|------|---------|--------|
| 42 | TBD | Running |
| 314 | TBD | Running |
| 999 | TBD | Running |
| **Mean** | **TBD** | — |

## Comparison vs PR #1489

| Technique | PR #1489 (SP1024) | Run 011 (SP8192) |
|-----------|-------------------|------------------|
| Tokenizer | SP1024 (vocab 1024) | SP8192 (vocab 8192) |
| Pre-quant TTT | ✓ 6 epochs | ✓ 6 epochs |
| Parallel Residuals | ✓ L7+ | ✓ L7+ |
| QK-Gain | 5.0 | 5.0 |
| EMA | 0.9965 | 0.9965 |
| Expected BPB | 1.07389 | 1.070-1.078 |

## Files

- `train_gpt.py` — Training script (copied from PR #1489)
- `run_all_seeds.sh` — 3-seed runner
- `job.tp.toml` — TensorPool job config (~/parameter-golf-project/jobs/run011.tp.toml)

## Next Steps

After completion:
1. Compare mean BPB vs PR #1489 (SP1024)
2. If SP8192 matches/exceeds SP1024 → tokenizer effect isolated
3. If SP8192 underperforms → SP1024 was key to our success
4. Submit as PR if beats SOTA 1.0810
