# Non-record: Mamba-3 Hybrid SSM + Multi-Epoch TTT + Dynamics-Protected Quant — 1.1456 bpb (3-seed mean)

**val_bpb: 1.1456** (3-seed mean, std 0.0011) | **15.93 MB total** (3-seed mean) | 8×H100

A follow-up SSM submission building on PR #1644 (1.1473 bpb). Same 7-layer Mamba-3/Attention hybrid; the −1.7 mBPB improvement comes from three quant/TTT-phase changes that don't touch the architecture.

| Seed | BF16 | Post-quant+TTT | Total submission |
|------|------|----------------|------------------|
| 1337 | 1.1389 | **1.1441** | 15,930,191 B |
| 42   | 1.1462 | **1.1460** | 15,961,203 B |
| 2025 | 1.1495 | **1.1468** | 15,975,083 B |
| **Mean** | **1.1449** | **1.1456** | **15,955,492 B** |
| **Std**  | **0.0045** | **0.0011** | **18,852 B** |

Submitted artifact corresponds to seed 1337 (1.1441, 15,930,191 B).

## What changed vs PR #1644

1. **`TTT_EPOCHS=2`**: PR #1644 used a single TTT epoch and saw a +8.3 mBPB BF16 → post-quant regression. With ep=2, the regression flips to approximately neutral (mean +0.7 mBPB across 3 seeds). The second epoch gives the model enough adaptation budget to recover the quant noise injected by INT6. Cost: 132s vs 76s for the TTT phase, both within the 600s eval budget.

2. **Mixed-precision SSM dynamics protection**: the `dd_A` and `dd_dt` rows of each Mamba-3 `in_proj.weight` (32 of 2232 rows per SSM block) are quantized at INT8 instead of INT6. Q-Mamba (ICLR 2025) showed that uniform 6-bit PTQ collapses Mamba perplexity from 5.5 to >21 because A/Ā errors compound through the recurrence. Promoting just these semantic-specific rows to INT8 costs ~0.01 MiB at our scale and recovers ~0.8 mBPB of quality. Implemented as per-row bit widths threaded through both the GPTQ path and the percentile-search path. New env var `QUANT_BITS_SSM_DYNAMICS=8` (default in `Hyperparameters`).

3. **Scale-floor quant bug fix**: an earlier mixed-precision commit accidentally hardcoded `scale.clamp_min(1.0/127)` (INT8 floor) for ALL rows, including INT6 rows that should floor at `1/31`. Consequence: INT6 q-values spread across [-31, 31] more uniformly, inflating LZMA entropy and starving selective ±1 pruning. Fixed to use per-row `1/qmax`. Net effect: ~1.4 MiB of spurious size inflation on prior runs disappears.

## Architecture (unchanged from PR #1644)

7-layer Mamba-3 SISO hybrid: 5 SSM blocks + 2 FlashAttention layers at positions 2 and 5, dim=512, d_state=64, expand=2, headdim=64, chunk_size=64, mlp_mult=3, 25.16M params. SP8192 BPE tokenizer trained from scratch on FineWeb. See PR #1644 for the full architectural rationale and Triton kernel analysis (no kernel-level changes here).

## Reproduction

```bash
SEED=1337 VOCAB_SIZE=8192 NUM_LAYERS=7 NUM_ATTN_LAYERS=2 \
  TRAIN_SEQ_LEN=4096 WARMDOWN_ITERS=2600 WARMDOWN_SHAPE=linear \
  MUON_EQ_R=1 LATE_QAT_THRESHOLD=0.15 \
  USE_GPTQ=1 QUANT_BITS=6 QUANT_BITS_EMBED=8 GPTQ_NUM_SEQS=32 \
  EVAL_OVERLAP=1024 USE_LZMA=1 EVAL_TEMP=0.9 TTT_EPOCHS=2 \
  WEIGHT_DECAY=0.04 MUON_MOMENTUM=0.99 MATRIX_LR=0.025 \
  torchrun --nproc_per_node=8 train_mamba3_hybrid.py
```

`QUANT_BITS_SSM_DYNAMICS=8` is the default in `Hyperparameters` and does not need to be set explicitly. Repeat with `SEED=42` and `SEED=2025` for the 3-seed mean.

## Data

Same as PR #1644: SP8192 BPE tokenizer trained from scratch on FineWeb-10B because the `kevclark/parameter-golf` SP8192 tokenizer was not consistent with this submission's tokenizer config. Tokenized shards and tokenizer artifacts available on a private HF dataset on request.

## What I tested and removed

This is a non-record submission and represents the cleaned production path from a much larger experimental sprint. The training script in this PR is the lean banked version. Many techniques that did not survive empirical validation at 25M / 10min / 16MB / SP8192 are not represented in this PR — including 1-attention ratio (works at SP4096, fails at SP8192 by +7.5 mBPB BF16), low-rank `in_proj` factorization (fails because random factored init destroys upstream's structured init for `dd_A`/`dd_dt` rows), depth recurrence at SP8192 (fails by +13.9 mBPB BF16 at expand=1.5), MLP INT5 quantization (+8 mBPB quality), and several others. Three structural findings emerged from this work:

- **LZMA compression penalty for SSM weights**: SSM `in_proj` rows compress ~3× worse than attention's homogeneous QKV under LZMA (33% vs 40% reduction measured), because the heterogeneous semantic row groups (z, xv, B, C, dd_dt, dd_A, trap, angles) have high within-tensor entropy.

- **Muon optimizer-vs-SSM discordance**: Muon's Newton-Schulz orthogonalization treats `in_proj` as one homogeneous matrix and flattens row-magnitude variance that SSM dynamics rows actually need. Co-varying signal: replacing 2-attn with 1-attn at SP8192 7L costs +7.5 mBPB BF16 even though it was a clean win at SP4096.

- **SP4096 → SP8192 architectural-finding non-transfer**: at 25M, embedding parameter cost shifts allocation enough that architectural sweeps at SP4096 don't generalize to SP8192. Two techniques validated at SP4096 (1-attention, expand=1.5 depth recurrence) flipped sign at SP8192.
