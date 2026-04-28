# Record: SmearGate BOS Fix — 3-Seed Reproduction of PR #1851

**val_bpb = 1.06145** (3-seed mean, std 0.00068) | **~15.95 MB** | 8xH100 SXM 80GB

## Summary

This is a **pure reproduction study** of [PR #1851](https://github.com/openai/parameter-golf/pull/1851) by @aquariouseworkman. The training script is byte-identical to the code in PR #1851. No new techniques or modifications are introduced.

PR #1851 submitted a single-seed result (seed 42, val_bpb = 1.06128). We extend this to a **3-seed evaluation** (seeds 42, 314, 1234) to confirm the result is robust and reproducible.

## 3-Seed Results

| Seed | Pre-Quant BPB | Quant BPB | **Post-TTT BPB** | Artifact (bytes) | Train Time | Eval Time |
|------|---------------|-----------|-------------------|-------------------|------------|-----------|
| 42*  | 1.06490240    | 1.07405660 | **1.06128183**   | 15,952,086        | 599.6s     | 519.5s    |
| 314  | 1.06467893    | 1.07358634 | **1.06086831**   | 15,952,419        | 599.6s     | 525.6s    |
| 1234 | 1.06593114    | 1.07503808 | **1.06220261**   | 15,952,690        | 599.5s     | 479.6s    |
| **Mean ± Std** | | | **1.06145 ± 0.00068** | | | |

\* Seed 42 result is from the original PR #1851 author @aquariouseworkman. Seeds 314 and 1234 are independent runs by @Christopher-Lee-McClendon.

## Key Change: SmearGate BOS Document Boundary Fix

PR #1851 identified and fixed a bug in the SmearGate mechanism's handling of beginning-of-sequence (BOS) document boundaries. The fix ensures SmearGate correctly resets at document boundaries instead of bleeding attention across documents.

This was a targeted one-line fix on top of the PR #1787 codebase. Credit for identifying the BOS bug goes to @cocohearts; the fix implementation is by @aquariouseworkman.

## Technique Stack

All techniques below are inherited from PR #1851 (and its lineage). No new techniques are introduced in this reproduction.

| Technique | Source | Author |
|-----------|--------|--------|
| Base architecture (11L, MLP 4x, MuonEq-R) | PR #1787 | @nprime06 |
| SmearGate attention | PR #1797 | @dexhunter |
| SmearGate BOS fix | PR #1851 | @aquariouseworkman |
| LQER Asymmetric quantization | PR #1797 | @dexhunter |
| CaseOps SP8192 | PR #1729 | @romeerp |
| GPTQ + SP8192 | PR #1394 | @clarkkev |
| Score-first TTT (3 phases) | PR #549 | @abaybektursun |
| BOS bug identification | Issue | @cocohearts |

## Architecture

Same as PR #1851 / PR #1787:
- 11 transformer layers, MLP multiplier 4x
- SmearGate attention with BOS boundary fix
- LQER asymmetric quantization
- CaseOps with SP8192 tokenization
- GPTQ post-training quantization
- Phased test-time training (3 phases)
- Embed clipping (15.0σ), MLP clipping (12.0σ)
- Embed bits: 7

## Compliance

| Budget | Limit | Worst-Case (across seeds) | Status |
|--------|-------|--------------------------|--------|
| Artifact size | 16,000,000 bytes | 15,952,690 bytes | ✅ |
| Training time | 600s | 599.6s | ✅ |
| Eval time | 600s | 525.6s | ✅ |

## Reproduction

The training script is byte-identical to PR #1851. To reproduce:

```bash
# 1. Install dependencies
pip install brotli python-minifier

# 2. Prepare CaseOps SP8192 data
#    Option A: Download pre-tokenized CaseOps data from HuggingFace
python3 prepare_caseops_data.py  # downloads from romeerp/parameter-golf-caseops-v1
#    Option B: Or use the standard data script
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest
#    Then apply CaseOps transform:
python3 lossless_caps.py  # transforms shards with CaseOps encoding

# 3. Run training (replace SEED with 42, 314, or 1234)
SEED=42 \
CASEOPS_ENABLED=1 \
EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
EMBED_CLIP_SIGMAS=15.0 \
MLP_CLIP_SIGMAS=12.0 \
GPTQ_RESERVE_SECONDS=0.5 \
PHASED_TTT_NUM_PHASES=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Environment variables (all required for exact reproduction):**

| Variable | Value | Purpose |
|----------|-------|---------|
| `CASEOPS_ENABLED` | `1` | Enable CaseOps SP8192 tokenization |
| `EMBED_BITS` | `7` | Embedding quantization bits |
| `SMEAR_GATE_ENABLED` | `1` | Enable SmearGate attention |
| `SPARSE_ATTN_GATE_ENABLED` | `1` | Enable sparse attention gating |
| `MIN_LR` | `0.1` | Minimum learning rate |
| `EMBED_CLIP_SIGMAS` | `15.0` | Embedding clipping threshold (σ) |
| `MLP_CLIP_SIGMAS` | `12.0` | MLP clipping threshold (σ) |
| `GPTQ_RESERVE_SECONDS` | `0.5` | Seconds reserved for GPTQ |
| `PHASED_TTT_NUM_PHASES` | `3` | Number of TTT phases |

**Hardware:** 8×H100 SXM 80GB (RunPod)

## Credits

- **@aquariouseworkman** — PR #1851 author (SmearGate BOS fix, seed 42 result)
- **@nprime06** — PR #1787 (base architecture)
- **@romeerp** — PR #1729 (CaseOps)
- **@dexhunter** — PR #1797 (SmearGate + LQER asymmetric quantization)
- **@cocohearts** — BOS document boundary bug identification
- **@abaybektursun** — PR #549 (score-first TTT)
- **@clarkkev** — PR #1394 (GPTQ + SP8192)

### Experimental train-only logit calibration variant

This branch adds an optional post-GPTQ, train-only logit calibration pass for testing on top of the reproduced #1851/#1868 stack. It fits a fixed global temperature plus coarse token-group bias using only training tokens, then applies the frozen affine correction before softmax in both the quantized diagnostic eval and the phased score-first TTT loss.

Default controls:

```bash
LOGIT_CALIB_ENABLED=1
LOGIT_CALIB_TOKENS=100000
LOGIT_CALIB_STRIDE=64
LOGIT_CALIB_BATCH_SEQS=8
LOGIT_CALIB_LR=0.003
LOGIT_CALIB_L2=0.01
LOGIT_CALIB_EPOCHS=1
LOGIT_CALIB_APPLY_TTT_UPDATE=1
```

Set `LOGIT_CALIB_ENABLED=0` to recover the byte-identical #1868 behavior. The calibration pass does not read validation targets or build validation-derived state; rank 0 fits on train shard tokens and broadcasts the frozen scale/bias to all ranks before eval.
