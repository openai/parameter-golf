# Record: SmearGate BOS Fix — 3-Seed Compliance Re-run of PR #1851

**val_bpb = 1.06141** (3-seed mean, std 0.00093) | **~15.95 MB** | 8xH100 SXM 80GB

## Summary

This is a **compliance re-run** of [PR #1868](https://github.com/openai/parameter-golf/pull/1868), which was itself a 3-seed reproduction of [PR #1851](https://github.com/openai/parameter-golf/pull/1851) by @aquariouseworkman.

The original submission used `GPTQ_RESERVE_SECONDS=0.5`, which was insufficient — the training loop ran until ~599.5s, and GPTQ hessian collection (which accesses training data) extended past the 600s budget. This re-run fixes the compliance issue by using `GPTQ_RESERVE_SECONDS=8.0`, so the training loop ends at ~592s and GPTQ hessian collection completes by ~595.5s, well within the 600s budget.

All 3 seeds were re-run fresh. Results are consistent with the originals (mean 1.06141 vs 1.06145), confirming the fix has negligible impact on model quality.

### What Changed

1. **`GPTQ_RESERVE_SECONDS` increased from 0.5 → 8.0** — Training loop now stops ~8s early to leave room for GPTQ hessian collection within the 600s budget.
2. **Serialize-before-diagnostic reordering** — The patched `train_gpt.py` writes the compressed artifact immediately after GPTQ quantization, before running pre-quant diagnostic evaluation. This ensures the artifact is saved as soon as possible.
3. **Timing instrumentation** — Added `serialize_wallclock` logging for transparency.

### Results vs Original

| Metric | Original (#1868) | Re-run (this) | Delta |
|--------|-------------------|---------------|-------|
| 3-seed mean BPB | 1.06145 | **1.06141** | −0.00004 |
| Std dev | 0.00068 | 0.00093 | +0.00025 |
| Train time | ~599.5s | ~592.1s | −7.4s |
| GPTQ hessians end | ~603s (❌ over budget) | ~595.5s (✅) | fixed |

## 3-Seed Results

| Seed | Pre-Quant BPB | Quant BPB | **Post-TTT BPB** | Artifact (bytes) | Train Time | Eval Time | Serialize |
|------|---------------|-----------|-------------------|-------------------|------------|-----------|-----------|
| 42   | 1.06457584    | 1.07361172 | **1.06083288**   | 15,949,701        | 592.1s     | 525.5s    | 81.8s     |
| 314  | 1.06474209    | 1.07364034 | **1.06090748**   | 15,951,777        | 592.0s     | 429.5s    | 81.4s     |
| 1234 | 1.06619783    | 1.07529543 | **1.06248776**   | 15,951,968        | 592.1s     | 481.2s    | 79.5s     |
| **Mean ± Std** | | | **1.06141 ± 0.00093** | | | | |

All 3 seeds run by @Christopher-Lee-McClendon on RunPod 8×H100 SXM.

### GPTQ Timing Breakdown (consistent across seeds)

| Phase | Time | Accesses Training Data? |
|-------|------|------------------------|
| Hessian collection | ~3.5s | ✅ Yes — must be within 600s |
| Quantization | ~10.1s | ❌ No — uses cached Hessians |
| Brotli compression | ~65-67s | ❌ No — pure I/O |
| **Total serialize** | **~80s** | |

Training loop ends at ~592s + hessian collection ~3.5s = **~595.5s** (< 600s ✅).

## Key Change: SmearGate BOS Document Boundary Fix

PR #1851 identified and fixed a bug in the SmearGate mechanism's handling of beginning-of-sequence (BOS) document boundaries. The fix ensures SmearGate correctly resets at document boundaries instead of bleeding attention across documents.

This was a targeted one-line fix on top of the PR #1787 codebase. Credit for identifying the BOS bug goes to @cocohearts; the fix implementation is by @aquariouseworkman.

## Technique Stack

All techniques below are inherited from PR #1851 (and its lineage). No new techniques are introduced.

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
| Artifact size | 16,000,000 bytes | 15,951,968 bytes | ✅ |
| Training time (loop + GPTQ hessians) | 600s | ~595.5s | ✅ |
| Eval time (scoring only) | 600s | 525.5s | ✅ |
| Eval time (incl. torch.compile warmup) | — | 709.4s (seed 42) | ⚠️ |

> **Note on eval time:** The `eval_time` reported above excludes `torch.compile` warmup, which is a one-time process startup cost (not evaluation work). All submissions using `torch.compile` have this overhead. Seed 42's warmup was 183.9s (first run on cold pod); seeds 314 and 1234 had ~108s warmup. Including warmup, seed 314 total is 538.2s and seed 1234 total is 588.2s — both within 600s.

**GPTQ compliance:** With `GPTQ_RESERVE_SECONDS=8.0`, training stops at ~592s. GPTQ hessian collection takes ~3.5s (accesses training data, so must be within 600s budget). Total training-data-access time: ~595.5s < 600s. Subsequent quantization and compression do not access training data.

## Reproduction

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
GPTQ_RESERVE_SECONDS=8.0 \
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
| `GPTQ_RESERVE_SECONDS` | `8.0` | Seconds reserved for GPTQ (was 0.5 in original) |
| `PHASED_TTT_NUM_PHASES` | `3` | Number of TTT phases |

**Hardware:** 8×H100 SXM 80GB (RunPod)

## Credits

- **@aquariouseworkman** — PR #1851 author (SmearGate BOS fix, original code)
- **@nprime06** — PR #1787 (base architecture)
- **@romeerp** — PR #1729 (CaseOps)
- **@dexhunter** — PR #1797 (SmearGate + LQER asymmetric quantization)
- **@cocohearts** — BOS document boundary bug identification
- **@abaybektursun** — PR #549 (score-first TTT)
- **@clarkkev** — PR #1394 (GPTQ + SP8192)
