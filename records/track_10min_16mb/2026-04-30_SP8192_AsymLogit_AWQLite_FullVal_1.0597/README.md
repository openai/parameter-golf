# SP8192 PR #1855 Base + Asymmetric Logit Rescale + AWQ-lite

**Score: 1.05971 BPB** (3-seed mean, full val partition, seeds 42 / 0 / 1234)

| Seed | val_bpb | val_loss | train wallclock | eval | artifact |
|------|--------:|---------:|----------------:|-----:|---------:|
| 42   | 1.06030 | 2.32033 | 599.6s | 532s | 15,944,044 B |
| 0    | 1.05970 | 2.31902 | 599.6s | 419s | 15,941,061 B |
| 1234 | 1.05912 | 2.31776 | 599.5s | 457s | 15,951,087 B |
| **mean** | **1.05971** | **2.31904** | **599.6s** | **469s** | **15,945,397 B** |
| std (pop) | 0.000478 | — | — | — | — |

`val_tokens: 47,851,520` on every seed (full validation partition, identical to #1855's measurement).

vs current rank 1 (PR #1855 1.06108): **−0.00137 BPB** — Welch t≈4.96 with 2 dof, p < 0.05.

## Submission history

This PR was originally opened on 2026-04-29 reporting 1.06577 BPB (3-seed mean). On review, @codemath3000 noted that the `val_tokens` line in our seed-42 log read `9,662,464` instead of the standard `47,851,520`, indicating measurement on a truncated validation partition. We confirmed and root-caused: a corrupted `fineweb_val_000000.bin` (19 MB instead of 95.7 MB) in the local network volume. Re-pulling the file directly from `huggingface.co/datasets/romeerp/parameter-golf-caseops-v1` restored the full 47.85M-token val partition.

While re-running on the full val we extended the stack with a second orthogonal addition on top of #1855 (AWQ-lite mixed-precision GPTQ, concurrently identified with @romeerp's PR #1908). The current submission reflects:

- corrected val measurement (full partition);
- the original Asymmetric Logit Rescale lever; and
- the AWQ-lite addition.

`submission.json` carries both revisions for transparency.

## Approach

Two **orthogonal eval-path additions** on top of the verbatim PR #1855 stack:

### 1. Asymmetric Logit Rescale

Two learnable scalars `softcap_pos`, `softcap_neg` replace the single `logit_softcap` in `forward_logits` and `forward_ttt`. Logits are split by sign:

```python
torch.where(logits > 0,
            softcap_pos * tanh(logits / softcap_pos),
            softcap_neg * tanh(logits / softcap_neg))
```

Both initialized to `logit_softcap = 30.0` so eval is identity at start. Trained inside Phased TTT (passed through global SGD as part of `model.parameters()`). They settle at slightly different values for the positive/negative tails of the next-token distribution.

The fused softcapped-CE Triton kernel on the training path is left unchanged (single softcap), so **train-time numerics match #1855 exactly**. The two scalars receive no gradient during training and stay at init until eval.

Artifact cost: 8 bytes (2 fp16 scalars in the passthrough float16 list). Compressed cost is below the lrzip block overhead, effectively free.

### 2. AWQ-lite mixed-precision GPTQ

During GPTQ calibration, collect activation RMS per layer; select the most-salient 64-column group; keep that group at int8 inside the GPTQ solve while the remainder stays at int6. Identical pattern to PR #1908 by @romeerp; we converged on the same recipe independently while iterating on the AsymLogit branch and confirmed it stacks orthogonally with AsymLogit (positive) on the full val partition.

Env vars: `AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64`.

## Stack inherited from #1855

- **SP-8192 tokenizer** with CaseOps lossless casing (PR #1729)
- **LQER asymmetric int4** (rank 4, group 64, top-3 layers) on tok_emb + selected MLP weights
- **Sparse Attn Gate** on attention out
- **BOS-Fixed SmearGate** in `_forward_hidden` and `forward_ttt` (cross-doc leak fix)
- **9-hparam greedy stack** (#1855)
- **Phased Multi-Phase Global SGD TTT** (3 phases, prefix=2500 docs)
- **Per-group lrzip ZPAQ** compression
- **8× H100 SXM**, FA3, fused softcapped CE, torch 2.9.1+cu128

## Hyperparameters (additions over #1855)

```bash
ASYM_LOGIT_RESCALE=1
AWQ_LITE_ENABLED=1
AWQ_LITE_BITS=8
AWQ_LITE_GROUP_TOP_K=1
AWQ_LITE_GROUP_SIZE=64
HASH_EMBED_ENABLED=0          # gated; not used here
```

All other env vars match #1855 verbatim (see PR #1855 README).

## Compliance (Issue #1017)

- **C1 strict causal dependence:** standard varlen + per-doc cu_seqlens; no future-token leakage
- **C2 full normalized distribution:** standard log-softmax over SP8192 vocab; AsymLogit is a deterministic monotone reshaping of logits before softmax — distribution remains a normalized full-vocab distribution
- **C3 score-before-update:** Phased TTT scores each chunk before any LoRA gradient step
- **C4 single pass:** each val token scored exactly once
- **No SLOT, no n-gram cache, no logit bias, no PPM, no pre-quant TTT on val data**
- **Compute caps:** train ≤599.6s, eval ≤532s, all 3 seeds. `MAX_WALLCLOCK_SECONDS=600`.
- **Artifact:** ≤15,985,176 bytes for all 3 seeds. Cap is 16,000,000.

## Reproduction

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
apt-get install -y lrzip

for seed in 42 0 1234; do
  SEED=$seed \
  ASYM_LOGIT_RESCALE=1 \
  AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 \
  HASH_EMBED_ENABLED=0 \
  CASEOPS_ENABLED=1 PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
  EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
  MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
  GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
  GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
  TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 SPARSE_ATTN_GATE_SCALE=0.5 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
  SMEAR_GATE_ENABLED=1 \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
  FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
  MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${seed}.log
done
```

## Attribution

- **codemath3000** — PR #1855 (full stack baseline 1.06108) + caught the val truncation issue on review
- **romeerp** — PR #1908 (AWQ-lite mixed-precision GPTQ) and PR #1729 (CaseOps lossless tokenizer)
- **classiclarryd** — modded-nanogpt PR #181 (Asymmetric Logit Rescale)
- **dexhunter** — PR #1797 (LQER asym + SmearGate base) and PR #1736 (CaseOps + GatedAttn + QuantGate)
- **nprime06** — PR #1787 (Polar Express NS + MIN_LR + Sparse Attn Gate + Fused CE)
- **MarioPaerle** — PR #1667 (SmearGate origin)
- **renqianluo** — PR #1767 (LoRA TTT improvements)
- **Jorge Asenjo** — PR #1700 (Phased Multi-Phase Global SGD TTT, retained from #1855 stack); this PR (AsymLogit + AWQ-lite stacking on full val)
