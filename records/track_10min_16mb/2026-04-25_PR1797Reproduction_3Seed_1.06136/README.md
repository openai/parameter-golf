# Record: PR #1787 + Smear Gate + LQER Asymmetric — val_bpb 1.06136 (3-seed mean)

**val_bpb: 1.06136** (3-seed mean, std=0.00059) | **val_loss: 2.32265 nats/token** | **~15.95 MB** | 8×H100 SXM | Phased TTT

**−0.00199 BPB vs PR #1797 reference** (1.06335). All 3 seeds individually beat PR #1797's best seed (1.06297). Independent reproduction validates the stack and improves on reported numbers via stochastic variance and hardware tuning.

**−0.01964 BPB vs current merged SOTA** (PR #1493: 1.0810). Clears the 0.005-nat significance threshold by a wide margin (3-seed std 0.00059, delta-to-SOTA / std = 33).

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, RunPod IN region)

| Seed | Steps | Pre-quant BPB | Quantized BPB | **Post-TTT BPB** | TTT Eval | Train | Artifact |
|------|------:|--------------:|--------------:|-----------------:|---------:|------:|---------:|
| 42   | 4948  | 1.06451       | 1.07345       | **1.06068**      | 533.5s   | 599.6s | 15,951,346 |
| 0    | 4920  | 1.06560       | 1.07458       | **1.06163**      | 435.1s   | 599.5s | 15,947,797 |
| 1234 | 4916  | 1.06557       | 1.07472       | **1.06177**      | 468.9s   | 599.6s | 15,952,843 |
| **Mean** | **4928** | **1.06523** | **1.07425** | **1.06136**  | 479.2s   | 599.6s | **15,950,662** |
| **Std**  |          | 0.00053     | 0.00057     | **0.00059**      |          |        |            |

All 3 seeds clear the 600s train budget (≤599.6s), the 600s TTT eval budget (≤533.5s), and the 16,000,000-byte artifact cap (≤15,952,843, ≥47,157 bytes headroom).

## Significance

3-seed std 0.00059 BPB ≈ 0.00136 nats/token. Delta to PR #1797 baseline (1.06335) is 0.00199 BPB ≈ 0.00138 nats/token. Significance ratio = 0.00199 / (0.00059 / √3) ≈ 5.8 — well past p < 0.01.

## Stack Description

This is a faithful reproduction of PR #1787's stack:

- **PR #1394** (Kevin Clark) — SP8192 + GPTQ Embeddings + SDClip + brotli + MuonEq-R foundation
- **PR #1493** (bigbag) — 3-layer depth recurrence (loops 3-5) + parallel residuals + QK-gain 5.25 + score-first TTT
- **PR #1736** (dexhunter) — Phased TTT + GatedAttn quant gate
- **PR #1797** (dexhunter) — **direct baseline**: Smear Gate + LQER Asymmetric (INT2+INT4 factors) on top of PR #1787

## Reproduction

```bash
# Data prep
MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets \
python3 cached_challenge_fineweb.py \
  --variant sp8192_lossless_caps_caseops_v1_reserved --train-shards 80

# Install deps
pip install --break-system-packages \
  flash-attn-interface sentencepiece triton brotli python-minifier

# Run (per seed)
for SEED in 42 0 1234; do
  NCCL_NET=Socket DATA_DIR=./data CASEOPS_ENABLED=1 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 MATRIX_LR=0.026 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  VAL_LOSS_EVERY=0 MIN_LR=0.10 FUSED_CE_ENABLED=1 \
  SPARSE_ATTN_GATE_ENABLED=1 GATED_ATTN_QUANT_GATE=1 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
  LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > train_seed${SEED}.log 2>&1
done
```

## Compliance (Issue #1017 Track A)

- ✅ **Fixed predictor**: scored artifact is int6-GPTQ + LQER + brotli, no eval-time adaptation outside score-first TTT
- ✅ **Score-first TTT**: phased per PR #1767 framework — each chunk scored under `torch.no_grad()` BEFORE LoRA updates
- ✅ **No SLOT, no RLS, no n-gram cache, no ETLB, no logit biasing**
- ✅ **Sliding-window eval**: strictly causal, stride 64, single pass, normalized softmax over full vocab
- ✅ **CaseOps byte sidecar** for honest BPB on original bytes (Title/AllCaps/CapNext control tokens don't inflate counts)
- ✅ **Train < 600s** (≤599.6s on all 3 seeds)
- ✅ **Eval < 600s** (≤533.5s on all 3 seeds)
- ✅ **Artifact < 16,000,000 bytes decimal** (≤15,952,843 on all 3 seeds, ≥47,157 bytes headroom)

## Phase 3 Ablation Note

We tested Gram Newton-Schulz (Dao AI Lab CuTeDSL kernels, arxiv 2505.16932 + April 2026 packages) as a drop-in replacement for our Polar Express NS iteration on seed 42. Result: **no measurable speedup** at our parameter-bank scale, slight numerical divergence from stock NS (pre-quant 1.06536 vs 1.06451). The Dao AI Lab claim of 2× speedup applies to larger matrices than our 512×2048 bank shards. We dropped Gram NS from the final submission. Logs in `gram_ns_ablation/`.

## Credits

This submission stands on the work of many contributors. Direct credit chain:

- **@clarkkev** (PR #1394) — SP8192 + GPTQ Embeddings + SDClip + Brotli + MuonEq-R
- **@bigbag** (PR #1493) — Merged SOTA 1.0810: depth recurrence + parallel residuals + QK-gain
- **@dexhunter** (PR #1736, #1797) — Phased TTT, Quant Gate, Smear Gate, LQER Asym
- **@nprime06** (PR #1787) — Polar Express NS port, Fused CE Triton kernel, Sparse Attn Gate, MIN_LR, bug fixes
- **@classiclarryd** — Smear Gate concept (modded-nanogpt origin)
- **@MarioPaerle** (PR #1667) — Smear Gate port to parameter-golf
- **@romeerp** (PR #1729) — CaseOps lossless-case tokenizer + byte sidecar

## Hardware

- 8× NVIDIA H100 80GB HBM3 SXM (RunPod secure cloud, IN region)
- PyTorch 2.9.1+cu128, CUDA 12.8
- Flash Attention 3, Triton 3.5.1, Brotli 1.2.0
- Python 3.12.3
