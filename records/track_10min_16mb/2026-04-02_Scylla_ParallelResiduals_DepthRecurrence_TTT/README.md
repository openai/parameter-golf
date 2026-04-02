# Record: Scylla + Parallel Residuals + Mini Depth Recurrence + Legal TTT

**val_bpb: 1.0876** (3-seed mean, std 0.00037) | **1.9469 nats** | **≤15.83 MB** | 8×H100 SXM, 600s + TTT

**Beats current merged SOTA** ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), **1.1147 BPB**, by @abaybektursun) **by −0.0271 BPB.** This is our own prior work — we are improving on our own merged record.

## Our Journey

This submission builds on our prior record-setting work in this competition:

1. **[PR #399](https://github.com/openai/parameter-golf/pull/399)** (1.1247 BPB, merged Mar 22) — We introduced **Parallel Muon optimizer + Parameter Banking** to the competition, based on @kellerjordan's modded-nanogpt. This reduced step time from ~110ms to ~82ms, enabling more training steps within the 600s budget.

2. **[PR #549](https://github.com/openai/parameter-golf/pull/549)** (1.1194 BPB, merged Mar 24) — We combined **LeakyReLU(0.5)²** (adopted from PR #493 by @parinzee and PR #518 by @sofiabod, -0.003 BPB) with **legal score-first TTT** (adapted from PR #461 by @Christopher-Lee-McClendon, we proved unfreezing all blocks is optimal at 3 epochs) into a unified record stack. This became the training base for multiple subsequent submissions by other participants, including PR #1242 (1.0903 BPB).

3. **[PR #1019](https://github.com/openai/parameter-golf/pull/1019)** (1.1147 BPB, merged Mar 30) — We introduced **AR self-generated GPTQ calibration** (model generates its own calibration data — no external data needed during quantization, a novel approach in this competition) and **all-layer XSA** (extending XSA from the last 4 layers to all 11). This is the current merged SOTA.

4. **PROTEUS EMA Notable** (1.1836 BPB, non-record Mar 25) — Our baseline documenting EMA weight averaging gains.

5. **PROTEUS+STYX N-gram Cache** (0.8495 BPB, non-record Mar 26) — Our exploration of backward-looking n-gram hash caches during sliding window eval.

Our PR #549 stack was subsequently used as the training base by PR #1242 (@Campbellb, 1.0903 BPB), which combined it with the Scylla tokenizer. This submission reclaims the lead by adding architectural innovations (parallel residuals + depth recurrence) on top of our own foundation.

## What's New (and What Changed from PR #1019)

**Added:**
- Scylla tokenizer ([@simon-marcus](https://github.com/openai/parameter-golf/pull/1143), 998-token TokenMonster)
- Parallel residual routing from layer 7 ([@msisovic](https://github.com/openai/parameter-golf/pull/1204))
- Mini depth recurrence on layers 4,5 with untied MLPs (PR #1204)
- Mixed INT5/INT6 per-row quantization + brotli-11 compression
- Learnable lane merge for parallel residuals

**Changed from PR #1019:**
- XSA reduced from all 11 layers to last 4 (Scylla base default; all-layer XSA added step time without TTT benefit on this tokenizer)
- BigramHash changed from 3072×112 to 2048×128 (budget tradeoff for recurrence params)
- GPTQ replaced with per-row INT5/INT6 (no Hessian calibration needed; simpler, fits budget with brotli)
- Compression changed from LZMA-9 to brotli-11 (better ratio for quantized weights)

## 3-Seed Results (8×H100 80GB SXM, 600s training + TTT eval)

| Seed | Steps | ms/step | Post-EMA BPB | Sliding BPB | **Legal TTT BPB** | Artifact |
|------|-------|---------|--------------|-------------|-------------------|----------|
| 42 | 5,875 | 102.2 | 1.0967 | 1.0981 | **1.0872** | 15,814,644 |
| 1337 | 5,878 | 102.1 | 1.0974 | 1.0973 | **1.0879** | 15,823,670 |
| 2024 | 5,884 | 102.0 | 1.0973 | 1.0982 | **1.0877** | 15,834,859 |
| **Mean** | **5,879** | **102.1** | **1.0971** | **1.0979** | **1.0876** | **15,824,391** |

All seeds stopped by 600s wallclock cap. All artifacts under 16,000,000 bytes.

Comparison vs current merged SOTA ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), @abaybektursun): **1.1147 BPB → 1.0876 BPB (−0.0271 BPB).** Note: nats are not directly comparable across tokenizers; BPB is the tokenizer-agnostic metric.

## Architecture

### Parallel Residuals (from [PR #1204](https://github.com/openai/parameter-golf/pull/1204), originally [modded-nanogpt #230](https://github.com/KellerJordan/modded-nanogpt/pull/230))

Starting from layer 7 (of 11), attention and MLP operate on separate residual lanes. Each sublayer writes back to both lanes through 4 learned routing scalars (`attn_to_attn`, `attn_to_mlp`, `mlp_to_attn`, `mlp_to_mlp`). Lanes merge via a learned scalar before the output head.

Each parallel block has an independent `resid_mix_mlp` parameter for the MLP lane's blending with the initial residual, allowing attn and MLP to specialize their input mixing.

### Mini Depth Recurrence (from [PR #1204](https://github.com/openai/parameter-golf/pull/1204))

Layers 4 and 5 are repeated once each (11 physical → 13 virtual layers). The repeated passes share attention weights but use untied MLP weights, adding ~3.1M parameters. Layer 4 is the last encoder layer; layer 5 is the first decoder layer (post-skip), placing recurrence at the U-Net hinge point.

### Scylla Tokenizer (from [PR #1143](https://github.com/openai/parameter-golf/pull/1143), @simon-marcus)

998-token TokenMonster vocabulary discovered via autoresearch. Full FineWeb retokenization (80 train + 1 val shard, ~7.9B tokens). Runtime byte accounting via per-token metadata.

### Mixed Quantization

Per-row INT5 (`clip_range=15`) for middle MLP layers (3–7), INT6 (`clip_range=31`) for attention + first/last 2 MLP layers, INT8 for small control tensors. Brotli quality=11 compression. This sensitivity-driven allocation keeps the artifact under 16 MB while preserving model quality where it matters most.

### Legal Score-First TTT

Score-first SGD following the accepted [PR #461](https://github.com/openai/parameter-golf/pull/461) framework. Each 32,768-token chunk is scored under `torch.inference_mode()` before any parameter update. BPB is always computed before adaptation. LR=0.005, 3 epochs, 2 freeze blocks. TTT runs ~490s.

### N-gram Two-Pass Rescoring

Orders 2–12, 16M buckets, entropy-adaptive alpha blending, leave-one-out. Two-pass eval: Pass 1 stores per-token neural probabilities, Pass 2 rescores with n-gram cache. N-gram BPB reported separately, not used as submission metric.

## Full Technique Stack

- **Scylla tokenizer** — 998 vocab TokenMonster (PR #1143)
- **Parallel residuals** — from layer 7, learned 4-scalar routing (PR #1204)
- **Mini depth recurrence** — layers 4,5 repeated, untied MLPs (PR #1204)
- **Legal TTT** — score-first SGD, LR=0.005, 3 epochs (our PR #549)
- **N-gram rescoring** — orders 2–12, two-pass eval
- **11L transformer** — 512d, GQA(8/4), MLP 3×, LeakyReLU(0.5)²
- **XSA** — last 4 layers
- **SmearGate** — gated previous-token blending
- **BigramHash** — 2048 vocab, 128 dim
- **ValueEmbedding** — shared, layers 9,10
- **EMA** (0.997) + **SWA** (every 50 steps)
- **Parallel Muon** optimizer + AdamW for scalars/embeddings
- **Mixed INT5/INT6 quantization** + **brotli-11** compression
- **Learnable lane merge** — single scalar for parallel lane averaging

## Statistical Significance

Welch t-test vs current merged SOTA (PR #1019, 3-seed mean 1.11474 BPB):
- **t = −91.92**, **df = 3.99**, **p ≪ 0.01**
- Delta: −0.0271 BPB (far exceeding the 0.005 nats ≈ 0.003 BPB threshold)

## Note on Cross-Tokenizer Comparison

This submission uses the Scylla tokenizer (998-token TokenMonster) while the merged SOTA (PR #1019) uses sp1024 SentencePiece. Raw nats (cross-entropy per token) are not directly comparable across tokenizers — a tokenizer with fewer, longer tokens will have higher per-token nats even when modeling the same bytes more efficiently. BPB (bits per byte) is the tokenizer-agnostic metric the competition uses for the leaderboard. Our BPB improvement of −0.0271 over merged SOTA is unambiguous.

This is consistent with other cross-tokenizer submissions (PR #1143 Scylla base, PR #1242 Scylla + TTT) which were evaluated on BPB.

## Legality

**TTT:** Each chunk scored under `torch.inference_mode()` before any parameter update. BPB is always computed before adaptation. Follows the accepted PR #461 score-first framework.

**N-gram:** `legal_ttt` score uses only the TTT-adapted neural model. N-gram two-pass is reported separately, not used as submission metric.

**Known log artifact:** The `final_int8_zlib_roundtrip_exact` line appears three times in each log — once after INT6 roundtrip, once after sliding window, and once after n-gram. Only the first occurrence reflects the quantized neural model. This is a logging quirk inherited from the Scylla base script, not a score manipulation.

No validation data accessed during training. TTT trains on validation tokens only after they have been scored (legal per FAQ).

## Reproduction

```bash
pip install tokenmonster brotli

# Retokenize FineWeb with Scylla
python3 data/retokenize_scylla.py \
    --vocab ./data/tokenizers/scylla/candidate.vocab \
    --output-dir ./data/datasets/fineweb_scylla \
    --sp-shards "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin" \
    --sp-model ./data/tokenizers/fineweb_1024_bpe.model

# Train (per seed)
for SEED in 42 1337 2024; do
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
  N_INT6_LAYERS=4 SEED=$SEED \
  DATA_PATH=./data/datasets/fineweb_scylla \
  TOKENIZER_PATH=./data/tokenizers/scylla/candidate.vocab \
  TOKENIZER_META_PATH=./data/tokenizers/scylla/candidate.meta.npz \
  VOCAB_SIZE=998 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
  PARALLEL_START_LAYER=7 RECUR_LAYERS=4,5 RECUR_UNTIE_MLP=1 \
  XSA_LAST_N=4 LN_SCALE=1 ROPE_DIMS=16 \
  BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
  VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
  ACTIVATION_MODE=leaky_relu_sq ACTIVATION_NEG_SLOPE=0.5 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  MUON_WD=0.04 GRAD_CLIP_NORM=0.3 \
  TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
  ITERATIONS=9000 WARMDOWN_ITERS=3500 MAX_WALLCLOCK_SECONDS=600 \
  EVAL_STRIDE=64 SWA_ENABLED=1 SWA_EVERY=50 EMA_DECAY=0.997 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Credits

### Our prior work (foundation for this submission)
- **PR #549** (@abaybektursun) — LeakyReLU², legal score-first TTT, Parallel Muon. Merged SOTA at 1.1194 BPB (Mar 24).
- **PR #1019** (@abaybektursun) — AR self-generated GPTQ calibration, all-layer XSA, BigramHash 3072×112. Current merged SOTA at 1.1147 BPB (Mar 30).
- **PROTEUS EMA Notable** (@abaybektursun) — EMA weight averaging baseline (Mar 25).
- **PROTEUS+STYX N-gram** (@abaybektursun) — Early n-gram eval cache exploration (Mar 26).

### External contributions integrated
- **Scylla tokenizer:** @simon-marcus (PR #1143) — 998-token TokenMonster vocabulary via autoresearch.
- **Parallel residuals + mini depth recurrence:** @msisovic (PR #1204, originally from modded-nanogpt #230 by @KellerJordan) — dual residual lanes + layer 4,5 repetition.
- **Legal TTT framework:** @Christopher-Lee-McClendon (PR #461) — score-first evaluation pattern.
- **Mixed quantization concept:** PR #1105 — per-layer bitwidth allocation.
- **Parallel Muon optimizer:** @kellerjordan (modded-nanogpt) — Newton-Schulz orthogonalization for bank params.
- **Competition infrastructure:** @signalrush (PR #414) — 11L EMA + GPTQ-lite base that PR #549 built upon.
