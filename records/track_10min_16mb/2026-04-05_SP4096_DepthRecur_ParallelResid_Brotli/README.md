# Record: SP4096 + Depth Recurrence + Parallel Residuals + QK-Gain + Brotli

**val_bpb: 1.1020** (3-seed mean, std 0.0011) | **~15.88 MB max** | 8xH100 SXM, 600s | No TTT

**Improvement over current merged SOTA ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), 1.1147 BPB):** -0.0127 BPB / -0.0088 nats (Welch t=-18.37, df=2.38, p<0.001)

## Results

| Seed | Steps | ms/step | **Sliding Window BPB** | Model Bytes | Total Artifact |
|------|-------|---------|------------------------|-------------|----------------|
| 42   | 5,733 | 104.67  | **1.10327**            | 15,748,095  | 15,824,545     |
| 314  | 5,945 | 100.94  | **1.10181**            | 15,792,991  | 15,869,441     |
| 999  | 5,936 | 101.10  | **1.10102**            | 15,799,271  | 15,875,721     |
| **Mean** | |        | **1.10203**            |             |                |

Spread across seeds: 0.0023 BPB (very tight). All 3 seeds fit under 16MB with >=124KB margin.

## Tokenizer Change: BPB Correctness Proof

This submission uses a SentencePiece 4096 BPE tokenizer (`fineweb_4096_bpe.model`) instead of the baseline SP1024. Per competition rules, we provide detailed proof that val_bpb is correctly calculated.

**How BPB is computed in this script:**

The `val_bpb` metric is computed by the same `sliding_window_bpb()` function used by all submissions in this repo. The function:

1. Evaluates cross-entropy loss in nats per token over the full validation set using a sliding window (stride=64)
2. Counts the total number of bytes in the validation text by summing `token_byte_lengths[token_id]` for each token
3. Computes `BPB = total_nats / (total_bytes * ln(2))`

The `token_byte_lengths` lookup table is built by `build_sentencepiece_luts()`, which inspects each token's UTF-8 byte length via `sp.id_to_piece(token_id)`. This is independent of vocabulary size — a token that represents "the" is 3 bytes whether the vocab is 1024 or 4096.

**Key invariant:** The total byte count of the validation set is identical regardless of tokenizer, because every tokenizer produces a lossless segmentation of the same byte sequence. More tokens (SP1024) or fewer tokens (SP4096) — the bytes sum is the same. Therefore BPB is a fair cross-tokenizer comparison.

**Verification from logs:** The validation set has `tokens:45508608` SP4096 tokens. At ~3.32 bytes/token average, this covers the same ~151M byte validation set used by SP1024 submissions (which have ~131M tokens at ~1.15 bytes/token). The per-token cross-entropy is higher with SP4096 (2.54 nats vs 1.88 nats) because each token covers more bytes, but the per-byte rate (BPB) is directly comparable.

---

## What Changed vs PR #1019

This submission replaces the SP1024 + BigramHash + LZMA stack with a SP4096-native architecture that gets more capacity from the larger vocabulary and recurrent/parallel techniques instead of explicit bigram features.

### 1. SP4096 Tokenizer + MLP 4x (from SP1024 + MLP 3x)

Switching to a 4096-token SentencePiece vocabulary with 4x MLP multiplier increases model capacity from ~27M to 34.4M parameters. The larger vocabulary captures more subword patterns natively, eliminating the need for BigramHash (which compresses 3.4x worse per parameter with SP4096).

### 2. Depth Recurrence (Layers 4-5 from Step 3000)

After step 3000, layers 4 and 5 are re-executed, effectively giving the model 13 logical layers for the cost of 11 layers' parameters. This adds zero parameters — it's purely a compute-time technique that trades ~10% wall-clock time for improved representation depth. Source: [PR #1260](https://github.com/openai/parameter-golf/pull/1260) ablation, estimated -0.0035 BPB.

### 3. Parallel Residuals (Layer 7+)

From layer 7 onward, the MLP and attention outputs are merged through a learned `lane_merge` scalar and `resid_mix_mlp` vector per layer (~20KB raw, ~3-5KB compressed). This allows the model to balance attention vs MLP contributions dynamically. Source: [PR #1289](https://github.com/openai/parameter-golf/pull/1289), estimated -0.0035 BPB.

### 4. QK-Gain 5.0

Initializes query and key projections with 5x scale, sharpening attention from the start of training without any parameter cost. Source: [PR #1217](https://github.com/openai/parameter-golf/pull/1217) (45 experiments), estimated -0.001 BPB.

### 5. MuonEq-R Optimizer

Row-norm normalization before Newton-Schulz iteration in Muon. ~15 lines of code, zero parameter cost, minor but consistent improvement. Source: [PR #1334](https://github.com/openai/parameter-golf/pull/1334).

### 6. ADAM_WD=0.090 + GPTQ Tuning

Increased Adam weight decay from 0.02 to 0.090 (matching Muon WD). GPTQ calibration increased from 64 to 128 AR self-generated sequences for denser Hessian estimates with the larger SP4096 model. Dampening factor tuned to 0.01.

### 7. Brotli Compression (from LZMA)

SP4096 int6 weights compress better under Brotli than LZMA. This switch recovers the size headroom that BigramHash removal freed up.

### Dropped vs PR #1019

| Removed | Reason |
|---------|--------|
| BigramHash 3072x112 | Compresses 3.4x worse per param with SP4096, net size-negative |
| TrigramHash | Same compression issue with SP4096 |
| LZMA preset=9 | Brotli compresses SP4096 int6 weights better |
| TTT | Neutral or negative on this stack (25 failed attempts, [PR #756](https://github.com/openai/parameter-golf/pull/756)) |

---

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | **4x** (2048) with LeakyReLU(0.5)^2 | [#493](https://github.com/openai/parameter-golf/pull/493) @parinzee |
| Tokenizer | **SentencePiece 4096** | [#1334](https://github.com/openai/parameter-golf/pull/1334) |
| Attention | XSA on all 11 layers | [#478](https://github.com/openai/parameter-golf/pull/478) @gowtham0992 |
| Depth Recurrence | **Layers 4-5 from step 3000** | [#1260](https://github.com/openai/parameter-golf/pull/1260) |
| Parallel Residuals | **Layer 7+ with learned merge** | [#1289](https://github.com/openai/parameter-golf/pull/1289) |
| QK-Gain | **5.0** | [#1217](https://github.com/openai/parameter-golf/pull/1217) |
| Optimizer | Parallel Muon + **MuonEq-R** + Parameter Banking | [#399](https://github.com/openai/parameter-golf/pull/399), [#1334](https://github.com/openai/parameter-golf/pull/1334) |
| RoPE | Partial (16/64 dims) | [#315](https://github.com/openai/parameter-golf/pull/315) @jfprincz |
| LN Scale | 1/sqrt(layer+1) | [#315](https://github.com/openai/parameter-golf/pull/315) @jfprincz |
| VE128 | Layers 9-10 | [#374](https://github.com/openai/parameter-golf/pull/374) @unnir |
| SmearGate | Position-mixing gate | [#65](https://github.com/openai/parameter-golf/pull/65) @aquariouseworkman |
| U-Net skips | Encoder-decoder connections | [#289](https://github.com/openai/parameter-golf/pull/289) |
| Weight avg | EMA(0.997) + Tight SWA(every 50) | [#401](https://github.com/openai/parameter-golf/pull/401) @newjordan |
| Quantization | Full Hessian GPTQ int6 (AR self-gen, **128 batch**) | [#535](https://github.com/openai/parameter-golf/pull/535) @raahilshah |
| Compression | **Brotli** | **This work** |
| Warmdown | 4000 iterations | [#364](https://github.com/openai/parameter-golf/pull/364) @shikhar1729 |
| Late QAT | STE at LR scale < 0.15 | [#286](https://github.com/openai/parameter-golf/pull/286) @chris-buckley |
| Selective pruning | +/-1 values by reconstruction error | [#609](https://github.com/openai/parameter-golf/pull/609) @saml212 |
| Flash Attention 3 | Hopper warp-specialized kernels | [#122](https://github.com/openai/parameter-golf/pull/122) @mtybadger |

## Requirements

**Flash Attention 3 (Hopper) is required.**

```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece zstandard brotli
python3 -c "from flash_attn_interface import flash_attn_func; import sentencepiece, zstandard, brotli; print('deps OK')"
```

## Run Command

```bash
VOCAB_SIZE=4096 MLP_MULT=4.0 QK_GAIN_INIT=5.0 MUON_EQ_R=1 \
RECUR_LAYERS="4,5" RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
MUON_WD=0.090 ADAM_WD=0.090 WARMDOWN_ITERS=4000 \
GPTQ_CALIB_BATCHES=128 GPTQ_DAMP=0.01 \
BIGRAM_VOCAB_SIZE=0 TRIGRAM=0 TARGET_MB=15.9 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

```
PR #1019 (Merged SOTA, 1.1147 BPB) -- SP1024 + BigramHash + LZMA
    +-- This work replaces with:
        +-- SP4096 + MLP 4x (native vocabulary capacity, no bigram needed)
        +-- Depth recurrence layers 4-5 from step 3000 (from #1260)
        +-- Parallel residuals layer 7+ with learned merge (from #1289)
        +-- QK-Gain 5.0 (from #1217)
        +-- MuonEq-R optimizer (from #1334)
        +-- ADAM_WD=0.090, GPTQ 128-batch calibration, damp=0.01
        +-- Brotli compression (better for SP4096 int6)
        +-- Guided by 37 GPU runs (~$266) and PR #670 negative results
```
