# Record: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT

**val_bpb (TTT) = 1.08083** (3-seed mean, std 0.00062) | **~15.97 MB** artifact | 8xH100 SXM

Current SOTA PR #1413: **1.0810 BPB** (std 0.0002). Our mean lands 0.00017 below that, within our own std.

## 3-Seed Results

| Seed | Pre-Quant | Sliding | **TTT** | Artifact |
|------|-----------|---------|---------|----------|
| 1337 | 1.08609 | 1.08083 | **1.08032** | 15,971,929 |
| 42   | 1.08733 | 1.08214 | **1.08152** | 15,973,790 |
| 7    | 1.08635 | 1.08135 | **1.08064** | 15,971,863 |
| **Mean** | **1.08659** | **1.08144** | **1.08083** | **15,972,527** |
| **Std**  | 0.00064 | 0.00066 | **0.00062** | |

## Key Techniques

1. **SP8192 + GPTQ SDClip:** int6 matrices (k=12.85), int8 embeddings (k=20.0), zero pruning (PR #1394 @clarkkev)
2. **3-Layer Depth Recurrence:** loops layers 3,4,5 twice, activates at frac=0.35 (PR #1331, #1437 @dexhunter)
3. **Parallel Residuals** (layers 7+): GPT-J style, attention and MLP read same input (PR #1412 @Robby955, PR #1204 @msisovic)
4. **QK-Gain 5.25:** learnable per-head query scaling (PR #1413 @dexhunter)
5. **Legal Score-First TTT:** SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk, freeze first 9 blocks, cross-rank `dist.all_reduce` on gradients, cosine LR decay (PR #549 @abaybektursun, PR #1413 @dexhunter)
6. **Tuned Hyperparameters:** MUON_WD=0.095, MATRIX_LR=0.022, EMA=0.9965, WARMDOWN_FRAC=0.72 (PR #1445 @X-Abhishek-X)
7. **FA3/SDPA backend switch:** `USE_FA3=1` uses `flash_attn_3_func` on Hopper; `USE_FA3=0` falls back to `F.scaled_dot_product_attention(..., enable_gqa=True)` for non-Hopper GPUs.

## Architecture

11L × 512d, 8 heads / 4 KV heads, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64 head dims), layerwise LN scale, tied embeddings, logit softcap 30.0.
Depth recurrence: encoder `[0,1,2,3,4,5,3,4]`, decoder `[5,3,4,5,6,7,8,9,10]`, loops layers 3-5 (three times total), activated at ~step 2016.
Parallel residuals from layer 7. Skip gates (sigmoid-gated U-Net connections).

## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps) for matrices; AdamW for embeddings and scalars. **4550 steps in 588s** on 8×H100 SXM. Linear warmdown to LR=0 over the final 72% of training. EMA decay 0.9965. Data: LightSpeedUp public SP8192 pre-tokenized FineWeb mirror.

## Quantization

Full-Hessian GPTQ with SDClip (`clip = k × std(row)`): int6 for attention/MLP matrices (k=12.85), int8 for token embeddings (k=20.0). 64 calibration batches. Byte-shuffle + Brotli-11 compression. Zero pruning needed; model fits natively at ~15.97 MB.

## TTT (Test-Time Training)

Score-first, chunk-based SGD adaptation at eval time:

- Val tokens divided into 32K-token chunks (~1240 chunks over full val)
- For each chunk: (1) score all sliding windows under `torch.no_grad()`, (2) SGD on chunk tokens with frozen first 9 blocks
- 3 epochs per chunk, cosine LR decay across chunks
- Gradient clip 1.0, cross-rank `dist.all_reduce` on gradients before every optimizer step
- Total TTT eval time ~420s per seed (within 600s eval budget)

## Compliance (Issue #1017, Track B legal eval-time adaptation)

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Each position scored from prefix tokens only.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab. No n-gram cache, no logit biasing.
- **Condition 3 (Score before update):** Each chunk fully scored under `torch.no_grad()` BEFORE any SGD update. Training only on already-scored tokens.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring, no multi-pass selection.

Additional:

- No SLOT (standard or causal)
- No pre-quant TTT on val data (model quantized once during training, TTT adapts at eval time)
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- All 3 seeds' artifacts under 16,000,000 bytes
- Training ≤ 600s on all 3 seeds (~588s actual)
- Eval (sliding + TTT) ≤ 600s on all 3 seeds (~545s actual)

## Reproduction

```bash
pip install brotli sentencepiece
# Hopper only (FA3). On non-Hopper GPUs skip this line and set USE_FA3=0 below.
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Download LSU SP8192 pre-tokenized FineWeb into the expected layout:
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('LightSpeedUp/parameter-golf-data', repo_type='dataset', local_dir='./data_lsu', allow_patterns=['fineweb_sp8192/*.bin','tokenizers/fineweb_8192_bpe.*'])"
mkdir -p data/datasets data/tokenizers
ln -s "$(pwd)/data_lsu/fineweb_sp8192"                   data/datasets/fineweb10B_sp8192
ln -s "$(pwd)/data_lsu/tokenizers/fineweb_8192_bpe.model" data/tokenizers/fineweb_8192_bpe.model

SEED=42 QK_GAIN_INIT=5.25 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
PARALLEL_RESIDUAL_START=7 MATRIX_LR=0.022 MUON_WD=0.095 EMBED_WD=0.095 \
EMA_DECAY=0.9965 WARMDOWN_FRAC=0.72 TTT_ENABLED=1 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@clarkkev:** SP8192 + GPTQ Embeddings + SDClip + MuonEq-R + U-Net skips (PR #1394)
- **@dexhunter:** 3-layer depth recurrence (PR #1331, #1437), Legal TTT on SP8192 (PR #1413)
- **@abaybektursun:** Score-first TTT framework (PR #549)
- **@Robby955:** Parallel residuals on SP8192 (PR #1412)
- **@msisovic:** Parallel residuals concept (PR #1204)
- **@X-Abhishek-X:** Hyperparameter tuning WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 (PR #1445)
- **@LightSpeedUp:** Public SP8192 pre-tokenized FineWeb mirror (HuggingFace)
