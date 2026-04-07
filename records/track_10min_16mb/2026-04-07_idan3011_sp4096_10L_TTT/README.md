# sp4096 Custom Tokenizer + 10L 3.5x MLP + GPTQ + Score-First TTT

**val_bpb: 1.1266** | **artifact: 15.99 MB** | **8xH100** | **600s wallclock**

## Headline metrics

| Stage | val_bpb | val_loss |
|-------|--------:|---------:|
| Pre-quant (step 5952) | 1.1427 | 2.6289 |
| Post-quant (int6+brotli roundtrip) | 1.1439 | 2.6318 |
| Sliding window (stride=64) | 1.1277 | — |
| **Score-first TTT (final)** | **1.1266** | — |

## Architecture

- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA 2:1)
- 3.5x MLP expansion (1792 hidden dim per block)
- Tied input/output embeddings (single 4096×512 matrix)
- LeakyReLU(0.5)² MLP activation
- Logit softcap at 30 via tanh
- U-Net skip connections (encoder layers feed matching decoder layers via per-layer scale weights)
- Last 4 blocks use cross-sequence attention (XSA)
- 28.3M parameters total

## Custom sp4096 SentencePiece tokenizer

A 4096-vocab SentencePiece BPE tokenizer trained on FineWeb, hosted at `idan3011/parameter-golf-sp4096` on HuggingFace. The script auto-downloads the dataset + tokenizer on first run.

Compared to the default sp1024 tokenizer:
- **~26% fewer tokens per byte** (more efficient compression)
- More room in the param budget — fewer tokens means more "value" per parameter spent
- Required tuning the embedding quantization separately (see below)

## Training

- **786,432 token batch** (8 GPUs × 1 grad accum) — chosen over 524K for smoother warmdown trajectory
- **Muon optimizer** for matrix parameters in transformer blocks
- **Adam** for embeddings, output head, and scalar/vector parameters
- **Wallclock-fraction warmdown**: cosine LR decay over the last 35% of remaining wallclock (rather than fixed step count) to maximize useful training time
- **EMA** with decay 0.997 maintained throughout
- **SWA** averaging in the last 50% of training, blended with EMA at the end (weighted average of 198 checkpoints)
- **QAT** (fake-quantized weights during forward) on MLP CastedLinear layers to make the model more robust to int5 quantization
- Hit wallclock cap at step 5952/20000 in 600.054s
- Pre-quant val_loss: 2.6289, val_bpb: **1.1427**

## Quantization & compression

### GPTQ with AR self-generated calibration

Rather than calibrating GPTQ on a separate dataset, the trained model **generates its own calibration sequences** via autoregressive sampling. This produces 16 sequences of 512 tokens each, sampled from the model's own distribution — perfectly matched to its activation statistics.

GPTQ then uses Hessian-aware error compensation to quantize each weight column-by-column, propagating the rounding error to the remaining columns. This minimizes the L2 reconstruction error of the layer outputs.

### Mixed quantization scheme

| Tensor class | Bits | Reasoning |
|---|---|---|
| Attention weights (q/k/v/proj) | int5 per-row | Aggressive but stable with GPTQ |
| MLP weights (fc/proj) | int5 per-row | Stable with QAT during training |
| **tok_emb.weight (tied)** | **int8 per-row** | int5 destroys tied embedding (input AND output projection) — discovered painfully via experimentation. int8 yields near-zero quant gap. |
| Control tensors (scales, mixes, q_gain, skip_weights) | fp32 passthrough | Small total size, needed for stability |
| All other small tensors | fp16 passthrough | <65K elements |

Final post-quant gap is only **0.0012 BPB** (1.1427 → 1.1439) — exceptionally small for such aggressive quantization.

### brotli + byte-shuffle compression

After int5 quantization, weights are compressed with **brotli (quality 11)** instead of LZMA. To boost compression further:

- **Byte-shuffle pre-filter**: int8 quantized values are stored as little-endian int8s. Most values cluster near zero, meaning the high bytes are mostly zero/uniform. Reordering bytes column-wise (all-byte-0 then all-byte-1 then ...) groups the structure together and lets brotli's context modeling exploit it.

This combination saved ~280KB vs plain LZMA, and ~700KB vs naive int8 + zlib. Final artifact: **15,918,111 bytes** (model) + 71,265 (code) = **15,989,376 bytes total** — 10KB under the 16MB cap.

## Score-First TTT (eval-time adaptation)

Test-Time Training adapts the model on validation data **after scoring it**, exploiting val/train distribution shift. Strictly legal under issue #402 / issue #1017 — every token is scored before any weight update on it, single left-to-right pass.

### Algorithm

1. Build sliding-window scoring positions globally over the validation set (stride=64, seq_len=2048)
2. Group windows into chunks based on which `chunk_tok` block their **scored region** falls in
3. For each chunk:
   - **Score** the chunk's windows under `inference_mode` (forward only) — accumulate L (loss), T (token count), B (byte count)
   - If not the last chunk: **train** on the chunk's contiguous sequences via SGD + grad clipping, with **all_reduce** of gradients across GPUs
4. Final BPB = L / (B × ln 2)

### Hyperparameters

- 348 chunks of 131,072 tokens each
- 20 SGD epochs per chunk
- SGD lr=0.003 with cosine decay across chunks (chunk i uses lr × 0.5 × (1 + cos(π·i/N)))
- Momentum 0.9, no weight decay
- Grad clipping at 1.0
- Freeze: 0 blocks (all 10 layers trainable)

### Distributed implementation

- Windows split across ranks contiguously: `windows[rank*N/W:(rank+1)*N/W]`
- Each GPU scores its windows independently
- During training: each GPU processes its own sequence partition, then `dist.all_reduce(grad, AVG)` synchronizes gradients before optimizer step
- Final L/T/B all-reduced via SUM

### Result

Sliding baseline: **1.1277** → TTT: **1.1266** (-0.0011 BPB)

The improvement is small because the base model is already well-optimized (only 28M params, fully trained for 600s on FineWeb).

## Reproduce

```bash
pip install -r requirements.txt
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Every hyperparameter is baked into the script. Data and tokenizer auto-download from HuggingFace on first run. No env vars, no shell scripts, no setup steps.

Total runtime: ~10 min training + ~7 min eval (post-quant + sliding + TTT) on 8xH100.

## Included files

- `train_gpt.py` — frozen training script (1387 lines)
- `train.log` — full training + eval log
- `submission.json` — leaderboard metadata
- `README.md` — this file
