# FreqGPTQ + GatedDeltaNet + Adaptive Quantization

**val_bpb: TBD** (pending GPU validation) | **~15.8 MB** | 8xH100 SXM

## Approach

Built on PR #1698 (GatedDeltaNet + Legal TTT, 1.00995 BPB) with quantization and compression improvements:

### 1. FreqGPTQ (frequency-weighted GPTQ calibration)
Top-100 most frequent tokens get `sqrt(2)` boosted activations during Hessian accumulation (`H = X^T X`), biasing quantization error minimization toward high-frequency tokens covering ~53% of text. Zero artifact cost.

### 2. PassthroughQuant
Control tensors (`attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`) quantized to per-tensor int8 instead of fp16 passthrough. Small 2D matrices also get per-row int8. Saves ~40KB compressed.

### 3. Sandwich Quantization
Final transformer block quantized to int8 instead of int6 to protect signal quality before the tied LM head.

### 4. Adaptive Embedding Precision
Int8 for top-100 frequent token embedding rows, int6/int5 for the rest. Higher precision where it matters most (Zipf's law).

### 5. Configurable Int5 GPTQ
Weight quantization pushed from 6-bit (`clip_range=31`) to 5-bit (`clip_range=15`) with FreqGPTQ, fitting ~38M params vs ~32M at int6 within the same 16MB budget. Late QAT clip range synced to match.

### 6. LZMA Self-Extracting Code Wrapper
Python source compressed from ~105KB to ~30KB via LZMA + base85 encoding. Frees ~73KB for model weights (~118K more parameters at int5).

## Architecture

Same as PR #1698 (Model K: K_KVShare_Wider):
- 10 GatedDeltaNet layers (FLA `fla-core==0.4.2`)
- 544 model dim, 8 heads, 64-dim head keys
- KV sharing stride 2
- 3x MLP, BigramHash embedding, SmearGate
- Tied embeddings, logit softcap 30.0

## Training

- Muon optimizer + EMA(0.997) + SWA(50)
- Late QAT with STE (threshold 0.15, clip range matches WEIGHT_BITS)
- Score-first TTT: SGD(lr=0.005, momentum=0.9), 3 epochs, 32K chunks, freeze first 2 blocks

## Run Command

```bash
ARCH_MODE=K GPTQ_ENABLED=1 TTT_ENABLED=1 WEIGHT_BITS=6 \
FREQ_GPTQ_BOOST=2.0 ADAPTIVE_EMBED=1 NUM_FREQ_TOKENS=100 \
TTT_LR=0.005 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=2 \
TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Status

WIP — code complete, pending GPU validation for BPB results and 3-seed statistical significance.

## Attribution

- GatedDeltaNet architecture: @resouer (PR #1687), @arsenis-cmd (PR #1698)
- Flash Linear Attention: @sustcsonglin (fla-core 0.4.2)
- Legal TTT framework: @Christopher-Lee-McClendon (PR #461)
- FreqGPTQ concept: PR #1707
- PassthroughQuant concept: PR #1716
- AttnOutGate + SmearGate: PR #1693
