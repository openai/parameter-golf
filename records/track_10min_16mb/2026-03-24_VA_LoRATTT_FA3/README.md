# 10L LoRA TTT 6ep + FlashAttention-3

**val_bpb: 0.7227** | **artifact: 15.45 MB** | **8xH100 SXM, 600s train + 569s eval**

## Architecture

- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA), MLP 3x
- ReLU-squared activation, tied embeddings, logit softcap 30.0
- SmearGate, BigramHash(2048, dim=128)
- U-Net skip connections (encoder/decoder split)
- RoPE with train_seq_len=1024

## Training

- Muon optimizer (Newton-Schulz, compiled separately) for matrices
- AdamW for embeddings/scalars, fused=True
- MATRIX_LR=0.025, warmdown 6000 steps
- EMA (decay=0.999, every 10 steps) + SWA (11 checkpoints)
- Late QAT during warmdown
- 7274 steps in 600s at 82.5ms/step on 8xH100 SXM
- Pre-quantization BPB: 1.1621

## Quantization

- Int6 uniform per-row quantization + zstd-22 compression
- FP16 passthrough for embeddings + control tensors
- Post-quantization BPB: 1.1750 (quant gap: 0.013)
- Artifact: 15,445,637 bytes (96.5% of 16MB limit)

## Test-Time Training (LoRA TTT)

Per-document low-rank adaptation — the key technique driving 0.45 BPB improvement.

### LoRA Architecture
- Rank-8 LoRA on Q and V projections (all layers)
- Rank-16 LoRA on LM-head (doubled capacity for output adaptation)
- Per-block learned bias vectors during TTT (cheap domain shift)
- BatchedLinearLoRA: per-batch-element LoRA (`Delta = x @ A^T @ B^T`)

### Per-Layer Learning Rates
- LM-head LoRA: 2x base LR (0.02)
- V LoRA: 1.5x base LR (0.015)
- Q LoRA: 0.5x base LR (0.005)
- Bias params: 3x base LR (0.03)
- Base LR: 0.01 (Adam optimizer)

### Per-Document Adaptation
- Documents segmented at BOS token boundaries
- Short docs (<1024 tokens): scored without TTT (2393 docs, 27s)
- Long docs: sorted by chunk count, processed in batches of 64 docs/GPU
- Fresh LoRA reset per document batch (prevents cross-contamination)
- 6 epochs per document batch with per-step cosine LR decay
- Score-every-epoch: overwrite scores each epoch (backward-looking, Issue #402 compliant)

### TTT Timing
- Short docs: 27s (base model scoring only)
- Long docs: 377s (61 batches of 64 docs across 8 GPUs)
- Post-TTT temperature rescaling: T=0.98
- Wall-clock deadline: 550s with base-model fallback for remaining docs
- Total eval: 569s (within 600s budget)

## Hardware Optimizations

- **FlashAttention-3**: `flash_attn_func` for causal attention (3% faster than SDPA)
- **Compiled Muon**: `zeropower_via_newtonschulz5 = torch.compile(...)` (4% faster)
- **train_seq_len=1024**: halves O(n^2) attention cost (12% faster vs 2048)
- **TF32 enabled**: `torch.backends.cuda.matmul.allow_tf32 = True`
- **Flash SDP forced**: `enable_flash_sdp(True); enable_math_sdp(False)`
- **Rotary cache .clone() fix**: prevents CUDA graph conflict with FA3
- Combined: 82.5ms/step (28% faster than our baseline 106ms/step)

## Results

- Training: 7274 steps at 82.5ms/step
- Pre-quant BPB: 1.1621
- Post-quant BPB: 1.1750
- **Post-TTT BPB: 0.7227**
- TTT gain: 0.4522 BPB
- Artifact: 15.45 MB
- Total wall time: 21.5 min (10 min train + 11.5 min eval)

## Based On

PR #596 (DeepQuant V10b by AriaAnima, 0.6430 BPB) with our additions:
- FlashAttention-3 integration
- Rotary cache compatibility fix

## Experiment History

38 experiments across steps 4-8, detailed in experiments.md:
- Steps 4-5: Architecture exploration (VR, GA, XSA4, SWA, EMA, GPTQ)
- Step 6: TTT rewrite (batched, per-step cosine, per-layer LR, grad sync fix)
- Step 7: TrigramHash, 11L attempts, GPTQ calibration fixes
- Step 8: Hardware optimization (FA3, compiled Muon, seq=1024), Variant B (int4/int5), Variant A (LoRA TTT)
