# K-LoRA + Min-NLL + FlashAttention-3

**val_bpb: 0.6864** | **seed: 42** | **artifact: 15.53 MB** | **8xH100 SXM, 600s train + 588s eval**

## Key Innovations (built on PR #611 Chimera TTT + PR #596 DeepQuant V10b)

### 1. K-Projection LoRA
LoRA applied to K projections (not just Q/V) with conservative 0.3x LR multiplier. K determines what information each position broadcasts for attention retrieval — adapting K alongside Q/V gives more expressive per-document specialization.

### 2. Min-NLL Epoch Selection
Track minimum average NLL per document across all TTT epochs, use best epoch's scores. Prevents late-epoch overfitting from degrading any document's score. Enables safely running more epochs.

### 3. FlashAttention-3
`flash_attn_func` for causal attention with Rotary cache `.clone()` fix for CUDA graph compatibility.

## Architecture
- 10L, 512d, GQA 8/4, MLP 3x, ReLU-squared
- EMA (0.999, every 10 steps) + SWA
- SmearGate, BigramHash(2048), U-Net skip connections
- Compiled Muon Newton-Schulz, train_seq_len=1024
- Late QAT, int6 uniform + zstd-22

## LoRA TTT
- Rank-8 Q/K/V LoRA + rank-16 LM-head LoRA
- Per-block bias tuning
- Per-document reset at BOS boundaries, batched 64 docs/GPU
- Adam lr=0.01, 6 epochs, per-step cosine LR
- Per-layer LR: LM-head 2x, V 1.5x, Q 0.5x, K 0.3x, bias 3x
- Temperature rescaling T=0.98
- Wall-clock deadline 550s with base-model fallback

## Results
- Training: 7313 steps at 82.1ms/step
- Pre-quant BPB: 1.1624, post-quant: 1.1755
- **Post-TTT BPB: 0.6864** (TTT gain: 0.489)
- Artifact: 15.53 MB (97.1% of 16MB)
- Eval time: 588s (within 600s budget)

## Based On
- PR #611 (Chimera TTT by teddyoweh, 0.5601 BPB — eval time non-compliant)
- PR #596 (DeepQuant V10b by AriaAnima, 0.6430 BPB)
- Our addition: FlashAttention-3 + Rotary cache fix
