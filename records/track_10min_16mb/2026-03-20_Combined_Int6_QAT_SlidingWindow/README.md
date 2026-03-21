# V5: TTT LoRA + Dual Bigram + Label Smoothing + Z-Loss

## Strategy
thwu1's winning 1.1428 bpb architecture + 4 novel innovations that no competitor uses.

## Architecture (from thwu1)
- **10 transformer layers**, GQA (8 heads, 4 KV heads), model_dim=512
- **3x MLP** (hidden=1536), ReLU^2 activation
- **Learned BigramHashEmbedding** (10240 entries, 128-dim, XOR hash)
- **SmearGate**: learned gate blending each token with previous
- **U-Net skip connections** with learned blending weights
- **Tied embeddings** (FP16 passthrough for tok_emb + last layer c_k)

## Training (from thwu1)
- **Muon optimizer**: momentum=0.99, weight_decay=0.04, Newton-Schulz
- **Linear warmdown** (wallclock-aware, 3000 iters)
- **SWA** (last 40%, every 50 steps, accumulated on CPU)
- **Low LRs**: matrix=0.02, scalar=0.02, tied_embed=0.03
- Train at seq_len=2048, batch=786,432 tokens

## Quantization & Compression (from thwu1)
- **Mixed Int5/Int6**: Int5 (clip=15) for MLP, Int6 (clip=31) for attention
- **3% magnitude pruning** (zeros compress well)
- **zstd level 22** compression

## Novel Innovations (V5)

### 1. TTT LoRA at Eval (biggest gain: -0.010 to -0.025 bpb)
Per-document LoRA adapters (rank=8) on Q, V projections and LM head.
Each document gets its own adapter trained chunk-by-chunk (256 tokens).
Adapters reset between documents. Uses Adam (lr=0.01).
**No competitor in the top 5 uses TTT.**

### 2. Dual Bigram inside TTT (-0.001 to -0.003 bpb)
Post-hoc statistical [1024,1024] bigram residual table added to logits
**inside** the TTT eval loop. TTT adapts with bigram context.

### 3. Label Smoothing (0.05) (-0.001 to -0.003 bpb)
Prevents overconfident predictions, improves calibration and
quantization robustness.

### 4. Z-Loss Regularization (1e-4) (-0.001 to -0.002 bpb)
Penalizes large logit magnitudes: `z_loss = 1e-4 * mean(logsumexp(logits)^2)`.
Directly improves post-quantization performance.

## Reproduction
```bash
pip install zstandard
cd /workspace/parameter-golf
python data/cached_challenge_fineweb.py
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_Combined_Int6_QAT_SlidingWindow/train_gpt.py
```

## Fallback
If TTT eval exceeds time budget: `TTT_ENABLED=0` falls back to sliding window eval.

## Expected Results
Target: ~1.115-1.135 bpb (beat thwu1's 1.1428 by 0.008-0.028)
