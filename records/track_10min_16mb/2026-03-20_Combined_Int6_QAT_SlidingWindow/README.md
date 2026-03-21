# V4: BigramHash + SWA + Int5-MLP + SmearGate

## Strategy
Full rewrite based on thwu1's winning 1.1428 bpb architecture, plus novel dual-bigram innovation.

## Architecture (from thwu1)
- **10 transformer layers**, GQA (8 heads, 4 KV heads), model_dim=512
- **3x MLP** (hidden=1536), SiLU activation
- **Learned BigramHashEmbedding** (10240 entries, 128-dim, XOR hash → project to model_dim)
- **SmearGate**: learned gate blending each token with previous token's embedding
- **Tied embeddings** (FP16 passthrough for tok_emb + last layer c_k)
- **Train at seq_len=2048**, batch=786,432 tokens

## Training (from thwu1)
- **Muon optimizer**: momentum=0.99, weight_decay=0.04, Newton-Schulz orthogonalization
- **AdamW** for embeddings + scalar params (weight_decay=0.01)
- **Linear warmdown** (wallclock-aware, 3000 iters)
- **Low LRs**: matrix=0.02, scalar=0.02, tied_embed=0.03
- **Grad clipping**: norm=0.3
- **SWA** (Stochastic Weight Averaging): last 40% of training, every 50 steps, accumulated on CPU

## Quantization & Compression (from thwu1)
- **Mixed Int5/Int6**: Int5 (clip=15) for MLP weights, Int6 (clip=31) for attention weights
- **Magnitude pruning**: zero out smallest 3% of weights (zeros compress well)
- **zstd level 22** compression (with zlib fallback)
- **FP16 passthrough**: tok_emb + last layer key weights

## Novel Contributions
1. **Dual bigram**: Learned BigramHash during training + post-hoc full [1024,1024] statistical bigram residual table at eval time (scale=0.3)
2. **Compiled forward_logits**: `torch.compile` on eval forward pass for faster sliding window
3. **PyTorch 2.4 GQA fallback**: `repeat_interleave` when `enable_gqa` not available

## Reproduction
```bash
pip install zstandard
cd /workspace/parameter-golf
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-20_Combined_Int6_QAT_SlidingWindow/train_gpt.py
```

## Expected Results
Target: < 1.1428 bpb (beat thwu1's #1 score)
