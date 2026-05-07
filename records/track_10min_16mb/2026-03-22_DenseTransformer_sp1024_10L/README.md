# Record Submission: 10L d=512 Int5-MLP + Int6-Attn + BigramHash + SmearGate

**Author:** LoquiAuris (@LoquiAuris)
**val_bpb:** 1.1508 (mean of 3 seeds, std=0.00012)
**Artifact size:** 15,680,288 bytes (15.68 MB)
**Training time:** ~10 minutes on 8×H100

## Results

| Seed | Post-quant val_bpb | Artifact (bytes) |
|------|-------------------|-----------------|
| 42   | 1.15097           | 15,680,288      |
| 1337 | 1.15077           | 15,654,632      |
| 2024 | 1.15074           | 15,639,761      |
| **Mean** | **1.15083 ±0.00012** | **—** |

## Approach

### Architecture

Standard PR #162 transformer stack with the following configuration:

- 10 layers, d_model=512, 8 attention heads, 4 KV heads (GQA)
- 3× FFN expansion (hidden=1536) with ReLU² activation
- SmearGate: learned blend with previous token representation
- BigramHash: 4096 buckets, dim=128, projected to 512
- U-Net skip connections between symmetric layer pairs
- RMSNorm, logit softcap=30.0, orthogonal initialization
- RoPE positional encoding (persistent=False)
- Tied embeddings via F.linear(x, tok_emb.weight)
- Vocabulary: sp1024 (1,024 BPE tokens)

### Training

- Optimizer: Muon (matrix_lr=0.02, momentum=0.99 with warmup from 0.92 over 1500 steps) + AdamW for embeddings and scalars
- Weight decay: 0.04 (Muon), 0.01 (AdamW)
- Gradient clipping: 0.3
- Sequence length: 2048
- Batch size: 786,432 tokens
- Warmup: 20 steps
- Warmdown: 3000 iterations (cosine schedule)
- SWA: start_frac=0.5, checkpoint every 50 steps, 29 checkpoints averaged
- Steps completed: ~7,600 in 10 minutes

### Quantization & Compression

- MLP weights: Int5 per-row symmetric (clip=15)
- Attention weights: Int6 per-row symmetric (clip=31)
- Embeddings: FP16 passthrough
- Norms, gates, control tensors: FP16 passthrough
- Compression: zstd level 22

### Evaluation

- Sliding window with stride=64, seq_len=2048

## Key Finding: Int6 Embedding Quantization

During development, we explored using sp8192 (8,192-token vocabulary) to improve tokenizer efficiency. The sp8192 tokenizer encodes at 3.79 bytes/token vs sp1024's 2.44 — a 55% improvement that directly reduces bits-per-byte.

The challenge: sp8192's embedding table at d=512 costs 8.39 MB in FP16, consuming over half the 16 MB budget and limiting the model to 6-8 layers.

We discovered that embedding tables can be quantized to Int6 (6-bit per-row symmetric) with negligible quality loss:

| Embed quantization | val_bpb | Penalty vs FP16 |
|-------------------|---------|-----------------|
| FP16 (baseline)   | 2.2352  | —               |
| Int8              | 2.2354  | +0.0002         |
| Int6              | 2.2357  | +0.0005         |

A penalty of +0.0005 bpb is within noise. This enabled sp8192 at d=512 — a combination previously considered impossible under the 16 MB constraint.

### sp8192 + Int6 Embed Results (H100)

| Config | Post-quant bpb | Artifact | Headroom |
|--------|---------------|----------|----------|
| sp8192 d=512 6L Int6-embed | 1.2010 | 11.97 MB | 4.0 MB |
| sp8192 d=512 7L Int6-embed | 1.1863 | 13.57 MB | 2.4 MB |
| sp8192 d=512 8L Int6-embed | 1.1794 | 14.99 MB | 1.0 MB |
| sp8192 d=384 9L FP16-embed | 1.1889 | 12.63 MB | 3.4 MB |
| sp1024 d=512 10L (this submission) | 1.1510 | 15.68 MB | 0.3 MB |

Despite the tokenizer efficiency advantage, sp1024 with 10 layers at full d=512 width outperformed all sp8192 configurations. The layer count advantage (10L vs 6-8L) at d=512 exceeds the tokenizer efficiency gain on H100 with full training.

However, the Int6 embedding finding remains significant: it enables large-vocabulary models within severe artifact constraints and may prove valuable as quantization techniques improve and more layers become feasible at larger vocab sizes.

## Development Process

This submission was developed through systematic architecture search:

1. **Tokenizer exploration:** Tested sp1024, sp2048, sp4096, sp8192 — identified the embedding size vs model capacity trade-off as the key constraint
2. **Width vs depth analysis:** Confirmed d=512 (width) > d=384/448 (narrower + deeper) across all tokenizer sizes at this parameter budget
3. **Int6 embedding discovery:** Found that embedding quantization to 6-bit has negligible quality impact (+0.0005 bpb), unlocking large vocabularies at full model width
4. **8 H100 configurations tested** across 2 pod sessions, plus extensive local testing on Apple Silicon (500-step ablations)
5. **Final result:** sp1024 d=512 10L produces the best bpb by maximizing layer count at full width within the 16 MB budget

### Local Testing Methodology

All architecture decisions were validated through 500-step local runs on Apple Silicon (MPS backend) using AdamW, then confirmed on 8×H100 with the full Muon + SWA + PR #162 stack. Local-to-H100 scaling ratio was approximately 1.85-1.95×.

### Hardware & Cost

- Training: 8×H100 SXM (RunPod)
- Local testing: Apple Silicon (MPS)
- Total H100 time: ~2.5 hours across 2 pod sessions
- Estimated cost: ~$65 in RunPod credits

## Files

- `train_gpt.py` — Complete training script with environment variable configuration
- `train.log` — Training log from seed 42 (primary submission)
- `submission.json` — Submission metadata
- `README.md` — This file
