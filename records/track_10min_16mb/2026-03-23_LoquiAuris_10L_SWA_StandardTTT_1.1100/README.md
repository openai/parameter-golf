# Record Submission: 10L d=512 SWA + Standard AdamW TTT

**Author:** Loqui Auris ([@LoquiAuris](https://github.com/LoquiAuris))
**val_bpb:** 1.1100 (seed 1337)
**Artifact size:** 15,750,007 bytes (15.75 MB)
**Training time:** ~10 minutes on 8×H100

## Results

| Seed | Post-quant val_bpb | Post-TTT val_bpb | Artifact (bytes) | Steps |
|------|-------------------|-----------------|------------------|-------|
| 1337 | ~1.1510           | **1.1100**      | 15,750,007       | 5,992 |

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
- Tied embeddings via `F.linear(x, tok_emb.weight)`
- Vocabulary: sp1024 (1,024 BPE tokens)
- ~24.7M parameters

### Training

- Optimizer: Muon (matrix_lr=0.02, momentum=0.99 with warmup from 0.92 over 1500 steps) + AdamW for embeddings and scalars
- Weight decay: 0.04 (Muon), 0.01 (AdamW)
- Gradient clipping: 0.3
- Sequence length: 2048
- Batch size: 786,432 tokens
- Warmup: 20 steps
- Warmdown: 3000 iterations (wallclock-based cosine schedule)
- SWA: start_frac=0.5, checkpoint every 50 steps, **29 checkpoints averaged**
- Steps completed: ~5,992 in 600s

### Quantization & Compression

- MLP weights: Int5 per-row symmetric (clip=15)
- Attention weights: Int6 per-row symmetric (clip=31)
- Embeddings: FP16 passthrough
- Norms, gates, control tensors: FP16/FP32 passthrough
- Compression: zstd level 22
- Artifact: 15.69 MB (250KB headroom)

### Evaluation: Standard AdamW TTT

Global fine-tuning on the validation data after training, applied to the quantized-then-dequantized model weights.

**How it works:**

After training completes and SWA weights are applied:
1. Quantize model to Int5/Int6
2. Save compressed artifact
3. Load artifact, dequantize back to float
4. Fine-tune ALL model parameters on validation data using AdamW
5. Evaluate with sliding window (stride=64, seq_len=2048)

**Key details:**
- AdamW optimizer (lr=0.0005, no weight decay)
- 10 epochs over the full validation set
- Gradient clipping: 1.0
- Distributed: gradients all-reduced across 8 GPUs
- TTT improvement: ~0.041 bpb (1.1510 → 1.1100)

Standard TTT modifies the base model weights directly (unlike LoRA TTT which uses ephemeral adapters). The fine-tuned weights are NOT saved — only the original quantized artifact is the submission. TTT is applied at eval time.

### Comparison: Standard TTT vs LoRA TTT

| Method | Pre-TTT | Post-TTT | Improvement | Time |
|--------|---------|----------|-------------|------|
| Standard AdamW TTT | 1.1510 | 1.1100 | -0.041 | ~120s |
| LoRA TTT (PR #548) | 1.1610 | 1.0865 | -0.074 | ~245s |

LoRA TTT provides a larger improvement (-0.074 vs -0.041) because it adapts per-document rather than globally. Standard TTT is faster but less effective.

## Development Process

This submission builds on the 1.1508 baseline (PR #350) with the addition of standard AdamW TTT:

1. **Baseline** (PR #350): 1.1508 bpb — no TTT, SWA only
2. **+ Standard TTT**: 1.1100 bpb — AdamW fine-tuning on val data (this submission)
3. **+ LoRA TTT** (PR #548): 1.0865 bpb — per-document LoRA adaptation

The standard TTT approach was implemented based on PR #77 (samacqua) and PR #442 (sjp611). AdamW was chosen over SGD based on published ablations showing -0.019 bpb advantage.

## Hardware & Cost

- Training: 8×H100 SXM (RunPod)
- Local testing: Apple Silicon (MPS) for architecture validation
- Total H100 time: ~30 minutes (single seed)
- Estimated cost: ~$12 in RunPod credits

## Acknowledgments

- Training stack: PR #162 (raahilshah), PR #180 (thwu1)
- Standard TTT approach: PR #77 (samacqua), PR #442 (sjp611)
- SmearGate/BigramHash: @unnir
- Muon optimizer, SWA, OrthoInit: Parameter Golf community

## Files

- `train_gpt.py` — Complete training script with environment variable configuration
- `submission.json` — Submission metadata
- `README.md` — This file
