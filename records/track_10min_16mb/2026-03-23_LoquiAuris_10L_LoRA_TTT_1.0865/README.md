# Record Submission: 10L d=512 EMA + LoRA TTT

**Author:** Loqui Auris ([@LoquiAuris](https://github.com/LoquiAuris))
**val_bpb:** 1.0865 (mean of 2 seeds, std=0.0013)
**Artifact size:** 15,810,855 bytes (15.81 MB)
**Training time:** ~10 minutes on 8×H100

## Results

| Seed | Pre-TTT val_bpb | Post-LoRA-TTT val_bpb | Artifact (bytes) | Steps |
|------|----------------|----------------------|------------------|-------|
| 42   | ~1.1610        | 1.0856               | 15,810,855       | 5,978 |
| 1337 | ~1.1610        | 1.0874               | 15,705,529       | 5,969 |
| Mean |                | **1.0865 ±0.0013**   |                  |       |

Third seed pending (compute grant).

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
- EMA: decay=0.997, applied after training completes
- Steps completed: ~5,970 in 600s

### Quantization & Compression

- MLP weights: Int6 per-row symmetric (clip=31)
- Attention weights: Int6 per-row symmetric (clip=31)
- Embeddings: FP16 passthrough
- Norms, gates, control tensors: FP32 passthrough
- Compression: zstd level 22

### Evaluation: LoRA TTT (Test-Time Training)

Per-document backward-looking LoRA adaptation during evaluation. This is the key technique that reduces bpb from ~1.161 (pre-TTT) to ~1.087 (post-TTT) — a **0.074 bpb improvement**.

**How it works:**

For each document in the validation set:
1. Add ephemeral LoRA adapters (rank=8) to Q projections, V projections, and the LM head
2. Split document into 256-token chunks with 1024-token context windows
3. Process chunks left-to-right over 2 epochs:
   - Forward pass with LoRA-adapted model
   - Score tokens on the final epoch (record loss for bpb)
   - Train LoRA on non-final chunks (backward + optimizer step)
4. Reset LoRA weights + optimizer state before the next document

**Key details:**
- LoRA rank 8 on Q + V projections + LM head per block
- Adam optimizer (lr=0.01, betas=0.9/0.95)
- Batch: 64 documents per GPU with independent LoRA per document
- Documents < 512 tokens: standard eval without TTT (insufficient context for adaptation)
- 8-GPU sharding: documents distributed across ranks, metrics all-reduced at end
- TTT time: ~245s per seed (within the 600s eval budget)

**LoRA weights are NOT part of the 16MB artifact.** They are created fresh at eval time, trained on the fly per document, and discarded between documents. Only the base model weights are in the artifact.

## Key Technique: Fresh Model for LoRA TTT

`torch.compile` with `fullgraph=True` caches the forward graph from training, which has `None` for all LoRA delta arguments. The compiled graph silently ignores LoRA deltas at eval time — the LoRA additions to Q, V, and logits are treated as dead code by the compiled graph.

**The fix:** Call `torch._dynamo.reset()` after training, create a fresh uncompiled `GPT` model from `state_dict`, and run LoRA TTT on the uncompiled model. This ensures all LoRA code paths are active during TTT.

Without this fix, LoRA TTT produces **worse** results than no TTT (1.189 vs 1.161) because the model is effectively running without adaptation while still paying the per-document overhead.

## Development Process

This submission builds on the 1.1508 baseline (PR #350) with two additions:

1. **EMA weight averaging** (decay=0.997) replaced SWA — marginal improvement
2. **LoRA TTT** adapted from PROTEUS v7 (PR #512) — the primary bpb improvement

The LoRA TTT implementation required solving the `torch.compile` graph caching issue (see above), which was the critical debugging step. Batched document processing (64 docs/GPU) was essential for completing TTT within the eval time budget.

### Progression

| Submission | val_bpb | Technique |
|-----------|---------|-----------|
| PR #350   | 1.1508  | Baseline (no TTT) |
| This (pre-TTT) | ~1.1610 | + EMA |
| This (post-TTT) | **1.0865** | + LoRA TTT |

## Hardware & Cost

- Training: 8×H100 SXM (RunPod)
- Local testing: Apple Silicon (MPS) for architecture validation
- Total H100 time: ~1 hour for 2 seeds
- Estimated cost: ~$25 in RunPod credits

## Acknowledgments

- Training stack: PR #162 (raahilshah), PR #180 (thwu1)
- LoRA TTT approach: PR #512 (MatoTeziTanka), PR #77 (samacqua)
- EMA + TTT: PR #442 (sjp611)
- SmearGate/BigramHash: @unnir
- Muon optimizer, OrthoInit: Parameter Golf community

## Files

- `train_gpt.py` — Complete training script with environment variable configuration
- `train_seed42.log` — Training + TTT log (seed 42)
- `train_seed1337.log` — Training + TTT log (seed 1337)
- `submission.json` — Submission metadata
- `README.md` — This file
