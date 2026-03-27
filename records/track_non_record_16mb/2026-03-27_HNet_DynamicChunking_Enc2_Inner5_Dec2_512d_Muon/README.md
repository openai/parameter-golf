# H-Net with Dynamic Sequence Chunking

**Non-Record Submission (Research Contribution) | First H-Net Architecture in Parameter Golf**
**Author:** Tim Shen ([@TimS-ml](https://github.com/TimS-ml))
**Hardware:** 1x RTX 4090 (24 GB)
**Architecture:** H-Net (Enc=2, Inner=5, Dec=2) at 512d, 17.5M params

---

## Summary

This submission introduces **H-Net (Hierarchical Network) with Dynamic Sequence Chunking** to Parameter Golf -- the first submission to use learned hierarchical sequence compression. Instead of processing all 1024 tokens through every layer, H-Net learns to detect natural boundaries in the token sequence, downsamples to chunk representatives, processes the compressed sequence through a deeper inner transformer, then upsamples back to full resolution.

The core idea: not all tokens need the same compute. Boundary tokens (e.g., sentence starts, topic shifts) carry more structural information and deserve deeper processing, while interior tokens can be reconstructed from their chunk context.

**Key metric: val_bpb = `TODO` (pending full training run)**

---

## Architecture

### H-Net Overview

```
Input tokens (B, 1024)
    |
    v
[Embedding + RMSNorm]
    |
    v
[Encoder: 2x TransformerBlock]     -- full sequence (1024 tokens)
    |
    v
[DynamicSequenceChunker]            -- learns boundaries, downsamples to ~170 chunks
    |                                   (target_avg_token_length=6.0)
    v
[Inner: 5x TransformerBlock]        -- compressed sequence (~170 tokens)
    |
    v
[Upsample via associative scan]     -- back to 1024 tokens
    |
    v
[Decoder: 2x TransformerBlock]      -- full sequence (1024 tokens)
    |
    v
[LM Head + softcap]
```

### Dynamic Sequence Chunking (from H-Net paper)

The chunker learns to detect token boundaries using cosine distance between adjacent token representations projected to a query-key space:

1. **Boundary detection**: Project tokens to queries and keys (`dim_qk=128`). Compute `prob = (1 - cos_sim(q_i, k_{i-1})) / 2`. Tokens with `prob > 0.5` are boundaries.
2. **Downsampling**: Extract boundary tokens, scale by their boundary probability. Apply `frac_gradient` for learning rate modulation.
3. **Inner processing**: The compressed sequence (~170 tokens for 1024 input with `target_avg_token_length=6.0`) goes through the inner transformer.
4. **Upsampling**: Associative scan smooths the inner output, `repeat_interleave` expands chunks back to full length. A residual projection from the encoder output is added.
5. **Auxiliary ratio loss**: Regularizes boundary frequency toward the target average chunk length using a straight-through estimator.

Reference implementation: [lucidrains/h-net-dynamic-chunking](https://github.com/lucidrains/h-net-dynamic-chunking)

### Transformer Blocks

Each block follows the modded-nanogpt pattern:

- **Attention**: GQA with 8 query heads, 4 KV heads, RoPE, QK-norm via `rms_norm`, learnable `q_gain`
- **MLP**: ReLU-squared activation (`relu(x)^2`) with 2x expansion
- **Residual**: Learnable `attn_scale`, `mlp_scale`, and `resid_mix` (blends current state with original input)
- **Precision**: CastedLinear (fp32 weights, bf16 compute), RMSNorm without learnable params
- **Init**: Zero-init on output projections for stable training

### Optimizer

- **Muon** for matrix parameters: lr=0.04, momentum=0.95, 5 Newton-Schulz steps
- **Adam** for token embeddings (tied): lr=0.05, betas=(0.9, 0.95)
- **Adam** for scalars/vectors: lr=0.04
- **Warmdown**: 1200 iterations with wallclock-aware schedule
- **Grad accum**: 8 micro-steps, 524K tokens/batch

### Quantization

- Int8 per-row with clipping at 99.99984th percentile
- Small tensors (<65K elements) kept as fp16
- zlib compression (level 9)
- Full roundtrip validation after quantization

---

## Configuration

| Parameter | Value |
|-----------|-------|
| `model_dim` | 512 |
| `num_heads` | 8 |
| `num_kv_heads` | 4 |
| `mlp_mult` | 2 |
| `hnet_enc_layers` | 2 |
| `hnet_inner_layers` | 5 |
| `hnet_dec_layers` | 2 |
| `hnet_dim_qk` | 128 |
| `hnet_boundary_threshold` | 0.5 |
| `hnet_target_avg_token_length` | 6.0 |
| `hnet_ratio_loss_weight` | 0.03 |
| `hnet_learning_rate_difference` | 0.75 |
| `vocab_size` | 1024 |
| `train_seq_len` | 1024 |
| `logit_softcap` | 30.0 |
| `tie_embeddings` | True |
| Total parameters | 17,451,208 |

---

## Why H-Net for Parameter Golf

### The Thesis

Standard transformers apply uniform compute across all sequence positions. H-Net exploits the observation that natural language has hierarchical structure -- some tokens (content words, sentence boundaries) carry more information than others (articles, prepositions within a phrase). By learning to identify and compress chunks, the inner transformer processes a ~6x shorter sequence, enabling:

1. **Deeper processing per chunk**: 5 inner layers operate on ~170 tokens instead of 1024
2. **Compute efficiency**: Attention is O(n^2), so processing 170 tokens costs ~36x less attention compute than 1024
3. **Learned compression**: The boundary detector adapts to the data, unlike fixed-window approaches

### Differences from Reference

Our implementation adapts the H-Net chunking mechanism to the parameter-golf competitive stack:

| Aspect | Reference (lucidrains) | Ours |
|--------|----------------------|------|
| Encoder/Decoder | LocalTransformer (windowed attn) | Full causal attention + GQA |
| Inner network | TwoSimplicialTransformer | Standard causal attention |
| Position encoding | Learned absolute | RoPE per-layer |
| MLP activation | Standard FFN | ReLU-squared |
| Normalization | `nn.RMSNorm` (learnable) | `F.rms_norm` (no learnable params) |
| Residual projection | Kaiming-uniform init | Zero-init |
| Implementation | Vectorized nested_tensor | Python loops (compile-friendly path WIP) |

These choices integrate H-Net with proven competition techniques (Muon optimizer, CastedLinear, modded-nanogpt blocks) rather than using the reference's research-oriented components.

---

## Results

> **Pending full training run.** Partial results from development:
>
> | Run | Steps | val_bpb | Notes |
> |-----|-------|---------|-------|
> | 15-min scout | 200 | 1.9466 | Early convergence, ~3.1s/step |
>
> The model is training correctly (loss decreasing steadily) but H-Net's per-step overhead from Python-loop chunking makes it slower than flat transformers. Full results will be added after a complete training run.

---

## Reproducing

```bash
# From the repository root:
pip install -r records/track_non_record_16mb/2026-03-27_HNet_DynamicChunking_Enc2_Inner5_Dec2_512d_Muon/requirements.txt

# Download data (if not already present):
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 20

# Run training (default 10-min budget):
CUDA_VISIBLE_DEVICES=0 \
python records/track_non_record_16mb/2026-03-27_HNet_DynamicChunking_Enc2_Inner5_Dec2_512d_Muon/train_gpt.py

# Extended run (e.g., 60 min):
CUDA_VISIBLE_DEVICES=0 \
MAX_WALLCLOCK_SECONDS=3600 \
python records/track_non_record_16mb/2026-03-27_HNet_DynamicChunking_Enc2_Inner5_Dec2_512d_Muon/train_gpt.py
```

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file |
| `submission.json` | Leaderboard metadata (results pending) |
| `train_gpt.py` | Self-contained training script (1473 lines) |
| `requirements.txt` | Python dependencies |

---

## Future Work

- **torch.compile support**: Replace Python loops in chunker with vectorized nested_tensor ops for ~2-3x speedup
- **Hyperparameter tuning**: Adjust `target_avg_token_length`, layer distribution (enc/inner/dec ratio), and learning rates
- **Int6 + lzma quantization**: Match the compression scheme from our x-transformers experiments (best: 1.210 bpb at 180-min)
- **Sliding-window evaluation**: Add stride-64 eval for better BPB
- **Batch size tuning**: Smaller batches showed gains in x-transformers track (327K tokens optimal)
