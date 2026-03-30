# First H-Net Tokenization Submission: 0.6846 BPB — Hierarchical Token Processing

Learned Token Merge/Unmerge + Multi-Resolution Processing + Gated Merge + Per-Position Unmerge + 11L Transformer + int6 GPTQ + sliding window eval

**val_bpb: 0.6846 (seed=42)** | 14.00 MB artifact | 8×H100 SXM, 555s training + 171s eval

## Results (seed=42, 8×H100 SXM)

| Metric | Value |
|--------|-------|
| Sliding BPB | 0.6846 |
| val_bpb (pre-quant) | 0.6604 |
| val_loss | 1.1558 |
| Steps | 2,730 |
| ms/step | 203.31 |
| Training time | 555s |
| GPTQ time | 40s |
| Eval time | 171s |
| Peak memory | 27,567 MiB |
| Artifact | 14,000,183 bytes (14.00 MB) |
| Model bytes | 13,926,222 |
| Code bytes | 73,961 |
| Parameters | ~27M |

## Method

Adds learned token merge/unmerge layers that create a hierarchical processing pipeline within the transformer:

### Token Merge (before encoder layers)
```
Groups of 2 adjacent tokens → 1 super-token
merge_gate = softmax([g_0, g_1])           # learned per-position weights
super_token = g_0 * token_0 + g_1 * token_1  # gated weighted sum
super_token = linear_proj(super_token)       # mixing projection
```

### Multi-Resolution Processing
- **Encoder (5 layers)**: processes at HALF resolution (T/2 tokens)
  - 4x less attention compute (attention is O(T²))
  - Each super-token represents 2 original tokens
- **Unmerge boundary**: expand back to full resolution
- **Decoder (6 layers)**: processes at FULL resolution (T tokens)

### Token Unmerge (before decoder layers)
```
Each super-token → 2 fine tokens via per-position projections:
fine_0 = proj_0(super_token)   # position-specific learned projection
fine_1 = proj_1(super_token)   # different projection for each position
output = interleave(fine_0, fine_1)
```

### Skip Connection Handling
- Encoder skip connections stored at coarse resolution
- Unmerged before decoder skip-add to match fine resolution
- VRL v0_raw recomputed at full resolution at the boundary

## Key Techniques

| Technique | Impact | Notes |
|-----------|--------|-------|
| Gated merge (softmax weights) | Learned token grouping | Model learns which token to emphasize in each pair |
| Per-position unmerge projections | Asymmetric expansion | Each position in the group gets its own projection |
| Multi-resolution processing | Compute savings | Encoder at T/2 → 4x less attention compute |
| Adapted skip connections | Structural correctness | Skips unmerged at resolution boundary |

## Architecture

- merge_factor = 2 (pairs of adjacent tokens)
- Encoder: 5 layers at half resolution
- Decoder: 6 layers at full resolution
- d_model=512 at both resolutions
- Standard attention, MLP, BigramHash, SmearGate at each resolution

## Command

```bash
TORCH_COMPILE_DISABLE=1 \
HNET_ENABLED=1 \
HNET_MERGE_FACTOR=2 \
HNET_COARSE_LAYERS=5 \
NGRAM_EVAL=0 \
KNN_LAMBDA=0 \
SEED=42 \
python3 -m torch.distributed.run --nproc_per_node=8 train_gpt.py
```

## Compliance

- [x] Artifact ≤16,000,000 bytes (14,000,183)
- [x] Training ≤600s on 8×H100 SXM (555s)
- [x] Eval ≤600s (171s)
- [x] GPTQ calibration inside training budget (40s, on training data)
- [x] No validation data during training
- [x] No network calls during evaluation
- [x] No external compute
- [x] No n-gram cache or kNN (clean sliding window eval only)
- [x] Reproducible from `train_gpt.py`

## References

- Token Merging: [arXiv:2210.09461](https://arxiv.org/abs/2210.09461) (Bolya et al., 2022)
- Hierarchical tokenization concepts

## Included Files

- `train_gpt.py` — full training script
- `train_seed42.txt` — training log
- `submission.json` — metadata
- `run.sh` — reproduction script
- `requirements.txt` — dependencies
