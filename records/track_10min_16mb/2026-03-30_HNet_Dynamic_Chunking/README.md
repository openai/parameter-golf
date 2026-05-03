# First H-Net Submission: 1.3639 BPB — Dynamic Chunking for Hierarchical Sequence Modeling

H-Net Dynamic Chunking + Cosine Similarity Routing + EMA Smoothing (Mamba2 kernel) + Multi-Resolution Processing + 11L Transformer + int6 GPTQ + sliding window eval

**val_bpb: 1.3639 (seed=42)** | 10.39 MB artifact | 8×H100 PCIe, 555s training + 253s eval

## Results (seed=42, 8×H100 PCIe)

| Metric | Value |
|--------|-------|
| Sliding BPB | 1.3639 |
| val_bpb (pre-quant) | 1.3595 |
| val_loss | 2.3029 |
| Steps | 1,706 |
| ms/step | 325.35 |
| Training time | 555s |
| GPTQ time | 43s |
| Eval time | 253s |
| Peak memory | 28,840 MiB |
| Artifact | 10,394,755 bytes (10.39 MB) |
| Model bytes | 10,313,565 |
| Code bytes | 81,190 |
| Parameters | 27,780,220 |

## Method

Implements the Dynamic Chunking mechanism from H-Net (Hwang, Wang, Gu 2025, arXiv:2507.07955), adapted for the token-level competition setting. The model learns content-dependent chunk boundaries end-to-end.

### Routing Module (Eq. 4)

Separate W_q and W_k projections measure cosine similarity between adjacent representations:

```
q_t = W_q * x_t,  k_t = W_k * x_t
p_t = 0.5 * (1 - cos_sim(q_t, k_{t-1}))    # boundary probability
b_t = 1{p_t >= 0.5}                          # hard boundary indicator
```

Low similarity between adjacent representations indicates a semantic shift — a natural chunk boundary. First position always has p_0 = 1.0.

### Downsampler

Selects vectors at boundary positions (b_t = 1) using scatter-based accumulation with STE for gradient flow through hard boundary decisions. Non-boundary vectors are discarded.

### Smoothing Module (Eq. 5)

EMA on the decoder side converts discrete chunks into smooth interpolations:

```
z̄_t = P_t * ẑ_t + (1 - P_t) * z̄_{t-1}
```

High-confidence boundaries (P_t ≈ 1.0) maintain discrete boundaries. Low-confidence boundaries are smoothed with previous context, creating a self-correcting mechanism.

### Upsampler (Eq. 6-9)

Confidence scoring + STE + causal expansion:

```
c_t = p_t if b_t=1, else 1-p_t              # confidence scoring
STE(c_t) = c_t + stopgrad(1 - c_t)          # rounds to 1.0 in forward
z̃_t = z̄_{cumsum(b)_t}                       # causal expansion (repeat until next boundary)
output_t = STE(c_t) * z̃_t                    # confidence-weighted decompression
```

### Residual Skip Connection (Eq. 3)

```
z^s = Dechunk(ẑ^{s+1}, p^s) + Linear(x̂^s)
```

Encoder output is projected and added to the unchunked decoder output, preserving fine-grained information.

### Ratio Loss (Eq. 10)

Guides compression toward target ratio N without imposing rigid rules:

```
L_ratio = (N/(N-1)) * ((N-1)*F*G + (1-F)*(1-G))
F = mean(b_t),  G = mean(p_t)
```

Combined with AR loss: L = L_AR + 0.03 * L_ratio (α = 0.03 as in paper).

## Key Techniques

| Technique | Paper Reference | Notes |
|-----------|----------------|-------|
| Cosine similarity routing (W_q, W_k) | Eq. 4 | Separate projections for boundary detection |
| Hard boundary selection + STE | Eq. 4, 7 | Gradient flows through hard decisions |
| EMA smoothing module | Eq. 5 | Differentiable chunking, self-correcting |
| Confidence-weighted upsampler | Eq. 6-9 | Causal expansion with STE |
| Residual skip connection | Eq. 3 | Encoder output projected and added to decoder |
| Ratio loss | Eq. 10 | Guides compression toward target ratio (α=0.03) |
| Multi-resolution processing | Architecture | Encoder at ~T/2, decoder at full T |

### Differences from Paper
- We operate on SP1024 tokens (not raw bytes) due to competition constraints
- We use Transformer layers for encoder/decoder (paper uses Mamba-2 SSM layers)
- Single-stage hierarchy (paper shows 2-stage is better, but adds complexity)

## Architecture

- target_ratio = 2 (approximately 2x compression)
- Encoder: 5 layers at compressed resolution
- Decoder: 6 layers at full resolution
- d_model=512 at both resolutions
- Routing: cosine similarity + learned threshold + temperature
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

- [x] Artifact ≤16,000,000 bytes (10,394,755)
- [x] Training ≤600s on 8×H100 (555s)
- [x] Eval ≤600s (253s)
- [x] GPTQ calibration inside training budget (43s, on training data)
- [x] No validation data during training
- [x] No network calls during evaluation
- [x] No external compute
- [x] No n-gram cache or kNN (clean sliding window eval only)
- [x] Reproducible from `train_gpt.py`

## References

- H-Net: [arXiv:2507.07955](https://arxiv.org/abs/2507.07955) (Hwang, Wang, Gu 2025)

## Included Files

- `train_gpt.py` — full training script
- `train_seed42.txt` — training log
- `submission.json` — metadata
- `run.sh` — reproduction script
- `requirements.txt` — dependencies
