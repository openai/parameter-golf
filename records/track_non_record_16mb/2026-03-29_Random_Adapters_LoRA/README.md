# Notable Non-Record: Learning Adapters on Random Linear Maps — 1.3705 BPB

Frozen Random Orthogonal Weights (0 bytes) + LoRA rank-32 Adapters + 30M Effective Params + 5.19 MB Artifact

**val_bpb: 1.3705 (seed=42)** | 5.19 MB artifact | 8×H100 SXM, 555s training + 105s eval

## Results (seed=42, 8×H100 SXM)

| Metric | Value |
|--------|-------|
| val_bpb (post-quant sliding) | 1.3705 |
| val_bpb (pre-quant) | 1.3959 |
| val_loss | 2.3140 |
| Steps | 4,307 |
| ms/step | 128.88 |
| Training time | 555s |
| GPTQ time | 29s |
| Eval time | 105s |
| Peak memory | 24,407 MiB |
| Artifact | 5,191,021 bytes (5.19 MB) |
| Model bytes | 5,115,117 |
| Code bytes | 75,904 |
| Trainable params | 3,744,892 |
| Frozen random params | 25,952,256 (NOT stored) |
| Effective total params | 29,697,148 |
| **Artifact usage** | **32% of 16 MB limit** |

## Method

Standard 11L Transformer, but every attention Q/K/V/proj and MLP fc/proj weight is a `FrozenRandomLinearWithLoRA`:

```
y = x @ W_frozen^T + LoRA(x)
  = x @ W_frozen^T + x @ B^T @ A^T
```

- **W_frozen**: random orthogonal matrix via QR decomposition, generated from a deterministic seed. Registered as `persistent=False` buffer — **NOT saved in state_dict**. At eval time, regenerated from the same seed. **Cost: 0 bytes.**
- **LoRA A**: (out, rank=32), initialized to zeros
- **LoRA B**: (rank=32, in), initialized to N(0, 1/√in)
- **alpha/rank = 1.0** (standard LoRA scaling)

### Why This Works

Random orthogonal projections provide a rich, well-conditioned feature space (reservoir computing principle). The LoRA adapters learn to select and combine features from this random basis. The orthogonal initialization ensures no information is lost in the projection.

### Size Impact

| Component | Params | Stored |
|-----------|--------|--------|
| Frozen random weights | 26M | 0 bytes (regenerated from seed) |
| LoRA adapters | 3.7M | ~5 MB compressed |
| Embeddings, norms, etc. | ~0.5M | included above |
| **Total effective** | **30M** | **5.19 MB** |

### Implementation Details

- `FrozenRandomLinearWithLoRA` overrides `_save_to_state_dict` to exclude frozen weights
- `_load_from_state_dict` regenerates frozen weights from seed on load
- Save/load roundtrip verified: 0.0 logit difference
- Each block gets unique seeds (layer_idx × 100 + offset) for independent random projections

## Architecture

- 11 layers, d_model=512, 8 heads, 4 KV heads (GQA)
- All attention and MLP projections: FrozenRandomLinearWithLoRA (rank 32)
- XSA on all 11 layers, Partial RoPE (16/64), LN Scale
- LeakyReLU(0.5)² MLP (3x expansion)
- BigramHash(2048), SmearGate, VRL
- int6 GPTQ (only 2 layers have quantizable weights — the LoRA params are small)
- EMA(0.997), SWA

## Command

```bash
USE_RANDOM_ADAPTERS=1 \
RANDOM_ADAPTER_RANK=32 \
RANDOM_ADAPTER_SEED=12345 \
NGRAM_EVAL=0 \
KNN_LAMBDA=0 \
SEED=42 \
OMP_NUM_THREADS=1 \
python3 -m torch.distributed.run --nproc_per_node=8 train_gpt.py
```

## Compliance

- [x] Artifact ≤16,000,000 bytes (5,191,021 — 32% of limit)
- [x] Training ≤600s on 8×H100 SXM (555s)
- [x] Eval ≤600s (105s)
- [x] GPTQ calibration inside training budget (29s, on training data)
- [x] No validation data during training
- [x] No network calls during evaluation
- [x] No external compute
- [x] No n-gram cache or kNN (clean sliding window eval only)
- [x] Reproducible from `train_gpt.py`

## References

- LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- Reservoir computing / random features
- Orthogonal initialization: [arXiv:1312.6120](https://arxiv.org/abs/1312.6120) (Saxe et al., 2013)

## Included Files

- `train_gpt.py` — full training script
- `train_seed42.txt` — training log
- `submission.json` — metadata
- `run.sh` — reproduction script
- `requirements.txt` — dependencies
