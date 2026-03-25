# 10L Enhanced: Int5 MLP + BigramHash(12288) + SmearGate + SWA + WD

**Expected val_bpb: ~1.1428** (to be measured after training)

> **Note:** This submission is based on the 10L SOTA configuration with an increased BigramHash vocabulary size from 10240 to 12288. The goal is to capture more bigram pairs with minimal increase in artifact size while maintaining the 16MB budget.

## Run Command

```bash
# Setup (once)
bash prepare.sh

# Train + evaluate (default seed=42)
bash eval/eval.sh

# With specific seed
SEED=42 bash eval/eval.sh
```

All parameters are set as defaults in `train_gpt.py`. To adjust BigramHash vocab size, set `BIGRAM_VOCAB_SIZE` environment variable.

## Key Techniques

### Mixed Int5/Int6 Quantization
- **Int5 [-16,15]** for MLP weights (most compressible)
- **Int6 [-32,31]** for attention weights (precision-sensitive)
- **FP16** for tied embeddings and last-layer key projections
- Int5 MLP saves ~1.86MB vs uniform int6, enabling 10 layers

### BigramHash(12288)
- Hash consecutive token pairs into 12288-bucket embedding table (dim=128)
- Projected to model_dim=512 via learned linear
- Larger bucket count reduces collisions compared to 10240, potentially improving bigram signal

### SWA with start_frac=0.4
- Collect checkpoints from last 40% of warmdown
- Averaged every 50 steps for smoother weights

### Additional Techniques
- SmearGate: lightweight embedding-level context mixing
- Orthogonal initialization with muP scaling for output projections
- Muon optimizer with weight_decay=0.04; AdamW with WD=0.04 for embeddings/scalars
- Sliding window evaluation (stride=64)
- Magnitude pruning (3%) before quantization
- Longer sequences: train_seq_len=2048

## Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3× expansion (hidden=1536), ReLU² activation
- U-Net skip connections, tied embeddings

## Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| num_layers | 10 |
| model_dim | 512 |
| mlp_mult | 3.0 |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| warmdown_iters | 3000 |
| matrix_lr (Muon) | 0.02 |
| scalar_lr (AdamW) | 0.02 |
| tied_embed_lr | 0.03 |
| muon_momentum | 0.99 (warmup from 0.92 over 1500 steps) |
| muon_weight_decay | 0.04 |
| adamw_weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| eval_stride | 64 |
| swa_every | 50 |
| swa_start_frac | 0.4 |
| bigram_vocab_size | 12288 |
| bigram_dim | 128 |
| compressor | zstd (level 22) |

## Expected Results

After training on 8×H100 for 10 minutes, fill in the table:

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|----------------|-------|
| 42   | TODO    | TODO           | yes   |
| 1337 | TODO    | TODO           | yes   |
| 2024 | TODO    | TODO           | yes   |
| **Mean** | **TODO** | | |
| **Std**  | **TODO** | | |

## Reproduction

1. Ensure data is prepared: `python3 data/cached_challenge_fineweb.py --variant sp1024`
2. Run training on 8×H100 SXM with wallclock cap 600s.
3. The script will automatically:
   - Perform sliding window evaluation after quantization roundtrip
   - Validate artifact size ≤ 16,000,000 bytes
   - Save `final_model.int8.ptz` and logs under `logs/`

## Notes

- The increase in BigramHash vocabulary from 10240 to 12288 adds approximately 262k parameters (128×2048), which after int6 quantization and zstd-22 compression is expected to add <0.1 MB to the artifact. The SOTA used 10240; this tweak aims for a marginal BPB improvement.
- All other hyperparameters match the 10L SOTA (1.14276 mean) to preserve proven performance.
- If artifact size exceeds limit, the script will raise an assertion error during serialization.

## Comparison with SOTA

| Configuration | val_bpb (mean) | Bigram Vocab |
|---------------|----------------|--------------|
| 10L SOTA (1.14276) | 1.14276 | 10240 |
| This submission (expected) | ~1.1425? | 12288 |

Theoretical improvement: reducing bigram hash collisions may lower cross-entropy by ~0.0005-0.001 bpb. Actual result depends on training randomness.

---

*Built on the work of the 10L Int5-MLP SOTA submission and PR #162 (SmearGate, BigramHash, OrthoInit).*
