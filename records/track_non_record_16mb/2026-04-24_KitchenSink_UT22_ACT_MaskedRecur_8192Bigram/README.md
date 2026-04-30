# Non-record: Kitchen Sink — UT×22, ACT vs Masked Recurrence, 8192-Bigram

**val_bpb: 1.4011 (honest, pre-TTT)** | 1×A10G-24GB, 20k steps, sp1024, ACT run

## What This PR Is

Ablation study on top of the kitchen-sink Universal Transformer baseline
(UT×22, Echo Training, Gradient Quilting, Adaptive Density, 8192-bigram buckets,
XSA, LeakyReLU², EMA, late QAT, Brotli-11) comparing two recurrence control
mechanisms added to the UT loop:

1. **ACT (Adaptive Computation Time, Graves 2016)** — learned per-token halting
   probability with ponder cost λ=0.01. Encoder and decoder have separate halting
   budgets; each UT iteration may conditionally exit early per token.

2. **Masked Recurrence** — learned per-token soft gate that interpolates between
   applying or skipping a block update at each UT iteration.

Both variants use 22 total UT iterations with the Adaptive Density sparse-to-dense
curriculum (sparsity decays from ~85% at step 400 to 0% at step 12000).

## TTT Note

Both runs were trained with `TTT_ENABLED=1`, which runs 10 epochs of adaptation on
`val_tokens` before the final roundtrip evaluation. This is the same pattern flagged
in PR #1376 and corrected in PR #1193. The headline val_bpb values below are the
pre-TTT checkpoint at step 20000. Post-TTT scores are reported for reference only.

## Results

| Run | Recurrence | val_bpb (pre-TTT) | TTT roundtrip (int6+zlib-9) | ms/step | Train time |
|-----|-----------|------------------|----------------------------|---------|------------|
| ACT | ACT, ponder_cost=0.01 | **1.4011** | 1.26482870 | 6087 | ~33.8 h |
| Masked-Recur | Soft gate | 1.4044 | 1.26742860 | 6130 | ~34.1 h |

Common config: UT=True, num_iters=22, echo=True, quilt=True, sparse=True, film=False,
20k steps, ITERATIONS=20000, VOCAB_SIZE=1024, NUM_LAYERS=11, MODEL_DIM=512,
BIGRAM_BUCKETS=8192, BIGRAM_EMBED_DIM=128.

## Convergence

| Step | ACT val_bpb | Masked-Recur val_bpb |
|------|-------------|----------------------|
| 0 | 3.4657 | 3.4645 |
| 1000 | 1.6507 | 1.6340 |
| 5000 | 1.4702 | 1.4684 |
| 10000 | 1.4210 | 1.4209 |
| 15000 | 1.4090 | 1.4123 |
| 20000 | **1.4011** | **1.4044** |

## Model Config

| Parameter | Value |
|-----------|-------|
| vocab_size | 1024 (sp1024) |
| num_layers | 11 (weight-shared, UT) |
| num_iters | 22 |
| model_dim | 512 |
| num_heads / kv_heads | 8 / 8 |
| mlp_hidden | 1792 |
| bigram_buckets | 8192 |
| bigram_embed_dim | 128 |
| rope_base / rope_dims | 10000 / 16 (partial RoPE) |
| logit_softcap | 30.0 |
| ln_scale | True |
| Activation | LeakyReLU² |
| Attention | XSA (cross-shaped) |
| Compression | Brotli-11 |
| Optimizer | Muon (momentum=0.99, wd=0.04, backend_steps=5) |
| embed_lr / matrix_lr | 0.6 / 0.025 |
| warmup / warmdown | 20 / 6000 steps |

## Findings

1. **ACT and masked recurrence converge to nearly identical BPB** (delta: 0.003). Neither
   is clearly superior at this model and training budget — the recurrence control mechanism
   matters less than the underlying UT compute budget.

2. **Learning plateaus mid-training**: both runs drop from 1.43 → 1.41 between steps 8k–14k
   with minimal improvement during the final 6k warmdown. Additional training steps beyond
   20k or a wider model (d=640) are likely higher-leverage than recurrence variant choice.

3. **TTT-on-val headroom is large**: 10-epoch adaptation on the validation set moves
   val_bpb from ~1.40 to an int6+zlib-9 roundtrip of ~1.265 (~0.14 bpb gap). This
   represents an upper bound on what a legal score-first TTT would achieve on unseen data.

4. **8192 bigram buckets not ablated**: both runs use BIGRAM_BUCKETS=8192. Its marginal
   contribution vs smaller bigram tables is not isolated here.

## Artifact Size

- Serialized model (raw): 24,664,446 bytes
- int6+zlib-9 compressed: 10,827,806–10,852,792 bytes (~10.3 MB)
- Total submission with code: ~10.9 MB — within the 16 MB track limit

## Reproduction

```bash
pip install sentencepiece brotli
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# ACT variant (reported metric)
TTT_ENABLED=0 ACT_ENABLED=1 MASKED_RECUR_ENABLED=0 \
  BIGRAM_BUCKETS=8192 BIGRAM_EMBED_DIM=128 ITERATIONS=20000 \
  VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_ITERS=22 \
  ECHO_ENABLED=1 QUILT_ENABLED=1 SPARSE_ENABLED=1 \
  python3 records/track_non_record_16mb/2026-04-24_KitchenSink_UT22_ACT_MaskedRecur_8192Bigram/train_gpt.py

# Masked recurrence variant
TTT_ENABLED=0 ACT_ENABLED=0 MASKED_RECUR_ENABLED=1 \
  BIGRAM_BUCKETS=8192 BIGRAM_EMBED_DIM=128 ITERATIONS=20000 \
  VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_ITERS=22 \
  ECHO_ENABLED=1 QUILT_ENABLED=1 SPARSE_ENABLED=1 \
  python3 records/track_non_record_16mb/2026-04-24_KitchenSink_UT22_ACT_MaskedRecur_8192Bigram/train_gpt.py
```

## Related

- PR #1193 Non-record: Universal Transformer + Adaptive Density (same base script, DGX Spark ablations)
- PR #1193 update: TTT-on-val disabled, honest BPB from 200-step DGX Spark run
- PR #1204 msisovic Mini Depth Recurrence
- PR #1334 aryanbhosale depth recurrence + parallel residuals (1.0897 BPB)
