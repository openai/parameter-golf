## Pre-Enrichment + Encoder Recurrence + XSA + SmearGate + BigramHash

**val_bpb: 1.1629** (sliding window, stride=64) | 15.05 MB | 8xH100 SXM, 600s

---

### Progress

| | v1 | v2 | v3 | v4 (this) |
|---|---|---|---|---|
| val_bpb (sliding) | 1.1855 | 1.1709 | 1.1668 | **1.1629** |
| Params | 19.4M | 24.7M | 25.2M | 25.2M |
| Artifact | 15.75 MB | 15.57 MB | 15.02 MB | 15.05 MB |
| Steps (600s) | 8,004 | 6,423 | 5,373 | 5,636 |
| Step time | 75ms | 93ms | 112ms | 106ms |
| Quant gap | 0.020 | 0.020 | 0.004 | 0.004 |

---

### Key Contributions

#### GELU Pre-Enrichment (512→768→512)

Raw token embeddings carry no relational structure. I add a wider nonlinear transformation before the residual stream:
embedding → BigramHash add → SmearGate → Linear(512→768) → GELU → Linear(768→512) → RMS Norm → transformer blocks

The wider bottleneck (768) gives the embedding transformation more capacity than the original 512→512. Cost: ~0.8M params, negligible step time.

#### 2x Encoder Recurrence

Depth recurrence is a known technique (ALBERT, Universal Transformers). My contribution is applying it to only the encoder half of a U-Net transformer architecture, with RMS norm stabilization between passes.

With 10 layers (5 encoder + 5 decoder), the forward pass becomes:
1. Run encoder blocks 0-4 (first pass)
2. RMS norm (stabilize between passes)
3. Run encoder blocks 0-4 again (second pass, refine)
4. Run decoder blocks 5-9 with skip connections from second encoder pass

**15 effective layers from 10 physical blocks**, zero extra parameters.

**A/B Comparison — MLP 3x + seq 2048 config (8xH100, 10 minutes):**

| Metric | With recurrence | Without recurrence |
|---|---|---|
| Steps completed | 6,423 | 8,950 |
| Step time | 93ms | 67ms |
| Sliding window BPB | **1.1709** | 1.1740 |

**A/B Comparison — MLP 2x + seq 1024 config (8xH100, 10 minutes):**

| Metric | With recurrence | Without recurrence |
|---|---|---|
| Steps completed | 8,004 | 11,955 |
| Step time | 75ms | 50ms |
| Sliding window BPB | **1.1855** | 1.1947 |

Recurrence wins across both configs despite 28-40% fewer gradient updates.

#### XSA (Exclusive Self Attention) on Last 4 Layers

Removes self-value bias from attention output via orthogonal projection (arXiv:2603.09078). After computing attention output Y, XSA subtracts the component aligned with each token's own value vector:

```
Vn = normalize(V, dim=-1)
Y = Y - (Y · Vn).sum(dim=-1, keepdim=True) * Vn
```

Forces attention layers to capture purely contextual information from other tokens. Zero new parameters. Applied to last 4 layers only — early layers retain self-attention for basic feature building. Requires GQA-aware expansion of V to match Q head count before projection.

v3 → v4 improvement: 1.1668 → 1.1629 (-0.004 BPB).

---

### Additional Techniques

- **SmearGate**: Per-dim learnable gate blending each token with previous token's embedding. 512 params.
- **BigramHash** (4096×64): Hash-table embedding for token bigrams, projected to model dim. ~590K params.
- **EMA** (decay=0.997): Exponential moving average replacing SWA. Quant gap reduced from 0.020 to 0.004 across versions.
- **Int6 QAT**: Fake quantization with straight-through estimator during training. Model learns int6-friendly weights.
- **lzma compression**: Stdlib replacement for zlib. Zero dependency risk.

Also: MLP 3x, seq 2048, overtone init, Muon+AdamW WD=0.04, sliding window eval stride=64.

Overtone init, Muon weight decay, and sliding window eval adapted from notapplica and Matthew Li's work.

---

### What Didn't Work

- **FP16 embedding passthrough**: Reduced quant error by ~0.006 BPB but added ~520KB, pushing artifact over 16MB.
- **3x encoder recurrence**: Exceeded Triton's per-SM shared memory limit on A100 and RTX 4050.
- **Reverse encoder recurrence** (second pass in reverse order): Worse than forward-only (1.4140 vs 1.4077 on A100).
- **Auxiliary encoder loss**: Hurt performance. Encoder works better optimized purely for decoder consumption.
- **Phase-transition resid_mix + gradient clipping**: Borrowed from top submissions, hurt our config. Techniques tuned for non-recurrence setups don't always transfer.
- **12L MLP 2x with recurrence (18 effective layers)**: Numbers were significantly worse than 10L MLP 3x. Width beats depth at this scale.
- **Warmdown scheduler on A100**: Wallclock-aware warmdown decayed LR from step 0 on A100 (~1100ms/step). Override to WARMDOWN_ITERS=120 required for local development.

---

### Configuration
TRAIN_BATCH_TOKENS=393216 MATRIX_LR=0.028 MUON_WD=0.04 ADAM_WD=0.04
WARMDOWN_ITERS=3300 NUM_LAYERS=10 MLP_MULT=3 TRAIN_SEQ_LEN=2048
ENCODER_RECURRENCE=1 EMA_DECAY=0.997 XSA_LAST_N=4

Model parameters: 25,222,224
Submission size (int6+lzma): 15,051,927 bytes (code: 59,427 bytes)

### Reproduction

All defaults are baked into the script — no env vars needed.

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Key Metrics

| Metric | Value |
|---|---|
| Pre-quant val_bpb | 1.1809 |
| Post-quant val_bpb (standard) | 1.1848 |
| Post-quant val_bpb (sliding window) | **1.1629** |
| Quant gap (standard - pre-quant) | 0.004 |
| Training time | 599,886ms (5,636 steps at ~106ms) |
| Peak memory | 14,147 MiB |
| Submission size (int6+lzma) | 15,051,927 bytes |
| Model parameters | 25,222,224 |

### Included Files

- `train_gpt.py` — standalone training script with all modifications
- `train.log` — full 8xH100 training log (seed 1337)
- `submission.json` — leaderboard metadata
- `README.md` — this file
