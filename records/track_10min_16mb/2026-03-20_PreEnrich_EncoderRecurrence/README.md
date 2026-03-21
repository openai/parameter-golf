## Pre-Enrichment + Encoder Recurrence + SmearGate + BigramHash

Architectural modifications to the baseline transformer achieving **val_bpb 1.1668** in a 15.02MB artifact trained in 10 minutes on 8xH100. Key techniques: GELU pre-enrichment (512→768→512), 2x encoder recurrence with RMS norm stabilization, SmearGate for lightweight bigram context, BigramHash for explicit bigram embeddings, and EMA weight averaging for quantization-friendly weights.

---

### Key Contributions

#### GELU Pre-Enrichment (512→768→512)

Two `CastedLinear` projections with a GELU activation between them, applied after the embedding lookup and before the first transformer block. The wider hidden dimension (768 vs baseline 512) gives the model a richer nonlinear transformation before the residual stream begins.

```
embedding → BigramHash add → SmearGate → Linear(512→768) → GELU → Linear(768→512) → RMS Norm → transformer blocks
```

#### 2x Encoder Recurrence

I reuse the encoder blocks for a second pass before running the decoder, with RMS norm stabilization between passes. With 10 layers (5 encoder + 5 decoder), this produces **15 effective layers from 10 physical blocks** with zero extra parameters.

**A/B Comparison — MLP 3x, seq 2048, int6 QAT (8xH100, 10 minutes):**

| Metric              | With recurrence    | Without recurrence |
|---------------------|--------------------|-----------------------|
| Steps completed     | 6,423              | 8,950                 |
| Step time           | 93ms               | 67ms                  |
| Sliding window BPB  | **1.1709**         | 1.1740                |

Encoder recurrence consistently wins — deeper processing per step beats more gradient updates.

#### SmearGate

Learned per-dimension gate (512 params) that blends each token's embedding with the previous token's embedding. Provides lightweight bigram context at the embedding layer. Initialized with gate bias 3.0 (sigmoid(3.0)≈0.95, near-identity at init).

#### BigramHash

Hash-table embedding mapping token bigrams to learned vectors. Hash formula: `(prev_token * 92821 + curr_token) % 4096`. Lookup table 4096×64, projected to model_dim via Linear(64, 512). Adds explicit bigram context to the token embedding.

#### EMA Weight Averaging

Exponential moving average (decay=0.997) updated every step, replacing SWA. EMA weights are loaded before quantization. Produces smoother weights that quantize significantly better — quant gap dropped from 0.020 (SWA) to **0.004** (EMA).

---

### Additional Techniques

Int6 quantization-aware training (fake quant with STE in CastedLinear), lzma compression, MLP 3x expansion, overtone embedding init, decoupled Muon weight decay (0.04), AdamW weight decay (0.04), batched sliding window eval (stride=64), fp16 embedding passthrough in quantization.

Hyperparameters: NUM_LAYERS=10, TRAIN_SEQ_LEN=2048, TRAIN_BATCH_TOKENS=393216, MATRIX_LR=0.028, SCALAR_LR=0.025, TIED_EMBED_LR=0.035, MUON_MOMENTUM=0.99, WARMDOWN_ITERS=3300.

---

### What Didn't Work

- **Phase-transition resid_mix init**: Sigmoid-scheduled initialization of resid_mix. Slowed convergence at our step count, hurt final score.

- **Late-K passthrough**: Keeping last 2 layers' c_k.weight in fp16 during quantization. Added artifact size without enough BPB improvement.

- **Gradient clipping (GRAD_CLIP_NORM=1.0)**: Constrained the optimizer, slower per-step learning.

- **12 layers + MLP 2x**: 18 effective layers with recurrence but MLP 2x bottleneck was too narrow. 10L MLP 3x wins.

- **Full dataset (80 shards) with WD=0.04**: More diverse data didn't improve pre-quant BPB. Only helped quant gap when combined with higher WD.

- **3x encoder recurrence**: Exceeded Triton's per-SM shared memory limit. Compiler limitation.

- Also tried: full U-Net recurrence (too slow), reverse encoder pass order (worse), auxiliary encoder prediction loss (hurt performance), 6+3 encoder/decoder split (worse than 5+5).

---

### Configuration

```
RUN_CONFIG=A
VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
TIE_EMBEDDINGS=1 TIED_EMBED_LR=0.035 MATRIX_LR=0.028 SCALAR_LR=0.025
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3300 WARMUP_STEPS=20 TRAIN_BATCH_TOKENS=393216 TRAIN_SEQ_LEN=2048
ENCODER_RECURRENCE=1 MUON_WD=0.04 ADAM_WD=0.04 EMA_DECAY=0.997
```

### Reproduction

All defaults are baked into the script:
```bash
RUN_CONFIG=A torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Key Metrics

| Metric | Value |
|---|---|
| Pre-quant val_bpb | 1.1848 |
| Post-quant val_bpb (standard) | 1.1889 |
| Post-quant val_bpb (sliding window) | **1.1668** |
| Quant gap (standard - pre-quant) | 0.004 |
| Training time | 600,011ms (5,373 steps at ~112ms) |
| Peak memory | 14,124 MiB |
| Submission size (int6+lzma) | 15,022,232 bytes |
| Model parameters | 25,222,224 |
