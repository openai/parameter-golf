## Pre-Enrichment + Encoder Recurrence

Two architectural modifications to the baseline transformer: (1) a GELU pre-enrichment block that transforms raw embeddings before they enter the residual stream, and (2) 2x encoder recurrence that runs the encoder blocks twice with RMS norm stabilization between passes. Combined with int6 QAT, lzma compression, MLP 3x, SWA, sliding window evaluation (stride=64), and overtone embedding initialization, this achieves **val_bpb 1.1709** in a 15.57MB artifact trained in 10 minutes on 8xH100.

---

### Key Contributions

#### GELU Pre-Enrichment

Raw token embeddings are a poor starting point for the residual stream. A 1024-token vocabulary maps each token to a 512-dimensional vector initialized from a normal distribution — these vectors carry no relational structure and every transformer layer must compensate for this weak initialization.

I add two `CastedLinear(512→512)` projections with a GELU activation between them, applied after the embedding lookup and before the first transformer block:

```
embedding → Linear(512→512) → GELU → Linear(512→512) → RMS Norm → transformer blocks
```

This gives the model a learned nonlinear transformation to produce richer representations before the residual stream begins. Cost: 0.5M extra parameters (~2% of total), negligible step time overhead.

#### 2x Encoder Recurrence

Depth recurrence is a known technique (ALBERT, Universal Transformers). My contribution is applying it to only the encoder half of a U-Net transformer architecture, with RMS norm stabilization between passes, and providing A/B data showing it consistently beats additional training steps across two different model configurations.

The baseline uses a U-Net architecture with encoder and decoder halves connected by skip connections. I reuse the encoder blocks for a second pass before running the decoder.

With 10 layers (5 encoder + 5 decoder), the forward pass becomes:
1. Run encoder blocks 0-4 (first pass, build initial features)
2. RMS norm (stabilize between passes)
3. Run encoder blocks 0-4 again (second pass, refine features)
4. Run decoder blocks 5-9 with skip connections from the refined second encoder pass

This produces **15 effective layers from 10 physical blocks** with zero extra parameters.

**A/B Comparison — Config 2 (MLP 3x, seq 2048, int6 QAT, SWA):**

| Metric              | With recurrence    | Without recurrence |
|---------------------|--------------------|-----------------------|
| Steps completed     | 6,423              | 8,950                 |
| Step time           | 93ms               | 67ms                  |
| Standard BPB        | 1.1929             | 1.1959                |
| Sliding window BPB  | **1.1709**         | 1.1740                |
| Submission size     | 15.57MB            | 15.54MB               |

**A/B Comparison — Config 1 (MLP 2x, seq 1024, int8+zlib):**

| Metric              | With recurrence    | Without recurrence |
|---------------------|--------------------|-----------------------|
| Steps completed     | 8,004              | 11,955                |
| Step time           | 75ms               | 50ms                  |
| Standard BPB        | 1.2211             | 1.2299                |
| Sliding window BPB  | **1.1855**         | 1.1947                |
| Submission size     | 15.75MB            | 15.82MB               |

Encoder recurrence wins across both configurations — different model sizes, different sequence lengths, different step counts. In both cases, 30-40% fewer training steps could not overcome the depth advantage. The pattern is consistent: deeper processing per step beats more gradient updates with shallower processing.

---

### Additional Techniques

Int6 quantization-aware training (fake quant with STE in CastedLinear), lzma compression, MLP 3x expansion, stochastic weight averaging (11 checkpoints during warmdown), overtone embedding init, decoupled Muon weight decay (0.04), AdamW weight decay (0.04), batched sliding window eval (stride=64), fp16 embedding passthrough in quantization.

Hyperparameters: NUM_LAYERS=10, TRAIN_SEQ_LEN=2048, MATRIX_LR=0.035, SCALAR_LR=0.025, TIED_EMBED_LR=0.035, MUON_MOMENTUM=0.99, WARMDOWN_ITERS=2100.

---

### What Didn't Work

- **FP16 embedding passthrough (without int6)**: Keeping the tied embedding in fp16 instead of int8 reduced quantization error by ~0.006 BPB but pushed the int8+zlib artifact over 16MB. Switching to int6 quantization solved this — fp16 embedding fits comfortably in the int6+lzma budget.

- **3x encoder recurrence**: The tripled computation graph exceeded Triton's per-SM shared memory limit on A100 (168,096 > 166,912 bytes). A compiler limitation, not an architectural one.

- **Warmdown scheduler on A100**: The wallclock-aware warmdown schedule estimates remaining time as `warmdown_iters × avg_step_time`. On A100 (~1100ms/step), this exceeds the total 600-second budget from step 0, causing the learning rate to decay throughout the entire run. Not relevant to 8xH100 but was a significant debugging finding during development.

- Also tried: full U-Net recurrence (too slow), reverse encoder pass order (worse), auxiliary encoder prediction loss (hurt performance), 6+3 encoder/decoder split (worse than 5+5).

---

### Configuration

```
RUN_CONFIG=A
VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
TIE_EMBEDDINGS=1 TIED_EMBED_LR=0.035 MATRIX_LR=0.035 SCALAR_LR=0.025
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=2100 WARMUP_STEPS=20 TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048
ENCODER_RECURRENCE=1 MUON_WD=0.04 ADAM_WD=0.04 SWA_EVERY=200
```

### Reproduction

All defaults are baked into the script:
```bash
RUN_CONFIG=A torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Key Metrics

| Metric | Value |
|---|---|
| Pre-quant val_bpb | 1.1730 |
| Post-quant val_bpb (standard) | 1.1929 |
| Post-quant val_bpb (sliding window) | **1.1709** |
| Training time | 600,034ms (6,423 steps at ~93ms) |
| Peak memory | 18,506 MiB |
| Submission size (int6+lzma) | 15,567,990 bytes |
| Model parameters | 24,664,656 |
