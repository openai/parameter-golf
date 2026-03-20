## Pre-Enrichment + Encoder Recurrence

Two architectural modifications to the baseline transformer: (1) a GELU pre-enrichment block that transforms raw embeddings before they enter the residual stream, and (2) 2x encoder recurrence that runs the encoder blocks twice with RMS norm stabilization between passes. Combined with sliding window evaluation (stride=64), overtone embedding initialization, and decoupled Muon weight decay, this achieves **val_bpb 1.1855** in a 15.75MB artifact trained in 10 minutes on 8xH100.

---

### Key Contributions

#### GELU Pre-Enrichment

Raw token embeddings are a poor starting point for the residual stream. A 1024-token vocabulary maps each token to a 512-dimensional vector initialized from a normal distribution — these vectors carry no relational structure and every transformer layer must compensate for this weak initialization.

I add two `CastedLinear(512→512)` projections with a GELU activation between them, applied after the embedding lookup and before the first transformer block:

```
embedding → Linear(512→512) → GELU → Linear(512→512) → RMS Norm → transformer blocks
```

This gives the model a learned nonlinear transformation to produce richer representations before the residual stream begins. Cost: 0.5M extra parameters (~3% of total), negligible step time overhead.

#### 2x Encoder Recurrence

Depth recurrence is a known technique (ALBERT, Universal Transformers). My contribution is applying it to only the encoder half of a U-Net transformer architecture, with RMS norm stabilization between passes, and providing A/B data showing it beats additional training steps.

The baseline uses a U-Net architecture with encoder and decoder halves connected by skip connections. I reuse the encoder blocks for a second pass before running the decoder.

With 10 layers (5 encoder + 5 decoder), the forward pass becomes:
1. Run encoder blocks 0-4 (first pass, build initial features)
2. RMS norm (stabilize between passes)
3. Run encoder blocks 0-4 again (second pass, refine features)
4. Run decoder blocks 5-9 with skip connections from the refined second encoder pass

This produces **15 effective layers from 10 physical blocks** with zero extra parameters. The only cost is step time: ~75ms vs ~50ms without recurrence (~50% overhead from running 5 extra blocks).

The critical question: does the architectural depth advantage justify 50% fewer training steps?

**A/B Comparison (8xH100, 10 minutes, identical config except recurrence):**

| Metric              | With recurrence    | Without recurrence |
|---------------------|--------------------|-----------------------|
| Steps completed     | 8,004              | 11,955                |
| Step time           | 75ms               | 50ms                  |
| Standard BPB        | 1.2211             | 1.2299                |
| Sliding window BPB  | **1.1855**         | 1.1947                |
| Submission size     | 15.75MB            | 15.82MB               |

50% more training steps could not overcome the depth advantage of encoder recurrence. At step 8000 (where the recurrence run stopped), the pre-quant val_bpb was 1.2065 vs 1.3020 for the no-recurrence run — a 0.0955 gap that the extra 4,000 steps narrowed but never closed.

I find encoder recurrence to be a parameter-efficient alternative to adding physical layers: it doubles the effective encoder depth with zero parameters and predictable step time overhead.

---

### Additional Techniques

Overtone embedding init, decoupled Muon weight decay (0.02), batched sliding window eval (stride=64), 10 layers, MATRIX_LR=0.06, TIED_EMBED_LR=0.1, WARMDOWN_ITERS=2500.

---

### What Didn't Work

- **FP16 embedding passthrough**: Keeping the tied embedding in fp16 instead of int8 reduced quantization error by ~0.006 BPB (the tied embedding is used twice — input and output — so int8 errors compound). However, the extra ~520KB pushed the artifact over the 16MB cap. I had to revert to int8.

- **3x encoder recurrence**: The tripled computation graph exceeded Triton's per-SM shared memory limit on both A100 (168,096 > 166,912 bytes) and RTX 4050. A compiler limitation, not an architectural one.

- **Warmdown scheduler on A100**: The wallclock-aware warmdown schedule (`WARMDOWN_ITERS=1200`) estimates remaining time as `warmdown_iters × avg_step_time`. On A100 (~1100ms/step), this exceeds the total 600-second budget from step 0, causing the learning rate to decay throughout the entire run. Not relevant to 8xH100 submissions but was a significant debugging finding.

- Also tried: full U-Net recurrence (too slow), reverse encoder pass order (worse), auxiliary encoder prediction loss (hurt performance).

---

### Configuration

```
VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
TIE_EMBEDDINGS=1 TIED_EMBED_LR=0.1 MATRIX_LR=0.06 SCALAR_LR=0.04
WARMDOWN_ITERS=2500 WARMUP_STEPS=20 TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024
ENCODER_RECURRENCE=1
```

Model parameters: 19,421,776
Submission size (int8+zlib): 15,753,781 bytes (code: 53,089 bytes)

### Reproduction

All defaults are baked into the script:
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Key Metrics

| Metric | Value |
|---|---|
| Pre-quant val_bpb | 1.2065 |
| Post-quant val_bpb (standard) | 1.2211 |
| Post-quant val_bpb (sliding window) | **1.1855** |
| Training time | 599,979ms (8,004 steps at ~75ms) |
| Peak memory | 16,592 MiB |
| Submission size (int8+zlib) | 15,753,781 bytes |
| Model parameters | 19,421,776 |
