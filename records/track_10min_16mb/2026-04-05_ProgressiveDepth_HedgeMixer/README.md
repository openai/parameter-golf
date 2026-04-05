# Progressive Depth + Hedge Mixer (Depth Recurrence)

**val_bpb: 1.1441** (3-seed mean, std 0.0051) | **~15.88 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM, PyTorch 2.5.1)

| Seed | Steps | Step avg | Roundtrip bpb | Sliding bpb | **Hedge bpb** | Eval time |
|------|-------|----------|---------------|-------------|---------------|-----------|
| 1337 | 5,668 | 105.8ms | 1.2302 | 1.1965 | **1.1441** | 580s |
| 42 | 5,170 | 116.1ms | 1.2298 | 1.1962 | **1.1491** | 580s |
| 7 | 5,405 | 111.0ms | 1.2286 | 1.1952 | **1.1390** | 587s |
| **Mean** | **5,414** | **111.0ms** | **1.2295** | **1.1960** | **1.1441 (std 0.0051)** | **~582s** |

## Architecture: Depth Recurrence

Instead of 9-11 unique transformer blocks, we use **3 shared blocks repeated 4 times** (12 effective layers). This trades unique parameters for effective depth, fitting more computation into the 16MB budget.

```
3 blocks × 4 repeats = 12 effective layers, 17.14M params
```

### Key components

- **Cross-Repeat Skip**: Each block receives a weighted residual from its own output in the previous repeat, turning stateless recurrence into stateful. Per-repeat learned scales.
- **Loop Embedding**: Learned per-layer vector added before each block — depth-wise positional encoding for shared weights.
- **Value Embeddings**: 2 extra embedding tables mixed into the residual stream at each effective layer with learned scales.
- **XSA (Exclusive Self-Attention)**: On last 4 effective layers — prevents attention collapse in deep recurrent models.
- **LeakyReLU(0.5)²**: Better gradient flow than ReLU² for deep/recurrent models.

### Model config

| Parameter | Value |
|-----------|-------|
| Layers × Repeats | 3 × 4 (12 effective) |
| Model dim | 832 |
| Heads / KV heads | 8 / 4 |
| MLP multiplier | 2× |
| Vocab size | 1024 (SP BPE) |
| Logit softcap | 30.0 |

## Key Innovation: Progressive Depth Training

Unique to shared-weight architectures — train with increasing recurrence depth over time:

| Phase | Time fraction | Repeats | Step speed |
|-------|--------------|---------|------------|
| Phase 1 | 0–40% | 2 | ~80ms |
| Phase 2 | 40–65% | 3 | ~90ms |
| Phase 3 | 65–100% | 4 | ~105ms |

This gives **+30% more training steps** compared to training at full depth the entire time (5,414 vs ~4,300 steps). Early phases are cheaper because fewer repeats = faster forward/backward pass. The model learns basic representations quickly at shallow depth, then refines with full recurrence.

`torch._dynamo.reset()` + recompile on phase transitions (~10s × 2 = 20s overhead).

Controlled by env var: `PROG_DEPTH="0.4:2,0.65:3,1.0:4"`

## Eval: Hedge Mixer (5-Expert Online Ensemble)

Eval-time improvement via online mixture of 5 experts using the Hedge (multiplicative weights) algorithm:

| Expert | Description |
|--------|-------------|
| Neural | Model's own logits (log-softmax) |
| Unigram | Global token frequency with Laplace smoothing |
| Bigram | Conditional P(token | prev_token) |
| Trigram | Hashed trigram context (65K buckets) |
| Entropy | Model's own entropy as calibration signal |

The mixer processes validation windows sequentially, updating n-gram statistics and expert weights after scoring each window. Initial bias toward the neural expert (log_weight = 2.0). Learning rate η = 0.1.

**Hedge provides −0.052 bpb improvement** over sliding window eval (1.1960 → 1.1441 mean).

### Timing budget

| Phase | Time |
|-------|------|
| Training (10 min cap) | 600s |
| Roundtrip eval | ~14s |
| Sliding window eval | ~67s |
| Hedge Mixer eval | ~582s |

## Training details

- **Optimizer**: Muon (matrix params) + Adam (scalars, embeddings)
- **LR**: matrix 0.012, scalar 0.012, tied_embed 0.015
- **Muon WD**: 0.04
- **Warmdown**: 3000 steps (wallclock-proportional)
- **SWA**: During warmdown, every 50 steps, 13-16 checkpoints averaged
- **Grad clip**: 0.3
- **Quantization**: int8 + zstd-22 (~15.88 MB artifact)

## Evolution & Prior PRs

This submission is the result of iterative development across several PRs in this repo:

| PR | Date | Score | What changed |
|----|------|-------|-------------|
| [#148](https://github.com/openai/parameter-golf/pull/148) | Mar 20 | 1.2196 | Depth recurrence (3×4), cross-repeat skip, value embeddings, sliding window eval |
| [#784](https://github.com/openai/parameter-golf/pull/784) | Mar 25 | 1.2065 | + XSA(4), LeakyReLU², GPTQ-lite, zstd-22 |
| [#835](https://github.com/openai/parameter-golf/pull/835) | Mar 26 | 1.1980 | + Progressive depth training (+30% steps) |
| [#856](https://github.com/openai/parameter-golf/pull/856) | Mar 26 | 1.1454 | + Hedge Mixer (5-expert eval-time ensemble) |
| **This PR** | Apr 5 | **1.1441** | Clean submission with 3-seed validation |

This PR supersedes the above with a clean diff and proper 3-seed statistical validation.

## Lineage

- Depth recurrence architecture is original to this submission line
- XSA from PR #198 (unnir), LeakyReLU² from PR #493 (parinzee)
- SWA and Muon WD from modded-nanogpt community
