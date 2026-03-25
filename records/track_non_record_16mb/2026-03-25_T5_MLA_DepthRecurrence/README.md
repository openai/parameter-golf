# T5 Entry: Phase-Based Depth Recurrence + MLA + Graduated Precision

## Core Idea

Instead of 11 unique layers (SOTA), we use **8 unique transformer blocks repeated across 4 specialized phases** to achieve **40 effective layers at full d=512 width** — with only 24M unique parameters fitting in ~13MB.

### Phase-Based Recurrence (not ALBERT-style uniform cycling)

```
ALBERT-style:  A B C D E A B C D E A B C D E A B C D E
               ← every layer must be a generalist →

Ours:          [A B]×5    [C D]×5    [E F]×5    [G H]×5
               ← lexical → ← syntax → ← semantic → ← predict →

Each phase pair SPECIALIZES for its depth range.
Within each phase, recurrence = iterative refinement.
```

Transformer layers exhibit clear phase differentiation: early layers learn lexical/positional features, middle layers handle syntax, late layers do prediction. Sharing across phases forces generalization; sharing within phases enables iterative refinement — analogous to fixed-point iteration in Universal Transformers.

### Graduated Precision

Early layers (coarse features) quantized to FP4, late layers (fine prediction) to Int6. The model trains with per-layer QAT simulation and full FP8 training with stochastic rounding on H100.

## Variants

| Variant | Architecture | Unique Params | Artifact | Depth vs SOTA |
|---|---|---|---|---|
| **looped40** | 8 unique × 4 phases × 5 reps, d=512, MLP3x | 24M | ~13 MB | 3.6× |
| deep20 | 20 unique layers, d=384, MLP3x | 34M | ~15.5 MB | 1.8× |

## Key Techniques

- **MLA** (Multi-Head Latent Attention): Low-rank KV compression, ~20% fewer attention params
- **DeepNorm init**: Output projections scaled by (8·N)^(-1/4) for deep stability
- **Muon + AdamW**: Newton-Schulz orthogonalization for 2D weights
- **QK-Clip**: Post-step attention score rescaling (Kimi K2, Section 3.3)
- **Z-Loss**: Logit magnitude penalty (PaLM/Gemini)
- **FP8 Training**: All persistent state in FP8 with stochastic rounding (CUDA)
- **EMA**: Exponential moving average for evaluation weights

## Gradient Sharing Benefit

With 8 unique blocks appearing 5× each, every weight set receives 5× gradient signal per step — acting as implicit gradient averaging. This reduces noise and improves convergence, partially compensating for fewer training steps at 40-layer depth.

## Local Validation

Trained a 13L×448d baseline on Mac MPS for 6 hours (25K steps, 1 FineWeb shard):
- **val_bpb = 1.50** at ~1% of the full compute budget
- Loss curve: 6.69 → 2.81 (stable, no NaN)
- Quantization roundtrip gap: 0.002 BPB

## Running

```bash
# H100 (8x)
VARIANT=looped40 torchrun --standalone --nproc_per_node=8 train_gpt.py
VARIANT=deep20 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Local smoke test
VARIANT=test DEVICE=cpu ITERATIONS=10 VAL_LOSS_EVERY=0 python3 train_gpt.py
```
