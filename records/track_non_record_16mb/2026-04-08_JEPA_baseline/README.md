# JEPA Baseline — First LLM-JEPA from Scratch in Parameter Golf

**val_bpb: 1.2699** | 135MB | 8×H100 SXM, 600s | Seed 1337

## What is JEPA?

Joint Embedding Predictive Architecture (JEPA) — originally from vision models
(LeCun et al.) — predicts embeddings instead of tokens. Rather than training
against absolute truth (one correct next token), the model learns to predict
the meaning of what comes next in embedding space.

This is the first implementation of JEPA for language model pretraining from
scratch in the Parameter Golf challenge.

## Results

| Seed | Steps | ms/step | val_bpb | Model Size |
|------|-------|---------|---------|------------|
| 1337 | 7000  | 64-69ms | 1.2699  | 135MB      |

## val_bpb progression

| Step | val_bpb |
|------|---------|
| 1000 | 1.3831  |
| 2000 | 1.3248  |
| 3000 | 1.3010  |
| 4000 | 1.2855  |
| 5000 | 1.2776  |
| 6000 | 1.2702  |
| 7000 | 1.2646  |
| final (post-quantization) | 1.2699 |

## Architecture

### JEPAPredictor
Small 2-layer MLP mapping context embedding → predicted target embedding.

### Target Encoder (EMA)
Full copy of all 11 transformer layers — frozen, no gradients.
Updated each step: target = 0.996 × target + 0.004 × context

### Forward Pass
Two parallel paths through the full sequence:
- Context encoder (gradients flow normally)
- Target encoder (EMA, no gradients)

Loss = CE + 0.1 × MSE(normalize(predicted_emb), normalize(target_emb))

## Known Limitations & Next Steps

Model size: 135MB — over 16MB record track limit.
Two full encoder copies doubles parameter count.

Planned:
1. Mid-layer JEPA — target encoder on layers 0-5 only
2. Shared embeddings between encoders
3. Smaller model_dim (384) to fit 16MB
4. Hybrid — JEPA loss in middle layers, CE throughout

## References
- LLM-JEPA: https://arxiv.org/abs/2509.14252
- I-JEPA: https://arxiv.org/abs/2301.08243
