**val_bpb: 1.4525** (sliding window, stride=128, GA+VR combined) | **13.2 MB** (control model) | 1xRTX3090, 1000 steps, 131K batch

## Non-record: Value Residual (-0.015 BPB) + Gated Attention (-0.003 BPB) with ablations

Two novel architecture modifications and one negative result. The goal is to share validated techniques with controlled ablation data.

### Contributions

1. **Value Residual (ResFormer)** -- -0.015 BPB standalone. Caches raw V vectors from layer 0 and mixes them into all subsequent layers: `V_n = lambda1 * V_0 + lambda2 * V_current`. Both lambdas are learnable (init 0.5), 18 scalars total for 9L. Preserves token identity through depth, especially beneficial for deep-narrow architectures (512d). Based on arXiv:2410.17897 (ACL 2025). Enable: `VALUE_RESIDUAL=1`.

2. **Gated Attention** -- -0.003 BPB standalone. Per-head sigmoid gate after SDPA: `Y' = Y * sigmoid(X @ W_gate + b_gate)`. Bias init 4.0 (sigmoid ~0.98, near no-op at start). Eliminates attention sinks where softmax forces distribution over irrelevant keys. ~37K params for 9L/8H. Based on arXiv:2505.06708 (NeurIPS 2025 Best Paper). Enable: `GATED_ATTENTION=1`.

3. **PPM-C Context Mixer** -- +0.0018 BPB (negative result). Classical Prediction by Partial Matching (order 2) blended with neural softmax at eval time (`mixed = 0.95*neural + 0.05*ppm`). On SmearGate+BigramHash models, the neural predictions already capture bigram patterns; PPM just dilutes them. Zero artifact cost but no benefit.

The two positive techniques **stack additively** for -0.017 BPB combined (no interference).

### Ablation Results

Controlled A/B test: v1024 9L 2xMLP, SmearGate + BigramHash(4096) + OrthoInit + WD 0.04, 131K batch, 1000 steps, RTX 3090.

| Config | Sliding BPB | Delta vs Control |
|--------|-------------|------------------|
| Control (no novel techniques) | 1.4697 | -- |
| Gated Attention only | 1.4665 | -0.0032 |
| **Value Residual only** | **1.4546** | **-0.0151** |
| **GA + VR combined** | **1.4525** | **-0.0172** |
| PPM-C (alpha=0.95, order=2) | 1.2900* | +0.0018 (worse) |

*PPM-C tested on stronger pre-trained model (524K batch, 2000 steps, baseline 1.2882 BPB).

### Relationship to community techniques

**Value Residual vs VE128 (Shared Value Embeddings):** Both preserve value information across depth, but through different mechanisms. VE128 shares a learned embedding matrix; Value Residual skip-connects layer 0's computed V vectors. Whether they are complementary or redundant has not been tested.

**Gated Attention vs XSA:** Both address attention quality through different failure modes. XSA removes self-value bias; Gated Attention allows heads to suppress output entirely. They may be complementary.

### Reproducibility

Ablation cost: ~$0.70 (3x RTX 3090 pods, ~20 min each).

```bash
TORCHDYNAMO_DISABLE=1 VOCAB_SIZE=1024 NUM_LAYERS=9 MLP_MULT=2 \
TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=131072 ITERATIONS=1000 \
SMEAR_GATE=1 BIGRAM_HASH=1 BIGRAM_BUCKETS=4096 ORTHO_INIT=1 \
WEIGHT_DECAY_MUON=0.04 WEIGHT_DECAY_ADAM=0.04 \
GATED_ATTENTION=1 VALUE_RESIDUAL=1 \
EVAL_SEQ_LEN=1024 EVAL_STRIDE=128 QUANT_BITS=6 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### What we'd do with more compute

A production run (11L MLP3x + full community stack + VR + GA, 9500 steps, 524K batch) is in progress on 1xA6000. If Value Residual's -0.015 BPB holds at the frontier, it could push SOTA from ~1.125 to ~1.110 BPB. Results will be submitted in a follow-up record PR if competitive.

### Files

- `README.md` -- This writeup
- `submission.json` -- Submission metadata
- `train_gpt.py` -- Training script with Value Residual, Gated Attention, XSA, EMA, Partial RoPE, LN Scale implementations
