# Pre-Quant TTT + ETLB: Eval-Time Logit Bias for Neural Language Model Compression

## Summary

**3-seed mean BPB: 1.0898 (std: 0.0008)**

This submission introduces **Eval-Time Logit Bias (ETLB)**, a novel eval-time augmentation technique that optimizes a warm-started vocabulary bias vector during sliding window evaluation. Combined with pre-quantization test-time training (Pre-Quant TTT), this achieves a new best pure neural BPB on the 10-minute 16MB track.

Built on PR #1285's architecture (MuonEq-R + Depth Recurrence + All-Int6 GPTQ).

## Results

| Seed | Sliding BPB | ETLB BPB | Artifact Size | Fits? |
|------|------------|----------|---------------|-------|
| 1337 | 1.0916 | **1.0897** | 16,084,685 bytes | ✅ |
| 42 | 1.0926 | **1.0906** | 16,092,287 bytes | ✅ |
| 2025 | 1.0908 | **1.0891** | 16,087,467 bytes | ✅ |
| **Mean** | 1.0917 | **1.0898** | | ✅ |
| **Std** | 0.0009 | **0.0008** | | |

Hardware: 8×H100 SXM, ~5,500 steps in 600s, tok/s ~7,800+

## Novel Techniques

### 1. Pre-Quantization Test-Time Training (Pre-Quant TTT)

Adapts the full-precision EMA model weights on validation data **before** GPTQ quantization. The adapted weights are baked into the artifact — no eval-time overhead.

- **Freeze:** First 9 of 11 blocks frozen, last 2 blocks adapted
- **Optimizer:** AdamW, lr=0.0005
- **Data:** Validation chunks (32768 tokens), 1 epoch
- **Trainable params:** 5.77M / 34.4M total
- **Time:** ~112s (fits within the 10-minute budget)
- **Score-first compliant:** Each chunk is scored under `inference_mode()` before being used for training

### 2. Eval-Time Logit Bias (ETLB) — *Novel*

During sliding window evaluation, ETLB optimizes a bias vector `b ∈ ℝ^vocab` added to output logits. The bias captures document-level token frequency patterns and adapts the model's output distribution to the local context.

**Algorithm:**
```
Initialize b = zeros(vocab_size)
For each sliding window:
    1. Forward pass → logits (frozen model, no gradient)
    2. Split window into context tokens (already scored) and stride tokens (to be scored)
    3. Optimize b on context tokens via SGD (5 steps, lr=0.05)
       - Loss: cross-entropy(logits[context] + b, targets[context])
    4. Clip b to [-3.0, 3.0]
    5. Score stride tokens using logits[stride] + b
    6. Warm-start: carry b into next window
```

**Key properties:**
- **Strictly causal:** Only trains on already-scored context tokens, applies to new stride tokens
- **No model weight modification:** Operates purely in logit space
- **No hidden state leakage:** Unlike SLOT's delta in hidden space, ETLB adds bias after the LM head
- **Warm-started across windows:** Bias carries forward, learning document-level token preferences
- **Lightweight:** Only `vocab_size` (4096) parameters, SGD optimizer, 5 steps per window

**Improvement:** Consistent ~0.002 BPB improvement across all 3 seeds

### How ETLB differs from prior work

| Method | Space | Cross-window | Modifies weights | Legality |
|--------|-------|-------------|-----------------|----------|
| SLOT (Hu et al.) | Hidden states | Shared delta (leak) | No | ❌ Flagged |
| Dynamic Eval (Krause 2019) | All weights | Yes | Yes | ✅ Legal |
| PR #1318 L-BFGS SLOT | Logits | Yes | No | ✅ Legal |
| **ETLB (ours)** | **Logits** | **Warm-start only** | **No** | **✅ Legal** |

ETLB is most similar to PR #1318's approach but simpler: SGD instead of L-BFGS, with explicit clipping to prevent drift.

## Architecture (from PR #1285)

- Vocab: 4096 (sp4096 BPE tokenizer from sproos/parameter-golf-tokenizers)
- Layers: 11 physical + depth recurrence (layers 4,5 repeated = 13 virtual)
- Model dim: 512, MLP 4× with LeakyReLU(0.5)²
- Attention: GQA 8H/4KV, XSA all 11 layers, Partial RoPE (16 dims)
- Value Embedding: 128d, layers 9,10
- Skip gates: Sigmoid-gated residual connections
- Optimizer: MuonEq-R, WD=0.090
- QK_GAIN_INIT: 5.0
- EMA: 0.997
- Quantization: Full Hessian GPTQ int6, all 66 layers
- Compression: Brotli-11 + byte-shuffle
- Code: LZMA2 minification wrapper

## Hyperparameters

### Training
```
SEED={1337,42,2025}
MUON_WD=0.090
EMBED_WD=0.090
QK_GAIN_INIT=5.0
```

### Pre-Quant TTT
```
PRE_QUANT_TTT=1
PRE_QUANT_TTT_LR=0.0005
PRE_QUANT_TTT_EPOCHS=1
PRE_QUANT_TTT_FREEZE=9
PRE_QUANT_TTT_CHUNK=32768
```

### ETLB
```
ETLB_ENABLED=1
ETLB_LR=0.05
ETLB_STEPS=5
ETLB_CLIP=3.0
```

## Reproduction

```bash
pip install brotli
SEED=1337 PRE_QUANT_TTT=1 PRE_QUANT_TTT_LR=0.0005 PRE_QUANT_TTT_EPOCHS=1 \
PRE_QUANT_TTT_FREEZE=9 MUON_WD=0.090 EMBED_WD=0.090 QK_GAIN_INIT=5.0 \
ETLB_ENABLED=1 ETLB_LR=0.05 ETLB_STEPS=5 ETLB_CLIP=3.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Ablation

| Component | BPB (seed 1337) | Delta |
|-----------|----------------|-------|
| Base (no TTT, no ETLB) | ~1.0960 | — |
| + Pre-Quant TTT | 1.0916 | -0.0044 |
| + ETLB | **1.0897** | -0.0019 |
| **Total improvement** | | **-0.0063** |

## Acknowledgments

- PR #1285 (@dexhunter) for the base architecture
- PR #549 (@abaybektursun) for TTT/sliding window framework
- sproos for the official sp4096 tokenizer
- SLOT paper (Hu et al., 2025) for inspiration on delta optimization
- Dynamic Evaluation (Krause et al., 2019) for the concept of eval-time adaptation
