# Ternary Reasoner — Capsule-Feedback Universal Transformer

**Author**: Aki Gogikar (OneNewAI)

Submission for the 10-minute / 16MB track of OpenAI's Parameter Golf Challenge.

## Core Thesis

**Low-bit inference + structured semantic state + backward semantic correction.**

Standard transformers do a single forward pass. The Ternary Reasoner iterates:
encode once, then run multiple decoder passes where each pass is corrected by a
compressed semantic sketch from the previous pass via Hadamard-gated adapters.
Recurrent capsule state accumulates global structure across iterations.

## What Makes This Different

While most submissions use int6/int8 quantization, we go to the extreme: **ternary weights {-1, 0, +1}**. This packs ~87M parameters into ~12MB compressed — 3-4x more parameters than int6 submissions at the same budget. The trade-off (noisier per-parameter signal) is compensated by:

1. **Iterative backward correction**: later decoder layers send compressed semantic sketches back to correct earlier representations — the model gets multiple shots at refining its answer
2. **Recurrent capsule state**: structured semantic slots that persist and accumulate across correction iterations, giving the model persistent working memory
3. **Hadamard-gated adapters**: element-wise multiplicative modulation (not just additive bias) — richer correction signal

Combined with proven competition techniques (XSA, LeakyReLU², BigramHash, VRL, Partial RoPE, EMA, GPTQ-lite, n-gram cache, legal TTT), this creates a unique architecture that maximizes both parameter count AND inference-time reasoning.

## Architecture (Default: 12L/768d)

- **Ternary U-Net trunk**: 12-layer encoder-decoder with skip connections, ~87M ternary params
- **XSA**: Exclusive Self-Attention on last 4 layers (zero parameters, forces context reliance)
- **LeakyReLU(0.5)²**: proven -0.003 BPB over ReLU²
- **Iterative correction**: 1 feedback pass during training, 2 at eval (configurable)
- **Hadamard-gated feedback**: multiplicative + additive backward semantic correction
- **Recurrent capsules**: 16 structured state slots persisting across iterations
- **Partial RoPE**: 16/96 dims rotated, rest attend without position
- **VRL**: first-layer values blended into deep-layer attention (layers 10+)
- **LN Scale Damping**: 1/sqrt(layer+1) for training stability
- **BigramHash**: 4096-bucket bigram hashing for local context
- **EMA** (decay=0.997): weight averaging for smoother quantization
- **GPTQ-lite**: per-row clip percentile search before ternary packing

## Eval Stack

- **Sliding window** (stride=64) with temperature scaling
- **N-gram cache** (order=5, entropy-adaptive mixing)
- **Legal score-first TTT** (3 epochs, feedback scope, SGD with momentum)

## Training Configuration

- **Muon optimizer**: lr=0.025, momentum=0.95, WD=0.04, 5 Newton-Schulz steps
- **Batch**: 786K tokens/step, seq_len=2048
- **Warmdown**: 50% of wallclock time (time-based, not step-based)
- **Gradient clipping**: 0.3
- **8xH100 SXM**: 10 minutes training, 10 minutes eval

## Run

```bash
bash setup.sh
conda activate golf
bash run_cuda_feedback.sh
```

### Ablation variants

```bash
# Minimal baseline (ternary trunk only, no extras)
FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 BIGRAM_HASH_ENABLED=0 VRL_ENABLED=0 \
XSA_START_LAYER=-1 EMA_ENABLED=0 GPTQ_LITE_ENABLED=0 TTT_ENABLED=0 \
NGRAM_CACHE_ENABLED=0 bash run_cuda_feedback.sh

# Full stack minus feedback (isolate feedback contribution)
FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 bash run_cuda_feedback.sh

# Training-only (no eval tricks, fast iteration)
SLIDING_EVAL=0 TEMP_SCALING=0 TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 bash run_cuda_feedback.sh

# Quick smoke test (1 GPU, 60s)
ITERATIONS=200 MAX_WALLCLOCK_SECONDS=60 SLIDING_EVAL=0 TEMP_SCALING=0 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 NPROC_PER_NODE=1 bash run_cuda_feedback.sh
```

## Key env knobs

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full config table.

## Ablation plan

See [ABLATION_TODO.md](ABLATION_TODO.md) for the prioritized experiment checklist.
