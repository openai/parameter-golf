# Mamba3-GPTQ-Long-Context: Mamba3 Hybrid with Post-Training Quantization

Mamba3 hybrid architecture (Mamba3 SSM + attention) with full Hessian-based GPTQ post-training quantization and EMA weight averaging.

## Result

| Metric | Value |
|--------|-------|
| Pre-GPTQ BPB | 1.1755 |
| Post-GPTQ BPB | 1.1875 |
| Size | 15.51 MB ✅ |
| Steps | 5557 (10 min) |
| Hardware | 8×H100 (~108ms/step) |

## Architecture

```
Mamba3-Hybrid 10-Layer Hybrid (8 Mamba3 + 2 Attention):
├── Embedding(1024, 512) — tied
├── Encoder (5 layers):
│   ├── Blocks 0-3: Mamba3 + MLP (relu²)
│   └── Block 4: Attention (GQA 8/4) + MLP
├── Decoder (5 layers):
│   ├── Blocks 5-8: Mamba3 + MLP
│   └── Block 9: Attention (GQA 8/4) + MLP
└── RMSNorm → tied linear → softcap → loss
```

- 10 layers: 8 Mamba3 SISO + 2 GQA attention (layers 4, 9)
- model_dim=512, mlp_mult=2, vocab_size=1024 (sp1024 BPE tokenizer)
- U-Net skip connections across all layer types
- train_seq_len=16384 (long context for Mamba's linear-time advantage)

## Compression Pipeline

1. Train at bf16 for 10 min on 8×H100 (~108ms/step, ~5557 steps)
2. Apply EMA weights (decay=0.997) — smooths weights, improves GPTQ resilience
3. AR self-generation: model generates 64 × 2048 calibration tokens (temp=0.8, seeded) — autoregressive calibration consistent with current SOTA, qualifying under competition rules
4. Full Hessian GPTQ: int6, block-128, column reordering, 5-percentile clip search
5. Int8 serialization + LZMA preset=9 compression
6. Sliding window roundtrip validation

## Journey & Observations

Several days of exploration went into this submission, motivated by genuine love for the SSM design and OpenAI's own wishlist item ("State-space models + super long context").

**Pure Mamba doesn't work — hybrid does.** A pure Mamba architecture underperforms, but mixing in a small number of attention layers produces a model that converges faster than pure attention in the early training steps. The sweet spot found was 2 attention layers at the boundaries of a U-Net encoder-decoder, with Mamba3 everywhere else.

**The fundamental hardware problem.** Mamba is less parallelizable than attention and currently less optimized for H100s. FA3 is specifically designed for Hopper hardware, exploiting its async execution and warp specialization to dramatically accelerate attention — combined with H100's 2× bf16 FLOPS and 2× memory bandwidth, SOTA runs at ~86ms/step on H100 vs Mamba3-Hybrid's ~108ms/step. That gap compounds over 10 minutes: SOTA gets ~7000 steps, Mamba3-Hybrid gets ~5557. Without FA3, Mamba would likely be competitive — possibly ahead.

**Hyperparameter sensitivity.** Mamba3-Hybrid requires its own hyperparameter tuning. Learning rate in particular is much more sensitive (LR=0.02 optimal; 0.03 noticeably worse, 0.04 much worse). Most of the competition's accumulated wisdom — BigramHash, SmearGate, OrthoInit — was tuned for a 7000-step attention regime and actively hurts here. The one major exception is **GPTQ**, which was the key unlock: Mamba weights are denser than attention weights and resist standard int8 compression badly, but full Hessian GPTQ finally made it possible to compress properly and fit under 16MB.

**A note on sliding window evaluation.** This submission uses sliding window eval because the leaderboard uses it, but it's worth being honest: sliding eval is measuring something fundamentally different from the training objective — it's not obviously "better", it's just a different metric. The effect here is negligible anyway, since training at seq_len=16384 means every token already sees up to 16K tokens of context. The real advantage of the Mamba hybrid is **linear-time long context scaling**, which would show up directly in BPB without any eval tricks. Sliding eval obscures this by handing all architectures richer context regardless of how efficiently they use it.

**Where the juice is.** On unlimited compute, this architecture likely converges lower than the current attention SOTA — early convergence curves suggest it. And it would produce a better model *in practice*: linear-time inference, genuine long-context capability, not just a benchmark artifact from sliding eval. The 10-minute constraint and FA3 speed advantage for attention are what prevent this from being a record submission today.

## Key Findings

**Mamba3-Hybrid converges faster per step early, but SOTA pulls ahead with more steps**

Early convergence comparison (same batch size, same wallclock):

| Step | Mamba3-Hybrid BPB | SOTA BPB |
|------|-----------|----------|
| 1000 | **1.2951** | 1.2986 |
| 2000 | **1.2324** | 1.2365 |
| 3000 | 1.1989 | **1.1953** |
| Final | 1.1947 (fewer steps) | **1.1576** (more steps) |

Mamba3-Hybrid wins early but falls behind as SOTA accumulates more steps — the step time gap is the decisive factor.

**EMA meaningfully helps post-GPTQ**
- Pre-quant improvement: −0.0009 BPB (small)
- Post-GPTQ improvement: −0.0024 BPB (meaningful)
- Smoother EMA weights quantize better

**SOTA features don't transfer to Mamba3-Hybrid**
- BigramHash + SmearGate + OrthoInit made things worse (1.2166 vs 1.2047)
- OrthoInit disrupts Mamba's tuned internal initialization
- BigramHash needs ~7000 steps to converge; Mamba3-Hybrid only gets ~4400

**Mamba3 > Mamba2**: ~0.01 BPB improvement AND smaller compressed size

**Optimal config found**: 2 attention layers, LR=0.02, batch=524k, warmdown=4000

## Running

```bash
# 8×H100 (~10 min)
NCCL_TIMEOUT=1800 USE_GPTQ=1 USE_LZMA=1 USE_MAMBA3=1 \
NUM_LAYERS=10 ATTN_LAYERS=4,9 \
MODEL_DIM=512 MLP_MULT=2 TRAIN_SEQ_LEN=16384 \
TRAIN_BATCH_TOKENS=524288 VAL_BATCH_SIZE=131072 \
MATRIX_LR=0.02 SCALAR_LR=0.02 WARMDOWN_ITERS=4000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
SEED=1337 VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-04-02_Mamba3-Hybrid_GPTQ/train_gpt.py
```

## Installing Mamba-3

```bash
pip install causal-conv1d>=1.4.0 --no-build-isolation

MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --no-build-isolation --no-deps \
  git+https://github.com/state-spaces/mamba.git
```

See `setup.sh` for full environment setup including automatic CUDA version detection.
