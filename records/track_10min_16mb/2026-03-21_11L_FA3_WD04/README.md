# 2026-03-21_11L_FA3_WD04

**val_bpb = 1.1907** (single seed=1337) | artifact = 12.44 MB

11-layer model with Flash Attention 2, WD=0.04, SmearGate/BigramHash optimizer bug fix.

---

## Approach

This experiment adds one layer (9→11) to the Int6 3xMLP base and introduces Flash Attention 2
for kernel-level attention speedup. The key contribution of this record is identifying and fixing
a silent optimizer bug: two feature modules were never trained.

### Configuration

All defaults (no env overrides needed):

```
NUM_LAYERS=11 MLP_MULT=3 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
QUANT_BITS=6 USE_ZSTD=1 TIE_EMBEDDINGS=1 FP16_EMBED_EXPORT=1
ROPE_BASE=10000 TRAIN_SEQ_LEN=2048 EVAL_STRIDE=64
MUON_MOMENTUM=0.95 MUON_MOMENTUM_WARMUP_START=0.85 MUON_MOMENTUM_WARMUP_STEPS=500
MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0 TIED_EMBED_LR=0.05
SMEAR_GATE=1 BIGRAM_HASH=1 SWA=1 ORTHO_INIT=1 QAT=1
```

### Key Metrics

| Metric | Int6 3xMLP (9L) | This run (11L) |
|--------|----------------|---------------|
| model_params | 21,778,504 | 26,830,000 |
| val_bpb | 1.1708 | 1.1907 |
| artifact size | 15.2 MB | 12.44 MB |
| steps | ~12,500 | ~7,200 |
| step_avg | 48ms | ~83ms |

Note: 11L is slower per step (+35ms) due to two extra transformer blocks. Artifact is smaller
because the model has more parameters to encode the same information more efficiently, compressed
at zstd-22.

---

## Critical Bug Found and Fixed: Optimizer Coverage

**SmearGate and BigramHashEmbedding were not in any optimizer parameter group.**

Both modules were registered as submodules with `smear_gate: bool` and `bigram_hash: bool`
flags, but the optimizer setup code only iterated `base_model.blocks` for matrix/scalar params
and handled `tok_emb` / `lm_head` explicitly. Neither `smear_gate` nor `bigram_hash` appeared
anywhere in the optimizer groups.

**Consequence:** Both modules trained frozen from initialization:
- `SmearGate.gate` started at `zeros` → sigmoid(0) = 0.5 for all 512 channels, never updated
- `BigramHash.proj.weight` initialized with `nn.init.zeros_` → permanent zero projection
- `BigramHash.embed.weight` initialized with `std=0.01` noise → never refined

This means every prior submission using `SMEAR_GATE=1 BIGRAM_HASH=1` (the default) was
implicitly computing: embed(x) + 0.5 * embed(x-1) (fixed smear) and proj(embed_noise(bigram))
→ 0 (zeroed projection). The hash embedding contributed nothing; the smear gate was stuck at
0.5 globally instead of learning per-channel gates.

**The fix** (Option B — fix optimizer, not disable features):

```python
# bigram_hash.proj is a dense 2D weight — Muon handles matrix-shaped params
if base_model.bigram_hash is not None:
    matrix_params.append(base_model.bigram_hash.proj.weight)

# smear_gate is a 1D parameter vector — AdamW at scalar_lr
if base_model.smear_gate is not None:
    scalar_params.extend(base_model.smear_gate.parameters())

# bigram_hash.embed is an embedding table — AdamW alongside tok_emb
embed_params = [base_model.tok_emb.weight]
if base_model.bigram_hash is not None:
    embed_params.append(base_model.bigram_hash.embed.weight)
```

The same bug was independently identified by other participants at roughly the same time.

---

## What Changed vs Int6_3xMLP

| Component | Int6_3xMLP | 11L_FA3_WD04 |
|-----------|-----------|--------------|
| Layers | 9 | 11 |
| Flash Attention | ❌ (SDPA) | ✅ FA2 |
| SmearGate optimizer | ❌ (frozen bug) | ✅ fixed |
| BigramHash optimizer | ❌ (frozen bug) | ✅ fixed |
| SWA | ✅ | ✅ (still enabled) |
| rope_base | 10000 | 10000 |
| muon_momentum | 0.99 | 0.95 |

---

## Known Limitations

- val_bpb **regressed** vs 9L baseline (1.1907 vs 1.1708): 11 layers at 83ms/step completes
  fewer steps (~7200 vs ~12500) in 600s. Depth gain does not compensate for the step count loss
  at these scales with current hyperparameters.
- muon_momentum=0.95 is suboptimal; top entries use 0.99 with warmup from 0.92 over 1500 steps.
- grad_clip_norm=0 (disabled); top entries use 0.3.
- rope_base=10000 is the default; top entries use 50000 for better long-range positional encoding.
- These deficiencies informed the next experiment (11L_XSA_EMA_TTT).

---

## Environment

- Hardware: 8×H100 SXM (RunPod Parameter Golf template)
- PyTorch: 2.9.1+cu128
- Flash Attention: FA2 (`flash_attn` package, `--no-build-isolation` install required)
- FA3 (`flash_attn_interface`) not available — cross-device link error on RunPod filesystem

### Flash Attention install (RunPod)

```bash
pip install flash-attn --no-cache-dir --no-build-isolation
```

`--no-build-isolation` is required so the build finds the already-installed torch.
`--no-cache-dir` avoids a cross-device link error specific to RunPod's filesystem.

---

## Reproduction

```bash
cd /workspace
git clone https://github.com/mrdavtan/parameter-golf.git
cd parameter-golf && git checkout 11l-fa3-wd04
pip install flash-attn --no-cache-dir --no-build-isolation
pip install zstandard sentencepiece huggingface_hub
python3 data/cached_challenge_fineweb.py --variant sp1024

# IMPORTANT: unset any lingering env vars from prior runs
unset MLP_HIDDEN QUANT_BITS RUN_ID SEED

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_FA3_WD04/train_gpt.py
```

---

## Acknowledgments

- Optimizer bug independently identified (same bug found by others in the community)
- FA2 integration pattern from flash-attn library docs
- Int6 + zstd compression from 2026-03-20_Int6_3xMLP

## Author

GitHub: [@mrdavtan](https://github.com/mrdavtan)
Date: 2026-03-21
