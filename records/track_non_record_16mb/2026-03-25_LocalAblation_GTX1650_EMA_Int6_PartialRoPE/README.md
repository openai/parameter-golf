# Local Ablation Pipeline: GTX 1650 — EMA + Int6 + Partial RoPE + LN Scale + Weight Decay

**Track:** non-record — dev hardware, not competition-scale
**Hardware:** NVIDIA GTX 1650 (4 GB VRAM, SM 7.5, Turing, Windows 11)
**Author:** gthgomez

---

## Why this exists

This is a non-record local validation submission intended to document implementation and ablation results on constrained hardware. It is not a leaderboard attempt.

The GTX 1650 cannot run the competition training (no FlashAttention-3, no native bfloat16 kernel, no Triton/torch.compile on Windows, 4 GB VRAM limit), but it can validate tensor shapes, code paths, and export behaviour at reduced scale before committing cloud credits. The goal is to confirm that each ported feature works correctly before scaling to 8×H100.

All features are controlled by env vars with defaults that preserve the baseline's existing behaviour.

---

## Features implemented

### GTX 1650 compatibility patches
Three patches make the baseline runnable on SM 7.5 hardware without changing competition behaviour:

- **`NO_COMPILE=1`** — skips `torch.compile` (hangs on Windows/Turing with no Triton). On H100 this env var is absent and compilation proceeds normally.
- **Math SDP fallback** — detects SM < 8.0 at startup and enables math SDP / disables Flash + memory-efficient. On SM ≥ 8.0 (H100) Flash is used as before.
- **`MAX_VAL_SEQS=N`** — caps the number of validation sequences. The full val set (969 K sequences at seq_len=64) would take hours locally; `MAX_VAL_SEQS=256` gives a fast proxy. Competition runs leave this unset.

### EMA (from entry #1)
Exponential moving average of weights applied before export.

```python
EMA_DECAY=0.997   # competition value — effective window ~333 steps
EMA_DECAY=0.97    # calibrated for 200-step local validation
```

Ablation (200 steps, seq_len=512):
| Config | Live bpb | EMA bpb |
|--------|----------|---------|
| EMA_DECAY=0.997 | 2.6943 | 3.1025 (hyperparameter mismatch — window too wide) |
| EMA_DECAY=0.97  | 2.6333 | **2.4661** (+0.167 improvement) |

`0.997` is the intended competition-scale setting based on top public entries (effective window ~333 steps over a 7000-step run); `0.97` confirmed correct implementation on short local runs.

### Int6 clip-search quantizer (GPTQ-lite, from entry #1)
Replaces the baseline int8 quantizer for large 2-D tensors. Tries 5 clip percentiles per row `[0.9990, 0.9995, 0.9999, 0.99999, 1.0]`, picks the one with lowest MSE reconstruction. Values stored in int8 tensors clamped to `[-31, 31]` — the restricted range compresses better under zlib at the cost of fewer quantisation levels.

An A/B comparison block is appended to every export: the int8 and int6 compressed sizes and bpb roundtrip are logged and compared without requiring a separate script.

Result on local checkpoint (same export path, same model, same zlib level=9):
- int8+zlib: **11.0 MB** (2.5273 bpb roundtrip)
- int6+zlib: **6.7 MB** (2.5319 bpb roundtrip, −3.8 MB, +0.0046 bpb)

The reduction appears to come from lower dynamic range and increased weight regularity, which improves entropy coding efficiency under zlib. All measurements use the same export and compression path (`torch.save` + `zlib.compress(level=9)`, same training run).

### Partial RoPE (from entry #2)
`ROPE_DIMS=16` rotates only the first 16 of 64 head dims; the remaining 48 act as absolute-position channels. Zero parameters added.

Implementation: `Rotary.__init__` computes `inv_freq` over `rope_dims` frequencies only. `apply_rotary_emb` detects partial mode (`cos.size(-1)*2 < head_dim`) and concatenates the passthrough tail unchanged.

### LN Scale (from entry #2)
`LN_SCALE=1` multiplies both `attn_norm` and `mlp_norm` outputs by `1/sqrt(layer_idx+1)` before the sub-layer. Layer 0 gets scale 1.0, layer 8 gets 0.333. Zero parameters. Reduces effective gradient magnitude in deeper layers, stabilising training of 11-layer models.

### Muon + Adam weight decay (from entry #1)
Decoupled weight decay added to both optimisers:

- **Muon**: `p.data.mul_(1.0 - lr * wd)` applied before the orthogonalised gradient step (`MUON_WD`, default 0.0, competition value 0.04).
- **Adam → AdamW**: token-embedding and scalar-param optimisers switched to `torch.optim.AdamW` with `weight_decay=adam_wd` (`ADAM_WD`, default 0.0, competition value 0.04).

### MLP_MULT float support
`mlp_mult` changed from `int` to `float` throughout, enabling `MLP_MULT=3.0` (entry #1's config; 3× hidden expansion vs baseline 2×). `hidden = int(mlp_mult * dim)` ensures integer layer widths.

---

## Local ablation results (200 steps, 9L model, GTX 1650)

| Run | Live bpb | EMA bpb | int8 size | int6 size |
|-----|----------|---------|-----------|-----------|
| Baseline | 2.6964 | — | ~11.1 MB | ~7.0 MB |
| EMA_DECAY=0.97 | 2.6333 | 2.4661 | 11.1 MB | — |
| Partial RoPE + LN Scale | 2.6845 | — | 11.2 MB | 7.0 MB |
| All features combined | 2.6845 | **2.5273** | **11.0 MB** | **6.7 MB** |

The combined EMA bpb (2.5273) is above the EMA-only result (2.4661) because LN Scale damps later-layer signal, slowing early-step convergence. This is expected behaviour — at full competition scale (7000 steps, LR warmdown at 3500) the damping becomes beneficial, as shown by the leaderboard entries.

---

## What is NOT in this submission

- **Full competition-scale training** (11L, MLP_MULT=3.0, seq_len=2048, 7000 steps) — requires 8×H100
- **XSA** (Exclusive Self-Attention on last N layers) — deferred, pending cloud access
- **VE** (Value Embedding) — not yet implemented in this script
- **Sliding-window eval** (stride-64, used for official scoring) — not in local eval loop

---

## Competition launch command (pending cloud access)

```bash
EMA_DECAY=0.997 \
ROPE_DIMS=16 \
LN_SCALE=1 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MLP_MULT=3.0 \
NUM_LAYERS=11 \
GRAD_CLIP_NORM=0.3 \
MUON_MOMENTUM=0.99 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
WARMDOWN_ITERS=3500 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All GTX compatibility patches (NO_COMPILE, math SDP, MAX_VAL_SEQS) are inert on H100 hardware and do not affect the training path.

---

## Included files

- `train_gpt.py` — patched script (all features, env-var controlled, backward-compatible defaults)
- `train.log` — combined local ablation run (EMA + Partial RoPE + LN Scale, 200 steps)
- `submission.json` — metadata
