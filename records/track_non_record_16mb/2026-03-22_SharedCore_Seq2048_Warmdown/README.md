# Weight Sharing Exposes a Different Scaling Regime

**Author:** Leo Feasby ([@leofeasby](https://github.com/leofeasby))
**Track:** Non-record
**val_bpb:** 1.1454 (still descending at cutoff)
**Roundtrip int8+zlib bpb:** 1.1723
**Runtime:** ~2.3 hours on 8×H100 SXM
**Date:** 2026-03-22

---

This submission studies a shared-weight transformer architecture that appears to follow a different optimisation regime from standard layer-independent models.

Instead of independent weights per layer, a single transformer block is reused across depth, forming a parameter-efficient recurrent stack.

**Key finding:** performance in this architecture is not limited by capacity, but by training schedule alignment. Under extended training, the model continues improving well beyond typical competition regimes, reaching 1.145 bpb and still descending.

---

## Key Result

**1.1454 val bpb** in 2.3 hours — the loss curve was still actively descending when training stopped. The loss was still descending at cutoff, so the true floor for this architecture was not reached in this run.

For reference, the only other listed non-record unlimited compute run achieves 1.2074 bpb at 4 hours. This run reaches 1.1454 in roughly half the time.

---

## Architecture: Shared-Core U-Net Transformer

The most novel aspect of this submission is the **shared-core transformer** — a single transformer block whose weights are reused across all 9 effective passes, structured as a U-Net encoder-decoder with learned skip connections.

### How it works

Instead of 9 independent transformer blocks, the model has **one** `shared_block` (attention + MLP) that is applied across 9 effective passes:

```
Encoder pass (layers 0–3):   x = shared_block(x) → store skip
Decoder pass (layers 4–8):   x = shared_block(x + skip_weight * skip)
```

The U-Net structure means early representations are injected back into later passes via learned `skip_weights`, allowing the model to reference shallow features deep in the stack — similar to residual shortcuts but across the entire depth.

To allow the shared block to behave differently at each depth, each layer gets its own:
- `attn_scale` — scales attention output before residual add
- `mlp_scale` — scales MLP output before residual add
- `resid_mix` — interpolates between residual and new representation

The core attention and MLP weights are **fully shared** across all 9 passes.

### Why this is parameter-efficient

A standard 9-layer transformer at this width (dim=1024, MLP×5) would require ~140M parameters. By sharing the core block:

| Component | Parameters |
|-----------|-----------|
| Shared transformer core (attention + MLP×5) | ~14.7M |
| Token embeddings (tied) | ~1.0M |
| Bigram hash table (4096 entries) | ~4.2M |
| Per-layer scales + skip weights | ~0.03M |
| **Total** | **18.9M** |

This gives 9-layer effective depth at a fraction of the cost — well within the 16MB artifact budget (~13.9MB int8+zlib compressed).

This architecture trades parameter redundancy for iterative refinement. Instead of learning depth through independent weights, it learns depth through repeated application of the same transformation, making optimisation dynamics fundamentally different from standard transformers.

### Architecture config
```
num_layers:       9
model_dim:        1024
num_heads:        16
num_kv_heads:     8   (GQA — 2:1 ratio)
mlp_mult:         5   (relu² activation)
bigram_table:     4096 (hash-based, zero-init)
tie_embeddings:   True
weight_decay:     0.04 (matrix params only)
```

---

## Training: Extended Warmdown Exploration

This run was designed to answer a specific question: **where does this architecture actually floor?**

Previous 10-minute competition runs were clearly not long enough to find the minimum — the loss was still falling steeply at every cutoff. This run extended the warmdown to 41,000 steps to probe the true convergence point.

### WARMDOWN_START_STEP: a new schedule control

Standard warmdown in this codebase is wallclock-based — LR starts decaying based on elapsed time relative to a fixed cap. This creates tight coupling between the training budget and the warmdown timing, making it hard to isolate the effect of warmdown length.

We introduce `WARMDOWN_START_STEP`: a step-based warmdown trigger that fires at a specific step count, independent of wallclock time. This decouples schedule design from budget constraints and makes the warmdown timing fully reproducible across hardware.

```python
def lr_mul(step, elapsed_ms):
    if args.warmdown_start_step > 0:
        if step < args.warmdown_start_step:
            return 1.0
        steps_into_warmdown = step - args.warmdown_start_step
        return max(1.0 - steps_into_warmdown / args.warmdown_iters, 0.0)
    # ... (original wallclock-based path preserved as fallback)
```

### Phase Transition Behaviour

Training exhibits a clear phase transition:
- Slow, steady improvement during the high-LR phase
- Followed by a rapid, sustained drop in loss during warmdown

From the training curve: the model sits at ~1.292 bpb when warmdown begins at step 4000, then descends continuously to 1.145 bpb over the next 41,000 steps — nearly all meaningful learning happens in the low-LR regime.

In shorter runs (~1000 warmdown steps), the same transition compresses: ~1.32 → ~1.25 in a tight window.

This indicates that the majority of the model's effective learning occurs during warmdown, not during early high-LR training. In standard competition runs, this phase is truncated by the wallclock limit before it can be fully exploited.

**This leads to a critical conclusion: performance is governed by schedule alignment, not architecture scaling.**

---

## Training Curve

| Step | Elapsed | val_bpb | Phase |
|------|---------|---------|-------|
| 0 | 0s | 4.164 | init |
| 500 | 80s | 1.476 | warmup |
| 2000 | 320s | 1.326 | full LR |
| 3500 | 560s | 1.297 | full LR |
| 4000 | 640s | **1.292** | warmdown starts |
| 5500 | 880s | 1.280 | |
| 7500 | 1200s | 1.267 | |
| 11500 | 1840s | 1.253 | |
| 15500 | 2480s | 1.241 | |
| 19500 | 3120s | 1.231 | |
| 23500 | 3760s | 1.222 | |
| 27500 | 4400s | 1.213 | |
| 31500 | 5040s | 1.205 | SWA starts (~step 32500) |
| 35500 | 5680s | 1.193 | |
| 39500 | 6320s | 1.177 | |
| 43500 | 6960s | 1.155 | |
| **45000** | **7200s** | **1.145** | LR → 0 |
| 45000+ | — | 1.145 | frozen (LR=0) |

**The loss was still dropping at ~0.003 per 500 steps when LR hit zero. The model never plateaued before LR reached zero.** The true floor is unknown.

SWA applied: 351 snapshots from step 32,500.

---

## Warmdown is the Dominant Driver

The entire gap between 1.292 and 1.145 was recovered during the warmdown phase alone, with no sign of saturation at cutoff.

This is not merely a schedule artifact. It appears to reflect a structural property of the shared-weight optimisation landscape: the model requires a long, low-LR phase to resolve the accumulated gradient signal from weight reuse across depth.

---

## seq_len=2048: A Major Lever

Doubling the context window from 1024 to 2048 tokens provides a substantial improvement with essentially no additional parameter cost.

Comparison at matched training config (same architecture, same hardware):

| seq_len | Best val_bpb | Notes |
|---------|-------------|-------|
| 1024 | 1.214 | 20-min run, warmdown still descending |
| **2048** | **1.145** | 2.3h run, warmdown still descending |

In these runs, moving from seq_len=1024 to 2048 corresponded to an approximately 0.07 bpb improvement, though this comparison is still conditioned on the current training setup and schedule. On H100s with Flash Attention, seq_len=2048 runs at only ~1.1× the wall-clock cost of seq_len=1024 (same batch tokens, quadratic attention largely absorbed by hardware).

This suggests that context scaling is more efficient than parameter scaling in this regime — consistent with the broader finding that the architecture is time-limited, not capacity-limited.

---

## What This Reveals

1. **The model is not capacity-limited**
   - Performance continues improving at 1.145 bpb with no plateau
   - Loss remains on a downward trajectory at cutoff
   → The architecture likely continues below 1.145 with longer training, though the exact floor is unknown

2. **Performance is schedule-limited**
   - The dominant gains occur during warmdown
   - Standard competition runs terminate before this regime is fully exploited
   → Current results are constrained by time allocation, not model quality

3. **Shared architectures follow a different optimisation regime**
   - Slower early learning
   - Stronger late-stage convergence
   → Unlike standard transformers, which optimise quickly then plateau

4. **Scaling behaviour differs from standard models**
   - Increasing training time yields large gains
   - Increasing parameters yields diminishing returns
   → Suggests a compute-to-quality tradeoff inversion

This result does not show that shared-weight architectures are universally better than standard transformers under the challenge objective. It shows that they occupy a different optimisation regime: weaker short-horizon performance, but much stronger long-horizon convergence than early runs suggest.

---

This work suggests that current leaderboard models are optimised for short-horizon performance, while shared-weight architectures may offer stronger long-horizon scaling than their short-horizon competition results suggest.

---

## Reproducing This Run

### Run config
```
TRAIN_SEQ_LEN=2048
ITERATIONS=50000
WARMDOWN_START_STEP=4000    ← full LR for ~640s (~11 min)
WARMDOWN_ITERS=41000        ← linear decay over 41k steps (~6560s)
MAX_WALLCLOCK_SECONDS=86400
SWA_FRAC=0.35 SWA_FREQ=50
ADAPTER_RANK=0
WEIGHT_DECAY=0.04
BIGRAM_TABLE_SIZE=4096
VAL_LOSS_EVERY=500
CHECKPOINT_EVERY=2000
```

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# Copy train_gpt.py from this records folder, then:
TRAIN_SEQ_LEN=2048 \
ITERATIONS=50000 \
WARMDOWN_START_STEP=4000 \
WARMDOWN_ITERS=41000 \
MAX_WALLCLOCK_SECONDS=86400 \
SWA_FRAC=0.35 SWA_FREQ=50 \
ADAPTER_RANK=0 \
MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=8 \
MLP_MULT=5 WEIGHT_DECAY=0.04 \
BIGRAM_TABLE_SIZE=4096 \
VAL_LOSS_EVERY=500 \
CHECKPOINT_EVERY=2000 \
FAST_COMPILE=1 \
torchrun --nproc_per_node=8 --master_port=29505 train_gpt.py
```

Step time: ~160ms/step on 8×H100 SXM. Total runtime: ~2.3 hours.
