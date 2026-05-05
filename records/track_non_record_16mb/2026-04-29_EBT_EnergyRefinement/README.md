# EBT — Energy-Based Transformer with Closed-Form Quadratic Refinement

**Track:** Notable Non-Record on 10-min / 16MB. Negative result with diagnostics.

## TL;DR

Reference paper: [Gladstone et al., Energy-Based Transformers Are Scalable Learners and Thinkers, arXiv:2507.02092, 2025.](https://arxiv.org/abs/2507.02092)

First EBT entry in `openai/parameter-golf` (based on PR scan). Adds a learnable
closed-form quadratic energy `g(h) = ½‖A h‖² + b·h` over the final hidden state and unrolls
`K` analytic gradient-descent steps `h ← h − η_k · (AᵀA h + b)` before projecting to logits.
Trained with the standard softmax-CE plus an auxiliary CE on `h₀` (anti-collapse), evaluated
post-quantization (int8 + zlib).

**What we kept from the paper:** the EBT idea of an inner-loop, gradient-of-energy
refinement that gives the model "test-time compute" knobs (`K_train`, `K_eval`, learned
per-step `η_k`) over a stock decoder.

**What we changed for the budget:**
1. Energy lives on the hidden state `h`, not on a continuous next-token candidate `y`
2. Standard cross-entropy training instead of denoising score matching
3. Closed-form quadratic energy** (`A`, `b` only) instead of a learned MLP energy net

**At the 8×H100 SXM5 / 16 MB / 600 s budget, EBT does not pay back its compute cost.** EBT
v4 (K=8, r=32) trails the matched 9L/448D baseline by **0.0044 BPB on wallclock** (3-seed,
paired t≈9.5, p≈0.011) and by **0.0022 BPB at iso-step** — so the gap is an architecture
issue, not just a wallclock-cost artifact. A compute-trade attempt (8L/384D + K=8 vs
9L/448D no-EBT) lost an additional 0.040 BPB. The K_eval ablation **is** monotonic in
every config we tested (K=0 → K=8 improves val BPB by 0.0003–0.0009), so the refinement
loop is doing real test-time compute — it just isn't enough to outweigh the ~11% per-step
slowdown or to substitute for backbone capacity at this scale.

## Method

```
input_ids → blocks → final_norm → h₀
       (training only) optional N(0, σ²) noise on h₀
  K analytic refinement steps:  h_{k+1} = h_k − η_k · (AᵀA h_k + b)
       loss = CE(logits(h_K)) + α · CE(logits(h₀))   (α=0 at eval)
```

Energy module: `A ∈ ℝ^{r×d}`, `b ∈ ℝ^d`, plus per-step learnable `η_k ∈ ℝ^K`
(~14.8k params at d=448, r=32; <0.1 % of total). Closed-form gradient avoids
`torch.autograd.grad(create_graph=True)` and nested `mx.grad`, so PyTorch and MLX backends
are byte-for-byte the same math and `torch.compile(fullgraph=True)` works.

Routing into the existing optimizer split: `A.weight` → Muon (matrix lr); `b` and `η_k`
→ Adam scalar group.

## Headline numbers (8×H100 SXM5, sp4096, 9L/448D, ~14.5M params)

| Run | wallclock | steps (mean) | step time | val_bpb (3-seed mean ± std) |
|---|---:|---:|---:|---:|
| **baseline_v3** (no EBT) | 600 s | 9 353 | 64.1 ms | **1.21669 ± 0.00018** |
| **ebt_v4** (K=8, r=32, η=0.05, α=0.3, σ=0.05) | 600 s | 8 434 | 71.2 ms | **1.22112 ± 0.00069** |
| Δ (EBT − baseline) | | −919 (−9.8 %) | +7.0 ms (+11 %) | **+0.00443 BPB** (worse) |

Per-seed quantized val_bpb:

| seed | baseline_v3 | ebt_v4 | diff |
|---:|---:|---:|---:|
| 1337 | 1.216834 | 1.221071 | +0.004237 |
| 42   | 1.216491 | 1.221837 | +0.005346 |
| 2026 | 1.216739 | 1.220464 | +0.003725 |
| **mean** | **1.216688** | **1.221124** | **+0.004436** |

Paired t-statistic on the per-seed deltas: t ≈ 9.5, df = 2, two-sided p ≈ 0.011.

## Diagnostic experiments (single seed = 1337)

### 1. Iso-step (8400 iters, no wallclock cap)

Same number of optimizer steps, same warmdown schedule, EBT enabled vs disabled. Splits
"architecture per-step quality" from "wallclock cost of refinement."

| Run | steps | wallclock | val_bpb (quantized) |
|---|---:|---:|---:|
| baseline_v3 iso-step | 8 400 | 529.6 s | 1.219053 |
| ebt_v4 iso-step      | 8 400 | 589.7 s | 1.221234 |
| Δ                    |       |         | **+0.00218** |

EBT loses 0.0022 BPB **even at iso-step**. The architecture itself is worse per step at
this scale, not just expensive — the wallclock cap then doubles the gap.

### 2. Compute-trade (smaller backbone + refinement)

Hypothesis: shrink the backbone enough that EBT-K=8 is iso-time vs baseline, then EBT's
refinement compute might pay off.

| Run | model_bytes | steps | step time | val_bpb |
|---|---:|---:|---:|---:|
| baseline_v3 (9L/448D, no EBT) | 13.5 MB | 9 357 | 64.1 ms | 1.216834 |
| ebt_v4 (8L/384D + K=8)        | 9.2 MB  | 11 375 | 52.7 ms | **1.256825** |

8L/384D + K=8 is **0.040 BPB worse** than the 9L/448D no-EBT baseline. The smaller
backbone burns more capacity than the refinement loop recovers. Compute-trade hypothesis
fails at this size — would need a much larger refinement effect (or a denser-than-quadratic
energy) to flip.

### 3. K_eval ablation (3-seed mean for ebt_v4 wallclock; single-seed for diagnostics)

Re-eval the trained quantized model at different K_eval ∈ {0, 1, 2, 4, 8, 16}. K_train = 8.

| K_eval | ebt_v4 wallclock (3-seed mean) | ebt_v4 iso-step (s=1337) | ebt_v4 8L/384D (s=1337) |
|---:|---:|---:|---:|
| 0 | 1.222000 | 1.222049 | 1.258247 |
| 1 | 1.221941 | 1.221943 | 1.257791 |
| 2 | 1.221901 | 1.221870 | 1.257669 |
| 4 | 1.221826 | 1.221761 | 1.257422 |
| 8 | **1.221641** | **1.221744** | **1.257368** |
| 16 | 1.223391 | 1.222336 | 1.258373 |
| Δ K_0 → K_8 | −0.000359 | −0.000305 | −0.000879 |

Monotonic K_0 → K_train across **every config**, with K=16 always overshooting (energy
descent diverges past the trained step count). This is the cleanest evidence in the
submission that the refinement loop is doing genuine test-time compute, not just acting
as a regularizer. The magnitude is small (3–9 ten-thousandths of a BPB) and consistently
smaller than the per-step training-quality gap, which is why the architecture loses
overall.

## Why EBT v4 loses at this scale

1. **Per-step quality is worse**, not better, when the K=8 inner loop is added on top of
   a 9L/448D baseline. The auxiliary CE on `h₀` and the sequential refinement constrain
   the optimizer enough that 8 400 EBT steps land 0.002 BPB above 8 400 baseline steps.
2. **Per-step time is +11 %.** Eight extra `AᵀA h + b` ops on a 14.5M model on 8×H100
   SXM5 cost ~7 ms of the 64 ms baseline step.
3. **Refinement is mild.** `refine_relchange` settles at 0.02–0.07 (full-size) and 0.19
   (8L/384D); the energy is barely moving `h`. The K_eval ablation says this small
   movement is doing useful work — but it isn't worth 11 % wallclock or two layers of
   backbone.

## DDP `find_unused_parameters` trap (worth flagging)

Initial baseline runs constructed the energy head + `refine_eta` parameters even when
EBT was disabled, and used `find_unused_parameters=True` so DDP wouldn't error on the
unused params. This silently added **~15 % per-step overhead** (82 ms vs 64 ms / step on
8×H100 SXM5), making the baseline look slower than it actually is. With that overhead
in place, EBT v4 *appeared* to win by 0.0014 BPB (3-seed mean) — a clean false positive.

Fix in this submission: skip constructing the EBT params entirely when
`refine_steps_train == refine_steps_eval == 0` and `aux_loss_weight == 0`, so DDP can
take its standard fast path. After the fix, the baseline goes from 82 → 64 ms / step
(+28 % more steps in 600 s) and the comparison flips.

Worth keeping in mind for any submission that mixes optional auxiliary heads with DDP:
unused parameters are not free.

## Reproducing

### Local (MLX, Mac)

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
# from this folder
RUN_ID=mlx_ebt_smoke_K2 ITERATIONS=1000 TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=4096 TRAIN_BATCH_TOKENS=4096 TRAIN_SEQ_LEN=512 \
NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 \
ENERGY_RANK=32 REFINE_STEPS_TRAIN=2 REFINE_STEPS_EVAL=2 \
WARMUP_STEPS=20 WARMDOWN_ITERS=200 MAX_WALLCLOCK_SECONDS=0 GRAD_ACCUM_STEPS=1 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
python3 train_gpt_mlx.py
```

### 8×H100 SXM5 (PyTorch, full submission)

```bash
# baseline_v3 (no EBT)
RUN_ID=baseline_v3 SEED=1337 \
NUM_LAYERS=9 MODEL_DIM=448 VOCAB_SIZE=4096 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048 VAL_BATCH_SIZE=131072 \
ENERGY_RANK=32 REFINE_STEPS_TRAIN=0 REFINE_STEPS_EVAL=0 AUX_LOSS_WEIGHT=0.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# ebt_v4
RUN_ID=ebt_v4 SEED=1337 \
NUM_LAYERS=9 MODEL_DIM=448 VOCAB_SIZE=4096 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048 VAL_BATCH_SIZE=131072 \
ENERGY_RANK=32 REFINE_STEPS_TRAIN=8 REFINE_STEPS_EVAL=8 \
REFINE_ETA_INIT=0.05 AUX_LOSS_WEIGHT=0.3 H0_NOISE_STD=0.05 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Multi-seed: repeat each with `SEED ∈ {1337, 42, 2026}`. Iso-step: add
`ITERATIONS=8400 WARMDOWN_ITERS=1200 MAX_WALLCLOCK_SECONDS=0`.

The script auto-runs the K_eval ablation post-quantization for any EBT run and emits
all results to `train_log.json`.

## EBT-specific environment variables

| Variable | Default | Description |
|---|---:|---|
| `ENERGY_RANK` | `64` | Bottleneck rank `r` of `A ∈ ℝ^{r×d}`. |
| `REFINE_STEPS_TRAIN` | `2` | Number of refinement steps `K` during training. |
| `REFINE_STEPS_EVAL` | `2` | `K` during evaluation; can differ from train. |
| `REFINE_ETA_INIT` | `0.05` | Initial value of each per-step learnable `η_k`. |
| `AUX_LOSS_WEIGHT` | `0.1` | Weight on the auxiliary CE on `h₀`. Set 0 to disable. |
| `H0_NOISE_STD` | `0.0` | Optional Gaussian noise on `h₀` during training. |
| `K_EVAL_SWEEP` | `0,1,2,4,8,16` | K values to sweep in the post-quant ablation. |

## Future improvements

**Minor Tweaks:**
- Lower per-step cost: smaller K, or share the refinement state across the batch.
- Stronger refinement: rank-up the energy, anneal `α` to 0 over training (paper does
  this; we don't), or replace the closed-form quadratic with a denser energy
  (e.g. `g(h) = uᵀ relu²(W h)` with autograd).
- Reuse the freed compute correctly: the 8L/384D run shows that naively shrinking the
  backbone is too costly; a more careful FLOPs-matched re-shaping might recover.

**Move closer to the paper:**

1. **Put the energy on a continuous next-token candidate, not on the hidden state.**
   Train it the way the paper does, with denoising score matching: corrupt the true
   next-token embedding with Gaussian noise and regress the gradient of energy back to
   that noise direction. Decode by running `K` Langevin steps from random noise and then
   snapping to the nearest vocab embedding. This is the version where the K refinement
   steps actually *sample* from an energy-based model rather than iterating a learned
   feed-forward block. The reason we didn't ship it: the noise schedule, the double-
   backward through the energy gradient, and the EMA all need a real stability sweep,
   and none of that fits cleanly into 600 s × few seeds without first doing a longer
   exploratory phase.
2. **Train the same energy contrastively instead.** Push energy down on the true next-
   token embedding and up on negatives — cheaper and more stable than (1), and still
   gives a real EBM as long as the negatives aren't the full vocabulary (in which case
   the loss collapses back to standard softmax CE and you've reinvented the baseline).
   Sampled hard negatives (e.g. top-32 from the current model) sit in the well-studied
   noise-contrastive-estimation regime; full MCMC-sampled negatives are the most
   faithful but the slowest.

If we revisit this, (1) with a short noise-schedule sweep is the most defensible next
step — what we shipped is mathematically close to a Universal Transformer with a
learned step size, and the iso-step result is consistent with that critique.

## File layout

- [`train_gpt.py`](train_gpt.py) — PyTorch submission script. Run from this folder via
  `torchrun --standalone --nproc_per_node=8 train_gpt.py`.
- [`train_gpt_mlx.py`](train_gpt_mlx.py) — MLX implementation, byte-identical math, used
  for local Mac smoke testing. Not counted toward the 16 MB cap.
- [`submission.json`](submission.json) — submission metadata.
- [`results.csv`](results.csv) — every H100 run, with hyperparameters, artifact sizes,
  and final BPB.
- `train_seed{1337,42,2026}.log` — full train logs for the 3-seed ebt_v4 wallclock runs
  (the headline numbers).
- `diag_iso_step_{baseline,ebt}.log`, `diag_compute_trade.log` — train logs for the
  three diagnostic runs cited in the tables above.
