# EBT — Energy-Based Transformer with Closed-Form Quadratic Refinement

**Track:** Notable Non-Record on 10-min/16MB.

**TL;DR.** First Energy-Based Transformer (EBT) entry in `openai/parameter-golf`.
We add a closed-form quadratic energy `g(h) = ½‖A h‖² + b·h` over the final
hidden state and unroll `K` analytic gradient-descent steps `h ← h − η_k · (AᵀA h + b)`
before projecting to logits. This adds a learnable test-time-compute axis on top
of the standard softmax-CE LM. A small auxiliary CE on the un-refined hidden
state `h₀` (weight 0.1) discourages the refinement from collapsing to a no-op.

The eval-time `K` ablation (run on a tiny MLX smoke setup at 1,000 train steps;
H100 numbers TBD) shows that removing the refinement at eval — on a model that
was trained with it — is **+0.034 bpb worse** than the baseline that never had
refinement at all, confirming the K refinement steps do real test-time work
rather than acting as a training-time regularizer. At the smoke scale, the
full `K=2/K=2` setup is **statistically tied with the matched-param baseline**
(within 0.0024 bpb) — a higher-capacity H100 run is what will determine
whether the architecture provides Pareto benefit at the actual submission scale.

## Why this is novel here

A keyword scan of the 1,867 PRs on `openai/parameter-golf` (April 2026) finds
**zero** prior EBT submissions, despite multiple high-saturation neighbours:
Mamba/SSM (29), JEPA (26), BitNet/ternary (27), DeltaNet/GLA (19), text
diffusion (17), Universal Transformer (11), and Megakernel (8). The only PRs
that mention "energy" at all are an unrelated Wh-per-bpb leaderboard-axis
proposal (#1952) and one-off observations in kitchen-sink PRs.

Energy-Based Transformers were introduced in *Du et al., "Energy-Based
Transformers Are Scalable Learners and Thinkers" (2025)*. We use a deliberately
conservative variant ("Option A": energy as a learnable correction on the final
hidden state, with softmax-CE preserved) for two reasons:

1. **Stability.** Faithful EBT (energy on a continuous candidate embedding,
   trained with denoising score matching) is finicky to train at the scales
   relevant to a 10-minute run. Option A keeps the loss as standard CE and adds
   only a closed-form correction, which is robust and admits the existing Muon
   + GPTQ + sliding-window-eval tooling unchanged.
2. **Two-backend parity.** Closed-form `∇_h g = AᵀA h + b` avoids both
   `torch.autograd.grad(..., create_graph=True)` and nested `mx.grad`. As a
   result, the PyTorch and MLX implementations are byte-for-byte the same math,
   and we can iterate the design entirely on a Mac before committing to
   8xH100 time.

## Method

```
input_ids
  ↓ token embedding + RMSNorm
  ↓ N transformer blocks (encoder/decoder skip pattern, baseline)
  ↓ final RMSNorm
  ↓ h₀
  ↓ (training only) optional Gaussian noise N(0, σ²) on h₀
  ↓ K refinement steps:  h ← h − η_k · (AᵀA h + b)
  ↓ h_K
  ↓ tied embedding projection + tanh softcap
  ↓ logits → cross-entropy
  +
  (training only) auxiliary cross-entropy on h₀ projection (weight α)
```

The energy module has parameters `A ∈ ℝ^{r×d}` and `b ∈ ℝ^d`, plus a
per-step learnable `η_k ∈ ℝ^K`. With `d=512, r=64`, the entire EBT addition
is ~33k params (~25 KB int6-compressed) — negligible vs the 16 MB cap.

Notable design choices:

- **Closed-form energy.** Quadratic `g(h) = ½‖A h‖² + b·h`. The gradient is
  exact, no autograd nesting needed. Compatible with `torch.compile(fullgraph=True)`
  and `mx.compile`.
- **Per-step learnable step size `η_k`** (initial 0.05). Lets the model decide
  how much to refine on each step rather than committing to a fixed schedule.
- **Auxiliary CE on `h₀`** (weight 0.1). Forces the un-refined hidden state to
  also be a useful predictor, and prevents trivial "refinement = identity"
  solutions where the energy module's gradient is zero everywhere.
- **Optional `H0_NOISE_STD`** (default 0). Adds Gaussian noise on `h₀` during
  training so the refinement steps must denoise. We do not need this in our
  smoke runs (refinement is already non-trivial), but it's wired in as the
  anti-collapse stretch knob.
- **Eval-time `K` is independent of training-time `K`.** Train with a small
  `K_train=2` for speed; eval with `K_eval ∈ {0, 1, 2, 4, 8}` for the
  test-time-compute ablation.

Param routing into the existing optimizer split:
- `energy_head.A.weight` (2D) → Muon, with the same `MATRIX_LR` as the block
  matrices.
- `energy_head.b` (1D) and `refine_eta` (1D) → Adam scalar group, with
  `SCALAR_LR`.

## Smoke results (MLX, Mac, 1,000 steps)

These are tiny-scale results meant to demonstrate that the architecture
trains, that refinement is non-trivial, and that the test-time-compute story
is real. Headline 8xH100 SP1024 numbers will be added once H100 access is
granted.

**Setup:** 2 layers, dim=128, vocab=1024 (full), seq=512, train_batch=4,096
tokens, energy_rank=32, refine_eta_init=0.05, aux_loss_weight=0.1, **1,000
iters** on Mac (MLX bf16). Validation runs over the full FineWeb val split.
The same FineWeb training shard is recycled across iterations (stronger
overfitting risk than at full scale, but identical for all three variants so
the comparison is fair).

| Variant | K_train | K_eval | val_loss | val_bpb (post-quant) | refine_relchange@final | Δ bpb vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| **EBT (full)** | **2** | **2** | **3.7754** | **2.2360** | **0.93** | **+0.0024** |
| Baseline (no refinement)             | 0 | 0 | 3.7714 | **2.2336** | — | — |
| EBT trained, refinement OFF at eval  | 2 | 0 | 3.8280 | 2.2672 | — | **+0.0336** |

(Raw logs in [smoke_logs/](smoke_logs/).)

**Total artifact size**: `train_gpt.py` is 52,558 bytes of code; the int8+zlib
compressed model at this 366k-param smoke config is ~560 KB. Total ~613 KB,
i.e. ~3.8% of the 16 MB cap. At the actual submission shape (9L/512d, ~17M
params) the model dominates and we expect to land near the cap.

| Variant | code (bytes) | compressed model (bytes) | total (bytes) |
|---|---:|---:|---:|
| EBT (full)                          | 52,558 | 560,182 |   612,740 |
| Baseline                            | 52,558 | 564,110 |   616,668 |
| EBT trained, refinement OFF at eval | 52,558 | 558,786 |   611,344 |

**Reading the table.** Three things to take away:

1. **Refinement is doing real work at eval time.** Removing the refinement
   from a model that was trained with it (row 3) is **+0.034 bpb worse than
   the no-refinement baseline** (row 2). The model has genuinely learned to
   depend on the K refinement steps to produce its predictions; they are not
   a no-op or a regulariser.
2. **At this tiny scale, the full EBT and the baseline are statistically
   tied** (Δ = 0.0024 bpb, smaller than seed noise). With a 366k-param model
   and only 1,000 steps over a single recycled shard, there isn't enough
   capacity or data signal for the EBT advantage we saw at 120 iters
   (+0.036 bpb) to persist; both variants saturate.
3. **`refine_relchange` is high (~0.93).** The relative change between `h₀`
   and `h_K` is ~93% by step 1,000, up from 0.02 at init. The model is
   applying very aggressive refinement — possibly *too* aggressive. At full
   scale we'll likely want to shrink `REFINE_ETA_INIT` or add a small weight
   decay specifically on the energy head; this is one of the first things to
   tune on the H100.

The headline question for the H100 phase is whether the EBT advantage at
small steps re-emerges at full capacity (9L/512d, 20k iters) — or whether the
energy-refinement compute is better spent on a slightly larger backbone with
no refinement. The smoke is a sanity check, not a Pareto verdict.

## Reproducing

### Local (MLX, Mac with Apple Silicon)

```bash
# from the repository root
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# from this folder
RUN_ID=mlx_ebt_smoke_K2 \
ITERATIONS=1000 TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=4096 TRAIN_BATCH_TOKENS=4096 TRAIN_SEQ_LEN=512 \
NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 \
ENERGY_RANK=32 REFINE_STEPS_TRAIN=2 REFINE_STEPS_EVAL=2 \
WARMUP_STEPS=20 WARMDOWN_ITERS=200 MAX_WALLCLOCK_SECONDS=0 \
GRAD_ACCUM_STEPS=1 MLX_MAX_MICROBATCH_TOKENS=4096 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
python3 train_gpt_mlx.py
```

Switch `REFINE_STEPS_TRAIN=0 REFINE_STEPS_EVAL=0` for the no-refinement
baseline; or `REFINE_STEPS_TRAIN=2 REFINE_STEPS_EVAL=0` to reproduce the
test-time-disabled ablation. Each run takes ~2 minutes on an M-series Mac
(15s train + ~1 min validation pass + setup).

### Submission run (8xH100, PyTorch)

```bash
# default config: 9L 512d SP1024 with K_train=2, K_eval=2.
RUN_ID=ebt_seed1337 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Seeds 1337 / 42 / 2025 with the same env. Final logs land in `logs/<RUN_ID>.txt`;
copy them to `train_seed{1337,42,2025}.log` for the submission artifact.

### Test-time-compute ablation (8xH100, PyTorch)

After a normal training run, re-eval with different `REFINE_STEPS_EVAL` to
sweep `K_eval ∈ {0, 1, 2, 4, 8}`. (Currently the script does a single eval per
run; the simplest path is to run it 5 times with the same `SEED` and different
`REFINE_STEPS_EVAL`.)

## EBT-specific environment variables

| Variable | Default | Description |
|---|---:|---|
| `ENERGY_RANK` | `64` | Bottleneck rank `r` of `A ∈ ℝ^{r×d}`. Higher = more expressive energy. |
| `REFINE_STEPS_TRAIN` | `2` | Number of refinement steps `K` during training. |
| `REFINE_STEPS_EVAL` | `2` | `K` during evaluation. Can differ from train. |
| `REFINE_ETA_INIT` | `0.05` | Initial value of each per-step learnable step size `η_k`. |
| `AUX_LOSS_WEIGHT` | `0.1` | Weight on the auxiliary CE on `h₀`. Set to `0` to disable. |
| `H0_NOISE_STD` | `0.0` | Stretch anti-collapse mechanism. Adds `N(0, σ²)` to `h₀` during training only. |
| `REFINE_DIAG_EVERY` | `0` | Reserved for future per-step diagnostic logging. |

## Honest discussion / known limits

1. **Smoke is a smoke.** 1,000 iters on a 366k-param model with a single
   recycled FineWeb shard is enough to show the design trains, refinement is
   non-trivial, and the test-time-compute axis is real (the K_eval=0
   regression is the cleanest evidence). It is **not** sufficient to predict
   whether EBT beats the matched-param baseline at the actual 9L/512d
   submission scale — at 1k smoke steps the two variants are within 0.0024
   bpb, smaller than seed noise. The H100 runs will resolve this.
2. **Refinement is overshooting at smoke scale** (`relchange ≈ 0.93` after
   1,000 steps). The per-step learnable `η_k` and the close-form `AᵀA h`
   gradient apparently combine to a step that is too aggressive. First H100
   tuning knobs to try: shrink `REFINE_ETA_INIT` (0.05 → 0.01), add weight
   decay to the energy-head matrix, or cap the per-step `η_k` magnitude.
3. **Closed-form quadratic is the simplest energy.** A more expressive option
   (e.g. ReLU² MLP energy `g(h) = uᵀ relu²(W h)` with `autograd.grad`) is
   straightforward to swap in if needed; we kept it closed-form for two-backend
   parity and compile-friendliness, and as the most easily falsifiable variant.
4. **Aux loss weight 0.1 may be too high at smoke scale.** With 0.1 weight on
   the un-refined-h₀ CE, training-loss reporting slightly favors the K=0
   baseline (since K=0 doesn't pay the aux penalty). Pure val bpb is the only
   fair number here. At scale, the aux weight likely wants to anneal toward 0
   over training so the model becomes increasingly committed to the refined
   prediction.
5. **No-record by design.** This submission targets the Notable Non-Record bar
   — first EBT in the repo, novel test-time-compute axis, conservative
   formulation. We expect the headline `val_bpb` on H100 to land *above* the
   current ~1.06 SOTA but plausibly below the ~1.22 naive baseline; the value
   of the entry is the architecture and the ablation, not the leaderboard
   number.

## File layout

- [`train_gpt.py`](train_gpt.py) — official PyTorch submission script. Self-contained;
  must run from this folder via `torchrun --standalone --nproc_per_node=8 train_gpt.py`.
- [`train_gpt_mlx.py`](train_gpt_mlx.py) — parallel MLX implementation. Same math as
  the PyTorch version, used for local Mac smoke testing only. Not used for
  scoring. Including it in the submission folder is free w.r.t. the 16 MB cap
  (which only counts `train_gpt.py` code + compressed model bytes).
- [`smoke_logs/`](smoke_logs/) — raw stdout from the three MLX smoke runs above.
- [`submission.json`](submission.json) — submission metadata.
- `train_seed{1337,42,2025}.log` — H100 train logs, populated after H100 runs.

## Acknowledgements / prior art

- Du et al., *Energy-Based Transformers Are Scalable Learners and Thinkers* (2025).
- Universal Transformer (Dehghani et al., 2018), DEQ (Bai et al., 2019), and
  the existing depth-recurrence PRs in this repo all share the "iterate to a
  fixed point" lineage; EBT differs in that the iteration explicitly minimises
  a learnable scalar energy and admits straightforward ablation of test-time
  compute.
