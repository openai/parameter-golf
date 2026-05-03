## 11L XSA4 + Partial RoPE + LN Scale + VE128 + ASQU + EMA + GPTQ-lite

Fork-local candidate built from the proven March 22 record because the March 31 causal-expert line missed the 10-minute target. This folder is the current rerun candidate for `8xH100`, `600s`, and `<=16 MB`.

**Base reference from the source record:** `1.1233 val_bpb` (3-seed mean), `15.55 MB`, `~7100` steps in `600s`. Current fork metrics are pending a fresh rerun.

This folder should be read as a differentiated hypothesis on top of a validated starting point, not as a new measured submission. Its value is that it preserves a strong March 22 control line while introducing one small architectural change with controlled local evidence over both `relu^2` and `leaky_relu^2`.

### Differentiated Hypothesis

This variant changes the MLP activation from a fixed scalar nonlinearity to **ASQU**:

`f(x) = x^2 if x > 0 else beta_i * x^2`

where `beta_i` is learned independently for each hidden channel.

Why this is a better candidate than the LeakyReLU2 fork:

- it is still a tiny change in runtime terms, but it is materially different from the already-covered LeakyReLU path
- it adds only `16,896` learned scalars across the whole 11-layer stack
- it already has controlled evidence that it beats both `relu^2` and `leaky_relu^2` in the same setup
- it does not require changing the training regime, artifact format, or legality profile

Local evidence already present in this repo:

- PR `#1035` reports a fixed-10k-step comparison where ASQU consistently beats both `relu^2` and `leaky_relu^2`
- in that controlled setting, mean post-quant BPB improves from `1.2331` (`relu^2`) to `1.2311` (`leaky_relu^2`) to `1.2300` (`ASQU`)
- the March 22 base still gives the hard external 8xH100 anchor for runtime and artifact discipline

That still does not prove this exact folder will beat the current SOTA. It does establish a more distinct and more evidence-backed path than the LeakyReLU rerun: transplant the stronger activation result onto the strongest clean non-TTT control we have and test whether the gain survives in the March 22 stack.

### Key Innovations Over PR #374

Two novel post-training optimizations plus training hyperparameter tuning on top of PR #374's architecture:

| Change | PR #374 | This | Impact |
|--------|---------|------|--------|
| **GPTQ-lite** | Fixed clip (row max) | 5 clip percentiles per row, pick min MSE | -0.0006 BPB (zero training cost) |
| **EMA** | None (Tight SWA only) | EMA decay=0.997 every step | -0.0006 BPB (smoother averaging) |
| **Warmdown** | 3000 | 3500 | -0.0002 BPB |
| **Late QAT threshold** | 0.1 | 0.15 | -0.0001 BPB (earlier fake quant, smaller quant gap) |
| **Total** | **1.1246** | **1.1233** | **-0.0013 BPB** |

### GPTQ-lite: Per-Layer Optimal Clip Percentile Search

Instead of using the row maximum for int6 quantization scale, we try 5 clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) per weight matrix row and pick the one minimizing reconstruction MSE. This is applied during post-training quantization with zero training cost.

### EMA Weight Averaging

Exponential moving average (decay=0.997) maintained every training step, applied before quantization. Stacks with Tight SWA — EMA provides continuous smoothing while SWA captures discrete checkpoints during warmdown.

### Base Reference Results (source record)

| Seed | Steps | val_loss | Sliding BPB (s64) | Artifact |
|------|-------|----------|-------------------|----------|
| **1337** | 7101 | 1.8958 | **1.1228** | 15.56 MB |
| 42 | ~7100 | 1.8972 | 1.1236 | 15.54 MB |
| 2024 | ~7100 | 1.8971 | 1.1236 | 15.59 MB |

**Mean: 1.1233 | Std: 0.0005** | Submitted: seed 1337 (best)

### Architecture (from PR #374)

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), `ASQU` activation in this variant (`relu^2` in the March 22 control)
- U-Net skip connections (5 encoder, 6 decoder)
- Efficient Partial XSA on last 4 layers (GQA-aware, zero-alloc)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10) with per-layer learned scales
- SmearGate + BigramHash (2048 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training

- FlashAttention 3 (Hopper-optimized)
- Muon optimizer (matrices): lr=0.025, momentum=0.99 (warmup 0.92->0.99 over 1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3500 iterations (wallclock-based)
- **EMA**: decay=0.997, every step
- **Tight SWA**: every 50 steps when scale<0.2
- **Late QAT**: STE int6 fake-quantization when LR scale<0.15
- OrthoInit + muP-scaled output projections

### Quantization

- **GPTQ-lite**: Per-row optimal clip percentile search (5 candidates) for int6
- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

### Validation Thesis

This folder is a low-entropy architectural bet. The near-term goal is to test a genuinely different activation family on the March 22 control line without blowing up runtime, legality, or artifact size.

Why this base matters:

- **It has a hard external performance anchor.** The source March 22 record already demonstrates `1.1233 val_bpb`, `15.55 MB`, and about `7100` steps in `600s` on `8xH100`.
- **It lives inside a strong family.** Closely related 11-layer FA3 records in the same lineage improve further, including a March 23 variant at `1.1194` with legal TTT.
- **The fork is intentionally low-entropy.** Almost all architectural behavior is unchanged from the March 22 source line; the main delta is the MLP activation family, plus operational cleanup for reruns and logging.
- **It keeps the control intact.** Because the change surface is so small, any measured movement is easier to interpret than it would be in a broad multi-change branch.

What is actually new here:

- `ASQU` as the default MLP activation for this variant
- dedicated low-LR optimization for the learned ASQU beta parameters
- stable seed-level logging for reruns
- clearer hardware/runbook instructions
- explicit positioning of this folder as a low-cost differentiated candidate rather than a broad rewrite

What is not new here:

- `ASQU` itself is not globally new to the repo; prior art already exists in non-record PR `#1035` and PR `#679`
- no measured architecture improvement over the March 22 source record yet
- no new submission metric from this fork yet
- no proof yet that the ASQU gain transfers one-for-one into this stack

The investable story is therefore disciplined execution:

- **Stage 1:** validate that the March 22 control still behaves correctly in this fork
- **Stage 2:** measure whether `ASQU` preserves runtime while improving score
- **Stage 3:** stop quickly if runtime, legality, artifact size, or quality miss predeclared bounds

Suggested paid-run decision gates:

- `GO`: the reproduced run stays within the 10-minute budget, artifact remains `<16 MB`, and the final score lands within about `0.005-0.010 BPB` of the source record
- `NO-GO`: step time is materially off target on `8xH100`, artifact size breaks the cap, legality becomes questionable, or reproduced quality misses badly enough that additional runs have poor expected value

If this line earns further capital, the next differentiated ideas should be treated as follow-on hypotheses on top of this activation variant, not mixed into the same test.

### Run Command

Before launching `torchrun`, verify how many GPUs the container can actually see:

```bash
nvidia-smi -L
python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
echo "$CUDA_VISIBLE_DEVICES"
```

Set `--nproc_per_node` to the visible GPU count. If you request `8` but the box only exposes `1` or `4`, PyTorch will fail with `CUDA error: invalid device ordinal`.

For the target `8xH100` run:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-02_11L_XSA4_PartialRoPE_LNScale_VE128_ASQU_EMA_GPTQlite

OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=base_11l_gptqlite_seed1337 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_ACTIVATION=asqu \
ASQU_BETA_INIT=0.25 \
ASQU_LR=0.001 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For an A/B against the original March 22 activation inside the same code path:

```bash
MLP_ACTIVATION=relu2
```

For a direct reference comparison to the earlier LeakyReLU variant:

```bash
MLP_ACTIVATION=leaky_relu2 MLP_LEAKY_SLOPE=0.5
```

If the container only sees a single GPU, do not treat that box as a submission reproduction environment. A longer 1-GPU run is still only a sanity check for startup, loss descent, export, and logging.

For a 2000-iteration 1-GPU debug run, use:

```bash
OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=debug_1gpu_2000_seed1337 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_ACTIVATION=asqu \
ASQU_BETA_INIT=0.25 \
ASQU_LR=0.001 \
ITERATIONS=2000 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Use the 1-GPU debug run only to answer:

- does the script launch cleanly
- does `world_size:1` appear as expected
- does training loss trend downward
- does a longer run stay numerically stable
- does the activation-specific log line confirm `mlp_activation:asqu`
- do `logs/<RUN_ID>.txt`, `train_seed<SEED>.log`, and `train.log` get written

Do not use the resulting step time or BPB as a decision-quality estimate for the intended `8xH100, 600s` submission regime.

### Cheap 1-GPU A/B Protocol

Use the same cheap regime for the March 22 control and the two activation variants. Change only the activation-related env vars.

Control (`relu^2`):

```bash
OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=ab_relu2_seed1337 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_ACTIVATION=relu2 \
ITERATIONS=2000 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Reference Leaky variant (`leaky_relu(0.5)^2`):

```bash
OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=ab_leakyrelu2_seed1337 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_ACTIVATION=leaky_relu2 \
MLP_LEAKY_SLOPE=0.5 \
ITERATIONS=2000 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

ASQU variant:

```bash
OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=ab_asqu_seed1337 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_ACTIVATION=asqu \
ASQU_BETA_INIT=0.25 \
ASQU_LR=0.001 \
ITERATIONS=2000 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Compare only these four numbers from the two logs:

- `step_avg` at step `2000`
- `step:2000 ... val_bpb`
- `DIAGNOSTIC post_ema ... val_bpb`
- `final_int6_roundtrip_exact ... val_bpb`

Use the A/B only as a local ranking signal between activations. It is still not a submission-quality estimate.

Provisional decision rule:

- `KEEP` the ASQU variant if it is no more than about `3%` slower and beats both the March 22 control and the Leaky reference by at least `0.003-0.005 BPB` on `post_ema` or `final_int6_roundtrip_exact` without clearly losing on the other.
- `KILL` the ASQU variant if it is slower and does not clearly improve over the March 22 control, or if it loses to the simpler Leaky reference as well.
- `INCONCLUSIVE` if the score movement is within about `0.003 BPB`; in that case, prefer the March 22 control for paid 8-GPU runs.

Defaults in `train_gpt.py` already encode the March 22 stack plus this variant's activation choice: `NUM_LAYERS=11`, `XSA_LAST_N=4`, `ROPE_DIMS=16`, `LN_SCALE=1`, `VE_ENABLED=1`, `VE_LAYERS=9,10`, `WARMDOWN_ITERS=3500`, `LATE_QAT_THRESHOLD=0.15`, `EVAL_STRIDE=64`, and `MLP_ACTIVATION=asqu`.

Each run writes:

- `logs/<RUN_ID>.txt`
- `train_seed<SEED>.log`
- `train.log` when `SEED=1337`

### Reproducibility

All 3 seeds produce valid artifacts under 16MB with tight variance (std=0.0005 BPB). The GPTQ-lite clip search is deterministic.
