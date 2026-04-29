# Lottery Ticket Hypothesis with few Floats

This record captures an unlimited-compute non-record submission that explores the **Lottery Ticket Hypothesis (LTH) / supermask** regime for parameter-golf: instead of training the weights of a transformer, we **freeze all weight matrices at a seed-derived random initialization and only learn per-element binary masks** over them, plus a small set of continuous scale parameters. The idea is that within a sufficiently large random network there already exists a subnetwork — a "winning ticket" — that performs well on the task, and training reduces to finding which elements to keep (1 bit per weight) and lightly calibrating a handful of continuous knobs.

**For inference, the frozen network is regenerated from the same seed and the learned binary mask is applied on top of it.**

**The submission artifact is therefore almost entirely incompressible bits plus a 4-byte seed, giving a very different size / quality tradeoff from a conventionally trained and quantized model.**

## Method

1. **Deterministic random init.** All weight matrices are generated on CPU from a single `uint32` seed using a fixed-order, fixed-dtype float32 generator:
   - Vocab embedding: Xavier uniform
   - MLP `fc` (pre-ReLU) layers: He (Kaiming) uniform with fan-in
   - All other 2D weights (attention q/k/v/proj, MLP proj): Xavier uniform
   - All 1D params (biases, norms): zeros
2. **Freeze weights, learn masks.** Every frozen `Linear` / `Embedding` gets a learnable `mask_scores` tensor of the same shape. At every forward pass, `mask = (sigmoid(scores / temp) >= 0.5).float()` using a straight-through estimator (sigmoid gradient on the backward pass). The effective weight is `frozen_weight * mask`.
3. **Temperature annealing.** `temp` anneals from `1.0 → 0.5` across training, sharpening the mask without collapsing gradients too early.
4. **A few continuous "scale" params stay trainable.** Per-block `attn_scale` and `mlp_scale` (shape `[D]`), per-head `head_scale` (shape `[H]`), and a `pre_logit_scale` (shape `[D]`). These are tiny — saved as fp16 directly in the artifact.
5. **Artifact.** Single `submission.ptz` file packing `[seed | bit-packed masks | fp16 scales]` compressed with zlib. At load time, the model is rebuilt in float32, weights are regenerated from the seed, converted to bf16, and the binary masks and scales are applied. (The float32→bf16 order matters: random numbers must be generated in float32 on CPU to be dtype-invariant, then cast.)

## Configuration

- **Track:** `non-record`, unlimited compute, under the `16,000,000` byte artifact cap
- **Layout:** `VOCAB_SIZE=1024 NUM_LAYERS=16 MODEL_DIM=1024 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- **Tied embeddings:** `TIE_EMBEDDINGS=1`
- **Mask optimizer:** Adam, `MASK_LR=0.1`, `MASK_INIT_SCORE=0.0` (starts at ~50% active)
- **Temperature schedule:** `MASK_TEMP_START=1.0 → MASK_TEMP_END=0.5`
- **Batching:** `TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=2048`
- **Iterations:** `ITERATIONS=50000`, `WARMDOWN_ITERS=200`, no wallclock cap
- **Hardware:** 1× NVIDIA A100-PCIE-40GB, run on a SLURM cluster

## Command

The run was launched from a SLURM cluster on a single A100-PCIE-40GB node:

```bash
RUN_ID=supermask_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
MASK_LR=0.1 MASK_INIT_SCORE=0.0 MASK_TEMP_END=0.5 \
NUM_LAYERS=16 MODEL_DIM=1024 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
TRAIN_SEQ_LEN=2048 \
ITERATIONS=50000 MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=200 \
TRAIN_BATCH_TOKENS=262144 \
torchrun --nproc_per_node=1 train_supermask.py
```

## Key metrics

From the training log:

- Training ran the full `50000/50000` steps (no wallclock cap).
- Final eval: `val_loss: 2.1810`, `val_bpb: 1.2917`
- Mask sparsity at stop: `mask_active: 0.461` (i.e. ~46% of weights are kept, ~54% zeroed)
- Exact printed roundtrip metric: `roundtrip_verify_exact val_bpb: 1.29171982`
- Roundtrip matches live eval to 4 decimal places, confirming artifact integrity.
- Train time: `206285390ms` (≈ 57.3 hours), `step_avg: 4125.71ms`
- Peak memory: `25549 MiB allocated`, `25682 MiB reserved`
- Serialized model (mask + scales + seed, zlib): `14,845,665 bytes`
- Code size: `46,184 bytes`
- **Total submission size: `14,891,849 bytes`** (under the 16 MB cap)

## Training volume

- Global batch: `262,144` tokens/step
- Total train tokens seen: `50,000 × 262,144 = 13,107,200,000`

## What the submission actually contains

The `submission.ptz` file is laid out as (pre-compression):

```
[ 4 bytes ]  network_seed            uint32 LE
[ 4 bytes ]  mask_section_length     uint32 LE
[ N bytes ]  bit-packed masks        1 bit per frozen weight element
[ remaining] scale params (fp16)     torch.save of small continuous tensors
```

Essentially: a seed (to regenerate the frozen random network), a big pile of bits (which of those random weights to keep), and a handful of floats (scale / gain knobs). No actual weight values are stored.

## Why this is interesting

- **Epistemic claim:** this is an empirical test of the LTH at the ~118M-frozen-weight scale with an SP-1024 tokenizer — a random init + learned binary mask is enough to reach `val_bpb ~1.29`, without ever updating a weight value.
- **Compression regime:** mask bits are essentially incompressible (they encode the learned signal), so artifact size scales linearly with parameters kept. At this architecture the mask is `~118M bits → ~14.1 MB` raw, compressing only slightly with zlib to ~14.16 MB, plus a few KB of fp16 scales. The floats-free tradeoff means quality is bounded more by the random basis than by quantization.
- **Reproducibility:** because everything traces back to a single `uint32` seed plus a deterministic float32 init routine, the full frozen network can be regenerated bit-exactly on any machine; the masks and scales are the only learned content that needs to be shipped.

## Included files

- `train_supermask.py` — code snapshot used for the run
- `train.log` — exact training log

## Files generated when this script is executed

- `submission.ptz` — the single submission artifact (seed + masks + scales)
- `submission.json` — leaderboard metadata