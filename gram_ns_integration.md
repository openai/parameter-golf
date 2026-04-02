# Gram Newton-Schulz Integration on Top of #1 Submission

## What is Gram Newton-Schulz?

Reference: https://github.com/Dao-AILab/gram-newton-schulz

Standard Newton-Schulz (what #1 uses) iterates on the full matrix `X` which is `(n, m)`:

```python
for each step:
    A = X @ X.T        # n×m times m×n = n×n
    B = b*A + c*A@A    # n×n
    X = a*X + B @ X    # n×n times n×m = n×m  ← expensive every step
```

Every iteration touches the full `n×m` matrix. For a 512×1536 MLP weight (transposed to 512×1536 since we ensure n≤m), that's a 512×1536 matmul every iteration.

Gram Newton-Schulz reformulates this to iterate on `R = X @ X.T` which is only `n×n`:

```python
R = X @ X.T          # compute once: 512×512
Q = I                 # accumulator: 512×512

for each step:
    Z = b*R + c*R@R   # 512×512 — small!
    Q = a*Q + Q@Z      # 512×512 — small!
    R = ...update...    # 512×512 — small!

X = Q @ X             # one final 512×1536 matmul at the end
```

All the iterative work happens on the small `512×512` Gram matrix. The expensive `512×1536` matmul with `X` only happens **once** at the end (plus at restart points for numerical stability) instead of every iteration. For 5 iterations, that's roughly 4 fewer large matmuls.

### Why it's even faster than the FLOP reduction suggests

All the inner-loop matrices (`R`, `Q`, `Z`) are **symmetric** (`R = XX^T` is always symmetric). The original Dao-AILab repo includes custom CuTeDSL CUDA kernels that exploit this symmetry — a symmetric GEMM only needs to compute half the output elements, giving another ~2x kernel-level speedup on H100/Blackwell. This integration does **not** use those custom kernels (to keep it dependency-free), but the algorithmic FLOP reduction still applies.

### Numerical stability: restarts

The Gram iteration accumulates floating-point errors in `R` and `Q` over iterations. To prevent this, a **restart** is performed at iteration 2 (of 5 total):

1. Apply accumulated `Q` to `X`: `X = Q @ X` (one expensive n×m matmul)
2. Recompute Gram fresh: `R = X @ X^T`
3. Reset `Q` to None (next iteration reinitializes from identity)

This keeps the iteration accurate while still saving most of the FLOPs.

## What was changed

**File:** `train_gpt_gram_ns.py` (copy of #1 submission: `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`)

**Changes made:**

1. Replaced `zeropower_via_newtonschulz5()` with a corrected Gram NS implementation matching the Dao-AILab reference
2. Added FlashAttention fallback chain (FA3 → FA2 → PyTorch SDPA) for non-H100 GPUs

**Zero changes to:**
- The Parallel Muon communication overlap (reduce-scatter/all-gather pipeline)
- The parameter bank structure
- Any other part of the training script
- The function signature — it's a drop-in replacement

## Implementation details

### Per-step coefficients (Polar Express)

The original #1 submission uses the same `(a, b, c) = (3.4445, -4.7750, 2.0315)` for all 5 iterations. The Dao-AILab reference uses **different coefficients per iteration** from the Polar Express paper (arxiv 2505.16932), with a 1.05x safety factor. The first iteration is more aggressive (a=7.89, large b and c), later iterations are gentler:

```python
_POLAR_EXPRESS_RAW = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),   # step 0: aggressive
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),  # step 1
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),  # step 2
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),    # step 3
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),  # step 4: gentle
]
# Each scaled by 1/safety^1, 1/safety^3, 1/safety^5 respectively
```

### Square vs rectangular dispatch

For **square matrices** (like `qo_bank` at 512×512), the Gram reformulation has no FLOP savings — the Gram matrix is the same size as the original. The implementation falls back to standard NS with `baddbmm` for these.

For **rectangular matrices** (like `mlp_up_bank` at 512×1536), the full Gram path runs — iterating on the 512×512 Gram while avoiding the 512×1536 matmuls until the end.

### Fused operations via `baddbmm`

All multiply-add patterns use `torch.baddbmm` instead of separate ops:
- `Z = b*R + c*R@R` → `torch.baddbmm(R, R, R, beta=b, alpha=c)`
- `Q = a*Q + Q@Z` → `torch.baddbmm(Q, Q, Z, beta=a)`
- `RZ = a*R + R@Z` → `torch.baddbmm(R, R, Z, beta=a)`

Each `baddbmm` fuses scalar multiply + matmul + add into a single CUDA kernel, saving memory bandwidth and kernel launch overhead.

### Q initialization

Q starts as `None`. On the first iteration (or after a restart), Q is initialized as:

```python
Q = Z + a * I   # = aI + bR + cR^2
```

This is the polynomial `p(R) = a + bR + cR²` applied to R, with the identity providing the constant term. On subsequent iterations, Q accumulates via `Q = a*Q + Q@Z`.

### R update skip optimization

The R update is skipped on the last iteration (R is never used again) and on iterations immediately before a restart (R will be recomputed from scratch). This saves 2 unnecessary `baddbmm` calls.

## FLOP savings per parameter bank

With `num_layers=11`, `model_dim=512`, `kv_dim=256`, `mlp_dim=1536`:

| Bank | Shape per slice | After transpose (n≤m) | n×n Gram size | Speedup potential |
|------|----------------|----------------------|---------------|-------------------|
| `qo_bank` | (22, 512, 512) | square — falls back to standard NS | 512×512 | None (n=m) |
| `kv_bank` | (22, 256, 512) | (22, 256, 512) | 256×256 | Moderate |
| `mlp_up_bank` | (11, 1536, 512) | transposed to (11, 512, 1536) | 512×512 | **Big** — inner loop avoids 512×1536 |
| `mlp_down_bank` | (11, 512, 1536) | (11, 512, 1536) | 512×512 | **Big** — same |

**Expected result:** On the MLP banks, the inner loop does 512×512 matmuls instead of 512×1536. With 5 steps, 1 restart at step 2: that's 2 cheap iterations (steps 0-1) + 1 restart (apply Q to X, recompute R) + 2 more cheap iterations (steps 3-4) + 1 final Q@X. Versus standard: 5 full 512×1536 matmuls. Roughly 40-50% fewer FLOPs on the MLP banks.

## The Gram Newton-Schulz algorithm (detailed)

```python
def _gram_newtonschulz(X, coefficients, restart_at):
    R = X @ X.mT                  # (B, n, n) Gram matrix
    I = eye(n).expand(B, n, n)
    Q = None

    for i, (a, b, c) in enumerate(coefficients):
        # Restart: fold Q into X, recompute R, reset Q
        if i in restart_at and i != 0:
            X = Q @ X              # expensive n×m matmul
            R = X @ X.mT           # recompute Gram
            Q = None

        Z = baddbmm(R, R, R, beta=b, alpha=c)   # Z = b*R + c*R²

        if Q is None:
            Q = Z + a * I          # initialize: Q = aI + bR + cR²
        else:
            Q = baddbmm(Q, Q, Z, beta=a)  # Q = a*Q + Q@Z

        # Skip R update on last step and before restarts
        if i < num_steps - 1 and (i + 1) not in restart_at:
            RZ = baddbmm(R, R, Z, beta=a)    # RZ = a*R + R@Z
            R  = baddbmm(RZ, Z, RZ, beta=a)  # R  = a*RZ + Z@RZ

    X = Q @ X   # final: apply accumulated orthogonal factor
    return X
```

### Why the R update formula is `a*RZ + Z@RZ` where `RZ = a*R + R@Z`

In standard NS, after one step: `X_new = a*X + Z @ X` where `Z = b*A + c*A²` and `A = X @ X^T`.

The Gram of the new X is:
```
R_new = X_new @ X_new^T
      = (a*X + Z@X)(a*X + Z@X)^T
      = (a*X + Z@X)(a*X^T + X^T@Z^T)
      = a² * X@X^T + a*Z@X@X^T + a*X@X^T@Z^T + Z@X@X^T@Z^T
```

Since `Z` and `R = X@X^T` are symmetric (`Z^T = Z`, `R^T = R`):
```
R_new = a²*R + a*Z@R + a*R@Z + Z@R@Z
      = a*(a*R + R@Z) + Z@(a*R + R@Z)
      = a*RZ + Z@RZ
```

where `RZ = a*R + R@Z`. This is how we update the Gram matrix without ever touching `X`.

## Corrections from initial implementation

The initial version had several issues compared to the Dao-AILab reference. These have all been fixed:

| Issue | Initial attempt | Fixed version |
|-------|----------------|---------------|
| **Coefficients** | Same `(3.4445, -4.7750, 2.0315)` every step | Polar Express per-step coefficients from Dao-AILab repo |
| **Q initialization** | `Q = I` then `a*Q + Q@Z` on first step | `Q = None` then `Q = Z + a*I` on first step (matches reference) |
| **R update waste** | Computed R on every step including last | Skips R update on last step and before restarts |
| **Square matrices** | Ran full Gram path (more ops, same size matmuls) | Falls back to standard NS (no Q/R overhead) |
| **Fused ops** | `a*R + R@Z` as separate ops | `torch.baddbmm(R, R, Z, beta=a)` — single fused kernel |

## FlashAttention fallback

The #1 submission imports `flash_attn_interface` (FlashAttention 3, H100-only). The modified script adds a fallback chain:

1. **FA3** (`flash_attn_interface`) — H100/Blackwell, fastest
2. **FA2** (`flash_attn`) — Ampere+ (RTX 3090/4090/A100), fast
3. **PyTorch SDPA** (`F.scaled_dot_product_attention`) — any GPU, transposes B,T,H,D → B,H,T,D

This lets the script run on a 4090 or any CUDA GPU without code changes.

## How to run

### Prerequisites

- CUDA GPU (H100 for full reproduction, 4090 for testing, any Ampere+ with FA2)
- PyTorch 2.6+
- `sentencepiece`, `lzma` (stdlib)
- Optional: `flash-attn` (FA2) or `flash_attn_interface` (FA3)
- FineWeb dataset: `python3 data/cached_challenge_fineweb.py --variant sp1024`

### Run on 8xH100 (full reproduction)

```bash
cd /path/to/parameter-golf

RUN_ID=gram_ns_test \
SEED=1337 \
NUM_LAYERS=11 \
MLP_MULT=3.0 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
ITERATIONS=9000 \
WARMDOWN_ITERS=3500 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
GRAD_CLIP_NORM=0.3 \
BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_LAYERS=9,10 \
EMA_ENABLED=1 \
LATE_QAT_THRESHOLD=0.15 \
EVAL_STRIDE=64 \
TTT_ENABLED=1 \
TTT_EPOCHS=3 \
TTT_FREEZE_BLOCKS=0 \
TTT_LR=0.002 \
TTT_CHUNK_TOKENS=32768 \
torchrun --standalone --nproc_per_node=8 train_gpt_gram_ns.py
```

### Run on 1x4090 (24GB, testing)

```bash
cd /path/to/parameter-golf

RUN_ID=gram_ns_4090 \
SEED=1337 \
NUM_LAYERS=11 \
MLP_MULT=3.0 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=131072 \
ITERATIONS=2000 \
WARMDOWN_ITERS=400 \
MAX_WALLCLOCK_SECONDS=600 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=500 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
GRAD_CLIP_NORM=0.3 \
BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_LAYERS=9,10 \
EVAL_STRIDE=64 \
TTT_ENABLED=0 \
VAL_LOSS_EVERY=500 \
VAL_BATCH_SIZE=65536 \
torchrun --standalone --nproc_per_node=1 train_gpt_gram_ns.py
```

If OOM on 4090, reduce `TRAIN_BATCH_TOKENS` to `65536`.

### Run on 1xGPU (smoke test, any GPU)

```bash
cd /path/to/parameter-golf

RUN_ID=gram_ns_smoke \
SEED=1337 \
NUM_LAYERS=11 \
MLP_MULT=3.0 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=65536 \
ITERATIONS=200 \
WARMDOWN_ITERS=50 \
MAX_WALLCLOCK_SECONDS=300 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
GRAD_CLIP_NORM=0.3 \
BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_LAYERS=9,10 \
EVAL_STRIDE=64 \
TTT_ENABLED=0 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt_gram_ns.py
```

### What to compare

Run the original #1 submission and the Gram NS version with the same seed and config. Compare:
1. **step_avg (ms)** — Gram NS should be lower (faster per step)
2. **total steps in 600s** — Gram NS should fit more steps
3. **final val_bpb** — should be equal or slightly better (same math, more steps)

The key metric is `step_avg`. If it drops by even 2-3ms, that's ~250 extra training steps in 10 minutes.
