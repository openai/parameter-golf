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
3. Reset `Q` to identity

This keeps the iteration accurate while still saving most of the FLOPs.

## What was changed

**File:** `train_gpt_gram_ns.py` (copy of #1 submission: `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`)

**Only one function replaced** — lines 102-178:

1. New `zeropower_via_gram_newtonschulz()` replaces the old `zeropower_via_newtonschulz5()` body
2. Old `zeropower_via_newtonschulz5()` becomes a thin wrapper that calls the Gram variant
3. The call site in `Muon.step()` (line 299) still calls `zeropower_via_newtonschulz5(update, steps=backend_steps)` — unchanged, it just routes through the alias

**Zero changes to:**
- The Parallel Muon communication overlap (reduce-scatter/all-gather pipeline)
- The parameter bank structure
- Any other part of the training script
- The function signature — it's a drop-in replacement

## FLOP savings per parameter bank

With `num_layers=11`, `model_dim=512`, `kv_dim=256`, `mlp_dim=1536`:

| Bank | Shape per slice | After transpose (n≤m) | n×n Gram size | Speedup potential |
|------|----------------|----------------------|---------------|-------------------|
| `qo_bank` | (22, 512, 512) | square — no gain | 512×512 | None (n=m) |
| `kv_bank` | (22, 256, 512) | (22, 256, 512) | 256×256 | Moderate |
| `mlp_up_bank` | (11, 1536, 512) | transposed to (11, 512, 1536) | 512×512 | **Big** — inner loop avoids 512×1536 |
| `mlp_down_bank` | (11, 512, 1536) | (11, 512, 1536) | 512×512 | **Big** — same |

The biggest win is on MLP banks — the Gram iteration does 512×512 matmuls instead of 512×1536 for most iterations. For `qo_bank` (square matrices), there's no win since `n = m = 512` — the Gram matrix is the same size as the original. But it doesn't hurt either.

**Expected result:** On the MLP banks (the largest parameters), the inner loop does 512×512 matmuls instead of 512×1536. With 5 steps, 1 restart at step 2: that's 2 cheap iterations (steps 0-1) + 1 restart (apply Q to X, recompute R) + 2 more cheap iterations (steps 3-4) + 1 final Q@X. Versus standard: 5 full 512×1536 matmuls. Roughly 40-50% fewer FLOPs on the MLP banks, which should translate to measurably faster Muon steps and more training steps within the 10-minute cap.

## The Gram Newton-Schulz algorithm (detailed)

```python
def zeropower_via_gram_newtonschulz(G, steps=5, eps=1e-7, restart_at=(2,)):
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Ensure X is (B, n, m) with n <= m
    X = G.bfloat16()
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    # Initial Gram matrix R = X @ X^T, shape (B, n, n)
    R = X @ X.mT

    # Q accumulates the orthogonal factor. Starts as identity.
    Q = I  # (B, n, n)

    for i in range(steps):
        # Restart: apply Q to X, recompute R, reset Q
        if i in restart_at and i != 0:
            X = Q @ X              # expensive n×m matmul
            R = X @ X.mT           # recompute Gram
            Q = I                   # reset

        # All n×n matmuls from here:
        Z  = b*R + c*(R @ R)       # NS polynomial on Gram
        Q  = a*Q + Q @ Z           # accumulate orthogonal factor
        RZ = a*R + R @ Z           # intermediate for R update
        R  = a*RZ + Z @ RZ         # update Gram

    X = Q @ X                      # final: apply Q to get result
    return X
```

### Why the R update formula is `a*RZ + Z@RZ` where `RZ = a*R + R@Z`

In standard NS, after one step: `X_new = a*X + (b*A + c*A^2) @ X = a*X + Z @ X` where `Z = b*A + c*A^2`.

The Gram of the new X is:
```
R_new = X_new @ X_new^T
      = (a*X + Z@X)(a*X + Z@X)^T
      = (a*X + Z@X)(a*X^T + X^T@Z^T)
      = a^2 * X@X^T + a*Z@X@X^T + a*X@X^T@Z^T + Z@X@X^T@Z^T
```

Since `Z` and `R = X@X^T` are symmetric (`Z^T = Z`, `R^T = R`):
```
R_new = a^2*R + a*Z@R + a*R@Z + Z@R@Z
      = a*(a*R + R@Z) + Z@(a*R + R@Z)
      = a*RZ + Z@RZ
```

where `RZ = a*R + R@Z`. This is how we update the Gram matrix without ever touching `X`.

## How to run

### Prerequisites

- 8xH100 SXM (or other CUDA GPU for testing)
- PyTorch 2.6+
- `sentencepiece`, `flash-attn` (for FlashAttention 3 on H100), `lzma` (stdlib)
- FineWeb dataset downloaded: `python3 data/cached_challenge_fineweb.py --variant sp1024`

### Run on 8xH100 (full reproduction)

```bash
cd /path/to/parameter-golf

# Full run matching #1 submission + Gram NS
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

### Run on 1xGPU (smoke test, won't reproduce score)

```bash
cd /path/to/parameter-golf

# Smoke test — reduced batch, no TTT, short run
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
