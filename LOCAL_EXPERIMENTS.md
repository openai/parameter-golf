## Local hardware notes

### 2026-03-18: RTX 3060 Ti 8 GB smoke run

Environment:
- GPU: `NVIDIA GeForce RTX 3060 Ti (8192 MiB)`
- Driver: `595.45.04`
- Python: repo-local `.venv` using `Python 3.13`
- PyTorch: `2.10.0+cu128`

Dataset setup:
- Command: `.venv/bin/python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`
- Result: local smoke dataset at `data/datasets/fineweb10B_sp1024/` plus tokenizer in `data/tokenizers/`

Confirmed working smoke config:

```bash
RUN_ID=local_3060ti_smoke \
ITERATIONS=20 \
WARMUP_STEPS=2 \
TRAIN_BATCH_TOKENS=65536 \
VAL_BATCH_SIZE=8192 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=5 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Observed results:
- Completed successfully on a single `3060 Ti`
- `step_avg: 447.17ms`
- `peak memory allocated: 1548 MiB`
- `peak memory reserved: 1682 MiB`
- `final_int8_zlib_roundtrip_exact val_bpb: 3.48092090`
- Log file: `logs/local_3060ti_smoke.txt`

Interpretation:
- Local iteration on an `8 GB` card is viable.
- Stock leaderboard settings do not fit on this GPU, but reduced-batch smoke runs do.
- Next step after this result: increase `TRAIN_BATCH_TOKENS` to `131072` and continue the VRAM sweep.

### 2026-03-18: RTX 3060 Ti 8 GB smoke run, larger batch

Confirmed working config:

```bash
RUN_ID=local_3060ti_131072 \
ITERATIONS=20 \
WARMUP_STEPS=2 \
TRAIN_BATCH_TOKENS=131072 \
VAL_BATCH_SIZE=8192 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=5 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Observed results:
- Completed successfully on a single `3060 Ti`
- `step_avg: 848.95ms`
- `peak memory allocated: 2797 MiB`
- `peak memory reserved: 2844 MiB`
- `final_int8_zlib_roundtrip_exact val_bpb: 3.46109434`
- Log file: `logs/local_3060ti_131072.txt`

Interpretation:
- `TRAIN_BATCH_TOKENS=131072` is still comfortably inside `8 GB`.
- The prior `65536` run was extremely conservative.
- Next step after this result: test `TRAIN_BATCH_TOKENS=262144`.

### 2026-03-18: RTX 3060 Ti 8 GB smoke run, failed larger batch

Attempted config:

```bash
RUN_ID=local_3060ti_262144 \
ITERATIONS=20 \
WARMUP_STEPS=2 \
TRAIN_BATCH_TOKENS=262144 \
VAL_BATCH_SIZE=8192 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=5 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Observed result:
- Failed during warmup with `torch.OutOfMemoryError`
- Error point: attempted allocation of `32.00 MiB`
- Context from error: about `4.64 GiB` allocated by PyTorch and about `5.27 GiB` total process memory in use at failure time
- Failure happened before any steady-state training metrics were logged
- Log file: `logs/local_3060ti_262144.txt`

Interpretation:
- `TRAIN_BATCH_TOKENS=262144` is above the practical limit for this exact setup on a `3060 Ti 8 GB`.
- The current known-good / known-bad bracket is:
  - good: `131072`
  - bad: `262144`
- Next step after this result: test midpoint `196608`.

### 2026-03-18: RTX 3060 Ti 8 GB smoke run, midpoint failure

Attempted config:

```bash
RUN_ID=local_3060ti_196608 \
ITERATIONS=20 \
WARMUP_STEPS=2 \
TRAIN_BATCH_TOKENS=196608 \
VAL_BATCH_SIZE=8192 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=5 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Observed result:
- Failed during warmup/backward compilation before steady-state training began
- Failure mode was not a plain CUDA allocator OOM
- Compiler error: `No valid triton configs`
- Root message: Triton kernel resource limit exceeded for the compiled backward path
- Log file: `logs/local_3060ti_196608.txt`

Interpretation:
- The practical limit on this GPU is constrained by both memory and compiled kernel shape.
- Current bracket for the stock compiled script:
  - good: `131072`
  - bad: `196608`
- Next step after this result: test `163840`.

### 2026-03-18: RTX 3060 Ti 8 GB smoke run, lower midpoint failure

Attempted config:

```bash
RUN_ID=local_3060ti_163840 \
ITERATIONS=20 \
WARMUP_STEPS=2 \
TRAIN_BATCH_TOKENS=163840 \
VAL_BATCH_SIZE=8192 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=5 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Observed result:
- Failed during warmup/backward compilation before steady-state training began
- Failure mode matched the `196608` run
- Compiler error: `No valid triton configs`
- Root message: Triton kernel resource limit exceeded for the compiled backward path
- Log file: `logs/local_3060ti_163840.txt`

Interpretation:
- The stock compiled script still fails above `131072` at `163840`.
- Current bracket for the stock compiled script:
  - good: `131072`
  - bad: `163840`
- Next step after this result: test the smallest step above the current good point, `139264`.

### 2026-03-18: RTX 3060 Ti 8 GB smoke run, smallest step above current good point

Confirmed working config:

```bash
RUN_ID=local_3060ti_139264 \
ITERATIONS=20 \
WARMUP_STEPS=2 \
TRAIN_BATCH_TOKENS=139264 \
VAL_BATCH_SIZE=8192 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=5 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Observed results:
- Completed successfully on a single `3060 Ti`
- `step_avg: 975.51ms`
- `peak memory allocated: 3063 MiB`
- `peak memory reserved: 3180 MiB`
- `final_int8_zlib_roundtrip_exact val_bpb: 3.45945004`
- Log file: `logs/local_3060ti_139264.txt`

Interpretation:
- `TRAIN_BATCH_TOKENS=139264` works on the stock compiled script.
- Current best-known bracket for this exact local setup:
  - good: `139264`
  - bad: `163840`
- The failure above this point is not simple VRAM exhaustion; it is a Triton backward-kernel resource limit in the compiled path.
