---
name: triton-sequential-stateful-blocks
description: Write Triton kernels with sequential stateful processing inside a single thread block, with mutable register state visible across iterations.
---

# Sequential Stateful Processing in a Single Triton Block

> **Targets:** Triton >= 2.1, SM70+/CDNA2+

## Overview

Some workloads require one thread block to process a sequence of items with mutable register state (e.g., LRU cache routing, sequential assignment). This pattern uses grid `(B,)` — one block per batch element — and updates registers in a sequential loop so each iteration sees the exact mutated state from previous iterations.

**When to use:** When output of iteration `t` depends on state mutations from iteration `t-1` and parallel processing would give wrong results (e.g., two candidates claiming the same victim slot).

## Architecture: grid=(B,), sequential candidate loop

```python
@triton.jit
def _sequential_kernel(
    # ... input/output pointers, strides ...
    H_KV: tl.constexpr,    # number of KV heads
    T: tl.constexpr,       # number of candidates (typically 8-16)
    ME: tl.constexpr,      # number of slots (typically 64)
    DH: tl.constexpr,      # head dimension
    AE: tl.constexpr,      # active capacity (<= ME)
):
    off_b = tl.program_id(0)
    offs_me = tl.arange(0, ME)
    offs_dh = tl.arange(0, DH)

    # Active slot mask: only slots [0, AE) participate
    active_mask = offs_me < AE
```

## Phase 1: Load shared state into SRAM

Load all mutable state into registers BEFORE the candidate loop. Never write intermediate state to HBM.

```python
# Verified pattern from production LRU bank routing kernel
# used: (ME,) bool — loaded as int8, converted to int1, masked by active slots
sram_used = tl.load(used_ptrs).to(tl.int1) & active_mask

# last: (ME,) int64 — LRU timestamps
sram_last = tl.load(last_ptrs)

# Track whether ANY slot is used (scalar, kept as int32 for type stability)
any_used = tl.max(sram_used.to(tl.int32), axis=0)
```

## Phase 2: Sequential candidate processing

Each iteration loads one candidate, computes scores, classifies, and mutates register state immediately.

```python
for t in range(T):
    # Default outputs: not-overwrite, not-touch, idx=0
    idx_t: tl.int64 = tl.zeros([], dtype=tl.int64)
    overwrite_t: tl.int1 = tl.zeros([], dtype=tl.int1)
    touch_t: tl.int1 = tl.zeros([], dtype=tl.int1)

    gate = tl.load(gate_ptr + off_b * stride_gb + t * stride_gt)
    keep = gate >= TAU_GATE

    if keep:
        # ------ Multi-head similarity scoring ------
        avg_scores = tl.zeros([ME], dtype=tl.float32)
        for h in range(H_KV):
            # Load candidate vector for this head: (DH,)
            v_tok = tl.load(v_ptrs + h * stride_vh + t * stride_vt + offs_dh * stride_vd).to(tl.float32)
            # Load cached bank vectors: (ME, DH)
            mem_tile = tl.load(mem_ptrs + h * stride_mh + offs_me[:, None] * stride_mm + offs_dh[None, :] * stride_md).to(tl.float32)
            # Dot product: (ME, DH) * (DH,) → (ME,)
            scores_h = tl.sum(mem_tile * v_tok[None, :], axis=1)
            avg_scores += scores_h
        avg_scores = avg_scores / H_KV

        # Mask unused and inactive slots
        avg_scores = tl.where(sram_used & active_mask, avg_scores, -1e9)
        best_score = tl.max(avg_scores, axis=0)
        best_idx = tl.argmax(avg_scores, axis=0).to(tl.int64)

        # ------ Classify: novel, hit, or skip ------
        is_novel = (any_used == 0) | (best_score < TAU_NOVEL)
        is_hit = (any_used != 0) & (best_score >= TAU_MATCH)

        if is_novel:
            # LRU victim: unused slots get -inf timestamp (picked first),
            # inactive slots get +inf (never picked)
            lru_key = tl.where(
                active_mask,
                tl.where(sram_used, sram_last, tl.full([ME], value=-2**62, dtype=tl.int64)),
                tl.full([ME], value=2**62, dtype=tl.int64),
            )
            victim = tl.argmin(lru_key, axis=0).to(tl.int64)
            idx_t = victim
            overwrite_t = tl.full([], value=1, dtype=tl.int1)
            touch_t = tl.full([], value=1, dtype=tl.int1)

            # IMMEDIATE state mutation — visible to next iteration
            pos_t = tl.load(pos_ptr + off_b * stride_pb + t * stride_pt)
            sram_used = sram_used | (offs_me == victim)
            sram_last = tl.where(offs_me == victim, pos_t, sram_last)
            any_used = 1

        elif is_hit:
            idx_t = best_idx
            overwrite_t = tl.full([], value=1, dtype=tl.int1)
            touch_t = tl.full([], value=1, dtype=tl.int1)
            pos_t = tl.load(pos_ptr + off_b * stride_pb + t * stride_pt)
            sram_last = tl.where(offs_me == best_idx, pos_t, sram_last)

        else:
            idx_t = best_idx  # skip — no state mutation

    # Store per-candidate outputs (separate pointers per output type)
    tl.store(idx_ptr + off_b * stride_ib + t * stride_it, idx_t)
    tl.store(overwrite_ptr + off_b * stride_ob + t * stride_ot, overwrite_t)
    tl.store(touch_ptr + off_b * stride_tb + t * stride_tt, touch_t)
```

## Phase 3: Write final state to HBM

```python
# Only write SRAM state back at the very end
tl.store(last_out_ptrs, sram_last)
```

## Launcher pattern

```python
def launch(v_sel_norm, mem_v_norm, used, last, gate_sel, pos_sel, **kwargs):
    B, Hkv, T, Dh = v_sel_norm.shape
    Me = mem_v_norm.shape[2]
    # Ensure contiguous, allocate outputs
    idx_tok = torch.zeros((B, T), device=device, dtype=torch.int64)
    overwrite_tok = torch.zeros((B, T), device=device, dtype=torch.bool)
    touch_tok = torch.zeros((B, T), device=device, dtype=torch.bool)
    last_out = last.clone()  # clone so original is preserved

    grid = (B,)
    kernel[grid](..., H_KV=Hkv, T=T, ME=Me, DH=Dh, AE=active_capacity)
```

## Key constraints

- **Sequential semantics:** The loop body MUST see updated register state — no parallelism across `t` iterations.
- **Type consistency:** Use `int32` for mutable boolean-like registers; Triton requires consistent dtypes across all `if/elif/else` branches.
- **Scalar constants:** `tl.zeros([], dtype=tl.int1)` for False, `tl.full([], value=1, dtype=tl.int1)` for True.
- **Index casting:** `tl.argmax`/`tl.argmin` return indices; always `.to(tl.int64)` before pointer arithmetic.
- **Register state updates via `tl.where`:** You cannot index-assign into Triton tensors (`ts_r[idx] = val`). Instead: `sram_last = tl.where(offs_me == victim, new_val, sram_last)`.
- **Active vs used masking:** Separate `active_mask` (capacity limit) from `sram_used` (occupancy). Inactive slots should never be picked as LRU victims.
- **Fallback:** Always provide a PyTorch reference implementation for CPU/non-Triton environments with identical sequential semantics.
