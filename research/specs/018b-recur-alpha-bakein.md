# Spec 018b — Recur-Alpha blend op baked into block forward

**Slug:** `recur-alpha-bakein`
**Created:** 2026-04-21
**Links to:** spec 018 (lerp), `research/ideas/beating-1736-note.md`

## Hypothesis

The recur-alpha blend is algebraically equivalent to modifying the block's coefficient structure:

```
x_blended = x_before + α(x_new − x_before)
          = (1 + α(mix[0]−1))·x_before + α·mix[1]·x0 + α·attn_scale·attn + α·mlp_scale·mlp
          = effective_mix_0·x_before + effective_mix_1·x0 + effective_attn_scale·attn + effective_mlp_scale·mlp
```

All four "effective" coefficients are per-dim vectors; α-scaling is pure scalar math. If we rewrite the block's final residual to compute this 4-term linear combination directly (instead of: compute `x_new`, then apply external blend), we eliminate the blend as a distinct tensor op. The 4-term sum fuses into a single kernel, skipping `x_new` materialization.

**Projected memory traffic**: ~50% reduction vs current (from ~256 MB to ~128 MB per site at best-case fusion), vs ~25% reduction from lerp (018).

**Caveat**: torch.compile's inductor may already fuse `torch.lerp(x_before, x_new, α)` with the preceding block's mlp-residual-add, achieving the same memory footprint automatically. If 018 shows big gains, 018b may be redundant. If 018 shows only modest gains, 018b is the needed manual refactor.

## Baseline

Run alongside 018 on the same 2×H100 mini-model pod. Baseline = same 4dd2d63 config. Compare tok/s at steady state (step 100-150) across:
- Run B (016b): current 4-op blend
- Run C (018): torch.lerp
- **Run D (018b): bake-in refactor**

## Decision criterion

Let L = tok/s of lerp (Run C), K = tok/s of bake-in (Run D), B = tok/s of current blend (Run B).

| Scenario | Interpretation | Action |
|---|---|---|
| K ≥ 1.05 × L | Bake-in materially better than lerp | Ship bake-in. Lerp was not enough. |
| 1.01 × L ≤ K < 1.05 × L | Modest improvement over lerp | Evaluate engineering cost vs gain. Likely ship lerp. |
| K ≈ L (within 1%) | torch.compile auto-fused lerp optimally | Ship lerp, shelve bake-in as redundant |
| K < L | Bake-in broke something | Debug — maybe fused sum has worse numerics or fusion pattern |

## Code changes

**Branch:** `exp/recur-alpha-bakein` forking from `4dd2d63` (same base as 018 — clean A/B/C on the same start point).
**Commit:** `4c06275` on `fork/exp/recur-alpha-bakein`.

Includes:
- Bake-in refactor in `Block.forward` and `_block_with_lora` (both accept optional `alpha` parameter, fold into 4-term residual sum when supplied).
- `forward_logits` and `forward_ttt` both pass `alpha_param` to the blocks at recur-alpha sites.
- TTT forward-path fix included (closes the α=1-at-TTT bug from spec 015's original patch).

**Note:** spec 018b is a training-only throughput test (`ENABLE_LOOPING_AT=0`, `ITERATIONS=150`), so the TTT fix is *included* in the commit but *not exercised* by this diagnostic. It's shipped here so downstream full-pipeline specs inherit the fix cleanly.

**Scope**: ~50 LOC refactor. Core changes:

### 1. `Block.forward` accepts optional `alpha` parameter

```python
def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
            cu_seqlens=None, max_seqlen=0, alpha=None):
    mix = self.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor,
                         q_w, k_w, v_w, out_w,
                         cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    mid = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
    mlp_out = self.mlp(self.mlp_norm(mid) * self.ln_scale_factor, up_w, down_w)

    if alpha is None:
        # Baseline path — unchanged from original
        return mid + self.mlp_scale.to(dtype=mid.dtype)[None, None, :] * mlp_out

    # Bake-in path: 4-term linear combination
    a = alpha.to(dtype=x.dtype)
    eff_mix_0     = 1.0 + a * (mix[0] - 1.0)              # [dim]
    eff_mix_1     = a * mix[1]                             # [dim]
    eff_attn_s    = a * self.attn_scale.to(dtype=x.dtype)  # [dim]
    eff_mlp_s     = a * self.mlp_scale.to(dtype=x.dtype)   # [dim]
    # Hopefully torch.compile fuses this 4-term sum into a single kernel.
    return (
        eff_mix_0[None, None, :] * x
        + eff_mix_1[None, None, :] * x0
        + eff_attn_s[None, None, :] * attn_out
        + eff_mlp_s[None, None, :] * mlp_out
    )
```

### 2. `forward_logits` passes `alpha` to blocks at recur-alpha sites

Replace the external blend:
```python
# OLD (in both encoder and decoder loops):
x_before = x
x_new = self.blocks[i](x_before, x0, ...)
if alpha_info is not None and alpha_info[step_idx] is not None:
    pass_off, local_idx = alpha_info[step_idx]
    alpha = self.recur_alpha[pass_off, local_idx].to(x_new.dtype)
    x = torch.lerp(x_before, x_new, alpha)  # or the 4-op version in current code
else:
    x = x_new

# NEW:
alpha_param = None
if alpha_info is not None and alpha_info[step_idx] is not None:
    pass_off, local_idx = alpha_info[step_idx]
    alpha_param = self.recur_alpha[pass_off, local_idx]
x = self.blocks[i](x, x0, ..., alpha=alpha_param)
```

Same change at both encoder (line ~1197) and decoder (line ~1260) loops.

### 3. `forward_ttt` equivalent changes

TTT has its own forward path (`_block_with_lora`) that mirrors `Block.forward`. Same bake-in refactor needs to land there for the TTT path to benefit.

**File scope**: `records/track_10min_16mb/2026-04-19_.../train_gpt.py` — Block.forward (~20 LOC), GPT.forward_logits (~10 LOC), _block_with_lora (~20 LOC). Total ~50 LOC.

### 4. Critical: verify gradient correctness

Recur_alpha's gradient must match baseline. In the old code:
```
dL/dα = dL/dx_blended · (x_new − x_before)
```

In the bake-in:
```
dL/dα = dL/dx_blended · ((mix[0]−1)·x + mix[1]·x0 + attn_scale·attn + mlp_scale·mlp)
      = dL/dx_blended · (x_new − x_before)
```

Algebraically identical. But verify numerically with `torch.autograd.gradcheck` on a toy example before launching.

## Model config (same mini proxy as 018)

Same as 018.

## Hardware

Same **2×H100 NA US-NE-1** pod. Ideally run 018 Run C and 018b Run D on the same pod back-to-back for apples-to-apples.

## Execution protocol

Run D (bake-in) on the same pod as 018's Run C. Ideally:

```bash
git checkout 4c06275

mkdir -p /workspace/runs/018b-recur-alpha-bakein/run-d-bakein

NCCL_NET=Socket DATA_DIR=/workspace/parameter-golf/data \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache \
ARTIFACT_DIR=/workspace/runs/018b-recur-alpha-bakein/run-d-bakein \
CASEOPS_ENABLED=1 GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
ENABLE_LOOPING_AT=0 TRAIN_LOG_EVERY=25 SEED=42 ITERATIONS=150 \
NUM_LAYERS=6 MODEL_DIM=256 XSA_LAST_N=6 PARALLEL_START_LAYER=99 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  > /workspace/runs/018b-recur-alpha-bakein/run-d-bakein/train.log 2>&1
```

Different commit than 018's Run C → fresh compile (~5 min). Read tok/s at steps 50, 100, 150.

## Expected artifacts

```
runs/018b-recur-alpha-bakein/
  run-d-bakein/train.log
  notes.md  # ratios + decision
```

`notes.md` must compare Run D's tok/s against Run C (lerp) and Run B (current).

## Stop-early criteria

- NaN / inf in step time → halt
- Compile failure → halt (if block.forward refactor broke something, catch here)
- Gradient mismatch at toy-example check (run on pod before main run) → halt
- Step 150 reached → done

## Cost estimate

| item | cost |
|---|---|
| Pod boot + SSH (if fresh pod; reuse if chaining from 018 pod) | ~$0.10 |
| Run D (bake-in): ~5 min compile + 25s training | ~$0.40 |
| Rsync + pod stop (if last run) | ~$0.05 |
| **Total 018b diagnostic (assuming chained after 018)** | **~$0.45** |

Cheaper if chained onto 018's pod. Total 018 + 018b on one pod: ~$1-1.50.

## Open questions for interview

1. **Chain onto 018's pod or spin fresh?** Chained is cheaper and cleaner (same hardware). Defer to execution.
2. **Gradient check first?** Short `torch.autograd.gradcheck` on a toy model (2 layers, tiny dim) verifies dL/dα matches baseline before committing to the 150-step run. ~30 sec extra. Recommended.
3. **If bake-in breaks something** (rare but possible with the refactor), do we fall back to lerp-only path? Yes — the default `alpha=None` path is unchanged from baseline, so disabling RECUR_ALPHA_ENABLED=0 gives us the original behavior.

## Sequencing

- Run **after** 018 (lerp) — that's the comparison anchor.
- Ideally on the **same pod** to eliminate pod-variance confound.
- Decision feeds into the next full-pipeline spec (017-style but on the winning optimization).

## What 018b does NOT do

- Does not eliminate blend cost entirely (still ~128 MB/site; ~50% of current)
- Does not test val_bpb (mini-model, no bpb measurement relevant)
- Does not refactor for α=0 optimizations (we never hit α=0 in practice; optimize for α≠1 and ≠0)
- Does not touch non-recur-alpha code paths (default `alpha=None` preserves baseline behavior exactly)
