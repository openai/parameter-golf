# Experiment NNN — short title

**Date:** YYYY-MM-DD
**Hypothesis:** What we expect and why (1–2 sentences).
**Baseline:** What we're comparing against (e.g. SOTA seed=42 = 1.08079).
**Cost:** Estimated $ and wallclock.

## Config

Diff from baseline (env vars, code patches, etc.):

```bash
TTT_LR=0.003 TTT_EPOCHS=5 TTT_CHUNK_TOKENS=16384 ...
```

If a code patch: link to `Opus/<patch_file>` and quote the relevant hunk.

## Command

Exact command(s) run:

```bash
SEED=42 ... torchrun --standalone --nproc_per_node=N train_gpt.py
```

## Result

| Metric | Value |
|--------|-------|
| `val_bpb_sliding` | |
| `val_bpb_ttt` | |
| Wallclock train | |
| Wallclock eval | |
| Artifact bytes | |

## Decision

- ✅ Promising → next step
- ⚠️ Marginal → re-run with different seed?
- ❌ Killed — reason

Notes / surprises / things to follow up.
