# Spec 019b seed_42 — execution notes

**Run dir:** `runs/019b-recur-alpha-manual-constant-full/seed_42/`
**Commit:** `9517a3b` on `exp/recur-alpha-manual-constant-full` (forked from `3c3a134` = 019)
**Pod:** `6pyy9q7aatvgpb` — 8×H100 SXM, AP-JP-1, JP volume `jlxvxeiol4`
**Date:** 2026-04-21
**Status:** completed

## Result — closest recur-alpha run to #1736 so far

| metric | #1736 | 008 | 017 | 019 | **019b** |
|---|---|---|---|---|---|
| final step | 4854 | 4828 | 4784 | 4697 | **4716** |
| pre-quant post-EMA val_bpb | 1.06907 | 1.06922 | 1.07083 | 1.07063 | **1.06951** |
| post-GPTQ val_bpb | 1.07847 | 1.08010 | — | 1.07989 | **1.07877** |
| **post-TTT val_bpb (GATE)** | **1.06610** | **1.06728** | **1.06733** | **1.06744** | **1.06628** |
| submission size | 15,978,834 | 15,946,577 | — | 15,980,998 | **15,981,844** |

**019b post-TTT 1.06628 misses #1736 by only 0.00018** — within seed std.

## Path to launch (non-trivial)

1. **Attempt 1 (commit `e93d77d`, initial manual-add form):** CUDA OOM during `_run_cu_bucket_warmup` on all 8 ranks on JP pod `cuh1g7lxkdhi5z`.
   - Root cause: `α * x_new + (1 - α) * x_before` is 4 FX nodes (mul, rsub, mul, add). AoT partitioner's min-cut saved 2 intermediate tensors per blend site × 6 sites × ~512 MiB = 3-6 GiB extra activation memory, pushing peak compile memory over cap.
   - **Resolution:** rewrite as algebraic lerp form `x_before + α * (x_new - x_before)` — 3 ops, 1 intermediate `x_new - x_before`. Gradient formula requires only `grad_y` and scalar `α`, so partitioner saves nothing intermediate. Memory matches `torch.lerp`.
2. **Attempt 2 (commit `9517a3b`, algebraic form, pod `6pyy9q7aatvgpb`):** clean completion.

## Code change vs 019 (4 sites in `train_gpt.py`)

```python
# 019 (lerp):
x = torch.lerp(x_before, x_new, alpha)

# 019b (algebraic):
x = x_before + alpha * (x_new - x_before)
```

Identical for forward_logits enc/dec and forward_ttt enc/dec (with `x` vs `x_new` per site).

## Throughput — JP pod lottery, familiar

Same drift pattern as 019's pod:
- Steady **~8.08M tok/s for minutes 1-3** (matching 008/017's best).
- **Drifts to 6.4M by minute 8** — typical JP pod cooling / contention.

Step-matched train_loss tracks 019 essentially perfectly — the algebraic form is a true no-op vs lerp at the loss level. Confirmed by **step 4000 val_bpb identical to 019 (1.1071)**.

## What made 019b beat 019

- Same JP node drift, but 019b completed 19 more training steps than 019 (4716 vs 4697). Within one log-tick interval; amounts to slightly luckier loss timing at the wallclock boundary.
- Those 19 steps produced a pre-quant post-EMA improvement of **0.00113** (1.07063 → 1.06951).
- GPTQ cost was identical (+0.00926 vs +0.00927).
- TTT recovery was **larger**: −0.01249 for 019b vs −0.01245 for 019 — essentially identical.
- Net: pre-quant headroom carried all the way through to post-TTT.

Sign: **019b's improvement over 019 is dominated by 19 extra training steps and TTT, not by any code-change mechanism.** The lerp → algebraic rewrite contributed zero measurable throughput/quality signal (within noise).

## Execution issues resolved during this run

1. **Tokenizer path bug** (first attempt on pod `cuh1g7lxkdhi5z`): `DATA_DIR=/workspace/parameter-golf/data` was wrong for CaseOps on this volume. Correct path: `/workspace/data/` (root of jlxvxeiol4). Saved as `train.tokenizer_path_bug.log` for reference.
2. **CUDA OOM in warmup compile** (same pod, after tokenizer fix): fixed via algebraic rewrite on `9517a3b`.
3. **`git fetch fork` missing** — pod's repo clone has origin only. Needed to add `fork` remote manually during preflight on each pod.

## Cost

| item | cost |
|---|---|
| `6vc98xb3qwpzri` 2×H100 smoke (killed by user) | ~$1 |
| `cuh1g7lxkdhi5z` 8×H100 JP OOM attempt (~10 min) | ~$4 |
| `imfw8twiq2ipsw` 8×H100 NA create+delete (user preferred JP) | ~$0.50 |
| `6pyy9q7aatvgpb` 8×H100 JP successful run (~36 min) | ~$14.50 |
| **Total spec 019b execution spend** | **~$20** |

## Deliverables

- `train.log` (35 KB) — full pipeline output from successful run
- `final_model.int6.ptz` (15.95 MB) — submission artifact, under 16 MB cap
- `final_model.pt` on JP volume (135 MB — not git-synced; `/workspace/runs/019b-recur-alpha-manual-constant-full/seed_42/final_model.pt`)
- `final.json` — all metrics, comparison tables, and cost accounting
- `train.tokenizer_path_bug.log` (22 KB) — first-attempt log with tokenizer error, for reference

## Recommendation for research

Bucket outcome is **(1.06550, 1.06710] — close, within seed std**. Per spec 019b decision criterion, next action is **3-seed confirmation** (seeds 43 + 44, ~$20-24) to resolve whether 019b's 0.00018 miss vs #1736 is real or noise.

On reflection: 019b's apparent gain over 019 may be pod-lottery on step count rather than a real code-change benefit. A 3-seed study would both (a) establish 019b's mean/std and (b) indirectly tell us whether 019 on an equally-fast pod would have landed similarly.
