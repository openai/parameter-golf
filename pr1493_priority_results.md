# PR1493 Priority Experiments — Results Log

Comparator from `pr1493_priority_experiments.md`:

```text
PR1493 reproduced, seed 42, QK_GAIN_INIT=5.25:
  quantized_ttt: 1.08103358

Best local TTT sweep (TTT_LR=0.007, TTT_EPOCHS=5):
  quantized_ttt: 1.08079274
```

To matter for leaderboard acceptance margin, we likely need another `~0.0017–0.0020 BPB`,
not noise-level movement. Anything that does not beat ~`1.08079` on this seed is not a
real win.

## Environment

- 8x NVIDIA H100 80GB HBM3
- torch 2.9.1+cu128, flash_attn_3 cu128_torch291
- Dataset: `kevclark/parameter-golf` SP8192, 128 train shards, 1 val shard
- Common env: `SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5`
- Wallclock cap: 600s training + GPTQ + post-train eval (~15-20 min total per run)

## Results table

Filled in as each experiment finishes. `pre` = pre-quantization post-EMA val_bpb,
`q` = quantized val_bpb, `q_sw` = quantized_sliding_window val_bpb,
`q_ttt` = quantized_ttt val_bpb (primary metric), `size` = Total submission size bytes,
`stop_step` = step reached at wallclock cap.

| # | Experiment | flag(s) | pre | q | q_sw | q_ttt | size | stop_step | status |
|---|------------|---------|-----|---|------|-------|------|-----------|--------|
| 0 | baseline_ttt | — | — | — | — | — | — | — | aborted (user requested skip after warmup) |
| 1 | docshuffle | `DOC_SHUFFLE_ENABLED=1` | 1.09005 | 1.10121 | 1.08448 | **1.08279** | 16,033,898 | 4526/20000 | done |
| 2 | wd | `WD_SCHEDULE_ENABLED=1` | 1.08650 | 1.09951 | 1.08269 | **1.08029** | 16,031,886 | 4567/20000 | done — small win |
| 3 | iha | `IHA_ENABLED=1` | 1.08820 | — | — | — | — | 4524/20000 | **failed during GPTQ** |
| 4 | mtp | `MTP_WEIGHT=0.10 MTP_STEPS=1` | 1.11283 | 1.12678 | 1.11018 | **1.09023** | 16,035,001 | 4438/20000 | done — clear regression |
| 5 | evalloop3 | `EVAL_NUM_LOOPS=3` | — | — | — | — | — | — | running |
| 6 | wd_paired | `WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1` | 1.08610 | 1.09891 | 1.08209 | **1.07974** | 16,029,924 | 4596/20000 | done — small real stack win, see `pr1493_wd_paired_session.md` |
| 7 | wd_strong_paired | `WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 WD_SCHED_LOW_FACTOR=0.50 WD_SCHED_HIGH_FACTOR=1.75` | 1.08573 | 1.09874 | 1.08194 | **1.07971** | 16,030,578 | 4602/20000 | done — pre-quant edge but no q_ttt stack vs wd_paired, see `pr1493_wd_strong_paired_session.md` |
| 8 | wd_paired_iha | `WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 IHA_ENABLED=1` | 1.08666 | — | — | — | — | 4528/20000 | killed by pre-quant gate (worse than wd_paired by 0.00056), see `pr1493_wd_paired_iha_session.md` |
| 9 | wd_paired requant: 16-shard no-AR | `GPTQ_ALL_REDUCE=0 GPTQ_DAMP=0.01 GPTQ_BLOCK_SIZE=128` (re-quant, no training) | 1.08610 | 1.10117 | 1.08437 | **1.08060** | 16,034,159 | n/a | done — see `pr1493_gptq_allreduce_session.md` |
| 10 | wd_paired requant: 16-shard AR | `GPTQ_ALL_REDUCE=1` (else same) | 1.08610 | 1.09855 | 1.08178 | **1.07977** | 16,032,979 | n/a | done — −0.00084 vs (9), confirms AR theory |
| 11 | wd_paired requant: 128-shard no-AR | `GPTQ_ALL_REDUCE=0` (else same) | 1.08610 | 1.09852 | 1.08175 | **1.07976** | 16,032,799 | n/a | done — reproduces HF 1.07976130 |
| 12 | wd_paired requant: 128-shard AR | `GPTQ_ALL_REDUCE=1` (else same) | 1.08610 | 1.09850 | 1.08174 | **1.07975** | 16,032,588 | n/a | done — AR saturated, Δ vs (11) within noise |
| 13 | damp sweep at (12) | `GPTQ_DAMP={0.005,0.02,0.03,0.05}` (TTT off, q_sw proxy) | — | 1.0985–1.0986 | 1.08174–1.08182 | n/a | 16,030–16,033 K | n/a | done — 0.01 default optimal |
| 14 | block_size sweep at (12) | `GPTQ_BLOCK_SIZE={64,256}` (TTT off) | — | 1.0985 | 1.08173–1.08174 | n/a | 16,032–16,033 K | n/a | done — all tied within 3e-6 BPB |

## Per-experiment notes

### baseline_ttt (aborted)

Started successfully but stopped by user after warmup completed (20/20 + loop_warmup 20/20).
Not run to completion. Machine parity vs. the runbook's expected shape
(`pre ≈ 1.0875–1.0880`, `q_sw ≈ 1.083`, `q_ttt ≈ 1.08079–1.08103`) is therefore unverified.

### docshuffle (done — clear regression)

`DOC_SHUFFLE_ENABLED=1` activates `DocumentShuffleLoader` instead of
`ShuffledSequenceLoader`. Loader log: `doc_shuffle:bos=1 files=16 docs=1929107`
per rank (16 shards × 8 ranks = 128 train shards total, ~15.4M docs total).

**Result: q_ttt = 1.08279 vs comparator 1.08079 → Δ = +0.00200 BPB (worse).**
Not noise — that's roughly the same magnitude as the gap we'd need to *win*, but in the
wrong direction. Reasons it likely hurt:

- Tokens/sec dropped from ~7.6M/s (steps 500-1500) to ~6.6M/s by step 3000 and ~6.0M/s
  by step 4500. Wallclock cap is fixed at 588s post-reserve, so the doc loader's per-batch
  overhead cost ~10% of total training steps. docshuffle stopped at step 4526/20000;
  baseline's expected stop is around 4900-5000.
- Document boundaries make many short windows that span a single doc, so the model sees
  fewer cross-doc contexts per step. At this training budget that's net-negative.

Submission size **16,033,898 bytes — over the 16M limit by 33,898 bytes**. Even if it
*had* been better, it wouldn't be submittable without code-size minification.

**Verdict: drop.**

### wd (done — small real win, still below the leaderboard bar)

`WD_SCHEDULE_ENABLED=1` engages a piecewise weight-decay schedule:
- Hold at 1× until `WD_SCHED_HOLD_FRAC` (default 0.40) of training
- Ramp linearly to `WD_SCHED_LOW_FACTOR` (default 0.65×) by `WD_SCHED_RAMP_FRAC` (0.85)
- Ramp up linearly to `WD_SCHED_HIGH_FACTOR` (default 1.5×) at end of training

**Result: q_ttt = 1.08029 vs comparator 1.08079 → Δ = −0.00050 BPB (better).**
Above the runbook's `0.0002 BPB` "treat as noise" threshold, so this is a real win,
but well short of the `~0.0017–0.0020 BPB` margin needed to clear the leaderboard
acceptance bar on its own. Worth re-running across seeds before declaring it stackable.

Stop step 4567/20000 (≈ same as docshuffle's 4526), tok/s held ~7.7M through step 2k
then drifted to ~6.7M by step 3k — comparable to baseline cost, no extra slowdown.
Submission size 16,031,886 B — also over the 16M limit by 31,886 B (code itself is
55,168 B, model+brotli is 15,976,718 B; minification still needed for submit).

**Verdict: keep, but re-run on multiple seeds and combine with another candidate
(MTP/evalloop3) to stack toward leaderboard threshold.**

### iha (failed — harness bug)

`IHA_ENABLED=1` adds `q_mix`/`k_mix` head-mixing parameters and replaces the standard
`self.c_q(x)` / `self.c_k(x)` calls with `F.linear(x, self._mixed_weight(...))` inside
`CausalSelfAttention.forward`. Training itself ran normally; pre-quantization
post-EMA val_bpb landed at **1.08820** (between docshuffle's 1.09005 and wd's 1.08650),
stop_step 4524/20000.

**Crashed during GPTQ quantization with:**

```
File "train_pr1493.py", line 309, in gptq_mixed_quantize
    gptq_quantize_weight(t, hessians[name], ...)
KeyError: 'blocks.0.attn.c_q.weight'
```

Root cause: `collect_hessians` registers forward hooks on `nn.Linear` submodules
(`self.c_q`, `self.c_k`, `self.c_v`). With IHA enabled, those Linear modules are
bypassed — the forward pass uses `F.linear(x, self._mixed_weight(self.c_q.weight,
self.q_mix, ...))` directly. So the hook never fires and `hessians[]` has no entry
for `c_q.weight`/`c_k.weight`/`c_v.weight`. When `gptq_mixed_quantize` then iterates
the state dict, it looks up the missing key and raises `KeyError`.

**Verdict: harness bug, not a feature evaluation.** Two fixes possible:
1. Run a no-op `self.c_q(zero)` pass during forward when IHA is on, just to keep the
   hook hot, OR
2. Change `collect_hessians` to also handle the IHA path by hooking on the parent
   attention module and recording the activations directly for c_q/c_k/c_v.

Either way, the experiment as committed doesn't run. Skipping for now; flag for the
runbook author. No usable q_ttt result.

### mtp (done — clear regression)

`MTP_WEIGHT=0.10 MTP_STEPS=1` adds a 10%-weighted auxiliary loss for predicting the
t+2 token alongside the standard next-token loss. The runbook intentionally disables
MTP during the TTT adaptation phase, so this measures whether the auxiliary objective
helps the *trained* model.

**Result: q_ttt = 1.09023 vs comparator 1.08079 → Δ = +0.00944 BPB (much worse).**
The hit shows up everywhere — pre-quant 1.11283 (vs baseline expected 1.0875-1.0880),
q_sw 1.11018 (vs baseline 1.083). At this short training budget (~4438 steps before
wallclock cap), the gradient capacity spent on t+2 prediction is straight-up stolen
from t+1 fitting; there's no offsetting representation benefit visible by step 4438.

Stop step 4438/20000 (slightly fewer than wd's 4567 — MTP head adds a small per-step
cost), tok/s shape similar to baseline. Submission size 16,035,001 B — also over the
16M limit by 35,001 B.

**Verdict: drop at this weight/budget.** Worth an MTP_WEIGHT=0.05 sweep eventually,
but not before stacking confirmed wins (currently only wd).

## Errors / learnings (live)

- **iha is broken at the harness level.** GPTQ's Hessian collection hooks on
  `nn.Linear` forward, but IHA bypasses those Linears in favor of an inline
  `F.linear(x, mixed_weight)`. Result: `KeyError: 'blocks.0.attn.c_q.weight'`
  in `gptq_mixed_quantize`. Needs a fix to either keep the hooks hot or change
  the calibration path before this idea can be evaluated.
- **Orchestrator `set -e` doesn't catch torchrun failures** because the run is
  wrapped in `... | tee log | tail -200 >> orchestrator.log`. Bash only checks the
  last command in the pipeline, so a failed torchrun leaves exit code 0 at the
  pipeline level. This is fortunate for us — mtp/evalloop3 still ran after iha
  crashed — but it means downstream "done at" markers in the orchestrator log
  do NOT imply success. Use the presence of `quantized_ttt val_loss` in
  `pr1493_<name>_s42.txt` as the success gate.
- **Submission size 16,031,886 B / 16,033,898 B is OVER the 16 MB limit by
  ~30 KB.** The pr1493 train script (~55 KB) is too big as-is — code minification
  is required before any of these can be submitted, regardless of BPB.

## How this file is updated

A scheduled cron (`7,17,27,37,47,57 * * * *`) re-runs the documentation prompt every
10 minutes, parses any newly-finished experiment logs (gated by the orchestrator's
`=== [N/5] <name> done at` line), and pushes to the `shikhar` branch only when the
markdown actually changed. Current cron job ID `324942df`. Stops on session close or
via `CronDelete 324942df`.
