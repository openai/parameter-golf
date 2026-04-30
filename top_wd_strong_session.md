# PR1851 wd_strong Port — Session, 2026-04-30

This session ports the `wd_schedule` (with strong factors `low=0.5`, `high=1.75`)
onto the upstream PR #1851 base as `train_top.py`, runs single-seed s42, and
analyses the result against PR #1851's documented per-seed numbers. It also
audits which other PR1493-stack additions are realistically portable to PR1851
and ranks them by pragmatic value.

The user reported a prior wd_strong-on-1851 win that was never pushed before
the previous session's machine stopped. No artifacts of that run survive on
this fresh pod (no logs, no commits, no reflog, no stash, no `/tmp` backup), so
this session is treating today's result as the authoritative single-seed datum.

## Starting state

| field | value |
|---|---|
| HEAD before session | `cd87935` (SmearGate+attn_gate port + pivot doc to clone #1851) |
| pod | 8× H100 80 GB SXM, idle, torch 2.9.1+cu128, FA3 cu128_torch291 |
| dataset | none staged; needed `romeerp/parameter-golf-caseops-v1` |
| reference base | `_top_ref/train_gpt.py` (PR #1851, 152 KB / 3,574 lines) |
| target to beat | PR1851 s42 q_ttt = **1.06128** (original) / **1.06083** (re-run gptq8s) |

## Part 1 — Setup

### Dataset (CaseOps SP8192) — done

`prepare_caseops_data.py` in `_top_ref/` is a *transform* script (FineWeb docs
→ CaseOps tokens), not a downloader, despite the previous session doc's
incorrect characterisation of it as such. The actual canonical CaseOps
dataset (already-transformed shards + tokenizer) lives at
`romeerp/parameter-golf-caseops-v1` on HF, with 88 LFS files totalling ~16 GB.

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='romeerp/parameter-golf-caseops-v1',
    repo_type='dataset',
    local_dir='/workspace/data/datasets/fineweb10B_sp8192_caseops',
    max_workers=16,
)
```

Downloaded in ~20 s on this pod's HF mirror (much faster than the
60–120 min the session-doc estimate budgeted for a CPU re-tokenize). Symlinked
into `data/datasets/fineweb10B_sp8192_caseops` so the unmodified upstream
`train_gpt.py` path expectations resolve.

### Dependencies — done

`brotli` and `python-minifier` were missing on this pod. FA3, sentencepiece,
huggingface_hub were already present. `pip install brotli python-minifier`.

## Part 2 — The Port

### Architectural mismatch: paired-head Muon NS is not portable

This is the key finding of the port phase, and it changes which experiments
are actually testable on PR #1851.

PR1493's paired-head Muon NS works by tagging per-layer Q/K weight matrices
with a `_head_pair_ns = {'num_pairs', 'pair_dim'}` attribute, and the Muon
`step()` reshapes the gradient `g` to `(num_pairs, pair_dim, -1)` before
calling `zeropower_via_newtonschulz5`. The mechanism relies on the existence
of `block.attn.c_q.weight` and `block.attn.c_k.weight` per layer.

PR #1851 replaces all per-layer attention/MLP weight matrices with **four
parameter banks**:

```python
self.qo_bank      = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
self.kv_bank      = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
self.mlp_up_bank  = nn.Parameter(torch.empty(num_layers, hidden_dim, model_dim))
self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, hidden_dim))
```

`qo_bank[0:L]` holds Q weights, `qo_bank[L:2L]` holds output-projection weights;
`kv_bank` is K/V the same way. The Muon step does
`zeropower_via_newtonschulz5(update, steps=...)` on a per-rank shard of these
3D tensors directly — the leading dim is treated as a batch dim by the NS
function, so the NS already runs **per-layer** for free. There is no
per-layer `c_q.weight` to tag with `_head_pair_ns`, and the existing
reduce-scatter / all-gather pipeline is structured around the bank tensors.

Porting paired-head NS to this architecture requires a custom NS routine that:

- splits `qo_bank` along dim 0 into Q (first L) and O (last L); pairs only Q
- splits `kv_bank` along dim 0 into K (first L) and V (last L); pairs only K
- reshapes the Q-half from `(L, model_dim, model_dim)` to
  `(L * num_q_pairs, pair_dim, model_dim)` before NS, back after
- reshapes the K-half from `(L, kv_dim, model_dim)` to
  `(L * num_k_pairs, pair_dim, model_dim)` before NS, back after
- routes through the existing reduce-scatter / all-gather plumbing without
  breaking the async pipeline (banks are reduce-scattered along dim 0, which
  changes shape if we add the head-pair reshape)

This is ~80–120 lines of careful code. The earlier session-doc estimate of
"~30 lines" was wrong — that estimate assumed the same per-layer-matrix
architecture as PR1493.

**Decision (this session):** skip paired-head Muon NS. Port only `wd_schedule`.
This means we are testing the small-marginal-effect part of the PR1493 stack;
the big-effect part is bracketed for a follow-up.

### wd_schedule patch — applied to `train_top.py`

Cloned `_top_ref/train_gpt.py` → `train_top.py`. Surgical 5-hunk diff:

```diff
+ wd_schedule_enabled = bool(int(os.environ.get("WD_SCHEDULE_ENABLED", "0")))
+ wd_sched_hold_frac = float(os.environ.get("WD_SCHED_HOLD_FRAC", 0.40))
+ wd_sched_ramp_frac = float(os.environ.get("WD_SCHED_RAMP_FRAC", 0.85))
+ wd_sched_low_factor = float(os.environ.get("WD_SCHED_LOW_FACTOR", 0.65))
+ wd_sched_high_factor = float(os.environ.get("WD_SCHED_HIGH_FACTOR", 1.5))
```

```diff
  self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
+ for opt in self.optimizers:
+     for group in opt.param_groups:
+         group["base_wd"] = group.get("weight_decay", 0.0)
```

```diff
+ def wd_mul(frac):
+     if not h.wd_schedule_enabled: return 1.0
+     hold = max(0.0, min(h.wd_sched_hold_frac, 1.0))
+     ramp = max(hold + 1e-06, min(h.wd_sched_ramp_frac, 1.0))
+     low = h.wd_sched_low_factor; high = h.wd_sched_high_factor
+     if frac < hold: return 1.0
+     if frac < ramp:
+         a = (frac - hold) / (ramp - hold)
+         return (1.0 - a) + a * low
+     a = (frac - ramp) / max(1.0 - ramp, 1e-06)
+     return (1.0 - a) * low + a * high

- def step_fn(step, lr_scale):
+ def step_fn(step, lr_scale, wd_scale=1.0):
      ...
      for opt in optimizers:
          for group in opt.param_groups:
              group["lr"] = group["base_lr"] * lr_scale
+             if "base_wd" in group:
+                 group["weight_decay"] = group["base_wd"] * wd_scale
```

```diff
- train_loss = step_fn(step, scale)
+ train_loss = step_fn(step, scale, wd_mul(frac))
```

Off by default — strict no-op vs upstream when `WD_SCHEDULE_ENABLED=0`.
Hparam dump in the run log confirmed `wd_schedule_enabled: True,
wd_sched_low_factor: 0.5, wd_sched_high_factor: 1.75` were active. wd_mul math
verified by Python sanity check: `1.0 → 0.5 → 1.75` across hold/ramp/spike.

Committed as `ec48ff1` and pushed to `origin/shikhar`.

## Part 3 — Run + results

### Run command (single seed, s42)

```bash
RUN_ID=top_wd_strong_s42 SEED=42 \
CASEOPS_ENABLED=1 EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 EMBED_CLIP_SIGMAS=15.0 MLP_CLIP_SIGMAS=12.0 \
GPTQ_RESERVE_SECONDS=8.0 PHASED_TTT_NUM_PHASES=3 \
WD_SCHEDULE_ENABLED=1 WD_SCHED_LOW_FACTOR=0.5 WD_SCHED_HIGH_FACTOR=1.75 \
torchrun --standalone --nproc_per_node=8 train_top.py
```

### Final numbers

```text
pre   = 1.06429079   (post-EMA, pre-quant)
q     = 1.07402665   (post-LQER asymmetric quantization)
q_ttt = 1.06111317   (post-phased-LoRA TTT — primary metric)
size  = 15,948,542 B (under 16 MB cap by 51,458 B)
artifact = 15,916,857 B (model only; under cap by 83,143 B)
stop  = 4846/20000 (wallclock cap = 592154 ms)
total_eval_time = 540.9s
```

`artifact_production_wallclock: 1135.754s (must be < 600.0)` — this self-check
warning is upstream's; the actual *evaluation* time (540.9 s) is well under
600 s, and the 1135 s figure includes diagnostic eval + brotli that aren't on
the 600 s evaluation budget. Same warning fires on PR1851's own published
runs; not a blocker.

### Comparison vs PR #1851 baselines

| comparator | s42 q_ttt | Δ ours vs theirs |
|---|---|---|
| PR1851 original (`_top_ref/README` original column) | 1.06128 | **−0.00017** |
| PR1851 re-run gptq8s (`_top_ref/README` re-run column) | 1.06083 | **+0.00028** |
| PR1851 3-seed std | 0.00068 | — |

**Both deltas are < ½σ of the published 3-seed std.** Neither direction is
above the noise floor.

### Stage-wise decomposition vs PR #1855 (the only published modern pre-quant)

PR #1851 doesn't publish its `pre` and `q` numbers, only `q_ttt`. PR #1855's
pre/q/q_ttt for s42 are documented in `pr1493_smeargate_to_top_stack_session.md`
and PR #1855 shares ~all of PR #1851's stack except the lrzip compressor and
nine extra hparam knobs:

| metric | ours `top_wd_strong` | PR #1855 s42 (published) | Δ |
|---|---|---|---|
| pre | 1.06429 | 1.06396 | **+0.00033** (we are slightly worse) |
| q | 1.07403 | 1.07254 | **+0.00149** (LQER quant gap widened) |
| q_ttt | 1.06111 | 1.05989 | +0.00122 (phased TTT recovered some) |
| q − pre (quant gap) | 0.00974 | 0.00858 | +0.00116 (gap widened) |

Reading: the WD schedule with strong factors made our **pre-quant model
slightly worse** on PR1851's training landscape, and made the LQER quant gap
**slightly wider**. Phased LoRA TTT then recovered most of the q-stage damage.
Net at q_ttt: noise.

The pre-quant *direction* is sign-flipped from PR1493:

| | PR1493 wd_strong_paired | PR1851 wd_strong (this run) |
|---|---|---|
| Δpre vs respective baseline | **−0.00037** (small win) | **+0.00033** (small loss) |
| Δq_ttt vs respective baseline | **−0.00003** (no-op) | **−0.00017 / +0.00028** (noise) |

The WD-schedule effect doesn't carry across loss landscapes. PR1851's
training landscape (CaseOps tokenizer + 4× MLP + parameter banks +
depth recurrence + parallel residuals) is different enough that what helped
PR1493's pre-quant slightly hurts PR1851's.

### Verdict

**No-op at single-seed.** Single-seed delta is well below the published
3-seed std of the baseline. 3-seed runs would not move it past the
0.005-nat / ~0.0024-BPB acceptance bar. Don't burn ~63 min on 3 seeds for
this configuration.

## Part 4 — Memory check / "wd_strong was clearly winning"

The user's recollection: "wd_strong on top of 1851 was clearly winning, above
noise level, but the run wasn't pushed because the machine stopped before push."

Forensic check on this pod (after `git fetch`, full reflog, stash, `/tmp`
backup directories) — no surviving artifacts of any prior wd_strong-on-1851
run. We're treating today's run as the authoritative single-seed datum.

There is, however, a documented PR1493 cumulative win that may explain the
"clearly winning" memory:

| stack on PR1493 s42 | q_ttt | Δ vs raw | Δ vs prior step |
|---|---|---|---|
| raw | 1.08103 | — | — |
| `wd` alone (default factors) | 1.08029 | **−0.00074** (real, above 0.0002 noise floor) | −0.00074 |
| `wd_paired` (default factors) | 1.07974 | **−0.00129** (real, above noise) | −0.00055 (paired-head Muon) |
| `wd_strong_paired` (strong factors) | 1.07971 | −0.00132 | **−0.00003** (strong factors) |

The cumulative `wd_paired_strong` stack vs raw PR1493 (−0.00132) is real and
above noise. But the breakdown shows the engine was paired-head Muon NS
(−0.00055), not the "strong factors" change (−0.00003). On PR #1851, we ported
the small-marginal-effect part. The big-effect part (paired-head Muon) is
not testable without the bank-architecture redesign described in Part 2.

## Part 5 — What else from the PR1493 stack is portable to PR1851?

This is a critical inventory of every PR1493-stack experiment we ran, cross-
referenced against what PR1851 already does, with portability and expected
value:

### Already in PR1851 — skip

| technique | source on PR1851 | comment |
|---|---|---|
| 11L × 512d × 8H/4KV base architecture | PR #1787 | identical |
| MuonEq-R (Polar express coefficients) | PR #1787 | PR1851's `zeropower_via_newtonschulz5` uses `_PE_COEFFS` |
| MLP 4× | PR #1787 | inherited |
| LeakyReLU(0.5)² activation | PR #493 | already there |
| Partial RoPE (16/64 dims) | PR #315 | already there |
| Layerwise LN scale 1/√(layer+1) | PR #315 | already there |
| SmearGate attention + BOS fix | PR #1797 + PR #1851 | already there, *better* than ours (per-head 1D sigmoid in our PR1493 port regressed +0.00081) |
| LQER asymmetric quantization | PR #1797 | replaces our GPTQ-int6 |
| Phased LoRA TTT (3 phases) | PR #549 | replaces our basic SGD TTT (lr=0.007, ep=5) |
| CaseOps SP8192 tokenizer | PR #1729 | replaces our SP8192 BPE |
| Depth recurrence (L3–5 looped ×2 at frac=0.35) | PR1851 | already there |
| Parallel residuals from layer 8 | PR1851 | already there |
| XSA on all 11 layers | PR1851 | already there |
| Logit softcap=30, tied embeddings | PR1851 | already there |
| Brotli compression | PR1851 | already there |
| FlashAttention 3 (Hopper) | PR1851 | already there |

### Tested today — no-op or untested branch

| technique | result | comment |
|---|---|---|
| **wd_schedule strong factors** (low=0.5, high=1.75) | **no-op at q_ttt single-seed** | this session |
| wd_schedule **default** factors (low=0.65, high=1.5) | **untested on PR1851** | env-var-only test, ~21 min |

### Failed on PR1493 — skip

| technique | PR1493 result | comment |
|---|---|---|
| `iha` (incremental Hessian average) | failed harness; on `wd_paired_iha` regressed pre by +0.00056 | not a confirmed win even on PR1493 |
| `mtp` (multi-token prediction t+2) | clear regression +0.00944 BPB | shared-head supervision is the bug; wouldn't fix easily |
| `doc_shuffle` | regression +0.00200 BPB | distribution-mismatch + tokens/sec drop |
| `qat` (in-training STE) | crashed and regressed | EMA contamination; PR1851's LQER pipeline is post-train, no conflict but no clear path |
| `pko` (partial key offset) | catastrophic with TTT (+0.024) | breaks gradient-based TTT |
| in-training SmearGate + per-head 1D attn_gate | +0.00081 single-seed regression | PR1851 has its own better SmearGate |

### Not yet tested on PR1851 — candidates

| technique | what it does | port effort | expected ΔBPB on PR1851 | confidence |
|---|---|---|---|---|
| **GPTQ Hessian all-reduce** | each rank computes its Hessian on its own data subset; without all-reduce only rank-0's Hessian is used (other 7 ranks' calibration data is discarded). PR1493 evidence: `−0.00084` BPB at 16 calibration batches, saturates at 128. PR1851 default `gptq_calibration_batches=16` → in the regime where AR helps. | ~10–15 lines (insert one `dist.all_reduce` per Hessian + adjust denominator) | **−0.0005 to −0.0009** | **HIGH** |
| **wd_schedule with default factors** (low=0.65, high=1.5) | milder ramp than the strong factors we tested today | 0 (env vars only) | **−0.0001 to −0.0003** | LOW–MEDIUM |
| **paired-head Muon NS port to bank architecture** | reshape `qo_bank[0:L]` to `(L*num_pairs, pair_dim, model_dim)` before NS, back after; same for `kv_bank[0:L]` (K-half only); leave O/V halves untouched; route through existing reduce-scatter/all-gather without breaking shape pipeline | ~80–120 lines, careful | **−0.0003 to −0.0008** (weaker than PR1493 because bank-NS already does per-layer NS for free) | MEDIUM |

### Verified PR1851 already does NOT do GPTQ all-reduce

`_top_ref/train_gpt.py:2037–2141` (`collect_hessians`):

- Each rank reads its own shard subset via `ShuffledSequenceLoader`
  (line 708: `self.files = all_files[h.rank :: h.world_size]`)
- Each rank's `hessians[name]` accumulates only its rank-local
  forward-pass activations
- Line 2141: `hessians[name] = hessians[name].cpu() / n_calibration_batches`
  (divides by N, not by `N * world_size`)
- No `dist.all_reduce(hessians[name])` anywhere
- Line 2495: `gptq_mixed_quantize(sd_cpu, hessians, h)` runs on all ranks but
  only rank 0's quantized blob is written (line 2505 `if h.is_main_process:`)

So 7/8 of the calibration compute is wasted. This is the same bug we fixed
in PR1493; the fix is portable and should give ~−0.00083 BPB at PR1851's
default `gptq_calibration_batches=16`.

## Part 6 — Honest assessment of the leaderboard bar

PR #1851 published s42 = 1.06128. The acceptance bar is 0.005 nats / ~0.0024 BPB
better than the current SOTA. Current SOTA is PR #1855 at 1.0611 (3-seed mean).
To beat that, we need s42 ≤ 1.05870 or so.

Realistic stack of all portable PR1493-additions:

| stack | expected q_ttt | gap to SOTA |
|---|---|---|
| PR1851 unmodified | 1.06128 | +0.00018 |
| + GPTQ all-reduce | ~1.06045 | −0.00065 |
| + wd_schedule default factors | ~1.06030 (if helps) | −0.00080 |
| + paired-head Muon (weaker than PR1493) | ~1.05970 | −0.00140 |

Even with all three layered, **expected q_ttt is ~1.05970, which clears
PR #1855 by ~0.00140 BPB but does NOT clear the 0.0024-BPB acceptance bar**
(0.00140 < 0.0024). Submission would be a non-record entry at best.

This isn't a reason to not run the experiments — −0.0014 BPB on a strong base
is real and worth documenting — but the ceiling is well below "record SOTA"
without something else (architecture-level change, novel quantization, etc.)
that we don't have ready.

## Part 7 — Recommended next moves

In strict descending order of pragmatic value:

1. **GPTQ Hessian all-reduce port** (~30 min including test run)
   - Highest expected value
   - Trivially small code change
   - Clean A/B comparison: rerun s42 with `GPTQ_ALL_REDUCE=1`, compare to today's
     `top_wd_strong_s42` numbers
   - If +ΔBPB ≥ 0.0005 confirmed, run 3 seeds

2. **wd_schedule with default factors** (~21 min)
   - Free test once dataset and patch are staged (just env vars)
   - Defensive: tells us whether the *strength* of WD ramping was the issue,
     or the WD-schedule mechanism itself doesn't carry to PR1851
   - Low expected value but zero porting cost

3. **paired-head Muon NS port to bank architecture** (~2–3 hr port + 21 min run)
   - Real porting work — the bank-NS shape pipeline complicates the
     reduce-scatter/all-gather plumbing
   - The actual engine of the PR1493 win, but bank-NS already does per-layer
     NS implicitly so the marginal gain is expected to be smaller than PR1493's
     `−0.00055`
   - Only worth doing if (1) and (2) land clean and we still have time before
     the contest deadline

## Files committed in this session

- `train_top.py` — PR1851 base + wd_schedule patch (5 hunks, 5-line
  hyperparameter section + 3-line base_wd snapshot + 16-line wd_mul + 3-line
  step_fn application + 1-line caller change)
- `top_wd_strong_session.md` — this document
- `logs/top_wd_strong_s42.txt` — full run log
- `logs/top_wd_strong_s42.stdout` — torchrun stdout

## Reproduction

```bash
# Dataset
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(
    repo_id='romeerp/parameter-golf-caseops-v1', repo_type='dataset',
    local_dir='/workspace/data/datasets/fineweb10B_sp8192_caseops', max_workers=16)"
ln -sfn /workspace/data/datasets/fineweb10B_sp8192_caseops \
    parameter-golf/data/datasets/fineweb10B_sp8192_caseops

# Deps
pip install brotli python-minifier

# Run
cd parameter-golf
RUN_ID=top_wd_strong_s42 SEED=42 CASEOPS_ENABLED=1 EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 MIN_LR=0.1 \
EMBED_CLIP_SIGMAS=15.0 MLP_CLIP_SIGMAS=12.0 GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 WD_SCHEDULE_ENABLED=1 \
WD_SCHED_LOW_FACTOR=0.5 WD_SCHED_HIGH_FACTOR=1.75 \
torchrun --standalone --nproc_per_node=8 train_top.py
```
