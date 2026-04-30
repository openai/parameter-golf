# PR1493 → Top Stack Pivot Session — 2026-04-30

This session ports BOS-fixed SmearGate + per-head attention output gate onto the
PR1493 wd_strong_paired baseline, validates the port empirically across two
real 8×H100 runs, then pivots to a much larger plan: cloning the current
top-of-leaderboard stack (PR #1851) as the new base and layering only our
PR1493-family differentiators on top.

## Starting state

| field | value |
|---|---|
| HEAD before session | `468be92` (GPTQ all-reduce + damp/block sweep) |
| baseline (single seed s42) | `wd_strong_paired` q_ttt = 1.07971 |
| local file | `train_pr1493.py` (553 lines, golfed) |
| pod | 8× H100 80GB SXM, torch 2.9.1+cu128, FA3 cu128_torch291 |
| dataset | `kevclark/parameter-golf` SP8192, 80 train shards (16 GB) at `/workspace/data/` |

## Part 1 — BOS-fixed SmearGate + attn output gate port (delivered)

### What was added to `train_pr1493.py`

5 new env-driven hparams:

```
SMEARGATE_ENABLED   default 1
SMEARGATE_BOS_ID    default 1   (SP8192 BOS)
SMEARGATE_INIT      default 3.0 (sigmoid≈0.95 ≈ near pass-through)
ATTN_GATE_ENABLED   default 1
ATTN_GATE_INIT      default 0.0 (additive form, identity at init)
```

`SmearGate` module (one-line golf form). BOS-fix mask: at positions where
`input_ids[t] == bos_id`, the smear contribution is forced to zero, so the
final token of doc N can never leak into the BOS of doc N+1. Verified by a
focused unit test (positions where `input_ids == 1` pass through exactly).

`CausalSelfAttention.attn_gate`:
- v1 (1D, sigmoid): `(num_heads,) → sigmoid → multiplicative`. Per-head scalar.
- v2 (2D, additive): `(num_heads, head_dim) → (1 + g) → multiplicative`.
  Identity at init when `init=0.0`. Matches upstream PR #1667's
  "weight-init zero, identity at start" wording.

`smeargate.smear_gate` is a top-level GPT-level parameter (not inside `blocks`),
so an explicit `Optimizers.scalar_params.append(base_model.smeargate.smear_gate)`
was added — the existing `block_named_params` loop wouldn't catch it.

`CONTROL_TENSOR_NAME_PATTERNS` extended with `smear_gate, attn_gate`.

100% of model parameters covered by some optimizer (verified via smoke test).

### Real-run results (single seed s42, 8×H100, PR1493 base + wd_strong_paired)

| run | pre | q | q_sw | q_ttt | Δ q_ttt vs baseline 1.07971 | artifact |
|---|---|---|---|---|---|---|
| baseline (`wd_strong_paired`) | 1.08573 | 1.09874 | 1.08194 | **1.07971** | — | 16,030,578 (over by 30,578 B) |
| `smear_attngate_s42` (1D sigmoid gate) | 1.08663 | 1.09887 | 1.08220 | **1.08052** | +0.00081 | 16,038,047 (over by 38,047 B) |
| `smearonly_s42` (gate disabled) | 1.08601 | 1.09834 | 1.08170 | **1.07998** | +0.00027 | 16,035,853 (over by 35,853 B) |
| `smear_gate2d_s42` (2D additive gate) | killed at step ~4000, mid-train val 1.1051 (best mid-train) | killed | killed | killed | — | — |

### Findings

1. **Per-head 1D sigmoid attn_gate is bad on our stack.** +0.00081 q_ttt — mostly
   driven by a +0.00090 pre-quant regression, not a quant artifact. Our 8-per-layer
   per-head scalar is much smaller than upstream PR #1667's gate (96/layer = 8 × 12,
   width=12 per head). Likely undercapacity, but also possibly the wrong functional form
   (sigmoid attenuates to 0.95×, not 1.0× pass-through).

2. **BOS-fixed SmearGate alone is best at q and q_sw** (-0.00040 q, -0.00024 q_sw vs
   baseline) — small consistent quant-and-window-level win. **But the q_ttt gain
   evaporates** because the SmearGate-trained residual stream changes the stats our
   SGD TTT was tuned for: TTT lift drops from 0.0022 (baseline) to 0.0017 (smearonly).
   Net: q_ttt = +0.00027 (within seed noise).

3. **2D additive gate looked promising mid-training** (val 1.1051 at step 4000 vs
   smearonly's 1.1098) but was killed before final eval. Not enough evidence to
   draw conclusions.

4. **Code budget cost.** Adding the SmearGate + gate code adds ~7 KB to the
   submission. Even with the slightly smaller quant blob (~52 KB smaller weights
   compress better), total submission goes from baseline 16,030,578 to
   smear_attngate 16,038,047 (+7,469 B). All single-seed runs on this stack
   bust the 16 MB cap; the wedges that actually shrink the artifact (per-group
   compression) are not in PR1493.

### Conclusion of Part 1

**The architectural port is correct (BOS mask verified, parameter coverage
verified, smoke + unit tests pass).** It just doesn't help on our PR1493
base at single-seed. Upstream PR #1667 won 0.0035 BPB on a different base
(PR #1586) — that base apparently has different residual-stream stats that
make the gate productive. On PR1493 + wd_strong_paired, SmearGate by itself
is +/- noise; the per-head 1D gate is a small regression.

### Files added / modified in Part 1

```
train_pr1493.py                   modified (5 hparams + SmearGate class +
                                  attn_gate in CausalSelfAttention + plumbing)
run_smear_attngate.sh             new (1D sigmoid gate)
run_smearonly.sh                  new (gate off, smear on)
run_smear_gate2d.sh               new (2D additive gate)
run_chain_smear_experiments.sh    new (chains smearonly → smear_gate2d)
run_mom97.sh                      new (drafted but not run; superseded by pivot)
logs/smear_attngate_s42.txt       new (full run log)
logs/smearonly_s42.txt            new (full run log)
logs/smear_gate2d_s42.txt         new (partial — killed mid-training)
```

## Part 2 — Critical leaderboard analysis

### Top-of-leaderboard PR status (verified via GitHub API)

| rank | PR | val_bpb | author | merged? | merged by | open disputes? |
|---|---|---|---|---|---|---|
| #1 | #1855 | 1.0611 | codemath3000 | ✅ 2026-04-29T19:18:38Z | cocohearts | **YES — val_docs=10_000 (jfc43, 2026-04-30 — unresolved)**; lrzip-as-runtime-dep (resolved in favor of submission) |
| #2 | #1851 | 1.0614 | aquariouseworkman | ✅ 2026-04-30T01:10:16Z | cocohearts | shared val_docs=10_000 risk (inherited); 3-seed validation via PR #1868 by Christopher-Lee-McClendon |
| #3 | #1787 | 1.0634 | nprime06 | ✅ | — | — |

### "Scylla-shaped" risk

The val_docs=10_000 dispute (`prepare_caseops_data.py` uses 10k val docs vs the
canonical 50k from `data/download_hf_docs_and_tokenize.py`) is **open** as of
2026-04-30. If the maintainer rules against the CaseOps chain, **6 leaderboard
positions vacate at once** (PRs #1736, #1769, #1787, #1851, #1855, #1868), and
the new top would be CaseOps-free stacks — i.e., the PR1493 family our work
descends from.

### Pre/q/q_ttt breakdown comparison (head-to-head with #1855 from real logs)

Earlier framing of "the gap is LQER + phased-TTT only" was wrong. Walked it back.

| metric | our `wd_strong_paired` | PR #1855 s42 (top) | gap |
|---|---|---|---|
| pre  | 1.08573 | **1.06396** | **+0.02205** ← biggest single gap |
| q (with LQER for theirs) | 1.09874 | **1.07254** | +0.02580 |
| q_ttt | 1.07971 | **1.05989** | +0.01982 |
| artifact | 16.030 MB (over) | 15.897 MB (under) | — |

**The pre-quant gap of 0.022 BPB is bigger than the entire 0.020 BPB final gap.**
That means the leaderboard wedge is dominated by training-level wins (CaseOps
tokenizer, SparseAttnGate, PolarNS coefficients, hparam compounding), NOT by
LQER or phased-TTT alone.

Decomposition:
- pre-quant gap (0.022): CaseOps + SparseAttnGate + PolarNS + 9-knob hparam stack
- (q − pre) gap: ours 0.013, theirs 0.009 — LQER closes ~0.004 of this for us
- (q − q_ttt) gap: ours 0.019 (basic SGD TTT), theirs 0.013 (phased LoRA TTT) —
  **phased TTT actually has a *smaller* lift than our SGD TTT**, because their
  q is much better, leaving less room. Phased TTT may not help us.

## Part 3 — Pivot decision: clone #1851's `train_gpt.py` as the new base

### Why #1851 not #1855

Both are merged + listed on the README leaderboard, both share the val_docs risk.
Differences:

| | #1855 (top) | #1851 (#2) |
|---|---|---|
| q_ttt | 1.0611 | 1.0614 (+0.0003) |
| compressor | per-group lrzip+brotli | brotli only |
| `lrzip` system dep | yes (rule-3 dispute, resolved in favor) | **no** |
| comment count / drama | 10 | 5 |
| extra hparam stack | 9 knobs | inherits PR #1787 stack |

**Picked #1851 because: same q_ttt within noise, no lrzip system dep, fewer
moving parts, fewer disputes.** Trade-off: 0.0003 BPB worse than #1855 (within
seed noise) and skips the 9-knob hparam compounding (we can layer those later).

### Why clone-then-add, not port-into-ours

Their `train_gpt.py` is 152 KB / 3,574 lines vs our `train_pr1493.py` at 553
lines. Porting their 2,500+ new lines into ours = ~25–30 hr engineering with
high integration risk (compile graph, distributed work queue, byte sidecar,
chunk-window math, BatchedLinearLoRA). Cloning their file as a new base
(`train_top.py`) and layering our few small differentiators is ~5–8 hr with
much lower risk.

Trade-off: codebase identity changes. `train_pr1493.py` stays preserved as a
rollback path; the new file is named distinctly.

### Our differentiators to layer onto #1851's base

(Confirmed by reading both files. Will be ported as patches to `train_top.py`.)

1. **`paired_head_muon_enabled`** (~30 lines)
   - 3D Newton-Schulz reshape on q/k matrices in pairs of attention heads
   - In our wd_strong_paired baseline, gives `tagged=22` (= 11 layers × q/k)
2. **`wd_schedule_enabled`** with hold/ramp/low/high factors (~20 lines)
   - Used: `WD_SCHED_LOW_FACTOR=0.5`, `WD_SCHED_HIGH_FACTOR=1.75`
3. **`gptq_all_reduce`** (~5 lines)
   - Per-rank Hessians dist.all_reduce'd before averaging — recovers low-shard configs
4. **`gptq_damp` / `gptq_block_size`** env knobs (~5 lines, drop-in)
5. **(skip)** our SmearGate / attn_gate port — #1851 has its own (better-validated) version
6. **(skip)** iha, mtp, doc_shuffle — failed experiments

### The val_docs=10_000 decision

I'm going to **match #1851 exactly** (val_docs=10_000) so we can A/B against the
leaderboard. Same risk profile as the actual #1 and #2. If the ruling lands
against, we toggle (will plumb a `CASEOPS_VAL_DOCS` env var so we can flip
without code changes).

### Compute-budget facts (from #1851's actual log)

- 600 s training cap
- Phased TTT compile warmup: ~138 s
- Phased TTT eval: ~509 s
- Brotli compression: ~65–67 s (CPU-bound, post-quant)
- Total per run wallclock: ~21 min
- Peak GPU memory: 41.7 GB / 47 GB reserved (fits comfortably on 80 GB H100)

This means we get ~3 ablation runs per hour — fewer cycles than the basic
PR1493 (15 min) flow.

### CaseOps data — already published, no retokenization needed

`romeerp/parameter-golf-caseops-v1` on HF has the canonical CaseOps dataset:
- 80 train shards (`fineweb_train_*.bin` ≈ 16 GB)
- val + val_bytes sidecar
- CaseOps tokenizer (`fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`)
- manifest

Layout matches `#1851`'s `_default_caseops_data` expectation when materialized
under `data/datasets/fineweb10B_sp8192_caseops/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/`.
Saves us 1–2 hr of CPU retokenization. Background download in progress at
session-end.

### Concrete plan for next session

1. **Reproduce #1851 unmodified** (s42, target q_ttt = 1.06128 ± 0.0005). 
   ⚠️ If we can't reproduce, STOP. Do not layer anything onto a non-reproducing base.
2. **Layer paired-head Muon NS only** — single seed, check delta.
3. **Layer wd_schedule** — single seed, check delta.
4. If both improve, run 3 seeds and write a record submission.

### Key compute numbers we should defend in any future submission

- Training cap: 600 s
- Artifact cap: 16,000,000 bytes
- val_tokens (CaseOps val): 47,851,520 (confirmed in #1851 log)
- Hardware envelope: 8× H100 80 GB SXM at FA3 cu128_torch291 / torch 2.9.1+cu128

## Reference files cached locally

```
_top_ref/
  train_gpt.py                      152 KB / 3,574 lines (#1851 base)
  lossless_caps.py                   29 KB / 833 lines (CaseOps transform)
  prepare_caseops_data.py             7 KB / 168 lines (downloader/prep)
  README.md                                    (#1851 reproduction recipe)
```

Pulled from
`https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141`.

## Open questions / risks to track

- val_docs=10_000 ruling (jfc43 → maintainer cocohearts) — recheck daily
- Whether our paired-head Muon NS still helps on phased-LoRA-TTT base
- Whether our wd_schedule still helps when CaseOps changes the loss landscape
- Memory headroom: we run at 41.7 GB peak; layering paired-head NS adds ~0 (just
  a reshape), but if anything else gets layered, watch for OOM
