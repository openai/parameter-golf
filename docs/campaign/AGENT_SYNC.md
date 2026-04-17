# Agent Sync

Date: 2026-04-01

## Current Objective

Reproduce PR `#1610` directly and layer a full-vocab posterior corrector to push below 1.070 BPB.

Execution plan: `docs/campaign/PLAN_PR1610_CORRECTOR.md` (locked Revision 3, 2026-04-14).

Key decisions:
1. source base is `#1610` `train_gpt.py` at SHA `ca191953` -- NOT patched D variant
2. non-record PR `#1598` remains open and frozen; do not edit unless reviewers ask
3. D / R1 evidence bundle is frozen; no more synthesis on that stack
4. budget: $212 RunPod (~35 runs), deadline Apr 30
5. fallback cascade defined if corrector < 0.001 BPB gain (export-only -> retrain -> writeup)

The mainline is now:
- reproduce `#1610` faithfully (Gate A: seed-0 within 0.003 of published 1.07258)
- validate 3-seed reproduction (Gate B: mean within 0.002 of published 1.07336)
- add full-vocab posterior corrector (eval-only on existing checkpoint first)
- multi-seed validation if corrector shows >= 0.002 BPB gain
- target: record-track PR at <= 1.070 BPB

## Challenge Reality

- Official leaderboard entry is **record-gated**, not top-5-open-entry.
- A record submission must beat the current official SOTA by at least `0.005` nats and show `p < 0.01`.
- Clean legal frontier: PR `#1610` at 1.0728 BPB (3-seed mean).
- Merged SOTA: PR `#1493` at 1.0810 BPB.

## Current Mainline Plan

### Phase 1: Session 05c-plus training bundle (MEASURED — quality positive, throughput regressed)

Plan: `docs/superpowers/plans/2026-03-30-session-05c-plus.md`
Code: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`

Four changes on the Session 03 anchor:
1. **XSA 4 → 11** — XSA on all layers (trivial constant)
2. **VE128 on layers 9-10** — shared ValueEmbedding (new module)
3. **Warmdown 3000 → 3500** — trivial constant
4. **LeakyReLU(0.5)²** — replaces relu in MLP (one line, aligns with PR #1019)

SWA is **not included** — dead code in both PR #1019 and #634 (collected but only EMA applied).

#### 8xH100 measured result (2026-03-31)

| Metric | 05c-plus | Anchor | Delta |
|--------|----------|--------|-------|
| sliding s64 val_bpb | **1.12557920** | 1.12904446 | **-0.00347** |
| pre_quant EMA exact | 1.14186715 | 1.14472403 | -0.00286 |
| int6 roundtrip exact | 1.14933197 | 1.15247273 | -0.00314 |
| step_avg_ms | **100.39** | 91.37 | **+9.02** |
| steps | 5977 | 6564 | -587 |
| bytes_total | 15,589,271 | 15,751,324 | -162,053 |

**Assessment**: Quality-positive (sliding s64 improved by 0.00347 despite 587 fewer steps). Throughput regressed materially (+9ms, exceeds the +5ms gate). Not a seed-validation branch yet. 05c-plus remains the best measured branch even after the 05f follow-up failed to improve it.

### Phase 3: Session 05f refinement (MEASURED — NEGATIVE vs 05c-plus)

Code: `records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/train_gpt.py`

Three additional changes on the 05c-plus base:
1. **BigramHash vocab 2048 → 3072** — reduces hash collisions
2. **BigramHash dim 128 → 112** — partially offsets parameter increase
3. **Warmdown 3500 → 4000** — more gradual cooldown

#### 8xH100 measured result (2026-03-31)

| Metric | 05f | 05c-plus | Delta vs 05c-plus |
|--------|-----|----------|-------------------|
| sliding s64 val_bpb | **1.12660664** | 1.12557920 | **+0.00103** |
| pre_quant EMA exact | 1.14190308 | 1.14186715 | +0.00004 |
| int6 roundtrip exact | 1.15026661 | 1.14933197 | +0.00093 |
| step_avg_ms | 100.51 | 100.39 | +0.12 |
| steps | 5977 | 5977 | 0 |
| bytes_total | 15,630,854 | 15,589,271 | +41,583 |

**Assessment**: Negative follow-up. BigramHash 3072x112 + warmdown 4000 did not improve 05c-plus and did not recover throughput. 05c-plus remains the best measured branch in this line.

### Phase 4: Checkpoint diagnostics (DONE — conclusions below)

Diagnostic conclusions (from 05c-plus and 05f checkpoint analysis):
- Export / quantization is not the current bottleneck
- 05c-plus and 05f have nearly identical float→int6 damage profiles
- No strong late-block-only quantization hotspot
- mlp.proj zeroing is real but diffuse, not the next tactical lever
- 05f failed because of training-side changes, not quantization
- VE is alive enough to keep
- Bigram is not the next lever
- XSA scope is the first knob to relax for throughput recovery

### Phase 5: Session 05g — XSA-8 throughput recovery (MEASURED — NEGATIVE)

Code: `records/track_non_record_16mb/2026-03-31_05g_xsa8_throughput/train_gpt.py`

Single change on the 05c-plus base:
- `xsa_last_n` 11 → 8 (XSA on layers 3-10, removed from layers 0-2)

#### 8xH100 measured result (2026-03-31)

| Metric | 05g | 05c-plus | Delta vs 05c-plus |
|--------|-----|----------|-------------------|
| sliding s64 val_bpb | **1.12584234** | 1.12557920 | **+0.00026** |
| pre_quant EMA exact | 1.14203044 | 1.14186715 | +0.00016 |
| int6 roundtrip exact | 1.14963535 | 1.14933197 | +0.00030 |
| step_avg_ms | **98.67** | 100.39 | **-1.72** |
| steps | 6080 | 5977 | +103 |
| bytes_total | **16,475,467** | 15,589,271 | **+886,196** |
| cap status | **+475,467 OVER** | -410,729 under | — |

**Assessment**: Negative. Small throughput gain (-1.72ms) but quality regressed (+0.00026 BPB), artifact blew the 16MB cap by 475KB, and is not a valid submission. The critical lesson: compression entropy is fragile enough that small training changes can blow the cap even without parameter count changes.

### Phase 6: Compression-path upgrade — brotli + byte-shuffle (GATE RESULT: PARTIAL)

Probe script: `compress_probe.py`

**Original thesis**: Compression headroom can buy meaningful quality through MLP width expansion.
**Measured result**: Brotli helps, but **width is capped at ~3.15x under uniform int6**. The bottleneck is bit-width, not the compressor.

#### Measured probe results (2026-03-31)

Best measured export candidate on both saved artifacts:
- `custom-shuffle + brotli-10`

05c-plus probe:
- baseline total with code: `15,596,605`
- best total with code: `15,446,614`
- gain vs baseline: `149,991` bytes (1.0%)
- headroom under cap: `+553,386`

05g probe:
- baseline total with code: `16,475,486` (over cap)
- best total with code: `15,442,565`
- gain vs baseline: `1,032,921` bytes (6.3%)
- headroom under cap: `+557,435`

#### Corrected width estimates (model_dim=512, mlp_mult=3.0, 11 layers)

| MLP mult | hidden | headroom | status |
|----------|--------|----------|--------|
| 3.10x | 1587 | +228 KB | FITS (safe) |
| 3.15x | 1612 | +69 KB | FITS (tight) |
| 3.20x | 1638 | -97 KB | OVER |
| 3.50x | 1792 | -1.08 MB | OVER |

Key findings:
- byte-shuffle contributes only `~8-10 KB`; custom serialization + brotli is the real win
- uniform int6 is the binding constraint on width, not the compressor
- PR #1089 achieves 3.5x MLP via mixed-precision GPTQ (int5/6/7), not via brotli alone
- mixed-precision *naive* quantization (int5 for safe layers) is the actual lever and does NOT require GPTQ

### Phase 7: Int5 tolerance probe (CONFIRMED — 924 KB savings, but BPB gate FAILED)

Mixed int5/int6 on 05c-plus float checkpoint:
- Baseline (custom-shuffle + brotli-10): 15,377,614 bytes
- Conservative schedule (9 tensors): 14,453,033 bytes
- Real savings: 924,581 bytes
- Byte savings are real, **but BPB damage is too large**.

#### Export A/B gate result (2026-04-01, serv-3336, A100-80GB)

| Metric | Path A (uniform int6) | Path B (conservative int5/int6) | Delta |
|--------|----------------------|--------------------------------|-------|
| Roundtrip BPB | 1.14983123 | 1.15627013 | **+0.006439** |
| Sliding s512 BPB | 1.12625753 | 1.13258369 | **+0.006326** |
| Model bytes | 15,377,614 | 14,453,033 | -924,581 |

**GATE FAIL**: `delta_sw = +0.006326 >= 0.002 threshold`. Cosine similarity (≥0.9954) is a poor predictor of BPB damage — forward pass nonlinearities amplify weight errors.

Conservative int5 tensor names (from probe):
`blocks.{1,2,4,5,6,7,8,9,10}.mlp.fc.weight`

### Phase 8: Session 06a — width fork (GATE FAILED, exploratory run in progress)

Code: `records/track_non_record_16mb/2026-04-01_06a_width325_mixed_int5_coprime_lateqat/train_gpt.py`

Five deltas on 05c-plus base:
1. `mlp_mult` 3.0 → 3.25 (hidden 1536 → 1664)
2. Mixed int5/int6 + custom-shuffle + brotli-10 export
3. Coprime distributed data loader (env: `LOADER_MODE=coprime`)
4. Late QAT via STE fake-quant (env: `LATE_QAT_THRESHOLD=0.15`)
5. Anchor/experiment string update

**Gate**: FAILED. Delta +0.006 BPB (3× threshold).

**Exploratory 8×H100 run** launched anyway (job 2734510, serv-3341) to test whether late QAT during training can close the gap. First run was confounded — QAT triggered at step 1 due to wallclock-based `lr_mul` instability on slow early steps. Bug fixed: QAT now requires `step >= 200`.

**Interpretation**: If the exploratory run's mixed-bit roundtrip gap is still > +0.002 vs uniform int6, mixed int5/int6 is dead on this model family. Salvageable parts (brotli, coprime loader, late QAT) can still go into a 3.0x or 3.10x fork without int5.

**Not bundled** (06b scope):
- Parameter banking + Parallel Muon
- Turbo-Muon optimizer
- EngramLite (bigram+trigram hash)

### Phase 2: GPTQ probe on 05c-plus architecture (DONE — NEGATIVE)

Session 05e: `records/track_non_record_16mb/2026-03-31_05e_gptq_probe/`
- Pre-quant EMA exact replay: `3.95543154`
- Naive int6 roundtrip exact replay: `3.96902897`
- GPTQ int6 roundtrip exact replay: `3.96902897`
- Result: **worse_than_naive_rowmax = 44/66 (67%)** — kill threshold exceeded
- No same-checkpoint BPB gain over naive replay
- LeakyReLU(0.5)² + VE128 did NOT unblock GPTQ
- GPTQ parked permanently for this model family

### Parked

- Session 05b GPTQ on current anchor (7 ablations, all failed, code proven correct)
- Saved-container FA3 throughput path
- TTT
- Broad novelty probes

## Session 05b GPTQ: Conclusive Parking Summary

Seven ablations on the same Session 03 checkpoint:

| # | Variant | gptq_diag | Roundtrip gap | Outcome |
|---|---------|-----------|---------------|---------|
| 1 | Initial smoke (1xH100) | 66/66 | +0.212 | Bug found |
| 2 | Loop fix + percentile search | 66/66 | +0.335 | Still bad |
| 3 | actorder=False | 66/66 | +0.395 | Worse |
| 4 | block_size=full | 66/66 | +0.395 | No change |
| 5 | Hessian normalize+damp | 66/66 | +0.337 | Identical |
| 6 | PR #1019 verbatim transplant | 66/66 | +0.337 | **Byte-identical MSE** |
| 7 | AR self-gen calibration | crash | N/A | Non-PD Hessian |

Key conclusion: ablation #6 proves the GPTQ code is functionally correct. The failure is model-specific, not a code bug. PR #1019 uses `leaky_relu(0.5)` while our anchor uses `relu`.

### Session 05e GPTQ Probe: Falsification Result (2026-03-31)

Tested whether LeakyReLU(0.5)² + VE128 unblocks GPTQ. **It does not.**

- Architecture: 05c-plus (warmdown 3500, XSA 11, VE128, LeakyReLU(0.5)²)
- 66 layers GPTQ'd, 68 Hessians collected
- **worse_than_naive_rowmax = 44/66 (67%)** — exceeds 50% kill threshold
- Same-checkpoint replay was flat vs naive:
  - pre-quant exact: `3.95543154`
  - naive roundtrip exact: `3.96902897`
  - GPTQ roundtrip exact: `3.96902897`
- Hessian collection: 2979ms, GPTQ quantization: 31488ms
- Hardware: RTXA6000 serv-2108 (export-only comparison, GPU speed irrelevant)

GPTQ is now parked permanently for this model family. The activation function is not the root cause.

## Fixed Reference Results

- Session 03 anchor (`8xH100`, `serv-3342`)
  - sliding s64 `val_bpb=1.12904446`
  - pre-quant EMA `val_bpb=1.14472403`
  - int6 roundtrip `val_bpb=1.15247273`
  - steps `6564`
  - step_avg `91.37 ms`
  - artifact `15751324` bytes

## Canonical Files

- Shared mutable state: `docs/campaign/AGENT_SYNC.md`
- Append-only measured results: `docs/campaign/results_log.jsonl`
- Stable rules: `CLAUDE.md`
- Checkpoint diagnostic utility: `scripts/diagnostics/diagnose_weights.py`
- Compression feasibility probe: `scripts/diagnostics/compress_probe.py`
- 05c-plus plan: `docs/superpowers/plans/2026-03-30-session-05c-plus.md`
- 05c-plus code: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`
- 05f follow-up: `records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/`
- GPTQ experiment (parked): `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/`
- GPTQ probe on 05c-plus: `records/track_non_record_16mb/2026-03-31_05e_gptq_probe/`
- 05c-plus checkpoint diagnostics: `diagnostics/2026-03-31_05c_plus/`
- Codex memory:
  - `docs/codex-memory/decisions.md`
  - `docs/codex-memory/project-state.md`
  - `docs/codex-memory/next-session.md`

## Session C Results (2026-04-15) — Corrector Skeleton + Legality Tests + Microbenchmark

**Status: COMPLETE. All Session C gates cleared. Ready for Session D (RunPod eval-only).**

### Deliverables

| Item | Status | Notes |
|------|--------|-------|
| `PrefixNgramCorrector` class | Done | Lines 15-59 of `train_gpt.py` |
| Hyperparameters (`CORRECTOR_ALPHA`, `CORRECTOR_ORDERS`) | Done | Lines 157-158 of `train_gpt.py` |
| `forward_ttt` `logit_bias` param | Done | Line 1049, broadcast at line 1122 |
| `eval_val_ttt_phased` corrector wiring | Done | Bias inject, update loop, SGD reset |
| `tests/test_corrector.py` (8 legality tests) | Done | **8/8 pass** |
| `scripts/bench_corrector_cpu.py` | Done | **GATE PASS** |

### Microbenchmark Gate Result

```
B=64, V=8192, orders=[8], alpha=0.3, chunk_size=32
get_logit_bias() + stack: 7921 μs / chunk-step
update():                    8.0 μs / token
Projected total overhead:   23.7s  (< 50s gate threshold)
```

### Corrector Design Notes

- `[B,1,V]` bias broadcasts across sequence dim — no dense `[B,S,V]` allocation
- Cache (`_lu`, `_lz`) invalidated on each `update()`, recomputed lazily on `get_logit_bias()`
- Two reset points: (1) per-doc-batch (co-located with `reusable_lora.reset()`), (2) after global SGD
- Chunk-level approximation: bias reflects state at chunk START (same for all 32 positions in chunk)
- `torch.compile` guard specialization handles `logit_bias=None` vs `Tensor` automatically

### Codex Review Findings and Resolutions (2026-04-15)

| Finding | Severity | Resolution |
|---------|----------|------------|
| Per-batch corrector reset contradicts spec | HIGH | **Fixed** — removed lines 2521–2523; only SGD-boundary reset remains (line 2688) |
| Chunk-static bias ≠ per-position prefix-scan | HIGH | **By design** — per-position scan requires [B,S,V] tensor or 32× GPU passes (both forbidden); LEGALITY_SPEC.md updated to document chunk-static as deliberate approximation |
| Multi-order `+=` log-deltas ≠ plan pseudocode `=` | MEDIUM | Deferred — single-order run 1a unaffected; deviation documented; revisit if run 1b shows unexpected behavior |
| Warmup uses real val tokens | MEDIUM | Pre-existing in #1610 base; not introduced by Session C; deferred |
| Test suite doesn't cover chunk-level integration ordering | LOW | Tests cover class legality properties; integration coverage deferred |

### Next: Session D — Eval-Only Corrector Trial

Blocked on: **Gate B pass** (3-seed reproduction mean within 0.002 of published 1.07336).

Once Gate B clears:
1. Launch eval-only ablation run 1a: `CORRECTOR_ALPHA=0.3 CORRECTOR_ORDERS=8`
2. Launch run 1b: `CORRECTOR_ALPHA=0.3 CORRECTOR_ORDERS=5,8,12`
3. Kill criterion: all 1a–1c < 0.001 BPB gain → kill corrector, go to fallback cascade

Key env vars added:
- `CORRECTOR_ALPHA` (default `"0.0"` = corrector disabled)
- `CORRECTOR_ORDERS` (default `"8"`, comma-sep for multi-order)

## Workspace

- Local repo: `/home/amay/Work/parameter-golf`
- Remote repo: `/netscratch/$USER/parameter-golf`

Use `git clone` and `git pull` by default.

## Session 1 corrector closeout — 2026-04-17

Audit of the PR #1610 posterior-corrector implementation on branch
`submission/pr1610-corrector` ground-truthed against live working-tree
files. Two prior reviews disagreed about correctness; adjudication below.

**Q1 chunk-static eval, Q2 reset semantics, Q3 multi-order formulation:
spec-compliant, no code change.**
- Q1: `train_gpt.py:2563-2566` computes `_logit_bias` once per chunk,
  reused at `:2568, :2598`. `LEGALITY_SPEC.md:104-120` explicitly accepts
  this as a deliberate legal approximation.
- Q2: No per-batch reset; `train_gpt.py:2521-2522` comment and reset at
  `:2689-2690` match `LEGALITY_SPEC.md:72-79` (phase-global state, SGD
  boundary only).
- Q3: `train_gpt.py:35` guard handles single-order and multi-order
  identically; `CORRECTOR_ORDERS="8"` runs the loop once cleanly.

**Q4 compile warmup: real bug, FIXED this session.** Warmup block in
`train_gpt.py` previously read `val_data.val_tokens` and ran
forward+backward+step on those tokens before the official eval timer.
Patched to use a device-local `torch.Generator(device=device).manual_seed(0)`
producing synthetic `torch.randint` tokens; global torch RNG state is
untouched. Marker block `# BEGIN warmup synthetic tokens` …
`# END warmup synthetic tokens` now brackets the region, and
`tests/test_corrector.py::TestWarmupLegality::test_warmup_does_not_reference_val_tokens`
pins the markers in place (9/9 tests pass).

**Measurements:**

| key                     | value                   |
|-------------------------|-------------------------|
| `head_wrapper`          | 28,616 bytes            |
| `pre_fix_wrapper`       | 29,451 bytes (Δ HEAD +835) |
| `post_fix_wrapper`      | 29,439 bytes (Δ HEAD +823) |
| warmup incremental      | −12 bytes (net reduction) |
| seed 0 headroom_after   | 2,480 bytes             |
| seed 1 headroom_after   | 3,192 bytes             |
| seed 2 headroom_after   | 10,372 bytes            |
| CPU bench projection    | 26.1 s (GATE PASS, < 50 s) |

The warmup fix is a net 12-byte compressed-code-size reduction vs the
pre-fix working tree, because eliminating `val_tokens_idx`, `ds0`,
`col_w`, and `idx_w` offsets the generator setup. Per-seed headroom
increased, not consumed. `(pre_fix_wrapper − head_wrapper) = +835`
exactly matches prior Codex measurement (deviation 0, within ±200 band).

**Deferred** (non-blocking, flagged for downstream sessions):
- Corrector hot-path optimization (scatter_add into logits, cached
  `orders` tensor, batch-level bias) — CPU bench currently PASSES; only
  needed if GPU eval shows real wall-clock pressure.
- Committing the currently-untracked LEGALITY_SPEC.md,
  tests/test_corrector.py, scripts/bench_corrector_cpu.py,
  DEPENDENCY_GATE.md, and requirements.txt.

**Refs**: `LEGALITY_SPEC.md:104-120, 72-79`;
`train_gpt.py:35, 2521-2522, 2689-2690, warmup marker block`.

## Session 3 pipeline launch state — 2026-04-17

RunPod pipeline for PR `#1610` + posterior corrector is now committed and
pushed at:

- branch: `submission/pr1610-corrector`
- launch SHA: `218b623f8962a301e41180b6050186a3c189d063`
- required warmup-fix ancestor: `a33191f572430566b88c4d61badb0369e1e6f9a3`

Targeted re-audit against `origin/submission/pr1610-corrector` confirmed:

- local HEAD == remote HEAD at `218b623f8962a301e41180b6050186a3c189d063`
- `scripts/runpod_pipeline/` is tracked and contains 11 files
- `02_gate_a.sh` now persists the seed-0 checkpoint before any log parsing
- `03_ablations.sh` / `04_decide_and_proceed.sh` implement the three-way fork:
  - `best_delta >= 0.002` → primary / Gate B
  - `0.001 <= best_delta < 0.002` → hold / human decision
  - `all deltas < 0.001` → fallback
- `04b_fallback_level1a.sh` now fails closed on missing BPB or missing variant results
- Stage 5 upload targets are `hf:` and `rsync:` only; stale `s3://` docs removed

Session 3 operator entry point:

- start: `bash scripts/runpod_pipeline/run_all.sh`
- review fork: `bash scripts/runpod_pipeline/04_decide_and_proceed.sh`
- preserve artifacts before teardown:
  - `UPLOAD_TARGET="hf:<repo>:<path>" bash scripts/runpod_pipeline/05_preserve_artifacts.sh`
  - `UPLOAD_TARGET="rsync:<user@host>:<path>" bash scripts/runpod_pipeline/05_preserve_artifacts.sh`
