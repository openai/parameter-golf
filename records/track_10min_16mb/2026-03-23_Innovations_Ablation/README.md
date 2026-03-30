# 2026-03-23: Innovation Ablation Tests

## Motivation

Current best non-TTT score: **1.1496** (run3, perlayer-lr-stack).
Target: **1.1181** (PR #505, best non-TTT entry).
Gap: **0.0315 BPB**.

Pre-eval TTT is not allowed. We need training/architecture/eval innovations to close the gap.
Three novel approaches designed from competition intel analysis — none have been tried by any competitor.

## Competitive Intel Summary

| PR | Score | TTT? | Key Differentiator |
|----|-------|------|--------------------|
| #505 | 1.1181 | No | SwiGLU h=1792, sigmoid skip gates, Late QAT |
| #445 | 1.1236 | No | Late Training Replay (100 batch, 2 epoch) |
| #374 | 1.1246 | No | Tight SWA (<0.2), VE128 |
| #486 | 1.1101 | Yes | TrigramHash, Value Residual, GradQuant |
| Ours | 1.1496 | No | Per-layer LR, VE128, TrigramHash, Value Residual |

Key discovery: sigmoid skip gates (`SIGMOID_SKIP_GATES=1`) and decoder 2x LR (`DECODER_LR_MULT=2.0`) already exist in our code as defaults.

## Three Innovations

### F: Progressive Layer Freezing During Warmdown

**Hypothesis:** Freezing encoder blocks (0-4) when warmdown `scale < 0.3` halves the backward pass (~35ms savings/step), yielding ~1700 extra training steps focused on decoder layers — which benefit most from continued training (supported by 2x decoder LR finding).

**Implementation:**
- `PROGRESSIVE_FREEZE=1`, `PROGRESSIVE_FREEZE_THRESHOLD=0.3`
- When scale drops below threshold, `requires_grad_(False)` on encoder blocks
- Muon optimizer patched to skip weight decay on frozen params (critical — without this fix, frozen weights decay toward zero)
- EMA continues tracking frozen params unchanged (safe)

**Expected:** -0.002 to -0.008 BPB

### G/H: Hyper-Connections (arxiv 2409.19606)

**Hypothesis:** Standard residual connections (and U-Net skips) only connect adjacent layers or mirrored encoder↔decoder pairs. Hyper-connections let each layer i attend to ALL prior layer outputs via learned mixing weights. This strictly generalizes both U-Net skips and the resid_mix x0 blending.

**Implementation:**
- `HYPER_CONNECTIONS=1`, `HYPER_CONN_MODE=scalar|vector`
- `hyper_alpha[i]` has shape `(i+2,)` (scalar) or `(i+2, model_dim)` (vector)
- Initialized to replicate standard residual: weight 1.0 on most recent output
- Softmax-normalized weights — always sums to 1
- Disables U-Net skips when active (subsumed)

**Params:** scalar=77, vector=39,424 — negligible either way.

**Expected:** -0.005 to -0.015 BPB

### I: Logit Ensemble from EMA Trajectory

**Hypothesis:** Averaging logits from 2 diverged checkpoints (EMA model + raw training model) is strictly more powerful than weight averaging. Logit ensembles can capture multi-modal predictions that weight averaging destroys.

**Implementation:**
- `LOGIT_ENSEMBLE=1`, `LOGIT_ENSEMBLE_N=2`, `LOGIT_ENSEMBLE_STRIDE=128`
- Saves raw (pre-EMA) checkpoint before applying EMA
- Both checkpoints quantized independently through int6+zstd roundtrip
- New `eval_val_sliding_ensemble()` averages log-probabilities across checkpoints
- Eval-only — zero impact on training, zero impact on artifact size

**Expected:** -0.003 to -0.010 BPB

## Testing Protocol

### Phase 1: TIER2 Quick Tests (1 GPU, 3 min, ~$1 each)

```bash
cd /workspace/parameter-golf
git fetch origin && git reset --hard origin/perlayer-lr-stack

# Baseline
TIER2=1 NGPU=1 bash run_ablation_innovations.sh F 1337  # Progressive Freeze
TIER2=1 NGPU=1 bash run_ablation_innovations.sh G 1337  # Hyper-Conn scalar
TIER2=1 NGPU=1 bash run_ablation_innovations.sh H 1337  # Hyper-Conn vector
```

Note: Logit Ensemble (I) cannot be tested in TIER2 — EMA/SWA are disabled in short runs.

**What to measure:**
- F: step_avg before/after freeze activation, final val_bpb
- G/H: param count in log, training convergence, val_bpb vs baseline

### Phase 2: Full Runs (8 GPU, 10 min, ~$3.60 each)

```bash
bash run_ablation_innovations.sh F 1337  # Progressive Freeze
bash run_ablation_innovations.sh G 1337  # Hyper-Connections scalar (or H if vector won)
bash run_ablation_innovations.sh I 1337  # Logit Ensemble
```

### Phase 3: Combined

Stack the winners from Phase 2 into a single run.

## Prior Neural Cache Results

Neural cache eval has failed twice — the model produces garbage predictions through the cached KV path:

| Run | Config | Post-quant BPB | Notes |
|-----|--------|---------------|-------|
| Run 2 | `max_len=8192, pos_offset=on` | 5.3528 | OOD positions (8192+ RoPE) |
| Run 4 | `max_len=2048, no_pos_offset=1` | 5.7259 | Still broken — deeper issue in `forward_logits_cached` |

Root cause: the `forward_logits_cached` path likely has incompatibilities beyond just position encoding — possibly attention mask handling, or the model simply cannot generalize to cached KV it was never trained with. **Recommend shelving neural cache** and focusing on the three innovations above.

## Session Learnings (2026-03-23)

### Run Results

| Run | Config | Pre-quant | Post-quant | Quant Gap | Steps | Step Avg | Key Finding |
|-----|--------|-----------|------------|-----------|-------|----------|-------------|
| 1 | no_ttt, 8xH100 | — | 1.1556 | — | 7346 | 81ms | First baseline (overwritten by run2) |
| 2 | neural_cache, 8xH100 | — | 5.3528 | — | 7240 | 82ms | Neural cache broken — OOD RoPE |
| 3 | no_ttt, 8xH100 | ~1.1499 | **1.1496** | **-0.0003** | 8454 | 71ms | **Best score** — fast pod |
| 4 | neural_cache v2, 8xH100 | — | 5.7259 | — | ~7200 | ~83ms | Still broken w/ no_pos_offset |
| 5 | QAT=1 trigram=0, 8xH100 | 1.1582 | **1.1492** | **-0.009** | ~8500 | ~70ms | QAT makes quant gap negative |
| 6 | sigmoid gates + dec2x, 8xH100 | **1.1457** | 1.1635 | **+0.018** | 8552 | ~71ms | Best pre-quant but fragile |
| 7 | Prog Freeze (F), 1xH100 | — | 1.5486 | — | 987 | 608ms | Freeze at step 1 — invalid |
| 8 | Hyper-Conn scalar (G), 1xH100 | — | 1.5226 | — | 1029 | 583ms | Too few steps — invalid |
| 9 | LR=0.03 + LeakyReLU + QAT, 8xH100 | 1.1583 | 1.1551 | -0.003 | 7018 | ~85ms | LeakyReLU lost Star-ReLU params |
| 10 | Hyper-Conn vector (H), 1xH100 | — | 2.1524 | — | 684 | 878ms | O(layers²) too slow — invalid |
| 11 | MATRIX_LR=0.03 only, 8xH100 | **1.1520** | 1.1664 | **+0.014** | 7058 | 84ms | Higher LR = quant-fragile weights |

### Confirmed Findings

1. **Per-layer LR is our unique innovation** — 5 competition PRs adopted it, 4 credited us by name. All use it for TTT; our use during main training is still unique.

2. **524K batch >> 786K batch on our hardware** — at ~70ms/step with 524K we get 8454 steps (4.4B tokens). At ~108ms/step with 786K we'd get ~5500 steps (4.3B tokens). The breakeven is ~80ms/step; PR #505 runs at 48ms so 786K works for them, not for us.

3. **Sigmoid skip gates and decoder 2x LR already existed in our code** — `SIGMOID_SKIP_GATES=1` and `DECODER_LR_MULT=2.0` are defaults. Discovery via competitive intel analysis saved implementation time.

4. **Stale env vars silently poison runs** — `MLP_HIDDEN=1792` leaked from a reverted commit and inflated the QAT run's model to 27.8M params (vs 27.5M). All run scripts now include `MLP_HIDDEN` in their `unset` blocks.

5. **Timestamped checkpoints prevent data loss** — before the fix, run2 overwrote run1's checkpoint. Now all artifacts include `{run_tag}_{timestamp}` in filenames.

### Emerging Pattern: Pre-Quant vs Post-Quant Tradeoff

The most important finding of the session. Changes that improve pre-quant quality often
make weights harder to quantize, causing a net regression:

| Change | Pre-quant | Quant gap | Post-quant | Net |
|--------|-----------|-----------|------------|-----|
| Run3 baseline | 1.1499 | -0.0003 | **1.1496** | Reference |
| +sigmoid gates +dec2x (run6) | **1.1457** | +0.018 | 1.1635 | **Worse** |
| +MATRIX_LR=0.03 (run11) | 1.1520 | +0.014 | 1.1664 | **Worse** |
| +QAT (run5) | 1.1582 | **-0.009** | **1.1492** | **Better** |

**Key insight:** Any change that trains "better" weights (larger magnitudes, more expressive)
also makes those weights harder to compress to int6. QAT is the antidote — it teaches the
model to find solutions that are both good AND quantization-friendly.

**Implication:** MATRIX_LR=0.03 and sigmoid gates both need QAT to work. The winning combo
is likely: MATRIX_LR=0.03 + QAT=1 (run5 proved QAT makes quant gap negative).

### Falsified

1. **Neural cache eval is fundamentally broken** — `forward_logits_cached` path produces garbage (5.3-5.7 BPB vs 1.15 expected). Tested twice with different configs. Root cause likely in `forward_logits_cached` missing Value Residual and Gated Attention paths. **Shelved.**

2. **MLP_HIDDEN=1792 doesn't fit** — artifact size goes to 17.6MB (cap is 16MB). **PR #505 also doesn't fit** (confirmed ~20MB by community). Reverted.

3. **LeakyReLU(0.5)^2 hurt us** — removed Star-ReLU's learned scale/bias (33K params). Run9 scored 1.1551 vs baseline 1.1496. The "proven -0.003 BPB" from PR #518 was on a different architecture. **Star-ReLU is better for us.**

4. **MATRIX_LR=0.03 without QAT is worse** — strong pre-quant signal (-0.06 at step 500) but quant gap blows up (+0.014). Needs QAT. **Must pair with QAT=1.**

5. **Progressive Freeze is marginal** — on 8xH100 would only activate for last 1050 steps, saving ~450 steps (5%). Not worth the encoder-decoder mismatch risk. **Shelved.**

6. **Hyper-Connections are too expensive** — O(layers²) overhead from torch.stack. 583ms/step (scalar) and 878ms/step (vector) vs 70ms baseline. Only ~700-1000 steps in 600s. **Shelved.**

7. **QAT + no-trigram is a wash** — 1.1492 vs 1.1496 baseline. Two changes cancel out.

### Infrastructure Improvements

- `download_pod.sh` — one-command SCP download of all artifacts from any pod
- Timestamped checkpoint filenames via `RUN_TAG` env var
- Removed git operations from all run scripts (manual pull before run)
- Fixed `unset` blocks across all scripts to prevent env var leaks

### Competitive Intelligence

- Best non-TTT score: PR #505 at 1.1181 (SwiGLU h=1792, sigmoid gates, Late QAT, full MHA)
- Best TTT score: PR #512 at 0.9512 (LoRA TTT)
- Our gap to non-TTT leader: 0.0315 BPB
- Our per-layer LR technique cited in 4 competition PRs
- Key missing technique vs #505: wider MLP (blocked by 16MB cap) and full MHA (8 KV heads)

### Innovation F: Progressive Layer Freezing — SHELVED

**Result:** 1.5486 BPB (vs 1.1496 baseline) — catastrophic regression.

**Root cause:** On 1xH100, the wallclock-based warmdown formula produced `scale=0.2369` at step 1 (below the 0.3 threshold), so the encoder froze immediately. The model trained for 987 steps with only decoder layers active. Invalid test.

**8xH100 reverse-engineering:** On 8xH100 at 71ms/step, scale would hit 0.3 at step ~7400 out of ~8454. The freeze would only be active for the last ~1050 steps, saving ~32s of backward time = ~450 extra steps (5% more). But those steps have a frozen encoder while the decoder evolves, creating an EMA mismatch.

**Verdict:** The step-time savings are too small when the freeze triggers late enough to be safe. Freezing earlier (threshold=0.5) risks the encoder not being done learning. The mechanism is mathematically marginal for our architecture — at most ±0.002 BPB. **Shelved.**

### Innovation G/H: Hyper-Connections — SHELVED

| Test | BPB | Steps | Step Avg | Issue |
|------|-----|-------|----------|-------|
| G (scalar) | 1.5226 | 1,029 | ~583ms | Only 1029 steps in 600s (1GPU too slow) |
| H (vector) | 2.1524 | ~800? | ~700ms? | Even worse; 52 min eval time |

**Root cause:** The `torch.stack` of all layer outputs per layer creates O(layers²) memory
and compute overhead. On 1GPU at 583ms/step, only ~1000 steps completed — not enough to
converge. Even on 8xH100, the overhead would likely cost 1000-2000 steps, outweighing the
architectural benefit.

The 1GPU tests were also invalid for the same reason as F: 600s wallclock on 1GPU with
grad_accum=8 only yields ~1000 steps regardless of architecture.

**Additionally:** Both G and H ran without U-Net skips (`UNET_SKIPS=0`), which removes the
proven skip connections. Hyper-connections are supposed to subsume skips, but with so few
training steps, they couldn't learn meaningful cross-layer mixing. The softmax-initialized
weights (mostly on the previous layer) don't provide the same inductive bias as explicit
encoder↔decoder skip connections.

**Verdict:** The O(layers²) cost is too high for a competition with a 600s wallclock.
Hyper-connections might work in longer training regimes but not here. **Shelved.**

## Local Artifacts

```
checkpoints/pod_runs/
├── no_ttt_run1.txt                             # Run 1: no_ttt, 1.1556 BPB
├── final_model_neural_cache_run2.pt            # Run 2: neural cache, 5.3528 BPB
├── final_model_neural_cache_run2.int8.ptz
├── neural_cache_run2.txt
├── no_ttt_run3/                                # Run 3: no_ttt, 1.1496 BPB (BASELINE)
│   ├── final_model_no_ttt_20260323_142955.pt
│   ├── final_model_no_ttt_20260323_142955.int8.ptz
│   └── no_ttt_run3.txt
├── neural_cache_run4/                          # Run 4: neural cache v2, 5.7259 BPB
│   ├── final_model_neural_cache_20260323_145307.pt
│   ├── final_model_neural_cache_20260323_145307.int8.ptz
│   └── neural_cache_run4.txt
├── qat_notrigram_run5/                         # Run 5: QAT=1 trigram=0, 1.1492 BPB
│   ├── final_model_qat_notrigram_20260323_154446.pt
│   ├── final_model_qat_notrigram_20260323_154446.int8.ptz
│   └── qat_notrigram_run5.txt
├── no_ttt_run6/                                # Run 6: sigmoid gates + dec2x LR, 1.1635 BPB (REGRESSION)
│   ├── final_model_no_ttt_20260323_161030.pt
│   ├── final_model_no_ttt_20260323_161030.int8.ptz
│   └── no_ttt_run6.txt
├── freeze_F_run7/                              # Run 7: Progressive Freeze, 1.5486 BPB (SHELVED)
│   ├── final_model_freeze_F_20260323_155839.pt
│   ├── final_model_freeze_F_20260323_155839.int8.ptz
│   └── freeze_F_run7.txt
├── hyper_G_run8/                               # Run 8: Hyper-Conn scalar, 1.5226 BPB (SHELVED)
│   └── hyper_G_run8.txt
├── best_run9/                                  # Run 9: LR=0.03+LeakyReLU+QAT, 1.1551 BPB
│   └── best_run9.txt
├── hyper_H_run10/                              # Run 10: Hyper-Conn vector, 2.1524 BPB (SHELVED)
│   └── hyper_H_run10.txt
└── lr03_leaky_qat_run11/                       # Run 11: MATRIX_LR=0.03 only, 1.1664 BPB
    └── lr03_run11.txt
```

## Commit History

- `f4fae72` Neural cache: add no_pos_offset option, clamp max_len to 2048
- `ff8baad` Remove git fetch/checkout/reset from all run scripts
- `edb9002` Fix run_tag: use args.run_tag instead of bare variable
- `57270e2` Timestamp checkpoint filenames to prevent overwrite between runs
- `00ea5f6` Add three novel innovations for ablation testing (freeze/hyper/ensemble)
- `acea2ca` Add MLP_HIDDEN to unset block in all run scripts
- `f15cac0` Fix batch size: 786K→524K to match run3 baseline
