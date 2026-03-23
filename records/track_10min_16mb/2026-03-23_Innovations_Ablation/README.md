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

| Run | Config | BPB | Steps | Step Avg | Key Finding |
|-----|--------|-----|-------|----------|-------------|
| 1 | no_ttt, 8xH100 | 1.1556 | 7346 | 81ms | First baseline (overwritten by run2) |
| 2 | neural_cache, 8xH100 | 5.3528 | 7240 | 82ms | Neural cache broken — OOD RoPE positions |
| 3 | no_ttt, 8xH100 | **1.1496** | 8454 | 71ms | Best score — faster pod, more steps |
| 4 | neural_cache v2, 8xH100 | 5.7259 | ~7200 | ~83ms | Still broken even with no_pos_offset fix |
| 5 | QAT=1 + trigram=0, 8xH100 | **1.1492** | ~8500 | ~70ms | Wash — QAT and no-trigram cancel out |

### Confirmed Findings

1. **Per-layer LR is our unique innovation** — 5 competition PRs adopted it, 4 credited us by name. All use it for TTT; our use during main training is still unique.

2. **524K batch >> 786K batch on our hardware** — at ~70ms/step with 524K we get 8454 steps (4.4B tokens). At ~108ms/step with 786K we'd get ~5500 steps (4.3B tokens). The breakeven is ~80ms/step; PR #505 runs at 48ms so 786K works for them, not for us.

3. **Sigmoid skip gates and decoder 2x LR already existed in our code** — `SIGMOID_SKIP_GATES=1` and `DECODER_LR_MULT=2.0` are defaults. Discovery via competitive intel analysis saved implementation time.

4. **Stale env vars silently poison runs** — `MLP_HIDDEN=1792` leaked from a reverted commit and inflated the QAT run's model to 27.8M params (vs 27.5M). All run scripts now include `MLP_HIDDEN` in their `unset` blocks.

5. **Timestamped checkpoints prevent data loss** — before the fix, run2 overwrote run1's checkpoint. Now all artifacts include `{run_tag}_{timestamp}` in filenames.

### Falsified

1. **Neural cache eval is fundamentally broken** — `forward_logits_cached` path produces garbage (5.3-5.7 BPB vs 1.15 expected). Tested twice with different configs. The model cannot use cached KV it wasn't trained with. Root cause likely in `forward_logits_cached` missing Value Residual and Gated Attention paths. **Shelved.**

2. **MLP_HIDDEN=1792 doesn't fit** — artifact size goes to 17.6MB (cap is 16MB). PR #505 fits with h=1792 because they have tighter quantization or fewer other params. Reverted.

3. **QAT + no-trigram is a wash** — 1.1492 vs 1.1496 baseline. The two changes appear to cancel each other out. Need isolated tests (QAT-only, trigram-only) to determine individual effects, but the combined result suggests neither is a large lever.

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

### Still Running (1xH100 pods)

- **F: Progressive Layer Freezing** — freeze encoder during warmdown for more decoder-focused steps
- **G: Hyper-Connections scalar** — learned mixing of all prior layer outputs
- **H: Hyper-Connections vector** — per-dim mixing weights

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
└── qat_notrigram_run5/                         # Run 5: QAT=1 trigram=0, 1.1492 BPB
    ├── final_model_qat_notrigram_20260323_154446.pt
    ├── final_model_qat_notrigram_20260323_154446.int8.ptz
    └── qat_notrigram_run5.txt
```

## Commit History

- `f4fae72` Neural cache: add no_pos_offset option, clamp max_len to 2048
- `ff8baad` Remove git fetch/checkout/reset from all run scripts
- `edb9002` Fix run_tag: use args.run_tag instead of bare variable
- `57270e2` Timestamp checkpoint filenames to prevent overwrite between runs
- `00ea5f6` Add three novel innovations for ablation testing (freeze/hyper/ensemble)
- `acea2ca` Add MLP_HIDDEN to unset block in all run scripts
- `f15cac0` Fix batch size: 786K→524K to match run3 baseline
