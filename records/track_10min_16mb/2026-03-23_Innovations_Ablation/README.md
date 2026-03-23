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

## Local Artifacts

```
checkpoints/pod_runs/
├── no_ttt_run1.txt                             # Run 1: no_ttt, 1.1556 BPB
├── final_model_neural_cache_run2.pt            # Run 2: neural cache, 5.3528 BPB
├── final_model_neural_cache_run2.int8.ptz
├── neural_cache_run2.txt
├── no_ttt_run3/                                # Run 3: no_ttt, 1.1496 BPB
│   ├── final_model_no_ttt_20260323_142955.pt
│   ├── final_model_no_ttt_20260323_142955.int8.ptz
│   └── no_ttt_run3.txt
└── neural_cache_run4/                          # Run 4: neural cache v2, 5.7259 BPB
    ├── final_model_neural_cache_20260323_145307.pt
    ├── final_model_neural_cache_20260323_145307.int8.ptz
    └── neural_cache_run4.txt
```

## Commit History

- `f4fae72` Neural cache: add no_pos_offset option, clamp max_len to 2048
- `ff8baad` Remove git fetch/checkout/reset from all run scripts
- `edb9002` Fix run_tag: use args.run_tag instead of bare variable
- `57270e2` Timestamp checkpoint filenames to prevent overwrite between runs
- `00ea5f6` Add three novel innovations for ablation testing (freeze/hyper/ensemble)
