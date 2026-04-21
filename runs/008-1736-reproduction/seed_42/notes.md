# Spec 008 seed_42 — execution notes

**Run dir:** `runs/008-1736-reproduction/seed_42/` (local + JP volume `jlxvxeiol4:/workspace/runs/008-1736-reproduction/seed_42/`)
**Commit:** `154c9b8` on `research` + in-place patch for `SAVE_PRE_GPTQ` in `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py`
**Date:** 2026-04-20
**Final pod:** `xy1bfwkcfds0ax` (8×H100 SXM AP-JP-1, $23.92/hr) — stopped after artifacts pulled

## Status

**Primary deliverable — MET.** `pre_gptq.pt` (135,595,881 bytes) saved on JP volume AND backed up locally. This is the hotstart input for all spec 009+ quant experiments.

**Submission artifact — produced.** `final_model.int6.ptz` (15,946,577 bytes, **under 16 MB cap**).

**Accept-gate number (post-TTT val_bpb) — NOT CAPTURED.** Training completed cleanly and all post-training stages ran through `Total submission size`, but I stopped the pod before the post-quant diagnostic eval and phased-TTT eval ran. Watcher trigger fired on the wrong log marker.

## Results table (what we have)

| stage | our value | #1736 seed 42 | Δ |
|---|---|---|---|
| stopping_early at step | 4828 | 4854 | −26 (3% slower due to hardware variance) |
| train_time wallclock | 596.14 s | 596.18 s | ≈ equal |
| step-4828 val_bpb (bare) | 1.0697 | 1.0696 (step 4854) | ≈ equal |
| diagnostic pre-quant post-EMA val_bpb | **1.06922** | **1.06906** | **+0.00016** |
| artifact size (quant+brotli) | 15,946,577 | 15,978,834 | −32,257 (both under 16 MB) |
| diagnostic quantized val_bpb | **1.08010** *(via 009 baseline)* | 1.07847 | +0.00163 |
| **quantized_ttt_phased val_bpb (GATE)** | **1.06728** *(via 009 baseline)* | **1.06610** | **+0.00118** |

## Implication for the accept gate

Gate: post-TTT val_bpb within ±0.003 of 1.06610 → range [1.06310, 1.06910].

Our pre-quant matches #1736 within 0.00016, so if TTT delivers the same gain:
- #1736 TTT gain = 1.06906 − 1.06610 = **−0.01237**? No — actually 1.07847 → 1.06610 = −0.01237 (diff from post-quant, not pre-quant).
- Better framing: *quantization cost* was +0.00941 (1.06906 → 1.07847), *TTT recovery* was −0.01237 (1.07847 → 1.06610), *net vs pre-quant* was −0.00296.
- Projected post-TTT for us: 1.06922 − 0.00296 ≈ **1.06626**
- Well within gate (1.06310, 1.06910). Projected pass.

Caveat: quantization cost and TTT gain are both seed/weight-specific; our numbers could drift ±0.001-0.002 from the projection. Still overwhelmingly likely we're inside the gate.

## Deliverables locally

| file | size | notes |
|---|---|---|
| `pre_gptq.pt` | 135.60 MB | EMA-blended FP32 weights, pre-quantization. Hotstart input for specs 009+. |
| `final_model.int6.ptz` | 15.95 MB | Compressed submission artifact (INT6 GPTQ + brotli). |
| `train.log` | 6.5 KB | Training log through `Total submission size`. |

## Deliverables on JP volume (`jlxvxeiol4`)

Same three files in `/workspace/runs/008-1736-reproduction/seed_42/`, plus the crash logs from three earlier attempts:
- `train.crashed_brotli.log` — attempt 1 (brotli missing on pod; pre-quant val_bpb 1.06975 here too)
- `train.oom1.log` — attempt 2 (stale CUDA contexts OOM'd init)
- `train_attempt1_brotli_missing.log` — duplicate of attempt 1

## Post-TTT — measured in spec 009 baseline run

**UPDATE:** The missing post-TTT number was measured in spec 009. The 009 baseline run loaded `pre_gptq.pt` from this run and executed the full GPTQ + phased-TTT pipeline:

| stage | value |
|---|---|
| diagnostic quantized val_bpb | 1.08010 |
| **quantized_ttt_phased val_bpb** | **1.06728** |

Source: `runs/009-spinquant-hotstart/baseline/final.json`

008's actual post-TTT = **1.06728** — misses #1736 (1.06610) by 0.00118. Consistent with 017 (1.06733) and 019 (1.06744); all three cluster within 0.00016 of each other.

## Next action — how to get the missing post-TTT gate number

Three options, ranked:

### (a) **Eval-only rerun on saved artifact** — recommended, ~$3
Write a small script (~50 lines) that imports train_gpt.py's functions and runs **only** the post-serialize path:
1. Load `final_model.int6.ptz` via `deserialize(h, device)`
2. Run `eval_val()` → "diagnostic quantized val_bpb"
3. Run the `if h.ttt_enabled:` block → "quantized_ttt_phased val_bpb"

train_gpt.py lines 2978-end is the template. No retraining needed; the quantized .ptz already encodes the post-training state.

Cost: 5-8 min on 8×H100 = ~$3. Risk: getting the DDP-aware ttt wrapper exactly right. Reusable for spec 009+ evals.

### (b) **Full Phase 3 rerun with correct completion trigger** — ~$10
Launch the exact same command as today's attempt 4, but change watcher grep to `quantized_ttt_phased val_loss:` before firing stop.

Wasted training cost but simplest. Gives a second independent data point on reproducibility (minor bonus).

### (c) **Ship as-is** — $0
Report pre-quant 1.06922 (matches #1736 within 0.00016) + projected post-TTT ~1.0663 + projection-passes-gate. No empirical post-TTT measurement. 

Defensible given the within-noise pre-quant match, but not fully spec-compliant.

## Lessons to carry forward (saved to memory)

1. `feedback_preflight_deps_and_gpu_clean.md` — install brotli + all import-audit deps upfront; verify GPU clean before relaunch.
2. `feedback_never_delete_checkpoints.md` — never `rm .pt/.ptz/.ckpt`; only rename logs.
3. **New, to add:** completion watcher patterns should grep for the *last* meaningful log line in a pipeline (here: `quantized_ttt_phased val_loss:`), not the first post-training serialization marker.

## Cost accounting for spec 008 today

| item | cost |
|---|---|
| smoke test (1×H100 JP, ~8 min) | $0.40 |
| prep pod (NA RTX PRO 6000, aborted) | $0.10 |
| 8×H100 attempt 1 (brotli crash) | $4.00 |
| 8×H100 attempt 2 (OOM) | $0.50 |
| 8×H100 attempt 3 (silent fail) | $0.20 |
| 8×H100 attempt 4 (successful up to line 176) | $5.50 |
| HF CaseOps data download (on 8×H100, inefficient — should've used 1×H100) | $3.50 |
| **Total spec 008 Phase 3 spend** | **~$14** |

Plus prior spec 008 Phase 0/1 (prior audit + HF shortcut check): ~$2.

## Handback

Primary deliverable `pre_gptq.pt` is secure. Execution recommends option (a) — eval-only rerun (~$3) — to close the loop on the post-TTT gate number. Research's call.
