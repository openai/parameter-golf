# Spec 009 — SpinQuant hotstart (baseline + internal_only) — execution summary

**Date:** 2026-04-20
**Pod:** `kp6hrvolde7vav` (8×H100 SXM AP-JP-1, `runpod/parameter-golf:latest` template, $23.92/hr)
**Runtime:** 02:45 → 03:19 UTC (~34 min)
**Cost:** ~$13.50 successful run + $1.20 aborted (wrong image); total **~$14.70**
**Commit:** `6456188` (rebank fix on top of `1572115` research implementation)
**Hotstart ckpt:** `runs/008-1736-reproduction/seed_42/pre_gptq.pt` (EMA-blended FP32, 135.6 MB)

## Headline numbers

| Metric | baseline (no rot) | internal_only (R_a) | Δ | #1736 ref |
|---|---|---|---|---|
| `diagnostic_pre_quant_post_rotation val_bpb` | 1.2216 | 1.2216 | −0.00002 | — |
| `diagnostic_quantized val_bpb` (post-GPTQ, pre-TTT) | 1.0801 | 1.0801 | −0.00003 | 1.07847 |
| **`quantized_ttt_phased val_bpb` (GATE)** | **1.0673** | **1.0673** | **+0.000026** | 1.06610 |
| artifact bytes (< 16,000,000 cap) | 15,948,105 | 15,947,721 | **−384** | 15,978,834 |
| TTT eval time (ms) | 498,902 | 433,735 | −65 s (compile cache) | — |

**Gate result:** **BOTH PASS** the spec 008 band `[1.06310, 1.06910]`. Spec 008's previously-missing post-TTT gate number is now empirically **1.0673** (+0.0012 vs #1736's 1.06610 — within bf16 cross-pod noise; consistent with spec 008's +0.00016 pre-quant delta).

**SpinQuant R_a verdict:** **null result on final val_bpb.** Expected Δ per spec was −0.001 to −0.002; observed Δ = +0.000026 (noise). Rotation *did* reduce weight outliers (artifact 384 B smaller, `diagnostic_quantized` tied to 4 decimals) — below the TTT-adaptation noise floor.

## TTT running-avg trajectory (interpolated to matched batch counts)

| batches done | baseline rb | internal_only rb | Δ (int − base) |
|---|---|---|---|
| 5 | 1.1142 | **1.0900** | **−0.0242** |
| 10 | 1.1009 | 1.0950 | −0.0059 |
| 25 | 1.0771 | 1.0811 | +0.0040 |
| 50 | 1.0680 | 1.0705 | +0.0026 |
| 100 | 1.0625 | 1.0659 | +0.0034 |
| 150 | 1.0613 | 1.0648 | +0.0035 |
| 200 | 1.0605 | 1.0637 | +0.0032 |
| 300 | 1.0594 | 1.0614 | +0.0020 |
| 400 | 1.0581 | 1.0607 | +0.0026 |
| 500 | 1.0601 | 1.0618 | +0.0017 |
| 600 | 1.0623 | 1.0638 | +0.0016 |
| 700 | 1.0645 | 1.0661 | +0.0016 |
| 750 | 1.0657 | 1.0673 | +0.0016 |
| 780 | **1.0663** | **1.0679** | **+0.0016** |

Raw per-line data: `ttt_trajectory.csv` (94 baseline points, 97 internal_only points).

### Shape of the curves

**Both curves:** U-shape. Start high (batch-0 bpb ~1.09–1.12 on un-adapted quantized model), plunge to minimum ~1.058 around batches 300–500 as the LoRA adapts, drift back up to final ~1.066–1.068 as cumulative average absorbs "harder" late-eval batches.

**internal_only vs baseline:**
- **Batches 1–15:** internal_only *below* baseline (rotation reduces raw quant error; LoRA hasn't adapted yet).
- **Batches 20–780:** internal_only consistently **+0.0016 to +0.0035 above baseline**, converging tightly. Crossover at ~batch 20.
- The late-eval Δ is *remarkably stable* at ~+0.0016, suggesting not noise but a real (small) regression from R_a once TTT is adapted.

**Phase markers** (`gd` column, prefix-SGD phase counter): `gd: 0→1` flip visible around batch 48 in baseline, ~batch 50 in internal_only. Boundaries `[666, 1333, 2000]` are inside the 2000-doc prefix; the 782 `ttp:` batches are the 48k-doc suffix eval.

## Why internal_only was null / slightly worse

Three hypotheses, ranked by prior likelihood:

1. **TTT LoRA substitutes for rotation.** Phased TTT already learns the structure of the quant error (that's its entire purpose). R_a's contribution — spreading V-output / O-input outliers — is a subset of what the LoRA picks up from calibration data. With TTT in the stack, the "rotation headroom" is near-zero.
2. **R_a alone is too small a lever.** SpinQuant's published gains assume full {R₀, R_a, R_m} — residual-stream + internal-attn + internal-MLP. R_a is one of three; expecting a third of the total is already optimistic, but when compounded with (1) the expected Δ collapses below noise.
3. **Minor regression from rotation + bf16 noise.** The late-eval +0.0016 stable gap is smaller than seed-to-seed variance in #1736 (~0.001–0.002 at a train pass), so probably not meaningful, but the *consistency* of the gap (not drifting) weakly suggests a real effect rather than noise.

## Artifacts (all local + on JP volume)

Per mode (`baseline/` and `internal_only/` each):
- `run.log` (27 KB) — full training log including all `ttp:` batch lines
- `final_model.int6.ptz` (≈ 15.95 MB) — quantized + brotli submission artifact
- `final_model.pt` (135.6 MB) — pre-GPTQ fp32 (rotated for internal_only)
- `rotation_manifest.json` — rotation seeds per (layer, kv-group); empty for baseline
- `final.json` — machine-readable metrics

Plus top-level:
- `ttt_trajectory.csv` — both curves in CSV for plotting
- `summary.md` — this file

## Decisions for research

1. **Spec 008 reproduction confirmed.** Post-TTT gate 1.0673 is within ±0.003 of #1736's 1.06610. Spec 008 evaluation can be written / experiments.md row appended.
2. **`internal_only` SpinQuant variant scores null.** Not worth pursuing a 3-seed confirmation. If we want a standalone-record claim from SpinQuant, the `full` variant (+R₀ residual-stream rotation, per-channel folds, resid_mix handling) is still unmeasured.
3. **`full` variant — decision point.** Script currently `NotImplementedError`s for `full`. Writing it is ~2–3 hours of research work (R₀ + fold logic + resid_mix freeze-to-mean) plus a CPU FP-invariance pre-test. Given R_a alone landed at 0 on top of TTT, the prior on full SpinQuant clearing the 0.005-bpb standalone-record threshold is weakened — but not eliminated, because R₀ is the largest rotation (residual stream sees every layer).
4. **`port_1695` variant** — separate research question; needs `gh pr diff 1695` read, unaffected by this spec's null.
5. **Alternative pivot:** spec 010 (tapered WD retrain) doesn't depend on SpinQuant succeeding.

## Things that went right

- Pre-flight CPU invariance test (`test_rotation_invariance.py`) passed on real ckpt before any pod spend — good investment.
- Per-variant scp on completion (watcher) means both sets of artifacts landed locally the instant each variant finished — no bulk-transfer at the end.
- Baseline and internal_only ran back-to-back on one pod with compile-cache reuse (internal_only TTT compile: 93.9 s vs baseline's 149.3 s).

## Things that cost money / attention

- **Wrong pod image attempt 1 (+$1.20):** used generic `runpod/pytorch:2.4.0-cu124` template; flash_attn_3 wheel requires torch 2.9.1+cu128. Now mandatorily fixed — always use `--template-id y5cejece4j` (Parameter Golf template).
- **Path typo in launch script (DATA_DIR=./data instead of /workspace/data):** caught on first torchrun attempt of the right pod, cost ~$0.30.
- **Stale `pre_gptq.pt` format (unbanked) vs script's load path:** caught pre-launch, fixed with a one-line `_rebank_state_dict` call before `load_state_dict`. Would otherwise have silently loaded random-init weights and burned the whole sweep — zero pod dollars lost thanks to the fix going in before the first successful launch.
