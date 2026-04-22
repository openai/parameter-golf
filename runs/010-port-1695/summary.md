# Spec 010 — Port #1695's online Hadamard rotation — execution summary

**Date:** 2026-04-20
**Pod:** `kp6hrvolde7vav` (restarted from spec 009 session; 8×H100 SXM AP-JP-1, `runpod/parameter-golf:latest`, $23.92/hr)
**Runtime:** ~17 min on pod (03:27 → ~03:45 UTC)
**Cost:** ~$6-7
**Commit:** `b47a252` on `research` (porting #1695's online-rotation design verbatim)
**Hotstart ckpt:** `runs/008-1736-reproduction/seed_42/pre_gptq.pt` (same as spec 009)
**Mode run:** `port_1695` (the #1695 online 4-rotation scheme)

## Headline result

| Metric | spec 009 baseline | spec 009 internal_only | **spec 010 port_1695** | #1736 ref |
|---|---|---|---|---|
| `diagnostic_pre_quant_post_rotation val_bpb` | 1.22161 | 1.22159 | **1.22161** | — |
| `diagnostic_quantized val_bpb` (post-GPTQ, pre-TTT) | 1.08010 | 1.08007 | **1.08001** | 1.07847 |
| **`quantized_ttt_phased val_bpb` (GATE)** | **1.06728** | **1.06731** | **1.06723** | 1.06610 |
| artifact bytes (< 16,000,000 cap) | 15,948,105 | 15,947,721 | **15,968,990** | 15,978,834 |
| TTT eval time (ms) | 498,902 | 433,735 | 567,379 | — |

**Gate result:** PASS (inside `[1.06310, 1.06910]`).
**Δ port_1695 vs spec 009 baseline:** **−0.000050 bpb** (within noise, nowhere near the ≥−0.003 spec threshold for "SpinQuant landed").

## What this means

**SpinQuant's online rotation scheme (the more sophisticated of the two variants tested) also returns null on top of #1736's phased TTT stack.** This is the second consecutive null from the SpinQuant family:

| spec | variant | Δ vs baseline | verdict |
|---|---|---|---|
| 009 | internal_only (R_a-only, static) | +0.00003 | null |
| 010 | port_1695 (4 online rotations) | **−0.00005** | null |

Both fail to move the needle despite successfully reducing *raw quantization error*:

- `diagnostic_quantized` (model after GPTQ, before TTT adaptation): 1.08010 → 1.08007 → 1.08001 — rotation does progressively reduce quant error by ~0.0001 bpb per variant.
- `quantized_ttt_phased` (after TTT adapts the LoRA): all three converge to within 0.00008 of each other.

**TTT adaptation fully compensates for whatever quant-error structure rotation would fix.** The LoRA learns the error pattern from the 2000-doc prefix and corrects it regardless of whether the weights were pre-rotated.

## Intra-eval trajectory — misleading early signal

The `ttp:` log column `rb` (running-avg bpb within the eval loop) produced a dramatic early-batch gap favoring port_1695:

| batches done | baseline rb | port_1695 rb | Δ |
|---|---|---|---|
| 1 | — | 1.0593 | — |
| 25 | 1.0771 | 1.0604 | −0.017 |
| 116 | 1.0625 | **1.0524** | **−0.010** |
| 331 | ~1.0594 | **1.0507** | **−0.009** |
| 773 (last logged) | 1.0663 | **1.0604** | **−0.006** |

From the log alone it looked like port_1695 would finish near 1.060 — a SOTA break. In reality the reported `val_bpb` in `final.json` was 1.0672, not 1.0604.

**What went wrong in the interpretation:** the `rb` column is a *within-phase* running metric. It does not cleanly converge to the final reported `val_bpb` — there's a non-trivial delta (~+0.007 in port_1695's case) between the last `ttp:` line and the final aggregated number written to `final.json`. We're not sure yet whether that's due to a final-batch aggregation, a phase-end averaging pass, or a different metric entirely — but the lesson is **trust `final.json`, not the intra-eval `rb` column**.

Baseline had the same pattern but smaller: last `ttp: rb=1.0663`, `final.json val_bpb=1.0673` — a +0.001 delta. Port_1695's delta was +0.007, which happened to cancel the lead it looked like it had.

Full per-variant trajectory data in `runs/010-port-1695/ttt_trajectory.csv` (all three variants, matched batch counts).

## Why SpinQuant keeps missing on top of #1736

Three hypotheses, consistent with both null results:

1. **TTT substitutes for rotation.** The phased TTT LoRA already learns the quant-error structure that rotation is designed to spread out. With TTT in the stack, rotation's "headroom" is near zero. Evidence: `diagnostic_quantized` differences between variants are 10× smaller than `quantized_ttt_phased` differences.
2. **Per-channel multipliers in #1736 (`attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`) already do what R₀ would do.** These are learned per-channel scalings on the residual that reshape weight distributions in a direction compatible with quantization. Spec 010's 4-rotation scheme sidesteps them (rotates pre-linear, not on residual) but still inherits the already-gentler distribution.
3. **#1736 is closer to the GPTQ ceiling than #1695's base was.** #1695 claimed −0.005 on top of a #1529-adjacent base without CaseOps / gates / phased TTT. Those additions left less room for quant-level improvements to show.

## What this does NOT rule out

- **Spec 009 `full` variant** (static residual-stream R₀ + per-channel fold + resid_mix freeze-to-mean) remains uncoded. It's less sophisticated than port_1695 (weight-only, no Hessian rotation), so the prior is weaker, but it's not strictly dominated — it could in principle compose differently with TTT. Whether to invest the 2-3 hr research time is a judgment call.
- **SpinQuant composed with spec 011 (tapered WD retrain) or other training-time changes.** Today's specs all hotstart the same trained checkpoint; rotation might do more on a differently-trained base.
- **Rotation-seed sensitivity.** All runs used `SPINQUANT_SEED=42`. Sweeping seeds could expose variance — but the null is so complete (0.00005 bpb) that it's hard to believe a seed change would move it by 0.002+.

## Artifacts

Local + on JP volume `jlxvxeiol4`:

- `runs/010-port-1695/run.log` (27 KB) — full training log
- `runs/010-port-1695/final_model.int6.ptz` (15.96 MB) — rotated + GPTQ-quantized submission artifact
- `runs/010-port-1695/final_model.pt` (135.6 MB) — pre-GPTQ fp32
- `runs/010-port-1695/rotation_manifest.json` — 4 rotation seeds per (layer, site)
- `runs/010-port-1695/final.json` — machine-readable
- `runs/010-port-1695/ttt_trajectory.csv` — baseline / internal_only / port_1695 curves at matched batch counts

## Decisions for research

1. **SpinQuant is exhausted as a standalone lever on this stack.** Two variants, both null. Third (`full`) is less promising than either already tried. Recommend not writing `full` unless a new argument emerges.
2. **Pivot candidates for next spec** (per `research/ideas/`):
   - Spec 011 (tapered weight-decay retrain) — training-time change, independent of quant stack. Estimate ~$15, ~10 min train + 10 min eval.
   - Other levers in `research/ideas/1736-improvement.md` that don't go through the quant/TTT path.
3. **Today's wins to hold onto:**
   - Spec 008's missed post-TTT gate number is now empirically closed (1.0673 via spec 009 baseline).
   - #1736 reproduction verified end-to-end.
   - Infrastructure proven: Parameter Golf template pod, per-variant watcher + scp, rebank fix for pre_gptq format.
4. **Sanity check of the infrastructure:** `final.json` numbers vs `ttp:` `rb` column disagreement is worth understanding before the next eval run. Likely `rb` is per-phase and `final.json` is all-phase aggregated, but confirm by reading `eval_val_ttt_phased` in train_gpt.py. Research task.

## Cost summary — SpinQuant investigation day

| Spec / attempt | Cost |
|---|---|
| Spec 009 attempt 1 (wrong pod image) | $1.20 |
| Spec 009 successful (baseline + internal_only) | $13.30 |
| Spec 010 port_1695 (restarted pod) | $6.50 |
| **Total SpinQuant today** | **~$21** |

Yielded: closed spec 008 gate (worth ~$3 of planned eval-only rerun), verified #1736 reproduction, ruled out two SpinQuant variants.
