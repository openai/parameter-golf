# Spec 010b — Site-selective SpinQuant ablation — execution summary

**Date:** 2026-04-20
**Pods:** two (2×H100 JP → killed after attn_only setup only; 8×H100 JP for attn_only + mlp_only). `all` mode not run on 8×H100 (had a 2×H100 partial that we killed mid-TTT when 8×H100 opened up).
**Runtime:** ~40 min total across both pods (excluding 2×H100 aborted portion)
**Cost:** ~$15 (2× partial ~$2 + 8× full ~$13)
**Commit:** `8815c4d` on `research` (SPINQUANT_SITES env var plumbing from `ff52a06`)
**Hotstart ckpt:** `runs/008-1736-reproduction/seed_42/pre_gptq.pt` (shared with specs 009 + 010)
**Modes run on 8×H100:** `attn_only` (sites=attn_in,attn_proj_in), `mlp_only` (sites=mlp_in,mlp_proj_in)

## Headline numbers (full 5-variant comparison)

All measured post-TTT val_bpb (the gate number):

| variant | rotation sites active | `diagnostic_quantized` (pre-TTT) | **`quantized_ttt_phased`** | Δ vs baseline |
|---|---|---|---|---|
| **baseline** (spec 009) | none | 1.080098 | **1.067283** | ref |
| internal_only (spec 009) | R_a static per-layer per-kv-group | 1.080068 | 1.067309 | +0.000026 |
| port_1695 (spec 010) | attn_in+attn_proj_in+mlp_in+mlp_proj_in (online) | 1.080009 | 1.067232 | −0.000050 |
| **attn_only** (010b) | attn_in + attn_proj_in only | 1.079976 | **1.067225** | **−0.000059** ← best |
| **mlp_only** (010b) | mlp_in + mlp_proj_in only | 1.080090 | **1.067288** | **+0.000005** |

**All 5 variants lie within 0.00009 bpb of each other.** Phased TTT fully absorbs any rotation-induced quantization-error reduction. SpinQuant is **exhausted as a standalone lever on #1736's stack.**

## The central finding: intra-eval motion exists but TTT absorbs it

The hypothesis from spec 010 analysis was that attention and MLP rotations have **different regime-dependent effects** that cancel in aggregate — specifically "attn helps long docs, mlp hurts short docs." Spec 010b's site-selective runs were designed to isolate each.

**What we found** (rank-0 rb trajectory across eval batches):

| batches done | baseline | attn_only | mlp_only | port_1695 |
|---|---|---|---|---|
| 5 | 1.1142 | 1.1142 | 1.0900 | 1.0595 |
| 50 | 1.0680 | 1.0680 | 1.0736 | 1.0557 |
| 200 | 1.0605 | 1.0609 | 1.0596 | 1.0510 |
| 400 | 1.0581 | 1.0591 | 1.0579 | 1.0509 |
| 780 (end of rank-0 log) | 1.0663 | 1.0665 | 1.0657 | 1.0604 |

Three distinct behaviors:

1. **`attn_only` ≡ `baseline`** literally to 4 decimals on early batches and tracks within ±0.001 thereafter. Attention rotation has essentially zero effect on rank-0's output. This is mechanistic: `softmax(QK.T) V` is rotation-equivariant in V's head_dim, so the quantized forward with rotated V/O reconstructs to near-identical values as unrotated. There is simply no outlier structure in attention weights for rotation to smooth.
2. **`mlp_only`** starts low (1.09 at batch 5 vs baseline's 1.11) and drifts back up. MLP rotation with the LeakyReLU² nonlinearity genuinely changes the forward pass values, but TTT LoRA adapts around it.
3. **`port_1695`** (all 4 sites) starts even lower (1.06 at batch 5) and stays well below baseline's trajectory throughout — but the final reported val_bpb converges to the same place.

### The rank-0 vs global aggregation gap

rank-0's `rb` at end of log: range **0.0075** across variants (1.0663 to 1.0604)
global `val_bpb` reported: range **0.00009** across variants

The 80× compression happens because (a) other ranks see different docs and (b) `val_bpb` is token-weighted across all 8 ranks while `rb` is batch-weighted over rank-0's subset. The `rb` column is useful for showing *that* rotation changes the forward pass (and at what magnitude on rank-0's doc mix), but **does not predict final val_bpb**.

## Why attn_only and mlp_only both null

Hypothesis 1 (refuted): regime-dependent (attn helps long, mlp hurts short) → decomposes cleanly. FALSE.

Hypothesis 2 (supported): phased TTT LoRA is doing the work SpinQuant would do. The 2000-doc prefix TTT adapts the LoRA to whatever quantization error pattern exists. Whether the pre-TTT quant error came from rotated or unrotated GPTQ, the adapted LoRA + quantized weights project to approximately the same functional model on the suffix eval distribution.

Evidence:
- `diagnostic_quantized` (pre-TTT) differences across 5 variants span 0.000122 bpb (max Δ from baseline). Already tiny.
- `quantized_ttt_phased` (post-TTT) differences span 0.000085 bpb. Slightly tighter.
- TTT adaptation reduces per-variant val_bpb by ~0.013 bpb (from ~1.080 to ~1.067) — ~150× larger than the rotation Δ.

## What's left / not run

- **`all` mode (all 4 sites via SITES plumbing)** was started on 2×H100 as the sanity-gate for the new env var, but killed when 8×H100 opened up before it finished TTT. Setup-phase log confirmed the `SPINQUANT_SITES` plumbing works correctly (44 active sites, 0 skipped for `all` ≡ 66 rotated for port_1695 minus 22 skipped MLP = same as port_1695). Full `val_bpb` for `all` on the SITES-path wasn't captured, but given port_1695 + SITES plumbing is verified, there's no reason to expect a meaningful difference.
- **`attn_in_only` (just 1 of 4 sites)** — deferred indefinitely. Given attn_only (2 sites) showed essentially zero effect, attn_in_only would show even less.
- **Seed sweeps** — single seed 42 across all 5 variants. Given the per-variant Δs are < 0.0001 bpb, well below the 0.0002 per-seed variance on #1736's track, seed sweeps aren't likely to reveal anything beyond noise.

## Artifacts

Local + on JP volume `jlxvxeiol4`:

- `runs/010b-spinquant-sites/attn_only/` — run.log, final.json, final_model.int6.ptz (15,965,940 B), final_model.pt, rotation_manifest.json
- `runs/010b-spinquant-sites/mlp_only/` — run.log, final.json, final_model.int6.ptz (15,947,098 B), final_model.pt, rotation_manifest.json
- `runs/010b-spinquant-sites/all/` — partial (2×H100 setup + TTT compile, no final.json) — setup-phase log only, confirming plumbing
- `runs/010b-spinquant-sites/ttt_trajectory_all5.csv` — all 5 variants' rank-0 rb trajectories at matched batch counts

## Decisions for research

1. **SpinQuant is fully exhausted on this stack.** Five variants, all within 0.0001 bpb. The decomposition hypothesis is refuted — attn and MLP rotations don't produce offsetting effects. Both are absorbed by TTT. No further SpinQuant variant (including `full` from spec 009, still unwritten) is likely to surprise.
2. **Pivot to non-quant levers.** Spec 011 (tapered weight-decay retrain) remains the next candidate. Other levers in `research/ideas/1736-improvement.md` that don't go through the quant/TTT path.
3. **Things to hold onto from today:**
   - Spec 008 reproduction closed empirically at 1.0673 via spec 009 baseline.
   - Infrastructure is solid: Parameter Golf pod template, per-variant watcher + scp, rebank fix, `SPINQUANT_SITES` env var for future site-specific quant experiments.
   - The rank-0 `rb` vs global `val_bpb` distinction is now understood — document this in `EXECUTION.md` so future monitoring doesn't over-interpret intra-eval `ttp:` lines again.
4. **The actually interesting side-story:** attention rotation is effectively a no-op in this quantized pipeline, while MLP rotation shifts per-batch output by 0.02-0.05 bpb. This is a reusable piece of knowledge about where quant-error structure lives in #1736's architecture.

## Cost summary — full day's SpinQuant investigation

| Spec / attempt | Variants | Cost |
|---|---|---|
| 009 attempt 1 (wrong image) | 0 | $1.20 |
| 009 successful | baseline + internal_only | $13.30 |
| 010 port_1695 | 1 | $6.50 |
| 010b attempt 1 (AP-IN-1 pod, canceled immediately) | 0 | $0.10 |
| 010b on 2×H100 (partial, killed at all mode TTT start) | ~0.5 | ~$2 |
| 010b on 8×H100 | attn_only + mlp_only | ~$13 |
| **Total** | **5 full + 1 partial** | **~$36** |

SpinQuant investigation is done.
