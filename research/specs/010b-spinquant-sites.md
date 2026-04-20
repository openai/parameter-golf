# Spec 010b — Site-selective SpinQuant ablation

**Slug:** `spinquant-sites`
**Created:** 2026-04-20
**Links to idea:** `research/ideas/spinquant-integration-notes.md` and spec 010 result trajectory analysis.
**Depends on:** spec 008 complete (`pre_gptq.pt`), spec 010 complete (provides the `port_1695` reference number).

## Hypothesis

Spec 010 showed rotation has a **regime-dependent** effect:

- Longest docs (dl > 1000 tokens): port_1695 is **−0.006 bpb** vs baseline.
- Shortest docs (dl < 300 tokens): port_1695 is **+0.015 bpb** vs baseline.
- Aggregate: ~0, because the eval distribution balances out.

Our hypothesis: **the "helpful on long" and "hurtful on short" effects come from different rotation sites.** Specifically:

- **Attention rotations** (R_attn_in, R_attn_proj_in) aggregate across tokens → should be the primary source of long-context help.
- **MLP rotations** (R_mlp_in, R_mlp_proj_in) are per-token feature transforms → should be the primary source of short-context hurt.

If this decomposition is correct, **running attention-only rotation should land a net-positive aggregate** by keeping the benefit and dropping the penalty.

## Baseline

Spec 009's `baseline` mode (our measured #1736 reproduction): `val_bpb = 1.06728`.

## Modes to sweep

Four env-var-selectable modes, all `SPINQUANT_ENABLED=1` with different `SPINQUANT_SITES` values:

| Mode | `SPINQUANT_SITES` | Rotations | Expected Δ (aggregate) |
|---|---|---|---|
| `attn_only` | `attn_in,attn_proj_in` | both attn, no MLP | **−0.001 to −0.003** (if hypothesis holds) |
| `mlp_only` | `mlp_in,mlp_proj_in` | both MLP, no attn | **+0.001 to +0.003** (hurt isolated) |
| `all` (sanity) | `attn_in,attn_proj_in,mlp_in,mlp_proj_in` | all 4 (= port_1695) | ~0 (should reproduce spec 010's +0.00005) |
| `attn_in_only` | `attn_in` | just residual→QKV | unknown; finer-grained attn decomposition |

The `all` mode is a sanity check that the new `SPINQUANT_SITES` plumbing doesn't accidentally change the behavior of the already-measured port_1695 mode. If `all`'s number doesn't match spec 010's 1.06723 within noise, the site-selection code is buggy.

## Accept criteria

- **Sanity:** `all` mode within ±0.0005 of spec 010's 1.06723 (confirms the env-var plumbing doesn't alter existing behavior).
- **Primary:** `attn_only` Δ ≤ −0.001 vs baseline → confirms hypothesis, attention rotation is the net-helpful subset. A clean signal ≥ −0.002 is a standalone win.
- **Secondary diagnostic:** `mlp_only` Δ ≥ +0.001 vs baseline → confirms MLP rotation is the hurt source. Informative even if unflattering.
- **Null case:** if `attn_only` and `mlp_only` are both within ±0.0005 of baseline, the regime-dependence doesn't decompose cleanly by site — probably uniform across sites. SpinQuant truly exhausted.

Intra-eval trajectory also worth logging: even if aggregate is null, per-doc-length-bucket breakdown (same analysis we did on spec 010) will tell us whether attn-only rotation has a FLATTER regime profile than port_1695 (less swing between long-doc help and short-doc hurt).

## Code changes

Minimal — we already have the full SpinQuant infrastructure from spec 010. Just add a `SPINQUANT_SITES` env var that filters which of the 4 tags get installed/rotated.

### `train_gpt.py`

1. **Hyperparameters:** add `spinquant_sites` field. Default `"attn_in,attn_proj_in,mlp_in,mlp_proj_in"` (all) so existing SPINQUANT_ENABLED runs are unaffected. Parse into a frozenset inline or via a helper.
2. **`install_spinquant_rotations`:** before each `register_buffer` call, check if the corresponding tag is in the allowed-sites set. Skip if not.
3. **`_spinquant_rotate_sd_and_H`:** before rotating a given `name`'s weight and Hessian, check if its tag is in the allowed-sites set. Skip if not.
4. The forward-pass hooks in `CausalSelfAttention.forward`, `MLP.forward`, and the TTT mirrors already check `hasattr(self, "_sq_R_...")`. If the buffer isn't installed, the branch doesn't fire. So no changes needed in the hook code.

That's ~15 LOC added across 3 insertion points, no control-flow changes.

### `spinquant_hotstart.py`

No code change. `SPINQUANT_SITES` is read inside `Hyperparameters`, picked up automatically.

## Hardware ladder

8×H100, single seed (42), one pod session running all four modes back-to-back. Same hotstart checkpoint as specs 009/010.

## Execution protocol

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

COMMON_ENV=(
  NCCL_NET=Socket DATA_DIR=./data
  CASEOPS_ENABLED=1
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0
  GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16
  GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
  SPINQUANT_MODE=port_1695
  SPINQUANT_SEED=42
  HOTSTART_FP_CKPT=/workspace/runs/008-1736-reproduction/seed_42/pre_gptq.pt
  SEED=42
)

for SITES in "attn_in,attn_proj_in" "mlp_in,mlp_proj_in" "attn_in,attn_proj_in,mlp_in,mlp_proj_in" "attn_in"; do
  case "$SITES" in
    "attn_in,attn_proj_in")                           name="attn_only" ;;
    "mlp_in,mlp_proj_in")                             name="mlp_only"  ;;
    "attn_in,attn_proj_in,mlp_in,mlp_proj_in")        name="all"       ;;
    "attn_in")                                        name="attn_in_only" ;;
  esac
  mkdir -p /workspace/runs/010b-spinquant-sites/$name
  env "${COMMON_ENV[@]}" \
    ARTIFACT_DIR=/workspace/runs/010b-spinquant-sites/$name \
    SPINQUANT_SITES="$SITES" \
    torchrun --standalone --nproc_per_node=8 spinquant_hotstart.py \
    > /workspace/runs/010b-spinquant-sites/$name/run.log 2>&1
done
```

Gate between runs: if `all` mode's final val_bpb differs from spec 010's 1.06723 by more than ±0.001, halt before running other modes — indicates `SPINQUANT_SITES` plumbing is broken.

## Seed plan

Single seed 42 to stay comparable with everything measured so far.

## Inputs

- Same as spec 010: `runs/008-1736-reproduction/seed_42/pre_gptq.pt` as hotstart.
- `train_gpt.py` on `research` branch at the commit that lands the `SPINQUANT_SITES` patch.

## Stop-early criteria

- `all` mode out of range → halt, debug before running other variants.
- Any mode produces `val_bpb > 1.080` → GPTQ or rotation error propagation, halt and flag.
- NaN / OOM → halt.

## Cost estimate

| Item | Cost |
|---|---|
| Pod spin-up + compile warm-up (reuse cache across variants) | $2 |
| 4 modes × ~10 min each | $20 |
| Buffer | $3 |
| **Total** | **~$25** |

Hotstart reuses spec 008's checkpoint; no training.

## Extra artifacts

Per mode, under `runs/010b-spinquant-sites/<mode>/`:
- `run.log` — full log
- `final_model.int6.ptz` — rotated + quantized artifact
- `final.json` — val_bpb numbers + rotation manifest
- `rotation_manifest.json` — which sites fired

Top-level:
- `summary.md` — side-by-side table of all four modes' (diagnostic_quantized, quantized_ttt_phased) plus per-doc-length-bucket analysis.
- `ttt_trajectory.csv` — all four modes' intra-eval trajectories at matched batch counts.

## Open questions

1. Does the `all` mode exactly match spec 010's 1.06723? If it drifts (even by 0.0001), the new site-filtering code has a subtle bug — check the parsing.
2. For the per-bucket analysis: at what doc length does `attn_only` transition from help to hurt? Compare to port_1695's ~500-token crossover. A more favorable crossover (e.g., 200 tokens) would mean attn-only is almost strictly better than baseline.
3. If `attn_in_only` (just 1 of the 4 rotations) delivers most of the `attn_only` benefit, we've localized the effect to a single rotation site — even tighter signal.
4. If both `attn_only` and `mlp_only` are net null individually but `all` is net null too, the regime-dependence is uniform across sites → nothing to exploit.

## What this spec does NOT do

- Does not retrain. Pure post-training sweep.
- Does not attempt layer-selective rotation (only applying to some layers). Potential follow-up (010c) if site-selection lands.
- Does not sweep `SPINQUANT_SEED`. Single seed 42 for all four modes.
- Does not combine site-selective rotation with training-time changes (spec 011 territory).

## Decision tree after this runs

- **`attn_only` ≤ −0.001:** confirmed lever. Promote this variant; consider stacking with spec 011 (tapered WD retrain on top of attn-only SpinQuant).
- **`attn_only` null (|Δ| ≤ 0.0005):** regime-dependence doesn't decompose by site. Spec 010c (layer-selective) still worth trying, but prior is weaker. Pivot to spec 011 as the primary lever.
- **`mlp_only` clearly hurts and `attn_only` clearly helps:** clean decomposition, attn_only is the working variant. Consider a combined (attn_only + tapered WD) run.
- **Both `attn_only` and `mlp_only` null, `all` null:** SpinQuant truly done on this stack. Full pivot to spec 011 and non-quant levers.
