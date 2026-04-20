# Spec 010 — Port #1695's online Hadamard rotation scheme

**Slug:** `port-1695-online-rotation`
**Created:** 2026-04-20
**Links to idea:** `research/ideas/1736-improvement.md` and `research/ideas/spinquant-integration-notes.md` (Addendum section: "how #1695 actually does it").
**Depends on:** spec 008 complete (`pre_gptq.pt` available). Can run **after** (or in parallel with) spec 009.
**Implementation state:** code landed in commit (see below). Float-pass invariance verified by construction (`F.linear(x @ R, W @ R) == F.linear(x, W)` for orthogonal R); GPU behavior unverified until pod runs.

## Hypothesis

PR #1695's scheme — **online activation rotation with rotated-basis GPTQ** — delivers ~−0.005 bpb on the #1529 base. Porting it to #1736's stack should yield similar or better gain because #1736's stack is strictly richer (CaseOps + gates + phased TTT) without conflicts with the rotation design.

## Baseline

Spec 009's `baseline` mode (our reproduced #1736 seed-42 number, measured end-to-end by spec 009).

## Expected Δ

−0.003 to −0.005 bpb vs baseline. Stronger than spec 009's `internal_only` mode (~−0.002) because it rotates in four positions instead of one, and handles the MLP via the post-nonlinearity hook.

If `internal_only` already delivered ≥ −0.003 in spec 009, this lever's incremental gain on top may be smaller (~−0.001 to −0.002) — in that case the combined delta against spec 009 baseline could be ~−0.004 total.

## Approach overview (see integration notes addendum for full design)

#1695 uses **four Hadamard rotations applied online in the forward pass** — not baked into weights, not folded through nonlinearities.

| Rotation | Dim | Site |
|---|---|---|
| `R_attn_in` | d_model (512) | `x_qkv = x @ R_attn_in` before Q/K/V linear |
| `R_attn_proj_in` | d_model (512) | `y = y @ R_attn_proj_in` before attn output proj |
| `R_mlp_in` | d_model (512) | `x = x @ R_mlp_in` before fc |
| `R_mlp_proj_in` | d_ff (2048) | `hidden = hidden @ R_mlp_proj_in` before proj (applied AFTER `LeakyReLU.square`) |

Rotations are `register_buffer`s (non-persistent, regenerated deterministically from `SPINQUANT_SEED`). Gated by `CastedLinear._sq_active` class flag — OFF during training (Dynamo constant-folds branch away), ON after `deserialize()` for quantized eval + TTT.

GPTQ Hessian must be rotated to match: `H_new = R.T @ H @ R` for each linear whose input is rotated.

**Why it works where static rotation doesn't:**

- `R_mlp_proj_in` applies after LeakyReLU² → no non-linearity to commute through.
- Rotations operate on per-linear-input, never the residual stream → per-channel multipliers (`attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`) stay in trained basis, untouched.
- Float pass is different from unrotated trained model — **no invariance check**. The bet: rotated-basis GPTQ error is lower, perturbation ≪ savings.

## Accept criteria

### Preflight
- CPU-side sanity test: rotate a tiny model, verify GPTQ calibration runs without numerical blow-up on the rotated Hessian (no NaN, no inf in rotated H's eigenvalues). Optional — this is less critical than spec 009's invariance test because we're not claiming float invariance.

### On-pod
- Script loads `pre_gptq.pt`, installs rotation buffers via `install_spinquant_rotations(...)`, sets `CastedLinear._sq_active = True`.
- GPTQ runs (rotated Hessian path) without error.
- Artifact < 16 MB.
- Phased TTT completes within 600 s.
- `final.json` with pre-quant, quantized, and post-TTT bpb.

### Primary success
- **val_bpb < spec 009 baseline by ≥ 0.002** → SpinQuant online rotation lands on #1736, matches #1695's witnessed gain.
- Ideally beats spec 009's `internal_only` by ≥ 0.001 → confirms the 4-rotation approach is worth the invasiveness.

## Config diff

```
SPINQUANT_ENABLED=1
SPINQUANT_SEED=42
HOTSTART_FP_CKPT=/workspace/runs/008-1736-reproduction/seed_42/pre_gptq.pt
```

Plus `ARTIFACT_DIR=/workspace/runs/010-port-1695/`.

## Code changes (as implemented)

Landed in `train_gpt.py` + driver dispatch in `spinquant_hotstart.py`. All changes env-var-gated (`SPINQUANT_ENABLED`, default 0) so this spec does not perturb spec 008 or spec 009's `baseline`/`internal_only` modes.

### `train_gpt.py`

1. **Import:** `hashlib` (for `_stable_seed`).
2. **Hyperparameters:** `spinquant_enabled` (bool, env `SPINQUANT_ENABLED`), `spinquant_seed` (int, env `SPINQUANT_SEED`, default 42).
3. **`CastedLinear._sq_active`:** class-level bool flag, default `False`.
4. **Utility block** (after `_rebank_state_dict`):
   - `_stable_seed(seed, tag)` — SHA-256-derived deterministic seed.
   - `_hadamard_rotation(n, seed, tag)` — Sylvester-Hadamard × random sign diag, QR re-orthogonalized. Cached by `(seed, tag, n)`.
   - `install_spinquant_rotations(model, h, seed, log_fn)` — registers `_sq_R_attn_in`, `_sq_R_attn_proj_in` buffers on every `CausalSelfAttention` and `_sq_R_mlp_in`, `_sq_R_mlp_proj_in` on every `MLP`.
   - `_SQ_KEY_TO_TAG` — suffix → tag map for the 6 rotated state_dict keys.
   - `_spinquant_rotate_sd_and_H(sd_cpu, hessians, h, log_fn)` — in-place rotates matching weights (`W ← W @ R`) and Hessians (`H ← R.T @ H @ R`).
5. **Forward hooks (4 modules × 2 sites = 8 hook insertions):**
   - `CausalSelfAttention.forward` — pre-QKV (`x_qkv = x @ R_attn_in`) and pre-attn-proj (`y @ R_attn_proj_in`).
   - `MLP.forward` — pre-fc (`x @ R_mlp_in`) and post-activation pre-proj (`hidden @ R_mlp_proj_in`). Disables fused kernel when active.
   - `_block_with_lora` (TTT sequential path) — matching two sites with LoRA using unrotated `n`.
   - `_parallel_block_with_lora` (TTT parallel path) — matching two sites.
6. **`serialize()`:** after `collect_hessians()` returns and before `gptq_mixed_quantize()`, calls `_spinquant_rotate_sd_and_H` if `h.spinquant_enabled`.
7. **`deserialize()`:** after `load_state_dict`, calls `install_spinquant_rotations` and sets `CastedLinear._sq_active = True` if `h.spinquant_enabled`.

### `spinquant_hotstart.py`

`port_1695` mode now sets `h.spinquant_enabled = True` and `h.spinquant_seed = base_seed`. All the actual rotation work is inside `train_gpt.py`'s `serialize()` and `deserialize()`. The driver contributes no new rotation logic; it's just a dispatch flag.

### Compatibility

- Spec 008's env block with `SPINQUANT_ENABLED=0` (default) → unchanged behavior.
- Spec 009's `baseline` and `internal_only` modes don't touch `SPINQUANT_ENABLED` — they continue to use their in-driver static R_a rotation. Unaffected.
- Only `spinquant_hotstart.py` launched with `SPINQUANT_MODE=port_1695` activates the new code path.

### Reference

PR #1695's diff, lines ~920–1080 (utility block) and ~1329 (forward hooks) and ~2880–2945 (serialize/deserialize integration). Porting was largely mechanical; only adjustment to #1736's stack was the TTT-path hooks, which go in `_block_with_lora` / `_parallel_block_with_lora` rather than the single-forward `forward_ttt` their base had.

## Hardware ladder

8×H100, single seed (42). Same pod shape as spec 009. ~10 min compute + eval + TTT.

## Seed plan

Single seed 42. If it wins clearly (>−0.002 over baseline and > spec 009 internal_only), 3-seed confirmation becomes the next spec.

## Inputs

- **FP checkpoint:** `runs/008-1736-reproduction/seed_42/pre_gptq.pt` (spec 008 output).
- **Data:** same CaseOps dataset as spec 008.
- **Tokenizer:** bundled.
- **Prior result needed first:** spec 009's `baseline` mode (gives us a measured spec-008-equivalent post-TTT number to compare against).

## Execution protocol

Execution runs `spinquant_hotstart.py` with `SPINQUANT_MODE=port_1695`. The driver toggles the Hyperparameters flag, train_gpt.py's serialize/deserialize do the rest.

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

mkdir -p /workspace/runs/010-port-1695

NCCL_NET=Socket DATA_DIR=./data \
ARTIFACT_DIR=/workspace/runs/010-port-1695 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
SPINQUANT_MODE=port_1695 \
SPINQUANT_SEED=42 \
HOTSTART_FP_CKPT=/workspace/runs/008-1736-reproduction/seed_42/pre_gptq.pt \
SEED=42 \
torchrun --standalone --nproc_per_node=8 spinquant_hotstart.py \
  > /workspace/runs/010-port-1695/run.log 2>&1
```

Note: `SPINQUANT_MODE=port_1695` is the driver flag (read by `spinquant_hotstart.py`). The driver sets `h.spinquant_enabled = True` internally — no need to pass `SPINQUANT_ENABLED=1` as env.

## Stop-early criteria

- GPTQ Hessian rotation produces non-finite values → halt, debug Hessian math.
- Artifact > 16 MB → halt.
- val_bpb > spec 009 baseline + 0.003 → likely a forward-pass hook bug, halt.

## Checkpoints to emit

None. Reuses spec 008's `pre_gptq.pt` as sole input. Output is the rotated-and-quantized `.ptz` artifact.

## Cost estimate

| Item | Cost |
|---|---|
| Pod spin-up + compile warm-up | $2 |
| Port setup (Hessian rotation debug if needed) | $3 |
| Single run (8×H100, ~10 min GPU) | $5 |
| **Total** | **~$10** |

Cheaper than spec 008 because no training.

## Extra artifacts

- `runs/010-port-1695/run.log`
- `runs/010-port-1695/final_model.int6.ptz`
- `runs/010-port-1695/rotation_manifest.json`
- `runs/010-port-1695/final.json`

## Open questions for interview

1. **Hessian-rotation math:** does `H_new = R.T @ H @ R` correctly capture the relationship for all four rotation sites? `R_mlp_proj_in` acts on the post-nonlinearity hidden, so its corresponding Hessian is collected from `hidden.detach()` at line ~822. Double-check the collected-tensor identity before rotating.
2. **GPTQ clip-sigma behavior:** `MLP_CLIP_SIGMAS=12.0`, `ATTN_CLIP_SIGMAS=13.0` were tuned for #1736's unrotated distributions. After rotation, weight/activation variance may shift. Initial run with original sigmas — if calibration fails or clip triggers excessively, sweep `*_CLIP_SIGMAS` wider.
3. **Training-time flag:** `CastedLinear._sq_active` must be `False` during any TTT training step (so LoRA trains on unrotated forward consistently). The spec-009 TTT code path would be affected too if we ever composed the two. For spec 010 alone this is fine — we never retrain.
4. **`_spinquant_rotate_sd_and_H` exact contents:** read the function in `#1695`'s diff and port it verbatim; their implementation handles the state_dict side too (are any weights rotated statically in addition to activations? check during porting).
5. **Seed convention:** `#1695` uses `SPINQUANT_SEED=20260416` (their date). Spec 010 uses 42 to match our seed convention. If sensitivity to rotation-seed is detectable, a sweep can come later.

## What this spec does NOT do

- Does not touch any non-quant lever.
- Does not retrain. Hotstart only.
- Does not sweep rotation seed, clip sigmas, or schedule — single config port.
- Does not attempt a hybrid with spec 009's `internal_only` R_a rotation. If both land positive, a follow-up spec can try the combination.
- Does not modify #1736's training loop — rotation is only active post-deserialize for eval.
