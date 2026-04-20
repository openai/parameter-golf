# Spec 009 — SpinQuant hotstart on top of #1736 (ABC sweep)

**Slug:** `spinquant-hotstart`
**Created:** 2026-04-20
**Links to idea:** `research/ideas/1736-improvement.md` and `research/ideas/spinquant-integration-notes.md` (full integration analysis).
**Depends on:** spec 008 complete, `runs/008-1736-reproduction/seed_42/final_model.pt` present (auto-saved by `serialize()` before GPTQ — no code patch needed in spec 008).

## Scope

**This session implements two modes. Two more are deferred to a follow-up spec while modes 1–2 run on the pod.**

Modes selectable by `SPINQUANT_MODE` env var:

- **`baseline`** (implemented) — no rotation. Loads `final_model.pt`, calls `serialize()` → `deserialize()` → eval + TTT. Produces our local post-TTT number for #1736 reproduction (the spec 008 gate number that was missed by the watcher-trigger bug). Doubles as the apples-to-apples reference that SpinQuant Δs are measured against.
- **`internal_only`** (implemented) — per-layer attention internal rotation **R_a only** (V-out / O-in, per KV-group). Float-invariant by construction (softmax(QKᵀ)V is rotation-equivariant in V's d_head axis). Does NOT include MLP internal rotation R_m, because #1736's MLP nonlinearity (LeakyReLU(slope=0.5)→square) is not rotation-equivariant. Expected Δ: −0.001 to −0.002 bpb vs baseline (R_a alone is ~half of full internal SpinQuant's benefit).
- **`full`** (DEFERRED to follow-up spec) — would add residual-stream R₀ with per-channel multiplier folds (attn_scale, mlp_scale, skip_weights) and `resid_mix` freeze-to-mean. Deferred because the fold design needs discussion (see integration notes + `resid_mix` issue). Script will reject `SPINQUANT_MODE=full` with a helpful error.
- **`port_1695`** (DEFERRED) — pending read of `#1695`'s actual diff. Script rejects with "not implemented yet."

Modes 1–2 hotstart off the same `final_model.pt` and run back-to-back on one pod (~10 min compute each). Total ~20 min GPU, ~$12–15. `baseline` closes spec 008's missed gate number in the same session.

**Why the scope cut:** the MLP LeakyReLU breaks `R_m` float-invariance without either (a) sign-only rotations (tiny benefit) or (b) accepting model perturbation (non-zero-cost transform, needs empirical justification). The residual-stream rotation in `full` has the separate `resid_mix` problem. Both are legitimate research directions but shouldn't block the 2-mode first pass.

## Hypothesis

Hadamard rotation of weight matrices before GPTQ quantization (SpinQuant V1) spreads weight-distribution outliers uniformly across input dimensions. This reduces quantization error at fixed bit-width, improving post-quant val_bpb without touching the float-precision forward pass. Witnessed on PR #1695 (X-Abhishek-X) at claimed −0.005 bpb on top of a #1529-adjacent base; expected to compose cleanly with #1736 since the quant stage is orthogonal to CaseOps / attention gates / phased TTT.

## Baseline

Spec 008's reproduced seed-42 val_bpb (target ~1.06610 ± 0.003). Exact number is whatever spec 008 actually lands; this spec compares Δ against that.

## Expected Δ (per mode, all vs local `baseline` mode)

| Mode | Expected Δ (bpb) | Notes |
|---|---|---|
| `baseline` | 0 (reference) | closes the loop on spec 008's gate number; target ≈ 1.066 (within ±0.001 of #1736's 1.06610 reproduces training) |
| `internal_only` | −0.002 to −0.004 | majority of SpinQuant benefit comes from internal rotations; clean, low-risk |
| `full` | −0.004 to −0.007 | only if `resid_mix` freeze doesn't degrade too much |
| `port_1695` | depends on #1695 | runs unconditionally — if their approach matches A, confirms implementation; if different, third data point |

Null or positive Δ on `internal_only` → implementation bug (almost certainly a banked-slice misalignment); halt the sweep and debug before proceeding to `full`.

## Accept criteria

### Phase 1 — FP invariance sanity (before GPTQ)
- Load `pre_gptq.pt`, run one forward pass on a small batch, record logits.
- Apply all rotation classes (see "Rotation structure" below).
- Re-run the same forward pass post-rotation, compare logits.
- **Must match within float tolerance** (max abs diff ≤ 1e-3 bf16, ≤ 1e-5 fp32). If not, rotation pairs are inconsistent → halt, debug.

### Phase 2 — GPTQ + TTT + eval
- GPTQ quantization completes under same config as #1736 (`EMBED_BITS=7`, `MLP_CLIP_SIGMAS=12.0`, `ATTN_CLIP_SIGMAS=13.0`, etc.).
- Artifact < 16,000,000 bytes.
- Phased TTT eval completes within 600 s.
- val_bpb is reported.

### Primary success
- **val_bpb < spec 008 baseline by ≥ 0.003** (moves in the expected direction with plausible magnitude).
- Ideally ≥ 0.0072 for a standalone-record claim (0.005 nat threshold), but not required — this is a screen.

## Config diff

Same env block as spec 008 (identical GPTQ / TTT / gate settings). Two additions:

```
SPINQUANT_ENABLED=1
SPINQUANT_SEED=42                  # seed for the random orthogonal / signed-Hadamard generator
HOTSTART_FP_CKPT=/workspace/runs/008-1736-reproduction/seed_42/pre_gptq.pt
```

No training run. No `MATRIX_LR` / `MUON` / dataset settings matter — this spec skips training entirely.

## Rotation structure

Three classes of rotations, all fixed (not learned), stored as non-parameter buffers. Derived from reading #1736's `GPT` class (line 876) and `_unbank_state_dict` (line 2003).

### Class 1: Residual-stream R₀ (one shared [512, 512] orthogonal matrix)

Applied to every weight tensor that reads from or writes into the residual stream. In #1736's banked layout (n = num_layers = 11):

| Banked tensor | Slice / index | Semantic | Rotation op |
|---|---|---|---|
| `qo_bank` | `[0:n]` | Q projections (c_q) — reads residual | `W[i] ← W[i] @ R₀.T` (input-side) |
| `qo_bank` | `[n:2n]` | attn output proj — writes residual | `W[i] ← R₀ @ W[i]` (output-side) |
| `kv_bank` | `[0:n]` | K projections — reads residual | `W[i] ← W[i] @ R₀.T` |
| `kv_bank` | `[n:2n]` | V projections — reads residual (V output is internal to attn) | `W[i] ← W[i] @ R₀.T` |
| `mlp_up_bank` | `[0:n]` | MLP fc — reads residual | `W[i] ← W[i] @ R₀.T` |
| `mlp_down_bank` | `[0:n]` | MLP proj — writes residual | `W[i] ← R₀ @ W[i]` |
| `tok_emb.weight` | — | embedding (lookup writes residual) | `W ← W @ R₀.T` (rotates d_model columns) |
| `lm_head.weight` | — | output head (reads residual) | `W ← W @ R₀.T` |
| `skip_weights` | — | U-net skip-channel weights (line 956), shape `[num_skip_weights, d_model]`, per-channel mixing on residual | needs equivalent channel rotation — per-channel diagonal absorbed into R₀ context; see "skip-stream complication" below |

### Class 2: Per-layer attention R_a^ℓ (11 × [d_head, d_head] orthogonal)

Internal to each attention block, rotates the V-output / O-input basis. Not on residual, so independent per layer (no Loop45 consistency issue — each layer just gets its own seeded rotation).

Applied to: V projection rows (output dim of V) and O projection columns (input dim of O), matched per layer. Exact banked indices: `kv_bank[n + i]` output rows and `qo_bank[n + i]` input columns.

### Class 3: Per-layer MLP R_m^ℓ (11 × [d_ff, d_ff] orthogonal)

Internal to each MLP block, rotates the fc-output / proj-input basis. Independent per layer.

Applied to: `mlp_up_bank[i]` output rows and `mlp_down_bank[i]` input columns.

### R construction

Preferred: `R = diag(±1) · Hadamard(d)` — signed Hadamard is structured, outlier-spreading, and cheap. Random sign chosen via `SPINQUANT_SEED`. Fallback for non-power-of-2 dims: `R = Q` where `Q, _ = torch.linalg.qr(torch.randn(d, d))` — random orthogonal.

For R₀ at d=512: Hadamard(512) exists (512 = 2⁹), so signed Hadamard is the clean choice.

### Critical constraints

1. **R₀ is shared across all 11 layers AND across Loop45 recurrence passes.** Key the rotation state by "residual stream identity," not by "invocation index." If layers 4 or 5 somehow end up with different R₀ across their multiple invocations (e.g., if rotation were applied inside `block.forward` dynamically), float invariance breaks — but since we rotate the weights statically before eval, this isn't a runtime concern.
2. **R_a and R_m are per-layer and completely independent** — no Loop45 issue. Layers 4 and 5 each get one R_a and one R_m; those rotations apply on every Loop45 pass through those layers naturally.
3. **The integration point is line 2978 of `train_gpt.py`** (just before `serialize(h, base_model, ...)`). Rotations apply to `base_model`'s parameters in-place. `serialize()` then GPTQ-quantizes the rotated weights without modification.

## Code changes

- **Branch:** `research` (this is a commitment-class change — quant lever becomes part of our baseline if it lands).
- **New file:** `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/spinquant_hotstart.py` (~200–300 LOC). Approximate structure:

  ```python
  from train_gpt import (
      Hyperparameters, GPT, ValidationData,
      serialize, deserialize, eval_val, eval_val_ttt_phased,
      BatchedTTTLoRA, timed_eval,
      restore_fp32_params,
  )

  def build_rotations(d_model, d_head, d_ff, num_layers, seed):
      # Signed-Hadamard for R₀, R_a^ℓ, R_m^ℓ
      ...

  def fold_rmsnorm_into_next_linear(state_dict, num_layers):
      # Fold attn_norm.gamma into qo_bank[:n] and kv_bank[:n] (input-side),
      # mlp_norm.gamma into mlp_up_bank, final_norm.gamma into lm_head.
      # Then set all gammas to 1. See "RMSNorm fold" below.
      ...

  def apply_spinquant_rotations(state_dict, R0, R_a_list, R_m_list, num_layers):
      # Apply rotation map from the Rotation Structure table above.
      # Also rotates skip_weights if present.
      ...

  def verify_fp_invariance(model_orig, model_rotated, val_data, device, tol):
      # Run forward on a small batch pre/post rotation. max abs logit diff < tol.
      ...

  def main():
      h = Hyperparameters()
      # 1. Build empty GPT(h), load state_dict from final_model.pt
      # 2. Fold RMSNorm gammas
      # 3. Build rotations, apply in-place
      # 4. FP invariance check (Phase 1 accept)
      # 5. Call serialize(h, base_model, code=...) — triggers GPTQ on rotated weights
      # 6. Call deserialize(h, device) — loads quantized
      # 7. Reproduce the quantized-eval + TTT blocks from train_and_eval() (lines 2969-3075)
      # 8. Write final.json with bpb, delta vs baseline, rotation seeds
  ```

- **Small refactor in `train_gpt.py` (preferred path):** factor the TTT eval block at lines 2997–3075 of `train_and_eval` into a named helper:

  ```python
  def run_ttt_eval(h, device, val_data):
      """Loads quantized model from h.quantized_model_path, warms up TTT LoRA compile,
      runs eval_val_ttt_phased, logs the result. Returns (val_loss, val_bpb)."""
      ...
  ```

  Then `train_and_eval` calls `run_ttt_eval(...)` instead of inlining, and `spinquant_hotstart.py` can too. This is a pure refactor with no behavior change — safe to include as part of spec 009's commit. Alternative: copy-paste the block verbatim into the hotstart script (~80 LOC dup), simpler but maintains two copies.

- **No other modifications** to `train_gpt.py`.

- **Reference code:** PR #1695's diff (SpinQuant V1) once visible — port rotation bookkeeping if their layout is compatible. Otherwise re-derive from the SpinQuant paper (Liu et al. 2024) cross-referenced against #1736's banked layout.

## RMSNorm fold (required before rotation)

RMSNorm does `y = gamma · x / ||x||_RMS`. For the rotation to preserve float invariance, RMSNorm must be in its "gamma=1" form, because only then does RMSNorm commute with orthogonal rotation:

`RMSNorm(R·x) = (R·x) / ||R·x||_RMS = (R·x) / ||x||_RMS = R · RMSNorm(x)` (orthogonal preserves norms).

With gamma ≠ 1, the per-channel scaling breaks commutativity unless `gamma` is also rotated, but then it's no longer an elementwise scaling.

**Fold procedure:** before applying rotations, fold each RMSNorm's gamma into the weight rows/columns of the next linear layer on the residual-read side, then set gamma to 1. Specifically in #1736:

| RMSNorm | Location | Fold target (next linear on residual-read side) |
|---|---|---|
| `attn_norm` (per-block) | before attention | `qo_bank[i]` columns (for c_q), `kv_bank[i]` columns (for c_k), `kv_bank[n+i]` columns (for c_v) — all get `W[:, j] ← W[:, j] · gamma_j` |
| `mlp_norm` (per-block) | before MLP | `mlp_up_bank[i]` columns: `W[:, j] ← W[:, j] · gamma_j` |
| `final_norm` | before lm_head | `lm_head.weight` columns: `W[:, j] ← W[:, j] · gamma_j` |

After folding, set each RMSNorm's gamma parameter to 1 (or replace the module with a plain `x / ||x||_RMS` that has no gamma). The forward pass is numerically unchanged at this stage — the fold is identity-preserving as long as RMSNorm is immediately followed by one of the listed linear layers.

## Skip-stream complication

Lines 893–960 of `train_gpt.py` reveal a U-net encoder/decoder split:

- `num_encoder_layers = num_layers // 2 = 5`
- `num_decoder_layers = 6`
- `skip_weights` tensor of shape `[num_skip_weights, model_dim]` — per-channel weights that mix encoder outputs into decoder residual stream.

Because the skip path carries residual-stream values, it has to be rotation-consistent too. The naive per-channel multiplier doesn't compose with a full rotation R₀.

**Two options** (execution to pick based on the actual forward code):

- **(a) Element-wise skip:** if skip is applied as `decoder_residual = decoder_residual + skip_weights[k] * encoder_residual` (elementwise), then rotating both residual streams with the same R₀ is consistent AS LONG AS we leave `skip_weights` untouched — because elementwise multiplication commutes with rotation only when the multiplier is a rotation-equivariant operator, which for general per-channel vectors it is not. This breaks float invariance. **Fix:** either fold `skip_weights` into the nearest linear on the decoder side (similar to RMSNorm fold) so the residual path passes through a rotated linear, or replace the per-channel scaling with a full [d, d] linear that can absorb R₀. The first is cleaner if the skip-application code permits.
- **(b) Matmul skip:** if skip is applied as a proper linear `decoder_residual = decoder_residual + skip_linear_k(encoder_residual)`, then rotate `skip_linear_k.weight` by R₀ on both input and output sides: `W ← R₀ @ W @ R₀.T`. Float invariance preserved.

**Action for execution:** read the `GPT.forward` / block-orchestration code around line 956 (`skip_weights`) to determine which form #1736 uses, then apply the matching fix. This is the single highest-risk part of the integration — get this wrong and FP invariance fails, Phase 1 accept fails immediately.

## Hardware ladder

- [x] **1×H100** — sufficient. No training, just rotate + GPTQ (~2 min) + TTT eval (~6–10 min). Could also use 2×H100 if DDP is needed for TTT parallelization, but TTT eval in #1736 runs on 8 ranks per its phased-TTT setup — may need to check whether single-rank eval is supported.
- **Fallback:** 8×H100 if phased TTT requires multi-rank.

## Seed plan

Single seed (42), matching spec 008. Compares directly against spec 008's seed-42 number.

## Inputs

- **FP checkpoint:** `runs/008-1736-reproduction/seed_42/pre_gptq.pt` (spec 008 output).
- **Data:** same CaseOps dataset as spec 008 (on persistent volume, already prepared).
- **Tokenizer:** bundled with #1736 submission dir, unchanged.

## Execution protocol

Single pod, four sequential runs — `baseline` first (closes spec 008's missed gate number and establishes local eval reference), then `internal_only`, `full`, `port_1695`.

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
  SPINQUANT_SEED=42
  HOTSTART_FP_CKPT=/workspace/runs/008-1736-reproduction/seed_42/final_model.pt
  SEED=42
)

# Variant 0: baseline (closes the loop on spec 008's missed gate number)
mkdir -p /workspace/runs/009-spinquant-hotstart/baseline
env "${COMMON_ENV[@]}" \
  ARTIFACT_DIR=/workspace/runs/009-spinquant-hotstart/baseline \
  SPINQUANT_MODE=baseline \
  torchrun --standalone --nproc_per_node=8 spinquant_hotstart.py \
  > /workspace/runs/009-spinquant-hotstart/baseline/run.log 2>&1

# Gate: baseline post-TTT val_bpb should be within 0.003 of #1736's 1.06610.
# If not, spec 008 reproduction issue — halt before running SpinQuant variants.

# Variant B: internal-only
mkdir -p /workspace/runs/009-spinquant-hotstart/internal_only
env "${COMMON_ENV[@]}" \
  ARTIFACT_DIR=/workspace/runs/009-spinquant-hotstart/internal_only \
  SPINQUANT_MODE=internal_only \
  torchrun --standalone --nproc_per_node=8 spinquant_hotstart.py \
  > /workspace/runs/009-spinquant-hotstart/internal_only/run.log 2>&1

# Gate: if Variant B failed FP invariance or val_bpb > spec-008 baseline + 0.003, HALT.
# Almost certainly means banked-slice misalignment — fix before proceeding to A.

# Variant A: full (internal + residual + folds)
mkdir -p /workspace/runs/009-spinquant-hotstart/full
env "${COMMON_ENV[@]}" \
  ARTIFACT_DIR=/workspace/runs/009-spinquant-hotstart/full \
  SPINQUANT_MODE=full \
  torchrun --standalone --nproc_per_node=8 spinquant_hotstart.py \
  > /workspace/runs/009-spinquant-hotstart/full/run.log 2>&1

# Variant C: always run.
mkdir -p /workspace/runs/009-spinquant-hotstart/port_1695
env "${COMMON_ENV[@]}" \
  ARTIFACT_DIR=/workspace/runs/009-spinquant-hotstart/port_1695 \
  SPINQUANT_MODE=port_1695 \
  torchrun --standalone --nproc_per_node=8 spinquant_hotstart.py \
  > /workspace/runs/009-spinquant-hotstart/port_1695/run.log 2>&1
```

After all three complete: `runpodctl pod stop $POD_ID` per the default memory policy.

## Kill protocol

- FP invariance check fails (Phase 1) → halt, save rotation seeds + diff stats, flag research.
- GPTQ calibration fails or hangs > 5 min → halt.
- Eval hangs > 15 min → halt, stop pod.
- After successful completion: stop pod per memory default.

## Stop-early criteria

- Phase 1 FP invariance: max abs logit diff > 1e-3 (bf16) → halt before wasting GPTQ time.
- Artifact size > 16 MB → halt, flag.
- val_bpb > spec 008 baseline + 0.003 (got *worse*) → likely rotation error, halt.

## Checkpoints to emit

**None.** Spec 009 is pure post-training, no new FP state worth saving. The only artifact is the rotated-and-quantized `.ptz` submission + the log.

## Cost estimate

| Item | Cost |
|---|---|
| Pod spin-up + framework compile warm-up | $2 |
| `baseline` (~10 min GPU) — closes spec 008's missed gate number | $5 |
| `internal_only` (~10 min GPU) | $5 |
| `full` (~10 min GPU) | $5 |
| `port_1695` (~10 min GPU) | $5 |
| Buffer for debug (invariance mismatches, GPTQ hiccups) | $5 |
| **Total** | **~$27** |

Folds in spec 008's otherwise-separate ~$3 eval-only rerun, so net ~$24 of new spend (and 4 measured numbers vs 2 across two specs).

Same `final_model.pt` hotstarts all three, so marginal cost per variant is just compute. The up-front integration work (building the unified `spinquant_hotstart.py` with toggle flags) is research-side, not execution-side — not on this budget line.

## Extra artifacts

Per variant (one subdir per mode):

- `runs/009-spinquant-hotstart/<mode>/run.log` — full log
- `runs/009-spinquant-hotstart/<mode>/final_model.int6.ptz` — rotated + quantized submission artifact
- `runs/009-spinquant-hotstart/<mode>/rotation_seeds.json` — rotation parameters (base seed + which modes were active + per-layer seed offsets)
- `runs/009-spinquant-hotstart/<mode>/invariance_report.json` — FP forward-pass diff stats pre vs post rotation
- `runs/009-spinquant-hotstart/<mode>/final.json` — val_bpb, Δ vs spec 008, artifact size, wall times

Top-level summary written by research during evaluation:

- `runs/009-spinquant-hotstart/summary.md` — side-by-side table of all three modes' final val_bpb, Δ, artifact size, and commentary on which won and why.

## Open questions for interview

1. **Skip-stream form** — is the U-net skip at line 956 element-wise multiply or a full linear matmul? Read `GPT.forward` or the block-loop code; apply the matching rotation fix (see "Skip-stream complication" section). High-impact: this is the most likely failure mode for FP invariance.
2. **TTT refactor vs copy-paste** — option (b) refactor `train_and_eval`'s TTT block (lines 2997–3075) into `run_ttt_eval(...)` helper and call it from both the hotstart script and `train_and_eval`, OR option (a) duplicate the block verbatim in the hotstart script. Preference: (b) — cleaner, single source of truth. Execution's call.
3. **GPTQ already handles mixed precision** (see `gptq_mixed_quantize` at 1863) — need to confirm that the rotated weights don't violate whatever the mixed-precision allocator assumes (e.g., per-matrix outlier heuristics). Likely fine since rotation *reduces* outliers, but worth a sanity assert during GPTQ.
4. **Phased-TTT ordering** — phased TTT at line 3064 (`eval_val_ttt_phased`) adapts LoRA on top of the quantized model. Rotation happens once, pre-GPTQ. TTT LoRA is then trained on top of the rotated-and-quantized base. This should be fine: LoRA is post-hoc and doesn't rely on the un-rotated basis, but worth verifying that `BatchedTTTLoRA` doesn't make assumptions about the weight distribution shape.
5. **Reference code availability** — is #1695's diff visible (is the PR source readable via `gh pr diff 1695`)? If yes, port rotation bookkeeping directly. If not, re-derive from the SpinQuant paper (Liu et al. 2024) and cross-check against the banked layout.
6. **Hadamard impl** — does torch ship a Hadamard primitive, or do we need `scipy.linalg.hadamard` / a hand-rolled one? For d=512, `scipy.linalg.hadamard(512)` returns a `{±1}` matrix that we scale by `1/√512` for orthogonality. Cheap either way.

## What this spec does NOT do

- Does not retrain any weights. All three variants are post-training transforms on spec 008's `final_model.pt`.
- Does not tune `SPINQUANT_SEED` — first run uses 42; if rotation-seed sensitivity matters (unlikely for Hadamard) we can sweep later.
- Does not change CaseOps, gates, TTT, or any other non-quant lever.
- Does not run multi-seed — matches spec 008's single-seed convention. If one of the three modes wins and we later need 3-seed confirmation, a follow-up spec runs all three data seeds on the winning mode.
- Does not attempt to fold `resid_mix` in Option A's fold path — `resid_mix` is frozen to its channel-mean value, accepting a small float-pass perturbation. Alternative treatments are deferred to follow-up specs if Option A shows signal but invariance is problematic.
