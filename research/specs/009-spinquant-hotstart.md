# Spec 009 — SpinQuant hotstart on top of #1736 (2-mode sweep)

**Slug:** `spinquant-hotstart`
**Created:** 2026-04-20
**Links to idea:** `research/ideas/1736-improvement.md` and `research/ideas/spinquant-integration-notes.md` (full integration analysis).
**Depends on:** spec 008 complete, `runs/008-1736-reproduction/seed_42/pre_gptq.pt` present on the JP volume (saved by execution's `SAVE_PRE_GPTQ` patch in spec 008 — this file is the hotstart input).

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

## Expected Δ (per mode, both vs local `baseline` mode)

| Mode | Expected Δ (bpb) | Notes |
|---|---|---|
| `baseline` | 0 (reference) | closes the loop on spec 008's missed gate number; target ≈ 1.066 (within ±0.003 of #1736's 1.06610 reproduces training) |
| `internal_only` | −0.001 to −0.002 | R_a-only rotation; roughly half the benefit of the full SpinQuant stack (R_m deferred because LeakyReLU breaks float-invariance there) |

Null or positive Δ on `internal_only` → implementation bug (almost certainly a banked-slice or head-group indexing misalignment); halt, flag research. The CPU invariance test (`test_rotation_invariance.py`) already verifies the math on both synthetic and real weights — strong prior against a bug, but GPU bf16 could still reveal something CPU fp32 doesn't.

## Accept criteria

### Preflight (CPU, before pod spin-up)
Run `python3 test_rotation_invariance.py --ckpt <path-to-pre_gptq.pt>` on the JP volume (or any box with torch installed). Must print `ALL TESTS PASS` — baseline is bit-exact, `internal_only` rel_max < 1e-4. The test is standalone (no flash-attn-3 dependency), so it runs anywhere torch is available.

### Phase 2 — per-mode (on pod)
For each mode (`baseline`, `internal_only`):
- Script completes without error (no NotImplementedError for the deferred modes).
- GPTQ quantization runs under the same config as spec 008 (`EMBED_BITS=7`, `MLP_CLIP_SIGMAS=12.0`, `ATTN_CLIP_SIGMAS=13.0`, etc.).
- Artifact < 16,000,000 bytes.
- Phased TTT eval completes within 600 s.
- `final.json` written with `diagnostic_quantized` and `quantized_ttt_phased` numbers.

### Primary success
- **`baseline`:** post-TTT val_bpb within ±0.003 of #1736's 1.06610 — confirms spec 008 training reproduced #1736.
- **`internal_only` vs `baseline`:** Δ negative (lower bpb); magnitude ≥ 0.001 is a clear signal, ≥ 0.002 is strong.
- **`internal_only` vs #1736's 1.06610:** Δ negative by 0.001–0.003. If positive, SpinQuant R_a didn't land on this stack.

## Config diff

Same env block as spec 008 (identical GPTQ / TTT / gate settings). Three additions:

```
SPINQUANT_MODE=baseline|internal_only     # selector
SPINQUANT_SEED=42                         # base seed; per-(layer, kv-group) offsets derived deterministically
HOTSTART_FP_CKPT=/workspace/runs/008-1736-reproduction/seed_42/pre_gptq.pt
```

Plus `ARTIFACT_DIR=/workspace/runs/009-spinquant-hotstart/<mode>/` per variant.

No training run. No `MATRIX_LR` / `MUON` / dataset-*path* settings matter (dataset is only used for GPTQ calibration + eval, reads the same path spec 008 did).

## Rotation structure (as implemented)

Only one rotation class is implemented for this spec: **per-layer, per-KV-group attention internal R_a** (d_head × d_head, one per attention kv-group per layer).

| Banked tensor | Slice | Op per (layer i, kv-group g) |
|---|---|---|
| `kv_bank[n + i]` | rows `g*d_head : (g+1)*d_head` | `W_slice ← R_a · W_slice` (rotate V output basis) |
| `qo_bank[n + i]` | for each Q-head `h` in the KV-group (`h // group_size == g`), cols `h*d_head : (h+1)*d_head` | `W_slice ← W_slice · R_aᵀ` (counter-rotate O input basis) |

`group_size = num_heads // num_kv_heads = 8 / 4 = 2`. So per layer there are 4 distinct R_a matrices (one per KV-group), each applied to 1 V-head slice + 2 Q-head O-input slices.

R_a is a signed Hadamard of size d_head = 64, seeded by `SPINQUANT_SEED + layer_idx * 1000 + kv_group_idx`. Structured, outlier-spreading, and cheap to construct.

**Float invariance:** `softmax(QKᵀ)V` is rotation-equivariant in V's d_head axis — rotating V by R_a and O's input cols by R_aᵀ exactly cancels in float. Verified by `test_rotation_invariance.py` on synthetic and real checkpoints (rel_max < 1e-6 on the spec 008 checkpoint).

### Not implemented (deferred to follow-up spec)

- **R₀ residual-stream rotation** — requires folding `attn_scale`, `mlp_scale`, `skip_weights` (per-channel multipliers live on the residual path) and handling `resid_mix` (pre-RMSNorm per-channel mix that doesn't fold cleanly). See `research/ideas/spinquant-integration-notes.md` for the full fold analysis.
- **R_m per-layer MLP internal rotation** — MLP nonlinearity `LeakyReLU(slope=0.5)→square` (line 821 of train_gpt.py) is NOT rotation-equivariant. An arbitrary R_m breaks float invariance. Options (sign-only R_m or accepting perturbation) require a separate design call.
- **Skip-stream rotation** — only relevant if R₀ is being applied (skip lives on residual).
- **RMSNorm gamma fold** — NOT needed because #1736's RMSNorm is gamma-free (line 529: `F.rms_norm(x, (x.size(-1),), eps=self.eps)` with no weight arg). This was a false alarm in earlier notes.

## Code changes — what's on disk

- **`spinquant_hotstart.py`** (new, 360 LOC) in the #1736 submission directory. Imports from `train_gpt.py`. Modes: `baseline`, `internal_only` implemented; `full` and `port_1695` raise `NotImplementedError` with explanatory messages.
- **`test_rotation_invariance.py`** (new, 250 LOC) in the same dir. Standalone (no flash-attn-3 / triton dep), runs on any CPU with torch. Supports `--ckpt` (real checkpoint) or `--synthetic` (random weights). Passes on both for baseline (bit-exact) and internal_only (rel < 1e-4).
- **`train_gpt.py`** — unmodified. The TTT eval block from `train_and_eval` (lines 2997–3075) is inlined into `_run_ttt_eval()` in the hotstart script rather than refactoring the source.

## Hardware ladder

- [x] **8×H100** — required. `eval_val_ttt_phased` and the multi-phase global SGD TTT from #1736 assume 8-rank DDP; the script calls `torchrun --standalone --nproc_per_node=8`. Same hardware as spec 008. ~20 min GPU total across both modes.

## Seed plan

Single seed (42), matching spec 008. Compares directly against spec 008's seed-42 number.

## Inputs

- **FP checkpoint:** `runs/008-1736-reproduction/seed_42/pre_gptq.pt` (spec 008 output — unbanked per-layer state_dict saved by execution's `SAVE_PRE_GPTQ` patch after `_unbank_state_dict` ran). The hotstart script re-banks it on load (or handles unbanked-keyed state_dict via `strict=False`).
- **Data:** same CaseOps dataset as spec 008 (on the same persistent volume, already prepared).
- **Tokenizer:** bundled with #1736 submission dir, unchanged.

## Execution protocol

**Preflight (CPU, anywhere torch is available):**

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
python3 test_rotation_invariance.py --ckpt /workspace/runs/008-1736-reproduction/seed_42/pre_gptq.pt
# Expect: "ALL TESTS PASS" — baseline max abs diff = 0, internal_only rel_max < 1e-4.
# If this fails, halt; rotation math needs fixing before spending GPU time.
```

**Pod sweep (2 modes, single pod, ~20 min total):**

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
  HOTSTART_FP_CKPT=/workspace/runs/008-1736-reproduction/seed_42/pre_gptq.pt
  SEED=42
)

# Mode 1: baseline — closes spec 008's missed gate number, also the reference for mode 2's Δ.
mkdir -p /workspace/runs/009-spinquant-hotstart/baseline
env "${COMMON_ENV[@]}" \
  ARTIFACT_DIR=/workspace/runs/009-spinquant-hotstart/baseline \
  SPINQUANT_MODE=baseline \
  torchrun --standalone --nproc_per_node=8 spinquant_hotstart.py \
  > /workspace/runs/009-spinquant-hotstart/baseline/run.log 2>&1

# Gate: baseline post-TTT val_bpb should be within ±0.003 of #1736's 1.06610.
# If not, spec 008 reproduction is off — halt and flag research before mode 2.

# Mode 2: internal_only — per-layer per-KV-group attention R_a rotation.
mkdir -p /workspace/runs/009-spinquant-hotstart/internal_only
env "${COMMON_ENV[@]}" \
  ARTIFACT_DIR=/workspace/runs/009-spinquant-hotstart/internal_only \
  SPINQUANT_MODE=internal_only \
  torchrun --standalone --nproc_per_node=8 spinquant_hotstart.py \
  > /workspace/runs/009-spinquant-hotstart/internal_only/run.log 2>&1
```

After both complete: `runpodctl pod stop $POD_ID` per the default memory policy.

## Kill protocol

- FP invariance check fails (Phase 1) → halt, save rotation seeds + diff stats, flag research.
- GPTQ calibration fails or hangs > 5 min → halt.
- Eval hangs > 15 min → halt, stop pod.
- After successful completion: stop pod per memory default.

## Stop-early criteria

- CPU preflight (`test_rotation_invariance.py`) fails → do NOT spin up the pod. Flag research.
- Any mode's artifact > 16 MB → halt, flag (quant config mismatch).
- `baseline` post-TTT val_bpb > 0.003 off #1736's 1.06610 → halt before running `internal_only`. Spec 008 reproduction needs investigation before any rotation experiment is meaningful.
- `internal_only` post-TTT val_bpb > `baseline` + 0.003 (got *worse*) → likely GPU-specific bf16 drift manifesting in softmax saturation. Halt and compare rotation_manifest.json seeds against the CPU preflight.

## Checkpoints to emit

**None.** Spec 009 is pure post-training, no new FP state worth saving. The only artifact is the rotated-and-quantized `.ptz` submission + the log.

## Cost estimate

| Item | Cost |
|---|---|
| CPU preflight (`test_rotation_invariance.py`) | $0 |
| Pod spin-up + compile warm-up | $2 |
| `baseline` (~10 min GPU) — closes spec 008's missed gate number | $5 |
| `internal_only` (~10 min GPU) | $5 |
| Buffer for debug (GPU bf16 drift, GPTQ hiccups) | $3 |
| **Total** | **~$15** |

Folds in spec 008's otherwise-separate ~$3 eval-only rerun, so net ~$12 of new spend for two measured numbers. Same `pre_gptq.pt` hotstarts both modes.

The `full` and `port_1695` follow-up spec (whenever we design it) reuses the same checkpoint and can run back-to-back with modes already measured here, so per-variant compute cost there will stay ~$5.

## Extra artifacts

Per mode (`baseline` and `internal_only` each get a subdir):

- `runs/009-spinquant-hotstart/<mode>/run.log` — full log
- `runs/009-spinquant-hotstart/<mode>/final_model.int6.ptz` — (possibly rotated) + GPTQ-quantized submission artifact
- `runs/009-spinquant-hotstart/<mode>/rotation_manifest.json` — seed + which rotations were applied (per layer, per KV-group)
- `runs/009-spinquant-hotstart/<mode>/final.json` — pre-quant, quantized, and post-TTT val_bpb + artifact size + wall times

Top-level summary written by research during evaluation:

- `runs/009-spinquant-hotstart/summary.md` — side-by-side table of both modes' val_bpb at each eval stage, Δ vs #1736's 1.06610 and vs local baseline, artifact sizes, commentary.

## Open questions for interview

1. **Checkpoint format** — spec 008's `pre_gptq.pt` was saved via execution's `SAVE_PRE_GPTQ` patch at line 2080 of train_gpt.py, which is *after* `_unbank_state_dict` ran — so the saved keys are per-layer (`blocks.N.attn.c_q.weight`, etc.), not banked (`qo_bank`). The hotstart script calls `load_state_dict(..., strict=False)` and logs missing/unexpected keys. If execution sees many missing `qo_bank`/`kv_bank`/`mlp_*_bank` keys plus many unexpected `blocks.N.*` keys, the script needs a rebank step: import `_rebank_state_dict` from `train_gpt.py` (line 2028) and call it on the loaded dict before `load_state_dict`. Confirm at preflight.
2. **GPTQ mixed precision interaction** — `gptq_mixed_quantize` (line 1863) uses per-matrix outlier heuristics. Rotation *reduces* outliers, which should only help, but worth a sanity glance at the calibration output during the `internal_only` run. If calibration error rises catastrophically vs `baseline`, flag.
3. **Phased-TTT ordering** — phased TTT at line 3064 adapts LoRA on the quantized model. Rotation is pre-GPTQ only, so TTT LoRA trains on top of the rotated-and-quantized base. Should be fine (LoRA doesn't depend on the un-rotated basis), but if `internal_only` mode shows a BPB *regression* only on the TTT-phased step, flag.
4. **bf16 drift on GPU** — the CPU preflight runs in fp32. On pod we operate in bf16. The rotation math is still correct in bf16 but float drift is ~100× larger. Pre-quant post-rotation diagnostic eval (the "diagnostic_pre_quant_post_rotation" logged value) should still match the un-rotated pre-quant value within ~0.001 val_bpb. If the diagnostic shows much bigger drift, that's a signal of accumulation issues we'd want to investigate before trusting the final number.
5. **#1695 diff** — `gh pr diff 1695` for their rotation scheme. Not a blocker for this spec, but useful to check how their approach compares to R_a-only and whether they handled `resid_mix` / R_m; informs the deferred `full` and `port_1695` modes.

## What this spec does NOT do

- Does not retrain any weights. Both modes are post-training transforms on spec 008's `pre_gptq.pt`.
- Does not tune `SPINQUANT_SEED` — first run uses 42; if rotation-seed sensitivity matters (unlikely for Hadamard) we can sweep later.
- Does not change CaseOps, gates, TTT, or any other non-quant lever.
- Does not run multi-seed — matches spec 008's single-seed convention. If `internal_only` wins and we later need 3-seed confirmation, a follow-up spec runs all three data seeds on the winning mode.
- Does not implement residual-stream R₀, MLP internal R_m, `resid_mix` fold, skip-stream rotation, or any per-channel-multiplier folds. All of those are deferred to a follow-up spec — see `research/ideas/spinquant-integration-notes.md` for the design analysis.
- Does not attempt the `port_1695` mode — separate research task, needs #1695's diff to be read.
