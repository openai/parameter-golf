# Distilled Guidance — Competition Profile (post `skc_engram_conv_20260418_031037`)

Scope: concrete edits to `train_gpt_verbose.py`, `run_god_tier_skc.sh`, and the matrix harness, derived from the 16-run convergence matrix. All loss deltas are in a ~5.965–6.114 band at 242 steps (10-min horizon), so changes here are targeted at (a) removing confirmed regressions, (b) locking in the mild SKC win, (c) neutralizing the untrained-engram footgun, and (d) fixing the A0 comparability bug.

---

## 1. Confirmed signals (what the matrix actually showed)

| Signal | Evidence | Confidence |
|---|---|---|
| `HEAD_LR_MULT=0.7` is toxic at 10-min horizon | A4 = 6.1142 (+2.22% vs B0), C3 = 6.1104 (+2.2% vs C0). Two independent tracks. | **High** |
| SKC amp/LR combo (A3) is neutral-to-mildly-positive | A3 = 5.9776 vs B0 = 5.9817 (−0.007%). Inside noise, but consistent in C0/C2. | **Medium** |
| Engram ON underperforms engram OFF at this horizon | B4 (OFF) = 5.9653 beats every ON variant in track B. C4 (OFF) = 5.9683 beats C0/C1/C2/C3. | **High** |
| Engram gate is untrained | `eng_gate_mean ≈ 0.500 ± 0.03` across all 16 runs; `eng_causal_delta ≈ 2.4` is the bias of an uninitialized gate, not recall signal. | **High** |
| Coupling modes (`gate`/`bias`) give no measurable lift | C1, C2 ≤ 0.1% from C0. | **Medium** |
| Batch-token size dominates the loss comparison | A0 = 5.4625 with `TRAIN_BATCH_TOKENS=32768` (vs 49152 elsewhere) and stopped 10 steps earlier. Not a win; artifact. | **High** |
| No numerical instability | No NaNs; `gw_ratio_mean ≈ 2.5k` stable; `ternary_zeros ≈ 0.30` stable. | **High** |

---

## 2. Edits to `train_gpt_verbose.py`

### 2a. Defaults worth flipping

Current defaults at the `Hyperparameters` dataclass (≈ lines 359–372, 433, 4517):

```
engram_taper_start = 0.4       # line 359
engram_taper_end   = 0.8       # line 360
eng_write_every    = 1         # line 362
eng_to_skc_mode    = 'off'     # line 363
skc_residual_scale_init = 0.15 # line 369
skc_amp_ramp_fraction   = 0.3  # line 370
skc_struct_lr_mult      = 1.5  # line 371
head_lr_mult            = 1.0  # line 372
recurrence_start_fraction = 0.0 # line 433
SCALES_LR_MULT = 3.0           # line 4517 (env default)
```

**Recommended change block inside the competition-profile gate (around line 216 — the `_unset('BIGRAM_HASH_ENABLED')` / `competition_profile` branch):**

1. **Lock in the A3 SKC combo** when `competition_profile=1` — these are already the dataclass defaults, but make them sticky in profile mode so a stray env does not silently regress:
   ```python
   if _unset('SKC_RESIDUAL_SCALE_INIT'): args.skc_residual_scale_init = 0.15
   if _unset('SKC_AMP_RAMP_FRACTION'):   args.skc_amp_ramp_fraction   = 0.3
   if _unset('SKC_STRUCT_LR_MULT'):      args.skc_struct_lr_mult      = 1.5
   ```
2. **Guardrail `HEAD_LR_MULT`** — reject values below ~0.9 in competition profile unless `ALLOW_HEAD_LR_UNDERSCALE=1` is set. Add next to the existing validator at line 518:
   ```python
   if args.competition_profile and args.head_lr_mult < 0.9 and not _e('ALLOW_HEAD_LR_UNDERSCALE', 0, bool):
       raise ValueError(f'HEAD_LR_MULT={args.head_lr_mult} regresses loss +2.2% at 10-min horizon (A4/C3). '
                        'Set ALLOW_HEAD_LR_UNDERSCALE=1 to override.')
   ```
3. **Gate engram-ON behind a trainedness check** — at present `BIGRAM_HASH_ENABLED=1` under competition profile (line ≈ 217). Since B4/C4 both beat engram-ON, either:
   - (preferred, low-risk): flip the competition default to `BIGRAM_HASH_ENABLED=0` until the gate actually learns, **or**
   - (bolder): keep engram ON but warm-start the gate away from 0.5. Add a bias init in the engram gate module: initialize pre-sigmoid bias to `+1.5` so the gate starts ≈ 0.82 and is forced to learn downward, breaking the symmetric stuck point. Wire via `ENG_GATE_BIAS_INIT` (default `1.5` in competition profile, `0.0` otherwise).
4. **Tail taper window** — `ENGRAM_TAPER_START/END = 0.4/0.8` default is too early; the B1/B3 runs used `0.95/0.99` and did not regress. Widen profile default to `0.9/0.99` so engram contribution dominates the middle of training and fades before ternarization lock.
5. **`SCALES_LR_MULT=3.0` (line 4517) was not in the sweep** — keep as-is, but add a TODO: this is a candidate for the next matrix since it directly interacts with `skc_struct_lr_mult=1.5`.

### 2b. Probe / diagnostics hardening

- The engram gate mean stayed at 0.500 ± 0.03 for *every* run, including the 3-step `P0_dryrun`. That is consistent with a gate that never receives gradient. **Action:** add an assertion in `ENG_GATE_LOG` path — after N warmup steps (say 100), if `|gate_mean − 0.5| < 0.02` and `gate_std < 0.01`, emit a `WARN: engram gate may be detached from graph` line. This would have caught the issue in the first run.
- `eng_causal_delta` reported ≈ 2.4 across the board — treat as untrustworthy until the gate unsticks. Consider logging `eng_causal_delta_normalized = delta / (1 − |gate_mean − 0.5| * 2)` so a dead gate reads 0.

---

## 3. Edits to `run_god_tier_skc.sh`

### 3a. Bugfix — batch-token auto-sizing

`pick_train_batch_tokens()` at line 41 picks batch size from *current* free VRAM. When a prior run has not fully released memory (A0 hit this), the first run in a matrix gets a smaller batch than siblings. That is exactly what invalidated A0.

Patch:
1. Add explicit barrier before measurement — after the existing `sleep 2` on line 27, add:
   ```bash
   nvidia-smi --gpu-reset 2>/dev/null || true
   for _ in 1 2 3; do
     sleep 3
     FREE_CHECK="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits $SMI_ARGS | head -n "${NPROC}" | awk 'NR==1{m=$1}$1<m{m=$1}END{print int(m)}')"
     if (( FREE_CHECK >= FREE_MB_EXPECTED_MIN )); then break; fi
   done
   ```
   with `FREE_MB_EXPECTED_MIN` defaulting to 70000 on H100.
2. **Matrix-mode override:** when `MATRIX_LOCK_BATCH_TOKENS` is set (e.g. to 49152), skip `pick_train_batch_tokens` entirely and use the locked value. The matrix harness must set this so A0 and A4 are comparable by construction.

Add near line 63:
```bash
if [[ -n "${MATRIX_LOCK_BATCH_TOKENS:-}" ]]; then
  TRAIN_BATCH_TOKENS_AUTO="${MATRIX_LOCK_BATCH_TOKENS}"
fi
```

### 3b. Promote A3 combo into the default `COMMON_ENV`

Add explicit defaults to the `COMMON_ENV=( ... )` block (line 93):
```bash
SKC_RESIDUAL_SCALE_INIT="${SKC_RESIDUAL_SCALE_INIT:-0.15}"
SKC_AMP_RAMP_FRACTION="${SKC_AMP_RAMP_FRACTION:-0.3}"
SKC_STRUCT_LR_MULT="${SKC_STRUCT_LR_MULT:-1.5}"
HEAD_LR_MULT="${HEAD_LR_MULT:-1.0}"
ENGRAM_TAPER_START="${ENGRAM_TAPER_START:-0.9}"
ENGRAM_TAPER_END="${ENGRAM_TAPER_END:-0.99}"
ENG_WRITE_EVERY="${ENG_WRITE_EVERY:-1}"
```
This makes the shell script self-documenting about which knobs the matrix validated, instead of relying on dataclass defaults that a future edit could silently change.

### 3c. Engram ON/OFF switch at the harness layer

Given B4/C4 both beat their engram-ON siblings, the harness should expose a **single toggle** rather than leaving `ENGRAM_COMPETITION_ENABLED=1` hard-wired (line 133). Replace with:
```bash
ENGRAM_COMPETITION_ENABLED="${ENGRAM_COMPETITION_ENABLED:-0}"   # default OFF until gate-trainedness verified
BIGRAM_HASH_ENABLED="${BIGRAM_HASH_ENABLED:-${ENGRAM_COMPETITION_ENABLED}}"
```
When the gate-bias-init fix from §2a.3 lands and a follow-up matrix confirms engram-ON ≥ engram-OFF, flip the default back to 1.

### 3d. Compile mode note

The matrix used `COMPILE_MODE=none`. Production runs use `COMPILE_MODE=reduce-overhead` (line 113). Any A3-vs-baseline promotion claim must be re-verified **once under compile=reduce-overhead** before being trusted — Inductor reorderings can change the effective LR schedule visibility.

---

## 4. Edits to `scratch/run_skc_engram_convergence_matrix.sh` (or successor)

- Set `export MATRIX_LOCK_BATCH_TOKENS=49152` at the top of the script so no run ever floats its batch size.
- Add a post-run assertion step after each run:
  ```bash
  python3 scratch/summarize_skc_matrix.py --run-dir "skc_matrix_${RUN}" \
    --assert-gate-trained --assert-no-nan --assert-batch-tokens 49152 \
    || { echo "RUN ${RUN} failed invariant"; exit 4; }
  ```
  so a stuck-gate or wrong-batch run aborts the matrix instead of being discovered post-hoc.
- For the *next* matrix, rotate in the knobs that were **not** swept this round and are candidates for real signal at this horizon:
  1. `SCALES_LR_MULT ∈ {1.0, 3.0, 6.0}` (currently 3.0 env default vs 1.0 matrix default — disagreement worth resolving).
  2. `RECURRENCE_START_FRACTION ∈ {0.0, 0.35, 0.6}` — 0.35 was fixed across all 16 runs.
  3. `ENG_GATE_BIAS_INIT ∈ {0.0, +1.5, −1.5}` — only meaningful once the gate is unstuck.
  4. Gate-trainedness repair runs with 3× horizon (≥ 720 steps) so engram-ON gets a fair shot.

---

## 5. Go / no-go summary for this matrix

- **GO (implement now):**
  1. Guardrail against `HEAD_LR_MULT < 0.9` in competition profile.
  2. Lock A3's SKC combo as competition-profile defaults in both the Python and the shell paths.
  3. `MATRIX_LOCK_BATCH_TOKENS` in the matrix harness + VRAM-settle loop in `run_god_tier_skc.sh`.
  4. Add gate-stuck warning in ENG_GATE_LOG.
  5. Flip competition default to engram-OFF until §2a.3 lands.

- **NO-GO (do not promote):**
  1. Any engram coupling mode (`gate`/`bias`) — no measurable lift.
  2. Any claim that A0's 5.4625 is a win — it's a batch-size artifact.
  3. Any engram taper/sparsity tuning as a win — engram itself is below floor.

- **Re-run required:**
  1. A0 with `TRAIN_BATCH_TOKENS=49152` to confirm ordering.
  2. A3 under `COMPILE_MODE=reduce-overhead` before calling it the competition recipe.
  3. Engram-ON variants after gate-bias-init fix, on a ≥720-step horizon.

---

## 6. One-liner for the next experiment pass

> "Land the three guardrails (head-LR floor, locked batch tokens, gate-stuck warning), ship A3 combo as profile defaults with engram OFF, then re-run A3/B4/C4 triad under compile=reduce-overhead to confirm; only after that, reopen the engram track with a warm-started gate on a 3× horizon."
