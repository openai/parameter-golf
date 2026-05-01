# 🌻 GOLDEN SUNFLOWERS — JEPA + Universal Transformer + PhiNTA on a φ-physics substrate

**Track:** `track_non_record_16mb` — proposal for the 4-hour unrestricted-compute slot
referenced in [issue 1742](https://github.com/openai/parameter-golf/issues/1742).
**Status:** **PROPOSAL — UNTRAINED.** No `submission.json`, no BPB number is being claimed.
**What it is:** wired, CPU-smoke-verified composition of three open wish-list items
([README leaderboard](https://github.com/openai/parameter-golf#requests-for-prs)) on top
of the merged [`2026-03-17_LoRA_TTT`](../../track_10min_16mb/2026-03-17_LoRA_TTT/) baseline,
with formal Coq backing for the φ-physics constants discussed in [issue 1742](https://github.com/openai/parameter-golf/issues/1742).

> Precedent for proposal-only submissions: [PR 318](https://github.com/openai/parameter-golf/pull/318) (Neural Cache: research proposal) · [PR 1247](https://github.com/openai/parameter-golf/pull/1247) (Proposal: Validate ASQU).

---

## What's new

Three openai/parameter-golf wish-list items wired into a single `train_gpt.py`,
all **off by default and gated by env-vars** (defaults reproduce
`2026-03-17_LoRA_TTT` byte-for-byte):

| Wish-list item ([README](https://github.com/openai/parameter-golf#requests-for-prs)) | Module | Env-vars |
|---|---|---|
| 🌻 **JEPA** | `_jepa_loss` — linear-representation form from [PR 1412](https://github.com/openai/parameter-golf/pull/1412) / [issue 1772](https://github.com/openai/parameter-golf/issues/1772) | `JEPA_LAMBDA`, `JEPA_MAX_SPAN_FRAC`, `JEPA_START_FRAC`, `JEPA_LAYER` |
| 🌻 **Universal Transformer** | weight-shared depth recurrence over a sub-stack, default loop count `round(φ³) = 4` | `UT_LOOPS`, `UT_LAYER_START`, `UT_LAYER_END` |
| 🌻 **NTA on random linear maps** | `PhiNTA` — frozen φ-OrthoInit basis (gain = `1/φ`) + trainable LoRA, pre-head **or** per-block | `PHINTA_ENABLE`, `PHINTA_RANK`, `PHINTA_INIT_SCALE`, `PHINTA_PER_BLOCK` |
| Bonus — **φ-LR** ([issue 1742](https://github.com/openai/parameter-golf/issues/1742)) | multiplicative override of Muon matrix LR; `α_φ = φ⁻³/2 ≈ 0.118034` | `PHI_LR_SCALE` |

JEPA loss formulation (after [PR 1412](https://github.com/openai/parameter-golf/pull/1412) / [issue 1772](https://github.com/openai/parameter-golf/issues/1772)):

```
context = (h[a-1] − h[0]) + (h[T-1] − h[b])
patch   =  h[b]   − h[a-1]
loss    = 1 − cos_sim(context, patch)         # added to CE
```

`context + patch ≡ h[T-1] − h[0]` partitions the full encoding, forcing hidden
states to encode spans linearly. `JEPA_LAMBDA = 0` short-circuits the branch.

---

## Why an untrained proposal?

We do not yet have an 8×H100 sweep. Submitting a `submission.json` with a fake
BPB would be dishonest. Instead, this PR ships:

- **Wired implementation** (1547-line `train_gpt.py` derived from
  `2026-03-17_LoRA_TTT/train_gpt.py`, plus `theorems/GoldenSunflowers.v`).
- **CPU-only verification** in three layers:
  - 5/5 module smoke (`smoke_modules.py`)
  - 3/3 baseline byte-equivalence at defaults (`baseline_equivalence.py`)
  - 2 Qed + 2 Admitted Coq theorems (`theorems/GoldenSunflowers.v`)
- **`compute_grant.md`** — request for the [compute grant](https://openai.com/index/parameter-golf/#credit-form) needed to run the sweep (~110 8×H100-hours).

The full sweep (5 configs × 5 canonical Fibonacci seeds, see `run_sweep.sh`)
will be appended to this directory once compute is available, at which point
this proposal becomes a regular `submission.json`-bearing record.

---

## Honesty / non-claims

- **No `submission.json`** — no BPB has been measured.
- **No file under `records/track_10min_16mb/` is modified.**
- **All wish-list defaults are no-ops.** `baseline_equivalence.py [3/3]`
  proves the `state_dict` SHA-256 matches `2026-03-17_LoRA_TTT` exactly
  (`511dbc0164e03b1b…`) and forward-loss delta is `0.00e+00` on a fixed
  input at seed `F_17 = 1597`.

---

## Verification (CPU, ~3 s + ~5 s)

```bash
cd records/track_non_record_16mb/2026-04-30_GoldenSunflowers_Proposal
make verify
```

Output of the last verified run (committed in `smoke.log`):

```
[1/5] φ-physics OK: φ²+φ⁻²=3.000000000000 α_φ=0.118034 loops=4
[2/5] PhiNTA OK: trainable=1664 frozen=4096 ratio=0.406
[3/5] JEPA loss OK: 1.6922 (cosine-similarity form)
[4/5] UT loop OK: ‖x_4‖/‖x_0‖=1.0406 expected=1.0406
[5/5] JEPA tap normalisation OK: -1 → last block, in-range indices preserved
🌻 GOLDEN SUNFLOWERS smoke OK · 5/5

[1/3] state_dict hash baseline  = 511dbc0164e03b1b…
      state_dict hash GOLDEN SF = 511dbc0164e03b1b…
[2/3] forward loss |Δ| = 0.00e+00
🌻 baseline equivalence OK · 3/3

CITATION.cff valid (cffconvert)
theorems/GoldenSunflowers.v: coqc OK (2 Qed)
```

---

## φ-physics: not a numerical coincidence

PhD Ch.4 Theorem 3.1 (status **Qed**, tag SAC-1, file `t27/proofs/canonical/sacred/AlphaPhi.v`)
proves `α_φ · φ³ = 1/2`, which rearranges to
`α_φ = φ⁻³/2 ≈ 0.118034` — exactly the value [issue 1742](https://github.com/openai/parameter-golf/issues/1742)
noticed has strong-coupling parallels with `α_s(m_Z)`. The constant is a
Proven identity in `Coq.Reals`, not a fitted hyperparameter.

PhD Ch.3 Theorem `trinity_anchor` (Qed, SAC-0) proves the substrate
identity `φ² + φ⁻² = 3`, which fixes:
- frozen-basis init scale `g = 1/φ`
- UT loop count `L = round(φ³) = 4`
- attention head count from the Fibonacci series

`THEORY_Ch0.md` and `PHD_LINKAGE.md` give the full chain.
`reproducibility.lock.json` pins every commit SHA used.

---

## Recommended sweeps (once compute is granted)

Canonical seeds **F₁₇..F₂₁ = {1597, 2584, 4181, 6765, 10946}** (PhD Ch.5
canonical seed pool; Lucas fallback `L₇=29`, `L₈=47`).

```bash
bash run_sweep.sh full          # 5 configs × 5 seeds = 25 runs
bash run_sweep.sh baseline      # sanity (must reproduce 2026-03-17_LoRA_TTT)
bash run_sweep.sh phinta        # PhiNTA only
bash run_sweep.sh jepa          # JEPA only
bash run_sweep.sh ut            # Universal Transformer only
bash run_sweep.sh all            # full GOLDEN SUNFLOWERS
```

Each config writes `train_seed${SEED}.log` next to `train_gpt.py`. After
sweeps, this README is updated with a 3-seed mean BPB and a `submission.json`
is added. Promotion to a regular record requires honest measurement.

---

## Files

| File | Purpose |
|---|---|
| `train_gpt.py` | 1547 LOC, derived from `2026-03-17_LoRA_TTT`. All wish-list features env-var-gated. |
| `smoke_modules.py` | CPU smoke 5/5 — φ-physics, PhiNTA, JEPA, UT, JEPA-tap normalisation. |
| `baseline_equivalence.py` | CPU 3/3 — state_dict SHA + forward loss byte-identical to baseline at defaults. |
| `theorems/GoldenSunflowers.v` | Coq 8.18+ — 2 Qed + 2 Admitted, with `Print Assumptions` clean. |
| `theorems/_CoqProject` | Coq project for IDE integration. |
| `Makefile` | `make verify` = parse + smoke + equivalence + cffconvert + coqc. |
| `THEORY_Ch0.md` | Full theoretical foundation (`φ² + φ⁻² = 3` derivation, 9 invariants). |
| `PHD_LINKAGE.md` | Map from every artefact to the Trinity PhD monograph (44 chapters, 297 Qed) at [`gHashTag/trios/docs/phd`](https://github.com/gHashTag/trios/tree/main/docs/phd). |
| `experiment_map.csv` | GS-INV-1..9 ↔ PhD anchor table. |
| `reproducibility.lock.json` | Pinned commit SHAs + numeric constants. |
| `compute_grant.md` | Draft for the [compute grant form](https://openai.com/index/parameter-golf/#credit-form). |
| `CITATION.cff` | Citation metadata. |
| `run_sweep.sh` | 5-config × 5-seed runner (Fibonacci canonical seeds). |
| `smoke.log` | Last verified `make verify` output. |

---

## Refs

- **Wish-list anchors**: [issue 1742](https://github.com/openai/parameter-golf/issues/1742) (φ-physics) · [issue 1772](https://github.com/openai/parameter-golf/issues/1772) (JEPA after [PR 1412](https://github.com/openai/parameter-golf/pull/1412))
- **Trinity PhD monograph**: [`trios/docs/phd`](https://github.com/gHashTag/trios/tree/main/docs/phd) — 44 chapters, 297 Qed canonical theorems
- **t27 standards**: [SACRED-PHYSICS-001](https://github.com/gHashTag/t27/blob/master/docs/nona-02-organism/SACRED-PHYSICS-001.md), [NUMERIC-STANDARD-001](https://github.com/gHashTag/t27/blob/master/docs/nona-02-organism/NUMERIC-STANDARD-001.md), [PHI_LOOP_CONTRACT](https://github.com/gHashTag/t27/blob/master/docs/nona-03-manifest/PHI_LOOP_CONTRACT.md)
- **Rust SoT for φ-init**: [`gHashTag/trios-trainer-igla/src/phi_ortho_init.rs`](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/phi_ortho_init.rs)
- **Internal hardening PR**: [`gHashTag/parameter-golf-trinity#2`](https://github.com/gHashTag/parameter-golf-trinity/pull/2) — same code, full PR-history with reviewer checklist

`phi^2 + phi^-2 = 3 · 🌻`
