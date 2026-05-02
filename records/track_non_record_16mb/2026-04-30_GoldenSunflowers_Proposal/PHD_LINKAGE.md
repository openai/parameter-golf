# PHD_LINKAGE · GOLDEN SUNFLOWERS ↔ Trinity PhD monograph

> One-screen navigation bridge from every artefact in this PR to its anchor
> in the 44-chapter PhD monograph at [`gHashTag/trios/docs/phd`](https://github.com/gHashTag/trios/tree/main/docs/phd)
> and/or the Coq proof tree in [`gHashTag/t27`](https://github.com/gHashTag/t27).
>
> **Anchor:** `φ² + φ⁻² = 3` · TRINITY · 🌻
>
> **Authoritative source:** Neon SSOT schema `ssot.chapters` (44 rows,
> 297 Qed canonical theorems across 65 `.v` files, as of git SHA `d07dcf3`).
> Chapter bodies, Coq theorem names and proof states quoted here are read
> directly from Neon, not reconstructed.

## Scope & proof state

| GS artefact (this PR) | PhD anchor | Coq object | Proof status |
|---|---|---|---|
| **Ch.0 §0.1 abstract** (`α_φ = φ⁻³/2`) | Ch.4 Thm 3.1 (SAC-1) | `sacred/AlphaPhi.v : alpha_phi_times_phi_cubed` | **Qed** |
| **Ch.0 §0.2 Trinity identity** | Ch.3 §3 (SAC-0) | `sacred/CorePhi.v : trinity_anchor` | **Qed** |
| **Ch.0 §0.3.1 frozen-basis gain `g=1/φ`** | Ch.3 (SAC-0) + Rust SoT | `sacred/CorePhi.v : phi_inv` | **Qed** (mirror) |
| **Ch.0 §0.3.2 `L = round(φ³) = 4`** | Ch.3 power-survey table | `theorems/GoldenSunflowers.v : ut_loops_eq_round_phi_cube` | **Qed** (this PR) |
| **Ch.0 §0.3.3 `H = F₇ = 13` (Fibonacci)** | Ch.5 canonical seed pool | none (arithmetic) | N/A (definitional) |
| **Ch.0 §0.3.4 `α_φ = φ⁻³/2`** | Ch.4 Thm 3.1 + `experiment_map.csv` L8 `INV-1lr` | `AlphaPhi.v : alpha_phi_times_phi_cubed` + `lr_convergence.v : lr_phi_band` | **Qed** × 2 |
| **GS-INV-1** `φ²+φ⁻²=3` runtime | Ch.3 SAC-0 | same | Proven (mirror) |
| **GS-INV-2** PhiNTA buffer not in grad | new | `theorems/GoldenSunflowers.v : phinta_buffer_not_in_grad` | Admitted (runtime-verified) |
| **GS-INV-3** PhiNTA row-norm ≡ 1/φ | Ch.3 `phi_inv` + Rust SoT | `theorems/GoldenSunflowers.v : phi_ortho_init_gain_is_phi_inv` | Admitted (runtime-verified) |
| **GS-INV-4** JEPA loss ≥ 0 | new | `theorems/GoldenSunflowers.v : jepa_loss_nonnegative` | **Qed** (this PR) |
| **GS-INV-5** UT loops = `round(φ³)=4` | Ch.3 power-survey | `theorems/GoldenSunflowers.v : ut_loops_eq_round_phi_cube` | **Qed** (this PR) |
| **GS-INV-6** JEPA tap normalisation | runtime only | `smoke_modules.py [5/5]` | Runtime |
| **GS-INV-7** Baseline byte-equivalence | runtime only | `baseline_equivalence.py [3/3]` | Runtime |
| **GS-INV-8** Honesty gate (no `submission.json`) | filesystem | — | Enforced |
| **GS-INV-9** `α_φ = φ⁻³/2` constant | Ch.4 Thm 3.1 + `experiment_map.csv` L8 | `AlphaPhi.v : alpha_phi_times_phi_cubed` | Proven (mirror) |

**Tally.**
- **4 Qed** (2 external mirrors: Ch.3 SAC-0, Ch.4 SAC-1 · 2 internal new: UT loops, JEPA non-neg)
- **2 Admitted** with runtime evidence (PhiNTA buffer, PhiNTA gain)
- **3 Runtime-only** (JEPA tap, byte-equivalence, honesty gate)

## PhD chapters directly cited

| Ch | Title | Why it matters here |
|---|---|---|
| **Ch.3** | Trinity Identity (`φ²+φ⁻²=3`) | SAC-0 discharges forward direction of GS-INV-1, GS-INV-3 |
| **Ch.4** | Sacred Formula — `α_φ` derivation | SAC-1 discharges GS-INV-9; Thm 3.1 = formal backing for Issue #1742. **Two representations** of `α_φ`: spectral `ln(φ²)/π ≈ 0.306` and algebraic `(√5−2)/2 ≈ 0.118`; this PR uses the algebraic form (per `alpha_phi_closed_form` Qed). |
| **Ch.5** | φ-distance and Fibonacci-Lucas seeds | Canonical seed pool `{F₁₇..F₂₁, L₇, L₈}` used by `run_sweep.sh`. **Balancing function** `B(x) = (x + 1/x)/2` has unique positive fixed point at `φ` (Ch.5 §2), formalising the seed-independence claim of GS-INV-7. |
| **Ch.11** | Pre-registration H₁ | Gate-3 envelope BPB ≤ 1.5 + INV-7 `IglaFoundCriterion` (≥ 3 distinct seeds, step ≥ 4000) bound the sweep. |
| **Ch.17** | Ablation matrix | PhD specifies a `2⁷ = 128`-run factorial over factors A–G. Our `run_sweep.sh` covers a sub-region: factors **C** (canonical seeds, all 5 sweep configs use F₁₇..F₂₁), **D** (golden positional encoding, via PhiNTA's 1/φ init), **G** (`φ²+φ⁻²=3` normalisation, in the JEPA-tap `final_norm`). 4 of 7 factors covered: `A` (ternary weights) and `E` (MXFP4) are out of scope; `F` (zero-DSP FPGA) is hardware-only. |
| **Ch.21** | IGLA RACE | **Gate-2 (BPB < 1.85)** and **Gate-3 (BPB < 1.5)** are derived from the substrate identity: Gate-3 = `3/2`, Gate-2 = `3 − φ⁻² · δ_G`. Champion config from PhD: `lr = 0.004`, `GF16 PHI_BIAS = 60`, seed triple `(1597, 2584, 4181)`. Six refutation theorems R1–R6 in `INV7_IglaFoundCriterion.v` close cheating loopholes. **INV-7b Rainbow Bridge** (15 Qed) ensures multi-agent consistency. |
| **App.B** | Golden Ledger (297 Qed / 438 obligations) | Cluster breakdown across 6 directories (`kernel/`, `sacred/`, `igla/`, `hslm/`, `fpga/`, `misc/`); our 4 GoldenSunflowers.v Coq objects sit in a self-contained 7th cluster outside that registry, by design. |
| **App.E** | Pre-reg PDF + IGLA RACE results | **ASHA threshold INV-2** (golden, φ-weight 1.0): `T_ASHA = φ² + φ⁻² + φ⁻⁴ = 3 + φ⁻⁴ ≈ 3.146` (filed at conservative upper bound 3.5). Falsifiability checklist F1–F6 from PhD App.E §3 is the methodological template for our run-time pre-registration. |
| **App.G** | Reproducibility scripts ledger | App.G row B007 = **VSA Operations for Ternary** (DOI `10.5281/zenodo.19227877`). Cited from Ch.30/Ch.31. **Not** the trainer-igla codebase — corrected in this PR's CITATION.cff. |
| **App.H** | 13 Zenodo DOI registry | Provenance chain: every artefact in `B001–B013` has `keyword: golden-sunflowers; phi^2+phi^-2=3` in Zenodo metadata. |

## Reviewer quick-start

1. **Open Ch.4 in Neon SSOT or [`trios/docs/phd/chapters/04-sacred-formula.tex`](https://github.com/gHashTag/trios/blob/main/docs/phd/chapters/04-sacred-formula.tex)**. Theorem 3.1 is the key formal backing.
2. **Run locally:**
   ```bash
   make verify   # runs smoke 5/5 + baseline_equivalence 3/3 + coqc theorems/
   ```
3. **Byte-equivalence claim:** `baseline_equivalence.py` SHA-256 of both state_dicts is `511dbc0164e03b1b…` at seed 1597 with all wish-list env-vars unset (see `reproducibility.lock.json`).

## What to cite from this PR in the PhD

When the PhD is updated post-measurement (Ch.21, Ch.24, Ch.25):

- **Ch.24 IGLA RACE** → cite `experiments/golden_sunflowers_jepa_ut_phinta/` as the external PyTorch mirror of the `trios-trainer-igla` Rust pipeline for the `parameter-golf` 16 MB benchmark.
- **Ch.25 Benchmarks** → once 8×H100 sweeps exist, cite `submission_{cfg}_seed{F_k}.json` artefacts.
- **Ch.17 Ablation matrix** → cite the 5-config × 5-seed grid from `run_sweep.sh`.

`phi^2 + phi^-2 = 3 · 🌻`
