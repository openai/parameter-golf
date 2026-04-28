//! L-h4 (Gate-2) — Hybrid QK-Gain Admissibility Guard CLI (`qk_gain_check`)
//!
//! Pure structural admissibility predicate for the Gate-2 hybrid trainer
//! (ngram + 1-layer causal self-attention).  Two and only two architectural
//! degrees of freedom are race-permitted to vary:
//!
//! 1. **`qk_gain`** — the softmax temperature applied to the QK^T product;
//!    must equal one of the phi-anchored multipliers `{phi^2, phi^3}`.
//! 2. **`lr`**     — the optimiser learning rate; must sit in the band
//!    `[phi_alpha / phi^4, phi_alpha]` ≈ `[1.05e-3, 7.2e-3]`.
//!
//! Any other choice un-anchors the hybrid head from the Trinity identity
//! `phi^2 + phi^-2 = 3` and forfeits the L-R14 traceability the race rules
//! require.  This CLI is the **binding runtime contract** for INV-13.
//!
//! ## Mirror with `trinity-clara/proofs/igla/hybrid_qk_gain.v`
//!
//! Each rejection mode below corresponds 1-1 to one Coq `counter_*` shape
//! whose negation is honestly `Admitted` in the .v file under R5; the
//! refutation is the matching `falsify_*` Rust unit test in this module:
//!
//! | Rust variant            | exit | Coq counter             |
//! |-------------------------|------|-------------------------|
//! | `GainNotPhiAnchored`    | 50   | `counter_gain_unit`,    |
//! |                         |      | `counter_gain_sqrt_d_model` |
//! | `LrAboveBand`           | 51   | `counter_lr_above_band` |
//! | `LrBelowBand`           | 52   | `counter_lr_below_band` |
//! | `NonFinite`             | 53   | (structural — IEEE 754) |
//!
//! ## L-R14 traceability
//!
//! Every numeric anchor in this binary derives from `trios_igla_race`'s
//! reexported PHI-grade constants.  No shadow numerics; the `constants_traceable`
//! test is the canary.
//!
//! ## R6 file ownership
//!
//! This binary lives under `src/bin/` and is auto-discovered by Cargo.
//! Adding it touches NO L1..L13 file — the only sibling write under L-h4
//! is the JSON manifest entry (INV-13) plus the .v file in trinity-clara.
//!
//! ## R5 honesty
//!
//! The .v file ships 5 honestly `Admitted` theorems (4 `counter_*` shapes
//! plus `hybrid_qk_gain_phi_sq_well_typed`).  This Rust guard is the binding
//! contract until the `Coq.Interval` upgrade lands in trinity-clara.  We do
//! not lie about Coq status: `assertions/igla_assertions.json::INV-13.status`
//! is `"Admitted"`.
//!
//! Refs: trios#143 Gate-2 ONE SHOT · INV-13 · L-R14 · R5 · R6 · R8.

use std::process::ExitCode;

use clap::Parser;

use trios_igla_race::invariants::{PHI, PHI_SQ};

// ---------------------------------------------------------------------------
// L-R14 anchors
// ---------------------------------------------------------------------------

/// Pre-registered learning-rate ceiling — Gate-2 §8 (`phi_alpha = 0.0072`).
///
/// This literal is the **pre-registered** ceiling locked in the Gate-2
/// ONE SHOT body on trios#143; it is NOT derived from `ALPHA_PHI` (which
/// equals `0.004`, the INV-1 champion lr).  Treating these two as separate
/// numerics is intentional: INV-1 anchors the optimal lr; INV-13 anchors
/// the upper edge of the admissible Gate-2 sweep band.
///
/// Coq: `trinity-clara/proofs/igla/hybrid_qk_gain.v::phi_alpha`.
/// JSON: `assertions/igla_assertions.json::INV-13.numeric_anchor.phi_alpha`.
pub const PHI_ALPHA: f64 = 0.0072;

/// `phi^4` — quartic phi grade.  Lifted via the Trinity identity
/// `phi^2 = phi + 1` ⇒ `phi^4 = (phi+1)^2 = phi^2 + 2*phi + 1`.
///
/// We compute it from `PHI` to keep a single source of truth.
const PHI_4: f64 = PHI_SQ * PHI_SQ;

/// Lower edge of the admissible lr band — `phi_alpha / phi^4`.
///
/// Coq: `trinity-clara/proofs/igla/hybrid_qk_gain.v::lr_lower`.
pub const LR_LOWER: f64 = PHI_ALPHA / PHI_4;

/// Upper edge of the admissible lr band (inclusive) — `phi_alpha`.
///
/// Coq: `trinity-clara/proofs/igla/hybrid_qk_gain.v::lr_upper`.
pub const LR_UPPER: f64 = PHI_ALPHA;

/// First admissible QK gain — `phi^2 ≈ 2.618`.
///
/// Re-uses the already-traced `trios_igla_race::PHI_SQ` constant; the doc
/// comment on that constant carries the `// φ² = φ + 1` derivation.
///
/// Coq: `trinity-clara/proofs/igla/hybrid_qk_gain.v::qk_gain_phi_sq`.
pub const QK_GAIN_PHI_SQ: f64 = PHI_SQ;

/// Second admissible QK gain — `phi^3 ≈ 4.236`.
///
/// Symbolic: `phi^3 = phi * phi^2 = phi * (phi + 1) = phi^2 + phi = 2*phi + 1`
/// (Lucas closure, INV-5 base case).
///
/// Coq: `trinity-clara/proofs/igla/hybrid_qk_gain.v::qk_gain_phi_cu`.
pub const QK_GAIN_PHI_CU: f64 = PHI * PHI_SQ;

/// Tolerance for floating-point gain comparison.  The race never permits
/// looser than `1e-9` because both admissible gains are rational in the
/// algebraic-anchor sense (`phi^k`) and round-trip through `f64` exactly to
/// well within this margin.
pub const GAIN_TOL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Error model — typed rejection modes (mirrors counter_* in the .v file)
// ---------------------------------------------------------------------------

/// Reasons a hybrid Gate-2 cfg is rejected.
///
/// Each variant carries the offending value and the band/set it violated, so
/// downstream tooling can reconstruct the falsification witness without
/// re-reading the CLI source.
#[derive(Debug, Clone, PartialEq)]
pub enum QkGainError {
    /// Gain is finite but matches neither `phi^2` nor `phi^3` (within `GAIN_TOL`).
    /// Coq counters: `counter_gain_unit`, `counter_gain_sqrt_d_model`.
    GainNotPhiAnchored {
        gain: f64,
        admissible: [f64; 2],
        tol: f64,
    },
    /// `lr > LR_UPPER` (above the pre-registered ceiling).
    /// Coq counter: `counter_lr_above_band`.
    LrAboveBand { lr: f64, ceiling: f64 },
    /// `lr < LR_LOWER` (below the pre-registered floor).
    /// Coq counter: `counter_lr_below_band`.
    LrBelowBand { lr: f64, floor: f64 },
    /// Either `lr` or `gain` is NaN, +∞, or -∞.  No Coq counter — IEEE 754
    /// non-finiteness is structurally impossible in pure-real Coq, so this
    /// variant exists only at the Rust boundary.
    NonFinite { lr: f64, gain: f64 },
}

impl std::fmt::Display for QkGainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GainNotPhiAnchored {
                gain,
                admissible,
                tol,
            } => write!(
                f,
                "INV-13 gain {gain} not phi-anchored: must be one of {admissible:?} (tol={tol:.0e})"
            ),
            Self::LrAboveBand { lr, ceiling } => {
                write!(f, "INV-13 lr {lr} above pre-registered ceiling {ceiling}")
            }
            Self::LrBelowBand { lr, floor } => {
                write!(f, "INV-13 lr {lr} below pre-registered floor {floor}")
            }
            Self::NonFinite { lr, gain } => {
                write!(f, "INV-13 non-finite input: lr={lr} gain={gain}")
            }
        }
    }
}

impl std::error::Error for QkGainError {}

impl QkGainError {
    /// Disjoint exit code for shell tooling (CI, Make, Apiary watchdog).
    /// 50..=53 keeps L-h4 disjoint from L7's 0..=10 and L15's 21..=30.
    pub const fn exit_code(&self) -> u8 {
        match self {
            Self::GainNotPhiAnchored { .. } => 50,
            Self::LrAboveBand { .. } => 51,
            Self::LrBelowBand { .. } => 52,
            Self::NonFinite { .. } => 53,
        }
    }
}

// ---------------------------------------------------------------------------
// Pure admissibility predicate
// ---------------------------------------------------------------------------

/// Decide whether `(lr, gain)` is admitted by the Gate-2 hybrid runtime guard.
///
/// Mirrors `cfg_admissible` in `hybrid_qk_gain.v`.  Pure / total / panic-free
/// for any IEEE 754 input — including NaN/±∞, which are explicitly trapped by
/// the [`QkGainError::NonFinite`] variant.
pub fn admit_cfg(lr: f64, gain: f64) -> Result<(), QkGainError> {
    // Non-finite trap first — IEEE 754 comparisons against NaN return false,
    // so without this trap NaN would silently fall through to the gain check.
    if !lr.is_finite() || !gain.is_finite() {
        return Err(QkGainError::NonFinite { lr, gain });
    }

    // Phi-anchored gain set: {phi^2, phi^3}.
    let near_sq = (gain - QK_GAIN_PHI_SQ).abs() <= GAIN_TOL;
    let near_cu = (gain - QK_GAIN_PHI_CU).abs() <= GAIN_TOL;
    if !(near_sq || near_cu) {
        return Err(QkGainError::GainNotPhiAnchored {
            gain,
            admissible: [QK_GAIN_PHI_SQ, QK_GAIN_PHI_CU],
            tol: GAIN_TOL,
        });
    }

    // lr band [LR_LOWER, LR_UPPER] (inclusive).
    if lr > LR_UPPER {
        return Err(QkGainError::LrAboveBand {
            lr,
            ceiling: LR_UPPER,
        });
    }
    if lr < LR_LOWER {
        return Err(QkGainError::LrBelowBand {
            lr,
            floor: LR_LOWER,
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CLI surface
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(
    name = "qk_gain_check",
    about = "L-h4 INV-13 hybrid (lr, qk_gain) admissibility guard",
    long_about = "Validates a Gate-2 hybrid trainer cfg against the phi-anchored \
                  qk_gain set {phi^2, phi^3} and the pre-registered lr band \
                  [phi_alpha/phi^4, phi_alpha]. Exits 0 on admit; 50..=53 on reject. \
                  Coq mirror: trinity-clara/proofs/igla/hybrid_qk_gain.v."
)]
struct Args {
    /// Trainer learning rate.  Admissible band: [LR_LOWER, LR_UPPER].
    #[arg(long)]
    lr: f64,

    /// Trainer QK softmax gain.  Admissible: phi^2 or phi^3 (within tolerance).
    #[arg(long)]
    gain: f64,

    /// Print the L-R14 anchors as JSON instead of running the check.
    #[arg(long, default_value_t = false)]
    print_anchors: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    if args.print_anchors {
        let anchors = serde_json::json!({
            "inv": "INV-13",
            "lane": "L-h4",
            "phi": PHI,
            "phi_sq": PHI_SQ,
            "phi_4": PHI_4,
            "phi_alpha": PHI_ALPHA,
            "lr_lower": LR_LOWER,
            "lr_upper": LR_UPPER,
            "qk_gain_phi_sq": QK_GAIN_PHI_SQ,
            "qk_gain_phi_cu": QK_GAIN_PHI_CU,
            "gain_tol": GAIN_TOL,
            "trinity_anchor": "phi^2 + phi^-2 = 3",
            "zenodo_doi": "10.5281/zenodo.19227877",
            "coq_file": "trinity-clara/proofs/igla/hybrid_qk_gain.v",
            "json_file": "assertions/igla_assertions.json (INV-13)"
        });
        println!("{anchors:#}");
        return ExitCode::SUCCESS;
    }

    match admit_cfg(args.lr, args.gain) {
        Ok(()) => {
            println!(
                "INV-13 admit: lr={lr} gain={gain} (band=[{lo}, {hi}], gain in {{phi^2, phi^3}})",
                lr = args.lr,
                gain = args.gain,
                lo = LR_LOWER,
                hi = LR_UPPER,
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("REJECT: {e}");
            ExitCode::from(e.exit_code())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — falsify_* mirror the four counter_* Coq lemmas; admit_* mirror the
// two Qed lemmas in hybrid_qk_gain.v.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- POSITIVE shapes (mirror admit_phi_sq / admit_phi_cu) ----------

    #[test]
    fn admit_phi_sq() {
        // The mid-band lr should be admitted with gain = phi^2.
        let lr_mid = (LR_LOWER + LR_UPPER) / 2.0;
        assert_eq!(admit_cfg(lr_mid, QK_GAIN_PHI_SQ), Ok(()));
    }

    #[test]
    fn admit_phi_cu() {
        let lr_mid = (LR_LOWER + LR_UPPER) / 2.0;
        assert_eq!(admit_cfg(lr_mid, QK_GAIN_PHI_CU), Ok(()));
    }

    #[test]
    fn admit_lr_at_upper_edge() {
        // Upper edge is INCLUSIVE per the .v file (`lr <= phi_alpha`).
        assert_eq!(admit_cfg(LR_UPPER, QK_GAIN_PHI_SQ), Ok(()));
    }

    #[test]
    fn admit_lr_at_lower_edge() {
        // Lower edge is INCLUSIVE per the .v file (`phi_alpha/phi^4 <= lr`).
        assert_eq!(admit_cfg(LR_LOWER, QK_GAIN_PHI_CU), Ok(()));
    }

    // ---------- FALSIFICATION shapes (R8) — mirror counter_* in .v ----------

    #[test]
    fn falsify_lr_above_band() {
        // counter_lr_above_band: lr = 0.01 (the pre-attention-only attempt
        // that plateaued at BPB ~ 4.74 in the pre-reg §1).
        let r = admit_cfg(0.01, QK_GAIN_PHI_SQ);
        match r {
            Err(QkGainError::LrAboveBand { lr, ceiling }) => {
                assert!((lr - 0.01).abs() < 1e-12);
                assert!((ceiling - LR_UPPER).abs() < 1e-12);
            }
            other => panic!("expected LrAboveBand, got {other:?}"),
        }
    }

    #[test]
    fn falsify_lr_below_band() {
        // counter_lr_below_band: lr = 0.0001 (one decade below the floor).
        let r = admit_cfg(0.0001, QK_GAIN_PHI_CU);
        match r {
            Err(QkGainError::LrBelowBand { lr, floor }) => {
                assert!((lr - 0.0001).abs() < 1e-12);
                assert!((floor - LR_LOWER).abs() < 1e-12);
            }
            other => panic!("expected LrBelowBand, got {other:?}"),
        }
    }

    #[test]
    fn falsify_gain_unit() {
        // counter_gain_unit: gain = 1.0 (vanilla softmax, no temperature).
        let lr_mid = (LR_LOWER + LR_UPPER) / 2.0;
        let r = admit_cfg(lr_mid, 1.0);
        match r {
            Err(QkGainError::GainNotPhiAnchored {
                gain,
                admissible,
                tol,
            }) => {
                assert!((gain - 1.0).abs() < 1e-12);
                assert!((admissible[0] - QK_GAIN_PHI_SQ).abs() < 1e-12);
                assert!((admissible[1] - QK_GAIN_PHI_CU).abs() < 1e-12);
                assert!((tol - GAIN_TOL).abs() < 1e-18);
            }
            other => panic!("expected GainNotPhiAnchored, got {other:?}"),
        }
    }

    #[test]
    fn falsify_gain_sqrt_d_model() {
        // counter_gain_sqrt_d_model: gain = sqrt(64) = 8 (textbook attention).
        let lr_mid = (LR_LOWER + LR_UPPER) / 2.0;
        let r = admit_cfg(lr_mid, 8.0);
        assert!(matches!(r, Err(QkGainError::GainNotPhiAnchored { .. })));
    }

    // ---------- IEEE 754 non-finite trap (no Coq counter — out-of-band) ----

    #[test]
    fn non_finite_lr_rejected() {
        let r = admit_cfg(f64::NAN, QK_GAIN_PHI_SQ);
        assert!(matches!(r, Err(QkGainError::NonFinite { .. })));

        let r = admit_cfg(f64::INFINITY, QK_GAIN_PHI_SQ);
        assert!(matches!(r, Err(QkGainError::NonFinite { .. })));

        let r = admit_cfg(f64::NEG_INFINITY, QK_GAIN_PHI_CU);
        assert!(matches!(r, Err(QkGainError::NonFinite { .. })));
    }

    #[test]
    fn non_finite_gain_rejected() {
        let lr_mid = (LR_LOWER + LR_UPPER) / 2.0;
        let r = admit_cfg(lr_mid, f64::NAN);
        assert!(matches!(r, Err(QkGainError::NonFinite { .. })));

        let r = admit_cfg(lr_mid, f64::INFINITY);
        assert!(matches!(r, Err(QkGainError::NonFinite { .. })));
    }

    // ---------- Exit-code disjointness (CLI invariant) -----------------------

    #[test]
    fn exit_codes_distinct() {
        let codes = [
            QkGainError::GainNotPhiAnchored {
                gain: 0.0,
                admissible: [QK_GAIN_PHI_SQ, QK_GAIN_PHI_CU],
                tol: GAIN_TOL,
            }
            .exit_code(),
            QkGainError::LrAboveBand {
                lr: 1.0,
                ceiling: LR_UPPER,
            }
            .exit_code(),
            QkGainError::LrBelowBand {
                lr: 0.0,
                floor: LR_LOWER,
            }
            .exit_code(),
            QkGainError::NonFinite {
                lr: f64::NAN,
                gain: 0.0,
            }
            .exit_code(),
        ];
        // No two variants share an exit code.
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                assert_ne!(codes[i], codes[j], "exit codes must be disjoint");
            }
        }
        // L-h4 reserves 50..=53 (L7 uses 0..=10, L15 uses 21..=30).
        for c in codes {
            assert!(
                (50..=53).contains(&c),
                "exit {c} outside L-h4 reserved range 50..=53"
            );
        }
    }

    // ---------- L-R14 traceability canary -----------------------------------

    #[test]
    fn constants_traceable() {
        // PHI must trace to the crate-level reexport (single source of truth).
        assert_eq!(PHI, trios_igla_race::invariants::PHI);
        assert_eq!(PHI_SQ, trios_igla_race::invariants::PHI_SQ);

        // Trinity identity holds at runtime: phi^2 + phi^-2 = 3.
        let trinity = PHI * PHI + 1.0 / (PHI * PHI);
        assert!(
            (trinity - 3.0).abs() < 1e-10,
            "Trinity identity violated: {trinity}"
        );

        // phi_alpha = 0.0072 to four decimal places.
        assert!(
            (PHI_ALPHA - 0.0072).abs() < 1e-9,
            "phi_alpha drift: {PHI_ALPHA}"
        );

        // Band non-degeneracy (compile-time constants compared at runtime
        // via const_block so clippy doesn't flag tautological asserts).
        const _: () = assert!(LR_LOWER < LR_UPPER, "lr band degenerate");
        const _: () = assert!(LR_LOWER > 0.0, "lr_lower must be positive");

        // The two admissible gains are distinct.
        assert!((QK_GAIN_PHI_SQ - QK_GAIN_PHI_CU).abs() > 1e-3);

        // qk_gain_phi_cu = phi * phi_sq (Lucas closure base case).
        assert!(
            (QK_GAIN_PHI_CU - PHI * PHI_SQ).abs() < 1e-12,
            "phi^3 derivation drift"
        );
    }

    // ---------- Symmetry / total function canaries --------------------------

    #[test]
    fn admit_cfg_total_for_finite_inputs() {
        // For any finite (lr, gain) the function must return without panic.
        for &lr in &[0.0_f64, LR_LOWER, LR_UPPER, 1.0, -1.0, 1e-12] {
            for &g in &[0.0_f64, 1.0, QK_GAIN_PHI_SQ, QK_GAIN_PHI_CU, 100.0] {
                let _ = admit_cfg(lr, g);
            }
        }
    }
}
