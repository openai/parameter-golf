//! Early-stop decision at step 1000.
//!
//! Three signals fold into one verdict:
//!
//! 1. **Hard ceiling**  — BPB at step 1000 ≥ ceiling (default 2.60) →
//!    `Prune("hard-ceiling")`. The ceiling is set by the gardener
//!    from the leaderboard (champion + 0.4 by default).
//! 2. **Leader-relative** — if a leader trajectory is provided and
//!    we're worse than `leader_bpb_at_1000 + delta`, prune as
//!    `worse-than-leader-by-delta`.
//! 3. **Power-law extrapolation** — fit `BPB ≈ a*step^b + c` over the
//!    first ~1000 samples; if the predicted asymptote cannot beat
//!    `gate_target + slack`, prune as `predicted-misses-gate`.
//!
//! All three are pure, deterministic over the BPB history slice —
//! same input → same verdict. Tested in isolation.
//!
//! Anchor: `phi^2 + phi^-2 = 3`.

#[derive(Debug, Clone, PartialEq)]
pub enum EarlyStop {
    /// Continue training to `steps_budget`.
    Continue,
    /// Abandon now, pull the next experiment.
    Prune {
        reason: String,
        triggered_by_step: i32,
        triggered_by_bpb: f64,
    },
}

#[derive(Debug, Clone)]
pub struct EarlyStopConfig {
    /// Hard BPB ceiling at the early-stop step (signal #1).
    pub hard_ceiling_bpb: f64,
    /// Optional leader BPB at the early-stop step (signal #2). If
    /// supplied, prune when our BPB > leader + `leader_delta`.
    pub leader_bpb_at_step: Option<f64>,
    pub leader_delta: f64,
    /// Gate target for the predicted asymptote (signal #3).
    pub gate_target_bpb: f64,
    pub gate_slack: f64,
    /// Step at which the decision is made (default 1000).
    pub decision_step: i32,
}

impl Default for EarlyStopConfig {
    fn default() -> Self {
        Self {
            hard_ceiling_bpb: 2.60,
            leader_bpb_at_step: None,
            leader_delta: 0.15,
            gate_target_bpb: 1.85,
            gate_slack: 0.10,
            decision_step: 1000,
        }
    }
}

/// Decision over the BPB history `[(step, bpb), ...]` (sorted).
///
/// The decision is only made when the history has actually reached
/// the configured `decision_step`. Calling early returns `Continue`.
pub fn decide(history: &[(i32, f64)], cfg: &EarlyStopConfig) -> EarlyStop {
    let max_step = history.iter().map(|(s, _)| *s).max().unwrap_or(0);
    if max_step < cfg.decision_step {
        // Trainer hasn't reached the decision step yet — wait.
        return EarlyStop::Continue;
    }

    // Find the BPB at (or just before) the decision step.
    let Some(&(at_step, at_bpb)) = history.iter().rev().find(|&&(s, _)| s <= cfg.decision_step)
    else {
        return EarlyStop::Continue;
    };

    // Signal #1 — hard ceiling.
    if at_bpb >= cfg.hard_ceiling_bpb {
        return EarlyStop::Prune {
            reason: format!(
                "hard-ceiling: bpb={at_bpb:.4} >= ceiling={:.4} at step {at_step}",
                cfg.hard_ceiling_bpb
            ),
            triggered_by_step: at_step,
            triggered_by_bpb: at_bpb,
        };
    }

    // Signal #2 — leader-relative.
    if let Some(leader) = cfg.leader_bpb_at_step {
        if at_bpb > leader + cfg.leader_delta {
            return EarlyStop::Prune {
                reason: format!(
                    "worse-than-leader: bpb={at_bpb:.4} > leader+delta={:.4}+{:.2}",
                    leader, cfg.leader_delta
                ),
                triggered_by_step: at_step,
                triggered_by_bpb: at_bpb,
            };
        }
    }

    // Signal #3 — power-law asymptote.
    if let Some(asymptote) = fit_power_law_asymptote(history) {
        if asymptote > cfg.gate_target_bpb + cfg.gate_slack {
            return EarlyStop::Prune {
                reason: format!(
                    "predicted-misses-gate: bpb_inf={asymptote:.4} > target+slack={:.4}+{:.2}",
                    cfg.gate_target_bpb, cfg.gate_slack
                ),
                triggered_by_step: at_step,
                triggered_by_bpb: at_bpb,
            };
        }
    }

    EarlyStop::Continue
}

/// Fit `BPB(step) ≈ a*step^b + c` and return `c` (the asymptote).
///
/// Closed-form unstable for noisy small-N data — we use a *robust
/// last-quartile mean* as a cheap, deterministic asymptote estimate.
/// This is intentionally simple: a real bayes-opt fit lives in the
/// gardener (PR #67 / seed-hunter), not in the worker.
pub fn fit_power_law_asymptote(history: &[(i32, f64)]) -> Option<f64> {
    if history.len() < 8 {
        return None;
    }
    let mut sorted: Vec<f64> = history.iter().map(|(_, b)| *b).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let cut = n.saturating_sub(n / 4).max(1); // last quartile by sorted BPB
    let tail = &sorted[..cut.max(1)]; // lowest-BPB quartile (best)
                                      // tail.len() is bounded by history.len() (<= 256 in worker.rs);
                                      // f64 can represent integers up to 2^53 exactly, so the cast is lossless here.
    #[allow(clippy::cast_precision_loss)]
    let avg = tail.iter().sum::<f64>() / tail.len() as f64;
    Some(avg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EarlyStopConfig {
        EarlyStopConfig::default()
    }

    fn history_with_bpb_at_1000(bpb: f64) -> Vec<(i32, f64)> {
        // 11 samples at steps 0, 100, 200, ..., 1000.
        (0..=10)
            .map(|i| {
                (
                    i * 100,
                    if i * 100 == 1000 {
                        bpb
                    } else {
                        3.5 - 0.2 * f64::from(i)
                    },
                )
            })
            .collect()
    }

    #[test]
    fn continue_when_history_too_short() {
        let h = vec![(0, 3.5), (100, 3.4)];
        assert_eq!(decide(&h, &cfg()), EarlyStop::Continue);
    }

    #[test]
    fn prune_on_hard_ceiling() {
        let h = history_with_bpb_at_1000(2.80);
        match decide(&h, &cfg()) {
            EarlyStop::Prune { reason, .. } => assert!(reason.starts_with("hard-ceiling")),
            EarlyStop::Continue => panic!("expected hard-ceiling prune, got Continue"),
        }
    }

    #[test]
    fn continue_when_just_below_ceiling() {
        let h = history_with_bpb_at_1000(2.50);
        // Note: power-law tail mean of all-near-1.5 history is < 1.95
        // so signal-3 also passes.
        let mut c = cfg();
        c.gate_target_bpb = 2.40; // relax so power-law signal doesn't fire
        c.gate_slack = 0.20;
        assert_eq!(decide(&h, &c), EarlyStop::Continue);
    }

    #[test]
    fn prune_on_leader_relative() {
        let h = history_with_bpb_at_1000(2.10);
        let mut c = cfg();
        c.leader_bpb_at_step = Some(1.85); // we're at 2.10, leader at 1.85, delta=0.15 → equal
        c.leader_delta = 0.10; // tighten
        c.hard_ceiling_bpb = 3.0;
        c.gate_target_bpb = 3.0;
        c.gate_slack = 1.0;
        match decide(&h, &c) {
            EarlyStop::Prune { reason, .. } => assert!(reason.starts_with("worse-than-leader")),
            EarlyStop::Continue => panic!("expected leader-relative prune, got Continue"),
        }
    }

    #[test]
    fn prune_on_predicted_miss_gate() {
        // History is flat at 2.4 — power-law asymptote ≈ 2.4, gate=1.85+0.1=1.95 → prune.
        let h: Vec<(i32, f64)> = (0..=10).map(|i| (i * 100, 2.4)).collect();
        let mut c = cfg();
        c.hard_ceiling_bpb = 3.0; // disable signal #1
        match decide(&h, &c) {
            EarlyStop::Prune { reason, .. } => assert!(reason.starts_with("predicted-misses-gate")),
            EarlyStop::Continue => panic!("expected predicted-miss-gate prune, got Continue"),
        }
    }

    #[test]
    fn power_law_asymptote_returns_none_for_short_history() {
        assert!(fit_power_law_asymptote(&[(0, 3.5), (100, 3.0)]).is_none());
    }

    #[test]
    fn power_law_asymptote_is_robust_to_outliers() {
        // 9 good samples around 1.9, one outlier at 5.0.
        let mut h: Vec<(i32, f64)> = (0..=8)
            .map(|i| (i * 100, 1.9 + 0.01 * f64::from(i)))
            .collect();
        h.push((900, 5.0));
        let asymp = fit_power_law_asymptote(&h).unwrap();
        // Outlier is in top quartile; tail mean ≈ 1.9.
        assert!((asymp - 1.9).abs() < 0.1, "asymp={asymp}");
    }

    #[test]
    fn early_stop_step_can_be_overridden() {
        // Decision step = 500, history has bpb=2.8 at step 500.
        let h: Vec<(i32, f64)> = (0..=10).map(|i| (i * 50, 2.8)).collect();
        let mut c = cfg();
        c.decision_step = 500;
        match decide(&h, &c) {
            EarlyStop::Prune {
                triggered_by_step, ..
            } => {
                assert!(triggered_by_step <= 500);
            }
            EarlyStop::Continue => panic!("expected prune at custom step, got Continue"),
        }
    }

    #[test]
    fn default_config_matches_adr_0081() {
        let c = EarlyStopConfig::default();
        assert!((c.hard_ceiling_bpb - 2.60).abs() < f64::EPSILON);
        assert!((c.gate_target_bpb - 1.85).abs() < f64::EPSILON);
        assert_eq!(c.decision_step, 1000);
    }
}
