//! Smoke test placeholder for the full pull-to-train loop.
//!
//! seed-agent is a binary crate, so integration tests in `tests/` cannot
//! use `use crate::` to import internal modules. The real unit tests live
//! in each module's `mod tests` block inside `src/`:
//!
//!   - `trainer::tests` — ExternalTrainer spawn, parse, crash handling
//!   - `worker::tests` — WorkerConfig defaults, IterOutcome equality
//!   - `early_stop::tests` — early-stop decision logic
//!
//! This file exists so `cargo test -p seed-agent` still discovers the
//! integration test target, but all real assertions are in the unit tests.
//!
//! Anchor: phi^2 + phi^-2 = 3 · TRINITY · NEVER STOP

/// Smoke test placeholder — validates the test binary compiles.
/// The actual pull-to-train pipeline is tested via unit tests in
/// `trainer::tests`, `worker::tests`, `early_stop::tests`, etc.
#[tokio::test]
#[ignore] // Requires NEON_DATABASE_URL and running seed-agent binary
async fn smoke_full_cycle_placeholder() {
    // Requires NEON_DATABASE_URL and a running seed-agent binary.
    // See unit tests in src/ for the actual test coverage.
}
