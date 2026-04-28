/// L-R14 anchor: BPB target for IGLA RACE victory (1.5).
/// Must match `hive_automaton::BPB_VICTORY_TARGET`.
pub const IGLA_TARGET_BPB: f64 = 1.5;

pub mod asha;
pub mod ema;
pub mod hive_automaton;
pub mod invariants;
pub mod lessons;
pub mod neon;
pub mod pull_queue;
pub mod race;
pub mod rungs;
pub mod sampler;
pub mod status;
pub mod victory;

pub use neon::{spawn_heartbeat, DashboardMeta, LessonEntry, NeonDb};

pub use pull_queue::{BpbSample, Experiment, ExperimentConfig, PullQueueDb, SelfDecision};
