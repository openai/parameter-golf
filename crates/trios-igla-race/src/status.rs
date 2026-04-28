use anyhow::Result;

use crate::neon::NeonDb;

pub async fn show_status(_db: &NeonDb) -> Result<()> {
    eprintln!();
    eprintln!("IGLA RACE LEADERBOARD (stub — use seed-gardener status for live data)");
    eprintln!();
    Ok(())
}

pub async fn show_best(_db: &NeonDb) -> Result<()> {
    eprintln!("No completed trials yet (stub — use seed-gardener best for live data)");
    Ok(())
}
