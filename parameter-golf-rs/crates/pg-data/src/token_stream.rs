use std::path::{Path, PathBuf};

use pg_core::error::{PgError, PgResult};

use crate::shard::DataShard;

/// Sequential token stream over multiple data shards.
/// Advances through shards in sorted order, wrapping around at the end.
///
/// Mirrors the Python `TokenStream` class from train_gpt.py.
pub struct TokenStream {
    files: Vec<PathBuf>,
    current_shard: DataShard,
    file_idx: usize,
    pos: usize,
}

impl TokenStream {
    /// Create a new token stream from a glob pattern (e.g., "fineweb_train_*.bin").
    pub fn from_glob(pattern: &str) -> PgResult<Self> {
        let mut files: Vec<PathBuf> = glob::glob(pattern)
            .map_err(|e| PgError::DataFormat(format!("invalid glob pattern: {}", e)))?
            .filter_map(Result::ok)
            .collect();
        files.sort();

        if files.is_empty() {
            return Err(PgError::DataFormat(format!(
                "no files found for pattern: {}",
                pattern
            )));
        }

        let current_shard = DataShard::open(&files[0])?;

        Ok(Self {
            files,
            current_shard,
            file_idx: 0,
            pos: 0,
        })
    }

    /// Create from a sorted list of file paths.
    pub fn from_files(files: Vec<PathBuf>) -> PgResult<Self> {
        if files.is_empty() {
            return Err(PgError::DataFormat("empty file list".into()));
        }
        let current_shard = DataShard::open(&files[0])?;
        Ok(Self {
            files,
            current_shard,
            file_idx: 0,
            pos: 0,
        })
    }

    /// Take `n` tokens from the stream, advancing through shards as needed.
    pub fn take(&mut self, n: usize) -> PgResult<Vec<u16>> {
        let mut result = Vec::with_capacity(n);
        let mut remaining = n;

        while remaining > 0 {
            let avail = self.current_shard.num_tokens() - self.pos;
            if avail == 0 {
                self.advance_file()?;
                continue;
            }
            let k = remaining.min(avail);
            let chunk = self.current_shard.tokens(self.pos, self.pos + k);
            result.extend_from_slice(chunk);
            self.pos += k;
            remaining -= k;
        }

        Ok(result)
    }

    fn advance_file(&mut self) -> PgResult<()> {
        self.file_idx = (self.file_idx + 1) % self.files.len();
        self.current_shard = DataShard::open(&self.files[self.file_idx])?;
        self.pos = 0;
        Ok(())
    }
}

/// Distributed token loader that partitions data across ranks.
/// Each rank gets a different slice of the global batch.
pub struct DistributedTokenLoader {
    stream: TokenStream,
    rank: usize,
    world_size: usize,
}

impl DistributedTokenLoader {
    pub fn new(pattern: &str, rank: usize, world_size: usize) -> PgResult<Self> {
        Ok(Self {
            stream: TokenStream::from_glob(pattern)?,
            rank,
            world_size,
        })
    }

    /// Get next batch of (input, target) token pairs.
    ///
    /// Returns (x, y) where x[i] = tokens[i..i+seq_len], y[i] = tokens[i+1..i+seq_len+1]
    /// Each GPU gets a disjoint slice of the global batch.
    pub fn next_batch(
        &mut self,
        global_tokens: usize,
        _seq_len: usize,
    ) -> PgResult<(Vec<u16>, Vec<u16>)> {
        let local_tokens = global_tokens / self.world_size;
        let per_rank_span = local_tokens + 1; // +1 for target offset

        // Take enough tokens for all ranks
        let chunk = self.stream.take(per_rank_span * self.world_size)?;

        // Slice for this rank
        let start = self.rank * per_rank_span;
        let local = &chunk[start..start + per_rank_span];

        let x = local[..local.len() - 1].to_vec();
        let y = local[1..].to_vec();

        Ok((x, y))
    }
}

/// Load all validation tokens from shard files matching a pattern.
pub fn load_validation_tokens(pattern: &str) -> PgResult<Vec<u16>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(
        Path::new(pattern)
            .parent()
            .ok_or_else(|| PgError::DataFormat("invalid pattern path".into()))?,
    )?
    .filter_map(Result::ok)
    .map(|e| e.path())
    .filter(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with("fineweb_val_") && n.ends_with(".bin"))
            .unwrap_or(false)
    })
    .collect();
    files.sort();

    if files.is_empty() {
        return Err(PgError::DataFormat("no validation files found".into()));
    }

    let mut all_tokens = Vec::new();
    for file in &files {
        let shard = DataShard::open(file)?;
        all_tokens.extend_from_slice(shard.all_tokens());
    }

    Ok(all_tokens)
}
