use std::path::PathBuf;

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

    /// Take `n` tokens into a reusable u32 buffer.
    ///
    /// Record-shaped Rust/CUDA runs immediately upload u32 token IDs, so this
    /// avoids per-step u16 batch allocation plus a second u16->u32 collect.
    pub fn take_u32_into(&mut self, n: usize, out: &mut Vec<u32>) -> PgResult<()> {
        out.clear();
        out.reserve(n.saturating_sub(out.capacity()));
        let mut remaining = n;

        while remaining > 0 {
            let avail = self.current_shard.num_tokens() - self.pos;
            if avail == 0 {
                self.advance_file()?;
                continue;
            }
            let k = remaining.min(avail);
            let chunk = self.current_shard.tokens(self.pos, self.pos + k);
            out.extend(chunk.iter().map(|&tok| tok as u32));
            self.pos += k;
            remaining -= k;
        }

        Ok(())
    }

    /// Take `n` input tokens and the shifted `n` target tokens directly into
    /// reusable u32 buffers.
    ///
    /// This reads `n + 1` stream tokens but avoids the temporary local span
    /// allocation used by `take(n + 1)` plus slicing/copying into input/target.
    pub fn take_shifted_u32_into(
        &mut self,
        n: usize,
        input: &mut Vec<u32>,
        target: &mut Vec<u32>,
    ) -> PgResult<()> {
        input.clear();
        target.clear();
        if n == 0 {
            return Ok(());
        }
        input.reserve(n.saturating_sub(input.capacity()));
        target.reserve(n.saturating_sub(target.capacity()));

        let mut remaining = n + 1;
        let mut consumed = 0usize;
        while remaining > 0 {
            let avail = self.current_shard.num_tokens() - self.pos;
            if avail == 0 {
                self.advance_file()?;
                continue;
            }
            let k = remaining.min(avail);
            let chunk = self.current_shard.tokens(self.pos, self.pos + k);

            let input_end = k.min(n.saturating_sub(consumed));
            input.extend(chunk[..input_end].iter().map(|&tok| tok as u32));

            let target_start = usize::from(consumed == 0);
            if target_start < k {
                target.extend(chunk[target_start..].iter().map(|&tok| tok as u32));
            }
            self.pos += k;
            remaining -= k;
            consumed += k;
        }

        debug_assert_eq!(input.len(), n);
        debug_assert_eq!(target.len(), n);
        Ok(())
    }

    /// Take the contiguous `n + 1` token span used to form shifted input and
    /// target windows. CUDA record runs can upload this compact u16 span and
    /// expand/shift into u32 input/target buffers on device.
    pub fn take_shifted_span_u16_into(&mut self, n: usize, span: &mut Vec<u16>) -> PgResult<()> {
        span.clear();
        span.reserve(n.saturating_add(1).saturating_sub(span.capacity()));
        self.take_into(n + 1, span)
    }

    fn take_into(&mut self, n: usize, out: &mut Vec<u16>) -> PgResult<()> {
        let mut remaining = n;
        while remaining > 0 {
            let avail = self.current_shard.num_tokens() - self.pos;
            if avail == 0 {
                self.advance_file()?;
                continue;
            }
            let k = remaining.min(avail);
            let chunk = self.current_shard.tokens(self.pos, self.pos + k);
            out.extend_from_slice(chunk);
            self.pos += k;
            remaining -= k;
        }
        Ok(())
    }

    /// Skip `n` tokens in the stream without allocating.
    pub fn skip(&mut self, n: usize) -> PgResult<()> {
        let mut remaining = n;
        while remaining > 0 {
            let avail = self.current_shard.num_tokens() - self.pos;
            if avail == 0 {
                self.advance_file()?;
                continue;
            }
            let k = remaining.min(avail);
            self.pos += k;
            remaining -= k;
        }
        Ok(())
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

        // Skip tokens for lower ranks, then take only this rank's span.
        // This avoids reading all ranks' data on every GPU.
        let skip_tokens = self.rank * per_rank_span;
        if skip_tokens > 0 {
            self.stream.skip(skip_tokens)?;
        }
        let local = self.stream.take(per_rank_span)?;

        // Skip remaining ranks' tokens to keep stream position consistent
        let remaining_skip = (self.world_size - self.rank - 1) * per_rank_span;
        if remaining_skip > 0 {
            self.stream.skip(remaining_skip)?;
        }

        let x = local[..local.len() - 1].to_vec();
        let y = local[1..].to_vec();

        Ok((x, y))
    }

    /// Fill reusable u32 input/target buffers for this rank's next batch.
    pub fn next_batch_u32_into(
        &mut self,
        global_tokens: usize,
        input: &mut Vec<u32>,
        target: &mut Vec<u32>,
    ) -> PgResult<()> {
        let local_tokens = global_tokens / self.world_size;
        let per_rank_span = local_tokens + 1;

        let skip_tokens = self.rank * per_rank_span;
        if skip_tokens > 0 {
            self.stream.skip(skip_tokens)?;
        }
        self.stream
            .take_shifted_u32_into(local_tokens, input, target)?;

        let remaining_skip = (self.world_size - self.rank - 1) * per_rank_span;
        if remaining_skip > 0 {
            self.stream.skip(remaining_skip)?;
        }
        Ok(())
    }

    /// Fill a reusable u16 span containing this rank's local input tokens plus
    /// the one-token target lookahead. The caller can build input/target pairs
    /// on device without uploading two full u32 buffers.
    pub fn next_batch_shifted_span_u16_into(
        &mut self,
        global_tokens: usize,
        span: &mut Vec<u16>,
    ) -> PgResult<()> {
        let local_tokens = global_tokens / self.world_size;
        let per_rank_span = local_tokens + 1;

        let skip_tokens = self.rank * per_rank_span;
        if skip_tokens > 0 {
            self.stream.skip(skip_tokens)?;
        }
        self.stream.take_shifted_span_u16_into(local_tokens, span)?;

        let remaining_skip = (self.world_size - self.rank - 1) * per_rank_span;
        if remaining_skip > 0 {
            self.stream.skip(remaining_skip)?;
        }
        Ok(())
    }
}

/// Load all validation tokens from shard files matching a pattern.
pub fn load_validation_tokens(pattern: &str) -> PgResult<Vec<u16>> {
    load_validation_tokens_limited(pattern, None)
}

/// Load validation tokens up to an optional prefix limit.
///
/// This preserves legal score-first ordering for proxy/smoke evaluation while
/// avoiding full-shard materialization when only a small eval prefix is needed.
pub fn load_validation_tokens_limited(
    pattern: &str,
    max_tokens: Option<usize>,
) -> PgResult<Vec<u16>> {
    load_validation_u16_shards_limited(pattern, max_tokens, false)
}

fn load_validation_u16_shards_limited(
    pattern: &str,
    max_tokens: Option<usize>,
    allow_byte_sidecars: bool,
) -> PgResult<Vec<u16>> {
    let mut files: Vec<PathBuf> = glob::glob(pattern)
        .map_err(|e| PgError::DataFormat(format!("invalid glob pattern: {}", e)))?
        .filter_map(Result::ok)
        .filter(|path| allow_byte_sidecars || !is_byte_sidecar_path(path))
        .collect();
    files.sort();

    if files.is_empty() {
        return Err(PgError::DataFormat(format!(
            "no validation files found for pattern: {}",
            pattern
        )));
    }

    let mut all_tokens = Vec::new();
    if let Some(limit) = max_tokens {
        all_tokens.reserve(limit);
    }
    for file in &files {
        let shard = DataShard::open(file)?;
        let remaining = max_tokens
            .map(|limit| limit.saturating_sub(all_tokens.len()))
            .unwrap_or_else(|| shard.num_tokens());
        if remaining == 0 {
            break;
        }
        let take = remaining.min(shard.num_tokens());
        all_tokens.extend_from_slice(shard.tokens(0, take));
    }

    Ok(all_tokens)
}

fn is_byte_sidecar_path(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.contains("_bytes_") || name.contains("val_bytes"))
        .unwrap_or(false)
}

/// Load a validation byte-count sidecar aligned one-to-one with validation
/// token shards. CaseOps/SP8192 validation uses this instead of deriving byte
/// counts from token surfaces after lossless case transforms.
pub fn load_validation_byte_sidecar_limited(
    pattern: &str,
    max_tokens: Option<usize>,
) -> PgResult<Vec<f32>> {
    let raw = load_validation_u16_shards_limited(pattern, max_tokens, true)?;
    Ok(raw.into_iter().map(|v| v as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_shard(path: &std::path::Path, values: &[u16]) {
        let mut file = std::fs::File::create(path).unwrap();
        let mut header = [0i32; 256];
        header[0] = 20240520;
        header[1] = 1;
        header[2] = values.len() as i32;
        file.write_all(bytemuck::cast_slice(&header)).unwrap();
        file.write_all(bytemuck::cast_slice(values)).unwrap();
    }

    #[test]
    fn byte_sidecar_loader_preserves_alignment_and_limit() {
        let path = std::env::temp_dir().join(format!(
            "pg_caseops_bytes_{}_{}.bin",
            std::process::id(),
            17
        ));
        write_test_shard(&path, &[0, 5, 1, 3]);
        let pattern = path.to_string_lossy().to_string();

        let all = load_validation_byte_sidecar_limited(&pattern, None).unwrap();
        assert_eq!(all, vec![0.0, 5.0, 1.0, 3.0]);

        let limited = load_validation_byte_sidecar_limited(&pattern, Some(3)).unwrap();
        assert_eq!(limited, vec![0.0, 5.0, 1.0]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn validation_token_loader_ignores_byte_sidecars_in_broad_glob() {
        let dir = std::env::temp_dir().join(format!(
            "pg_val_sidecar_filter_{}_{}",
            std::process::id(),
            23
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let token_path = dir.join("fineweb_val_000.bin");
        let sidecar_path = dir.join("fineweb_val_bytes_000.bin");
        write_test_shard(&token_path, &[10, 11]);
        write_test_shard(&sidecar_path, &[1, 2, 3, 4]);

        let pattern = dir.join("fineweb_val_*.bin").to_string_lossy().to_string();
        let tokens = load_validation_tokens_limited(&pattern, None).unwrap();
        assert_eq!(tokens, vec![10, 11]);

        let sidecar_pattern = dir
            .join("fineweb_val_bytes_*.bin")
            .to_string_lossy()
            .to_string();
        let sidecar = load_validation_byte_sidecar_limited(&sidecar_pattern, None).unwrap();
        assert_eq!(sidecar, vec![1.0, 2.0, 3.0, 4.0]);

        let _ = std::fs::remove_file(token_path);
        let _ = std::fs::remove_file(sidecar_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn shifted_u32_loader_fills_input_and_target_without_scratch_span() {
        let dir = std::env::temp_dir().join(format!(
            "pg_shifted_u32_loader_{}_{}",
            std::process::id(),
            31
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let shard0 = dir.join("train_000.bin");
        let shard1 = dir.join("train_001.bin");
        write_test_shard(&shard0, &[10, 11, 12]);
        write_test_shard(&shard1, &[13, 14, 15]);

        let mut stream = TokenStream::from_files(vec![shard0.clone(), shard1.clone()]).unwrap();
        let mut input = vec![999; 8];
        let mut target = vec![999; 8];
        stream
            .take_shifted_u32_into(5, &mut input, &mut target)
            .unwrap();

        assert_eq!(input, vec![10, 11, 12, 13, 14]);
        assert_eq!(target, vec![11, 12, 13, 14, 15]);

        stream
            .take_shifted_u32_into(0, &mut input, &mut target)
            .unwrap();
        assert!(input.is_empty());
        assert!(target.is_empty());

        let _ = std::fs::remove_file(shard0);
        let _ = std::fs::remove_file(shard1);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn shifted_u16_span_loader_preserves_target_lookahead() {
        let dir = std::env::temp_dir().join(format!(
            "pg_shifted_u16_span_loader_{}_{}",
            std::process::id(),
            37
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let shard0 = dir.join("train_000.bin");
        let shard1 = dir.join("train_001.bin");
        write_test_shard(&shard0, &[20, 21, 22]);
        write_test_shard(&shard1, &[23, 24, 25]);

        let mut stream = TokenStream::from_files(vec![shard0.clone(), shard1.clone()]).unwrap();
        let mut span = vec![999; 8];
        stream.take_shifted_span_u16_into(5, &mut span).unwrap();

        assert_eq!(span, vec![20, 21, 22, 23, 24, 25]);

        let _ = std::fs::remove_file(shard0);
        let _ = std::fs::remove_file(shard1);
        let _ = std::fs::remove_dir(dir);
    }
}
