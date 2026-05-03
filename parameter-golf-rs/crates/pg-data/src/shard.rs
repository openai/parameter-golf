use std::path::Path;

use memmap2::Mmap;
use pg_core::error::{PgError, PgResult};

/// Binary data shard format:
/// - 256 x int32 header
///   - header[0] = magic 20240520
///   - header[1] = version 1
///   - header[2] = num_tokens
/// - Payload: num_tokens x uint16 (little-endian token IDs)
///
/// We memory-map the file and interpret the payload as &[u16] with zero copy.
const HEADER_INTS: usize = 256;
const HEADER_BYTES: usize = HEADER_INTS * 4; // 256 * sizeof(i32)
const MAGIC: i32 = 20240520;
const VERSION: i32 = 1;

pub struct DataShard {
    _mmap: Mmap,
    tokens_ptr: *const u16,
    num_tokens: usize,
}

// SAFETY: The mmap is read-only and the pointer is derived from it.
// The mmap lifetime is tied to the DataShard, so the pointer is valid.
unsafe impl Send for DataShard {}
unsafe impl Sync for DataShard {}

impl DataShard {
    /// Open a binary shard file and memory-map it.
    pub fn open(path: &Path) -> PgResult<Self> {
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len() as usize;

        if file_len < HEADER_BYTES {
            return Err(PgError::DataFormat(format!(
                "shard {} too small for header ({} bytes)",
                path.display(),
                file_len
            )));
        }

        // SAFETY: read-only mmap, file must not be modified externally during lifetime
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse header (little-endian i32)
        let header_bytes = &mmap[..HEADER_BYTES];
        let header: &[i32] = bytemuck::cast_slice(&header_bytes[..HEADER_INTS * 4]);

        if header[0] != MAGIC {
            return Err(PgError::DataFormat(format!(
                "shard {} has wrong magic: {} (expected {})",
                path.display(),
                header[0],
                MAGIC
            )));
        }
        if header[1] != VERSION {
            return Err(PgError::DataFormat(format!(
                "shard {} has wrong version: {} (expected {})",
                path.display(),
                header[1],
                VERSION
            )));
        }

        let num_tokens = header[2] as usize;
        let expected_size = HEADER_BYTES + num_tokens * 2; // uint16 = 2 bytes
        if file_len != expected_size {
            return Err(PgError::DataFormat(format!(
                "shard {} size mismatch: file={} expected={}",
                path.display(),
                file_len,
                expected_size
            )));
        }

        let tokens_ptr = mmap[HEADER_BYTES..].as_ptr() as *const u16;

        Ok(Self {
            _mmap: mmap,
            tokens_ptr,
            num_tokens,
        })
    }

    /// Number of tokens in this shard.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get a slice of tokens. Panics if out of bounds.
    pub fn tokens(&self, start: usize, end: usize) -> &[u16] {
        assert!(end <= self.num_tokens, "token range out of bounds");
        assert!(start <= end);
        // SAFETY: tokens_ptr is valid for num_tokens u16 values,
        // derived from the mmap which outlives this reference
        unsafe { std::slice::from_raw_parts(self.tokens_ptr.add(start), end - start) }
    }

    /// Get all tokens.
    pub fn all_tokens(&self) -> &[u16] {
        self.tokens(0, self.num_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_open_train_shard() {
        let shard_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin");
        if !shard_path.exists() {
            eprintln!(
                "Skipping test: shard file not found at {}",
                shard_path.display()
            );
            return;
        }
        let shard = DataShard::open(&shard_path).expect("failed to open shard");
        assert!(shard.num_tokens() > 0, "shard should have tokens");
        // Tokens should be valid vocab IDs (0..1024)
        let first_100 = shard.tokens(0, 100);
        for &tok in first_100 {
            assert!(tok < 1024, "token {} out of vocab range", tok);
        }
        eprintln!(
            "Shard loaded: {} tokens, first 10: {:?}",
            shard.num_tokens(),
            &first_100[..10]
        );
    }
}
