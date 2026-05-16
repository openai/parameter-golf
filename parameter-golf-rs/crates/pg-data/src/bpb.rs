/// BPB (bits-per-byte) metric computation.
///
/// The competition uses a tokenizer-agnostic compression metric:
///   bpb = (cross_entropy_loss / ln(2)) * (tokens / bytes)
///
/// This requires mapping each token to its byte count, accounting for
/// SentencePiece's leading space convention:
/// - Each token has a base byte count from its UTF-8 surface form
/// - Tokens starting with "▁" (U+2581) have a leading space byte that
///   only counts if the previous token is NOT a boundary token
///
/// The LUTs are built once from the SentencePiece model file.
use std::path::Path;

use pg_core::error::{PgError, PgResult};

/// Lookup tables for tokenizer-aware byte counting.
pub struct BpbLuts {
    /// base_bytes[token_id] = number of UTF-8 bytes in token's surface form
    pub base_bytes: Vec<i16>,
    /// has_leading_space[token_id] = true if token starts with "▁"
    pub has_leading_space: Vec<bool>,
    /// is_boundary_token[token_id] = true for control/unknown/unused tokens
    pub is_boundary_token: Vec<bool>,
}

impl BpbLuts {
    /// Build LUTs for a vocab. Placeholder until sentencepiece-sys is integrated.
    /// For now, uses a simple byte-per-token estimate.
    pub fn placeholder(vocab_size: usize) -> Self {
        Self {
            base_bytes: vec![4; vocab_size], // conservative estimate
            has_leading_space: vec![false; vocab_size],
            is_boundary_token: vec![false; vocab_size],
        }
    }

    /// Build byte-count LUTs from a SentencePiece `.vocab` file.
    ///
    /// The file is expected to contain one token per line, with the piece in
    /// the first tab-separated column. This avoids linking sentencepiece into
    /// the Rust submission path while still using the real tokenizer surfaces.
    pub fn from_vocab_file(path: &Path) -> PgResult<Self> {
        let body = std::fs::read_to_string(path)?;
        let mut base_bytes = Vec::new();
        let mut has_leading_space = Vec::new();
        let mut is_boundary_token = Vec::new();

        for (line_idx, line) in body.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let piece = line.split('\t').next().ok_or_else(|| {
                PgError::DataFormat(format!("invalid vocab line {}", line_idx + 1))
            })?;
            let boundary = is_control_piece(piece);
            let leading = piece.starts_with('▁');
            let bytes = if boundary {
                0
            } else if is_byte_piece(piece) {
                1
            } else {
                piece
                    .trim_start_matches('▁')
                    .as_bytes()
                    .len()
                    .min(i16::MAX as usize) as i16
            };
            base_bytes.push(bytes);
            has_leading_space.push(leading);
            is_boundary_token.push(boundary);
        }

        if base_bytes.is_empty() {
            return Err(PgError::DataFormat(format!(
                "tokenizer vocab {} contained no pieces",
                path.display()
            )));
        }

        Ok(Self {
            base_bytes,
            has_leading_space,
            is_boundary_token,
        })
    }

    /// Count bytes for a sequence of (prev_token, target_token) pairs.
    pub fn count_bytes(&self, prev_tokens: &[u16], target_tokens: &[u16]) -> f64 {
        let mut total_bytes = 0.0;
        for (&prev, &tgt) in prev_tokens.iter().zip(target_tokens.iter()) {
            total_bytes += self.byte_count_pair(prev as usize, tgt as usize);
        }
        total_bytes
    }

    /// Per-target byte counts for a token stream. Entry `i` corresponds to
    /// the scored pair `(tokens[i], tokens[i + 1])`.
    pub fn pair_byte_counts_u32(&self, tokens: &[u32]) -> Vec<f32> {
        tokens
            .windows(2)
            .map(|w| self.byte_count_pair(w[0] as usize, w[1] as usize) as f32)
            .collect()
    }

    pub fn byte_count_pair(&self, prev: usize, target: usize) -> f64 {
        if target >= self.base_bytes.len() {
            return 1.0;
        }
        let mut bytes = self.base_bytes[target].max(0) as f64;
        let prev_is_boundary = prev >= self.is_boundary_token.len() || self.is_boundary_token[prev];
        if self.has_leading_space[target] && !prev_is_boundary {
            bytes += 1.0;
        }
        bytes
    }
}

/// Compute BPB from total loss, token count, and byte count.
pub fn compute_bpb(total_loss: f64, token_count: f64, byte_count: f64) -> f64 {
    let bits_per_token = total_loss / std::f64::consts::LN_2;
    let tokens_per_byte = token_count / byte_count;
    bits_per_token * tokens_per_byte
}

fn is_control_piece(piece: &str) -> bool {
    matches!(piece, "<pad>" | "<s>" | "</s>" | "<unk>") || piece.starts_with("<unused")
}

fn is_byte_piece(piece: &str) -> bool {
    piece.len() == 6
        && piece.starts_with("<0x")
        && piece.ends_with('>')
        && piece[3..5].chars().all(|c| c.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_piece_and_leading_space_counts_are_tokenizer_aware() {
        let luts = BpbLuts {
            base_bytes: vec![0, 1, 3],
            has_leading_space: vec![false, false, true],
            is_boundary_token: vec![true, false, false],
        };
        assert_eq!(luts.byte_count_pair(0, 2), 3.0);
        assert_eq!(luts.byte_count_pair(1, 2), 4.0);
        assert_eq!(luts.byte_count_pair(2, 1), 1.0);
    }
}
