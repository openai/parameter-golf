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

/// Lookup tables for tokenizer-agnostic byte counting.
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

    /// Count bytes for a sequence of (prev_token, target_token) pairs.
    pub fn count_bytes(&self, prev_tokens: &[u16], target_tokens: &[u16]) -> f64 {
        let mut total_bytes = 0.0;
        for (&prev, &tgt) in prev_tokens.iter().zip(target_tokens.iter()) {
            let mut b = self.base_bytes[tgt as usize] as f64;
            if self.has_leading_space[tgt as usize] && !self.is_boundary_token[prev as usize] {
                b += 1.0;
            }
            total_bytes += b;
        }
        total_bytes
    }
}

/// Compute BPB from total loss, token count, and byte count.
pub fn compute_bpb(total_loss: f64, token_count: f64, byte_count: f64) -> f64 {
    let bits_per_token = total_loss / std::f64::consts::LN_2;
    let tokens_per_byte = token_count / byte_count;
    bits_per_token * tokens_per_byte
}
