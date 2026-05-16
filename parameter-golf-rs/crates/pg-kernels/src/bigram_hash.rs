/// BigramHash — learned hash-based lexical memory.
///
/// XOR-hash two consecutive token IDs into a bucket, look up a learned embedding.
/// BigramHash(1536 buckets, 128 embed_dim) → 197K params.
///
/// Matches SOTA Python: hash = (36313 * cur) XOR (27191 * prev) % (num_buckets - 1)
/// Position 0 maps to (num_buckets - 1) as a sentinel bucket.

const HASH_A: u32 = 36313;
const HASH_B: u32 = 27191;

/// Compute bigram hash bucket index (SOTA convention).
/// For t=0 (no previous token), returns num_buckets - 1 (sentinel).
#[inline]
pub fn bigram_hash(prev: Option<u32>, cur: u32, num_buckets: usize) -> usize {
    match prev {
        None => num_buckets - 1,
        Some(p) => {
            let h = cur.wrapping_mul(HASH_A) ^ p.wrapping_mul(HASH_B);
            (h as usize) % (num_buckets - 1)
        }
    }
}

/// Forward: compute bigram embeddings for a sequence.
/// tokens: [seq_len]
/// embedding_table: [num_buckets, embed_dim]
/// output: [seq_len, embed_dim]
pub fn bigram_hash_forward(
    tokens: &[u32],
    embedding_table: &[f32],
    output: &mut [f32],
    num_buckets: usize,
    embed_dim: usize,
) {
    let seq_len = tokens.len();
    for t in 0..seq_len {
        let prev = if t == 0 { None } else { Some(tokens[t - 1]) };
        let bucket = bigram_hash(prev, tokens[t], num_buckets);
        let src = &embedding_table[bucket * embed_dim..(bucket + 1) * embed_dim];
        let dst = &mut output[t * embed_dim..(t + 1) * embed_dim];
        dst.copy_from_slice(src);
    }
}

/// Backward: accumulate gradients into embedding table.
/// grad_output: [seq_len, embed_dim]
/// grad_embedding: [num_buckets, embed_dim] (accumulated, NOT zeroed here)
pub fn bigram_hash_backward(
    tokens: &[u32],
    grad_output: &[f32],
    grad_embedding: &mut [f32],
    num_buckets: usize,
    embed_dim: usize,
) {
    let seq_len = tokens.len();
    for t in 0..seq_len {
        let prev = if t == 0 { None } else { Some(tokens[t - 1]) };
        let bucket = bigram_hash(prev, tokens[t], num_buckets);
        let go = &grad_output[t * embed_dim..(t + 1) * embed_dim];
        let ge = &mut grad_embedding[bucket * embed_dim..(bucket + 1) * embed_dim];
        for d in 0..embed_dim {
            ge[d] += go[d];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigram_hash_deterministic() {
        let h1 = bigram_hash(Some(100), 200, 1536);
        let h2 = bigram_hash(Some(100), 200, 1536);
        assert_eq!(h1, h2);
        assert!(h1 < 1536);
    }

    #[test]
    fn test_bigram_hash_different_inputs() {
        let h1 = bigram_hash(Some(100), 200, 1536);
        let h2 = bigram_hash(Some(200), 100, 1536);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_bigram_hash_sentinel() {
        // Position 0 (no prev) should map to sentinel bucket
        let h = bigram_hash(None, 42, 1536);
        assert_eq!(h, 1535);
    }

    #[test]
    fn test_bigram_forward_backward() {
        let num_buckets = 8;
        let embed_dim = 4;
        let tokens = vec![5u32, 3, 7, 1];
        let seq_len = tokens.len();

        let mut table = vec![0.0f32; num_buckets * embed_dim];
        for i in 0..table.len() {
            table[i] = (i as f32) * 0.1;
        }

        let mut output = vec![0.0f32; seq_len * embed_dim];
        bigram_hash_forward(&tokens, &table, &mut output, num_buckets, embed_dim);

        // Verify each position got the right bucket's embedding
        for t in 0..seq_len {
            let prev = if t == 0 { None } else { Some(tokens[t - 1]) };
            let bucket = bigram_hash(prev, tokens[t], num_buckets);
            for d in 0..embed_dim {
                assert_eq!(output[t * embed_dim + d], table[bucket * embed_dim + d]);
            }
        }

        // Backward: grad_output = 1.0 everywhere
        let grad_output = vec![1.0f32; seq_len * embed_dim];
        let mut grad_table = vec![0.0f32; num_buckets * embed_dim];
        bigram_hash_backward(
            &tokens,
            &grad_output,
            &mut grad_table,
            num_buckets,
            embed_dim,
        );

        let total_grad: f32 = grad_table.iter().sum();
        assert!((total_grad - (seq_len * embed_dim) as f32).abs() < 1e-6);
    }
}
