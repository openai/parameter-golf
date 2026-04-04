/// CPU reference causal self-attention with GQA support.
///
/// Q: [tokens, num_heads, head_dim]
/// K: [tokens, num_kv_heads, head_dim]
/// V: [tokens, num_kv_heads, head_dim]
/// Output: [tokens, num_heads, head_dim]
///
/// GQA: each KV head serves (num_heads / num_kv_heads) query heads.
/// On GPU this uses Flash Attention 3 (cuDNN). CPU reference is naive O(n²d).

/// Forward: causal scaled dot-product attention with GQA.
pub fn causal_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let group = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let kv_h = h / group;

        for t in 0..tokens {
            // Compute attention scores for query at position t
            // Only attend to positions 0..=t (causal)
            let q_offset = (t * num_heads + h) * head_dim;

            // Find max score for numerical stability
            let mut max_score = f32::NEG_INFINITY;
            for s in 0..=t {
                let k_offset = (s * num_kv_heads + kv_h) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_offset + d] * k[k_offset + d];
                }
                let score = dot * scale;
                if score > max_score {
                    max_score = score;
                }
            }

            // Compute softmax and weighted sum
            let mut sum_exp = 0.0f32;
            let o_offset = (t * num_heads + h) * head_dim;

            // Zero output first
            for d in 0..head_dim {
                output[o_offset + d] = 0.0;
            }

            for s in 0..=t {
                let k_offset = (s * num_kv_heads + kv_h) * head_dim;
                let v_offset = (s * num_kv_heads + kv_h) * head_dim;

                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_offset + d] * k[k_offset + d];
                }
                let weight = (dot * scale - max_score).exp();
                sum_exp += weight;

                for d in 0..head_dim {
                    output[o_offset + d] += weight * v[v_offset + d];
                }
            }

            // Normalize
            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                output[o_offset + d] *= inv_sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_token_attention() {
        // With a single token, attention output should equal V
        let head_dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![0.0, 1.0, 0.0, 0.0];
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let mut out = vec![0.0; head_dim];

        causal_attention_forward(&q, &k, &v, &mut out, 1, 1, 1, head_dim);

        for d in 0..head_dim {
            assert!(
                (out[d] - v[d]).abs() < 1e-5,
                "mismatch at {}: {} vs {}",
                d, out[d], v[d]
            );
        }
    }

    #[test]
    fn test_causal_masking() {
        // Two tokens. Token 0 should only attend to itself.
        // Token 1 attends to both.
        let head_dim = 2;
        let q = vec![1.0, 0.0, 1.0, 0.0]; // 2 tokens
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; 4];

        causal_attention_forward(&q, &k, &v, &mut out, 2, 1, 1, head_dim);

        // Token 0: only attends to itself → output = v[0]
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 0.0).abs() < 1e-5);

        // Token 1: attends to both with softmax weights based on q·k scores
        // q[1]·k[0] = 1*1+0*0 = 1, q[1]·k[1] = 1*0+0*1 = 0
        // After scale (1/sqrt(2)): scores ≈ [0.707, 0]
        // softmax([0.707, 0]) → higher weight on token 0
        assert!(out[2] > 0.5); // more of v[0]=[1,0]
    }

    #[test]
    fn test_gqa_attention() {
        // 2 query heads, 1 KV head
        let head_dim = 2;
        let tokens = 1;
        let q = vec![1.0, 0.0, 0.0, 1.0]; // head 0: [1,0], head 1: [0,1]
        let k = vec![1.0, 0.0]; // 1 KV head
        let v = vec![0.5, 0.5];
        let mut out = vec![0.0; 4];

        causal_attention_forward(&q, &k, &v, &mut out, tokens, 2, 1, head_dim);

        // Both heads attend to same KV, single token → both get v
        for d in 0..head_dim {
            assert!((out[d] - 0.5).abs() < 1e-5);
            assert!((out[head_dim + d] - 0.5).abs() < 1e-5);
        }
    }
}
