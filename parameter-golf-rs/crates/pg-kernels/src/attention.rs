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

/// Backward: causal attention gradients with GQA.
///
/// Given grad_output [tokens, num_heads, head_dim], computes:
///   grad_q [tokens, num_heads, head_dim]
///   grad_k [tokens, num_kv_heads, head_dim]
///   grad_v [tokens, num_kv_heads, head_dim]
///
/// Recomputes attention weights from Q, K (no saved state needed).
pub fn causal_attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    _output: &[f32], // forward output (reserved for fused softmax backward variant)
    grad_output: &[f32], // [tokens, num_heads, head_dim]
    grad_q: &mut [f32], // [tokens, num_heads, head_dim]
    grad_k: &mut [f32], // [tokens, num_kv_heads, head_dim]
    grad_v: &mut [f32], // [tokens, num_kv_heads, head_dim]
    tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let group = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Zero grad accumulators
    grad_q.iter_mut().for_each(|v| *v = 0.0);
    grad_k.iter_mut().for_each(|v| *v = 0.0);
    grad_v.iter_mut().for_each(|v| *v = 0.0);

    for h in 0..num_heads {
        let kv_h = h / group;

        for t in 0..tokens {
            let q_off = (t * num_heads + h) * head_dim;
            let o_off = (t * num_heads + h) * head_dim;

            // Recompute attention weights (softmax(Q·K^T / sqrt(d)))
            let mut scores = vec![0.0f32; t + 1];
            let mut max_s = f32::NEG_INFINITY;
            for s in 0..=t {
                let k_off = (s * num_kv_heads + kv_h) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                scores[s] = dot * scale;
                if scores[s] > max_s {
                    max_s = scores[s];
                }
            }
            let mut weights = vec![0.0f32; t + 1];
            let mut sum_exp = 0.0f32;
            for s in 0..=t {
                weights[s] = (scores[s] - max_s).exp();
                sum_exp += weights[s];
            }
            let inv_sum = 1.0 / sum_exp;
            for s in 0..=t {
                weights[s] *= inv_sum;
            }

            // grad_v: dL/dV_s += w_s * grad_output_t (for all s <= t)
            for s in 0..=t {
                let v_off = (s * num_kv_heads + kv_h) * head_dim;
                for d in 0..head_dim {
                    grad_v[v_off + d] += weights[s] * grad_output[o_off + d];
                }
            }

            // grad_weights: dL/dw_s = grad_output_t · V_s
            let mut grad_w = vec![0.0f32; t + 1];
            for s in 0..=t {
                let v_off = (s * num_kv_heads + kv_h) * head_dim;
                for d in 0..head_dim {
                    grad_w[s] += grad_output[o_off + d] * v[v_off + d];
                }
            }

            // Softmax backward: dL/dscore_s = w_s * (grad_w_s - sum_j(w_j * grad_w_j))
            let dot_wg: f32 = (0..=t).map(|s| weights[s] * grad_w[s]).sum();
            let mut grad_scores = vec![0.0f32; t + 1];
            for s in 0..=t {
                grad_scores[s] = weights[s] * (grad_w[s] - dot_wg);
            }

            // Score backward: score = q · k * scale
            // grad_q += grad_score_s * k_s * scale
            // grad_k_s += grad_score_s * q_t * scale
            for s in 0..=t {
                let k_off = (s * num_kv_heads + kv_h) * head_dim;
                let gs = grad_scores[s] * scale;
                for d in 0..head_dim {
                    grad_q[q_off + d] += gs * k[k_off + d];
                    grad_k[k_off + d] += gs * q[q_off + d];
                }
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
                d,
                out[d],
                v[d]
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
    fn test_attention_backward_numerical() {
        let tokens = 3;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;

        // Deterministic pseudo-random inputs
        let make_val = |i: usize| (i as f32 * 0.37 + 0.13).sin() * 0.5;
        let q: Vec<f32> = (0..tokens * num_heads * head_dim).map(make_val).collect();
        let k: Vec<f32> = (0..tokens * num_kv_heads * head_dim)
            .map(|i| make_val(i + 100))
            .collect();
        let v: Vec<f32> = (0..tokens * num_kv_heads * head_dim)
            .map(|i| make_val(i + 200))
            .collect();

        // Forward
        let mut out = vec![0.0f32; tokens * num_heads * head_dim];
        causal_attention_forward(
            &q,
            &k,
            &v,
            &mut out,
            tokens,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        // Use sum of outputs as scalar loss
        let grad_output = vec![1.0f32; tokens * num_heads * head_dim];

        // Analytical
        let mut grad_q = vec![0.0f32; tokens * num_heads * head_dim];
        let mut grad_k = vec![0.0f32; tokens * num_kv_heads * head_dim];
        let mut grad_v = vec![0.0f32; tokens * num_kv_heads * head_dim];
        causal_attention_backward(
            &q,
            &k,
            &v,
            &out,
            &grad_output,
            &mut grad_q,
            &mut grad_k,
            &mut grad_v,
            tokens,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        // Numerical gradient for Q
        let eps = 1e-3;
        for idx in 0..q.len().min(8) {
            let mut q_p = q.clone();
            let mut q_m = q.clone();
            q_p[idx] += eps;
            q_m[idx] -= eps;
            let mut out_p = vec![0.0f32; out.len()];
            let mut out_m = vec![0.0f32; out.len()];
            causal_attention_forward(
                &q_p,
                &k,
                &v,
                &mut out_p,
                tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            causal_attention_forward(
                &q_m,
                &k,
                &v,
                &mut out_m,
                tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            let numerical: f32 = out_p
                .iter()
                .zip(out_m.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();
            let diff = (grad_q[idx] - numerical).abs();
            let max_val = grad_q[idx].abs().max(numerical.abs()).max(1e-6);
            assert!(
                diff / max_val < 0.02 || diff < 1e-4,
                "Q grad mismatch at {}: analytical={}, numerical={}",
                idx,
                grad_q[idx],
                numerical
            );
        }

        // Numerical gradient for K
        for idx in 0..k.len().min(4) {
            let mut k_p = k.clone();
            let mut k_m = k.clone();
            k_p[idx] += eps;
            k_m[idx] -= eps;
            let mut out_p = vec![0.0f32; out.len()];
            let mut out_m = vec![0.0f32; out.len()];
            causal_attention_forward(
                &q,
                &k_p,
                &v,
                &mut out_p,
                tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            causal_attention_forward(
                &q,
                &k_m,
                &v,
                &mut out_m,
                tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            let numerical: f32 = out_p
                .iter()
                .zip(out_m.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();
            let diff = (grad_k[idx] - numerical).abs();
            let max_val = grad_k[idx].abs().max(numerical.abs()).max(1e-6);
            assert!(
                diff / max_val < 0.02 || diff < 1e-4,
                "K grad mismatch at {}: analytical={}, numerical={}",
                idx,
                grad_k[idx],
                numerical
            );
        }

        // Numerical gradient for V
        for idx in 0..v.len().min(4) {
            let mut v_p = v.clone();
            let mut v_m = v.clone();
            v_p[idx] += eps;
            v_m[idx] -= eps;
            let mut out_p = vec![0.0f32; out.len()];
            let mut out_m = vec![0.0f32; out.len()];
            causal_attention_forward(
                &q,
                &k,
                &v_p,
                &mut out_p,
                tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            causal_attention_forward(
                &q,
                &k,
                &v_m,
                &mut out_m,
                tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            let numerical: f32 = out_p
                .iter()
                .zip(out_m.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();
            let diff = (grad_v[idx] - numerical).abs();
            let max_val = grad_v[idx].abs().max(numerical.abs()).max(1e-6);
            assert!(
                diff / max_val < 0.02 || diff < 1e-4,
                "V grad mismatch at {}: analytical={}, numerical={}",
                idx,
                grad_v[idx],
                numerical
            );
        }
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
