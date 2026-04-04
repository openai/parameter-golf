/// XSA (Exclusive Self Attention) — removes self-value redundancy.
///
/// Standard attention output y_i has high cosine similarity with the token's
/// own value vector v_i. XSA projects out this component:
///   z_i = y_i - (y_i^T v_i / ||v_i||²) * v_i
///
/// Applied to the last 4 layers (XSA4). Zero new parameters, negligible compute.
/// BPB gain: ~0.003–0.005.
///
/// GQA-aware: with 8 heads and 4 KV heads (group=2), we reshape to
/// [B, T, Hkv, group, D] to avoid repeat_interleave.

/// Forward: project out self-value component.
/// y: [batch*seq, num_heads, head_dim] — attention output
/// v: [batch*seq, num_kv_heads, head_dim] — value vectors (before GQA expansion)
/// output: same shape as y
pub fn xsa_forward(
    y: &[f32],
    v: &[f32],
    output: &mut [f32],
    tokens: usize, // batch * seq_len
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let group = num_heads / num_kv_heads;

    for t in 0..tokens {
        for kv_h in 0..num_kv_heads {
            // Get v vector for this KV head
            let v_offset = (t * num_kv_heads + kv_h) * head_dim;
            let v_slice = &v[v_offset..v_offset + head_dim];

            // ||v||²
            let v_norm_sq: f32 = v_slice.iter().map(|&x| x * x).sum::<f32>() + 1e-8;

            // For each Q head in this group
            for g in 0..group {
                let h = kv_h * group + g;
                let y_offset = (t * num_heads + h) * head_dim;
                let y_slice = &y[y_offset..y_offset + head_dim];

                // y^T v
                let dot: f32 = y_slice.iter().zip(v_slice.iter()).map(|(&a, &b)| a * b).sum();
                let coeff = dot / v_norm_sq;

                // z = y - coeff * v
                for d in 0..head_dim {
                    output[y_offset + d] = y_slice[d] - coeff * v_slice[d];
                }
            }
        }
    }
}

/// Backward for XSA.
/// grad_y, grad_v computed from grad_output (same shape as output).
pub fn xsa_backward(
    y: &[f32],
    v: &[f32],
    grad_output: &[f32],
    grad_y: &mut [f32],
    grad_v: &mut [f32],
    tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let group = num_heads / num_kv_heads;

    // Zero grad_v first (accumulated from multiple heads in group)
    grad_v.iter_mut().for_each(|x| *x = 0.0);

    for t in 0..tokens {
        for kv_h in 0..num_kv_heads {
            let v_offset = (t * num_kv_heads + kv_h) * head_dim;
            let v_slice = &v[v_offset..v_offset + head_dim];
            let v_norm_sq: f32 = v_slice.iter().map(|&x| x * x).sum::<f32>() + 1e-8;

            for g in 0..group {
                let h = kv_h * group + g;
                let y_offset = (t * num_heads + h) * head_dim;
                let y_slice = &y[y_offset..y_offset + head_dim];
                let go_slice = &grad_output[y_offset..y_offset + head_dim];

                let y_dot_v: f32 = y_slice.iter().zip(v_slice.iter()).map(|(&a, &b)| a * b).sum();
                let coeff = y_dot_v / v_norm_sq;

                // grad_y = grad_out - (grad_out^T v / ||v||²) * v
                let go_dot_v: f32 = go_slice.iter().zip(v_slice.iter()).map(|(&a, &b)| a * b).sum();
                let go_coeff = go_dot_v / v_norm_sq;

                for d in 0..head_dim {
                    grad_y[y_offset + d] = go_slice[d] - go_coeff * v_slice[d];
                }

                // grad_v contributions (accumulated across group heads)
                // d/dv of (y - (y^T v / ||v||²) * v)
                // = -(go^T y / ||v||² + coeff * go^T) * something... complex
                // Simplified: grad_v += -coeff * go - (go^T v / ||v||²) * (y - 2*coeff*v) / ||v||²
                // For correctness, use the full derivative:
                // dL/dv_k = sum_g [ -go · (y·e_k/||v||² + coeff·e_k) + go · v · (2·y·v·e_k / ||v||⁴) ]
                // This simplifies per-element to:
                for d in 0..head_dim {
                    let dcoeff_dv_d = (y_slice[d] * v_norm_sq - y_dot_v * 2.0 * v_slice[d])
                        / (v_norm_sq * v_norm_sq);
                    grad_v[v_offset + d] +=
                        -go_slice[d] * coeff - (go_dot_v * v_slice[d] * 0.0); // simplified
                    // Full derivative: -go · (dcoeff/dv · v + coeff · e_d)
                    grad_v[v_offset + d] = grad_v[v_offset + d]
                        - go_dot_v * dcoeff_dv_d; // projection term
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xsa_removes_self_component() {
        // y = [1, 0], v = [1, 0] → z should be [0, 0] (y is entirely along v)
        let y = vec![1.0f32, 0.0];
        let v = vec![1.0f32, 0.0];
        let mut out = vec![0.0; 2];
        xsa_forward(&y, &v, &mut out, 1, 1, 1, 2);
        assert!(out[0].abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
    }

    #[test]
    fn test_xsa_preserves_orthogonal() {
        // y = [0, 1], v = [1, 0] → z should be [0, 1] (y is orthogonal to v)
        let y = vec![0.0f32, 1.0];
        let v = vec![1.0f32, 0.0];
        let mut out = vec![0.0; 2];
        xsa_forward(&y, &v, &mut out, 1, 1, 1, 2);
        assert!(out[0].abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_xsa_gqa() {
        // 2 heads, 1 KV head (group=2)
        // Both heads share the same v, each gets projected independently
        let y = vec![
            1.0, 0.0, // head 0
            0.5, 0.5, // head 1
        ];
        let v = vec![1.0, 0.0]; // single KV head
        let mut out = vec![0.0; 4];
        xsa_forward(&y, &v, &mut out, 1, 2, 1, 2);

        // head 0: y=[1,0], v=[1,0] → proj out → [0, 0]
        assert!(out[0].abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
        // head 1: y=[0.5, 0.5], v=[1,0] → remove v-component → [0, 0.5]
        assert!(out[2].abs() < 1e-6);
        assert!((out[3] - 0.5).abs() < 1e-6);
    }
}
