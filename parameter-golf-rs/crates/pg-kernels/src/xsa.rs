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
                let dot: f32 = y_slice
                    .iter()
                    .zip(v_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
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

                let y_dot_v: f32 = y_slice
                    .iter()
                    .zip(v_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let coeff = y_dot_v / v_norm_sq;

                // grad_y = grad_out - (grad_out^T v / ||v||²) * v
                let go_dot_v: f32 = go_slice
                    .iter()
                    .zip(v_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let go_coeff = go_dot_v / v_norm_sq;

                for d in 0..head_dim {
                    grad_y[y_offset + d] = go_slice[d] - go_coeff * v_slice[d];
                }

                // grad_v = -coeff*go - (go·v / ||v||²)*y
                //          + (2*(y·v)*(go·v) / ||v||⁴)*v
                for d in 0..head_dim {
                    grad_v[v_offset + d] += -coeff * go_slice[d]
                        - (go_dot_v / v_norm_sq) * y_slice[d]
                        + (2.0 * y_dot_v * go_dot_v / (v_norm_sq * v_norm_sq)) * v_slice[d];
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

    #[test]
    fn test_xsa_backward_numerical() {
        let tokens = 2;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 3;
        let y_len = tokens * num_heads * head_dim;
        let v_len = tokens * num_kv_heads * head_dim;
        let y: Vec<f32> = (0..y_len)
            .map(|i| 0.07 * ((i * 5 + 3) % 17) as f32 - 0.5)
            .collect();
        let v: Vec<f32> = (0..v_len)
            .map(|i| 0.05 * ((i * 7 + 2) % 19) as f32 - 0.4)
            .collect();
        let grad_out: Vec<f32> = (0..y_len)
            .map(|i| 0.03 * ((i * 11 + 1) % 13) as f32 - 0.2)
            .collect();

        let mut grad_y = vec![0.0; y_len];
        let mut grad_v = vec![0.0; v_len];
        xsa_backward(
            &y,
            &v,
            &grad_out,
            &mut grad_y,
            &mut grad_v,
            tokens,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        let loss = |yy: &[f32], vv: &[f32]| -> f32 {
            let mut out = vec![0.0; y_len];
            xsa_forward(yy, vv, &mut out, tokens, num_heads, num_kv_heads, head_dim);
            out.iter().zip(grad_out.iter()).map(|(&a, &b)| a * b).sum()
        };
        let eps = 1e-3;
        for &idx in &[0usize, 5, y_len - 1] {
            let mut yp = y.clone();
            let mut ym = y.clone();
            yp[idx] += eps;
            ym[idx] -= eps;
            let numerical = (loss(&yp, &v) - loss(&ym, &v)) / (2.0 * eps);
            let diff = (grad_y[idx] - numerical).abs();
            assert!(
                diff < 2e-3,
                "grad_y[{idx}] analytical={} numerical={} diff={diff}",
                grad_y[idx],
                numerical
            );
        }
        for &idx in &[0usize, 4, v_len - 1] {
            let mut vp = v.clone();
            let mut vm = v.clone();
            vp[idx] += eps;
            vm[idx] -= eps;
            let numerical = (loss(&y, &vp) - loss(&y, &vm)) / (2.0 * eps);
            let diff = (grad_v[idx] - numerical).abs();
            assert!(
                diff < 2e-3,
                "grad_v[{idx}] analytical={} numerical={} diff={diff}",
                grad_v[idx],
                numerical
            );
        }
    }
}
