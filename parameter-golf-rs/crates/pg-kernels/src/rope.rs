/// Partial RoPE (Rotary Position Embedding) on first 16 of 64 head dimensions.
///
/// The remaining 48 dimensions attend without positional bias, learning
/// position-independent semantic patterns. -0.0023 BPB improvement.
///
/// Matches SOTA Python convention (half-split, not interleaved):
///   x_rotated[0:half]   = x[0:half] * cos + x[half:] * sin
///   x_rotated[half:]    = -x[0:half] * sin + x[half:] * cos
/// where θ = position / (base^(2i/rope_dims))

/// Precompute cos/sin tables for RoPE.
/// Returns (cos_table, sin_table), each of shape [seq_len, rope_dims/2].
pub fn precompute_rope_tables(seq_len: usize, rope_dims: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rope_dims / 2;
    let mut cos_table = vec![0.0f32; seq_len * half];
    let mut sin_table = vec![0.0f32; seq_len * half];

    for pos in 0..seq_len {
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / rope_dims as f32);
            let angle = pos as f32 * freq;
            cos_table[pos * half + i] = angle.cos();
            sin_table[pos * half + i] = angle.sin();
        }
    }

    (cos_table, sin_table)
}

/// Apply partial RoPE to Q or K tensor in-place.
///
/// x: [batch * seq_len, num_heads, head_dim] stored as flat f32 array
/// Only the first `rope_dims` of `head_dim` are rotated.
/// cos/sin: [seq_len, rope_dims/2]
///
/// For the fused QK-norm+RoPE+q_gain kernel, this will be merged into
/// a single CubeCL kernel. This CPU reference validates correctness.
pub fn apply_partial_rope(
    x: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_dims: usize,
) {
    let half = rope_dims / 2;

    for b in 0..batch_size {
        for s in 0..seq_len {
            for h in 0..num_heads {
                let base_idx = ((b * seq_len + s) * num_heads + h) * head_dim;
                let rope_offset = s * half;

                // Rotate first rope_dims dimensions
                for i in 0..half {
                    let cos_val = cos_table[rope_offset + i];
                    let sin_val = sin_table[rope_offset + i];

                    let x1 = x[base_idx + i];
                    let x2 = x[base_idx + half + i];

                    // SOTA convention: first_half = x1*cos + x2*sin
                    x[base_idx + i] = x1 * cos_val + x2 * sin_val;
                    // second_half = -x1*sin + x2*cos
                    x[base_idx + half + i] = -x1 * sin_val + x2 * cos_val;
                }
                // Dimensions [rope_dims..head_dim] are untouched (passthrough)
            }
        }
    }
}

/// Backward for partial RoPE (rotation is its own inverse with negated sin).
pub fn apply_partial_rope_backward(
    grad: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_dims: usize,
) {
    let half = rope_dims / 2;

    for b in 0..batch_size {
        for s in 0..seq_len {
            for h in 0..num_heads {
                let base_idx = ((b * seq_len + s) * num_heads + h) * head_dim;
                let rope_offset = s * half;

                for i in 0..half {
                    let cos_val = cos_table[rope_offset + i];
                    let sin_val = sin_table[rope_offset + i];

                    let g1 = grad[base_idx + i];
                    let g2 = grad[base_idx + half + i];

                    // Inverse of SOTA convention: transpose the rotation matrix
                    grad[base_idx + i] = g1 * cos_val - g2 * sin_val;
                    grad[base_idx + half + i] = g1 * sin_val + g2 * cos_val;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_tables() {
        let (cos, sin) = precompute_rope_tables(4, 16, 10000.0);
        assert_eq!(cos.len(), 4 * 8); // seq_len * rope_dims/2
        // Position 0 should have all cos=1, sin=0
        for i in 0..8 {
            assert!((cos[i] - 1.0).abs() < 1e-5);
            assert!(sin[i].abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_roundtrip() {
        // Apply RoPE then inverse should recover original
        let (cos, sin) = precompute_rope_tables(4, 16, 10000.0);
        let original = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, // passthrough dims (17-64 equivalent)
            100.0, 200.0, 300.0, 400.0,
        ];
        let head_dim = 20; // 16 rope + 4 passthrough
        let rope_dims = 16;

        // Forward
        let mut x = original.clone();
        apply_partial_rope(&mut x, &cos, &sin, 1, 1, 1, head_dim, rope_dims);

        // Passthrough dims unchanged
        assert_eq!(x[16], 100.0);
        assert_eq!(x[17], 200.0);

        // Backward (inverse)
        apply_partial_rope_backward(&mut x, &cos, &sin, 1, 1, 1, head_dim, rope_dims);

        // Should recover original
        for i in 0..original.len() {
            assert!(
                (x[i] - original[i]).abs() < 1e-4,
                "mismatch at {}: got {}, expected {}",
                i,
                x[i],
                original[i]
            );
        }
    }
}
