/// Fused-kernel CPU references (TDD §2, Innovation 2).
///
/// These functions implement, in plain Rust, the same per-element operations
/// the future GPU CubeCL kernels will perform. They serve two purposes:
///
/// 1. They are *correctness oracles*: GPU kernels are validated by comparing
///    their output against these CPU references on the same input.
/// 2. They allow the rest of the Rust stack to take the "fused path" right
///    now, even on CPU, so that the call sites are already written and the
///    GPU swap is purely a backend choice (`#[cfg(feature = "cuda")]`).
///
/// Naming: a fused kernel always has both a `_fused` and an `_unfused`
/// implementation in tests, with a `_matches_unfused` test asserting they
/// agree to within float tolerance.
///
/// ### Fusions implemented (TDD §2.4)
///
/// | Name | Operations merged |
/// |------|------------------|
/// | **Fusion A**: `xsa_residual_norm_fused` | XSA projection + residual add + RMSNorm |
/// | **Fusion B**: `bigram_embed_fused`      | Token embedding + bigram lookup + bigram projection + add |
/// | **Fusion C**: `rmsnorm_qk_rope_qgain`   | RMSNorm(Q,K) + partial RoPE + per-head q_gain |
use crate::bigram_hash::bigram_hash;

/// Fusion A: XSA-projection epilogue + residual add + RMSNorm.
///
/// Computes (per token):
///   z[i]  = attn_proj_out[i] - (attn_proj_out[i]·v_self[i] / ||v_self[i]||²) * v_self[i]
///   h[i]  = residual[i] + z[i]
///   norm[i] = (h[i] / rms(h)) * rms_weight[i] * ln_scale
///
/// Stores both the new residual stream `h_out` and the normalized version
/// `normed_out` because both are needed downstream (h_out feeds the next
/// residual link, normed_out feeds the MLP block).
///
/// In the GPU version this is one CubeCL kernel that keeps `z` in registers
/// and only writes `h_out` and `normed_out` to global memory.
///
/// XSA is per-(token, KV-head); residual+norm is per-(token, dim). To match
/// what the actual model does we treat `attn_proj_out` and `v_self` as
/// `[tokens, dim]` after the output projection has folded heads into dim.
/// We expose XSA at the head granularity *before* the output projection to
/// keep the algebra honest. The realistic GPU fusion will fold this into the
/// post-output-projection kernel.
#[allow(clippy::too_many_arguments)]
pub fn xsa_residual_norm_fused(
    attn_proj_out: &[f32], // [tokens, dim] — attention output projected
    v_self_full: &[f32],   // [tokens, dim] — V vectors expanded to head-dim space
    residual: &[f32],      // [tokens, dim]
    rms_weight: &[f32],    // [dim]
    ln_scale: f32,
    h_out: &mut [f32],      // [tokens, dim] — residual stream after add
    normed_out: &mut [f32], // [tokens, dim] — normalized for next sublayer
    tokens: usize,
    dim: usize,
    eps: f32,
) {
    for t in 0..tokens {
        let off = t * dim;

        // 1. XSA projection: z = y - (y·v / (||v||² + ε)) v
        let mut y_dot_v = 0.0f32;
        let mut v_norm_sq = 0.0f32;
        for d in 0..dim {
            let y = attn_proj_out[off + d];
            let v = v_self_full[off + d];
            y_dot_v += y * v;
            v_norm_sq += v * v;
        }
        let coeff = y_dot_v / (v_norm_sq + eps);

        // 2. Residual add (z + residual) and accumulate sum-of-squares for RMSNorm
        let mut sq_sum = 0.0f32;
        for d in 0..dim {
            let y = attn_proj_out[off + d];
            let v = v_self_full[off + d];
            let z = y - coeff * v;
            let h = residual[off + d] + z;
            h_out[off + d] = h;
            sq_sum += h * h;
        }

        // 3. RMSNorm(h_out) * rms_weight * ln_scale
        let inv_rms = ln_scale / (sq_sum / dim as f32 + eps).sqrt();
        for d in 0..dim {
            normed_out[off + d] = h_out[off + d] * inv_rms * rms_weight[d];
        }
    }
}

/// Unfused reference: same operations, separate kernels. Used by tests to
/// prove `xsa_residual_norm_fused` is bit-equivalent (within float tolerance).
#[allow(clippy::too_many_arguments)]
pub fn xsa_residual_norm_unfused(
    attn_proj_out: &[f32],
    v_self_full: &[f32],
    residual: &[f32],
    rms_weight: &[f32],
    ln_scale: f32,
    h_out: &mut [f32],
    normed_out: &mut [f32],
    tokens: usize,
    dim: usize,
    eps: f32,
) {
    // 1. Standalone XSA: z[t] = y - coeff*v
    let mut z = vec![0.0f32; tokens * dim];
    for t in 0..tokens {
        let off = t * dim;
        let mut yv = 0.0f32;
        let mut vv = 0.0f32;
        for d in 0..dim {
            yv += attn_proj_out[off + d] * v_self_full[off + d];
            vv += v_self_full[off + d] * v_self_full[off + d];
        }
        let c = yv / (vv + eps);
        for d in 0..dim {
            z[off + d] = attn_proj_out[off + d] - c * v_self_full[off + d];
        }
    }
    // 2. Standalone residual add: h = residual + z
    for i in 0..tokens * dim {
        h_out[i] = residual[i] + z[i];
    }
    // 3. Standalone RMSNorm(h) * rms_weight * ln_scale
    for t in 0..tokens {
        let off = t * dim;
        let mut sq = 0.0f32;
        for d in 0..dim {
            sq += h_out[off + d] * h_out[off + d];
        }
        let inv_rms = ln_scale / (sq / dim as f32 + eps).sqrt();
        for d in 0..dim {
            normed_out[off + d] = h_out[off + d] * inv_rms * rms_weight[d];
        }
    }
}

/// Fusion B: token embedding lookup + BigramHash lookup + bigram projection + add.
///
/// Computes (per token t, per dim d):
///   tok_emb_val   = embed_table[token_ids[t], d]
///   bigram_val    = bigram_table[hash(prev, cur), :]   (vector of length bigram_dim)
///   bigram_proj_v = sum over bd of bigram_val[bd] * bigram_proj[bd, d]
///   output[t, d]  = tok_emb_val + bigram_scale * bigram_proj_v
///
/// This is one CubeCL kernel: each output element does its own gather +
/// matmul-row, no global memory writes for intermediate `tok_emb` or
/// `bigram_emb` tensors.
#[allow(clippy::too_many_arguments)]
pub fn bigram_embed_fused(
    token_ids: &[u32],
    embed_table: &[f32],  // [vocab, dim]
    bigram_table: &[f32], // [num_buckets, bigram_dim]
    bigram_proj: &[f32],  // [dim, bigram_dim]  (row-major: [d * bigram_dim + bd])
    bigram_scale: f32,
    output: &mut [f32], // [tokens, dim]
    dim: usize,
    bigram_dim: usize,
    num_buckets: usize,
) {
    let tokens = token_ids.len();
    for t in 0..tokens {
        let prev = if t == 0 { None } else { Some(token_ids[t - 1]) };
        let bucket = bigram_hash(prev, token_ids[t], num_buckets);
        let bigram_row = &bigram_table[bucket * bigram_dim..(bucket + 1) * bigram_dim];
        let tok = token_ids[t] as usize;
        let tok_row = &embed_table[tok * dim..(tok + 1) * dim];
        let out_row = &mut output[t * dim..(t + 1) * dim];

        for d in 0..dim {
            // bigram_proj[d, :] · bigram_row
            let proj_row = &bigram_proj[d * bigram_dim..(d + 1) * bigram_dim];
            let mut acc = 0.0f32;
            for bd in 0..bigram_dim {
                acc += proj_row[bd] * bigram_row[bd];
            }
            out_row[d] = tok_row[d] + bigram_scale * acc;
        }
    }
}

/// Unfused reference for Fusion B: separate gather, separate gather, separate matmul, separate add.
#[allow(clippy::too_many_arguments)]
pub fn bigram_embed_unfused(
    token_ids: &[u32],
    embed_table: &[f32],
    bigram_table: &[f32],
    bigram_proj: &[f32],
    bigram_scale: f32,
    output: &mut [f32],
    dim: usize,
    bigram_dim: usize,
    num_buckets: usize,
) {
    let tokens = token_ids.len();

    // 1. tok_emb gather
    let mut tok_emb = vec![0.0f32; tokens * dim];
    for t in 0..tokens {
        let tok = token_ids[t] as usize;
        tok_emb[t * dim..(t + 1) * dim].copy_from_slice(&embed_table[tok * dim..(tok + 1) * dim]);
    }
    // 2. bigram gather
    let mut bigram_emb = vec![0.0f32; tokens * bigram_dim];
    for t in 0..tokens {
        let prev = if t == 0 { None } else { Some(token_ids[t - 1]) };
        let bucket = bigram_hash(prev, token_ids[t], num_buckets);
        bigram_emb[t * bigram_dim..(t + 1) * bigram_dim]
            .copy_from_slice(&bigram_table[bucket * bigram_dim..(bucket + 1) * bigram_dim]);
    }
    // 3. matmul: bigram_proj_out[t, d] = sum_bd bigram_emb[t, bd] * bigram_proj[d, bd]
    let mut bigram_proj_out = vec![0.0f32; tokens * dim];
    for t in 0..tokens {
        for d in 0..dim {
            let mut acc = 0.0f32;
            for bd in 0..bigram_dim {
                acc += bigram_emb[t * bigram_dim + bd] * bigram_proj[d * bigram_dim + bd];
            }
            bigram_proj_out[t * dim + d] = acc;
        }
    }
    // 4. add
    for i in 0..tokens * dim {
        output[i] = tok_emb[i] + bigram_scale * bigram_proj_out[i];
    }
}

/// Fusion C: RMSNorm + per-head Q/K post-projection RoPE + q_gain scaling.
///
/// Inputs are the *post-GEMM* Q and K tensors (already projected through
/// `q_weight` and `k_weight`). The fused kernel:
///   1. Takes each (token, head) vector of length `head_dim`
///   2. Applies RMSNorm across `head_dim` (QK norm)
///   3. Applies partial RoPE to the first `rope_dims` dimensions
///   4. Multiplies Q (only) by per-head `q_gain`
///   5. Writes the result back
///
/// In the GPU version, `rope_dims` and `head_dim` are `#[comptime]` params so
/// the inner loops are fully unrolled at compile time.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_qk_rope_qgain_fused(
    q: &mut [f32],     // [tokens, num_heads, head_dim]
    k: &mut [f32],     // [tokens, num_kv_heads, head_dim]
    cos_table: &[f32], // [seq_len, rope_dims/2]
    sin_table: &[f32], // [seq_len, rope_dims/2]
    q_gain: &[f32],    // [num_heads]
    tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dims: usize,
    eps: f32,
) {
    let half = rope_dims / 2;

    // ---- Q ----
    for t in 0..tokens {
        for h in 0..num_heads {
            let off = (t * num_heads + h) * head_dim;
            // 1. RMSNorm across head_dim
            let mut sq = 0.0f32;
            for d in 0..head_dim {
                sq += q[off + d] * q[off + d];
            }
            let inv_rms = 1.0 / (sq / head_dim as f32 + eps).sqrt();
            for d in 0..head_dim {
                q[off + d] *= inv_rms;
            }
            // 2. Partial RoPE on first rope_dims dimensions
            let rope_off = t * half;
            for i in 0..half {
                let c = cos_table[rope_off + i];
                let s = sin_table[rope_off + i];
                let x1 = q[off + i];
                let x2 = q[off + half + i];
                q[off + i] = x1 * c + x2 * s;
                q[off + half + i] = -x1 * s + x2 * c;
            }
            // 3. q_gain
            let g = q_gain[h];
            for d in 0..head_dim {
                q[off + d] *= g;
            }
        }
    }

    // ---- K (no q_gain) ----
    for t in 0..tokens {
        for h in 0..num_kv_heads {
            let off = (t * num_kv_heads + h) * head_dim;
            let mut sq = 0.0f32;
            for d in 0..head_dim {
                sq += k[off + d] * k[off + d];
            }
            let inv_rms = 1.0 / (sq / head_dim as f32 + eps).sqrt();
            for d in 0..head_dim {
                k[off + d] *= inv_rms;
            }
            let rope_off = t * half;
            for i in 0..half {
                let c = cos_table[rope_off + i];
                let s = sin_table[rope_off + i];
                let x1 = k[off + i];
                let x2 = k[off + half + i];
                k[off + i] = x1 * c + x2 * s;
                k[off + half + i] = -x1 * s + x2 * c;
            }
        }
    }
}

/// Unfused reference: explicit RMSNorm → RoPE → q_gain in three passes.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_qk_rope_qgain_unfused(
    q: &mut [f32],
    k: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    q_gain: &[f32],
    tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dims: usize,
    eps: f32,
) {
    // 1. Standalone RMSNorm on Q and K (per head) using rms_norm_forward_cpu.
    {
        let buf = q.to_vec();
        crate::rms_norm::rms_norm_forward_cpu(&buf, q, head_dim, 1.0, eps);
    }
    {
        let buf = k.to_vec();
        crate::rms_norm::rms_norm_forward_cpu(&buf, k, head_dim, 1.0, eps);
    }
    // 2. Standalone partial RoPE on Q and K using apply_partial_rope.
    crate::rope::apply_partial_rope(
        q, cos_table, sin_table, 1, tokens, num_heads, head_dim, rope_dims,
    );
    crate::rope::apply_partial_rope(
        k,
        cos_table,
        sin_table,
        1,
        tokens,
        num_kv_heads,
        head_dim,
        rope_dims,
    );
    // 3. Standalone q_gain scaling.
    for t in 0..tokens {
        for h in 0..num_heads {
            let off = (t * num_heads + h) * head_dim;
            let g = q_gain[h];
            for d in 0..head_dim {
                q[off + d] *= g;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope::precompute_rope_tables;

    fn approx_eq_slice(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", label);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "{}: index {} differs: fused={} unfused={}",
                label,
                i,
                x,
                y
            );
        }
    }

    fn deterministic(seed: u32, n: usize, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s as f32 / u32::MAX as f32) - 0.5) * scale
            })
            .collect()
    }

    #[test]
    fn test_fusion_a_xsa_residual_norm_matches_unfused() {
        let tokens = 6;
        let dim = 16;
        let attn = deterministic(1, tokens * dim, 1.0);
        let v = deterministic(2, tokens * dim, 1.0);
        let residual = deterministic(3, tokens * dim, 0.5);
        let rms_w = deterministic(4, dim, 0.2)
            .iter()
            .map(|x| x + 1.0)
            .collect::<Vec<_>>();
        let ln_scale = 0.7;

        let mut h_a = vec![0.0f32; tokens * dim];
        let mut n_a = vec![0.0f32; tokens * dim];
        let mut h_b = vec![0.0f32; tokens * dim];
        let mut n_b = vec![0.0f32; tokens * dim];

        xsa_residual_norm_fused(
            &attn, &v, &residual, &rms_w, ln_scale, &mut h_a, &mut n_a, tokens, dim, 1e-6,
        );
        xsa_residual_norm_unfused(
            &attn, &v, &residual, &rms_w, ln_scale, &mut h_b, &mut n_b, tokens, dim, 1e-6,
        );

        approx_eq_slice(&h_a, &h_b, 1e-5, "h_out");
        approx_eq_slice(&n_a, &n_b, 1e-5, "normed_out");
    }

    #[test]
    fn test_fusion_b_bigram_embed_matches_unfused() {
        let tokens = 8;
        let vocab = 16;
        let dim = 12;
        let bigram_dim = 6;
        let num_buckets = 32;

        let token_ids: Vec<u32> = (0..tokens as u32)
            .map(|i| (i * 3 + 1) % vocab as u32)
            .collect();
        let embed = deterministic(11, vocab * dim, 0.3);
        let bigram_table = deterministic(12, num_buckets * bigram_dim, 0.4);
        let bigram_proj = deterministic(13, dim * bigram_dim, 0.2);
        let bigram_scale = 0.05;

        let mut a = vec![0.0f32; tokens * dim];
        let mut b = vec![0.0f32; tokens * dim];

        bigram_embed_fused(
            &token_ids,
            &embed,
            &bigram_table,
            &bigram_proj,
            bigram_scale,
            &mut a,
            dim,
            bigram_dim,
            num_buckets,
        );
        bigram_embed_unfused(
            &token_ids,
            &embed,
            &bigram_table,
            &bigram_proj,
            bigram_scale,
            &mut b,
            dim,
            bigram_dim,
            num_buckets,
        );

        approx_eq_slice(&a, &b, 1e-5, "bigram_embed_output");
    }

    #[test]
    fn test_fusion_c_rmsnorm_qk_rope_qgain_matches_unfused() {
        let tokens = 4;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let rope_dims = 4;
        let eps = 1e-6;

        let q = deterministic(21, tokens * num_heads * head_dim, 1.0);
        let k = deterministic(22, tokens * num_kv_heads * head_dim, 1.0);
        let q_gain = vec![1.5f32, 0.8];

        let (cos_table, sin_table) = precompute_rope_tables(tokens, rope_dims, 10000.0);

        let mut q_a = q.clone();
        let mut k_a = k.clone();
        let mut q_b = q.clone();
        let mut k_b = k.clone();

        rmsnorm_qk_rope_qgain_fused(
            &mut q_a,
            &mut k_a,
            &cos_table,
            &sin_table,
            &q_gain,
            tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            eps,
        );
        rmsnorm_qk_rope_qgain_unfused(
            &mut q_b,
            &mut k_b,
            &cos_table,
            &sin_table,
            &q_gain,
            tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            eps,
        );

        approx_eq_slice(&q_a, &q_b, 1e-5, "q post-fusion");
        approx_eq_slice(&k_a, &k_b, 1e-5, "k post-fusion");
    }
}
