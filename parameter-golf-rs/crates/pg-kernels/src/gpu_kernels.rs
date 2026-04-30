use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, PushKernelArg};
use pg_core::error::{PgError, PgResult};
/// GPU kernels for Parameter Golf — all element-wise operations.
///
/// Strategy: CUDA C source strings compiled to PTX at runtime via NVRTC,
/// loaded via `CudaContext::load_module()`, launched via `CudaStream::launch_builder()`.
///
/// All kernels operate on f32 data with f32 arithmetic internally.
/// Grid/block dimensions are set for H100 (132 SMs, 1024 threads/block).
use std::sync::{Arc, OnceLock};

/// Compiled GPU kernel module — initialized once, reused for all launches.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct CudaPtr(pub u64);
unsafe impl cudarc::driver::DeviceRepr for CudaPtr {}

pub struct GpuKernels {
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    // Element-wise kernels
    rms_norm_fwd: CudaFunction,
    rms_norm_fwd_bf16: CudaFunction,
    rms_norm_bwd: CudaFunction,
    rms_norm_bwd_go_bf16: CudaFunction,
    rms_norm_bwd_accum: CudaFunction,
    rms_norm_bwd_accum_residual_mix_input: CudaFunction,
    rms_norm_bwd_accum_residual_mix_input_go_bf16: CudaFunction,
    rms_norm_bwd_residual_mix: CudaFunction,
    rms_norm_bwd_residual_mix_chunked: CudaFunction,
    rms_norm_bwd_residual_mix_no_mix_grad: CudaFunction,
    rms_norm_bwd_residual_mix_recompute: CudaFunction,
    residual_mix_grad_reduce_chunks: CudaFunction,
    residual_mix_grad_reduce_from_stats: CudaFunction,
    leaky_relu_sq_fwd: CudaFunction,
    leaky_relu_sq_fwd_bf16: CudaFunction,
    leaky_relu_sq_fwd_x_bf16_only: CudaFunction,
    leaky_relu_sq_bwd: CudaFunction,
    leaky_relu_sq_bwd_bf16: CudaFunction,
    leaky_relu_sq_bwd_x_bf16: CudaFunction,
    leaky_relu_sq_bwd_x_bf16_only: CudaFunction,
    residual_mix: CudaFunction,
    residual_mix_rms_norm_fwd: CudaFunction,
    residual_mix_rms_norm_fwd_bf16: CudaFunction,
    residual_mix_bwd: CudaFunction,
    residual_add_scale: CudaFunction,
    residual_add_scale_bf16_proj: CudaFunction,
    residual_add_scale_from_base: CudaFunction,
    residual_add_scale_from_base_bf16_proj: CudaFunction,
    residual_add_scale_from_base_rms_norm: CudaFunction,
    residual_add_scale_from_base_rms_norm_bf16: CudaFunction,
    residual_add_scale_from_base_rms_norm_bf16_proj: CudaFunction,
    residual_add_scale_bwd: CudaFunction,
    residual_add_scale_bwd_bf16: CudaFunction,
    residual_add_scale_bwd_from_bf16_only: CudaFunction,
    residual_add_scale_bwd_bf16_only_atomic: CudaFunction,
    residual_add_scale_bwd_bf16_only_no_atomic: CudaFunction,
    residual_add_scale_grad_scale_reduce: CudaFunction,
    residual_add_scale_grad_scale_reduce_bf16: CudaFunction,
    smear_gate_fwd: CudaFunction,
    smear_gate_fwd_boundary: CudaFunction,
    smear_gate_bwd: CudaFunction,
    smear_gate_bwd_boundary: CudaFunction,
    embedding_gather: CudaFunction,
    embedding_gather_bwd: CudaFunction,
    qk_norm_fwd: CudaFunction,
    qk_norm_bwd: CudaFunction,
    partial_rope_fwd: CudaFunction,
    partial_rope_bwd: CudaFunction,
    q_gain_fwd: CudaFunction,
    q_gain_bwd: CudaFunction,
    q_gain_rope_qk_norm_fwd: CudaFunction,
    q_gain_rope_qk_norm_fwd_bf16_bhsd: CudaFunction,
    rope_qk_norm_fwd: CudaFunction,
    rope_qk_norm_fwd_bf16_bhsd: CudaFunction,
    q_gain_rope_qk_norm_bwd: CudaFunction,
    q_gain_rope_qk_norm_bwd_chunked: CudaFunction,
    q_gain_reduce_chunked: CudaFunction,
    rope_qk_norm_bwd: CudaFunction,
    dot_accumulate: CudaFunction,
    dot_accumulate_by_param: CudaFunction,
    clip_by_global_norm: CudaFunction,
    cross_entropy_fwd: CudaFunction,
    cross_entropy_fwd_bf16_logits: CudaFunction,
    cross_entropy_bwd: CudaFunction,
    cross_entropy_bwd_bf16: CudaFunction,
    cross_entropy_bwd_bf16_logits: CudaFunction,
    cross_entropy_loss_bwd_bf16: CudaFunction,
    cross_entropy_loss_bwd_bf16_logits: CudaFunction,
    output_ce_stats_init: CudaFunction,
    output_ce_tile_stats_update: CudaFunction,
    output_ce_finalize_loss: CudaFunction,
    loss_window_sum: CudaFunction,
    loss_window_accumulate: CudaFunction,
    output_ce_tile_grad_bf16: CudaFunction,
    bigram_hash_embed: CudaFunction,
    bigram_hash_embed_bwd: CudaFunction,
    add_scaled: CudaFunction,
    add_scaled_by_param: CudaFunction,
    add_scaled_by_param_index: CudaFunction,
    add_scaled_by_param_product: CudaFunction,
    causal_attention_naive: CudaFunction,
    causal_attention_naive_bwd: CudaFunction,
    causal_attention_online: CudaFunction,
    causal_attention_online_bwd: CudaFunction,
    attn_out_gate_fwd: CudaFunction,
    attn_out_gate_bwd: CudaFunction,
    sparse_attn_gate_fwd: CudaFunction,
    sparse_attn_gate_xsa_fwd: CudaFunction,
    sparse_attn_gate_xsa_fwd_bf16_bhsd: CudaFunction,
    sparse_attn_gate_bwd: CudaFunction,
    sparse_attn_gate_bwd_stage1: CudaFunction,
    sparse_attn_gate_xsa_bwd_bf16_bhsd: CudaFunction,
    sparse_attn_gate_xsa_bwd_bf16_bhsd_warpheads: CudaFunction,
    sparse_attn_gate_weight_grad: CudaFunction,
    xsa_fwd: CudaFunction,
    xsa_bwd: CudaFunction,
    xsa_bwd_bf16_bhsd: CudaFunction,
    copy_fwd: CudaFunction,
    copy_u16_fwd: CudaFunction,
    fill_u16: CudaFunction,
    bthd_to_bhsd_bf16: CudaFunction,
    bhsd_to_bthd_bf16: CudaFunction,
    scale_inplace: CudaFunction,
    pack_qkv_weights: CudaFunction,
    unpack_qkv_output: CudaFunction,
    pack_qkv_grads: CudaFunction,
    pack_qkv_grads_bf16: CudaFunction,
    unpack_qkv_weight_grad: CudaFunction,
    f32_to_bf16: CudaFunction,
    bf16_to_f32: CudaFunction,
    normalize_matrices: CudaFunction,
    decay_sgd_step: CudaFunction,
    adamw_step: CudaFunction,
}

/// CUDA C source for all element-wise kernels.
/// f32 data, f32 compute, one thread per element (or per row for reductions).
const CUDA_SOURCE: &str = r#"
__device__ __forceinline__ unsigned short pg_f32_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounding_bias = 0x7fffu + lsb;
    return static_cast<unsigned short>((bits + rounding_bias) >> 16);
}

__device__ __forceinline__ float pg_bf16_to_f32(unsigned short value) {
    unsigned int bits = static_cast<unsigned int>(value) << 16;
    return __uint_as_float(bits);
}

// ──────────────────────────────────────────────────────────────
// Copy forward: dst[idx] = src[idx]
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void copy_fwd(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

extern "C" __global__ void copy_u16_fwd(
    const unsigned short* __restrict__ src,
    unsigned short* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

extern "C" __global__ void fill_u16(
    unsigned short* __restrict__ dst,
    unsigned short value,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = value;
    }
}

// ──────────────────────────────────────────────────────────────
// Pack per-layer Q/K/V projection banks into one [layers, d+2kv, d]
// shadow bank. Authoritative params remain qo_bank/kv_bank; this is a
// tensor-core GEMM layout optimization for the hot path.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void pack_qkv_weights(
    const float* __restrict__ qo_bank,
    const float* __restrict__ kv_bank,
    float* __restrict__ qkv_bank,
    int layers,
    int d,
    int kv
) {
    int qkv = d + 2 * kv;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = layers * qkv * d;
    if (idx >= total) return;

    int col = idx % d;
    int row = (idx / d) % qkv;
    int layer = idx / (qkv * d);
    float value;
    if (row < d) {
        value = qo_bank[(layer * d + row) * d + col];
    } else if (row < d + kv) {
        int k_row = row - d;
        value = kv_bank[(layer * kv + k_row) * d + col];
    } else {
        int v_row = row - d - kv;
        value = kv_bank[((layers + layer) * kv + v_row) * d + col];
    }
    qkv_bank[idx] = value;
}

// combined[t, d+2kv] -> q[t,d], k[t,kv], v[t,kv]
extern "C" __global__ void unpack_qkv_output(
    const float* __restrict__ combined,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int tokens,
    int d,
    int kv
) {
    int qkv = d + 2 * kv;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * qkv;
    if (idx >= total) return;

    int row = idx / qkv;
    int col = idx - row * qkv;
    float value = combined[idx];
    if (col < d) {
        q[row * d + col] = value;
    } else if (col < d + kv) {
        k[row * kv + (col - d)] = value;
    } else {
        v[row * kv + (col - d - kv)] = value;
    }
}

// q/k/v grads -> combined[t, d+2kv]
extern "C" __global__ void pack_qkv_grads(
    const float* __restrict__ grad_q,
    const float* __restrict__ grad_k,
    const float* __restrict__ grad_v,
    float* __restrict__ combined,
    int tokens,
    int d,
    int kv
) {
    int qkv = d + 2 * kv;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * qkv;
    if (idx >= total) return;

    int row = idx / qkv;
    int col = idx - row * qkv;
    if (col < d) {
        combined[idx] = grad_q[row * d + col];
    } else if (col < d + kv) {
        combined[idx] = grad_k[row * kv + (col - d)];
    } else {
        combined[idx] = grad_v[row * kv + (col - d - kv)];
    }
}

// q/k/v f32 grads -> combined BF16[t, d+2kv]. This removes a
// separate full-size f32_to_bf16 pass from the fused QKV backward path.
extern "C" __global__ void pack_qkv_grads_bf16(
    const float* __restrict__ grad_q,
    const float* __restrict__ grad_k,
    const float* __restrict__ grad_v,
    unsigned short* __restrict__ combined,
    int tokens,
    int d,
    int kv
) {
    int qkv = d + 2 * kv;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * qkv;
    if (idx >= total) return;

    int row = idx / qkv;
    int col = idx - row * qkv;
    float value;
    if (col < d) {
        value = grad_q[row * d + col];
    } else if (col < d + kv) {
        value = grad_k[row * kv + (col - d)];
    } else {
        value = grad_v[row * kv + (col - d - kv)];
    }
    combined[idx] = pg_f32_to_bf16(value);
}

// combined grad [d+2kv,d] -> qo_grad[layer], kv_grad[layer], kv_grad[layers+layer]
extern "C" __global__ void unpack_qkv_weight_grad(
    const float* __restrict__ combined_grad,
    float* __restrict__ qo_grad,
    float* __restrict__ kv_grad,
    int layer,
    int layers,
    int d,
    int kv
) {
    int qkv = d + 2 * kv;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = qkv * d;
    if (idx >= total) return;

    int col = idx % d;
    int row = idx / d;
    float value = combined_grad[idx];
    if (row < d) {
        qo_grad[(layer * d + row) * d + col] = value;
    } else if (row < d + kv) {
        int k_row = row - d;
        kv_grad[(layer * kv + k_row) * d + col] = value;
    } else {
        int v_row = row - d - kv;
        kv_grad[((layers + layer) * kv + v_row) * d + col] = value;
    }
}

// ──────────────────────────────────────────────────────────────
// Scale in-place: x[i] *= scale
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void scale_inplace(
    float* __restrict__ x,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (scale == 0.0f) {
            x[idx] = 0.0f;
        } else {
            x[idx] *= scale;
        }
    }
}

// ──────────────────────────────────────────────────────────────
// BF16 conversion kernels. BF16 is stored as raw u16 so this file
// does not depend on cuda_bf16.h availability under NVRTC.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void f32_to_bf16(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = pg_f32_to_bf16(src[idx]);
    }
}

extern "C" __global__ void bf16_to_f32(
    const unsigned short* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = pg_bf16_to_f32(src[idx]);
    }
}

// ──────────────────────────────────────────────────────────────
// Per-matrix L2 normalization for rank-3 banks:
// dst[b, :, :] = src[b, :, :] / (||src[b, :, :]||_2 + eps)
// One block handles one matrix slice.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void normalize_matrices(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int matrix_numel,
    int batch,
    float eps
) {
    int b = blockIdx.x;
    if (b >= batch) return;
    int tid = threadIdx.x;
    int base = b * matrix_numel;

    float sum = 0.0f;
    for (int i = tid; i < matrix_numel; i += blockDim.x) {
        float v = src[base + i];
        sum += v * v;
    }

    __shared__ float shared[256];
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    float inv_norm = 1.0f / (sqrtf(shared[0]) + eps);
    for (int i = tid; i < matrix_numel; i += blockDim.x) {
        dst[base + i] = src[base + i] * inv_norm;
    }
}

// ──────────────────────────────────────────────────────────────
// Decoupled weight decay + SGD step:
// param = (1 - lr * wd) * param - lr * grad
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void decay_sgd_step(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float lr,
    float weight_decay,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float decay = 1.0f - lr * weight_decay;
    param[idx] = decay * param[idx] - lr * grad[idx];
}

// ──────────────────────────────────────────────────────────────
// Fused AdamW step.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void adamw_step(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float bc1,
    float bc2,
    float eps,
    float weight_decay,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad[idx];
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = m_new;
    v[idx] = v_new;

    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;
    if (weight_decay > 0.0f) {
        param[idx] *= 1.0f - lr * weight_decay;
    }
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// ──────────────────────────────────────────────────────────────
// RMSNorm forward: y[row] = (x[row] / rms) * ln_scale_factor
// Grid: (num_rows,), Block: (block_dim,) — warp reduction per row
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void rms_norm_forward(
    const float* __restrict__ x,
    float* __restrict__ y,
    int dim,
    float ln_scale_factor,
    float eps,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* x_row = x + row * dim;
    float* y_row = y + row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    // Cross-warp reduction via shared memory
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 32) {
        sum_sq = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float inv_rms_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
    }
    __syncthreads();
    float inv_rms = inv_rms_shared;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        y_row[i] = x_row[i] * inv_rms;
    }
}

extern "C" __global__ void rms_norm_forward_bf16(
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned short* __restrict__ y_bf16,
    int dim,
    float ln_scale_factor,
    float eps,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* x_row = x + row * dim;
    float* y_row = y + row * dim;
    unsigned short* y_bf16_row = y_bf16 + row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 32) {
        sum_sq = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float inv_rms_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
    }
    __syncthreads();
    float inv_rms = inv_rms_shared;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float out = x_row[i] * inv_rms;
        y_row[i] = out;
        y_bf16_row[i] = pg_f32_to_bf16(out);
    }
}

// ──────────────────────────────────────────────────────────────
// RMSNorm backward: dx = s/rms * (dy - x * dot(x,dy)/(rms²*dim))
// Grid: (num_rows,), Block: (block_dim,)
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void rms_norm_backward(
    const float* __restrict__ x,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int dim,
    float ln_scale_factor,
    float eps,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* x_row = x + row * dim;
    const float* go_row = grad_output + row * dim;
    float* gi_row = grad_input + row * dim;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = x_row[i];
        float go = go_row[i];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float shared_sum[32];
    __shared__ float shared_dot[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
        shared_dot[warp_id] = x_dot_go;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        x_dot_go = (threadIdx.x < warps) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        }
    }

    __shared__ float inv_rms_shared;
    __shared__ float coeff_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
        coeff_shared = x_dot_go / (rms * rms * (float)dim);
    }
    __syncthreads();

    float inv_rms = inv_rms_shared;
    float coeff = coeff_shared;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        gi_row[i] = inv_rms * (go_row[i] - x_row[i] * coeff);
    }
}

extern "C" __global__ void rms_norm_backward_go_bf16(
    const float* __restrict__ x,
    const unsigned short* __restrict__ grad_output_bf16,
    float* __restrict__ grad_input,
    int dim,
    float ln_scale_factor,
    float eps,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* x_row = x + row * dim;
    const unsigned short* go_row = grad_output_bf16 + row * dim;
    float* gi_row = grad_input + row * dim;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = x_row[i];
        float go = pg_bf16_to_f32(go_row[i]);
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float shared_sum[32];
    __shared__ float shared_dot[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
        shared_dot[warp_id] = x_dot_go;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        x_dot_go = (threadIdx.x < warps) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        }
    }

    __shared__ float inv_rms_shared;
    __shared__ float coeff_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
        coeff_shared = x_dot_go / (rms * rms * (float)dim);
    }
    __syncthreads();

    float inv_rms = inv_rms_shared;
    float coeff = coeff_shared;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float go = pg_bf16_to_f32(go_row[i]);
        gi_row[i] = inv_rms * (go - x_row[i] * coeff);
    }
}

// RMSNorm backward with accumulation:
// grad_input = beta * grad_input + dx_norm
extern "C" __global__ void rms_norm_backward_accum(
    const float* __restrict__ x,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int dim,
    float ln_scale_factor,
    float eps,
    float beta,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* x_row = x + row * dim;
    const float* go_row = grad_output + row * dim;
    float* gi_row = grad_input + row * dim;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = x_row[i];
        float go = go_row[i];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float shared_sum[32];
    __shared__ float shared_dot[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
        shared_dot[warp_id] = x_dot_go;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        x_dot_go = (threadIdx.x < warps) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        }
    }

    __shared__ float inv_rms_shared;
    __shared__ float coeff_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
        coeff_shared = x_dot_go / (rms * rms * (float)dim);
    }
    __syncthreads();

    float inv_rms = inv_rms_shared;
    float coeff = coeff_shared;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float dx = inv_rms * (go_row[i] - x_row[i] * coeff);
        gi_row[i] = beta * gi_row[i] + dx;
    }
}

// RMSNorm backward with accumulation, recomputing the RMSNorm input from the
// residual-mix sources. This removes the need to save the full mixed activation
// when the forward input is cheaply reproducible from x, x0, and mix.
extern "C" __global__ void rms_norm_backward_accum_residual_mix_input(
    const float* __restrict__ residual_x,
    const float* __restrict__ residual_x0,
    const float* __restrict__ mix,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int dim,
    float ln_scale_factor,
    float eps,
    float beta,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    int base = row * dim;
    const float* x_row = residual_x + base;
    const float* x0_row = residual_x0 + base;
    const float* go_row = grad_output + base;
    float* gi_row = grad_input + base;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = mix[i] * x_row[i] + mix[dim + i] * x0_row[i];
        float go = go_row[i];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float shared_sum[32];
    __shared__ float shared_dot[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
        shared_dot[warp_id] = x_dot_go;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        x_dot_go = (threadIdx.x < warps) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        }
    }

    __shared__ float inv_rms_shared;
    __shared__ float coeff_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
        coeff_shared = x_dot_go / (rms * rms * (float)dim);
    }
    __syncthreads();

    float inv_rms = inv_rms_shared;
    float coeff = coeff_shared;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = mix[i] * x_row[i] + mix[dim + i] * x0_row[i];
        float dx = inv_rms * (go_row[i] - xv * coeff);
        gi_row[i] = beta * gi_row[i] + dx;
    }
}

extern "C" __global__ void rms_norm_backward_accum_residual_mix_input_go_bf16(
    const float* __restrict__ residual_x,
    const float* __restrict__ residual_x0,
    const float* __restrict__ mix,
    const unsigned short* __restrict__ grad_output_bf16,
    float* __restrict__ grad_input,
    int dim,
    float ln_scale_factor,
    float eps,
    float beta,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    int base = row * dim;
    const float* x_row = residual_x + base;
    const float* x0_row = residual_x0 + base;
    const unsigned short* go_row = grad_output_bf16 + base;
    float* gi_row = grad_input + base;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = mix[i] * x_row[i] + mix[dim + i] * x0_row[i];
        float go = pg_bf16_to_f32(go_row[i]);
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float shared_sum[32];
    __shared__ float shared_dot[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
        shared_dot[warp_id] = x_dot_go;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        x_dot_go = (threadIdx.x < warps) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        }
    }

    __shared__ float inv_rms_shared;
    __shared__ float coeff_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
        coeff_shared = x_dot_go / (rms * rms * (float)dim);
    }
    __syncthreads();

    float inv_rms = inv_rms_shared;
    float coeff = coeff_shared;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = mix[i] * x_row[i] + mix[dim + i] * x0_row[i];
        float go = pg_bf16_to_f32(go_row[i]);
        float dx = inv_rms * (go - xv * coeff);
        gi_row[i] = beta * gi_row[i] + dx;
    }
}

// RMSNorm backward fused with the following residual-mix backward.
// Computes:
//   norm_dx = RMSNormBwd(x_norm, grad_norm)
//   go = beta * base_grad + norm_dx
//   residual_mix_backward(residual_x, residual_x0, go, mix)
// This avoids materializing go for the common block-backward tail.
extern "C" __global__ void rms_norm_backward_accum_residual_mix_backward(
    const float* __restrict__ x_norm,
    const float* __restrict__ grad_norm,
    const float* __restrict__ residual_x,
    const float* __restrict__ residual_x0,
    const float* __restrict__ mix,
    const float* __restrict__ base_grad,
    float* __restrict__ grad_x,
    float* __restrict__ grad_x0,
    float* __restrict__ grad_mix,
    int dim,
    float ln_scale_factor,
    float eps,
    float beta,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* x_row = x_norm + row * dim;
    const float* go_row = grad_norm + row * dim;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = x_row[i];
        float go = go_row[i];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float shared_sum[32];
    __shared__ float shared_dot[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
        shared_dot[warp_id] = x_dot_go;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        x_dot_go = (threadIdx.x < warps) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        }
    }

    __shared__ float inv_rms_shared;
    __shared__ float coeff_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
        coeff_shared = x_dot_go / (rms * rms * (float)dim);
    }
    __syncthreads();

    float inv_rms = inv_rms_shared;
    float coeff = coeff_shared;
    int base = row * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        int idx = base + i;
        float norm_dx = inv_rms * (grad_norm[idx] - x_norm[idx] * coeff);
        float go = beta * base_grad[idx] + norm_dx;
        float mix_x = mix[i];
        float mix_x0 = mix[dim + i];
        grad_x[idx] = go * mix_x;
        grad_x0[idx] += go * mix_x0;
        atomicAdd(&grad_mix[i], go * residual_x[idx]);
        atomicAdd(&grad_mix[dim + i], go * residual_x0[idx]);
    }
}

extern "C" __global__ void rms_norm_backward_accum_residual_mix_backward_chunked(
    const float* __restrict__ x_norm,
    const float* __restrict__ grad_norm,
    const float* __restrict__ residual_x,
    const float* __restrict__ residual_x0,
    const float* __restrict__ mix,
    const float* __restrict__ base_grad,
    float* __restrict__ grad_x,
    float* __restrict__ grad_x0,
    float* __restrict__ grad_mix_chunks,
    int dim,
    int num_chunks,
    int chunk_rows,
    float ln_scale_factor,
    float eps,
    float beta,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* x_row = x_norm + row * dim;
    const float* go_row = grad_norm + row * dim;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = x_row[i];
        float go = go_row[i];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float shared_sum[32];
    __shared__ float shared_dot[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
        shared_dot[warp_id] = x_dot_go;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        x_dot_go = (threadIdx.x < warps) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        }
    }

    __shared__ float inv_rms_shared;
    __shared__ float coeff_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
        coeff_shared = x_dot_go / (rms * rms * (float)dim);
    }
    __syncthreads();

    float inv_rms = inv_rms_shared;
    float coeff = coeff_shared;
    int base = row * dim;
    int chunk = row / chunk_rows;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        int idx = base + i;
        float norm_dx = inv_rms * (grad_norm[idx] - x_norm[idx] * coeff);
        float go = beta * base_grad[idx] + norm_dx;
        float mix_x = mix[i];
        float mix_x0 = mix[dim + i];
        grad_x[idx] = go * mix_x;
        grad_x0[idx] += go * mix_x0;
        atomicAdd(&grad_mix_chunks[i * num_chunks + chunk], go * residual_x[idx]);
        atomicAdd(&grad_mix_chunks[(dim + i) * num_chunks + chunk], go * residual_x0[idx]);
    }
}

extern "C" __global__ void residual_mix_grad_reduce_chunks(
    const float* __restrict__ grad_mix_chunks,
    float* __restrict__ grad_mix,
    int dim2,
    int num_chunks
) {
    int d = blockIdx.x;
    if (d >= dim2) return;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < num_chunks; c += blockDim.x) {
        sum += grad_mix_chunks[d * num_chunks + c];
    }
    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&grad_mix[d], scratch[0]);
    }
}

extern "C" __global__ void rms_norm_backward_accum_residual_mix_backward_no_mix_grad(
    const float* __restrict__ x_norm,
    const float* __restrict__ grad_norm,
    const float* __restrict__ residual_x,
    const float* __restrict__ residual_x0,
    const float* __restrict__ mix,
    const float* __restrict__ base_grad,
    float* __restrict__ grad_x,
    float* __restrict__ grad_x0,
    float* __restrict__ row_norm_stats,
    int dim,
    float ln_scale_factor,
    float eps,
    float beta,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* x_row = x_norm + row * dim;
    const float* go_row = grad_norm + row * dim;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = x_row[i];
        float go = go_row[i];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float shared_sum[32];
    __shared__ float shared_dot[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
        shared_dot[warp_id] = x_dot_go;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        x_dot_go = (threadIdx.x < warps) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        }
    }

    __shared__ float inv_rms_shared;
    __shared__ float coeff_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
        coeff_shared = x_dot_go / (rms * rms * (float)dim);
        row_norm_stats[row * 2] = inv_rms_shared;
        row_norm_stats[row * 2 + 1] = coeff_shared;
    }
    __syncthreads();

    float inv_rms = inv_rms_shared;
    float coeff = coeff_shared;
    int base = row * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        int idx = base + i;
        float norm_dx = inv_rms * (grad_norm[idx] - x_norm[idx] * coeff);
        float go = beta * base_grad[idx] + norm_dx;
        float mix_x = mix[i];
        float mix_x0 = mix[dim + i];
        grad_x[idx] = go * mix_x;
        grad_x0[idx] += go * mix_x0;
    }
}

extern "C" __global__ void residual_mix_grad_reduce_from_stats(
    const float* __restrict__ x_norm,
    const float* __restrict__ grad_norm,
    const float* __restrict__ residual_x,
    const float* __restrict__ residual_x0,
    const float* __restrict__ base_grad,
    const float* __restrict__ row_norm_stats,
    float* __restrict__ grad_mix,
    int dim,
    int num_rows,
    int chunk_rows,
    float beta
) {
    int param = blockIdx.x;
    int chunk = blockIdx.y;
    int dim2 = 2 * dim;
    if (param >= dim2) return;
    int feature = param < dim ? param : param - dim;
    bool use_x0 = param >= dim;
    int row_start = chunk * chunk_rows;
    int row_end = min(row_start + chunk_rows, num_rows);

    float sum = 0.0f;
    for (int row = row_start + threadIdx.x; row < row_end; row += blockDim.x) {
        int idx = row * dim + feature;
        float inv_rms = row_norm_stats[row * 2];
        float coeff = row_norm_stats[row * 2 + 1];
        float norm_dx = inv_rms * (grad_norm[idx] - x_norm[idx] * coeff);
        float go = beta * base_grad[idx] + norm_dx;
        float residual = use_x0 ? residual_x0[idx] : residual_x[idx];
        sum += go * residual;
    }

    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&grad_mix[param], scratch[0]);
    }
}

// Same fused tail as above, but recomputes x_norm from the residual-mix inputs
// instead of reading a saved mixed activation.
extern "C" __global__ void rms_norm_backward_accum_residual_mix_backward_recompute(
    const float* __restrict__ grad_norm,
    const float* __restrict__ residual_x,
    const float* __restrict__ residual_x0,
    const float* __restrict__ mix,
    const float* __restrict__ base_grad,
    float* __restrict__ grad_x,
    float* __restrict__ grad_x0,
    float* __restrict__ grad_mix,
    int dim,
    float ln_scale_factor,
    float eps,
    float beta,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    int base = row * dim;
    const float* residual_row = residual_x + base;
    const float* residual0_row = residual_x0 + base;
    const float* go_row = grad_norm + base;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xv = mix[i] * residual_row[i] + mix[dim + i] * residual0_row[i];
        float go = go_row[i];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float shared_sum[32];
    __shared__ float shared_dot[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
        shared_dot[warp_id] = x_dot_go;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        x_dot_go = (threadIdx.x < warps) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        }
    }

    __shared__ float inv_rms_shared;
    __shared__ float coeff_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
        coeff_shared = x_dot_go / (rms * rms * (float)dim);
    }
    __syncthreads();

    float inv_rms = inv_rms_shared;
    float coeff = coeff_shared;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        int idx = base + i;
        float xv = mix[i] * residual_row[i] + mix[dim + i] * residual0_row[i];
        float norm_dx = inv_rms * (grad_norm[idx] - xv * coeff);
        float go = beta * base_grad[idx] + norm_dx;
        float mix_x = mix[i];
        float mix_x0 = mix[dim + i];
        grad_x[idx] = go * mix_x;
        grad_x0[idx] += go * mix_x0;
        atomicAdd(&grad_mix[i], go * residual_x[idx]);
        atomicAdd(&grad_mix[dim + i], go * residual_x0[idx]);
    }
}

// ──────────────────────────────────────────────────────────────
// LeakyReLU(0.5)² forward: y = leaky_relu(x, 0.5)²
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void leaky_relu_sq_forward(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    float lr = (v >= 0.0f) ? v : 0.5f * v;
    y[idx] = lr * lr;
}

extern "C" __global__ void leaky_relu_sq_forward_bf16(
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned short* __restrict__ y_bf16,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    float lr = (v >= 0.0f) ? v : 0.5f * v;
    float out = lr * lr;
    y[idx] = out;
    y_bf16[idx] = pg_f32_to_bf16(out);
}

extern "C" __global__ void leaky_relu_sq_forward_x_bf16_only(
    const unsigned short* __restrict__ x_bf16,
    unsigned short* __restrict__ y_bf16,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = pg_bf16_to_f32(x_bf16[idx]);
    float lr = (v >= 0.0f) ? v : 0.5f * v;
    y_bf16[idx] = pg_f32_to_bf16(lr * lr);
}

// ──────────────────────────────────────────────────────────────
// LeakyReLU(0.5)² backward
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void leaky_relu_sq_backward(
    const float* __restrict__ x,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    float lr = (v >= 0.0f) ? v : 0.5f * v;
    float d_lr = (v >= 0.0f) ? 1.0f : 0.5f;
    grad_input[idx] = grad_output[idx] * 2.0f * lr * d_lr;
}

extern "C" __global__ void leaky_relu_sq_backward_bf16(
    const float* __restrict__ x,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    unsigned short* __restrict__ grad_input_bf16,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    float lr = (v >= 0.0f) ? v : 0.5f * v;
    float d_lr = (v >= 0.0f) ? 1.0f : 0.5f;
    float grad = grad_output[idx] * 2.0f * lr * d_lr;
    grad_input[idx] = grad;
    grad_input_bf16[idx] = pg_f32_to_bf16(grad);
}

extern "C" __global__ void leaky_relu_sq_backward_x_bf16(
    const unsigned short* __restrict__ x_bf16,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    unsigned short* __restrict__ grad_input_bf16,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = pg_bf16_to_f32(x_bf16[idx]);
    float lr = (v >= 0.0f) ? v : 0.5f * v;
    float d_lr = (v >= 0.0f) ? 1.0f : 0.5f;
    float grad = grad_output[idx] * 2.0f * lr * d_lr;
    grad_input[idx] = grad;
    grad_input_bf16[idx] = pg_f32_to_bf16(grad);
}

extern "C" __global__ void leaky_relu_sq_backward_x_bf16_only(
    const unsigned short* __restrict__ x_bf16,
    const float* __restrict__ grad_output,
    unsigned short* __restrict__ grad_input_bf16,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = pg_bf16_to_f32(x_bf16[idx]);
    float lr = (v >= 0.0f) ? v : 0.5f * v;
    float d_lr = (v >= 0.0f) ? 1.0f : 0.5f;
    float grad = grad_output[idx] * 2.0f * lr * d_lr;
    grad_input_bf16[idx] = pg_f32_to_bf16(grad);
}

// ──────────────────────────────────────────────────────────────
// Residual mixing: out[i] = mix0[i%d] * x[i] + mix1[i%d] * x0[i]
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void residual_mix(
    const float* __restrict__ x,
    const float* __restrict__ x0,
    const float* __restrict__ mix,
    float* __restrict__ out,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    out[idx] = mix[d] * x[idx] + mix[dim + d] * x0[idx];
}

// ──────────────────────────────────────────────────────────────
// Fused residual mix + RMSNorm.
// Writes the mixed residual stream and the normalized stream in one row pass:
//   mixed = mix[:d] * x + mix[d:] * x0
//   norm  = mixed * ln_scale / sqrt(mean(mixed^2) + eps)
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void residual_mix_rms_norm_forward(
    const float* __restrict__ x,
    const float* __restrict__ x0,
    const float* __restrict__ mix,
    float* __restrict__ mixed,
    float* __restrict__ norm,
    int dim,
    float ln_scale_factor,
    float eps,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* x_row = x + row * dim;
    const float* x0_row = x0 + row * dim;
    float* mixed_row = mixed + row * dim;
    float* norm_row = norm + row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = mix[i] * x_row[i] + mix[dim + i] * x0_row[i];
        mixed_row[i] = v;
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float inv_rms_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
    }
    __syncthreads();
    float inv_rms = inv_rms_shared;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        norm_row[i] = mixed_row[i] * inv_rms;
    }
}

extern "C" __global__ void residual_mix_rms_norm_forward_bf16(
    const float* __restrict__ x,
    const float* __restrict__ x0,
    const float* __restrict__ mix,
    float* __restrict__ mixed,
    float* __restrict__ norm,
    unsigned short* __restrict__ norm_bf16,
    int dim,
    float ln_scale_factor,
    float eps,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* x_row = x + row * dim;
    const float* x0_row = x0 + row * dim;
    float* mixed_row = mixed + row * dim;
    float* norm_row = norm + row * dim;
    unsigned short* norm_bf16_row = norm_bf16 + row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = mix[i] * x_row[i] + mix[dim + i] * x0_row[i];
        mixed_row[i] = v;
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float inv_rms_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
    }
    __syncthreads();
    float inv_rms = inv_rms_shared;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float out = mixed_row[i] * inv_rms;
        norm_row[i] = out;
        norm_bf16_row[i] = pg_f32_to_bf16(out);
    }
}

// ──────────────────────────────────────────────────────────────
// Residual mixing backward.
// grad_x is written, grad_x0 and grad_mix are accumulated.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void residual_mix_backward(
    const float* __restrict__ x,
    const float* __restrict__ x0,
    const float* __restrict__ grad_output,
    const float* __restrict__ mix,
    float* __restrict__ grad_x,
    float* __restrict__ grad_x0,
    float* __restrict__ grad_mix,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    float go = grad_output[idx];
    grad_x[idx] = go * mix[d];
    grad_x0[idx] += go * mix[dim + d];
    atomicAdd(&grad_mix[d], go * x[idx]);
    atomicAdd(&grad_mix[dim + d], go * x0[idx]);
}

// ──────────────────────────────────────────────────────────────
// Residual add with per-dim scale: x[i] += scale[i%d] * proj[i]
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void residual_add_scale(
    float* __restrict__ x,
    const float* __restrict__ proj,
    const float* __restrict__ scale,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    x[idx] += scale[d] * proj[idx];
}

extern "C" __global__ void residual_add_scale_bf16_proj(
    float* __restrict__ x,
    const unsigned short* __restrict__ proj_bf16,
    const float* __restrict__ scale,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    x[idx] += scale[d] * pg_bf16_to_f32(proj_bf16[idx]);
}

extern "C" __global__ void residual_add_scale_from_base(
    const float* __restrict__ base,
    const float* __restrict__ proj,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    out[idx] = base[idx] + scale[d] * proj[idx];
}

extern "C" __global__ void residual_add_scale_from_base_bf16_proj(
    const float* __restrict__ base,
    const unsigned short* __restrict__ proj_bf16,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    out[idx] = base[idx] + scale[d] * pg_bf16_to_f32(proj_bf16[idx]);
}

// Parallel-residual forward fusion:
//   out[row]  = base[row] + scale * proj[row]
//   norm[row] = rms_norm(base[row])
// This matches the parallel-residual block where the MLP branch normalizes the
// pre-attention residual stream while the attention branch is accumulated into
// the block output.
extern "C" __global__ void residual_add_scale_from_base_rms_norm_forward(
    const float* __restrict__ base,
    const float* __restrict__ proj,
    const float* __restrict__ scale,
    float* __restrict__ out,
    float* __restrict__ norm,
    int dim,
    float ln_scale_factor,
    float eps,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* base_row = base + row * dim;
    const float* proj_row = proj + row * dim;
    float* out_row = out + row * dim;
    float* norm_row = norm + row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = base_row[i];
        out_row[i] = v + scale[i] * proj_row[i];
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float inv_rms_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
    }
    __syncthreads();
    float inv_rms = inv_rms_shared;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        norm_row[i] = base_row[i] * inv_rms;
    }
}

extern "C" __global__ void residual_add_scale_from_base_rms_norm_forward_bf16(
    const float* __restrict__ base,
    const float* __restrict__ proj,
    const float* __restrict__ scale,
    float* __restrict__ out,
    float* __restrict__ norm,
    unsigned short* __restrict__ norm_bf16,
    int dim,
    float ln_scale_factor,
    float eps,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* base_row = base + row * dim;
    const float* proj_row = proj + row * dim;
    float* out_row = out + row * dim;
    float* norm_row = norm + row * dim;
    unsigned short* norm_bf16_row = norm_bf16 + row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = base_row[i];
        out_row[i] = v + scale[i] * proj_row[i];
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float inv_rms_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
    }
    __syncthreads();
    float inv_rms = inv_rms_shared;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float out_norm = base_row[i] * inv_rms;
        norm_row[i] = out_norm;
        norm_bf16_row[i] = pg_f32_to_bf16(out_norm);
    }
}

extern "C" __global__ void residual_add_scale_from_base_rms_norm_forward_bf16_proj(
    const float* __restrict__ base,
    const unsigned short* __restrict__ proj_bf16,
    const float* __restrict__ scale,
    float* __restrict__ out,
    float* __restrict__ norm,
    unsigned short* __restrict__ norm_bf16,
    int dim,
    float ln_scale_factor,
    float eps,
    int n
) {
    int row = blockIdx.x;
    int num_rows = n / dim;
    if (row >= num_rows) return;

    const float* base_row = base + row * dim;
    const unsigned short* proj_row = proj_bf16 + row * dim;
    float* out_row = out + row * dim;
    float* norm_row = norm + row * dim;
    unsigned short* norm_bf16_row = norm_bf16 + row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = base_row[i];
        out_row[i] = v + scale[i] * pg_bf16_to_f32(proj_row[i]);
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float inv_rms_shared;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)dim + eps);
        inv_rms_shared = ln_scale_factor / rms;
    }
    __syncthreads();
    float inv_rms = inv_rms_shared;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float out_norm = base_row[i] * inv_rms;
        norm_row[i] = out_norm;
        norm_bf16_row[i] = pg_f32_to_bf16(out_norm);
    }
}

// ──────────────────────────────────────────────────────────────
// Residual add-scale backward.
// forward: x_out = x_in + scale * proj
// grad_x_in and grad_scale are accumulated; grad_proj is written.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void residual_add_scale_backward(
    const float* __restrict__ proj,
    const float* __restrict__ grad_output,
    const float* __restrict__ scale,
    float* __restrict__ grad_x_in,
    float* __restrict__ grad_proj,
    float* __restrict__ grad_scale,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    float go = grad_output[idx];
    grad_x_in[idx] += go;
    grad_proj[idx] = go * scale[d];
    atomicAdd(&grad_scale[d], go * proj[idx]);
}

extern "C" __global__ void residual_add_scale_backward_bf16(
    const float* __restrict__ proj,
    const float* __restrict__ grad_output,
    const float* __restrict__ scale,
    float* __restrict__ grad_x_in,
    float* __restrict__ grad_proj,
    unsigned short* __restrict__ grad_proj_bf16,
    float* __restrict__ grad_scale,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    float go = grad_output[idx];
    float gp = go * scale[d];
    grad_x_in[idx] += go;
    grad_proj[idx] = gp;
    grad_proj_bf16[idx] = pg_f32_to_bf16(gp);
    atomicAdd(&grad_scale[d], go * proj[idx]);
}

extern "C" __global__ void residual_add_scale_backward_bf16_only(
    const float* __restrict__ proj,
    const float* __restrict__ grad_output,
    const float* __restrict__ scale,
    float* __restrict__ grad_x_in,
    unsigned short* __restrict__ grad_proj_bf16,
    float* __restrict__ grad_scale,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    float go = grad_output[idx];
    grad_x_in[idx] += go;
    grad_proj_bf16[idx] = pg_f32_to_bf16(go * scale[d]);
    atomicAdd(&grad_scale[d], go * proj[idx]);
}

extern "C" __global__ void residual_add_scale_backward_from_bf16_only(
    const unsigned short* __restrict__ proj_bf16,
    const float* __restrict__ grad_output,
    const float* __restrict__ scale,
    float* __restrict__ grad_x_in,
    unsigned short* __restrict__ grad_proj_bf16,
    float* __restrict__ grad_scale,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    float go = grad_output[idx];
    grad_x_in[idx] += go;
    grad_proj_bf16[idx] = pg_f32_to_bf16(go * scale[d]);
    atomicAdd(&grad_scale[d], go * pg_bf16_to_f32(proj_bf16[idx]));
}

extern "C" __global__ void residual_add_scale_backward_bf16_only_no_atomic(
    const float* __restrict__ grad_output,
    const float* __restrict__ scale,
    float* __restrict__ grad_x_in,
    unsigned short* __restrict__ grad_proj_bf16,
    int dim,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = idx % dim;
    float go = grad_output[idx];
    grad_x_in[idx] += go;
    grad_proj_bf16[idx] = pg_f32_to_bf16(go * scale[d]);
}

extern "C" __global__ void residual_add_scale_grad_scale_reduce(
    const float* __restrict__ proj,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_scale,
    int dim,
    int n
) {
    int d = blockIdx.x;
    if (d >= dim) return;
    int rows = n / dim;
    float sum = 0.0f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        int idx = r * dim + d;
        sum += grad_output[idx] * proj[idx];
    }

    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&grad_scale[d], scratch[0]);
    }
}

extern "C" __global__ void residual_add_scale_grad_scale_reduce_bf16(
    const unsigned short* __restrict__ proj_bf16,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_scale,
    int dim,
    int n
) {
    int d = blockIdx.x;
    if (d >= dim) return;
    int rows = n / dim;
    float sum = 0.0f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        int idx = r * dim + d;
        sum += grad_output[idx] * pg_bf16_to_f32(proj_bf16[idx]);
    }

    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&grad_scale[d], scratch[0]);
    }
}

// ──────────────────────────────────────────────────────────────
// SmearGate: out = (1 - sigmoid(gate)) * x + sigmoid(gate) * x_prev
// NOTE: high sigmoid → more x_prev (matching CPU reference)
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void smear_gate_forward(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    float* __restrict__ out,
    int tokens,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = tokens * dim;
    if (idx >= n) return;

    int tok = idx / dim;
    int d = idx % dim;
    float sig = 1.0f / (1.0f + expf(-gate[d]));
    float x_val = x[idx];
    float x_prev = (tok % seq_len > 0) ? x[idx - dim] : 0.0f;
    out[idx] = (1.0f - sig) * x_val + sig * x_prev;
}

extern "C" __global__ void smear_gate_forward_boundary(
    const float* __restrict__ x,
    const unsigned int* __restrict__ input_ids,
    const float* __restrict__ gate,
    float* __restrict__ out,
    int tokens,
    int seq_len,
    int dim,
    unsigned int boundary_token_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = tokens * dim;
    if (idx >= n) return;

    int tok = idx / dim;
    int d = idx % dim;
    float sig = 1.0f / (1.0f + expf(-gate[d]));
    float x_val = x[idx];
    bool has_prev = (tok % seq_len > 0) && (input_ids[tok] != boundary_token_id);
    float x_prev = has_prev ? x[idx - dim] : 0.0f;
    out[idx] = (1.0f - sig) * x_val + sig * x_prev;
}

// ──────────────────────────────────────────────────────────────
// SmearGate backward.
// x_prev is the previous token's x (token 0 uses zero), matching forward.
// grad_gate is accumulated.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void smear_gate_backward(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_x,
    float* __restrict__ grad_x_prev,
    float* __restrict__ grad_gate,
    int tokens,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = tokens * dim;
    if (idx >= n) return;

    int tok = idx / dim;
    int d = idx % dim;
    float sig = 1.0f / (1.0f + expf(-gate[d]));
    float go = grad_output[idx];
    float x_val = x[idx];
    float x_prev = (tok % seq_len > 0) ? x[idx - dim] : 0.0f;

    grad_x[idx] = (1.0f - sig) * go;
    grad_x_prev[idx] = (tok % seq_len > 0) ? sig * go : 0.0f;
    atomicAdd(&grad_gate[d], go * (x_prev - x_val) * sig * (1.0f - sig));
}

extern "C" __global__ void smear_gate_backward_boundary(
    const float* __restrict__ x,
    const unsigned int* __restrict__ input_ids,
    const float* __restrict__ gate,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_x,
    float* __restrict__ grad_x_prev,
    float* __restrict__ grad_gate,
    int tokens,
    int seq_len,
    int dim,
    unsigned int boundary_token_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = tokens * dim;
    if (idx >= n) return;

    int tok = idx / dim;
    int d = idx % dim;
    float sig = 1.0f / (1.0f + expf(-gate[d]));
    float go = grad_output[idx];
    float x_val = x[idx];
    bool has_prev = (tok % seq_len > 0) && (input_ids[tok] != boundary_token_id);
    float x_prev = has_prev ? x[idx - dim] : 0.0f;

    grad_x[idx] = (1.0f - sig) * go;
    grad_x_prev[idx] = has_prev ? sig * go : 0.0f;
    atomicAdd(&grad_gate[d], go * (x_prev - x_val) * sig * (1.0f - sig));
}

// ──────────────────────────────────────────────────────────────
// Embedding gather: out[t, :] = emb[ids[t], :]
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void embedding_gather(
    const int* __restrict__ ids,
    const float* __restrict__ emb,
    float* __restrict__ out,
    int dim,
    int tokens
) {
    int t = blockIdx.x;
    if (t >= tokens) return;
    int tok = ids[t];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[t * dim + i] = emb[tok * dim + i];
    }
}

// ──────────────────────────────────────────────────────────────
// Embedding gather backward: grad_emb[ids[t], :] += grad_output[t, :]
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void embedding_gather_backward(
    const int* __restrict__ ids,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_emb,
    int dim,
    int tokens
) {
    int t = blockIdx.x;
    if (t >= tokens) return;
    int tok = ids[t];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        atomicAdd(&grad_emb[tok * dim + i], grad_output[t * dim + i]);
    }
}

// ──────────────────────────────────────────────────────────────
// QK-norm: per-head RMSNorm on Q or K (in-place)
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void qk_norm_forward(
    float* __restrict__ qk,
    int head_dim,
    int total_heads,
    float eps
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    float* head = qk + head_idx * head_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = head[i];
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    // head_dim <= 64, so one warp is enough. Broadcast via shared memory.
    __shared__ float inv_rms_sh;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)head_dim + eps);
        inv_rms_sh = 1.0f / rms;
    }
    __syncthreads();
    float inv_rms = inv_rms_sh;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        head[i] *= inv_rms;
    }
}

// ──────────────────────────────────────────────────────────────
// QK-norm backward: RMSNorm backward with scale factor 1.0.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void qk_norm_backward(
    const float* __restrict__ x,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int head_dim,
    int total_heads,
    float eps
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    const float* x_head = x + head_idx * head_dim;
    const float* go_head = grad_output + head_idx * head_dim;
    float* gi_head = grad_input + head_idx * head_dim;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float xv = x_head[i];
        float go = go_head[i];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float inv_rms_sh;
    __shared__ float coeff_sh;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)head_dim + eps);
        inv_rms_sh = 1.0f / rms;
        coeff_sh = x_dot_go / (rms * rms * (float)head_dim);
    }
    __syncthreads();

    float inv_rms = inv_rms_sh;
    float coeff = coeff_sh;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        gi_head[i] = inv_rms * (go_head[i] - x_head[i] * coeff);
    }
}

// ──────────────────────────────────────────────────────────────
// Partial RoPE forward (half-split convention)
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void partial_rope_forward(
    float* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_dims,
    int total_heads
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    int tok = head_idx / num_heads;
    int half = rope_dims / 2;
    float* head = x + head_idx * head_dim;
    int pos = tok % seq_len;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float cos_val = cos_table[pos * half + i];
        float sin_val = sin_table[pos * half + i];
        float x1 = head[i];
        float x2 = head[half + i];
        head[i]        = x1 * cos_val + x2 * sin_val;
        head[half + i] = -x1 * sin_val + x2 * cos_val;
    }
}

// ──────────────────────────────────────────────────────────────
// Partial RoPE backward: apply the transpose rotation in-place.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void partial_rope_backward(
    float* __restrict__ grad,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_dims,
    int total_heads
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    int tok = head_idx / num_heads;
    int half = rope_dims / 2;
    float* head = grad + head_idx * head_dim;
    int pos = tok % seq_len;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float cos_val = cos_table[pos * half + i];
        float sin_val = sin_table[pos * half + i];
        float g1 = head[i];
        float g2 = head[half + i];
        head[i]        = g1 * cos_val - g2 * sin_val;
        head[half + i] = g1 * sin_val + g2 * cos_val;
    }
}

// ──────────────────────────────────────────────────────────────
// Q-gain: Q *= gain[head] per head (in-place)
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void q_gain_forward(
    float* __restrict__ q,
    const float* __restrict__ gain,
    int num_heads,
    int head_dim,
    int total_heads
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;
    int h = head_idx % num_heads;
    float g = gain[h];
    float* head = q + head_idx * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        head[i] *= g;
    }
}

// ──────────────────────────────────────────────────────────────
// Q-gain backward.
// grad_gain is accumulated by head.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void q_gain_backward(
    const float* __restrict__ q_pre_gain,
    const float* __restrict__ grad_q_scaled,
    const float* __restrict__ gain,
    float* __restrict__ grad_q_pre_gain,
    float* __restrict__ grad_gain,
    int num_heads,
    int head_dim,
    int total_heads
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;
    int h = head_idx % num_heads;
    float g = gain[h];

    const float* q_head = q_pre_gain + head_idx * head_dim;
    const float* go_head = grad_q_scaled + head_idx * head_dim;
    float* gi_head = grad_q_pre_gain + head_idx * head_dim;

    float dot = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float go = go_head[i];
        gi_head[i] = go * g;
        dot += go * q_head[i];
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    __shared__ float shared_dot[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_dot[warp_id] = dot;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        dot = (threadIdx.x < warps) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&grad_gain[h], dot);
    }
}

extern "C" __global__ void q_gain_rope_qk_norm_forward(
    float* __restrict__ q,
    float* __restrict__ q_post_rope,
    const float* __restrict__ gain,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_dims,
    int total_heads,
    float eps
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    int h = head_idx % num_heads;
    int tok = head_idx / num_heads;
    int pos = tok % seq_len;
    int half = rope_dims / 2;
    float* head = q + head_idx * head_dim;
    float* post_head = q_post_rope ? (q_post_rope + head_idx * head_dim) : nullptr;
    float g = gain[h];

    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float v = head[j];
        sum_sq += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float inv_rms_sh;
    if (threadIdx.x == 0) {
        inv_rms_sh = rsqrtf(sum_sq / (float)head_dim + eps);
    }
    __syncthreads();
    float inv_rms = inv_rms_sh;

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        if (j < half) {
            float cos_val = cos_table[pos * half + j];
            float sin_val = sin_table[pos * half + j];
            float x1 = head[j] * inv_rms;
            float x2 = head[half + j] * inv_rms;
            float r1 = x1 * cos_val + x2 * sin_val;
            float r2 = -x1 * sin_val + x2 * cos_val;
            if (post_head) {
                post_head[j] = r1;
                post_head[half + j] = r2;
            }
            head[j] = r1 * g;
            head[half + j] = r2 * g;
        } else if (j >= rope_dims) {
            float r = head[j] * inv_rms;
            if (post_head) post_head[j] = r;
            head[j] = r * g;
        }
    }
}

extern "C" __global__ void q_gain_rope_qk_norm_forward_bf16_bhsd(
    float* __restrict__ q,
    float* __restrict__ q_post_rope,
    unsigned short* __restrict__ q_bhsd_bf16,
    const float* __restrict__ gain,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_dims,
    int total_heads,
    float eps
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    int h = head_idx % num_heads;
    int tok = head_idx / num_heads;
    int b = tok / seq_len;
    int pos = tok - b * seq_len;
    int half = rope_dims / 2;
    float* head = q + head_idx * head_dim;
    float* post_head = q_post_rope ? (q_post_rope + head_idx * head_dim) : nullptr;
    unsigned short* out_head = q_bhsd_bf16
        ? q_bhsd_bf16 + ((b * num_heads + h) * seq_len + pos) * head_dim
        : nullptr;
    float g = gain[h];

    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float v = head[j];
        sum_sq += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float inv_rms_sh;
    if (threadIdx.x == 0) {
        inv_rms_sh = rsqrtf(sum_sq / (float)head_dim + eps);
    }
    __syncthreads();
    float inv_rms = inv_rms_sh;

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float out_value;
        if (j < half) {
            float cos_val = cos_table[pos * half + j];
            float sin_val = sin_table[pos * half + j];
            float x1 = head[j] * inv_rms;
            float x2 = head[half + j] * inv_rms;
            float r1 = x1 * cos_val + x2 * sin_val;
            float r2 = -x1 * sin_val + x2 * cos_val;
            if (post_head) {
                post_head[j] = r1;
                post_head[half + j] = r2;
            }
            out_value = r1 * g;
            head[j] = out_value;
            if (out_head) out_head[j] = pg_f32_to_bf16(out_value);
            float out_peer = r2 * g;
            head[half + j] = out_peer;
            if (out_head) out_head[half + j] = pg_f32_to_bf16(out_peer);
        } else if (j >= rope_dims) {
            float r = head[j] * inv_rms;
            if (post_head) post_head[j] = r;
            out_value = r * g;
            head[j] = out_value;
            if (out_head) out_head[j] = pg_f32_to_bf16(out_value);
        }
    }
}

extern "C" __global__ void rope_qk_norm_forward(
    float* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_dims,
    int total_heads,
    float eps
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    int tok = head_idx / num_heads;
    int pos = tok % seq_len;
    int half = rope_dims / 2;
    float* head = k + head_idx * head_dim;

    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float v = head[j];
        sum_sq += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float inv_rms_sh;
    if (threadIdx.x == 0) {
        inv_rms_sh = rsqrtf(sum_sq / (float)head_dim + eps);
    }
    __syncthreads();
    float inv_rms = inv_rms_sh;

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        if (j < half) {
            float cos_val = cos_table[pos * half + j];
            float sin_val = sin_table[pos * half + j];
            float x1 = head[j] * inv_rms;
            float x2 = head[half + j] * inv_rms;
            head[j] = x1 * cos_val + x2 * sin_val;
            head[half + j] = -x1 * sin_val + x2 * cos_val;
        } else if (j >= rope_dims) {
            head[j] *= inv_rms;
        }
    }
}

extern "C" __global__ void rope_qk_norm_forward_bf16_bhsd(
    float* __restrict__ k,
    unsigned short* __restrict__ k_bhsd_bf16,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_dims,
    int total_heads,
    float eps
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    int h = head_idx % num_heads;
    int tok = head_idx / num_heads;
    int b = tok / seq_len;
    int pos = tok - b * seq_len;
    int half = rope_dims / 2;
    float* head = k + head_idx * head_dim;
    unsigned short* out_head = k_bhsd_bf16
        ? k_bhsd_bf16 + ((b * num_heads + h) * seq_len + pos) * head_dim
        : nullptr;

    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float v = head[j];
        sum_sq += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float inv_rms_sh;
    if (threadIdx.x == 0) {
        inv_rms_sh = rsqrtf(sum_sq / (float)head_dim + eps);
    }
    __syncthreads();
    float inv_rms = inv_rms_sh;

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float out_value;
        if (j < half) {
            float cos_val = cos_table[pos * half + j];
            float sin_val = sin_table[pos * half + j];
            float x1 = head[j] * inv_rms;
            float x2 = head[half + j] * inv_rms;
            out_value = x1 * cos_val + x2 * sin_val;
            head[j] = out_value;
            if (out_head) out_head[j] = pg_f32_to_bf16(out_value);
            float out_peer = -x1 * sin_val + x2 * cos_val;
            head[half + j] = out_peer;
            if (out_head) out_head[half + j] = pg_f32_to_bf16(out_peer);
        } else if (j >= rope_dims) {
            out_value = head[j] * inv_rms;
            head[j] = out_value;
            if (out_head) out_head[j] = pg_f32_to_bf16(out_value);
        }
    }
}

extern "C" __global__ void bthd_to_bhsd_bf16(
    const float* __restrict__ in,
    unsigned short* __restrict__ out,
    int batch,
    int seq_len,
    int num_heads,
    int head_dim,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int h = tmp % num_heads;
    int token = tmp / num_heads;
    int b = token / seq_len;
    int s = token - b * seq_len;
    int out_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
    out[out_idx] = pg_f32_to_bf16(in[idx]);
}

extern "C" __global__ void bhsd_to_bthd_bf16(
    const unsigned short* __restrict__ in,
    unsigned short* __restrict__ out,
    int batch,
    int seq_len,
    int num_heads,
    int head_dim,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int h = tmp % num_heads;
    int token = tmp / num_heads;
    int b = token / seq_len;
    int s = token - b * seq_len;
    int in_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
    out[idx] = in[in_idx];
}

extern "C" __global__ void q_gain_rope_qk_norm_backward(
    const float* __restrict__ q_pre_norm,
    const float* __restrict__ q_post_rope,
    const float* __restrict__ grad_q_scaled,
    const float* __restrict__ gain,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ grad_q_proj,
    float* __restrict__ grad_gain,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_dims,
    int total_heads,
    float eps
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    int h = head_idx % num_heads;
    int tok = head_idx / num_heads;
    int pos = tok % seq_len;
    int half = rope_dims / 2;

    const float* x_head = q_pre_norm + head_idx * head_dim;
    const float* q_rope_head = q_post_rope + head_idx * head_dim;
    const float* go_scaled_head = grad_q_scaled + head_idx * head_dim;
    float* gi_head = grad_q_proj + head_idx * head_dim;
    float g = gain[h];

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    float gain_dot = 0.0f;

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float go_j_scaled = go_scaled_head[j];
        float go_j = go_j_scaled * g;
        gain_dot += go_j_scaled * q_rope_head[j];

        float go = go_j;
        if (j < rope_dims && rope_dims > 0) {
            if (j < half) {
                float cos_val = cos_table[pos * half + j];
                float sin_val = sin_table[pos * half + j];
                float peer = go_scaled_head[half + j] * g;
                go = go_j * cos_val - peer * sin_val;
            } else {
                int i = j - half;
                float cos_val = cos_table[pos * half + i];
                float sin_val = sin_table[pos * half + i];
                float peer = go_scaled_head[i] * g;
                go = peer * sin_val + go_j * cos_val;
            }
        }

        float xv = x_head[j];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        gain_dot += __shfl_down_sync(0xffffffff, gain_dot, offset);
    }

    __shared__ float inv_rms_sh;
    __shared__ float coeff_sh;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)head_dim + eps);
        inv_rms_sh = 1.0f / rms;
        coeff_sh = x_dot_go / (rms * rms * (float)head_dim);
        atomicAdd(&grad_gain[h], gain_dot);
    }
    __syncthreads();

    float inv_rms = inv_rms_sh;
    float coeff = coeff_sh;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float go_j = go_scaled_head[j] * g;
        float go = go_j;
        if (j < rope_dims && rope_dims > 0) {
            if (j < half) {
                float cos_val = cos_table[pos * half + j];
                float sin_val = sin_table[pos * half + j];
                float peer = go_scaled_head[half + j] * g;
                go = go_j * cos_val - peer * sin_val;
            } else {
                int i = j - half;
                float cos_val = cos_table[pos * half + i];
                float sin_val = sin_table[pos * half + i];
                float peer = go_scaled_head[i] * g;
                go = peer * sin_val + go_j * cos_val;
            }
        }
        float xv = x_head[j];
        gi_head[j] = inv_rms * (go - xv * coeff);
    }
}

extern "C" __global__ void q_gain_rope_qk_norm_backward_chunked(
    const float* __restrict__ q_pre_norm,
    const float* __restrict__ q_post_rope,
    const float* __restrict__ grad_q_scaled,
    const float* __restrict__ gain,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ grad_q_proj,
    float* __restrict__ grad_gain_chunks,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_dims,
    int total_heads,
    int q_gain_chunk_tokens,
    int q_gain_chunks,
    float eps
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    int h = head_idx % num_heads;
    int tok = head_idx / num_heads;
    int pos = tok % seq_len;
    int half = rope_dims / 2;

    const float* x_head = q_pre_norm + head_idx * head_dim;
    const float* q_rope_head = q_post_rope + head_idx * head_dim;
    const float* go_scaled_head = grad_q_scaled + head_idx * head_dim;
    float* gi_head = grad_q_proj + head_idx * head_dim;
    float g = gain[h];

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    float gain_dot = 0.0f;

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float go_j_scaled = go_scaled_head[j];
        float go_j = go_j_scaled * g;
        gain_dot += go_j_scaled * q_rope_head[j];

        float go = go_j;
        if (j < rope_dims && rope_dims > 0) {
            if (j < half) {
                float cos_val = cos_table[pos * half + j];
                float sin_val = sin_table[pos * half + j];
                float peer = go_scaled_head[half + j] * g;
                go = go_j * cos_val - peer * sin_val;
            } else {
                int i = j - half;
                float cos_val = cos_table[pos * half + i];
                float sin_val = sin_table[pos * half + i];
                float peer = go_scaled_head[i] * g;
                go = peer * sin_val + go_j * cos_val;
            }
        }

        float xv = x_head[j];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
        gain_dot += __shfl_down_sync(0xffffffff, gain_dot, offset);
    }

    __shared__ float inv_rms_sh;
    __shared__ float coeff_sh;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)head_dim + eps);
        inv_rms_sh = 1.0f / rms;
        coeff_sh = x_dot_go / (rms * rms * (float)head_dim);
        int chunk = (tok / q_gain_chunk_tokens);
        if (chunk >= q_gain_chunks) chunk = q_gain_chunks - 1;
        atomicAdd(&grad_gain_chunks[h * q_gain_chunks + chunk], gain_dot);
    }
    __syncthreads();

    float inv_rms = inv_rms_sh;
    float coeff = coeff_sh;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float go_j = go_scaled_head[j] * g;
        float go = go_j;
        if (j < rope_dims && rope_dims > 0) {
            if (j < half) {
                float cos_val = cos_table[pos * half + j];
                float sin_val = sin_table[pos * half + j];
                float peer = go_scaled_head[half + j] * g;
                go = go_j * cos_val - peer * sin_val;
            } else {
                int i = j - half;
                float cos_val = cos_table[pos * half + i];
                float sin_val = sin_table[pos * half + i];
                float peer = go_scaled_head[i] * g;
                go = peer * sin_val + go_j * cos_val;
            }
        }
        float xv = x_head[j];
        gi_head[j] = inv_rms * (go - xv * coeff);
    }
}

extern "C" __global__ void q_gain_reduce_chunked(
    const float* __restrict__ grad_gain_chunks,
    float* __restrict__ grad_gain,
    int q_gain_chunks
) {
    int h = blockIdx.x;
    __shared__ float scratch[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < q_gain_chunks; i += blockDim.x) {
        sum += grad_gain_chunks[h * q_gain_chunks + i];
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(&grad_gain[h], scratch[0]);
}

extern "C" __global__ void rope_qk_norm_backward(
    const float* __restrict__ k_pre_norm,
    const float* __restrict__ grad_k_attn,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ grad_k_proj,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_dims,
    int total_heads,
    float eps
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;

    int tok = head_idx / num_heads;
    int pos = tok % seq_len;
    int half = rope_dims / 2;

    const float* x_head = k_pre_norm + head_idx * head_dim;
    const float* go_head = grad_k_attn + head_idx * head_dim;
    float* gi_head = grad_k_proj + head_idx * head_dim;

    float sum_sq = 0.0f;
    float x_dot_go = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float go = go_head[j];
        if (j < rope_dims && rope_dims > 0) {
            if (j < half) {
                float cos_val = cos_table[pos * half + j];
                float sin_val = sin_table[pos * half + j];
                float peer = go_head[half + j];
                go = go * cos_val - peer * sin_val;
            } else {
                int i = j - half;
                float cos_val = cos_table[pos * half + i];
                float sin_val = sin_table[pos * half + i];
                float peer = go_head[i];
                go = peer * sin_val + go * cos_val;
            }
        }

        float xv = x_head[j];
        sum_sq += xv * xv;
        x_dot_go += xv * go;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        x_dot_go += __shfl_down_sync(0xffffffff, x_dot_go, offset);
    }

    __shared__ float inv_rms_sh;
    __shared__ float coeff_sh;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / (float)head_dim + eps);
        inv_rms_sh = 1.0f / rms;
        coeff_sh = x_dot_go / (rms * rms * (float)head_dim);
    }
    __syncthreads();

    float inv_rms = inv_rms_sh;
    float coeff = coeff_sh;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float go = go_head[j];
        if (j < rope_dims && rope_dims > 0) {
            if (j < half) {
                float cos_val = cos_table[pos * half + j];
                float sin_val = sin_table[pos * half + j];
                float peer = go_head[half + j];
                go = go * cos_val - peer * sin_val;
            } else {
                int i = j - half;
                float cos_val = cos_table[pos * half + i];
                float sin_val = sin_table[pos * half + i];
                float peer = go_head[i];
                go = peer * sin_val + go * cos_val;
            }
        }
        float xv = x_head[j];
        gi_head[j] = inv_rms * (go - xv * coeff);
    }
}

// ──────────────────────────────────────────────────────────────
// Dot accumulate: out[0] += alpha * sum_i(a[i] * b[i])
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void dot_accumulate(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, alpha * sum);
    }
}

// ──────────────────────────────────────────────────────────────
// Dot accumulate with a device-resident scalar:
// out[0] += alpha * scale[index] * dot(a, b)
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void dot_accumulate_by_param(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    const float* __restrict__ scale,
    int scale_index,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, alpha * scale[scale_index] * sum);
    }
}

// ──────────────────────────────────────────────────────────────
// Device-side global-norm clipping:
// x *= min(1, max_norm / sqrt(sum_sq[0]))
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void clip_by_global_norm(
    float* __restrict__ x,
    const float* __restrict__ sum_sq,
    float max_norm,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float norm = sqrtf(sum_sq[0]);
    float scale = (norm > max_norm) ? (max_norm / (norm + 1e-12f)) : 1.0f;
    x[idx] *= scale;
}

// ──────────────────────────────────────────────────────────────
// Fused cross-entropy forward (with softcap)
// One block per token, full reduction within CTA.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void cross_entropy_forward(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ losses,
    int vocab_size,
    float softcap,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    const float* row = logits + t * vocab_size;
    int target = targets[t];
    float inv_cap = 1.0f / softcap;

    // Phase 1: find max capped logit
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        if (capped > max_val) max_val = capped;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x < 32) {
        max_val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_max[threadIdx.x] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float max_broadcast;
    if (threadIdx.x == 0) max_broadcast = max_val;
    __syncthreads();
    max_val = max_broadcast;

    // Phase 2: sum of exp
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        sum_exp += expf(capped - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum_exp = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }

    if (threadIdx.x == 0) {
        float log_sum_exp = max_val + logf(sum_exp);
        float capped_target = softcap * tanhf(row[target] * inv_cap);
        losses[t] = log_sum_exp - capped_target;
    }
}

extern "C" __global__ void cross_entropy_forward_bf16_logits(
    const unsigned short* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ losses,
    int vocab_size,
    float softcap,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    const unsigned short* row = logits + t * vocab_size;
    int target = targets[t];
    float inv_cap = 1.0f / softcap;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(pg_bf16_to_f32(row[i]) * inv_cap);
        if (capped > max_val) max_val = capped;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x < 32) {
        max_val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_max[threadIdx.x] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float max_broadcast;
    if (threadIdx.x == 0) max_broadcast = max_val;
    __syncthreads();
    max_val = max_broadcast;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(pg_bf16_to_f32(row[i]) * inv_cap);
        sum_exp += expf(capped - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum_exp = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }

    if (threadIdx.x == 0) {
        float log_sum_exp = max_val + logf(sum_exp);
        float capped_target = softcap * tanhf(pg_bf16_to_f32(row[target]) * inv_cap);
        losses[t] = log_sum_exp - capped_target;
    }
}

// ──────────────────────────────────────────────────────────────
// Cross-entropy backward with softcap.
// One block per token; writes the full grad_logits row.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void cross_entropy_backward(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ grad_logits,
    int vocab_size,
    float softcap,
    float grad_loss,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    const float* row = logits + t * vocab_size;
    float* grad_row = grad_logits + t * vocab_size;
    int target = targets[t];
    float inv_cap = 1.0f / softcap;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        if (capped > max_val) max_val = capped;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        max_val = (threadIdx.x < warps) ? shared_max[threadIdx.x] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float max_broadcast;
    if (threadIdx.x == 0) max_broadcast = max_val;
    __syncthreads();
    max_val = max_broadcast;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        sum_exp += expf(capped - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_exp = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float sum_broadcast;
    if (threadIdx.x == 0) sum_broadcast = sum_exp;
    __syncthreads();
    sum_exp = sum_broadcast;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        float prob = expf(capped - max_val) / sum_exp;
        float one_hot = (i == target) ? 1.0f : 0.0f;
        float tv = tanhf(row[i] * inv_cap);
        float d_softcap = 1.0f - tv * tv;
        grad_row[i] = grad_loss * (prob - one_hot) * d_softcap;
    }
}

// ──────────────────────────────────────────────────────────────
// Cross-entropy backward with softcap, writing BF16 gradients directly.
// This is equivalent to cross_entropy_backward followed by f32_to_bf16 and
// avoids the full F32 grad_logits materialization in the BF16 output path.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void cross_entropy_backward_bf16(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    unsigned short* __restrict__ grad_logits,
    int vocab_size,
    float softcap,
    float grad_loss,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    const float* row = logits + t * vocab_size;
    unsigned short* grad_row = grad_logits + t * vocab_size;
    int target = targets[t];
    float inv_cap = 1.0f / softcap;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        if (capped > max_val) max_val = capped;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        max_val = (threadIdx.x < warps) ? shared_max[threadIdx.x] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float max_broadcast;
    if (threadIdx.x == 0) max_broadcast = max_val;
    __syncthreads();
    max_val = max_broadcast;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        sum_exp += expf(capped - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_exp = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float sum_broadcast;
    if (threadIdx.x == 0) sum_broadcast = sum_exp;
    __syncthreads();
    sum_exp = sum_broadcast;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        float prob = expf(capped - max_val) / sum_exp;
        float one_hot = (i == target) ? 1.0f : 0.0f;
        float tv = tanhf(row[i] * inv_cap);
        float d_softcap = 1.0f - tv * tv;
        float grad = grad_loss * (prob - one_hot) * d_softcap;
        unsigned int bits = __float_as_uint(grad);
        unsigned int lsb = (bits >> 16) & 1u;
        unsigned int rounding_bias = 0x7fffu + lsb;
        grad_row[i] = static_cast<unsigned short>((bits + rounding_bias) >> 16);
    }
}

extern "C" __global__ void cross_entropy_backward_bf16_logits(
    const unsigned short* __restrict__ logits,
    const int* __restrict__ targets,
    unsigned short* __restrict__ grad_logits,
    int vocab_size,
    float softcap,
    float grad_loss,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    const unsigned short* row = logits + t * vocab_size;
    unsigned short* grad_row = grad_logits + t * vocab_size;
    int target = targets[t];
    float inv_cap = 1.0f / softcap;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(pg_bf16_to_f32(row[i]) * inv_cap);
        if (capped > max_val) max_val = capped;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        max_val = (threadIdx.x < warps) ? shared_max[threadIdx.x] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float max_broadcast;
    if (threadIdx.x == 0) max_broadcast = max_val;
    __syncthreads();
    max_val = max_broadcast;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(pg_bf16_to_f32(row[i]) * inv_cap);
        sum_exp += expf(capped - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_exp = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float sum_broadcast;
    if (threadIdx.x == 0) sum_broadcast = sum_exp;
    __syncthreads();
    sum_exp = sum_broadcast;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float logit = pg_bf16_to_f32(row[i]);
        float capped = softcap * tanhf(logit * inv_cap);
        float prob = expf(capped - max_val) / sum_exp;
        float one_hot = (i == target) ? 1.0f : 0.0f;
        float tv = tanhf(logit * inv_cap);
        float d_softcap = 1.0f - tv * tv;
        float grad = grad_loss * (prob - one_hot) * d_softcap;
        grad_row[i] = pg_f32_to_bf16(grad);
    }
}

// ──────────────────────────────────────────────────────────────
// Fused softcapped cross-entropy loss + BF16 backward.
// This matches cross_entropy_forward followed by cross_entropy_backward_bf16
// while scanning each logits row once for max/sum and once for gradient.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void cross_entropy_loss_backward_bf16(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ losses,
    unsigned short* __restrict__ grad_logits,
    int vocab_size,
    float softcap,
    float grad_loss,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    const float* row = logits + t * vocab_size;
    unsigned short* grad_row = grad_logits + t * vocab_size;
    int target = targets[t];
    float inv_cap = 1.0f / softcap;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        if (capped > max_val) max_val = capped;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        max_val = (threadIdx.x < warps) ? shared_max[threadIdx.x] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float max_broadcast;
    if (threadIdx.x == 0) max_broadcast = max_val;
    __syncthreads();
    max_val = max_broadcast;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        sum_exp += expf(capped - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_exp = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float sum_broadcast;
    if (threadIdx.x == 0) sum_broadcast = sum_exp;
    __syncthreads();
    sum_exp = sum_broadcast;

    if (threadIdx.x == 0) {
        float log_sum_exp = max_val + logf(sum_exp);
        float capped_target = softcap * tanhf(row[target] * inv_cap);
        losses[t] = log_sum_exp - capped_target;
    }

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        float prob = expf(capped - max_val) / sum_exp;
        float one_hot = (i == target) ? 1.0f : 0.0f;
        float tv = tanhf(row[i] * inv_cap);
        float d_softcap = 1.0f - tv * tv;
        float grad = grad_loss * (prob - one_hot) * d_softcap;
        grad_row[i] = pg_f32_to_bf16(grad);
    }
}

extern "C" __global__ void cross_entropy_loss_backward_bf16_logits(
    const unsigned short* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ losses,
    unsigned short* __restrict__ grad_logits,
    int vocab_size,
    float softcap,
    float grad_loss,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    const unsigned short* row = logits + t * vocab_size;
    unsigned short* grad_row = grad_logits + t * vocab_size;
    int target = targets[t];
    float inv_cap = 1.0f / softcap;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(pg_bf16_to_f32(row[i]) * inv_cap);
        if (capped > max_val) max_val = capped;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        max_val = (threadIdx.x < warps) ? shared_max[threadIdx.x] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float max_broadcast;
    if (threadIdx.x == 0) max_broadcast = max_val;
    __syncthreads();
    max_val = max_broadcast;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float capped = softcap * tanhf(pg_bf16_to_f32(row[i]) * inv_cap);
        sum_exp += expf(capped - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        sum_exp = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float sum_broadcast;
    if (threadIdx.x == 0) sum_broadcast = sum_exp;
    __syncthreads();
    sum_exp = sum_broadcast;

    if (threadIdx.x == 0) {
        float log_sum_exp = max_val + logf(sum_exp);
        float capped_target = softcap * tanhf(pg_bf16_to_f32(row[target]) * inv_cap);
        losses[t] = log_sum_exp - capped_target;
    }

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float logit = pg_bf16_to_f32(row[i]);
        float capped = softcap * tanhf(logit * inv_cap);
        float prob = expf(capped - max_val) / sum_exp;
        float one_hot = (i == target) ? 1.0f : 0.0f;
        float tv = tanhf(logit * inv_cap);
        float d_softcap = 1.0f - tv * tv;
        float grad = grad_loss * (prob - one_hot) * d_softcap;
        grad_row[i] = pg_f32_to_bf16(grad);
    }
}

extern "C" __global__ void output_ce_stats_init(
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    float* __restrict__ target_logit,
    int num_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) return;
    row_max[idx] = -1.0e30f;
    row_sum[idx] = 0.0f;
    target_logit[idx] = 0.0f;
}

extern "C" __global__ void output_ce_tile_stats_update(
    const float* __restrict__ logits_tile,
    const int* __restrict__ targets,
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    float* __restrict__ target_logit,
    int vocab_start,
    int tile_vocab,
    int tile_stride,
    float softcap,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    const float* row = logits_tile + t * tile_stride;
    int target = targets[t];
    float inv_cap = 1.0f / softcap;

    float tile_max = -1.0e30f;
    for (int i = threadIdx.x; i < tile_vocab; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        tile_max = fmaxf(tile_max, capped);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = tile_max;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        tile_max = (threadIdx.x < warps) ? shared_max[threadIdx.x] : -1.0e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
    }
    __shared__ float tile_max_broadcast;
    if (threadIdx.x == 0) tile_max_broadcast = tile_max;
    __syncthreads();
    tile_max = tile_max_broadcast;

    float tile_sum = 0.0f;
    for (int i = threadIdx.x; i < tile_vocab; i += blockDim.x) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        tile_sum += expf(capped - tile_max);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = tile_sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        tile_sum = (threadIdx.x < warps) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
    }
    __shared__ float tile_sum_broadcast;
    if (threadIdx.x == 0) tile_sum_broadcast = tile_sum;
    __syncthreads();
    tile_sum = tile_sum_broadcast;

    if (threadIdx.x == 0) {
        float prev_max = row_max[t];
        float prev_sum = row_sum[t];
        float new_max = fmaxf(prev_max, tile_max);
        float new_sum = prev_sum * expf(prev_max - new_max)
            + tile_sum * expf(tile_max - new_max);
        row_max[t] = new_max;
        row_sum[t] = new_sum;
        if (target >= vocab_start && target < vocab_start + tile_vocab) {
            int local = target - vocab_start;
            target_logit[t] = softcap * tanhf(row[local] * inv_cap);
        }
    }
}

extern "C" __global__ void output_ce_finalize_loss(
    const float* __restrict__ row_max,
    const float* __restrict__ row_sum,
    const float* __restrict__ target_logit,
    float* __restrict__ losses,
    int num_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) return;
    losses[idx] = row_max[idx] + logf(row_sum[idx]) - target_logit[idx];
}

extern "C" __global__ void loss_window_sum(
    const float* __restrict__ losses,
    float* __restrict__ out,
    int start,
    int end
) {
    __shared__ float scratch[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = start + tid; i < end; i += blockDim.x) {
        sum += losses[i];
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = scratch[0];
}

extern "C" __global__ void loss_window_accumulate(
    const float* __restrict__ losses,
    float* __restrict__ out,
    int start,
    int end
) {
    __shared__ float scratch[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = start + tid; i < end; i += blockDim.x) {
        sum += losses[i];
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, scratch[0]);
}

extern "C" __global__ void output_ce_tile_grad_bf16(
    const float* __restrict__ logits_tile,
    const int* __restrict__ targets,
    const float* __restrict__ row_max,
    const float* __restrict__ row_sum,
    unsigned short* __restrict__ grad_tile,
    int vocab_start,
    int tile_vocab,
    int tile_stride,
    float softcap,
    float grad_loss,
    int num_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = num_tokens * tile_vocab;
    if (idx >= n) return;

    int t = idx / tile_vocab;
    int local = idx - t * tile_vocab;
    int vocab_id = vocab_start + local;
    float logit = logits_tile[t * tile_stride + local];
    float inv_cap = 1.0f / softcap;
    float tv = tanhf(logit * inv_cap);
    float capped = softcap * tv;
    float prob = expf(capped - row_max[t]) / row_sum[t];
    float one_hot = (targets[t] == vocab_id) ? 1.0f : 0.0f;
    float grad = grad_loss * (prob - one_hot) * (1.0f - tv * tv);
    grad_tile[idx] = pg_f32_to_bf16(grad);
}

// ──────────────────────────────────────────────────────────────
// Bigram hash embedding gather
// MUST match CPU reference: hash = (36313*cur) ^ (27191*prev) % (num_buckets-1)
// Position 0 (no prev) maps to sentinel bucket (num_buckets - 1)
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void bigram_hash_embed(
    const int* __restrict__ ids,
    const float* __restrict__ embed,
    float* __restrict__ out,
    int bigram_vocab,
    int bigram_dim,
    int tokens,
    int seq_len
) {
    int t = blockIdx.x;
    if (t >= tokens) return;

    int bucket;
    if (t % seq_len == 0) {
        bucket = bigram_vocab - 1;  // sentinel
    } else {
        unsigned int curr = (unsigned int)ids[t];
        unsigned int prev = (unsigned int)ids[t - 1];
        unsigned int h = (curr * 36313u) ^ (prev * 27191u);
        bucket = (int)(h % (unsigned int)(bigram_vocab - 1));
    }

    for (int i = threadIdx.x; i < bigram_dim; i += blockDim.x) {
        out[t * bigram_dim + i] = embed[bucket * bigram_dim + i];
    }
}

// ──────────────────────────────────────────────────────────────
// Bigram hash embedding backward: scatter-add token gradients to buckets.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void bigram_hash_embed_backward(
    const int* __restrict__ ids,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_embed,
    int bigram_vocab,
    int bigram_dim,
    int tokens,
    int seq_len
) {
    int t = blockIdx.x;
    if (t >= tokens) return;

    int bucket;
    if (t % seq_len == 0) {
        bucket = bigram_vocab - 1;
    } else {
        unsigned int curr = (unsigned int)ids[t];
        unsigned int prev = (unsigned int)ids[t - 1];
        unsigned int h = (curr * 36313u) ^ (prev * 27191u);
        bucket = (int)(h % (unsigned int)(bigram_vocab - 1));
    }

    for (int i = threadIdx.x; i < bigram_dim; i += blockDim.x) {
        atomicAdd(&grad_embed[bucket * bigram_dim + i], grad_output[t * bigram_dim + i]);
    }
}

// ──────────────────────────────────────────────────────────────
// add_scaled: x[i] += alpha * y[i]
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void add_scaled(
    float* __restrict__ x,
    const float* __restrict__ y,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] += alpha * y[idx];
}

// x[i] += alpha * scale[0] * y[i]
extern "C" __global__ void add_scaled_by_param(
    float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ scale,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] += (alpha * scale[0]) * y[idx];
}

// x[i] += alpha * scale[index] * y[i]
extern "C" __global__ void add_scaled_by_param_index(
    float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ scale,
    int scale_index,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] += (alpha * scale[scale_index]) * y[idx];
}

// x[i] += alpha * scale_a[index_a] * scale_b[index_b] * y[i]
extern "C" __global__ void add_scaled_by_param_product(
    float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ scale_a,
    int scale_a_index,
    const float* __restrict__ scale_b,
    int scale_b_index,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] += (alpha * scale_a[scale_a_index] * scale_b[scale_b_index]) * y[idx];
}

// ──────────────────────────────────────────────────────────────
// Causal attention forward (naive, for parity testing ONLY)
// Grid: (tokens * num_heads,), Block: (head_dim,)
// O(T²) — NOT for production (use cuDNN FA3).
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void causal_attention_naive(
    const float* __restrict__ q,    // [T, H, D]
    const float* __restrict__ k,    // [T, Hkv, D]
    const float* __restrict__ v,    // [T, Hkv, D]
    float* __restrict__ out,        // [T, H, D]
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int head_idx = blockIdx.x;
    if (head_idx >= tokens * num_heads) return;

    int t = head_idx / num_heads;
    int h = head_idx % num_heads;
    int hkv = h / (num_heads / num_kv_heads);  // GQA mapping
    int seq_start = (t / seq_len) * seq_len;

    const float* q_head = q + head_idx * head_dim;
    float* out_head = out + head_idx * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    int d = threadIdx.x;
    if (d >= head_dim) return;

    // First pass: compute max score for numerical stability
    float max_score = -1e30f;
    for (int s = seq_start; s <= t; s++) {
        float score = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            score += q_head[dd] * k[(s * num_kv_heads + hkv) * head_dim + dd];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // Second pass: compute exp sum and weighted value
    float sum_exp = 0.0f;
    float val_acc = 0.0f;
    for (int s = seq_start; s <= t; s++) {
        float score = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            score += q_head[dd] * k[(s * num_kv_heads + hkv) * head_dim + dd];
        }
        score *= scale;
        float w = expf(score - max_score);
        sum_exp += w;
        val_acc += w * v[(s * num_kv_heads + hkv) * head_dim + d];
    }

    out_head[d] = val_acc / sum_exp;
}

// ──────────────────────────────────────────────────────────────
// Causal attention backward (naive O(T²), parity-first).
// One CTA per (token, query-head). K/V grads use atomicAdd because multiple
// future tokens contribute to the same KV positions.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void causal_attention_naive_backward(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int MAX_TOKENS = 2048;
    int head_idx = blockIdx.x;
    int total_heads = tokens * num_heads;
    if (head_idx >= total_heads || seq_len > MAX_TOKENS) return;

    int tok = head_idx / num_heads;
    int h = head_idx % num_heads;
    int group = num_heads / num_kv_heads;
    int kv_h = h / group;
    int seq_start = (tok / seq_len) * seq_len;
    int local_tok = tok - seq_start;
    float scale = rsqrtf((float)head_dim);

    const float* q_head = q + head_idx * head_dim;
    const float* go_head = grad_output + head_idx * head_dim;
    float* gq_head = grad_q + head_idx * head_dim;

    __shared__ float weights[MAX_TOKENS];
    __shared__ float grad_scores[MAX_TOKENS];

    if (threadIdx.x == 0) {
        float max_score = -1e30f;
        for (int i = 0; i <= local_tok; ++i) {
            int s = seq_start + i;
            const float* k_head = k + (s * num_kv_heads + kv_h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_head[d] * k_head[d];
            }
            float score = dot * scale;
            weights[i] = score;
            if (score > max_score) max_score = score;
        }

        float sum_exp = 0.0f;
        for (int i = 0; i <= local_tok; ++i) {
            float w = expf(weights[i] - max_score);
            weights[i] = w;
            sum_exp += w;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i <= local_tok; ++i) {
            weights[i] *= inv_sum;
        }

        float dot_wg = 0.0f;
        for (int i = 0; i <= local_tok; ++i) {
            int s = seq_start + i;
            const float* v_head = v + (s * num_kv_heads + kv_h) * head_dim;
            float grad_w = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                grad_w += go_head[d] * v_head[d];
            }
            grad_scores[i] = grad_w;
            dot_wg += weights[i] * grad_w;
        }

        for (int i = 0; i <= local_tok; ++i) {
            grad_scores[i] = weights[i] * (grad_scores[i] - dot_wg) * scale;
        }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        gq_head[d] = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i <= local_tok; ++i) {
        int s = seq_start + i;
        const float* k_head = k + (s * num_kv_heads + kv_h) * head_dim;
        int kv_off = (s * num_kv_heads + kv_h) * head_dim;
        float w = weights[i];
        float gs = grad_scores[i];
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            atomicAdd(&grad_v[kv_off + d], w * go_head[d]);
            atomicAdd(&grad_k[kv_off + d], gs * q_head[d]);
            gq_head[d] += gs * k_head[d];
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Flash-style causal attention forward (online softmax, f32).
//
// One CTA computes one (token, query-head) row. Unlike the parity kernel,
// the QK dot product for each source token is reduced once across the CTA
// and streamed directly into the output accumulator; no score matrix is
// materialized and no output lane recomputes the full dot product.
//
// This is the production f32 path until the BF16 cuDNN/FA3 backend is wired.
// It preserves exact row-major tensor layouts used by the CPU reference.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void causal_attention_online(
    const float* __restrict__ q,    // [T, H, D]
    const float* __restrict__ k,    // [T, Hkv, D]
    const float* __restrict__ v,    // [T, Hkv, D]
    float* __restrict__ out,        // [T, H, D]
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int head_idx = blockIdx.x;
    if (head_idx >= tokens * num_heads || head_dim > blockDim.x) return;

    int tok = head_idx / num_heads;
    int h = head_idx % num_heads;
    int group = num_heads / num_kv_heads;
    int kv_h = h / group;
    int seq_start = (tok / seq_len) * seq_len;
    float scale = rsqrtf((float)head_dim);

    int tid = threadIdx.x;
    const float* q_head = q + head_idx * head_dim;
    float acc = 0.0f;

    __shared__ float reduce[256];
    __shared__ float max_score_s;
    __shared__ float denom_s;
    __shared__ float weight_s;

    float max_score = -1e30f;
    for (int s = seq_start; s <= tok; ++s) {
        const float* k_head = k + (s * num_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            dot += q_head[d] * k_head[d];
        }
        reduce[tid] = dot;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            float score = reduce[0] * scale;
            if (score > max_score) max_score = score;
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_score_s = max_score;
        denom_s = 0.0f;
    }
    __syncthreads();

    for (int s = seq_start; s <= tok; ++s) {
        const float* k_head = k + (s * num_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            dot += q_head[d] * k_head[d];
        }
        reduce[tid] = dot;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            float w = expf(reduce[0] * scale - max_score_s);
            weight_s = w;
            denom_s += w;
        }
        __syncthreads();

        if (tid < head_dim) {
            acc += weight_s * v[(s * num_kv_heads + kv_h) * head_dim + tid];
        }
        __syncthreads();
    }

    if (tid < head_dim) {
        out[head_idx * head_dim + tid] = acc / denom_s;
    }
}

// ──────────────────────────────────────────────────────────────
// Flash-style causal attention backward (f32).
//
// The backward pass mirrors the CPU reference but parallelizes the expensive
// q·k and grad_out·v reductions across the CTA. K/V gradients remain atomics:
// multiple future query rows legitimately contribute to the same KV row.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void causal_attention_online_backward(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int MAX_TOKENS = 2048;
    int head_idx = blockIdx.x;
    if (head_idx >= tokens * num_heads || seq_len > MAX_TOKENS || head_dim > blockDim.x) return;

    int tok = head_idx / num_heads;
    int h = head_idx % num_heads;
    int group = num_heads / num_kv_heads;
    int kv_h = h / group;
    int seq_start = (tok / seq_len) * seq_len;
    int local_tok = tok - seq_start;
    int tid = threadIdx.x;
    float scale = rsqrtf((float)head_dim);

    const float* q_head = q + head_idx * head_dim;
    const float* go_head = grad_output + head_idx * head_dim;
    float* gq_head = grad_q + head_idx * head_dim;

    __shared__ float weights[MAX_TOKENS];
    __shared__ float grad_scores[MAX_TOKENS];
    __shared__ float reduce[256];
    __shared__ float dot_wg_s;

    float max_score = -1e30f;
    for (int i = 0; i <= local_tok; ++i) {
        int s = seq_start + i;
        const float* k_head = k + (s * num_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            dot += q_head[d] * k_head[d];
        }
        reduce[tid] = dot;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            float score = reduce[0] * scale;
            weights[i] = score;
            if (score > max_score) max_score = score;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum_exp = 0.0f;
        for (int i = 0; i <= local_tok; ++i) {
            float w = expf(weights[i] - max_score);
            weights[i] = w;
            sum_exp += w;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i <= local_tok; ++i) {
            weights[i] *= inv_sum;
        }
        dot_wg_s = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i <= local_tok; ++i) {
        int s = seq_start + i;
        const float* v_head = v + (s * num_kv_heads + kv_h) * head_dim;
        float grad_w = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            grad_w += go_head[d] * v_head[d];
        }
        reduce[tid] = grad_w;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            grad_scores[i] = reduce[0];
            dot_wg_s += weights[i] * reduce[0];
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int i = 0; i <= local_tok; ++i) {
            grad_scores[i] = weights[i] * (grad_scores[i] - dot_wg_s) * scale;
        }
    }
    __syncthreads();

    if (tid < head_dim) {
        gq_head[tid] = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i <= local_tok; ++i) {
        int s = seq_start + i;
        int kv_off = (s * num_kv_heads + kv_h) * head_dim;
        const float* k_head = k + kv_off;
        float w = weights[i];
        float gs = grad_scores[i];
        if (tid < head_dim) {
            atomicAdd(&grad_v[kv_off + tid], w * go_head[tid]);
            atomicAdd(&grad_k[kv_off + tid], gs * q_head[tid]);
            gq_head[tid] += gs * k_head[tid];
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Attention output gate.
//
// gate[t,h] = 2 * sigmoid(b[h] + dot(W[h], gate_input[t, :width]))
// out[t,h,:] = attn_in[t,h,:] * gate[t,h]
//
// One CTA handles one (token, head). The gate dot and head-dim reductions
// stay inside the CTA; parameter/input gradients use atomics because tokens
// and heads legitimately share destinations.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void attn_out_gate_forward(
    const float* __restrict__ attn_in,
    const float* __restrict__ gate_input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float* __restrict__ gate_values,
    int tokens,
    int num_heads,
    int head_dim,
    int model_dim,
    int gate_width
) {
    int row = blockIdx.x;
    if (row >= tokens * num_heads) return;
    int tok = row / num_heads;
    int head = row % num_heads;
    int tid = threadIdx.x;

    __shared__ float reduce[256];
    float score_part = 0.0f;
    const float* w = weight + head * gate_width;
    const float* x = gate_input + tok * model_dim;
    for (int j = tid; j < gate_width; j += blockDim.x) {
        score_part += w[j] * x[j];
    }
    reduce[tid] = score_part;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }

    __shared__ float gate_s;
    if (tid == 0) {
        float score = reduce[0] + bias[head];
        float sig = 1.0f / (1.0f + expf(-score));
        gate_s = 2.0f * sig;
        gate_values[row] = gate_s;
    }
    __syncthreads();

    int base = row * head_dim;
    for (int j = tid; j < head_dim; j += blockDim.x) {
        out[base + j] = attn_in[base + j] * gate_s;
    }
}

extern "C" __global__ void attn_out_gate_backward(
    const float* __restrict__ attn_in,
    const float* __restrict__ gate_input,
    const float* __restrict__ gate_values,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_attn_in,
    float* __restrict__ grad_gate_input,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int tokens,
    int num_heads,
    int head_dim,
    int model_dim,
    int gate_width
) {
    int row = blockIdx.x;
    if (row >= tokens * num_heads) return;
    int tok = row / num_heads;
    int head = row % num_heads;
    int tid = threadIdx.x;
    int base = row * head_dim;
    float gate = gate_values[row];

    __shared__ float reduce[256];
    float grad_gate_part = 0.0f;
    for (int j = tid; j < head_dim; j += blockDim.x) {
        float go = grad_out[base + j];
        grad_attn_in[base + j] = go * gate;
        grad_gate_part += go * attn_in[base + j];
    }
    reduce[tid] = grad_gate_part;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }

    __shared__ float grad_score_s;
    if (tid == 0) {
        float sig = 0.5f * gate;
        grad_score_s = reduce[0] * 2.0f * sig * (1.0f - sig);
        atomicAdd(&grad_bias[head], grad_score_s);
    }
    __syncthreads();

    const float* x = gate_input + tok * model_dim;
    const float* w = weight + head * gate_width;
    for (int j = tid; j < gate_width; j += blockDim.x) {
        atomicAdd(&grad_weight[head * gate_width + j], grad_score_s * x[j]);
        atomicAdd(&grad_gate_input[tok * model_dim + j], grad_score_s * w[j]);
    }
}

// PR1787/PR1797 SparseAttnGate:
// gate[t,h] = sigmoid(scale * dot(W[h], gate_input[t, :width]))
// out[t,h,:] = attn_in[t,h,:] * gate[t,h]
extern "C" __global__ void sparse_attn_gate_forward(
    const float* __restrict__ attn_in,
    const float* __restrict__ gate_input,
    const float* __restrict__ weight,
    float* __restrict__ out,
    float* __restrict__ gate_values,
    int tokens,
    int num_heads,
    int head_dim,
    int model_dim,
    int gate_width,
    float gate_scale
) {
    int row = blockIdx.x;
    if (row >= tokens * num_heads) return;
    int tok = row / num_heads;
    int head = row % num_heads;
    int tid = threadIdx.x;

    __shared__ float reduce[256];
    float score_part = 0.0f;
    const float* w = weight + head * gate_width;
    const float* x = gate_input + tok * model_dim;
    for (int j = tid; j < gate_width; j += blockDim.x) {
        score_part += w[j] * x[j];
    }
    reduce[tid] = score_part;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }

    __shared__ float gate_s;
    if (tid == 0) {
        float score = gate_scale * reduce[0];
        gate_s = 1.0f / (1.0f + expf(-score));
        gate_values[row] = gate_s;
    }
    __syncthreads();

    int base = row * head_dim;
    for (int j = tid; j < head_dim; j += blockDim.x) {
        out[base + j] = attn_in[base + j] * gate_s;
    }
}

extern "C" __global__ void sparse_attn_gate_xsa_forward(
    const float* __restrict__ y,
    const float* __restrict__ v,
    const float* __restrict__ gate_input,
    const float* __restrict__ weight,
    float* __restrict__ out,
    unsigned short* __restrict__ out_bf16,
    float* __restrict__ gate_values,
    int tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int model_dim,
    int gate_width,
    float gate_scale
) {
    int row = blockIdx.x;
    if (row >= tokens * num_heads) return;
    int tok = row / num_heads;
    int head = row % num_heads;
    int hkv = head / (num_heads / num_kv_heads);
    int y_off = row * head_dim;
    int v_off = (tok * num_kv_heads + hkv) * head_dim;
    int tid = threadIdx.x;

    __shared__ float reduce_yv[256];
    __shared__ float reduce_vn[256];
    __shared__ float reduce_score[256];

    float y_dot_v = 0.0f;
    float v_norm_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float yv = y[y_off + i];
        float vv = v[v_off + i];
        y_dot_v += yv * vv;
        v_norm_sq += vv * vv;
    }

    float score_part = 0.0f;
    const float* w = weight + head * gate_width;
    const float* x = gate_input + tok * model_dim;
    for (int j = tid; j < gate_width; j += blockDim.x) {
        score_part += w[j] * x[j];
    }

    reduce_yv[tid] = y_dot_v;
    reduce_vn[tid] = v_norm_sq;
    reduce_score[tid] = score_part;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_yv[tid] += reduce_yv[tid + stride];
            reduce_vn[tid] += reduce_vn[tid + stride];
            reduce_score[tid] += reduce_score[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float coeff_s;
    __shared__ float gate_s;
    if (tid == 0) {
        coeff_s = reduce_yv[0] / (reduce_vn[0] + 1e-8f);
        float score = gate_scale * reduce_score[0];
        gate_s = 1.0f / (1.0f + expf(-score));
        gate_values[row] = gate_s;
    }
    __syncthreads();

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float value = (y[y_off + i] - coeff_s * v[v_off + i]) * gate_s;
        out[y_off + i] = value;
        out_bf16[y_off + i] = pg_f32_to_bf16(value);
    }
}

extern "C" __global__ void sparse_attn_gate_xsa_forward_bf16_bhsd(
    const unsigned short* __restrict__ y_bhsd,
    const unsigned short* __restrict__ v_bhsd,
    const float* __restrict__ gate_input,
    const float* __restrict__ weight,
    float* __restrict__ out,
    unsigned short* __restrict__ out_bf16,
    float* __restrict__ gate_values,
    int batch,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int model_dim,
    int gate_width,
    float gate_scale
) {
    int row = blockIdx.x;
    int tokens = batch * seq_len;
    if (row >= tokens * num_heads) return;
    int tok = row / num_heads;
    int b = tok / seq_len;
    int s = tok - b * seq_len;
    int head = row % num_heads;
    int hkv = head / (num_heads / num_kv_heads);
    int y_bf16_off = ((b * num_heads + head) * seq_len + s) * head_dim;
    int v_bf16_off = ((b * num_kv_heads + hkv) * seq_len + s) * head_dim;
    int out_off = row * head_dim;
    int tid = threadIdx.x;

    __shared__ float reduce_yv[256];
    __shared__ float reduce_vn[256];
    __shared__ float reduce_score[256];

    float y_dot_v = 0.0f;
    float v_norm_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float yv = pg_bf16_to_f32(y_bhsd[y_bf16_off + i]);
        float vv = pg_bf16_to_f32(v_bhsd[v_bf16_off + i]);
        y_dot_v += yv * vv;
        v_norm_sq += vv * vv;
    }

    float score_part = 0.0f;
    const float* w = weight + head * gate_width;
    const float* x = gate_input + tok * model_dim;
    for (int j = tid; j < gate_width; j += blockDim.x) {
        score_part += w[j] * x[j];
    }

    reduce_yv[tid] = y_dot_v;
    reduce_vn[tid] = v_norm_sq;
    reduce_score[tid] = score_part;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_yv[tid] += reduce_yv[tid + stride];
            reduce_vn[tid] += reduce_vn[tid + stride];
            reduce_score[tid] += reduce_score[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float coeff_s;
    __shared__ float gate_s;
    if (tid == 0) {
        coeff_s = reduce_yv[0] / (reduce_vn[0] + 1e-8f);
        float score = gate_scale * reduce_score[0];
        gate_s = 1.0f / (1.0f + expf(-score));
        gate_values[row] = gate_s;
    }
    __syncthreads();

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float yv = pg_bf16_to_f32(y_bhsd[y_bf16_off + i]);
        float vv = pg_bf16_to_f32(v_bhsd[v_bf16_off + i]);
        float value = (yv - coeff_s * vv) * gate_s;
        out[out_off + i] = value;
        out_bf16[out_off + i] = pg_f32_to_bf16(value);
    }
}

extern "C" __global__ void sparse_attn_gate_backward(
    const float* __restrict__ attn_in,
    const float* __restrict__ gate_input,
    const float* __restrict__ gate_values,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_attn_in,
    float* __restrict__ grad_gate_input,
    float* __restrict__ grad_weight,
    int tokens,
    int num_heads,
    int head_dim,
    int model_dim,
    int gate_width,
    float gate_scale
) {
    int row = blockIdx.x;
    if (row >= tokens * num_heads) return;
    int tok = row / num_heads;
    int head = row % num_heads;
    int tid = threadIdx.x;
    int base = row * head_dim;
    float gate = gate_values[row];

    __shared__ float reduce[256];
    float grad_gate_part = 0.0f;
    for (int j = tid; j < head_dim; j += blockDim.x) {
        float go = grad_out[base + j];
        grad_attn_in[base + j] = go * gate;
        grad_gate_part += go * attn_in[base + j];
    }
    reduce[tid] = grad_gate_part;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }

    __shared__ float grad_score_s;
    if (tid == 0) {
        grad_score_s = reduce[0] * gate_scale * gate * (1.0f - gate);
    }
    __syncthreads();

    const float* x = gate_input + tok * model_dim;
    const float* w = weight + head * gate_width;
    for (int j = tid; j < gate_width; j += blockDim.x) {
        atomicAdd(&grad_weight[head * gate_width + j], grad_score_s * x[j]);
        atomicAdd(&grad_gate_input[tok * model_dim + j], grad_score_s * w[j]);
    }
}

extern "C" __global__ void sparse_attn_gate_backward_stage1(
    const float* __restrict__ attn_in,
    const float* __restrict__ gate_input,
    const float* __restrict__ gate_values,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_attn_in,
    float* __restrict__ grad_gate_input,
    float* __restrict__ grad_score_out,
    int tokens,
    int num_heads,
    int head_dim,
    int model_dim,
    int gate_width,
    float gate_scale
) {
    int row = blockIdx.x;
    if (row >= tokens * num_heads) return;
    int tok = row / num_heads;
    int head = row % num_heads;
    int tid = threadIdx.x;
    int base = row * head_dim;
    float gate = gate_values[row];

    __shared__ float reduce[256];
    float grad_gate_part = 0.0f;
    for (int j = tid; j < head_dim; j += blockDim.x) {
        float go = grad_out[base + j];
        grad_attn_in[base + j] = go * gate;
        grad_gate_part += go * attn_in[base + j];
    }
    reduce[tid] = grad_gate_part;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }

    __shared__ float grad_score_s;
    if (tid == 0) {
        grad_score_s = reduce[0] * gate_scale * gate * (1.0f - gate);
        grad_score_out[row] = grad_score_s;
    }
    __syncthreads();

    const float* w = weight + head * gate_width;
    for (int j = tid; j < gate_width; j += blockDim.x) {
        atomicAdd(&grad_gate_input[tok * model_dim + j], grad_score_s * w[j]);
    }
}

extern "C" __global__ void sparse_attn_gate_weight_grad_reduce(
    const float* __restrict__ gate_input,
    const float* __restrict__ grad_score,
    float* __restrict__ grad_weight,
    int tokens,
    int num_heads,
    int model_dim,
    int gate_width
) {
    int head = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;
    if (head >= num_heads || j >= gate_width) return;

    __shared__ float reduce[256];
    float sum = 0.0f;
    for (int tok = tid; tok < tokens; tok += blockDim.x) {
        sum += grad_score[tok * num_heads + head] * gate_input[tok * model_dim + j];
    }
    reduce[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        grad_weight[head * gate_width + j] += reduce[0];
    }
}

extern "C" __global__ void sparse_attn_gate_xsa_backward_bf16_bhsd(
    const unsigned short* __restrict__ y_bhsd,
    const unsigned short* __restrict__ v_bhsd,
    const float* __restrict__ gate_input,
    const float* __restrict__ gate_values,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_y,
    float* __restrict__ grad_v,
    float* __restrict__ grad_gate_input,
    float* __restrict__ grad_score_out,
    int batch,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int model_dim,
    int gate_width,
    float gate_scale
) {
    int head_idx = blockIdx.x;
    int tokens = batch * seq_len;
    if (head_idx >= tokens * num_heads) return;

    int token = head_idx / num_heads;
    int b = token / seq_len;
    int s = token - b * seq_len;
    int h = head_idx % num_heads;
    int hkv = h / (num_heads / num_kv_heads);
    int y_f32_off = head_idx * head_dim;
    int v_f32_off = (token * num_kv_heads + hkv) * head_dim;
    int y_bf16_off = ((b * num_heads + h) * seq_len + s) * head_dim;
    int v_bf16_off = ((b * num_kv_heads + hkv) * seq_len + s) * head_dim;
    int tid = threadIdx.x;
    float gate = gate_values[head_idx];

    __shared__ float reduce_a[256];
    __shared__ float reduce_b[256];

    float y_dot_v = 0.0f;
    float v_norm_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float yv = pg_bf16_to_f32(y_bhsd[y_bf16_off + i]);
        float vv = pg_bf16_to_f32(v_bhsd[v_bf16_off + i]);
        y_dot_v += yv * vv;
        v_norm_sq += vv * vv;
    }
    reduce_a[tid] = y_dot_v;
    reduce_b[tid] = v_norm_sq;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_a[tid] += reduce_a[tid + stride];
            reduce_b[tid] += reduce_b[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float y_dot_v_s;
    __shared__ float v_norm_sq_s;
    __shared__ float coeff_s;
    if (tid == 0) {
        y_dot_v_s = reduce_a[0];
        v_norm_sq_s = reduce_b[0] + 1e-8f;
        coeff_s = y_dot_v_s / v_norm_sq_s;
    }
    __syncthreads();

    float grad_gate_part = 0.0f;
    float go_dot_v = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float yv = pg_bf16_to_f32(y_bhsd[y_bf16_off + i]);
        float vv = pg_bf16_to_f32(v_bhsd[v_bf16_off + i]);
        float go = grad_out[y_f32_off + i];
        float xsa = yv - coeff_s * vv;
        float gxsa = go * gate;
        grad_gate_part += go * xsa;
        go_dot_v += gxsa * vv;
    }
    reduce_a[tid] = grad_gate_part;
    reduce_b[tid] = go_dot_v;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_a[tid] += reduce_a[tid + stride];
            reduce_b[tid] += reduce_b[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float grad_score_s;
    __shared__ float go_dot_v_s;
    if (tid == 0) {
        grad_score_s = reduce_a[0] * gate_scale * gate * (1.0f - gate);
        go_dot_v_s = reduce_b[0];
        grad_score_out[head_idx] = grad_score_s;
    }
    __syncthreads();

    float go_coeff = go_dot_v_s / v_norm_sq_s;
    float v_norm_sq2 = v_norm_sq_s * v_norm_sq_s;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float yv = pg_bf16_to_f32(y_bhsd[y_bf16_off + i]);
        float vv = pg_bf16_to_f32(v_bhsd[v_bf16_off + i]);
        float gxsa = grad_out[y_f32_off + i] * gate;
        grad_y[y_f32_off + i] = gxsa - go_coeff * vv;
        float gv = -coeff_s * gxsa - go_coeff * yv
            + (2.0f * y_dot_v_s * go_dot_v_s / v_norm_sq2) * vv;
        atomicAdd(&grad_v[v_f32_off + i], gv);
    }

    const float* w = weight + h * gate_width;
    for (int j = tid; j < gate_width; j += blockDim.x) {
        atomicAdd(&grad_gate_input[token * model_dim + j], grad_score_s * w[j]);
    }
}

extern "C" __global__ void sparse_attn_gate_xsa_backward_bf16_bhsd_warpheads(
    const unsigned short* __restrict__ y_bhsd,
    const unsigned short* __restrict__ v_bhsd,
    const float* __restrict__ gate_input,
    const float* __restrict__ gate_values,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_y,
    float* __restrict__ grad_v,
    float* __restrict__ grad_gate_input,
    float* __restrict__ grad_score_out,
    int batch,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int model_dim,
    int gate_width,
    float gate_scale
) {
    int token = blockIdx.x;
    int tokens = batch * seq_len;
    if (token >= tokens) return;

    int lane = threadIdx.x & 31;
    int h = threadIdx.x >> 5;
    if (h >= num_heads) return;

    int b = token / seq_len;
    int s = token - b * seq_len;
    int hkv = h / (num_heads / num_kv_heads);
    int head_idx = token * num_heads + h;
    int y_f32_off = head_idx * head_dim;
    int v_f32_off = (token * num_kv_heads + hkv) * head_dim;
    int y_bf16_off = ((b * num_heads + h) * seq_len + s) * head_dim;
    int v_bf16_off = ((b * num_kv_heads + hkv) * seq_len + s) * head_dim;
    float gate = gate_values[head_idx];

    float y_dot_v = 0.0f;
    float v_norm_sq = 0.0f;
    for (int i = lane; i < head_dim; i += 32) {
        float yv = pg_bf16_to_f32(y_bhsd[y_bf16_off + i]);
        float vv = pg_bf16_to_f32(v_bhsd[v_bf16_off + i]);
        y_dot_v += yv * vv;
        v_norm_sq += vv * vv;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        y_dot_v += __shfl_down_sync(0xffffffff, y_dot_v, offset);
        v_norm_sq += __shfl_down_sync(0xffffffff, v_norm_sq, offset);
    }
    float y_dot_v_s = __shfl_sync(0xffffffff, y_dot_v, 0);
    float v_norm_sq_s = __shfl_sync(0xffffffff, v_norm_sq, 0) + 1e-8f;
    float coeff = y_dot_v_s / v_norm_sq_s;

    float grad_gate_part = 0.0f;
    float go_dot_v = 0.0f;
    for (int i = lane; i < head_dim; i += 32) {
        float yv = pg_bf16_to_f32(y_bhsd[y_bf16_off + i]);
        float vv = pg_bf16_to_f32(v_bhsd[v_bf16_off + i]);
        float go = grad_out[y_f32_off + i];
        float xsa = yv - coeff * vv;
        float gxsa = go * gate;
        grad_gate_part += go * xsa;
        go_dot_v += gxsa * vv;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        grad_gate_part += __shfl_down_sync(0xffffffff, grad_gate_part, offset);
        go_dot_v += __shfl_down_sync(0xffffffff, go_dot_v, offset);
    }
    float grad_gate_sum = __shfl_sync(0xffffffff, grad_gate_part, 0);
    float go_dot_v_s = __shfl_sync(0xffffffff, go_dot_v, 0);
    float grad_score = grad_gate_sum * gate_scale * gate * (1.0f - gate);
    if (lane == 0) {
        grad_score_out[head_idx] = grad_score;
    }

    float go_coeff = go_dot_v_s / v_norm_sq_s;
    float v_norm_sq2 = v_norm_sq_s * v_norm_sq_s;
    for (int i = lane; i < head_dim; i += 32) {
        float yv = pg_bf16_to_f32(y_bhsd[y_bf16_off + i]);
        float vv = pg_bf16_to_f32(v_bhsd[v_bf16_off + i]);
        float gxsa = grad_out[y_f32_off + i] * gate;
        grad_y[y_f32_off + i] = gxsa - go_coeff * vv;
        float gv = -coeff * gxsa - go_coeff * yv
            + (2.0f * y_dot_v_s * go_dot_v_s / v_norm_sq2) * vv;
        atomicAdd(&grad_v[v_f32_off + i], gv);
    }

    if (lane < gate_width) {
        atomicAdd(
            &grad_gate_input[token * model_dim + lane],
            grad_score * weight[h * gate_width + lane]
        );
    }
}

// ──────────────────────────────────────────────────────────────
// XSA (Exclusive Self Attention) forward
// Projects out self-value component: z = y - (y·v / ||v||²) * v
// Grid: (tokens * num_heads,), Block: (64,) — uses strided loops
// and cross-warp shared memory reduction for head_dim > 32.
// GQA-aware: multiple Q heads map to the same KV head.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void xsa_forward(
    const float* __restrict__ attn_out,  // [T, H, D]
    const float* __restrict__ v,         // [T, Hkv, D]
    float* __restrict__ out,             // [T, H, D]
    int tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int head_idx = blockIdx.x;
    if (head_idx >= tokens * num_heads) return;

    int t = head_idx / num_heads;
    int h = head_idx % num_heads;
    int hkv = h / (num_heads / num_kv_heads);

    // Strided accumulation for dot products (handles head_dim > blockDim.x)
    float y_dot_v_acc = 0.0f;
    float v_norm_sq_acc = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float y_d = attn_out[head_idx * head_dim + i];
        float v_d = v[(t * num_kv_heads + hkv) * head_dim + i];
        y_dot_v_acc += y_d * v_d;
        v_norm_sq_acc += v_d * v_d;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        y_dot_v_acc += __shfl_down_sync(0xffffffff, y_dot_v_acc, offset);
        v_norm_sq_acc += __shfl_down_sync(0xffffffff, v_norm_sq_acc, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float shared_dot[32];
    __shared__ float shared_norm[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_dot[warp_id] = y_dot_v_acc;
        shared_norm[warp_id] = v_norm_sq_acc;
    }
    __syncthreads();

    // Final reduction in first warp
    if (threadIdx.x < 32) {
        y_dot_v_acc = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_dot[threadIdx.x] : 0.0f;
        v_norm_sq_acc = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_norm[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            y_dot_v_acc += __shfl_down_sync(0xffffffff, y_dot_v_acc, offset);
            v_norm_sq_acc += __shfl_down_sync(0xffffffff, v_norm_sq_acc, offset);
        }
    }

    // Broadcast coefficient from thread 0
    __shared__ float coeff_shared;
    if (threadIdx.x == 0) {
        coeff_shared = y_dot_v_acc / (v_norm_sq_acc + 1e-8f);
    }
    __syncthreads();
    float coeff = coeff_shared;

    // z = y - coeff * v (strided write)
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float y_d = attn_out[head_idx * head_dim + i];
        float v_d = v[(t * num_kv_heads + hkv) * head_dim + i];
        out[head_idx * head_dim + i] = y_d - coeff * v_d;
    }
}

// ──────────────────────────────────────────────────────────────
// XSA backward. grad_v is accumulated because several Q heads share one KV head.
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void xsa_backward(
    const float* __restrict__ y,
    const float* __restrict__ v,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_y,
    float* __restrict__ grad_v,
    int tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int head_idx = blockIdx.x;
    if (head_idx >= tokens * num_heads) return;

    int t = head_idx / num_heads;
    int h = head_idx % num_heads;
    int hkv = h / (num_heads / num_kv_heads);
    int y_off = head_idx * head_dim;
    int v_off = (t * num_kv_heads + hkv) * head_dim;

    float y_dot_v = 0.0f;
    float v_norm_sq = 0.0f;
    float go_dot_v = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float yv = y[y_off + i];
        float vv = v[v_off + i];
        float go = grad_output[y_off + i];
        y_dot_v += yv * vv;
        v_norm_sq += vv * vv;
        go_dot_v += go * vv;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        y_dot_v += __shfl_down_sync(0xffffffff, y_dot_v, offset);
        v_norm_sq += __shfl_down_sync(0xffffffff, v_norm_sq, offset);
        go_dot_v += __shfl_down_sync(0xffffffff, go_dot_v, offset);
    }

    __shared__ float shared_yv[32];
    __shared__ float shared_vn[32];
    __shared__ float shared_gv[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_yv[warp_id] = y_dot_v;
        shared_vn[warp_id] = v_norm_sq;
        shared_gv[warp_id] = go_dot_v;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        y_dot_v = (threadIdx.x < warps) ? shared_yv[threadIdx.x] : 0.0f;
        v_norm_sq = (threadIdx.x < warps) ? shared_vn[threadIdx.x] : 0.0f;
        go_dot_v = (threadIdx.x < warps) ? shared_gv[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            y_dot_v += __shfl_down_sync(0xffffffff, y_dot_v, offset);
            v_norm_sq += __shfl_down_sync(0xffffffff, v_norm_sq, offset);
            go_dot_v += __shfl_down_sync(0xffffffff, go_dot_v, offset);
        }
    }

    __shared__ float yv_sh;
    __shared__ float vn_sh;
    __shared__ float gv_sh;
    if (threadIdx.x == 0) {
        yv_sh = y_dot_v;
        vn_sh = v_norm_sq + 1e-8f;
        gv_sh = go_dot_v;
    }
    __syncthreads();
    y_dot_v = yv_sh;
    v_norm_sq = vn_sh;
    go_dot_v = gv_sh;

    float coeff = y_dot_v / v_norm_sq;
    float go_coeff = go_dot_v / v_norm_sq;
    float v_norm_sq2 = v_norm_sq * v_norm_sq;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float yv = y[y_off + i];
        float vv = v[v_off + i];
        float go = grad_output[y_off + i];
        grad_y[y_off + i] = go - go_coeff * vv;
        float gv = -coeff * go - go_coeff * yv + (2.0f * y_dot_v * go_dot_v / v_norm_sq2) * vv;
        atomicAdd(&grad_v[v_off + i], gv);
    }
}

// XSA backward from saved cuDNN BF16 BHSD attention tensors. The public model
// tensors are logically [B, T, H, D], while cuDNN SDPA stores saved tensors as
// [B, H, T, D]. This path avoids saving/copying the full F32 attention output
// and V projection solely for XSA backward.
extern "C" __global__ void xsa_backward_bf16_bhsd(
    const unsigned short* __restrict__ y_bhsd,
    const unsigned short* __restrict__ v_bhsd,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_y,
    float* __restrict__ grad_v,
    int batch,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int head_idx = blockIdx.x;
    int tokens = batch * seq_len;
    if (head_idx >= tokens * num_heads) return;

    int token = head_idx / num_heads;
    int b = token / seq_len;
    int s = token - b * seq_len;
    int h = head_idx % num_heads;
    int hkv = h / (num_heads / num_kv_heads);
    int y_f32_off = head_idx * head_dim;
    int v_f32_off = (token * num_kv_heads + hkv) * head_dim;
    int y_bf16_off = ((b * num_heads + h) * seq_len + s) * head_dim;
    int v_bf16_off = ((b * num_kv_heads + hkv) * seq_len + s) * head_dim;

    float y_dot_v = 0.0f;
    float v_norm_sq = 0.0f;
    float go_dot_v = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float yv = pg_bf16_to_f32(y_bhsd[y_bf16_off + i]);
        float vv = pg_bf16_to_f32(v_bhsd[v_bf16_off + i]);
        float go = grad_output[y_f32_off + i];
        y_dot_v += yv * vv;
        v_norm_sq += vv * vv;
        go_dot_v += go * vv;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        y_dot_v += __shfl_down_sync(0xffffffff, y_dot_v, offset);
        v_norm_sq += __shfl_down_sync(0xffffffff, v_norm_sq, offset);
        go_dot_v += __shfl_down_sync(0xffffffff, go_dot_v, offset);
    }

    __shared__ float shared_yv[32];
    __shared__ float shared_vn[32];
    __shared__ float shared_gv[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_yv[warp_id] = y_dot_v;
        shared_vn[warp_id] = v_norm_sq;
        shared_gv[warp_id] = go_dot_v;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) / 32;
        y_dot_v = (threadIdx.x < warps) ? shared_yv[threadIdx.x] : 0.0f;
        v_norm_sq = (threadIdx.x < warps) ? shared_vn[threadIdx.x] : 0.0f;
        go_dot_v = (threadIdx.x < warps) ? shared_gv[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            y_dot_v += __shfl_down_sync(0xffffffff, y_dot_v, offset);
            v_norm_sq += __shfl_down_sync(0xffffffff, v_norm_sq, offset);
            go_dot_v += __shfl_down_sync(0xffffffff, go_dot_v, offset);
        }
    }

    __shared__ float yv_sh;
    __shared__ float vn_sh;
    __shared__ float gv_sh;
    if (threadIdx.x == 0) {
        yv_sh = y_dot_v;
        vn_sh = v_norm_sq + 1e-8f;
        gv_sh = go_dot_v;
    }
    __syncthreads();
    y_dot_v = yv_sh;
    v_norm_sq = vn_sh;
    go_dot_v = gv_sh;

    float coeff = y_dot_v / v_norm_sq;
    float go_coeff = go_dot_v / v_norm_sq;
    float v_norm_sq2 = v_norm_sq * v_norm_sq;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float yv = pg_bf16_to_f32(y_bhsd[y_bf16_off + i]);
        float vv = pg_bf16_to_f32(v_bhsd[v_bf16_off + i]);
        float go = grad_output[y_f32_off + i];
        grad_y[y_f32_off + i] = go - go_coeff * vv;
        float gv = -coeff * go - go_coeff * yv + (2.0f * y_dot_v * go_dot_v / v_norm_sq2) * vv;
        atomicAdd(&grad_v[v_f32_off + i], gv);
    }
}
"#;

fn residual_scale_reduce_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("PG_GPU_RESIDUAL_SCALE_REDUCE")
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

impl GpuKernels {
    /// Initialize GPU kernels by compiling CUDA source to PTX and loading all functions.
    pub fn new(ctx: Arc<CudaContext>, stream: Arc<CudaStream>) -> PgResult<Self> {
        // Compile CUDA source to PTX using NVRTC
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(
            CUDA_SOURCE,
            cudarc::nvrtc::safe::CompileOptions {
                // NOTE: fast_math disabled for parity. Enable after GPU-CPU match is confirmed.
                use_fast_math: None,
                arch: Some("sm_90"), // H100
                ..Default::default()
            },
        )
        .map_err(|e| PgError::InvalidOp(format!("NVRTC compilation failed: {:?}", e)))?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| PgError::InvalidOp(format!("PTX module load failed: {:?}", e)))?;

        Ok(Self {
            stream,
            _module: module.clone(),
            rms_norm_fwd: module
                .load_function("rms_norm_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rms_norm_fwd_bf16: module
                .load_function("rms_norm_forward_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rms_norm_bwd: module
                .load_function("rms_norm_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rms_norm_bwd_go_bf16: module
                .load_function("rms_norm_backward_go_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rms_norm_bwd_accum: module
                .load_function("rms_norm_backward_accum")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rms_norm_bwd_accum_residual_mix_input: module
                .load_function("rms_norm_backward_accum_residual_mix_input")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rms_norm_bwd_accum_residual_mix_input_go_bf16: module
                .load_function("rms_norm_backward_accum_residual_mix_input_go_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rms_norm_bwd_residual_mix: module
                .load_function("rms_norm_backward_accum_residual_mix_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rms_norm_bwd_residual_mix_chunked: module
                .load_function("rms_norm_backward_accum_residual_mix_backward_chunked")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rms_norm_bwd_residual_mix_no_mix_grad: module
                .load_function("rms_norm_backward_accum_residual_mix_backward_no_mix_grad")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rms_norm_bwd_residual_mix_recompute: module
                .load_function("rms_norm_backward_accum_residual_mix_backward_recompute")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_mix_grad_reduce_chunks: module
                .load_function("residual_mix_grad_reduce_chunks")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_mix_grad_reduce_from_stats: module
                .load_function("residual_mix_grad_reduce_from_stats")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            leaky_relu_sq_fwd: module
                .load_function("leaky_relu_sq_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            leaky_relu_sq_fwd_bf16: module
                .load_function("leaky_relu_sq_forward_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            leaky_relu_sq_fwd_x_bf16_only: module
                .load_function("leaky_relu_sq_forward_x_bf16_only")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            leaky_relu_sq_bwd: module
                .load_function("leaky_relu_sq_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            leaky_relu_sq_bwd_bf16: module
                .load_function("leaky_relu_sq_backward_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            leaky_relu_sq_bwd_x_bf16: module
                .load_function("leaky_relu_sq_backward_x_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            leaky_relu_sq_bwd_x_bf16_only: module
                .load_function("leaky_relu_sq_backward_x_bf16_only")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_mix: module
                .load_function("residual_mix")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_mix_rms_norm_fwd: module
                .load_function("residual_mix_rms_norm_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_mix_rms_norm_fwd_bf16: module
                .load_function("residual_mix_rms_norm_forward_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_mix_bwd: module
                .load_function("residual_mix_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale: module
                .load_function("residual_add_scale")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_bf16_proj: module
                .load_function("residual_add_scale_bf16_proj")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_from_base: module
                .load_function("residual_add_scale_from_base")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_from_base_bf16_proj: module
                .load_function("residual_add_scale_from_base_bf16_proj")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_from_base_rms_norm: module
                .load_function("residual_add_scale_from_base_rms_norm_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_from_base_rms_norm_bf16: module
                .load_function("residual_add_scale_from_base_rms_norm_forward_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_from_base_rms_norm_bf16_proj: module
                .load_function("residual_add_scale_from_base_rms_norm_forward_bf16_proj")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_bwd: module
                .load_function("residual_add_scale_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_bwd_bf16: module
                .load_function("residual_add_scale_backward_bf16")
                .map_err(|e| {
                PgError::InvalidOp(format!("Failed to load kernel {}", e))
            })?,
            residual_add_scale_bwd_from_bf16_only: module
                .load_function("residual_add_scale_backward_from_bf16_only")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_bwd_bf16_only_atomic: module
                .load_function("residual_add_scale_backward_bf16_only")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_bwd_bf16_only_no_atomic: module
                .load_function("residual_add_scale_backward_bf16_only_no_atomic")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_grad_scale_reduce: module
                .load_function("residual_add_scale_grad_scale_reduce")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale_grad_scale_reduce_bf16: module
                .load_function("residual_add_scale_grad_scale_reduce_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            smear_gate_fwd: module
                .load_function("smear_gate_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            smear_gate_fwd_boundary: module
                .load_function("smear_gate_forward_boundary")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            smear_gate_bwd: module
                .load_function("smear_gate_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            smear_gate_bwd_boundary: module
                .load_function("smear_gate_backward_boundary")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            embedding_gather: module
                .load_function("embedding_gather")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            embedding_gather_bwd: module
                .load_function("embedding_gather_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            qk_norm_fwd: module
                .load_function("qk_norm_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            qk_norm_bwd: module
                .load_function("qk_norm_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            partial_rope_fwd: module
                .load_function("partial_rope_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            partial_rope_bwd: module
                .load_function("partial_rope_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            q_gain_fwd: module
                .load_function("q_gain_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            q_gain_bwd: module
                .load_function("q_gain_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            q_gain_rope_qk_norm_fwd: module
                .load_function("q_gain_rope_qk_norm_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            q_gain_rope_qk_norm_fwd_bf16_bhsd: module
                .load_function("q_gain_rope_qk_norm_forward_bf16_bhsd")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rope_qk_norm_fwd: module
                .load_function("rope_qk_norm_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rope_qk_norm_fwd_bf16_bhsd: module
                .load_function("rope_qk_norm_forward_bf16_bhsd")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            q_gain_rope_qk_norm_bwd: module
                .load_function("q_gain_rope_qk_norm_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            q_gain_rope_qk_norm_bwd_chunked: module
                .load_function("q_gain_rope_qk_norm_backward_chunked")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            q_gain_reduce_chunked: module
                .load_function("q_gain_reduce_chunked")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            rope_qk_norm_bwd: module
                .load_function("rope_qk_norm_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            dot_accumulate: module
                .load_function("dot_accumulate")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            dot_accumulate_by_param: module
                .load_function("dot_accumulate_by_param")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            clip_by_global_norm: module
                .load_function("clip_by_global_norm")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            cross_entropy_fwd: module
                .load_function("cross_entropy_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            cross_entropy_fwd_bf16_logits: module
                .load_function("cross_entropy_forward_bf16_logits")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            cross_entropy_bwd: module
                .load_function("cross_entropy_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            cross_entropy_bwd_bf16: module
                .load_function("cross_entropy_backward_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            cross_entropy_bwd_bf16_logits: module
                .load_function("cross_entropy_backward_bf16_logits")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            cross_entropy_loss_bwd_bf16: module
                .load_function("cross_entropy_loss_backward_bf16")
                .map_err(|e| {
                PgError::InvalidOp(format!("Failed to load kernel {}", e))
            })?,
            cross_entropy_loss_bwd_bf16_logits: module
                .load_function("cross_entropy_loss_backward_bf16_logits")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            output_ce_stats_init: module
                .load_function("output_ce_stats_init")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            output_ce_tile_stats_update: module
                .load_function("output_ce_tile_stats_update")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            output_ce_finalize_loss: module
                .load_function("output_ce_finalize_loss")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            loss_window_sum: module
                .load_function("loss_window_sum")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            loss_window_accumulate: module
                .load_function("loss_window_accumulate")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            output_ce_tile_grad_bf16: module
                .load_function("output_ce_tile_grad_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            bigram_hash_embed: module
                .load_function("bigram_hash_embed")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            bigram_hash_embed_bwd: module
                .load_function("bigram_hash_embed_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            add_scaled: module
                .load_function("add_scaled")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            add_scaled_by_param: module
                .load_function("add_scaled_by_param")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            add_scaled_by_param_index: module
                .load_function("add_scaled_by_param_index")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            add_scaled_by_param_product: module
                .load_function("add_scaled_by_param_product")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            causal_attention_naive: module
                .load_function("causal_attention_naive")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            causal_attention_naive_bwd: module
                .load_function("causal_attention_naive_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            causal_attention_online: module
                .load_function("causal_attention_online")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            causal_attention_online_bwd: module
                .load_function("causal_attention_online_backward")
                .map_err(|e| {
                PgError::InvalidOp(format!("Failed to load kernel {}", e))
            })?,
            attn_out_gate_fwd: module
                .load_function("attn_out_gate_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            attn_out_gate_bwd: module
                .load_function("attn_out_gate_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            sparse_attn_gate_fwd: module
                .load_function("sparse_attn_gate_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            sparse_attn_gate_xsa_fwd: module
                .load_function("sparse_attn_gate_xsa_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            sparse_attn_gate_xsa_fwd_bf16_bhsd: module
                .load_function("sparse_attn_gate_xsa_forward_bf16_bhsd")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            sparse_attn_gate_bwd: module
                .load_function("sparse_attn_gate_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            sparse_attn_gate_bwd_stage1: module
                .load_function("sparse_attn_gate_backward_stage1")
                .map_err(|e| {
                PgError::InvalidOp(format!("Failed to load kernel {}", e))
            })?,
            sparse_attn_gate_xsa_bwd_bf16_bhsd: module
                .load_function("sparse_attn_gate_xsa_backward_bf16_bhsd")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            sparse_attn_gate_xsa_bwd_bf16_bhsd_warpheads: module
                .load_function("sparse_attn_gate_xsa_backward_bf16_bhsd_warpheads")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            sparse_attn_gate_weight_grad: module
                .load_function("sparse_attn_gate_weight_grad_reduce")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            xsa_fwd: module
                .load_function("xsa_forward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            xsa_bwd: module
                .load_function("xsa_backward")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            xsa_bwd_bf16_bhsd: module
                .load_function("xsa_backward_bf16_bhsd")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            copy_fwd: module
                .load_function("copy_fwd")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            copy_u16_fwd: module
                .load_function("copy_u16_fwd")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            fill_u16: module
                .load_function("fill_u16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            bthd_to_bhsd_bf16: module
                .load_function("bthd_to_bhsd_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            bhsd_to_bthd_bf16: module
                .load_function("bhsd_to_bthd_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            scale_inplace: module
                .load_function("scale_inplace")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            pack_qkv_weights: module
                .load_function("pack_qkv_weights")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            unpack_qkv_output: module
                .load_function("unpack_qkv_output")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            pack_qkv_grads: module
                .load_function("pack_qkv_grads")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            pack_qkv_grads_bf16: module
                .load_function("pack_qkv_grads_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            unpack_qkv_weight_grad: module
                .load_function("unpack_qkv_weight_grad")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            f32_to_bf16: module
                .load_function("f32_to_bf16")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            bf16_to_f32: module
                .load_function("bf16_to_f32")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            normalize_matrices: module
                .load_function("normalize_matrices")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            decay_sgd_step: module
                .load_function("decay_sgd_step")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            adamw_step: module
                .load_function("adamw_step")
                .map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
        })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    // ── Dispatch wrappers ──────────────────────────────────────

    /// RMSNorm forward: y = (x / rms(x)) * ln_scale_factor
    pub fn rms_norm_forward(
        &self,
        x: CudaPtr,
        y: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_fwd)
                .arg(&x)
                .arg(&y)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 128,
                })
                .map_err(|e| PgError::InvalidOp(format!("rms_norm launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// RMSNorm forward with an additional BF16 side output for tensor-core consumers.
    pub fn rms_norm_forward_bf16(
        &self,
        x: CudaPtr,
        y: CudaPtr,
        y_bf16: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_fwd_bf16)
                .arg(&x)
                .arg(&y)
                .arg(&y_bf16)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 128,
                })
                .map_err(|e| PgError::InvalidOp(format!("rms_norm_bf16 launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// RMSNorm backward.
    pub fn rms_norm_backward(
        &self,
        x: CudaPtr,
        grad_output: CudaPtr,
        grad_input: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_bwd)
                .arg(&x)
                .arg(&grad_output)
                .arg(&grad_input)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| PgError::InvalidOp(format!("rms_norm_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// RMSNorm backward where the upstream gradient is BF16. This is used on
    /// record-shaped BF16 GEMM paths to avoid materializing a full F32
    /// activation-gradient tensor before the norm boundary.
    pub fn rms_norm_backward_go_bf16(
        &self,
        x: CudaPtr,
        grad_output_bf16: CudaPtr,
        grad_input: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_bwd_go_bf16)
                .arg(&x)
                .arg(&grad_output_bf16)
                .arg(&grad_input)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| PgError::InvalidOp(format!("rms_norm_bwd_go_bf16 launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn copy_fwd(&self, src: CudaPtr, dst: CudaPtr, n: u32) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.copy_fwd)
                .arg(&src)
                .arg(&dst)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("copy_fwd failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn copy_u16_fwd(&self, src: CudaPtr, dst: CudaPtr, n: u32) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.copy_u16_fwd)
                .arg(&src)
                .arg(&dst)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("copy_u16_fwd failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn fill_u16(&self, dst: CudaPtr, value: u16, n: u32) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.fill_u16)
                .arg(&dst)
                .arg(&value)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("fill_u16 failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn scale_inplace(&self, x: CudaPtr, scale: f32, n: u32) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.scale_inplace)
                .arg(&x)
                .arg(&scale)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("scale_inplace failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn pack_qkv_weights(
        &self,
        qo_bank: CudaPtr,
        kv_bank: CudaPtr,
        qkv_bank: CudaPtr,
        layers: u32,
        d: u32,
        kv: u32,
    ) -> PgResult<()> {
        let n = layers * (d + 2 * kv) * d;
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.pack_qkv_weights)
                .arg(&qo_bank)
                .arg(&kv_bank)
                .arg(&qkv_bank)
                .arg(&(layers as i32))
                .arg(&(d as i32))
                .arg(&(kv as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("pack_qkv_weights failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn unpack_qkv_output(
        &self,
        combined: CudaPtr,
        q: CudaPtr,
        k: CudaPtr,
        v: CudaPtr,
        tokens: u32,
        d: u32,
        kv: u32,
    ) -> PgResult<()> {
        let n = tokens * (d + 2 * kv);
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.unpack_qkv_output)
                .arg(&combined)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&(tokens as i32))
                .arg(&(d as i32))
                .arg(&(kv as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("unpack_qkv_output failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn pack_qkv_grads(
        &self,
        grad_q: CudaPtr,
        grad_k: CudaPtr,
        grad_v: CudaPtr,
        combined: CudaPtr,
        tokens: u32,
        d: u32,
        kv: u32,
    ) -> PgResult<()> {
        let n = tokens * (d + 2 * kv);
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.pack_qkv_grads)
                .arg(&grad_q)
                .arg(&grad_k)
                .arg(&grad_v)
                .arg(&combined)
                .arg(&(tokens as i32))
                .arg(&(d as i32))
                .arg(&(kv as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("pack_qkv_grads failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn pack_qkv_grads_bf16(
        &self,
        grad_q: CudaPtr,
        grad_k: CudaPtr,
        grad_v: CudaPtr,
        combined: CudaPtr,
        tokens: u32,
        d: u32,
        kv: u32,
    ) -> PgResult<()> {
        let n = tokens * (d + 2 * kv);
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.pack_qkv_grads_bf16)
                .arg(&grad_q)
                .arg(&grad_k)
                .arg(&grad_v)
                .arg(&combined)
                .arg(&(tokens as i32))
                .arg(&(d as i32))
                .arg(&(kv as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("pack_qkv_grads_bf16 failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn unpack_qkv_weight_grad(
        &self,
        combined_grad: CudaPtr,
        qo_grad: CudaPtr,
        kv_grad: CudaPtr,
        layer: u32,
        layers: u32,
        d: u32,
        kv: u32,
    ) -> PgResult<()> {
        let n = (d + 2 * kv) * d;
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.unpack_qkv_weight_grad)
                .arg(&combined_grad)
                .arg(&qo_grad)
                .arg(&kv_grad)
                .arg(&(layer as i32))
                .arg(&(layers as i32))
                .arg(&(d as i32))
                .arg(&(kv as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("unpack_qkv_weight_grad failed: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn f32_to_bf16(&self, src: CudaPtr, dst: CudaPtr, n: u32) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.f32_to_bf16)
                .arg(&src)
                .arg(&dst)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("f32_to_bf16 failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn bf16_to_f32(&self, src: CudaPtr, dst: CudaPtr, n: u32) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.bf16_to_f32)
                .arg(&src)
                .arg(&dst)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("bf16_to_f32 failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn normalize_matrices(
        &self,
        src: CudaPtr,
        dst: CudaPtr,
        matrix_numel: u32,
        batch: u32,
        eps: f32,
    ) -> PgResult<()> {
        unsafe {
            self.stream
                .launch_builder(&self.normalize_matrices)
                .arg(&src)
                .arg(&dst)
                .arg(&(matrix_numel as i32))
                .arg(&(batch as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (batch, 1, 1).into(),
                    block_dim: (256, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("normalize_matrices failed: {:?}", e)))?;
        }
        Ok(())
    }

    /// RMSNorm backward with accumulation into grad_input.
    pub fn rms_norm_backward_accum(
        &self,
        x: CudaPtr,
        grad_output: CudaPtr,
        grad_input: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
        beta: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_bwd_accum)
                .arg(&x)
                .arg(&grad_output)
                .arg(&grad_input)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&beta)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| PgError::InvalidOp(format!("rms_norm_bwd_accum launch: {:?}", e)))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_accum_residual_mix_input(
        &self,
        residual_x: CudaPtr,
        residual_x0: CudaPtr,
        mix: CudaPtr,
        grad_output: CudaPtr,
        grad_input: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
        beta: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_bwd_accum_residual_mix_input)
                .arg(&residual_x)
                .arg(&residual_x0)
                .arg(&mix)
                .arg(&grad_output)
                .arg(&grad_input)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&beta)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "rms_norm_bwd_accum_residual_mix_input launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_accum_residual_mix_input_go_bf16(
        &self,
        residual_x: CudaPtr,
        residual_x0: CudaPtr,
        mix: CudaPtr,
        grad_output_bf16: CudaPtr,
        grad_input: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
        beta: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_bwd_accum_residual_mix_input_go_bf16)
                .arg(&residual_x)
                .arg(&residual_x0)
                .arg(&mix)
                .arg(&grad_output_bf16)
                .arg(&grad_input)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&beta)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "rms_norm_bwd_accum_residual_mix_input_go_bf16 launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_accum_residual_mix_bwd(
        &self,
        x_norm: CudaPtr,
        grad_norm: CudaPtr,
        residual_x: CudaPtr,
        residual_x0: CudaPtr,
        mix: CudaPtr,
        base_grad: CudaPtr,
        grad_x: CudaPtr,
        grad_x0: CudaPtr,
        grad_mix: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
        beta: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_bwd_residual_mix)
                .arg(&x_norm)
                .arg(&grad_norm)
                .arg(&residual_x)
                .arg(&residual_x0)
                .arg(&mix)
                .arg(&base_grad)
                .arg(&grad_x)
                .arg(&grad_x0)
                .arg(&grad_mix)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&beta)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("rms_norm_bwd_residual_mix launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_accum_residual_mix_bwd_chunked(
        &self,
        x_norm: CudaPtr,
        grad_norm: CudaPtr,
        residual_x: CudaPtr,
        residual_x0: CudaPtr,
        mix: CudaPtr,
        base_grad: CudaPtr,
        grad_x: CudaPtr,
        grad_x0: CudaPtr,
        grad_mix_chunks: CudaPtr,
        grad_mix: CudaPtr,
        num_rows: u32,
        dim: u32,
        num_chunks: u32,
        chunk_rows: u32,
        ln_scale_factor: f32,
        eps: f32,
        beta: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_bwd_residual_mix_chunked)
                .arg(&x_norm)
                .arg(&grad_norm)
                .arg(&residual_x)
                .arg(&residual_x0)
                .arg(&mix)
                .arg(&base_grad)
                .arg(&grad_x)
                .arg(&grad_x0)
                .arg(&grad_mix_chunks)
                .arg(&(dim as i32))
                .arg(&(num_chunks as i32))
                .arg(&(chunk_rows as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&beta)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("rms_norm_bwd_residual_mix_chunked launch: {:?}", e))
                })?;

            self.stream
                .launch_builder(&self.residual_mix_grad_reduce_chunks)
                .arg(&grad_mix_chunks)
                .arg(&grad_mix)
                .arg(&((2 * dim) as i32))
                .arg(&(num_chunks as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (2 * dim, 1, 1).into(),
                    block_dim: (256, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("residual_mix_grad_reduce_chunks launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_accum_residual_mix_bwd_split_reduce(
        &self,
        x_norm: CudaPtr,
        grad_norm: CudaPtr,
        residual_x: CudaPtr,
        residual_x0: CudaPtr,
        mix: CudaPtr,
        base_grad: CudaPtr,
        grad_x: CudaPtr,
        grad_x0: CudaPtr,
        row_norm_stats: CudaPtr,
        grad_mix: CudaPtr,
        num_rows: u32,
        dim: u32,
        chunk_rows: u32,
        ln_scale_factor: f32,
        eps: f32,
        beta: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        let num_chunks = num_rows.div_ceil(chunk_rows);
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_bwd_residual_mix_no_mix_grad)
                .arg(&x_norm)
                .arg(&grad_norm)
                .arg(&residual_x)
                .arg(&residual_x0)
                .arg(&mix)
                .arg(&base_grad)
                .arg(&grad_x)
                .arg(&grad_x0)
                .arg(&row_norm_stats)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&beta)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "rms_norm_bwd_residual_mix_no_mix_grad launch: {:?}",
                        e
                    ))
                })?;

            self.stream
                .launch_builder(&self.residual_mix_grad_reduce_from_stats)
                .arg(&x_norm)
                .arg(&grad_norm)
                .arg(&residual_x)
                .arg(&residual_x0)
                .arg(&base_grad)
                .arg(&row_norm_stats)
                .arg(&grad_mix)
                .arg(&(dim as i32))
                .arg(&(num_rows as i32))
                .arg(&(chunk_rows as i32))
                .arg(&beta)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (2 * dim, num_chunks, 1).into(),
                    block_dim: (256, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "residual_mix_grad_reduce_from_stats launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_accum_residual_mix_bwd_recompute(
        &self,
        grad_norm: CudaPtr,
        residual_x: CudaPtr,
        residual_x0: CudaPtr,
        mix: CudaPtr,
        base_grad: CudaPtr,
        grad_x: CudaPtr,
        grad_x0: CudaPtr,
        grad_mix: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
        beta: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm_bwd_residual_mix_recompute)
                .arg(&grad_norm)
                .arg(&residual_x)
                .arg(&residual_x0)
                .arg(&mix)
                .arg(&base_grad)
                .arg(&grad_x)
                .arg(&grad_x0)
                .arg(&grad_mix)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&beta)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "rms_norm_bwd_residual_mix_recompute launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    pub fn decay_sgd_step(
        &self,
        param: CudaPtr,
        grad: CudaPtr,
        lr: f32,
        weight_decay: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.decay_sgd_step)
                .arg(&param)
                .arg(&grad)
                .arg(&lr)
                .arg(&weight_decay)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("decay_sgd_step failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn adamw_step(
        &self,
        param: CudaPtr,
        grad: CudaPtr,
        m: CudaPtr,
        v: CudaPtr,
        lr: f32,
        beta1: f32,
        beta2: f32,
        bc1: f32,
        bc2: f32,
        eps: f32,
        weight_decay: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.adamw_step)
                .arg(&param)
                .arg(&grad)
                .arg(&m)
                .arg(&v)
                .arg(&lr)
                .arg(&beta1)
                .arg(&beta2)
                .arg(&bc1)
                .arg(&bc2)
                .arg(&eps)
                .arg(&weight_decay)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("adamw_step failed: {:?}", e)))?;
        }
        Ok(())
    }

    /// LeakyReLU² forward: y = leaky_relu(x, 0.5)²
    pub fn leaky_relu_sq_forward(&self, x: CudaPtr, y: CudaPtr, n: u32) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.leaky_relu_sq_fwd)
                .arg(&x)
                .arg(&y)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("leaky_relu_sq launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn leaky_relu_sq_forward_bf16(
        &self,
        x: CudaPtr,
        y: CudaPtr,
        y_bf16: CudaPtr,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.leaky_relu_sq_fwd_bf16)
                .arg(&x)
                .arg(&y)
                .arg(&y_bf16)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("leaky_relu_sq_bf16 launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn leaky_relu_sq_forward_x_bf16_only(
        &self,
        x_bf16: CudaPtr,
        y_bf16: CudaPtr,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.leaky_relu_sq_fwd_x_bf16_only)
                .arg(&x_bf16)
                .arg(&y_bf16)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("leaky_relu_sq_fwd_x_bf16_only launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// LeakyReLU² backward.
    pub fn leaky_relu_sq_backward(
        &self,
        x: CudaPtr,
        grad_output: CudaPtr,
        grad_input: CudaPtr,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.leaky_relu_sq_bwd)
                .arg(&x)
                .arg(&grad_output)
                .arg(&grad_input)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("leaky_relu_sq_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn leaky_relu_sq_backward_bf16(
        &self,
        x: CudaPtr,
        grad_output: CudaPtr,
        grad_input: CudaPtr,
        grad_input_bf16: CudaPtr,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.leaky_relu_sq_bwd_bf16)
                .arg(&x)
                .arg(&grad_output)
                .arg(&grad_input)
                .arg(&grad_input_bf16)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("leaky_relu_sq_bwd_bf16 launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn leaky_relu_sq_backward_x_bf16(
        &self,
        x_bf16: CudaPtr,
        grad_output: CudaPtr,
        grad_input: CudaPtr,
        grad_input_bf16: CudaPtr,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.leaky_relu_sq_bwd_x_bf16)
                .arg(&x_bf16)
                .arg(&grad_output)
                .arg(&grad_input)
                .arg(&grad_input_bf16)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("leaky_relu_sq_bwd_x_bf16 launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn leaky_relu_sq_backward_x_bf16_only(
        &self,
        x_bf16: CudaPtr,
        grad_output: CudaPtr,
        grad_input_bf16: CudaPtr,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.leaky_relu_sq_bwd_x_bf16_only)
                .arg(&x_bf16)
                .arg(&grad_output)
                .arg(&grad_input_bf16)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("leaky_relu_sq_bwd_x_bf16_only launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Residual mixing: out = mix[0,:] * x + mix[1,:] * x0
    pub fn residual_mix_fwd(
        &self,
        x: CudaPtr,
        x0: CudaPtr,
        mix: CudaPtr,
        out: CudaPtr,
        dim: u32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.residual_mix)
                .arg(&x)
                .arg(&x0)
                .arg(&mix)
                .arg(&out)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("residual_mix launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Fused residual mixing and RMSNorm.
    pub fn residual_mix_rms_norm_fwd(
        &self,
        x: CudaPtr,
        x0: CudaPtr,
        mix: CudaPtr,
        mixed: CudaPtr,
        norm: CudaPtr,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let rows = n / dim.max(1);
        unsafe {
            self.stream
                .launch_builder(&self.residual_mix_rms_norm_fwd)
                .arg(&x)
                .arg(&x0)
                .arg(&mix)
                .arg(&mixed)
                .arg(&norm)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 128,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("residual_mix_rms_norm launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Fused residual mixing and RMSNorm with BF16 norm side output.
    pub fn residual_mix_rms_norm_fwd_bf16(
        &self,
        x: CudaPtr,
        x0: CudaPtr,
        mix: CudaPtr,
        mixed: CudaPtr,
        norm: CudaPtr,
        norm_bf16: CudaPtr,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let rows = n / dim.max(1);
        unsafe {
            self.stream
                .launch_builder(&self.residual_mix_rms_norm_fwd_bf16)
                .arg(&x)
                .arg(&x0)
                .arg(&mix)
                .arg(&mixed)
                .arg(&norm)
                .arg(&norm_bf16)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 128,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("residual_mix_rms_norm_bf16 launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Residual mixing backward.
    pub fn residual_mix_bwd(
        &self,
        x: CudaPtr,
        x0: CudaPtr,
        grad_output: CudaPtr,
        mix: CudaPtr,
        grad_x: CudaPtr,
        grad_x0: CudaPtr,
        grad_mix: CudaPtr,
        dim: u32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.residual_mix_bwd)
                .arg(&x)
                .arg(&x0)
                .arg(&grad_output)
                .arg(&mix)
                .arg(&grad_x)
                .arg(&grad_x0)
                .arg(&grad_mix)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("residual_mix_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Residual add with scale: x += scale * proj (in-place)
    pub fn residual_add_scale_fwd(
        &self,
        x: CudaPtr,
        proj: CudaPtr,
        scale: CudaPtr,
        dim: u32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.residual_add_scale)
                .arg(&x)
                .arg(&proj)
                .arg(&scale)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("residual_add_scale launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn residual_add_scale_bf16_proj_fwd(
        &self,
        x: CudaPtr,
        proj_bf16: CudaPtr,
        scale: CudaPtr,
        dim: u32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.residual_add_scale_bf16_proj)
                .arg(&x)
                .arg(&proj_bf16)
                .arg(&scale)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("residual_add_scale_bf16_proj launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn residual_add_scale_from_base_fwd(
        &self,
        base: CudaPtr,
        proj: CudaPtr,
        scale: CudaPtr,
        out: CudaPtr,
        dim: u32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.residual_add_scale_from_base)
                .arg(&base)
                .arg(&proj)
                .arg(&scale)
                .arg(&out)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("residual_add_scale_from_base launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn residual_add_scale_from_base_bf16_proj_fwd(
        &self,
        base: CudaPtr,
        proj_bf16: CudaPtr,
        scale: CudaPtr,
        out: CudaPtr,
        dim: u32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.residual_add_scale_from_base_bf16_proj)
                .arg(&base)
                .arg(&proj_bf16)
                .arg(&scale)
                .arg(&out)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "residual_add_scale_from_base_bf16_proj launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn residual_add_scale_from_base_rms_norm_fwd(
        &self,
        base: CudaPtr,
        proj: CudaPtr,
        scale: CudaPtr,
        out: CudaPtr,
        norm: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.residual_add_scale_from_base_rms_norm)
                .arg(&base)
                .arg(&proj)
                .arg(&scale)
                .arg(&out)
                .arg(&norm)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 128,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "residual_add_scale_from_base_rms_norm launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn residual_add_scale_from_base_rms_norm_fwd_bf16(
        &self,
        base: CudaPtr,
        proj: CudaPtr,
        scale: CudaPtr,
        out: CudaPtr,
        norm: CudaPtr,
        norm_bf16: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.residual_add_scale_from_base_rms_norm_bf16)
                .arg(&base)
                .arg(&proj)
                .arg(&scale)
                .arg(&out)
                .arg(&norm)
                .arg(&norm_bf16)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 128,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "residual_add_scale_from_base_rms_norm_bf16 launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn residual_add_scale_from_base_rms_norm_fwd_bf16_proj(
        &self,
        base: CudaPtr,
        proj_bf16: CudaPtr,
        scale: CudaPtr,
        out: CudaPtr,
        norm: CudaPtr,
        norm_bf16: CudaPtr,
        num_rows: u32,
        dim: u32,
        ln_scale_factor: f32,
        eps: f32,
    ) -> PgResult<()> {
        let n = num_rows * dim;
        let block = 32u32.max(256u32.min(dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.residual_add_scale_from_base_rms_norm_bf16_proj)
                .arg(&base)
                .arg(&proj_bf16)
                .arg(&scale)
                .arg(&out)
                .arg(&norm)
                .arg(&norm_bf16)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_rows, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 128,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "residual_add_scale_from_base_rms_norm_bf16_proj launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    /// Residual add-scale backward.
    pub fn residual_add_scale_bwd(
        &self,
        proj: CudaPtr,
        grad_output: CudaPtr,
        scale: CudaPtr,
        grad_x_in: CudaPtr,
        grad_proj: CudaPtr,
        grad_scale: CudaPtr,
        dim: u32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.residual_add_scale_bwd)
                .arg(&proj)
                .arg(&grad_output)
                .arg(&scale)
                .arg(&grad_x_in)
                .arg(&grad_proj)
                .arg(&grad_scale)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("residual_add_scale_bwd launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn residual_add_scale_bwd_bf16(
        &self,
        proj: CudaPtr,
        grad_output: CudaPtr,
        scale: CudaPtr,
        grad_x_in: CudaPtr,
        grad_proj: CudaPtr,
        grad_proj_bf16: CudaPtr,
        grad_scale: CudaPtr,
        dim: u32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.residual_add_scale_bwd_bf16)
                .arg(&proj)
                .arg(&grad_output)
                .arg(&scale)
                .arg(&grad_x_in)
                .arg(&grad_proj)
                .arg(&grad_proj_bf16)
                .arg(&grad_scale)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("residual_add_scale_bwd_bf16 launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn residual_add_scale_bwd_bf16_only(
        &self,
        proj: CudaPtr,
        grad_output: CudaPtr,
        scale: CudaPtr,
        grad_x_in: CudaPtr,
        grad_proj_bf16: CudaPtr,
        grad_scale: CudaPtr,
        dim: u32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            if !residual_scale_reduce_enabled() {
                self.stream
                    .launch_builder(&self.residual_add_scale_bwd_bf16_only_atomic)
                    .arg(&proj)
                    .arg(&grad_output)
                    .arg(&scale)
                    .arg(&grad_x_in)
                    .arg(&grad_proj_bf16)
                    .arg(&grad_scale)
                    .arg(&(dim as i32))
                    .arg(&(n as i32))
                    .launch(cudarc::driver::LaunchConfig {
                        grid_dim: (grid, 1, 1).into(),
                        block_dim: (block, 1, 1).into(),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| {
                        PgError::InvalidOp(format!(
                            "residual_add_scale_bwd_bf16_only launch: {:?}",
                            e
                        ))
                    })?;
                return Ok(());
            }
            self.stream
                .launch_builder(&self.residual_add_scale_bwd_bf16_only_no_atomic)
                .arg(&grad_output)
                .arg(&scale)
                .arg(&grad_x_in)
                .arg(&grad_proj_bf16)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("residual_add_scale_bwd_bf16_only launch: {:?}", e))
                })?;
            self.stream
                .launch_builder(&self.residual_add_scale_grad_scale_reduce)
                .arg(&proj)
                .arg(&grad_output)
                .arg(&grad_scale)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (dim, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "residual_add_scale_grad_scale_reduce launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    pub fn residual_add_scale_bwd_from_bf16_only(
        &self,
        proj_bf16: CudaPtr,
        grad_output: CudaPtr,
        scale: CudaPtr,
        grad_x_in: CudaPtr,
        grad_proj_bf16: CudaPtr,
        grad_scale: CudaPtr,
        dim: u32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            if !residual_scale_reduce_enabled() {
                self.stream
                    .launch_builder(&self.residual_add_scale_bwd_from_bf16_only)
                    .arg(&proj_bf16)
                    .arg(&grad_output)
                    .arg(&scale)
                    .arg(&grad_x_in)
                    .arg(&grad_proj_bf16)
                    .arg(&grad_scale)
                    .arg(&(dim as i32))
                    .arg(&(n as i32))
                    .launch(cudarc::driver::LaunchConfig {
                        grid_dim: (grid, 1, 1).into(),
                        block_dim: (block, 1, 1).into(),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| {
                        PgError::InvalidOp(format!(
                            "residual_add_scale_bwd_from_bf16_only launch: {:?}",
                            e
                        ))
                    })?;
                return Ok(());
            }
            self.stream
                .launch_builder(&self.residual_add_scale_bwd_bf16_only_no_atomic)
                .arg(&grad_output)
                .arg(&scale)
                .arg(&grad_x_in)
                .arg(&grad_proj_bf16)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "residual_add_scale_bwd_from_bf16_only no_atomic launch: {:?}",
                        e
                    ))
                })?;
            self.stream
                .launch_builder(&self.residual_add_scale_grad_scale_reduce_bf16)
                .arg(&proj_bf16)
                .arg(&grad_output)
                .arg(&grad_scale)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (dim, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "residual_add_scale_grad_scale_reduce_bf16 launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    /// SmearGate forward
    pub fn smear_gate_fwd(
        &self,
        x: CudaPtr,
        gate: CudaPtr,
        out: CudaPtr,
        tokens: u32,
        seq_len: u32,
        dim: u32,
    ) -> PgResult<()> {
        let n = tokens * dim;
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.smear_gate_fwd)
                .arg(&x)
                .arg(&gate)
                .arg(&out)
                .arg(&(tokens as i32))
                .arg(&(seq_len as i32))
                .arg(&(dim as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("smear_gate launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Boundary-aware SmearGate forward. Masks previous-token mixing when the
    /// current token is BOS/boundary, preventing packed-document leakage.
    pub fn smear_gate_fwd_boundary(
        &self,
        x: CudaPtr,
        input_ids: CudaPtr,
        gate: CudaPtr,
        out: CudaPtr,
        tokens: u32,
        seq_len: u32,
        dim: u32,
        boundary_token_id: u32,
    ) -> PgResult<()> {
        let n = tokens * dim;
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.smear_gate_fwd_boundary)
                .arg(&x)
                .arg(&input_ids)
                .arg(&gate)
                .arg(&out)
                .arg(&(tokens as i32))
                .arg(&(seq_len as i32))
                .arg(&(dim as i32))
                .arg(&boundary_token_id)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("smear_gate_boundary launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// SmearGate backward.
    pub fn smear_gate_bwd(
        &self,
        x: CudaPtr,
        gate: CudaPtr,
        grad_output: CudaPtr,
        grad_x: CudaPtr,
        grad_x_prev: CudaPtr,
        grad_gate: CudaPtr,
        tokens: u32,
        seq_len: u32,
        dim: u32,
    ) -> PgResult<()> {
        let n = tokens * dim;
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.smear_gate_bwd)
                .arg(&x)
                .arg(&gate)
                .arg(&grad_output)
                .arg(&grad_x)
                .arg(&grad_x_prev)
                .arg(&grad_gate)
                .arg(&(tokens as i32))
                .arg(&(seq_len as i32))
                .arg(&(dim as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("smear_gate_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Boundary-aware SmearGate backward matching `smear_gate_fwd_boundary`.
    pub fn smear_gate_bwd_boundary(
        &self,
        x: CudaPtr,
        input_ids: CudaPtr,
        gate: CudaPtr,
        grad_output: CudaPtr,
        grad_x: CudaPtr,
        grad_x_prev: CudaPtr,
        grad_gate: CudaPtr,
        tokens: u32,
        seq_len: u32,
        dim: u32,
        boundary_token_id: u32,
    ) -> PgResult<()> {
        let n = tokens * dim;
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.smear_gate_bwd_boundary)
                .arg(&x)
                .arg(&input_ids)
                .arg(&gate)
                .arg(&grad_output)
                .arg(&grad_x)
                .arg(&grad_x_prev)
                .arg(&grad_gate)
                .arg(&(tokens as i32))
                .arg(&(seq_len as i32))
                .arg(&(dim as i32))
                .arg(&boundary_token_id)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("smear_gate_bwd_boundary launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Embedding gather: out[t, :] = emb[ids[t], :]
    pub fn embedding_gather_fwd(
        &self,
        ids: CudaPtr,
        emb: CudaPtr,
        out: CudaPtr,
        dim: u32,
        tokens: u32,
    ) -> PgResult<()> {
        let block = 256u32.min(dim.next_power_of_two());
        unsafe {
            self.stream
                .launch_builder(&self.embedding_gather)
                .arg(&ids)
                .arg(&emb)
                .arg(&out)
                .arg(&(dim as i32))
                .arg(&(tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("embedding_gather launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Embedding gather backward.
    pub fn embedding_gather_bwd(
        &self,
        ids: CudaPtr,
        grad_output: CudaPtr,
        grad_emb: CudaPtr,
        dim: u32,
        tokens: u32,
    ) -> PgResult<()> {
        let block = 256u32.min(dim.next_power_of_two());
        unsafe {
            self.stream
                .launch_builder(&self.embedding_gather_bwd)
                .arg(&ids)
                .arg(&grad_output)
                .arg(&grad_emb)
                .arg(&(dim as i32))
                .arg(&(tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("embedding_gather_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// QK-norm: in-place per-head RMSNorm
    pub fn qk_norm_fwd(
        &self,
        qk: CudaPtr,
        head_dim: u32,
        total_heads: u32,
        eps: f32,
    ) -> PgResult<()> {
        // Cap to 32 threads: the reduction uses a single warp shuffle pass
        // with no cross-warp shared memory step. block > 32 would silently
        // drop partial sums from the second warp.
        let block = 32u32;
        unsafe {
            self.stream
                .launch_builder(&self.qk_norm_fwd)
                .arg(&qk)
                .arg(&(head_dim as i32))
                .arg(&(total_heads as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 4,
                })
                .map_err(|e| PgError::InvalidOp(format!("qk_norm launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// QK-norm backward.
    pub fn qk_norm_bwd(
        &self,
        x: CudaPtr,
        grad_output: CudaPtr,
        grad_input: CudaPtr,
        head_dim: u32,
        total_heads: u32,
        eps: f32,
    ) -> PgResult<()> {
        let block = 32u32;
        unsafe {
            self.stream
                .launch_builder(&self.qk_norm_bwd)
                .arg(&x)
                .arg(&grad_output)
                .arg(&grad_input)
                .arg(&(head_dim as i32))
                .arg(&(total_heads as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 8,
                })
                .map_err(|e| PgError::InvalidOp(format!("qk_norm_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Partial RoPE: in-place rotate first rope_dims of each head
    pub fn partial_rope_fwd(
        &self,
        x: CudaPtr,
        cos_table: CudaPtr,
        sin_table: CudaPtr,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        rope_dims: u32,
        total_heads: u32,
    ) -> PgResult<()> {
        let half = rope_dims / 2;
        let block = 32u32.max(half.next_power_of_two().min(64));
        unsafe {
            self.stream
                .launch_builder(&self.partial_rope_fwd)
                .arg(&x)
                .arg(&cos_table)
                .arg(&sin_table)
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(rope_dims as i32))
                .arg(&(total_heads as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("partial_rope launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Partial RoPE backward in-place.
    pub fn partial_rope_bwd(
        &self,
        grad: CudaPtr,
        cos_table: CudaPtr,
        sin_table: CudaPtr,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        rope_dims: u32,
        total_heads: u32,
    ) -> PgResult<()> {
        let half = rope_dims / 2;
        let block = 32u32.max(half.next_power_of_two().min(64));
        unsafe {
            self.stream
                .launch_builder(&self.partial_rope_bwd)
                .arg(&grad)
                .arg(&cos_table)
                .arg(&sin_table)
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(rope_dims as i32))
                .arg(&(total_heads as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("partial_rope_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Q-gain: Q *= gain[head] per head (in-place)
    pub fn q_gain_fwd(
        &self,
        q: CudaPtr,
        gain: CudaPtr,
        num_heads: u32,
        head_dim: u32,
        total_heads: u32,
    ) -> PgResult<()> {
        let block = 32u32.max(64u32.min(head_dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.q_gain_fwd)
                .arg(&q)
                .arg(&gain)
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(total_heads as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("q_gain launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Q-gain backward.
    pub fn q_gain_bwd(
        &self,
        q_pre_gain: CudaPtr,
        grad_q_scaled: CudaPtr,
        gain: CudaPtr,
        grad_q_pre_gain: CudaPtr,
        grad_gain: CudaPtr,
        num_heads: u32,
        head_dim: u32,
        total_heads: u32,
    ) -> PgResult<()> {
        let block = 32u32.max(64u32.min(head_dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.q_gain_bwd)
                .arg(&q_pre_gain)
                .arg(&grad_q_scaled)
                .arg(&gain)
                .arg(&grad_q_pre_gain)
                .arg(&grad_gain)
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(total_heads as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 128,
                })
                .map_err(|e| PgError::InvalidOp(format!("q_gain_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn q_gain_rope_qk_norm_fwd(
        &self,
        q: CudaPtr,
        q_post_rope: Option<CudaPtr>,
        gain: CudaPtr,
        cos_table: CudaPtr,
        sin_table: CudaPtr,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        rope_dims: u32,
        total_heads: u32,
        eps: f32,
    ) -> PgResult<()> {
        let block = 32u32;
        let q_post_rope = q_post_rope.unwrap_or(CudaPtr(0));
        unsafe {
            self.stream
                .launch_builder(&self.q_gain_rope_qk_norm_fwd)
                .arg(&q)
                .arg(&q_post_rope)
                .arg(&gain)
                .arg(&cos_table)
                .arg(&sin_table)
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(rope_dims as i32))
                .arg(&(total_heads as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 4,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("q_gain_rope_qk_norm_fwd launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn q_gain_rope_qk_norm_fwd_bf16_bhsd(
        &self,
        q: CudaPtr,
        q_post_rope: Option<CudaPtr>,
        q_bhsd_bf16: CudaPtr,
        gain: CudaPtr,
        cos_table: CudaPtr,
        sin_table: CudaPtr,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        rope_dims: u32,
        total_heads: u32,
        eps: f32,
    ) -> PgResult<()> {
        let block = 32u32;
        let q_post_rope = q_post_rope.unwrap_or(CudaPtr(0));
        unsafe {
            self.stream
                .launch_builder(&self.q_gain_rope_qk_norm_fwd_bf16_bhsd)
                .arg(&q)
                .arg(&q_post_rope)
                .arg(&q_bhsd_bf16)
                .arg(&gain)
                .arg(&cos_table)
                .arg(&sin_table)
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(rope_dims as i32))
                .arg(&(total_heads as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 4,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("q_gain_rope_qk_norm_fwd_bf16_bhsd launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn rope_qk_norm_fwd(
        &self,
        k: CudaPtr,
        cos_table: CudaPtr,
        sin_table: CudaPtr,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        rope_dims: u32,
        total_heads: u32,
        eps: f32,
    ) -> PgResult<()> {
        let block = 32u32;
        unsafe {
            self.stream
                .launch_builder(&self.rope_qk_norm_fwd)
                .arg(&k)
                .arg(&cos_table)
                .arg(&sin_table)
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(rope_dims as i32))
                .arg(&(total_heads as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 4,
                })
                .map_err(|e| PgError::InvalidOp(format!("rope_qk_norm_fwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk_norm_fwd_bf16_bhsd(
        &self,
        k: CudaPtr,
        k_bhsd_bf16: CudaPtr,
        cos_table: CudaPtr,
        sin_table: CudaPtr,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        rope_dims: u32,
        total_heads: u32,
        eps: f32,
    ) -> PgResult<()> {
        let block = 32u32;
        unsafe {
            self.stream
                .launch_builder(&self.rope_qk_norm_fwd_bf16_bhsd)
                .arg(&k)
                .arg(&k_bhsd_bf16)
                .arg(&cos_table)
                .arg(&sin_table)
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(rope_dims as i32))
                .arg(&(total_heads as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 4,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("rope_qk_norm_fwd_bf16_bhsd launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn bthd_to_bhsd_bf16(
        &self,
        input: CudaPtr,
        output: CudaPtr,
        batch: u32,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> PgResult<()> {
        let total = batch
            .checked_mul(seq_len)
            .and_then(|v| v.checked_mul(num_heads))
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| PgError::InvalidOp("bthd_to_bhsd_bf16 size overflow".into()))?;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.bthd_to_bhsd_bf16)
                .arg(&input)
                .arg(&output)
                .arg(&(batch as i32))
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(total as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("bthd_to_bhsd_bf16 launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn bhsd_to_bthd_bf16(
        &self,
        input: CudaPtr,
        output: CudaPtr,
        batch: u32,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> PgResult<()> {
        let total = batch
            .checked_mul(seq_len)
            .and_then(|v| v.checked_mul(num_heads))
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| PgError::InvalidOp("bhsd_to_bthd_bf16 size overflow".into()))?;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.bhsd_to_bthd_bf16)
                .arg(&input)
                .arg(&output)
                .arg(&(batch as i32))
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(total as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("bhsd_to_bthd_bf16 launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn q_gain_rope_qk_norm_bwd(
        &self,
        q_pre_norm: CudaPtr,
        q_post_rope: CudaPtr,
        grad_q_scaled: CudaPtr,
        gain: CudaPtr,
        cos_table: CudaPtr,
        sin_table: CudaPtr,
        grad_q_proj: CudaPtr,
        grad_gain: CudaPtr,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        rope_dims: u32,
        total_heads: u32,
        eps: f32,
    ) -> PgResult<()> {
        let block = 32u32;
        unsafe {
            self.stream
                .launch_builder(&self.q_gain_rope_qk_norm_bwd)
                .arg(&q_pre_norm)
                .arg(&q_post_rope)
                .arg(&grad_q_scaled)
                .arg(&gain)
                .arg(&cos_table)
                .arg(&sin_table)
                .arg(&grad_q_proj)
                .arg(&grad_gain)
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(rope_dims as i32))
                .arg(&(total_heads as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 8,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("q_gain_rope_qk_norm_bwd launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn q_gain_rope_qk_norm_bwd_chunked(
        &self,
        q_pre_norm: CudaPtr,
        q_post_rope: CudaPtr,
        grad_q_scaled: CudaPtr,
        gain: CudaPtr,
        cos_table: CudaPtr,
        sin_table: CudaPtr,
        grad_q_proj: CudaPtr,
        grad_gain_chunks: CudaPtr,
        grad_gain: CudaPtr,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        rope_dims: u32,
        total_heads: u32,
        q_gain_chunk_tokens: u32,
        q_gain_chunks: u32,
        eps: f32,
    ) -> PgResult<()> {
        let block = 32u32;
        unsafe {
            self.stream
                .launch_builder(&self.q_gain_rope_qk_norm_bwd_chunked)
                .arg(&q_pre_norm)
                .arg(&q_post_rope)
                .arg(&grad_q_scaled)
                .arg(&gain)
                .arg(&cos_table)
                .arg(&sin_table)
                .arg(&grad_q_proj)
                .arg(&grad_gain_chunks)
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(rope_dims as i32))
                .arg(&(total_heads as i32))
                .arg(&(q_gain_chunk_tokens as i32))
                .arg(&(q_gain_chunks as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 8,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("q_gain_rope_qk_norm_bwd_chunked launch: {:?}", e))
                })?;
            self.stream
                .launch_builder(&self.q_gain_reduce_chunked)
                .arg(&grad_gain_chunks)
                .arg(&grad_gain)
                .arg(&(q_gain_chunks as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_heads, 1, 1).into(),
                    block_dim: (256, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("q_gain_reduce_chunked launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn rope_qk_norm_bwd(
        &self,
        k_pre_norm: CudaPtr,
        grad_k_attn: CudaPtr,
        cos_table: CudaPtr,
        sin_table: CudaPtr,
        grad_k_proj: CudaPtr,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        rope_dims: u32,
        total_heads: u32,
        eps: f32,
    ) -> PgResult<()> {
        let block = 32u32;
        unsafe {
            self.stream
                .launch_builder(&self.rope_qk_norm_bwd)
                .arg(&k_pre_norm)
                .arg(&grad_k_attn)
                .arg(&cos_table)
                .arg(&sin_table)
                .arg(&grad_k_proj)
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(rope_dims as i32))
                .arg(&(total_heads as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 8,
                })
                .map_err(|e| PgError::InvalidOp(format!("rope_qk_norm_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// out[0] += alpha * dot(a, b)
    pub fn dot_accumulate(
        &self,
        a: CudaPtr,
        b: CudaPtr,
        out: CudaPtr,
        alpha: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = ((n + block - 1) / block).min(256);
        unsafe {
            self.stream
                .launch_builder(&self.dot_accumulate)
                .arg(&a)
                .arg(&b)
                .arg(&out)
                .arg(&alpha)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 128,
                })
                .map_err(|e| PgError::InvalidOp(format!("dot_accumulate launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// out[0] += alpha * scale[scale_index] * dot(a, b)
    pub fn dot_accumulate_by_param(
        &self,
        a: CudaPtr,
        b: CudaPtr,
        out: CudaPtr,
        scale: CudaPtr,
        scale_index: u32,
        alpha: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = ((n + block - 1) / block).min(256);
        unsafe {
            self.stream
                .launch_builder(&self.dot_accumulate_by_param)
                .arg(&a)
                .arg(&b)
                .arg(&out)
                .arg(&scale)
                .arg(&(scale_index as i32))
                .arg(&alpha)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 128,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("dot_accumulate_by_param launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn clip_by_global_norm(
        &self,
        x: CudaPtr,
        sum_sq: CudaPtr,
        max_norm: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.clip_by_global_norm)
                .arg(&x)
                .arg(&sum_sq)
                .arg(&max_norm)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("clip_by_global_norm launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Cross-entropy forward with softcap
    pub fn cross_entropy_fwd(
        &self,
        logits: CudaPtr,
        targets: CudaPtr,
        losses: CudaPtr,
        vocab_size: u32,
        softcap: f32,
        num_tokens: u32,
    ) -> PgResult<()> {
        let block = 32u32.max(256u32.min(vocab_size.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.cross_entropy_fwd)
                .arg(&logits)
                .arg(&targets)
                .arg(&losses)
                .arg(&(vocab_size as i32))
                .arg(&softcap)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| PgError::InvalidOp(format!("cross_entropy launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Cross-entropy forward with BF16 logits and F32 loss output.
    pub fn cross_entropy_fwd_bf16_logits(
        &self,
        logits: CudaPtr,
        targets: CudaPtr,
        losses: CudaPtr,
        vocab_size: u32,
        softcap: f32,
        num_tokens: u32,
    ) -> PgResult<()> {
        let block = 32u32.max(256u32.min(vocab_size.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.cross_entropy_fwd_bf16_logits)
                .arg(&logits)
                .arg(&targets)
                .arg(&losses)
                .arg(&(vocab_size as i32))
                .arg(&softcap)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("cross_entropy_fwd_bf16_logits launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Cross-entropy backward with softcap.
    pub fn cross_entropy_bwd(
        &self,
        logits: CudaPtr,
        targets: CudaPtr,
        grad_logits: CudaPtr,
        vocab_size: u32,
        softcap: f32,
        grad_loss: f32,
        num_tokens: u32,
    ) -> PgResult<()> {
        let block = 32u32.max(256u32.min(vocab_size.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.cross_entropy_bwd)
                .arg(&logits)
                .arg(&targets)
                .arg(&grad_logits)
                .arg(&(vocab_size as i32))
                .arg(&softcap)
                .arg(&grad_loss)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| PgError::InvalidOp(format!("cross_entropy_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Cross-entropy backward with softcap, writing BF16 grad logits directly.
    pub fn cross_entropy_bwd_bf16(
        &self,
        logits: CudaPtr,
        targets: CudaPtr,
        grad_logits: CudaPtr,
        vocab_size: u32,
        softcap: f32,
        grad_loss: f32,
        num_tokens: u32,
    ) -> PgResult<()> {
        let block = 32u32.max(256u32.min(vocab_size.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.cross_entropy_bwd_bf16)
                .arg(&logits)
                .arg(&targets)
                .arg(&grad_logits)
                .arg(&(vocab_size as i32))
                .arg(&softcap)
                .arg(&grad_loss)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("cross_entropy_bwd_bf16 launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Cross-entropy backward from BF16 logits, writing BF16 grad logits.
    pub fn cross_entropy_bwd_bf16_logits(
        &self,
        logits: CudaPtr,
        targets: CudaPtr,
        grad_logits: CudaPtr,
        vocab_size: u32,
        softcap: f32,
        grad_loss: f32,
        num_tokens: u32,
    ) -> PgResult<()> {
        let block = 32u32.max(256u32.min(vocab_size.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.cross_entropy_bwd_bf16_logits)
                .arg(&logits)
                .arg(&targets)
                .arg(&grad_logits)
                .arg(&(vocab_size as i32))
                .arg(&softcap)
                .arg(&grad_loss)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("cross_entropy_bwd_bf16_logits launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Fused softcapped cross-entropy loss + backward with BF16 grad logits.
    pub fn cross_entropy_loss_bwd_bf16(
        &self,
        logits: CudaPtr,
        targets: CudaPtr,
        losses: CudaPtr,
        grad_logits: CudaPtr,
        vocab_size: u32,
        softcap: f32,
        grad_loss: f32,
        num_tokens: u32,
    ) -> PgResult<()> {
        let block = 32u32.max(256u32.min(vocab_size.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.cross_entropy_loss_bwd_bf16)
                .arg(&logits)
                .arg(&targets)
                .arg(&losses)
                .arg(&grad_logits)
                .arg(&(vocab_size as i32))
                .arg(&softcap)
                .arg(&grad_loss)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("cross_entropy_loss_bwd_bf16 launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Fused softcapped CE loss + backward from BF16 logits.
    pub fn cross_entropy_loss_bwd_bf16_logits(
        &self,
        logits: CudaPtr,
        targets: CudaPtr,
        losses: CudaPtr,
        grad_logits: CudaPtr,
        vocab_size: u32,
        softcap: f32,
        grad_loss: f32,
        num_tokens: u32,
    ) -> PgResult<()> {
        let block = 32u32.max(256u32.min(vocab_size.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.cross_entropy_loss_bwd_bf16_logits)
                .arg(&logits)
                .arg(&targets)
                .arg(&losses)
                .arg(&grad_logits)
                .arg(&(vocab_size as i32))
                .arg(&softcap)
                .arg(&grad_loss)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "cross_entropy_loss_bwd_bf16_logits launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    pub fn output_ce_stats_init(
        &self,
        row_max: CudaPtr,
        row_sum: CudaPtr,
        target_logit: CudaPtr,
        num_tokens: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (num_tokens + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.output_ce_stats_init)
                .arg(&row_max)
                .arg(&row_sum)
                .arg(&target_logit)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("output_ce_stats_init launch: {:?}", e)))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn output_ce_tile_stats_update(
        &self,
        logits_tile: CudaPtr,
        targets: CudaPtr,
        row_max: CudaPtr,
        row_sum: CudaPtr,
        target_logit: CudaPtr,
        vocab_start: u32,
        tile_vocab: u32,
        tile_stride: u32,
        softcap: f32,
        num_tokens: u32,
    ) -> PgResult<()> {
        let block = 32u32.max(256u32.min(tile_vocab.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.output_ce_tile_stats_update)
                .arg(&logits_tile)
                .arg(&targets)
                .arg(&row_max)
                .arg(&row_sum)
                .arg(&target_logit)
                .arg(&(vocab_start as i32))
                .arg(&(tile_vocab as i32))
                .arg(&(tile_stride as i32))
                .arg(&softcap)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("output_ce_tile_stats_update launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn output_ce_finalize_loss(
        &self,
        row_max: CudaPtr,
        row_sum: CudaPtr,
        target_logit: CudaPtr,
        losses: CudaPtr,
        num_tokens: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (num_tokens + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.output_ce_finalize_loss)
                .arg(&row_max)
                .arg(&row_sum)
                .arg(&target_logit)
                .arg(&losses)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("output_ce_finalize_loss launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn loss_window_sum(
        &self,
        losses: CudaPtr,
        out: CudaPtr,
        start: u32,
        end: u32,
    ) -> PgResult<()> {
        if end < start {
            return Err(PgError::InvalidOp(format!(
                "loss_window_sum invalid range: start={start} end={end}"
            )));
        }
        unsafe {
            self.stream
                .launch_builder(&self.loss_window_sum)
                .arg(&losses)
                .arg(&out)
                .arg(&(start as i32))
                .arg(&(end as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1).into(),
                    block_dim: (256, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("loss_window_sum launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn loss_window_accumulate(
        &self,
        losses: CudaPtr,
        out: CudaPtr,
        start: u32,
        end: u32,
    ) -> PgResult<()> {
        if end < start {
            return Err(PgError::InvalidOp(format!(
                "loss_window_accumulate invalid range: start={start} end={end}"
            )));
        }
        unsafe {
            self.stream
                .launch_builder(&self.loss_window_accumulate)
                .arg(&losses)
                .arg(&out)
                .arg(&(start as i32))
                .arg(&(end as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1).into(),
                    block_dim: (256, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("loss_window_accumulate launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn output_ce_tile_grad_bf16(
        &self,
        logits_tile: CudaPtr,
        targets: CudaPtr,
        row_max: CudaPtr,
        row_sum: CudaPtr,
        grad_tile: CudaPtr,
        vocab_start: u32,
        tile_vocab: u32,
        tile_stride: u32,
        softcap: f32,
        grad_loss: f32,
        num_tokens: u32,
    ) -> PgResult<()> {
        let n = num_tokens * tile_vocab;
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.output_ce_tile_grad_bf16)
                .arg(&logits_tile)
                .arg(&targets)
                .arg(&row_max)
                .arg(&row_sum)
                .arg(&grad_tile)
                .arg(&(vocab_start as i32))
                .arg(&(tile_vocab as i32))
                .arg(&(tile_stride as i32))
                .arg(&softcap)
                .arg(&grad_loss)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("output_ce_tile_grad_bf16 launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Bigram hash embedding gather
    pub fn bigram_hash_embed_fwd(
        &self,
        ids: CudaPtr,
        embed: CudaPtr,
        out: CudaPtr,
        bigram_vocab: u32,
        bigram_dim: u32,
        tokens: u32,
        seq_len: u32,
    ) -> PgResult<()> {
        let block = 128u32.min(bigram_dim.next_power_of_two());
        unsafe {
            self.stream
                .launch_builder(&self.bigram_hash_embed)
                .arg(&ids)
                .arg(&embed)
                .arg(&out)
                .arg(&(bigram_vocab as i32))
                .arg(&(bigram_dim as i32))
                .arg(&(tokens as i32))
                .arg(&(seq_len as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("bigram_hash_embed launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Bigram hash embedding backward.
    pub fn bigram_hash_embed_bwd(
        &self,
        ids: CudaPtr,
        grad_output: CudaPtr,
        grad_embed: CudaPtr,
        bigram_vocab: u32,
        bigram_dim: u32,
        tokens: u32,
        seq_len: u32,
    ) -> PgResult<()> {
        let block = 128u32.min(bigram_dim.next_power_of_two());
        unsafe {
            self.stream
                .launch_builder(&self.bigram_hash_embed_bwd)
                .arg(&ids)
                .arg(&grad_output)
                .arg(&grad_embed)
                .arg(&(bigram_vocab as i32))
                .arg(&(bigram_dim as i32))
                .arg(&(tokens as i32))
                .arg(&(seq_len as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("bigram_hash_embed_bwd launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// x += alpha * y (in-place)
    pub fn add_scaled_fwd(&self, x: CudaPtr, y: CudaPtr, alpha: f32, n: u32) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.add_scaled)
                .arg(&x)
                .arg(&y)
                .arg(&alpha)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("add_scaled launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// x += alpha * scale[0] * y (in-place)
    pub fn add_scaled_by_param_fwd(
        &self,
        x: CudaPtr,
        y: CudaPtr,
        scale: CudaPtr,
        alpha: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.add_scaled_by_param)
                .arg(&x)
                .arg(&y)
                .arg(&scale)
                .arg(&alpha)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("add_scaled_by_param launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// x += alpha * scale[index] * y (in-place)
    pub fn add_scaled_by_param_index_fwd(
        &self,
        x: CudaPtr,
        y: CudaPtr,
        scale: CudaPtr,
        scale_index: u32,
        alpha: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.add_scaled_by_param_index)
                .arg(&x)
                .arg(&y)
                .arg(&scale)
                .arg(&(scale_index as i32))
                .arg(&alpha)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("add_scaled_by_param_index launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// x += alpha * scale_a[index_a] * scale_b[index_b] * y (in-place)
    pub fn add_scaled_by_param_product_fwd(
        &self,
        x: CudaPtr,
        y: CudaPtr,
        scale_a: CudaPtr,
        scale_a_index: u32,
        scale_b: CudaPtr,
        scale_b_index: u32,
        alpha: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream
                .launch_builder(&self.add_scaled_by_param_product)
                .arg(&x)
                .arg(&y)
                .arg(&scale_a)
                .arg(&(scale_a_index as i32))
                .arg(&scale_b)
                .arg(&(scale_b_index as i32))
                .arg(&alpha)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("add_scaled_by_param_product launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Causal attention forward (naive O(T²), for parity testing only)
    /// q: [tokens * num_heads, head_dim], k/v: [tokens * num_kv_heads, head_dim]
    /// out: [tokens * num_heads, head_dim]
    pub fn causal_attention_naive_fwd(
        &self,
        q: CudaPtr,
        k: CudaPtr,
        v: CudaPtr,
        out: CudaPtr,
        tokens: u32,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> PgResult<()> {
        let total_heads = tokens * num_heads;
        let block = head_dim.min(64);
        unsafe {
            self.stream
                .launch_builder(&self.causal_attention_naive)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&out)
                .arg(&(tokens as i32))
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("causal_attention launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Causal attention backward (naive O(T²), parity-first).
    pub fn causal_attention_naive_bwd(
        &self,
        q: CudaPtr,
        k: CudaPtr,
        v: CudaPtr,
        grad_output: CudaPtr,
        grad_q: CudaPtr,
        grad_k: CudaPtr,
        grad_v: CudaPtr,
        tokens: u32,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> PgResult<()> {
        let total_heads = tokens * num_heads;
        let block = 32u32.max(64u32.min(head_dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.causal_attention_naive_bwd)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&grad_output)
                .arg(&grad_q)
                .arg(&grad_k)
                .arg(&grad_v)
                .arg(&(tokens as i32))
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("causal_attention_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Flash-style f32 causal attention forward.
    /// One CTA computes one row using online softmax; no score matrix is materialized.
    pub fn causal_attention_online_fwd(
        &self,
        q: CudaPtr,
        k: CudaPtr,
        v: CudaPtr,
        out: CudaPtr,
        tokens: u32,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> PgResult<()> {
        if head_dim == 0 || head_dim > 256 {
            return Err(PgError::InvalidOp(format!(
                "online attention supports 1..=256 head_dim, got {head_dim}"
            )));
        }
        let total_heads = tokens * num_heads;
        let block = 32u32.max(head_dim.next_power_of_two()).min(256);
        unsafe {
            self.stream
                .launch_builder(&self.causal_attention_online)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&out)
                .arg(&(tokens as i32))
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("causal_attention_online launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Flash-style f32 causal attention backward.
    pub fn causal_attention_online_bwd(
        &self,
        q: CudaPtr,
        k: CudaPtr,
        v: CudaPtr,
        grad_output: CudaPtr,
        grad_q: CudaPtr,
        grad_k: CudaPtr,
        grad_v: CudaPtr,
        tokens: u32,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> PgResult<()> {
        if head_dim == 0 || head_dim > 256 {
            return Err(PgError::InvalidOp(format!(
                "online attention backward supports 1..=256 head_dim, got {head_dim}"
            )));
        }
        let total_heads = tokens * num_heads;
        let block = 32u32.max(head_dim.next_power_of_two()).min(256);
        unsafe {
            self.stream
                .launch_builder(&self.causal_attention_online_bwd)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&grad_output)
                .arg(&grad_q)
                .arg(&grad_k)
                .arg(&grad_v)
                .arg(&(tokens as i32))
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("causal_attention_online_bwd launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    pub fn attn_out_gate_fwd(
        &self,
        attn_in: CudaPtr,
        gate_input: CudaPtr,
        weight: CudaPtr,
        bias: CudaPtr,
        out: CudaPtr,
        gate_values: CudaPtr,
        tokens: u32,
        num_heads: u32,
        head_dim: u32,
        model_dim: u32,
        gate_width: u32,
    ) -> PgResult<()> {
        if gate_width == 0 || gate_width > model_dim {
            return Err(PgError::InvalidOp(format!(
                "invalid AttnOutGate width {gate_width} for model_dim {model_dim}"
            )));
        }
        let block = 32u32
            .max(gate_width.max(head_dim).next_power_of_two())
            .min(256);
        unsafe {
            self.stream
                .launch_builder(&self.attn_out_gate_fwd)
                .arg(&attn_in)
                .arg(&gate_input)
                .arg(&weight)
                .arg(&bias)
                .arg(&out)
                .arg(&gate_values)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens * num_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("attn_out_gate_fwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn attn_out_gate_bwd(
        &self,
        attn_in: CudaPtr,
        gate_input: CudaPtr,
        gate_values: CudaPtr,
        grad_out: CudaPtr,
        weight: CudaPtr,
        grad_attn_in: CudaPtr,
        grad_gate_input: CudaPtr,
        grad_weight: CudaPtr,
        grad_bias: CudaPtr,
        tokens: u32,
        num_heads: u32,
        head_dim: u32,
        model_dim: u32,
        gate_width: u32,
    ) -> PgResult<()> {
        if gate_width == 0 || gate_width > model_dim {
            return Err(PgError::InvalidOp(format!(
                "invalid AttnOutGate width {gate_width} for model_dim {model_dim}"
            )));
        }
        let block = 32u32
            .max(gate_width.max(head_dim).next_power_of_two())
            .min(256);
        unsafe {
            self.stream
                .launch_builder(&self.attn_out_gate_bwd)
                .arg(&attn_in)
                .arg(&gate_input)
                .arg(&gate_values)
                .arg(&grad_out)
                .arg(&weight)
                .arg(&grad_attn_in)
                .arg(&grad_gate_input)
                .arg(&grad_weight)
                .arg(&grad_bias)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens * num_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("attn_out_gate_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn sparse_attn_gate_fwd(
        &self,
        attn_in: CudaPtr,
        gate_input: CudaPtr,
        weight: CudaPtr,
        out: CudaPtr,
        gate_values: CudaPtr,
        tokens: u32,
        num_heads: u32,
        head_dim: u32,
        model_dim: u32,
        gate_width: u32,
        gate_scale: f32,
    ) -> PgResult<()> {
        if gate_width == 0 || gate_width > model_dim {
            return Err(PgError::InvalidOp(format!(
                "invalid SparseAttnGate width {gate_width} for model_dim {model_dim}"
            )));
        }
        let block = 32u32
            .max(gate_width.max(head_dim).next_power_of_two())
            .min(256);
        unsafe {
            self.stream
                .launch_builder(&self.sparse_attn_gate_fwd)
                .arg(&attn_in)
                .arg(&gate_input)
                .arg(&weight)
                .arg(&out)
                .arg(&gate_values)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .arg(&gate_scale)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens * num_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("sparse_attn_gate_fwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn sparse_attn_gate_xsa_fwd(
        &self,
        y: CudaPtr,
        v: CudaPtr,
        gate_input: CudaPtr,
        weight: CudaPtr,
        out: CudaPtr,
        out_bf16: CudaPtr,
        gate_values: CudaPtr,
        tokens: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        model_dim: u32,
        gate_width: u32,
        gate_scale: f32,
    ) -> PgResult<()> {
        if gate_width == 0 || gate_width > model_dim {
            return Err(PgError::InvalidOp(format!(
                "invalid SparseAttnGate width {gate_width} for model_dim {model_dim}"
            )));
        }
        let block = 32u32
            .max(gate_width.max(head_dim).next_power_of_two())
            .min(256);
        unsafe {
            self.stream
                .launch_builder(&self.sparse_attn_gate_xsa_fwd)
                .arg(&y)
                .arg(&v)
                .arg(&gate_input)
                .arg(&weight)
                .arg(&out)
                .arg(&out_bf16)
                .arg(&gate_values)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .arg(&gate_scale)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens * num_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("sparse_attn_gate_xsa_fwd launch: {:?}", e))
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn_gate_xsa_fwd_bf16_bhsd(
        &self,
        y_bhsd: CudaPtr,
        v_bhsd: CudaPtr,
        gate_input: CudaPtr,
        weight: CudaPtr,
        out: CudaPtr,
        out_bf16: CudaPtr,
        gate_values: CudaPtr,
        batch: u32,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        model_dim: u32,
        gate_width: u32,
        gate_scale: f32,
    ) -> PgResult<()> {
        if gate_width == 0 || gate_width > model_dim {
            return Err(PgError::InvalidOp(format!(
                "invalid SparseAttnGate width {gate_width} for model_dim {model_dim}"
            )));
        }
        let tokens = batch
            .checked_mul(seq_len)
            .ok_or_else(|| PgError::InvalidOp("SparseAttnGate+XSA BF16 token overflow".into()))?;
        let block = 32u32
            .max(gate_width.max(head_dim).next_power_of_two())
            .min(256);
        unsafe {
            self.stream
                .launch_builder(&self.sparse_attn_gate_xsa_fwd_bf16_bhsd)
                .arg(&y_bhsd)
                .arg(&v_bhsd)
                .arg(&gate_input)
                .arg(&weight)
                .arg(&out)
                .arg(&out_bf16)
                .arg(&gate_values)
                .arg(&(batch as i32))
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .arg(&gate_scale)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens * num_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "sparse_attn_gate_xsa_fwd_bf16_bhsd launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    pub fn sparse_attn_gate_bwd(
        &self,
        attn_in: CudaPtr,
        gate_input: CudaPtr,
        gate_values: CudaPtr,
        grad_out: CudaPtr,
        weight: CudaPtr,
        grad_attn_in: CudaPtr,
        grad_gate_input: CudaPtr,
        grad_weight: CudaPtr,
        tokens: u32,
        num_heads: u32,
        head_dim: u32,
        model_dim: u32,
        gate_width: u32,
        gate_scale: f32,
    ) -> PgResult<()> {
        if gate_width == 0 || gate_width > model_dim {
            return Err(PgError::InvalidOp(format!(
                "invalid SparseAttnGate width {gate_width} for model_dim {model_dim}"
            )));
        }
        let block = 32u32
            .max(gate_width.max(head_dim).next_power_of_two())
            .min(256);
        unsafe {
            self.stream
                .launch_builder(&self.sparse_attn_gate_bwd)
                .arg(&attn_in)
                .arg(&gate_input)
                .arg(&gate_values)
                .arg(&grad_out)
                .arg(&weight)
                .arg(&grad_attn_in)
                .arg(&grad_gate_input)
                .arg(&grad_weight)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .arg(&gate_scale)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens * num_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| PgError::InvalidOp(format!("sparse_attn_gate_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn sparse_attn_gate_bwd_two_pass(
        &self,
        attn_in: CudaPtr,
        gate_input: CudaPtr,
        gate_values_and_grad_score: CudaPtr,
        grad_out: CudaPtr,
        weight: CudaPtr,
        grad_attn_in: CudaPtr,
        grad_gate_input: CudaPtr,
        grad_weight: CudaPtr,
        tokens: u32,
        num_heads: u32,
        head_dim: u32,
        model_dim: u32,
        gate_width: u32,
        gate_scale: f32,
    ) -> PgResult<()> {
        if gate_width == 0 || gate_width > model_dim {
            return Err(PgError::InvalidOp(format!(
                "invalid SparseAttnGate width {gate_width} for model_dim {model_dim}"
            )));
        }
        let block = 32u32
            .max(gate_width.max(head_dim).next_power_of_two())
            .min(256);
        unsafe {
            self.stream
                .launch_builder(&self.sparse_attn_gate_bwd_stage1)
                .arg(&attn_in)
                .arg(&gate_input)
                .arg(&gate_values_and_grad_score)
                .arg(&grad_out)
                .arg(&weight)
                .arg(&grad_attn_in)
                .arg(&grad_gate_input)
                .arg(&gate_values_and_grad_score)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .arg(&gate_scale)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens * num_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!("sparse_attn_gate_bwd_stage1 launch: {:?}", e))
                })?;
            self.stream
                .launch_builder(&self.sparse_attn_gate_weight_grad)
                .arg(&gate_input)
                .arg(&gate_values_and_grad_score)
                .arg(&grad_weight)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_heads, gate_width, 1).into(),
                    block_dim: (256, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "sparse_attn_gate_weight_grad_reduce launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    pub fn sparse_attn_gate_xsa_bwd_bf16_bhsd_two_pass(
        &self,
        y_bhsd: CudaPtr,
        v_bhsd: CudaPtr,
        gate_input: CudaPtr,
        gate_values_and_grad_score: CudaPtr,
        grad_out: CudaPtr,
        weight: CudaPtr,
        grad_y: CudaPtr,
        grad_v: CudaPtr,
        grad_gate_input: CudaPtr,
        grad_weight: CudaPtr,
        batch: u32,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        model_dim: u32,
        gate_width: u32,
        gate_scale: f32,
    ) -> PgResult<()> {
        if gate_width == 0 || gate_width > model_dim {
            return Err(PgError::InvalidOp(format!(
                "invalid SparseAttnGate width {gate_width} for model_dim {model_dim}"
            )));
        }
        let tokens = batch
            .checked_mul(seq_len)
            .ok_or_else(|| PgError::InvalidOp("SparseAttnGate XSA token count overflow".into()))?;
        let block = 32u32
            .max(gate_width.max(head_dim).next_power_of_two())
            .min(256);
        unsafe {
            self.stream
                .launch_builder(&self.sparse_attn_gate_xsa_bwd_bf16_bhsd)
                .arg(&y_bhsd)
                .arg(&v_bhsd)
                .arg(&gate_input)
                .arg(&gate_values_and_grad_score)
                .arg(&grad_out)
                .arg(&weight)
                .arg(&grad_y)
                .arg(&grad_v)
                .arg(&grad_gate_input)
                .arg(&gate_values_and_grad_score)
                .arg(&(batch as i32))
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .arg(&gate_scale)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens * num_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "sparse_attn_gate_xsa_backward_bf16_bhsd launch: {:?}",
                        e
                    ))
                })?;
            self.stream
                .launch_builder(&self.sparse_attn_gate_weight_grad)
                .arg(&gate_input)
                .arg(&gate_values_and_grad_score)
                .arg(&grad_weight)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_heads, gate_width, 1).into(),
                    block_dim: (256, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "sparse_attn_gate_weight_grad_reduce launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    pub fn sparse_attn_gate_xsa_bwd_bf16_bhsd_warpheads_two_pass(
        &self,
        y_bhsd: CudaPtr,
        v_bhsd: CudaPtr,
        gate_input: CudaPtr,
        gate_values_and_grad_score: CudaPtr,
        grad_out: CudaPtr,
        weight: CudaPtr,
        grad_y: CudaPtr,
        grad_v: CudaPtr,
        grad_gate_input: CudaPtr,
        grad_weight: CudaPtr,
        batch: u32,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        model_dim: u32,
        gate_width: u32,
        gate_scale: f32,
    ) -> PgResult<()> {
        if gate_width == 0 || gate_width > model_dim {
            return Err(PgError::InvalidOp(format!(
                "invalid SparseAttnGate width {gate_width} for model_dim {model_dim}"
            )));
        }
        if num_heads == 0 || num_heads * 32 > 1024 {
            return Err(PgError::InvalidOp(format!(
                "warp-head SparseAttnGate XSA backward requires 1..=32 heads; got {num_heads}"
            )));
        }
        if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
            return Err(PgError::InvalidOp(format!(
                "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )));
        }
        let tokens = batch
            .checked_mul(seq_len)
            .ok_or_else(|| PgError::InvalidOp("SparseAttnGate XSA token count overflow".into()))?;
        unsafe {
            self.stream
                .launch_builder(&self.sparse_attn_gate_xsa_bwd_bf16_bhsd_warpheads)
                .arg(&y_bhsd)
                .arg(&v_bhsd)
                .arg(&gate_input)
                .arg(&gate_values_and_grad_score)
                .arg(&grad_out)
                .arg(&weight)
                .arg(&grad_y)
                .arg(&grad_v)
                .arg(&grad_gate_input)
                .arg(&gate_values_and_grad_score)
                .arg(&(batch as i32))
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .arg(&gate_scale)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (tokens, 1, 1).into(),
                    block_dim: (num_heads * 32, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "sparse_attn_gate_xsa_backward_bf16_bhsd_warpheads launch: {:?}",
                        e
                    ))
                })?;
            self.stream
                .launch_builder(&self.sparse_attn_gate_weight_grad)
                .arg(&gate_input)
                .arg(&gate_values_and_grad_score)
                .arg(&grad_weight)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(model_dim as i32))
                .arg(&(gate_width as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_heads, gate_width, 1).into(),
                    block_dim: (256, 1, 1).into(),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| {
                    PgError::InvalidOp(format!(
                        "sparse_attn_gate_weight_grad_reduce launch: {:?}",
                        e
                    ))
                })?;
        }
        Ok(())
    }

    /// XSA forward: z = y - (y·v / ||v||²) * v
    /// attn_out: [tokens * num_heads, head_dim], v: [tokens * num_kv_heads, head_dim]
    /// out: [tokens * num_heads, head_dim]
    pub fn xsa_fwd(
        &self,
        attn_out: CudaPtr,
        v: CudaPtr,
        out: CudaPtr,
        tokens: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> PgResult<()> {
        let total_heads = tokens * num_heads;
        // Use 64 threads (2 warps) to cover head_dim=64 efficiently.
        // The kernel uses strided loops + cross-warp shmem reduction.
        let block = 32u32.max(64u32.min(head_dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.xsa_fwd)
                .arg(&attn_out)
                .arg(&v)
                .arg(&out)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 256,
                })
                .map_err(|e| PgError::InvalidOp(format!("xsa launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// XSA backward.
    pub fn xsa_bwd(
        &self,
        y: CudaPtr,
        v: CudaPtr,
        grad_output: CudaPtr,
        grad_y: CudaPtr,
        grad_v: CudaPtr,
        tokens: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> PgResult<()> {
        let total_heads = tokens * num_heads;
        let block = 32u32.max(64u32.min(head_dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.xsa_bwd)
                .arg(&y)
                .arg(&v)
                .arg(&grad_output)
                .arg(&grad_y)
                .arg(&grad_v)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 384,
                })
                .map_err(|e| PgError::InvalidOp(format!("xsa_bwd launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// XSA backward from saved cuDNN BF16 BHSD tensors.
    pub fn xsa_bwd_bf16_bhsd(
        &self,
        y_bhsd: CudaPtr,
        v_bhsd: CudaPtr,
        grad_output: CudaPtr,
        grad_y: CudaPtr,
        grad_v: CudaPtr,
        batch: u32,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> PgResult<()> {
        let total_heads = batch * seq_len * num_heads;
        let block = 32u32.max(64u32.min(head_dim.next_power_of_two()));
        unsafe {
            self.stream
                .launch_builder(&self.xsa_bwd_bf16_bhsd)
                .arg(&y_bhsd)
                .arg(&v_bhsd)
                .arg(&grad_output)
                .arg(&grad_y)
                .arg(&grad_v)
                .arg(&(batch as i32))
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (total_heads, 1, 1).into(),
                    block_dim: (block, 1, 1).into(),
                    shared_mem_bytes: 384,
                })
                .map_err(|e| PgError::InvalidOp(format!("xsa_bwd_bf16_bhsd launch: {:?}", e)))?;
        }
        Ok(())
    }
}
