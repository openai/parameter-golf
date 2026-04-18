/// GPU kernels for Parameter Golf — all element-wise operations.
///
/// Strategy: CUDA C source strings compiled to PTX at runtime via NVRTC,
/// loaded via `CudaContext::load_module()`, launched via `CudaStream::launch_builder()`.
///
/// All kernels operate on f32 data with f32 arithmetic internally.
/// Grid/block dimensions are set for H100 (132 SMs, 1024 threads/block).

use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaStream, CudaSlice, CudaModule, CudaFunction, PushKernelArg};
use pg_core::error::{PgError, PgResult};

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
    leaky_relu_sq_fwd: CudaFunction,
    residual_mix: CudaFunction,
    residual_add_scale: CudaFunction,
    smear_gate_fwd: CudaFunction,
    embedding_gather: CudaFunction,
    qk_norm_fwd: CudaFunction,
    partial_rope_fwd: CudaFunction,
    q_gain_fwd: CudaFunction,
    cross_entropy_fwd: CudaFunction,
    bigram_hash_embed: CudaFunction,
    add_scaled: CudaFunction,
    causal_attention_naive: CudaFunction,
    xsa_fwd: CudaFunction,
    copy_fwd: CudaFunction,
}

/// CUDA C source for all element-wise kernels.
/// f32 data, f32 compute, one thread per element (or per row for reductions).
const CUDA_SOURCE: &str = r#"
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

// ──────────────────────────────────────────────────────────────
// SmearGate: out = (1 - sigmoid(gate)) * x + sigmoid(gate) * x_prev
// NOTE: high sigmoid → more x_prev (matching CPU reference)
// ──────────────────────────────────────────────────────────────
extern "C" __global__ void smear_gate_forward(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    float* __restrict__ out,
    int tokens,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = tokens * dim;
    if (idx >= n) return;

    int tok = idx / dim;
    int d = idx % dim;
    float sig = 1.0f / (1.0f + expf(-gate[d]));
    float x_val = x[idx];
    float x_prev = (tok > 0) ? x[idx - dim] : 0.0f;
    out[idx] = (1.0f - sig) * x_val + sig * x_prev;
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
    int tokens
) {
    int t = blockIdx.x;
    if (t >= tokens) return;

    int bucket;
    if (t == 0) {
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
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int head_idx = blockIdx.x;
    if (head_idx >= tokens * num_heads) return;

    int t = head_idx / num_heads;
    int h = head_idx % num_heads;
    int hkv = h / (num_heads / num_kv_heads);  // GQA mapping

    const float* q_head = q + head_idx * head_dim;
    float* out_head = out + head_idx * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    int d = threadIdx.x;
    if (d >= head_dim) return;

    // First pass: compute max score for numerical stability
    float max_score = -1e30f;
    for (int s = 0; s <= t; s++) {
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
    for (int s = 0; s <= t; s++) {
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
"#;

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
        ).map_err(|e| PgError::InvalidOp(format!("NVRTC compilation failed: {:?}", e)))?;

        let module = ctx.load_module(ptx)
            .map_err(|e| PgError::InvalidOp(format!("PTX module load failed: {:?}", e)))?;

        
        Ok(Self {
            stream,
            _module: module.clone(),
            rms_norm_fwd: module.load_function("rms_norm_forward").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            leaky_relu_sq_fwd: module.load_function("leaky_relu_sq_forward").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_mix: module.load_function("residual_mix").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            residual_add_scale: module.load_function("residual_add_scale").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            smear_gate_fwd: module.load_function("smear_gate_forward").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            embedding_gather: module.load_function("embedding_gather").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            qk_norm_fwd: module.load_function("qk_norm_forward").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            partial_rope_fwd: module.load_function("partial_rope_forward").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            q_gain_fwd: module.load_function("q_gain_forward").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            cross_entropy_fwd: module.load_function("cross_entropy_forward").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            bigram_hash_embed: module.load_function("bigram_hash_embed").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            add_scaled: module.load_function("add_scaled").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            causal_attention_naive: module.load_function("causal_attention_naive").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            xsa_fwd: module.load_function("xsa_forward").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
            copy_fwd: module.load_function("copy_fwd").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?,
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
        let block = 256u32.min(dim.next_power_of_two());
        unsafe {
            self.stream.launch_builder(&self.rms_norm_fwd)
                .arg(&x)
                .arg(&y)
                .arg(&(dim as i32))
                .arg(&ln_scale_factor)
                .arg(&eps)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (num_rows, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 128 })
                .map_err(|e| PgError::InvalidOp(format!("rms_norm launch: {:?}", e)))?;
        }
        Ok(())
    }

    pub fn copy_fwd(
        &self,
        src: CudaPtr,
        dst: CudaPtr,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream.launch_builder(&self.copy_fwd)
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

    /// LeakyReLU² forward: y = leaky_relu(x, 0.5)²
    pub fn leaky_relu_sq_forward(
        &self,
        x: CudaPtr,
        y: CudaPtr,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream.launch_builder(&self.leaky_relu_sq_fwd)
                .arg(&x)
                .arg(&y)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 0 })
                .map_err(|e| PgError::InvalidOp(format!("leaky_relu_sq launch: {:?}", e)))?;
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
            self.stream.launch_builder(&self.residual_mix)
                .arg(&x)
                .arg(&x0)
                .arg(&mix)
                .arg(&out)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 0 })
                .map_err(|e| PgError::InvalidOp(format!("residual_mix launch: {:?}", e)))?;
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
            self.stream.launch_builder(&self.residual_add_scale)
                .arg(&x)
                .arg(&proj)
                .arg(&scale)
                .arg(&(dim as i32))
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 0 })
                .map_err(|e| PgError::InvalidOp(format!("residual_add_scale launch: {:?}", e)))?;
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
        dim: u32,
    ) -> PgResult<()> {
        let n = tokens * dim;
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream.launch_builder(&self.smear_gate_fwd)
                .arg(&x)
                .arg(&gate)
                .arg(&out)
                .arg(&(tokens as i32))
                .arg(&(dim as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 0 })
                .map_err(|e| PgError::InvalidOp(format!("smear_gate launch: {:?}", e)))?;
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
            self.stream.launch_builder(&self.embedding_gather)
                .arg(&ids)
                .arg(&emb)
                .arg(&out)
                .arg(&(dim as i32))
                .arg(&(tokens as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (tokens, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 0 })
                .map_err(|e| PgError::InvalidOp(format!("embedding_gather launch: {:?}", e)))?;
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
            self.stream.launch_builder(&self.qk_norm_fwd)
                .arg(&qk)
                .arg(&(head_dim as i32))
                .arg(&(total_heads as i32))
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig { grid_dim: (total_heads, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 4 })
                .map_err(|e| PgError::InvalidOp(format!("qk_norm launch: {:?}", e)))?;
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
            self.stream.launch_builder(&self.partial_rope_fwd)
                .arg(&x)
                .arg(&cos_table)
                .arg(&sin_table)
                .arg(&(seq_len as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(rope_dims as i32))
                .arg(&(total_heads as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (total_heads, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 0 })
                .map_err(|e| PgError::InvalidOp(format!("partial_rope launch: {:?}", e)))?;
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
        let block = 64u32.min(head_dim.next_power_of_two());
        unsafe {
            self.stream.launch_builder(&self.q_gain_fwd)
                .arg(&q)
                .arg(&gain)
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(total_heads as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (total_heads, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 0 })
                .map_err(|e| PgError::InvalidOp(format!("q_gain launch: {:?}", e)))?;
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
        let block = 256u32.min(vocab_size.next_power_of_two());
        unsafe {
            self.stream.launch_builder(&self.cross_entropy_fwd)
                .arg(&logits)
                .arg(&targets)
                .arg(&losses)
                .arg(&(vocab_size as i32))
                .arg(&softcap)
                .arg(&(num_tokens as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (num_tokens, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 256 })
                .map_err(|e| PgError::InvalidOp(format!("cross_entropy launch: {:?}", e)))?;
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
    ) -> PgResult<()> {
        let block = 128u32.min(bigram_dim.next_power_of_two());
        unsafe {
            self.stream.launch_builder(&self.bigram_hash_embed)
                .arg(&ids)
                .arg(&embed)
                .arg(&out)
                .arg(&(bigram_vocab as i32))
                .arg(&(bigram_dim as i32))
                .arg(&(tokens as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (tokens, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 0 })
                .map_err(|e| PgError::InvalidOp(format!("bigram_hash_embed launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// x += alpha * y (in-place)
    pub fn add_scaled_fwd(
        &self,
        x: CudaPtr,
        y: CudaPtr,
        alpha: f32,
        n: u32,
    ) -> PgResult<()> {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        unsafe {
            self.stream.launch_builder(&self.add_scaled)
                .arg(&x)
                .arg(&y)
                .arg(&alpha)
                .arg(&(n as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 0 })
                .map_err(|e| PgError::InvalidOp(format!("add_scaled launch: {:?}", e)))?;
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
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> PgResult<()> {
        let total_heads = tokens * num_heads;
        let block = head_dim.min(64);
        unsafe {
            self.stream.launch_builder(&self.causal_attention_naive)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&out)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (total_heads, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 0 })
                .map_err(|e| PgError::InvalidOp(format!("causal_attention launch: {:?}", e)))?;
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
        let block = 64u32.min(head_dim.next_power_of_two());
        unsafe {
            self.stream.launch_builder(&self.xsa_fwd)
                .arg(&attn_out)
                .arg(&v)
                .arg(&out)
                .arg(&(tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(cudarc::driver::LaunchConfig { grid_dim: (total_heads, 1, 1).into(), block_dim: (block, 1, 1).into(), shared_mem_bytes: 256 })
                .map_err(|e| PgError::InvalidOp(format!("xsa launch: {:?}", e)))?;
        }
        Ok(())
    }
}
