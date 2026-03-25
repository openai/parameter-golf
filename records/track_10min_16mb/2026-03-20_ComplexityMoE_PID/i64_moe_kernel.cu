/**
 * i64 MoE Kernel — Token-Routed Expert Dispatch for Parameter Golf
 *
 * Full pipeline: route → scatter → expert MLP (cuBLAS) → gather
 * All control flow is integer. Float only inside expert GEMMs + SiLU.
 *
 * From vllm-i64 (https://github.com/Complexity-ML/vllm-i64)
 * Apache 2.0 — INL / Complexity-ML, 2025
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error"); } while(0)

// Route: token_id & mask → expert_id + atomic count
__global__ void route_kernel(const int64_t* token_ids, int32_t* expert_ids,
    int32_t* expert_counts, int32_t num_tokens, int32_t mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tokens) return;
    int32_t e = (int32_t)(token_ids[i] & (int64_t)mask);
    expert_ids[i] = e;
    atomicAdd(&expert_counts[e], 1);
}

// Prefix sum (tiny, single thread)
__global__ void prefix_sum_kernel(const int32_t* counts, int32_t* offsets, int32_t N) {
    if (threadIdx.x != 0) return;
    int32_t s = 0;
    for (int i = 0; i < N; i++) { offsets[i] = s; s += counts[i]; }
}

// Scatter: group tokens by expert
__global__ void scatter_kernel(const half* in, half* out, const int32_t* expert_ids,
    int32_t* scatter_idx, const int32_t* offsets, int32_t* counters,
    int32_t num_tokens, int32_t dim) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;
    __shared__ int32_t slot;
    if (threadIdx.x == 0) {
        int32_t e = expert_ids[t];
        slot = offsets[e] + atomicAdd(&counters[e], 1);
        scatter_idx[t] = slot;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        out[slot * dim + d] = in[t * dim + d];
}

// Gather: restore original order
__global__ void gather_kernel(const half* in, half* out, const int32_t* scatter_idx,
    int32_t num_tokens, int32_t dim) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;
    int32_t src = scatter_idx[t];
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        out[t * dim + d] = in[src * dim + d];
}

// SiLU + Hadamard: silu(gate) * up
__global__ void silu_hadamard_kernel(const half* gate_up, half* inter,
    int32_t num_tokens, int32_t expert_inter) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;
    for (int d = threadIdx.x; d < expert_inter; d += blockDim.x) {
        float g = __half2float(gate_up[t * 2 * expert_inter + d]);
        float u = __half2float(gate_up[t * 2 * expert_inter + expert_inter + d]);
        inter[t * expert_inter + d] = __float2half(g / (1.0f + expf(-g)) * u);
    }
}

// ========================
// PyTorch binding
// ========================

torch::Tensor i64_moe_forward(
    torch::Tensor hidden,        // [N, dim] fp16
    torch::Tensor token_ids,     // [N] int64
    torch::Tensor gate_up_w,     // [E, dim, 2*inter] fp16
    torch::Tensor down_w,        // [E, inter, dim] fp16
    int num_experts
) {
    auto N = hidden.size(0);
    auto dim = hidden.size(1);
    auto expert_inter = down_w.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int32_t mask = num_experts - 1;

    // Allocate temporaries
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(hidden.device());
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(hidden.device());
    auto expert_ids = torch::empty({N}, opts_i32);
    auto expert_counts = torch::zeros({num_experts}, opts_i32);
    auto expert_offsets = torch::empty({num_experts}, opts_i32);
    auto expert_counters = torch::zeros({num_experts}, opts_i32);
    auto scatter_idx = torch::empty({N}, opts_i32);
    auto scattered = torch::empty_like(hidden);
    auto expert_out = torch::empty_like(hidden);
    auto output = torch::empty_like(hidden);

    // 1. Route (integer bitmask)
    route_kernel<<<(N+255)/256, 256, 0, stream>>>(
        token_ids.data_ptr<int64_t>(), expert_ids.data_ptr<int32_t>(),
        expert_counts.data_ptr<int32_t>(), N, mask);

    // 2. Prefix sum
    prefix_sum_kernel<<<1, 1, 0, stream>>>(
        expert_counts.data_ptr<int32_t>(), expert_offsets.data_ptr<int32_t>(), num_experts);

    // 3. Scatter
    scatter_kernel<<<N, min((int)dim, 256), 0, stream>>>(
        (half*)hidden.data_ptr(), (half*)scattered.data_ptr(),
        expert_ids.data_ptr<int32_t>(), scatter_idx.data_ptr<int32_t>(),
        expert_offsets.data_ptr<int32_t>(), expert_counters.data_ptr<int32_t>(), N, dim);

    // 4. Expert MLP via cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);

    // Copy counts to host
    int32_t h_counts[32], h_offsets[32];
    CUDA_CHECK(cudaMemcpyAsync(h_counts, expert_counts.data_ptr<int32_t>(),
        num_experts*4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_offsets, expert_offsets.data_ptr<int32_t>(),
        num_experts*4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Temp buffers
    int max_count = 0;
    for (int e = 0; e < num_experts; e++) if (h_counts[e] > max_count) max_count = h_counts[e];
    auto gu_buf = torch::empty({max_count, 2*(int)expert_inter}, opts_fp16);
    auto inter_buf = torch::empty({max_count, (int)expert_inter}, opts_fp16);

    for (int e = 0; e < num_experts; e++) {
        int cnt = h_counts[e], off = h_offsets[e];
        if (cnt == 0) continue;
        half* tok = (half*)scattered.data_ptr() + (int64_t)off * dim;
        half* out_e = (half*)expert_out.data_ptr() + (int64_t)off * dim;
        half* w_gu = (half*)gate_up_w.data_ptr() + (int64_t)e * dim * 2 * expert_inter;
        half* w_dn = (half*)down_w.data_ptr() + (int64_t)e * expert_inter * dim;

        // gate_up = tokens @ W_gate_up
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            2*expert_inter, cnt, dim, &alpha, w_gu, 2*expert_inter,
            tok, dim, &beta, (half*)gu_buf.data_ptr(), 2*expert_inter));

        // silu(gate) * up
        silu_hadamard_kernel<<<cnt, min((int)expert_inter, 256), 0, stream>>>(
            (half*)gu_buf.data_ptr(), (half*)inter_buf.data_ptr(), cnt, expert_inter);

        // down = inter @ W_down
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            dim, cnt, expert_inter, &alpha, w_dn, dim,
            (half*)inter_buf.data_ptr(), expert_inter, &beta, out_e, dim));
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    // 5. Gather (restore original order)
    gather_kernel<<<N, min((int)dim, 256), 0, stream>>>(
        (half*)expert_out.data_ptr(), (half*)output.data_ptr(),
        scatter_idx.data_ptr<int32_t>(), N, dim);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &i64_moe_forward, "I64 MoE forward (scatter dispatch)");
}
