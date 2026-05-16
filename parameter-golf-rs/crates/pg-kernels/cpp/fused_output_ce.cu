#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

static inline int cuda_status(cudaError_t status) {
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

static inline int cublas_status(cublasStatus_t status) {
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + static_cast<int>(status);
}

int get_thread_local_cublas_handle(cudaStream_t stream, cublasHandle_t* out) {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        int status = cublas_status(cublasCreate(&handle));
        if (status != 0) return status;
        status = cublas_status(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        if (status != 0) return status;
    }
    int status = cublas_status(cublasSetStream(handle, stream));
    if (status != 0) return status;
    *out = handle;
    return 0;
}

__device__ __forceinline__ unsigned short f32_to_bf16_bits(float x) {
    unsigned int bits = __float_as_uint(x);
    unsigned int lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<unsigned short>(bits >> 16);
}

__global__ void stats_init_kernel(
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    float* __restrict__ target_logit,
    int m
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m) return;
    row_max[idx] = -1.0e30f;
    row_sum[idx] = 0.0f;
    target_logit[idx] = 0.0f;
}

__global__ void stats_update_warp_rows_kernel(
    const float* __restrict__ logits_tile,
    const int* __restrict__ targets,
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    float* __restrict__ target_logit,
    int vocab_start,
    int tile_vocab,
    int tile_stride,
    float softcap,
    int m
) {
    constexpr int kWarpsPerBlock = 8;
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row_id = blockIdx.x * kWarpsPerBlock + warp_in_block;
    if (row_id >= m) return;

    const float* row = logits_tile + static_cast<size_t>(row_id) * tile_stride;
    int target = targets[row_id];
    float inv_cap = 1.0f / softcap;

    float tile_max = -1.0e30f;
    for (int i = lane; i < tile_vocab; i += 32) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        tile_max = fmaxf(tile_max, capped);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
    }
    tile_max = __shfl_sync(0xffffffff, tile_max, 0);

    float tile_sum = 0.0f;
    for (int i = lane; i < tile_vocab; i += 32) {
        float capped = softcap * tanhf(row[i] * inv_cap);
        tile_sum += expf(capped - tile_max);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
    }
    tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);

    if (lane == 0) {
        float prev_max = row_max[row_id];
        float prev_sum = row_sum[row_id];
        float new_max = fmaxf(prev_max, tile_max);
        float new_sum = prev_sum * expf(prev_max - new_max)
            + tile_sum * expf(tile_max - new_max);
        row_max[row_id] = new_max;
        row_sum[row_id] = new_sum;
        if (target >= vocab_start && target < vocab_start + tile_vocab) {
            int local = target - vocab_start;
            target_logit[row_id] = softcap * tanhf(row[local] * inv_cap);
        }
    }
}

__global__ void finalize_loss_kernel(
    const float* __restrict__ row_max,
    const float* __restrict__ row_sum,
    const float* __restrict__ target_logit,
    float* __restrict__ losses,
    int m
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m) return;
    losses[idx] = row_max[idx] + logf(row_sum[idx]) - target_logit[idx];
}

__global__ void grad_tile_warp_rows_kernel(
    const float* __restrict__ logits_tile,
    const int* __restrict__ targets,
    const float* __restrict__ row_max,
    const float* __restrict__ row_sum,
    unsigned short* __restrict__ grad_tile,
    int vocab_start,
    int tile_vocab,
    int tile_stride,
    float softcap,
    float loss_scale,
    int m
) {
    constexpr int kWarpsPerBlock = 8;
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row_id = blockIdx.x * kWarpsPerBlock + warp_in_block;
    if (row_id >= m) return;

    const float* row = logits_tile + static_cast<size_t>(row_id) * tile_stride;
    unsigned short* grad_row = grad_tile + static_cast<size_t>(row_id) * tile_vocab;
    int target = targets[row_id];
    float inv_cap = 1.0f / softcap;
    float rmax = row_max[row_id];
    float rsum = row_sum[row_id];
    for (int local = lane; local < tile_vocab; local += 32) {
        int vocab_id = vocab_start + local;
        float logit = row[local];
        float tv = tanhf(logit * inv_cap);
        float capped = softcap * tv;
        float prob = expf(capped - rmax) / rsum;
        float one_hot = target == vocab_id ? 1.0f : 0.0f;
        float grad = loss_scale * (prob - one_hot) * (1.0f - tv * tv);
        grad_row[local] = f32_to_bf16_bits(grad);
    }
}

int gemm_forward_tile(
    cublasHandle_t handle,
    const void* hidden_bf16,
    const void* weight_tile_bf16,
    float* logits_tile,
    int m,
    int tile_vocab,
    int d
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublas_status(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        tile_vocab,
        m,
        d,
        &alpha,
        weight_tile_bf16,
        CUDA_R_16BF,
        d,
        hidden_bf16,
        CUDA_R_16BF,
        d,
        &beta,
        logits_tile,
        CUDA_R_32F,
        tile_vocab,
        CUBLAS_COMPUTE_32F_FAST_16BF,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int gemm_backward_input(
    cublasHandle_t handle,
    const void* grad_tile_bf16,
    const void* weight_tile_bf16,
    float* d_hidden,
    int m,
    int tile_vocab,
    int d,
    float beta
) {
    const float alpha = 1.0f;
    return cublas_status(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d,
        m,
        tile_vocab,
        &alpha,
        weight_tile_bf16,
        CUDA_R_16BF,
        d,
        grad_tile_bf16,
        CUDA_R_16BF,
        tile_vocab,
        &beta,
        d_hidden,
        CUDA_R_32F,
        d,
        CUBLAS_COMPUTE_32F_FAST_16BF,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int gemm_backward_weight(
    cublasHandle_t handle,
    const void* grad_tile_bf16,
    const void* hidden_bf16,
    float* d_weight_tile,
    int m,
    int tile_vocab,
    int d
) {
    const float alpha = 1.0f;
    const float beta = 1.0f;
    return cublas_status(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        d,
        tile_vocab,
        m,
        &alpha,
        hidden_bf16,
        CUDA_R_16BF,
        d,
        grad_tile_bf16,
        CUDA_R_16BF,
        tile_vocab,
        &beta,
        d_weight_tile,
        CUDA_R_32F,
        d,
        CUBLAS_COMPUTE_32F_FAST_16BF,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace

extern "C" int run_fused_output_ce_stats_bf16(
    cudaStream_t stream,
    uint64_t hidden_bf16,
    uint64_t weight_bf16,
    uint64_t targets,
    uint64_t row_max,
    uint64_t row_sum,
    uint64_t target_logit,
    uint64_t losses,
    uint64_t logits_scratch_f32,
    int m,
    int v,
    int d,
    int tile_vocab,
    float softcap
) {
    if (m <= 0 || v <= 0 || d <= 0 || tile_vocab <= 0 || (v % tile_vocab) != 0) return 2;
    cublasHandle_t handle;
    int status = get_thread_local_cublas_handle(stream, &handle);
    if (status != 0) return status;

    int blocks = (m + 255) / 256;
    stats_init_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<float*>(row_max),
        reinterpret_cast<float*>(row_sum),
        reinterpret_cast<float*>(target_logit),
        m);
    status = cuda_status(cudaGetLastError());
    if (status != 0) {
        return status;
    }

    for (int vocab_start = 0; vocab_start < v; vocab_start += tile_vocab) {
        const char* weight_base = reinterpret_cast<const char*>(weight_bf16);
        const void* weight_tile = weight_base + static_cast<size_t>(vocab_start) * d * sizeof(unsigned short);
        status = gemm_forward_tile(
            handle,
            reinterpret_cast<const void*>(hidden_bf16),
            weight_tile,
            reinterpret_cast<float*>(logits_scratch_f32),
            m,
            tile_vocab,
            d);
        if (status != 0) {
            return status;
        }
        constexpr int kWarpsPerBlock = 8;
        int row_blocks = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
        stats_update_warp_rows_kernel<<<row_blocks, kWarpsPerBlock * 32, 0, stream>>>(
            reinterpret_cast<const float*>(logits_scratch_f32),
            reinterpret_cast<const int*>(targets),
            reinterpret_cast<float*>(row_max),
            reinterpret_cast<float*>(row_sum),
            reinterpret_cast<float*>(target_logit),
            vocab_start,
            tile_vocab,
            tile_vocab,
            softcap,
            m);
        status = cuda_status(cudaGetLastError());
        if (status != 0) {
            return status;
        }
    }

    finalize_loss_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const float*>(row_max),
        reinterpret_cast<const float*>(row_sum),
        reinterpret_cast<const float*>(target_logit),
        reinterpret_cast<float*>(losses),
        m);
    status = cuda_status(cudaGetLastError());
    return status;
}

extern "C" int run_fused_output_ce_backward_bf16(
    cudaStream_t stream,
    uint64_t hidden_bf16,
    uint64_t weight_bf16,
    uint64_t targets,
    uint64_t row_max,
    uint64_t row_sum,
    uint64_t d_hidden_f32,
    uint64_t d_weight_f32,
    uint64_t logits_scratch_f32,
    uint64_t grad_scratch_bf16,
    int m,
    int v,
    int d,
    int tile_vocab,
    float softcap,
    float loss_scale
) {
    if (m <= 0 || v <= 0 || d <= 0 || tile_vocab <= 0 || (v % tile_vocab) != 0) return 2;
    cublasHandle_t handle;
    int status = get_thread_local_cublas_handle(stream, &handle);
    if (status != 0) return status;

    constexpr int kWarpsPerBlock = 8;
    int row_blocks = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
    for (int vocab_start = 0; vocab_start < v; vocab_start += tile_vocab) {
        const char* weight_base = reinterpret_cast<const char*>(weight_bf16);
        const void* weight_tile = weight_base + static_cast<size_t>(vocab_start) * d * sizeof(unsigned short);
        char* d_weight_base = reinterpret_cast<char*>(d_weight_f32);
        float* d_weight_tile = reinterpret_cast<float*>(
            d_weight_base + static_cast<size_t>(vocab_start) * d * sizeof(float));
        status = gemm_forward_tile(
            handle,
            reinterpret_cast<const void*>(hidden_bf16),
            weight_tile,
            reinterpret_cast<float*>(logits_scratch_f32),
            m,
            tile_vocab,
            d);
        if (status != 0) {
            return status;
        }
        grad_tile_warp_rows_kernel<<<row_blocks, kWarpsPerBlock * 32, 0, stream>>>(
            reinterpret_cast<const float*>(logits_scratch_f32),
            reinterpret_cast<const int*>(targets),
            reinterpret_cast<const float*>(row_max),
            reinterpret_cast<const float*>(row_sum),
            reinterpret_cast<unsigned short*>(grad_scratch_bf16),
            vocab_start,
            tile_vocab,
            tile_vocab,
            softcap,
            loss_scale,
            m);
        status = cuda_status(cudaGetLastError());
        if (status != 0) {
            return status;
        }
        status = gemm_backward_input(
            handle,
            reinterpret_cast<const void*>(grad_scratch_bf16),
            weight_tile,
            reinterpret_cast<float*>(d_hidden_f32),
            m,
            tile_vocab,
            d,
            vocab_start == 0 ? 0.0f : 1.0f);
        if (status != 0) {
            return status;
        }
        status = gemm_backward_weight(
            handle,
            reinterpret_cast<const void*>(grad_scratch_bf16),
            reinterpret_cast<const void*>(hidden_bf16),
            d_weight_tile,
            m,
            tile_vocab,
            d);
        if (status != 0) {
            return status;
        }
    }

    return 0;
}
