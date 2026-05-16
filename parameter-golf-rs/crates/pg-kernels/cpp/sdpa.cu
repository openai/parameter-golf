#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

__global__ void sdpa_naive_f32_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_queries = tokens * num_heads;
    if (query_index >= total_queries) {
        return;
    }

    int token = query_index / num_heads;
    int head = query_index % num_heads;
    int group = num_heads / num_kv_heads;
    int kv_head = head / group;

    const float scale = rsqrtf((float)head_dim);
    const int q_offset = (token * num_heads + head) * head_dim;
    const int out_offset = q_offset;

    float max_score = -INFINITY;
    for (int src = 0; src <= token; ++src) {
        const int k_offset = (src * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            dot += q[q_offset + dim] * k[k_offset + dim];
        }
        float score = dot * scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    for (int dim = 0; dim < head_dim; ++dim) {
        out[out_offset + dim] = 0.0f;
    }

    float sum_exp = 0.0f;
    for (int src = 0; src <= token; ++src) {
        const int k_offset = (src * num_kv_heads + kv_head) * head_dim;
        const int v_offset = k_offset;
        float dot = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            dot += q[q_offset + dim] * k[k_offset + dim];
        }
        float weight = expf(dot * scale - max_score);
        sum_exp += weight;
        for (int dim = 0; dim < head_dim; ++dim) {
            out[out_offset + dim] += weight * v[v_offset + dim];
        }
    }

    float inv_sum = 1.0f / sum_exp;
    for (int dim = 0; dim < head_dim; ++dim) {
        out[out_offset + dim] *= inv_sum;
    }
}

}  // namespace

extern "C" {

int run_naive_sdpa_f32(
    void* stream,
    uint64_t q_ptr,
    uint64_t k_ptr,
    uint64_t v_ptr,
    uint64_t out_ptr,
    int batch_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    if (batch_tokens <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        return 1;
    }
    if (num_heads % num_kv_heads != 0) {
        return 2;
    }

    const float* q = reinterpret_cast<const float*>(q_ptr);
    const float* k = reinterpret_cast<const float*>(k_ptr);
    const float* v = reinterpret_cast<const float*>(v_ptr);
    float* out = reinterpret_cast<float*>(out_ptr);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    int total_queries = batch_tokens * num_heads;
    int threads = 128;
    int blocks = (total_queries + threads - 1) / threads;
    sdpa_naive_f32_kernel<<<blocks, threads, 0, cuda_stream>>>(
        q,
        k,
        v,
        out,
        batch_tokens,
        num_heads,
        num_kv_heads,
        head_dim
    );
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int>(err);
}

}
