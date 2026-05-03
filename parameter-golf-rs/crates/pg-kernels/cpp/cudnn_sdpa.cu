#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace {

constexpr int Q_UID = 1;
constexpr int K_UID = 2;
constexpr int V_UID = 3;
constexpr int O_UID = 4;
constexpr int STATS_UID = 5;
constexpr int DO_UID = 101;
constexpr int DQ_UID = 102;
constexpr int DK_UID = 103;
constexpr int DV_UID = 104;

struct Key {
    int device;
    int batch;
    int seq;
    int heads;
    int kv_heads;
    int head_dim;

    bool operator==(const Key& other) const {
        return device == other.device && batch == other.batch && seq == other.seq &&
               heads == other.heads && kv_heads == other.kv_heads && head_dim == other.head_dim;
    }
};

struct KeyHash {
    size_t operator()(const Key& key) const {
        size_t h = static_cast<size_t>(key.device);
        h = h * 1315423911u + static_cast<size_t>(key.batch);
        h = h * 1315423911u + static_cast<size_t>(key.seq);
        h = h * 1315423911u + static_cast<size_t>(key.heads);
        h = h * 1315423911u + static_cast<size_t>(key.kv_heads);
        h = h * 1315423911u + static_cast<size_t>(key.head_dim);
        return h;
    }
};

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    int ensure(size_t required) {
        if (bytes >= required) {
            return 0;
        }
        if (ptr != nullptr) {
            cudaError_t free_err = cudaFree(ptr);
            if (free_err != cudaSuccess) {
                return 1000 + static_cast<int>(free_err);
            }
        }
        ptr = nullptr;
        bytes = 0;
        if (required == 0) {
            return 0;
        }
        cudaError_t alloc_err = cudaMalloc(&ptr, required);
        if (alloc_err != cudaSuccess) {
            return 1100 + static_cast<int>(alloc_err);
        }
        bytes = required;
        return 0;
    }
};

struct GraphCache {
    std::mutex mutex;
    cudnnHandle_t handle = nullptr;
    std::shared_ptr<fe::graph::Graph> fwd_graph;
    std::shared_ptr<fe::graph::Graph> bwd_graph;
    DeviceBuffer workspace;
    DeviceBuffer q_bf16;
    DeviceBuffer k_bf16;
    DeviceBuffer v_bf16;
    DeviceBuffer o_bf16;
    DeviceBuffer do_bf16;
    DeviceBuffer dq_bf16;
    DeviceBuffer dk_bf16;
    DeviceBuffer dv_bf16;
    DeviceBuffer stats;
    size_t workspace_bytes = 0;
    bool has_forward_stats = false;

    ~GraphCache() {
        if (handle != nullptr) {
            cudnnDestroy(handle);
        }
    }
};

std::mutex g_cache_mutex;
std::unordered_map<Key, std::shared_ptr<GraphCache>, KeyHash> g_cache;

__global__ void f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16_rn(in[idx]);
    }
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __bfloat162float(in[idx]);
    }
}

__global__ void f32_bthd_to_bhsd_bf16_kernel(
    const float* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int batch,
    int seq,
    int heads,
    int head_dim,
    size_t n
) {
    size_t in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (in_idx >= n) {
        return;
    }
    int d = static_cast<int>(in_idx % head_dim);
    size_t tmp = in_idx / head_dim;
    int h = static_cast<int>(tmp % heads);
    tmp /= heads;
    int t = static_cast<int>(tmp % seq);
    int b = static_cast<int>(tmp / seq);
    size_t out_idx =
        (((static_cast<size_t>(b) * heads + h) * seq + t) * head_dim) + d;
    out[out_idx] = __float2bfloat16_rn(in[in_idx]);
}

__global__ void bf16_bhsd_to_bthd_f32_kernel(
    const __nv_bfloat16* __restrict__ in,
    float* __restrict__ out,
    int batch,
    int seq,
    int heads,
    int head_dim,
    size_t n
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= n) {
        return;
    }
    int d = static_cast<int>(out_idx % head_dim);
    size_t tmp = out_idx / head_dim;
    int h = static_cast<int>(tmp % heads);
    tmp /= heads;
    int t = static_cast<int>(tmp % seq);
    int b = static_cast<int>(tmp / seq);
    size_t in_idx =
        (((static_cast<size_t>(b) * heads + h) * seq + t) * head_dim) + d;
    out[out_idx] = __bfloat162float(in[in_idx]);
}

__global__ void bf16_bhsd_to_bthd_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int batch,
    int seq,
    int heads,
    int head_dim,
    size_t n
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= n) {
        return;
    }
    int d = static_cast<int>(out_idx % head_dim);
    size_t tmp = out_idx / head_dim;
    int h = static_cast<int>(tmp % heads);
    tmp /= heads;
    int t = static_cast<int>(tmp % seq);
    int b = static_cast<int>(tmp / seq);
    size_t in_idx =
        (((static_cast<size_t>(b) * heads + h) * seq + t) * head_dim) + d;
    out[out_idx] = in[in_idx];
}

int convert_f32_to_bf16(cudaStream_t stream, const float* in, void* out, size_t n) {
    if (n == 0) {
        return 0;
    }
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(in, reinterpret_cast<__nv_bfloat16*>(out), n);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int>(err);
}

int convert_bf16_to_f32(cudaStream_t stream, const void* in, float* out, size_t n) {
    if (n == 0) {
        return 0;
    }
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(in), out, n);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int>(err);
}

int convert_f32_bthd_to_bhsd_bf16(
    cudaStream_t stream,
    const float* in,
    void* out,
    int batch,
    int seq,
    int heads,
    int head_dim
) {
    const size_t n = static_cast<size_t>(batch) * seq * heads * head_dim;
    if (n == 0) {
        return 0;
    }
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    f32_bthd_to_bhsd_bf16_kernel<<<blocks, threads, 0, stream>>>(
        in,
        reinterpret_cast<__nv_bfloat16*>(out),
        batch,
        seq,
        heads,
        head_dim,
        n);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int>(err);
}

int convert_bf16_bhsd_to_bthd_f32(
    cudaStream_t stream,
    const void* in,
    float* out,
    int batch,
    int seq,
    int heads,
    int head_dim
) {
    const size_t n = static_cast<size_t>(batch) * seq * heads * head_dim;
    if (n == 0) {
        return 0;
    }
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    bf16_bhsd_to_bthd_f32_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in),
        out,
        batch,
        seq,
        heads,
        head_dim,
        n);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int>(err);
}

int convert_bf16_bhsd_to_bthd_bf16(
    cudaStream_t stream,
    const void* in,
    void* out,
    int batch,
    int seq,
    int heads,
    int head_dim
) {
    const size_t n = static_cast<size_t>(batch) * seq * heads * head_dim;
    if (n == 0) {
        return 0;
    }
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    bf16_bhsd_to_bthd_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in),
        reinterpret_cast<__nv_bfloat16*>(out),
        batch,
        seq,
        heads,
        head_dim,
        n);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int>(err);
}

bool deterministic_sdpa_backward_enabled() {
    const char* raw = std::getenv("PG_CUDNN_SDPA_DETERMINISTIC");
    if (raw == nullptr) {
        return false;
    }
    return raw[0] == '1' || raw[0] == 't' || raw[0] == 'T' ||
           raw[0] == 'y' || raw[0] == 'Y';
}

std::shared_ptr<fe::graph::Graph> create_forward_graph(
    int b,
    int hq,
    int hk,
    int seq,
    int d,
    float attn_scale
) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const int64_t b_stride_q = static_cast<int64_t>(hq) * seq * d;
    const int64_t b_stride_k = static_cast<int64_t>(hk) * seq * d;

    auto q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim({b, hq, seq, d})
                               .set_stride({b_stride_q, seq * d, d, 1}));
    auto k = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim({b, hk, seq, d})
                               .set_stride({b_stride_k, seq * d, d, 1}));
    auto v = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim({b, hk, seq, d})
                               .set_stride({b_stride_k, seq * d, d, 1}));

    auto options = fe::graph::SDPA_attributes()
                       .set_name("pg_cudnn_sdpa_fwd")
                       .set_generate_stats(true)
                       .set_attn_scale(attn_scale);
    options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
        .set_diagonal_band_right_bound(0);

    auto [o, stats] = graph->sdpa(q, k, v, options);
    o->set_output(true)
        .set_dim({b, hq, seq, d})
        .set_stride({b_stride_q, seq * d, d, 1})
        .set_uid(O_UID);
    stats->set_output(true)
        .set_dim({b, hq, seq, 1})
        .set_stride({static_cast<int64_t>(hq) * seq, seq, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT)
        .set_uid(STATS_UID);
    return graph;
}

std::shared_ptr<fe::graph::Graph> create_backward_graph(
    int b,
    int hq,
    int hk,
    int seq,
    int d,
    float attn_scale
) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const int64_t b_stride_q = static_cast<int64_t>(hq) * seq * d;
    const int64_t b_stride_k = static_cast<int64_t>(hk) * seq * d;

    auto q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim({b, hq, seq, d})
                               .set_stride({b_stride_q, seq * d, d, 1}));
    auto k = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim({b, hk, seq, d})
                               .set_stride({b_stride_k, seq * d, d, 1}));
    auto v = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim({b, hk, seq, d})
                               .set_stride({b_stride_k, seq * d, d, 1}));
    auto o = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("O")
                               .set_uid(O_UID)
                               .set_dim({b, hq, seq, d})
                               .set_stride({b_stride_q, seq * d, d, 1}));
    auto d_o = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("dO")
                                 .set_uid(DO_UID)
                                 .set_dim({b, hq, seq, d})
                                 .set_stride({b_stride_q, seq * d, d, 1}));
    auto stats = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Stats")
                                   .set_uid(STATS_UID)
                                   .set_dim({b, hq, seq, 1})
                                   .set_stride({static_cast<int64_t>(hq) * seq, seq, 1, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));

    auto options = fe::graph::SDPA_backward_attributes()
                       .set_name("pg_cudnn_sdpa_bwd")
                       .set_attn_scale(attn_scale);
    if (deterministic_sdpa_backward_enabled()) {
        options.set_deterministic_algorithm(true);
    }
    options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
        .set_diagonal_band_right_bound(0);

    auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, d_o, stats, options);
    dq->set_output(true)
        .set_uid(DQ_UID)
        .set_dim({b, hq, seq, d})
        .set_stride({b_stride_q, seq * d, d, 1});
    dk->set_output(true)
        .set_uid(DK_UID)
        .set_dim({b, hk, seq, d})
        .set_stride({b_stride_k, seq * d, d, 1});
    dv->set_output(true)
        .set_uid(DV_UID)
        .set_dim({b, hk, seq, d})
        .set_stride({b_stride_k, seq * d, d, 1});
    return graph;
}

int ensure_cache(GraphCache& cache, const Key& key) {
    const float scale = rsqrtf(static_cast<float>(key.head_dim));
    if (cache.handle == nullptr) {
        cudnnStatus_t st = cudnnCreate(&cache.handle);
        if (st != CUDNN_STATUS_SUCCESS) {
            return 2000 + static_cast<int>(st);
        }
    }
    if (!cache.fwd_graph) {
        cache.fwd_graph = create_forward_graph(
            key.batch, key.heads, key.kv_heads, key.seq, key.head_dim, scale);
        auto status = cache.fwd_graph->build(cache.handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
        if (!status.is_good()) {
            return 2100;
        }
    }
    if (!cache.bwd_graph) {
        cache.bwd_graph = create_backward_graph(
            key.batch, key.heads, key.kv_heads, key.seq, key.head_dim, scale);
        auto status = cache.bwd_graph->build(cache.handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
        if (!status.is_good()) {
            return 2200;
        }
    }
    int64_t fwd_workspace = 0;
    int64_t bwd_workspace = 0;
    if (!cache.fwd_graph->get_workspace_size(fwd_workspace).is_good()) {
        return 2300;
    }
    if (!cache.bwd_graph->get_workspace_size(bwd_workspace).is_good()) {
        return 2400;
    }
    cache.workspace_bytes = static_cast<size_t>(fwd_workspace > bwd_workspace ? fwd_workspace : bwd_workspace);
    int rc = cache.workspace.ensure(cache.workspace_bytes);
    if (rc != 0) return rc;

    const size_t q_elems = static_cast<size_t>(key.batch) * key.seq * key.heads * key.head_dim;
    const size_t kv_elems = static_cast<size_t>(key.batch) * key.seq * key.kv_heads * key.head_dim;
    const size_t stats_elems = static_cast<size_t>(key.batch) * key.heads * key.seq;
    const size_t q_bytes = q_elems * sizeof(__nv_bfloat16);
    const size_t kv_bytes = kv_elems * sizeof(__nv_bfloat16);
    rc = cache.q_bf16.ensure(q_bytes); if (rc != 0) return rc;
    rc = cache.k_bf16.ensure(kv_bytes); if (rc != 0) return rc;
    rc = cache.v_bf16.ensure(kv_bytes); if (rc != 0) return rc;
    rc = cache.o_bf16.ensure(q_bytes); if (rc != 0) return rc;
    rc = cache.do_bf16.ensure(q_bytes); if (rc != 0) return rc;
    rc = cache.dq_bf16.ensure(q_bytes); if (rc != 0) return rc;
    rc = cache.dk_bf16.ensure(kv_bytes); if (rc != 0) return rc;
    rc = cache.dv_bf16.ensure(kv_bytes); if (rc != 0) return rc;
    rc = cache.stats.ensure(stats_elems * sizeof(float)); if (rc != 0) return rc;
    return 0;
}

std::shared_ptr<GraphCache> get_cache(const Key& key) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    auto it = g_cache.find(key);
    if (it == g_cache.end()) {
        auto inserted = g_cache.emplace(key, std::make_shared<GraphCache>());
        it = inserted.first;
    }
    return it->second;
}

bool valid_shape(int tokens, int seq, int heads, int kv_heads, int head_dim) {
    return tokens > 0 && seq > 0 && heads > 0 && kv_heads > 0 && head_dim > 0 &&
           tokens % seq == 0 && heads % kv_heads == 0 && head_dim <= 128;
}

}  // namespace

extern "C" {

int run_cudnn_sdpa_bf16_f32_forward_with_stats(
    void* stream_ptr,
    uint64_t q_ptr,
    uint64_t k_ptr,
    uint64_t v_ptr,
    uint64_t out_ptr,
    uint64_t stats_ptr,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    if (!valid_shape(tokens, seq_len, num_heads, num_kv_heads, head_dim)) {
        return 1;
    }
    int device = 0;
    cudaError_t dev_err = cudaGetDevice(&device);
    if (dev_err != cudaSuccess) {
        return 1200 + static_cast<int>(dev_err);
    }
    Key key{device, tokens / seq_len, seq_len, num_heads, num_kv_heads, head_dim};
    auto cache = get_cache(key);
    std::lock_guard<std::mutex> cache_lock(cache->mutex);
    int status = ensure_cache(*cache, key);
    if (status != 0) {
        return status;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudnnStatus_t stream_status = cudnnSetStream(cache->handle, stream);
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        return 2500 + static_cast<int>(stream_status);
    }

    const int batch = tokens / seq_len;
    int rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(q_ptr),
        cache->q_bf16.ptr,
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 3000 + rc;
    rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(k_ptr),
        cache->k_bf16.ptr,
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 3100 + rc;
    rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(v_ptr),
        cache->v_bf16.ptr,
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 3200 + rc;

    void* stats_storage =
        stats_ptr == 0 ? cache->stats.ptr : reinterpret_cast<void*>(stats_ptr);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
        {Q_UID, cache->q_bf16.ptr},
        {K_UID, cache->k_bf16.ptr},
        {V_UID, cache->v_bf16.ptr},
        {O_UID, cache->o_bf16.ptr},
        {STATS_UID, stats_storage},
    };
    auto exec_status = cache->fwd_graph->execute(cache->handle, pack, cache->workspace.ptr);
    if (!exec_status.is_good()) {
        cache->has_forward_stats = false;
        return 3300;
    }
    // Internal stats are valid only when the call writes to the graph cache.
    // Caller-owned stats are used by saved-activation backward paths and must
    // not leave the cache marked as valid for a later internal-stats backward.
    cache->has_forward_stats = stats_ptr == 0;
    rc = convert_bf16_bhsd_to_bthd_f32(
        stream,
        cache->o_bf16.ptr,
        reinterpret_cast<float*>(out_ptr),
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 3400 + rc;
    return 0;
}

int run_cudnn_sdpa_bf16_f32_forward(
    void* stream_ptr,
    uint64_t q_ptr,
    uint64_t k_ptr,
    uint64_t v_ptr,
    uint64_t out_ptr,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    return run_cudnn_sdpa_bf16_f32_forward_with_stats(
        stream_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        0,
        tokens,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim);
}

int run_cudnn_sdpa_bf16_f32_forward_with_stats_saved_bf16(
    void* stream_ptr,
    uint64_t q_ptr,
    uint64_t k_ptr,
    uint64_t v_ptr,
    uint64_t out_ptr,
    uint64_t stats_ptr,
    uint64_t q_bf16_ptr,
    uint64_t k_bf16_ptr,
    uint64_t v_bf16_ptr,
    uint64_t out_bf16_ptr,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    if (!valid_shape(tokens, seq_len, num_heads, num_kv_heads, head_dim)) {
        return 1;
    }
    if (q_bf16_ptr == 0 || k_bf16_ptr == 0 || v_bf16_ptr == 0 || out_bf16_ptr == 0) {
        return 3;
    }
    int device = 0;
    cudaError_t dev_err = cudaGetDevice(&device);
    if (dev_err != cudaSuccess) {
        return 1200 + static_cast<int>(dev_err);
    }
    Key key{device, tokens / seq_len, seq_len, num_heads, num_kv_heads, head_dim};
    auto cache = get_cache(key);
    std::lock_guard<std::mutex> cache_lock(cache->mutex);
    int status = ensure_cache(*cache, key);
    if (status != 0) {
        return status;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudnnStatus_t stream_status = cudnnSetStream(cache->handle, stream);
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        return 2500 + static_cast<int>(stream_status);
    }

    const int batch = tokens / seq_len;
    void* q_bf16 = reinterpret_cast<void*>(q_bf16_ptr);
    void* k_bf16 = reinterpret_cast<void*>(k_bf16_ptr);
    void* v_bf16 = reinterpret_cast<void*>(v_bf16_ptr);
    void* o_bf16 = reinterpret_cast<void*>(out_bf16_ptr);

    int rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(q_ptr),
        q_bf16,
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 3000 + rc;
    rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(k_ptr),
        k_bf16,
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 3100 + rc;
    rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(v_ptr),
        v_bf16,
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 3200 + rc;

    void* stats_storage =
        stats_ptr == 0 ? cache->stats.ptr : reinterpret_cast<void*>(stats_ptr);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
        {Q_UID, q_bf16},
        {K_UID, k_bf16},
        {V_UID, v_bf16},
        {O_UID, o_bf16},
        {STATS_UID, stats_storage},
    };
    auto exec_status = cache->fwd_graph->execute(cache->handle, pack, cache->workspace.ptr);
    if (!exec_status.is_good()) {
        cache->has_forward_stats = false;
        return 3300;
    }
    cache->has_forward_stats = stats_ptr == 0;
    rc = convert_bf16_bhsd_to_bthd_f32(
        stream,
        o_bf16,
        reinterpret_cast<float*>(out_ptr),
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 3400 + rc;
    return 0;
}

int run_cudnn_sdpa_bf16_f32_forward_with_stats_prepacked_bf16(
    void* stream_ptr,
    uint64_t q_bf16_ptr,
    uint64_t k_bf16_ptr,
    uint64_t v_bf16_ptr,
    uint64_t out_ptr,
    uint64_t stats_ptr,
    uint64_t out_bf16_ptr,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    if (!valid_shape(tokens, seq_len, num_heads, num_kv_heads, head_dim)) {
        return 1;
    }
    if (q_bf16_ptr == 0 || k_bf16_ptr == 0 || v_bf16_ptr == 0 || out_bf16_ptr == 0) {
        return 3;
    }
    int device = 0;
    cudaError_t dev_err = cudaGetDevice(&device);
    if (dev_err != cudaSuccess) {
        return 1200 + static_cast<int>(dev_err);
    }
    Key key{device, tokens / seq_len, seq_len, num_heads, num_kv_heads, head_dim};
    auto cache = get_cache(key);
    std::lock_guard<std::mutex> cache_lock(cache->mutex);
    int status = ensure_cache(*cache, key);
    if (status != 0) {
        return status;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudnnStatus_t stream_status = cudnnSetStream(cache->handle, stream);
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        return 2500 + static_cast<int>(stream_status);
    }

    void* q_bf16 = reinterpret_cast<void*>(q_bf16_ptr);
    void* k_bf16 = reinterpret_cast<void*>(k_bf16_ptr);
    void* v_bf16 = reinterpret_cast<void*>(v_bf16_ptr);
    void* o_bf16 = reinterpret_cast<void*>(out_bf16_ptr);
    void* stats_storage =
        stats_ptr == 0 ? cache->stats.ptr : reinterpret_cast<void*>(stats_ptr);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
        {Q_UID, q_bf16},
        {K_UID, k_bf16},
        {V_UID, v_bf16},
        {O_UID, o_bf16},
        {STATS_UID, stats_storage},
    };
    auto exec_status = cache->fwd_graph->execute(cache->handle, pack, cache->workspace.ptr);
    if (!exec_status.is_good()) {
        cache->has_forward_stats = false;
        return 3300;
    }
    cache->has_forward_stats = stats_ptr == 0;

    const int batch = tokens / seq_len;
    int rc = convert_bf16_bhsd_to_bthd_f32(
        stream,
        o_bf16,
        reinterpret_cast<float*>(out_ptr),
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 3400 + rc;
    return 0;
}

int run_cudnn_sdpa_bf16_forward_with_stats_prepacked_bf16_only(
    void* stream_ptr,
    uint64_t q_bf16_ptr,
    uint64_t k_bf16_ptr,
    uint64_t v_bf16_ptr,
    uint64_t stats_ptr,
    uint64_t out_bf16_ptr,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    if (!valid_shape(tokens, seq_len, num_heads, num_kv_heads, head_dim)) {
        return 1;
    }
    if (q_bf16_ptr == 0 || k_bf16_ptr == 0 || v_bf16_ptr == 0 || out_bf16_ptr == 0) {
        return 3;
    }
    int device = 0;
    cudaError_t dev_err = cudaGetDevice(&device);
    if (dev_err != cudaSuccess) {
        return 1200 + static_cast<int>(dev_err);
    }
    Key key{device, tokens / seq_len, seq_len, num_heads, num_kv_heads, head_dim};
    auto cache = get_cache(key);
    std::lock_guard<std::mutex> cache_lock(cache->mutex);
    int status = ensure_cache(*cache, key);
    if (status != 0) {
        return status;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudnnStatus_t stream_status = cudnnSetStream(cache->handle, stream);
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        return 2500 + static_cast<int>(stream_status);
    }

    void* q_bf16 = reinterpret_cast<void*>(q_bf16_ptr);
    void* k_bf16 = reinterpret_cast<void*>(k_bf16_ptr);
    void* v_bf16 = reinterpret_cast<void*>(v_bf16_ptr);
    void* o_bf16 = reinterpret_cast<void*>(out_bf16_ptr);
    void* stats_storage =
        stats_ptr == 0 ? cache->stats.ptr : reinterpret_cast<void*>(stats_ptr);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
        {Q_UID, q_bf16},
        {K_UID, k_bf16},
        {V_UID, v_bf16},
        {O_UID, o_bf16},
        {STATS_UID, stats_storage},
    };
    auto exec_status = cache->fwd_graph->execute(cache->handle, pack, cache->workspace.ptr);
    if (!exec_status.is_good()) {
        cache->has_forward_stats = false;
        return 3300;
    }
    cache->has_forward_stats = stats_ptr == 0;
    return 0;
}

int run_cudnn_sdpa_bf16_f32_backward_with_stats(
    void* stream_ptr,
    uint64_t q_ptr,
    uint64_t k_ptr,
    uint64_t v_ptr,
    uint64_t out_ptr,
    uint64_t grad_out_ptr,
    uint64_t grad_q_ptr,
    uint64_t grad_k_ptr,
    uint64_t grad_v_ptr,
    uint64_t stats_ptr,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    if (!valid_shape(tokens, seq_len, num_heads, num_kv_heads, head_dim)) {
        return 1;
    }
    int device = 0;
    cudaError_t dev_err = cudaGetDevice(&device);
    if (dev_err != cudaSuccess) {
        return 1200 + static_cast<int>(dev_err);
    }
    Key key{device, tokens / seq_len, seq_len, num_heads, num_kv_heads, head_dim};
    auto cache = get_cache(key);
    std::lock_guard<std::mutex> cache_lock(cache->mutex);
    int status = ensure_cache(*cache, key);
    if (status != 0) {
        return status;
    }
    if (stats_ptr == 0 && !cache->has_forward_stats) {
        return 2;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudnnStatus_t stream_status = cudnnSetStream(cache->handle, stream);
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        return 2500 + static_cast<int>(stream_status);
    }

    const int batch = tokens / seq_len;
    int rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(q_ptr),
        cache->q_bf16.ptr,
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 4000 + rc;
    rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(k_ptr),
        cache->k_bf16.ptr,
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 4100 + rc;
    rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(v_ptr),
        cache->v_bf16.ptr,
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 4200 + rc;
    rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(out_ptr),
        cache->o_bf16.ptr,
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 4300 + rc;
    rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(grad_out_ptr),
        cache->do_bf16.ptr,
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 4400 + rc;

    void* stats_storage =
        stats_ptr == 0 ? cache->stats.ptr : reinterpret_cast<void*>(stats_ptr);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
        {Q_UID, cache->q_bf16.ptr},
        {K_UID, cache->k_bf16.ptr},
        {V_UID, cache->v_bf16.ptr},
        {O_UID, cache->o_bf16.ptr},
        {DO_UID, cache->do_bf16.ptr},
        {STATS_UID, stats_storage},
        {DQ_UID, cache->dq_bf16.ptr},
        {DK_UID, cache->dk_bf16.ptr},
        {DV_UID, cache->dv_bf16.ptr},
    };
    auto exec_status = cache->bwd_graph->execute(cache->handle, pack, cache->workspace.ptr);
    if (!exec_status.is_good()) {
        return 4500;
    }
    rc = convert_bf16_bhsd_to_bthd_f32(
        stream,
        cache->dq_bf16.ptr,
        reinterpret_cast<float*>(grad_q_ptr),
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 4600 + rc;
    rc = convert_bf16_bhsd_to_bthd_f32(
        stream,
        cache->dk_bf16.ptr,
        reinterpret_cast<float*>(grad_k_ptr),
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 4700 + rc;
    rc = convert_bf16_bhsd_to_bthd_f32(
        stream,
        cache->dv_bf16.ptr,
        reinterpret_cast<float*>(grad_v_ptr),
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 4800 + rc;
    return 0;
}

int run_cudnn_sdpa_bf16_f32_backward_with_saved_bf16_stats(
    void* stream_ptr,
    uint64_t q_bf16_ptr,
    uint64_t k_bf16_ptr,
    uint64_t v_bf16_ptr,
    uint64_t out_bf16_ptr,
    uint64_t grad_out_ptr,
    uint64_t grad_q_ptr,
    uint64_t grad_k_ptr,
    uint64_t grad_v_ptr,
    uint64_t stats_ptr,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    if (!valid_shape(tokens, seq_len, num_heads, num_kv_heads, head_dim)) {
        return 1;
    }
    if (q_bf16_ptr == 0 || k_bf16_ptr == 0 || v_bf16_ptr == 0 || out_bf16_ptr == 0) {
        return 3;
    }
    int device = 0;
    cudaError_t dev_err = cudaGetDevice(&device);
    if (dev_err != cudaSuccess) {
        return 1200 + static_cast<int>(dev_err);
    }
    Key key{device, tokens / seq_len, seq_len, num_heads, num_kv_heads, head_dim};
    auto cache = get_cache(key);
    std::lock_guard<std::mutex> cache_lock(cache->mutex);
    int status = ensure_cache(*cache, key);
    if (status != 0) {
        return status;
    }
    if (stats_ptr == 0 && !cache->has_forward_stats) {
        return 2;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudnnStatus_t stream_status = cudnnSetStream(cache->handle, stream);
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        return 2500 + static_cast<int>(stream_status);
    }

    const int batch = tokens / seq_len;
    int rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(grad_out_ptr),
        cache->do_bf16.ptr,
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 4400 + rc;

    void* stats_storage =
        stats_ptr == 0 ? cache->stats.ptr : reinterpret_cast<void*>(stats_ptr);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
        {Q_UID, reinterpret_cast<void*>(q_bf16_ptr)},
        {K_UID, reinterpret_cast<void*>(k_bf16_ptr)},
        {V_UID, reinterpret_cast<void*>(v_bf16_ptr)},
        {O_UID, reinterpret_cast<void*>(out_bf16_ptr)},
        {DO_UID, cache->do_bf16.ptr},
        {STATS_UID, stats_storage},
        {DQ_UID, cache->dq_bf16.ptr},
        {DK_UID, cache->dk_bf16.ptr},
        {DV_UID, cache->dv_bf16.ptr},
    };
    auto exec_status = cache->bwd_graph->execute(cache->handle, pack, cache->workspace.ptr);
    if (!exec_status.is_good()) {
        return 4500;
    }
    rc = convert_bf16_bhsd_to_bthd_f32(
        stream,
        cache->dq_bf16.ptr,
        reinterpret_cast<float*>(grad_q_ptr),
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 4600 + rc;
    rc = convert_bf16_bhsd_to_bthd_f32(
        stream,
        cache->dk_bf16.ptr,
        reinterpret_cast<float*>(grad_k_ptr),
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 4700 + rc;
    rc = convert_bf16_bhsd_to_bthd_f32(
        stream,
        cache->dv_bf16.ptr,
        reinterpret_cast<float*>(grad_v_ptr),
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 4800 + rc;
    return 0;
}

int run_cudnn_sdpa_bf16_f32_backward_with_saved_bf16_stats_bf16_grads(
    void* stream_ptr,
    uint64_t q_bf16_ptr,
    uint64_t k_bf16_ptr,
    uint64_t v_bf16_ptr,
    uint64_t out_bf16_ptr,
    uint64_t grad_out_ptr,
    uint64_t grad_q_bf16_ptr,
    uint64_t grad_k_bf16_ptr,
    uint64_t grad_v_bf16_ptr,
    uint64_t stats_ptr,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    if (!valid_shape(tokens, seq_len, num_heads, num_kv_heads, head_dim)) {
        return 1;
    }
    if (q_bf16_ptr == 0 || k_bf16_ptr == 0 || v_bf16_ptr == 0 || out_bf16_ptr == 0) {
        return 3;
    }
    if (grad_q_bf16_ptr == 0 || grad_k_bf16_ptr == 0 || grad_v_bf16_ptr == 0) {
        return 4;
    }
    int device = 0;
    cudaError_t dev_err = cudaGetDevice(&device);
    if (dev_err != cudaSuccess) {
        return 1200 + static_cast<int>(dev_err);
    }
    Key key{device, tokens / seq_len, seq_len, num_heads, num_kv_heads, head_dim};
    auto cache = get_cache(key);
    std::lock_guard<std::mutex> cache_lock(cache->mutex);
    int status = ensure_cache(*cache, key);
    if (status != 0) {
        return status;
    }
    if (stats_ptr == 0 && !cache->has_forward_stats) {
        return 2;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudnnStatus_t stream_status = cudnnSetStream(cache->handle, stream);
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        return 2500 + static_cast<int>(stream_status);
    }

    const int batch = tokens / seq_len;
    int rc = convert_f32_bthd_to_bhsd_bf16(
        stream,
        reinterpret_cast<const float*>(grad_out_ptr),
        cache->do_bf16.ptr,
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 4400 + rc;

    void* stats_storage =
        stats_ptr == 0 ? cache->stats.ptr : reinterpret_cast<void*>(stats_ptr);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
        {Q_UID, reinterpret_cast<void*>(q_bf16_ptr)},
        {K_UID, reinterpret_cast<void*>(k_bf16_ptr)},
        {V_UID, reinterpret_cast<void*>(v_bf16_ptr)},
        {O_UID, reinterpret_cast<void*>(out_bf16_ptr)},
        {DO_UID, cache->do_bf16.ptr},
        {STATS_UID, stats_storage},
        {DQ_UID, cache->dq_bf16.ptr},
        {DK_UID, cache->dk_bf16.ptr},
        {DV_UID, cache->dv_bf16.ptr},
    };
    auto exec_status = cache->bwd_graph->execute(cache->handle, pack, cache->workspace.ptr);
    if (!exec_status.is_good()) {
        return 4500;
    }
    rc = convert_bf16_bhsd_to_bthd_bf16(
        stream,
        cache->dq_bf16.ptr,
        reinterpret_cast<void*>(grad_q_bf16_ptr),
        batch,
        seq_len,
        num_heads,
        head_dim);
    if (rc != 0) return 4600 + rc;
    rc = convert_bf16_bhsd_to_bthd_bf16(
        stream,
        cache->dk_bf16.ptr,
        reinterpret_cast<void*>(grad_k_bf16_ptr),
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 4700 + rc;
    rc = convert_bf16_bhsd_to_bthd_bf16(
        stream,
        cache->dv_bf16.ptr,
        reinterpret_cast<void*>(grad_v_bf16_ptr),
        batch,
        seq_len,
        num_kv_heads,
        head_dim);
    if (rc != 0) return 4800 + rc;
    return 0;
}

int run_cudnn_sdpa_bf16_f32_backward(
    void* stream_ptr,
    uint64_t q_ptr,
    uint64_t k_ptr,
    uint64_t v_ptr,
    uint64_t out_ptr,
    uint64_t grad_out_ptr,
    uint64_t grad_q_ptr,
    uint64_t grad_k_ptr,
    uint64_t grad_v_ptr,
    int tokens,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    return run_cudnn_sdpa_bf16_f32_backward_with_stats(
        stream_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        grad_out_ptr,
        grad_q_ptr,
        grad_k_ptr,
        grad_v_ptr,
        0,
        tokens,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim);
}

}  // extern "C"
