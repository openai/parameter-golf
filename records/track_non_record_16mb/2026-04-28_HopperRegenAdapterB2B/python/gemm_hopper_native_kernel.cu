#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float read_value(const float* ptr) {
  return *ptr;
}

__device__ __forceinline__ float read_value(const __nv_bfloat16* ptr) {
  return __bfloat162float(*ptr);
}

__device__ __forceinline__ float read_value(const __half* ptr) {
  return __half2float(*ptr);
}

__device__ __forceinline__ void write_value(float* ptr, float value) {
  *ptr = value;
}

__device__ __forceinline__ void write_value(__nv_bfloat16* ptr, float value) {
  *ptr = __float2bfloat16(value);
}

__device__ __forceinline__ void write_value(__half* ptr, float value) {
  *ptr = __float2half_rn(value);
}

template <typename XType>
__global__ void adapter_z_kernel(int64_t M,
                                 int64_t K,
                                 int64_t r,
                                 const XType* X,
                                 const std::int8_t* A,
                                 float* Z) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = M * r;
  if (idx >= total) return;

  const int64_t m = idx / r;
  const int64_t j = idx - m * r;
  const XType* x_row = X + m * K;
  const std::int8_t* a_row = A + j * K;

  float acc = 0.0f;
  for (int64_t k = 0; k < K; ++k) {
    acc += read_value(x_row + k) * static_cast<float>(a_row[k]);
  }
  Z[idx] = acc;
}

template <typename OutType>
__global__ void adapter_y_kernel(int64_t M,
                                 int64_t N,
                                 int64_t r,
                                 const float* Z,
                                 const std::int8_t* B,
                                 OutType* Y) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = M * N;
  if (idx >= total) return;

  const int64_t m = idx / N;
  const int64_t n = idx - m * N;
  const float* z_row = Z + m * r;
  const std::int8_t* b_row = B + n * r;

  float acc = 0.0f;
  for (int64_t j = 0; j < r; ++j) {
    acc += z_row[j] * static_cast<float>(b_row[j]);
  }
  write_value(Y + idx, acc);
}

template <typename XType, typename OutType>
void launch_typed(const torch::Tensor& x,
                  const torch::Tensor& a,
                  const torch::Tensor& b,
                  torch::Tensor& z,
                  torch::Tensor& out,
                  int64_t M,
                  int64_t N,
                  int64_t K,
                  int64_t r) {
  constexpr int block = 256;
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const int z_grid = static_cast<int>((M * r + block - 1) / block);
  const int y_grid = static_cast<int>((M * N + block - 1) / block);

  adapter_z_kernel<<<z_grid, block, 0, stream.stream()>>>(
      M,
      K,
      r,
      reinterpret_cast<const XType*>(x.data_ptr()),
      a.data_ptr<std::int8_t>(),
      z.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  adapter_y_kernel<<<y_grid, block, 0, stream.stream()>>>(
      M,
      N,
      r,
      z.data_ptr<float>(),
      b.data_ptr<std::int8_t>(),
      reinterpret_cast<OutType*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

void gemm_hopper_native_adapter_forward_cuda(const torch::Tensor& x,
                                             const torch::Tensor& a,
                                             const torch::Tensor& b,
                                             torch::Tensor& z,
                                             torch::Tensor& out,
                                             int64_t M,
                                             int64_t N,
                                             int64_t K,
                                             int64_t r) {
  if (x.scalar_type() == at::kFloat) {
    launch_typed<float, float>(x, a, b, z, out, M, N, K, r);
  } else if (x.scalar_type() == at::kBFloat16) {
    launch_typed<__nv_bfloat16, __nv_bfloat16>(x, a, b, z, out, M, N, K, r);
  } else if (x.scalar_type() == at::kHalf) {
    launch_typed<__half, __half>(x, a, b, z, out, M, N, K, r);
  } else {
    TORCH_CHECK(false, "unsupported native adapter dtype");
  }
}
