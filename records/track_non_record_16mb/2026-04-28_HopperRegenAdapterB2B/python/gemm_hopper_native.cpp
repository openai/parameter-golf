#include <torch/extension.h>

#include "gemm_hopper/reference.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

void gemm_hopper_native_adapter_forward_cuda(const torch::Tensor& x,
                                             const torch::Tensor& a,
                                             const torch::Tensor& b,
                                             torch::Tensor& z,
                                             torch::Tensor& out,
                                             int64_t M,
                                             int64_t N,
                                             int64_t K,
                                             int64_t r);

namespace {

void check_cuda(cudaError_t err, const char* where) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(where) + " failed: " + cudaGetErrorString(err));
  }
}

void require_sm90(const torch::Tensor& x) {
  const c10::cuda::CUDAGuard guard(x.device());
  int device = x.get_device();
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  if (prop.major < 9) {
    throw std::runtime_error(
        "GEMM Hopper native backend requires an SM90/Hopper CUDA device; current device is " +
        std::to_string(prop.major) + "." + std::to_string(prop.minor));
  }
}

torch::Tensor copy_ternary_to_device(std::uint64_t seed,
                                     int64_t rows,
                                     int64_t cols,
                                     const torch::Tensor& like) {
  std::vector<std::int8_t> host(static_cast<std::size_t>(rows * cols));
  if (rows > 0 && cols > 0) {
    gemm_hopper::generate_ternary_matrix(
        seed,
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols),
        host.data());
  }

  auto options = torch::TensorOptions().device(like.device()).dtype(torch::kInt8);
  torch::Tensor out = torch::empty({rows, cols}, options);
  if (!host.empty()) {
    const c10::cuda::CUDAGuard guard(like.device());
    check_cuda(cudaMemcpy(
                   out.data_ptr<std::int8_t>(),
                   host.data(),
                   host.size() * sizeof(std::int8_t),
                   cudaMemcpyHostToDevice),
               "cudaMemcpy ternary H2D");
  }
  return out;
}

} // namespace

torch::Tensor adapter_forward(torch::Tensor x,
                              std::uint64_t seed_A,
                              std::uint64_t seed_B,
                              int64_t N,
                              int64_t K,
                              int64_t r) {
  TORCH_CHECK(x.is_cuda(), "adapter_forward expects a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "adapter_forward expects flattened 2D X");
  TORCH_CHECK(x.is_contiguous(), "adapter_forward expects contiguous X");
  TORCH_CHECK(K >= 0 && N >= 0 && r >= 0, "adapter_forward got negative shape");
  TORCH_CHECK(x.size(1) == K,
              "adapter_forward K mismatch: X.shape[1]=",
              x.size(1),
              " K=",
              K);
  TORCH_CHECK(x.scalar_type() == at::kFloat ||
                  x.scalar_type() == at::kBFloat16 ||
                  x.scalar_type() == at::kHalf,
              "adapter_forward supports float32, bfloat16, and float16 X");

  require_sm90(x);
  const c10::cuda::CUDAGuard guard(x.device());

  const int64_t M = x.size(0);
  torch::Tensor out = torch::empty({M, N}, x.options());
  if (M == 0 || N == 0) {
    return out;
  }
  if (K == 0 || r == 0) {
    return torch::zeros({M, N}, x.options());
  }

  torch::Tensor A = copy_ternary_to_device(seed_A, r, K, x);
  torch::Tensor B = copy_ternary_to_device(seed_B, N, r, x);
  torch::Tensor Z = torch::empty({M, r}, x.options().dtype(at::kFloat));

  gemm_hopper_native_adapter_forward_cuda(x, A, B, Z, out, M, N, K, r);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adapter_forward",
        &adapter_forward,
        "Seed-based Hopper regenerated adapter forward: (X @ A.T) @ B.T");
}
