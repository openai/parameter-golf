"""Experimental CUDA kernels for packed train-time ternary linear layers."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

extern "C" void pack_ternary_weight_launch(
    const void* weight,
    int dtype_code,
    int32_t* packed,
    float* scales,
    int64_t rows,
    int64_t cols,
    int64_t group_size,
    cudaStream_t stream);

extern "C" void materialize_ternary_weight_launch(
    const void* weight,
    void* out,
    int dtype_code,
    int64_t rows,
    int64_t cols,
    int64_t group_size,
    cudaStream_t stream);

extern "C" void packed_ternary_matmul_launch(
    const void* x,
    const int32_t* packed,
    const float* scales,
    void* out,
    int dtype_code,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t words_per_row,
    int64_t groups_per_row,
    int64_t group_size,
    cudaStream_t stream);

extern "C" void packed_ternary_input_grad_launch(
    const void* grad_out,
    const int32_t* packed,
    const float* scales,
    void* grad_x,
    int dtype_code,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t words_per_row,
    int64_t groups_per_row,
    int64_t group_size,
    cudaStream_t stream);

std::vector<torch::Tensor> pack_ternary_weight(torch::Tensor weight, int64_t group_size) {
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(group_size % 16 == 0, "group_size must be a multiple of 16");
  TORCH_CHECK(group_size <= 1024, "group_size must be <= 1024");
  TORCH_CHECK(
      weight.scalar_type() == torch::kFloat16 ||
      weight.scalar_type() == torch::kBFloat16 ||
      weight.scalar_type() == torch::kFloat32,
      "weight must be float16, bfloat16, or float32");
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);
  const int64_t words_per_row = (k + 15) / 16;
  const int64_t groups_per_row = (k + group_size - 1) / group_size;
  auto packed = torch::empty({n, words_per_row}, weight.options().dtype(torch::kInt32));
  auto scales = torch::empty({n, groups_per_row}, weight.options().dtype(torch::kFloat32));
  const int dtype_code = weight.scalar_type() == torch::kFloat16 ? 0 : (weight.scalar_type() == torch::kBFloat16 ? 2 : 1);
  pack_ternary_weight_launch(
      weight.data_ptr(),
      dtype_code,
      packed.data_ptr<int32_t>(),
      scales.data_ptr<float>(),
      n,
      k,
      group_size,
      at::cuda::getCurrentCUDAStream().stream());
  return {packed, scales};
}

torch::Tensor materialize_ternary_weight(torch::Tensor weight, int64_t group_size) {
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(group_size <= 1024, "group_size must be <= 1024");
  TORCH_CHECK(
      weight.scalar_type() == torch::kFloat16 ||
      weight.scalar_type() == torch::kBFloat16 ||
      weight.scalar_type() == torch::kFloat32,
      "weight must be float16, bfloat16, or float32");
  auto out = torch::empty_like(weight);
  const int dtype_code = weight.scalar_type() == torch::kFloat16 ? 0 : (weight.scalar_type() == torch::kBFloat16 ? 2 : 1);
  materialize_ternary_weight_launch(
      weight.data_ptr(),
      out.data_ptr(),
      dtype_code,
      weight.size(0),
      weight.size(1),
      group_size,
      at::cuda::getCurrentCUDAStream().stream());
  return out;
}

torch::Tensor packed_ternary_matmul(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t k,
    int64_t group_size) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(packed.is_cuda(), "packed must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(packed.dim() == 2, "packed must be 2D");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2D");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(x.size(1) == k, "x width does not match k");
  TORCH_CHECK(packed.scalar_type() == torch::kInt32, "packed must be int32");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat16 ||
      x.scalar_type() == torch::kBFloat16 ||
      x.scalar_type() == torch::kFloat32,
      "x must be float16, bfloat16, or float32");
  const int64_t n = packed.size(0);
  const int64_t words_per_row = packed.size(1);
  const int64_t groups_per_row = scales.size(1);
  const int dtype_code = x.scalar_type() == torch::kFloat16 ? 0 : (x.scalar_type() == torch::kBFloat16 ? 2 : 1);
  auto out = torch::empty({x.size(0), n}, x.options());
  packed_ternary_matmul_launch(
      x.data_ptr(),
      packed.data_ptr<int32_t>(),
      scales.data_ptr<float>(),
      out.data_ptr(),
      dtype_code,
      x.size(0),
      n,
      k,
      words_per_row,
      groups_per_row,
      group_size,
      at::cuda::getCurrentCUDAStream().stream());
  return out;
}

torch::Tensor packed_ternary_input_grad(
    torch::Tensor grad_out,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t k,
    int64_t group_size) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
  TORCH_CHECK(packed.is_cuda(), "packed must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2D");
  TORCH_CHECK(packed.dim() == 2, "packed must be 2D");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2D");
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(grad_out.size(1) == packed.size(0), "grad_out width must match packed rows");
  TORCH_CHECK(packed.scalar_type() == torch::kInt32, "packed must be int32");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      grad_out.scalar_type() == torch::kFloat16 ||
      grad_out.scalar_type() == torch::kBFloat16 ||
      grad_out.scalar_type() == torch::kFloat32,
      "grad_out must be float16, bfloat16, or float32");
  const int64_t n = packed.size(0);
  const int64_t words_per_row = packed.size(1);
  const int64_t groups_per_row = scales.size(1);
  const int dtype_code = grad_out.scalar_type() == torch::kFloat16 ? 0 : (grad_out.scalar_type() == torch::kBFloat16 ? 2 : 1);
  auto grad_x = torch::empty({grad_out.size(0), k}, grad_out.options());
  packed_ternary_input_grad_launch(
      grad_out.data_ptr(),
      packed.data_ptr<int32_t>(),
      scales.data_ptr<float>(),
      grad_x.data_ptr(),
      dtype_code,
      grad_out.size(0),
      n,
      k,
      words_per_row,
      groups_per_row,
      group_size,
      at::cuda::getCurrentCUDAStream().stream());
  return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_ternary_weight", &pack_ternary_weight, "pack ternary weight");
  m.def("materialize_ternary_weight", &materialize_ternary_weight, "materialize ternary weight");
  m.def("packed_ternary_matmul", &packed_ternary_matmul, "packed ternary matmul");
  m.def("packed_ternary_input_grad", &packed_ternary_input_grad, "packed ternary input grad");
}
"""


CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ __half float_to_scalar<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename scalar_t>
__device__ __forceinline__ float round_scale_to_work_dtype(float value) {
  return value;
}

template <>
__device__ __forceinline__ float round_scale_to_work_dtype<__half>(float value) {
  return __half2float(__float2half_rn(value));
}

template <>
__device__ __forceinline__ float round_scale_to_work_dtype<__nv_bfloat16>(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ int ternary_code(float value, float scale) {
  const float cutoff = 0.5f * scale;
  if (value > cutoff) {
    return 1;
  }
  if (value < -cutoff) {
    return 2;
  }
  return 0;
}

__device__ __forceinline__ float code_to_sign(int code) {
  return code == 1 ? 1.0f : (code == 2 ? -1.0f : 0.0f);
}

template <typename scalar_t>
__global__ void pack_ternary_weight_kernel(
    const scalar_t* __restrict__ weight,
    int32_t* __restrict__ packed,
    float* __restrict__ scales,
    int rows,
    int cols,
    int words_per_row,
    int groups_per_row,
    int group_size) {
  const int block = blockIdx.x;
  const int row = block / groups_per_row;
  const int group = block - row * groups_per_row;
  const int tid = threadIdx.x;
  const int group_start = group * group_size;
  const int group_len = max(0, min(group_size, cols - group_start));

  extern __shared__ float shared[];
  float local_abs = 0.0f;
  if (tid < group_len) {
    local_abs = fabsf(scalar_to_float(weight[row * cols + group_start + tid]));
  }
  shared[tid] = local_abs;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float raw_scale = fmaxf(shared[0] / fmaxf((float)group_size, 1.0f), 1.0e-8f);
  const float scale = round_scale_to_work_dtype<scalar_t>(raw_scale);
  if (tid == 0) {
    scales[row * groups_per_row + group] = scale;
  }
  __syncthreads();

  const int words_in_group = (group_len + 15) / 16;
  if (tid < words_in_group) {
    const int word_col = group_start + tid * 16;
    uint32_t bits = 0;
    #pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const int col = word_col + lane;
      int code = 0;
      if (col < cols && col < group_start + group_len) {
        const float v = scalar_to_float(weight[row * cols + col]);
        code = ternary_code(v, scale);
      }
      bits |= ((uint32_t)code) << (2 * lane);
    }
    packed[row * words_per_row + (word_col >> 4)] = (int32_t)bits;
  }
}

template <typename scalar_t>
__global__ void packed_ternary_matmul_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ packed,
    const float* __restrict__ scales,
    scalar_t* __restrict__ out,
    int m,
    int n,
    int k,
    int words_per_row,
    int groups_per_row,
    int group_size) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int word_idx = 0; word_idx < words_per_row; ++word_idx) {
    const uint32_t bits = (uint32_t)packed[col * words_per_row + word_idx];
    const int base_k = word_idx << 4;
    #pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const int kk = base_k + lane;
      if (kk < k) {
        const int code = (bits >> (2 * lane)) & 3;
        if (code != 0) {
          const int group = min(kk / group_size, groups_per_row - 1);
          const float w = code_to_sign(code) * scales[col * groups_per_row + group];
          acc += scalar_to_float(x[row * k + kk]) * w;
        }
      }
    }
  }
  out[row * n + col] = float_to_scalar<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void materialize_ternary_weight_kernel(
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int rows,
    int cols,
    int groups_per_row,
    int group_size) {
  const int block = blockIdx.x;
  const int row = block / groups_per_row;
  const int group = block - row * groups_per_row;
  const int tid = threadIdx.x;
  const int group_start = group * group_size;
  const int group_len = max(0, min(group_size, cols - group_start));

  extern __shared__ float shared[];
  float local_abs = 0.0f;
  if (tid < group_len) {
    local_abs = fabsf(scalar_to_float(weight[row * cols + group_start + tid]));
  }
  shared[tid] = local_abs;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float raw_scale = fmaxf(shared[0] / fmaxf((float)group_size, 1.0f), 1.0e-8f);
  const float scale = round_scale_to_work_dtype<scalar_t>(raw_scale);
  if (tid < group_len) {
    const int col = group_start + tid;
    const float v = scalar_to_float(weight[row * cols + col]);
    const int code = ternary_code(v, scale);
    out[row * cols + col] = float_to_scalar<scalar_t>(code_to_sign(code) * scale);
  }
}

template <typename scalar_t>
__global__ void packed_ternary_input_grad_kernel(
    const scalar_t* __restrict__ grad_out,
    const int32_t* __restrict__ packed,
    const float* __restrict__ scales,
    scalar_t* __restrict__ grad_x,
    int m,
    int n,
    int k,
    int words_per_row,
    int groups_per_row,
    int group_size) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int kk = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || kk >= k) {
    return;
  }

  const int word_idx = kk >> 4;
  const int lane = kk & 15;
  const int group = min(kk / group_size, groups_per_row - 1);
  float acc = 0.0f;
  for (int col = 0; col < n; ++col) {
    const uint32_t bits = (uint32_t)packed[col * words_per_row + word_idx];
    const int code = (bits >> (2 * lane)) & 3;
    if (code != 0) {
      const float w = code_to_sign(code) * scales[col * groups_per_row + group];
      acc += scalar_to_float(grad_out[row * n + col]) * w;
    }
  }
  grad_x[row * k + kk] = float_to_scalar<scalar_t>(acc);
}

static int next_power_of_two(int value) {
  int out = 1;
  while (out < value) {
    out <<= 1;
  }
  return out;
}

template <typename scalar_t>
static void pack_ternary_weight_launch_typed(
    const void* weight,
    int32_t* packed,
    float* scales,
    int rows,
    int cols,
    int group_size,
    cudaStream_t stream) {
  const int words_per_row = (cols + 15) / 16;
  const int groups_per_row = (cols + group_size - 1) / group_size;
  const int threads = next_power_of_two((int)group_size);
  const int blocks = rows * groups_per_row;
  pack_ternary_weight_kernel<scalar_t><<<blocks, threads, threads * sizeof(float), stream>>>(
        static_cast<const scalar_t*>(weight),
        packed,
        scales,
        rows,
        cols,
        words_per_row,
        groups_per_row,
        (int)group_size);
}

template <typename scalar_t>
static void materialize_ternary_weight_launch_typed(
    const void* weight,
    void* out,
    int rows,
    int cols,
    int group_size,
    cudaStream_t stream) {
  const int groups_per_row = (cols + group_size - 1) / group_size;
  const int threads = next_power_of_two((int)group_size);
  const int blocks = rows * groups_per_row;
  materialize_ternary_weight_kernel<scalar_t><<<blocks, threads, threads * sizeof(float), stream>>>(
      static_cast<const scalar_t*>(weight),
      static_cast<scalar_t*>(out),
      rows,
      cols,
      groups_per_row,
      (int)group_size);
}

extern "C" void materialize_ternary_weight_launch(
    const void* weight,
    void* out,
    int dtype_code,
    int64_t rows,
    int64_t cols,
    int64_t group_size,
    cudaStream_t stream) {
  if (dtype_code == 0) {
    materialize_ternary_weight_launch_typed<__half>(
        weight, out, (int)rows, (int)cols, (int)group_size, stream);
  } else if (dtype_code == 2) {
    materialize_ternary_weight_launch_typed<__nv_bfloat16>(
        weight, out, (int)rows, (int)cols, (int)group_size, stream);
  } else {
    materialize_ternary_weight_launch_typed<float>(
        weight, out, (int)rows, (int)cols, (int)group_size, stream);
  }
}

extern "C" void packed_ternary_matmul_launch(
    const void* x,
    const int32_t* packed,
    const float* scales,
    void* out,
    int dtype_code,
    int64_t m_raw,
    int64_t n_raw,
    int64_t k,
    int64_t words_per_row_raw,
    int64_t groups_per_row_raw,
    int64_t group_size,
    cudaStream_t stream) {
  const int m = (int)m_raw;
  const int n = (int)n_raw;
  const int words_per_row = (int)words_per_row_raw;
  const int groups_per_row = (int)groups_per_row_raw;
  const dim3 threads(16, 16);
  const dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
  if (dtype_code == 0) {
    packed_ternary_matmul_kernel<__half><<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(x),
        packed,
        scales,
        static_cast<__half*>(out),
        m,
        n,
        (int)k,
        words_per_row,
        groups_per_row,
        (int)group_size);
  } else if (dtype_code == 2) {
    packed_ternary_matmul_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        packed,
        scales,
        static_cast<__nv_bfloat16*>(out),
        m,
        n,
        (int)k,
        words_per_row,
        groups_per_row,
        (int)group_size);
  } else {
    packed_ternary_matmul_kernel<float><<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(x),
        packed,
        scales,
        static_cast<float*>(out),
        m,
        n,
        (int)k,
        words_per_row,
        groups_per_row,
        (int)group_size);
  }
}

extern "C" void packed_ternary_input_grad_launch(
    const void* grad_out,
    const int32_t* packed,
    const float* scales,
    void* grad_x,
    int dtype_code,
    int64_t m_raw,
    int64_t n_raw,
    int64_t k,
    int64_t words_per_row_raw,
    int64_t groups_per_row_raw,
    int64_t group_size,
    cudaStream_t stream) {
  const int m = (int)m_raw;
  const int n = (int)n_raw;
  const int words_per_row = (int)words_per_row_raw;
  const int groups_per_row = (int)groups_per_row_raw;
  const dim3 threads(16, 16);
  const dim3 blocks(((int)k + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
  if (dtype_code == 0) {
    packed_ternary_input_grad_kernel<__half><<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(grad_out),
        packed,
        scales,
        static_cast<__half*>(grad_x),
        m,
        n,
        (int)k,
        words_per_row,
        groups_per_row,
        (int)group_size);
  } else if (dtype_code == 2) {
    packed_ternary_input_grad_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(grad_out),
        packed,
        scales,
        static_cast<__nv_bfloat16*>(grad_x),
        m,
        n,
        (int)k,
        words_per_row,
        groups_per_row,
        (int)group_size);
  } else {
    packed_ternary_input_grad_kernel<float><<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(grad_out),
        packed,
        scales,
        static_cast<float*>(grad_x),
        m,
        n,
        (int)k,
        words_per_row,
        groups_per_row,
        (int)group_size);
  }
}

extern "C" void pack_ternary_weight_launch(
    const void* weight,
    int dtype_code,
    int32_t* packed,
    float* scales,
    int64_t rows,
    int64_t cols,
    int64_t group_size,
    cudaStream_t stream) {
  if (dtype_code == 0) {
    pack_ternary_weight_launch_typed<__half>(
        weight, packed, scales, (int)rows, (int)cols, (int)group_size, stream);
  } else if (dtype_code == 2) {
    pack_ternary_weight_launch_typed<__nv_bfloat16>(
        weight, packed, scales, (int)rows, (int)cols, (int)group_size, stream);
  } else {
    pack_ternary_weight_launch_typed<float>(
        weight, packed, scales, (int)rows, (int)cols, (int)group_size, stream);
  }
}
"""


@lru_cache(maxsize=1)
def _load_extension():
    try:
        from tools.cuda126_env import configure_cuda126_env

        configure_cuda126_env()
    except Exception:
        force_packed = os.environ.get("TRAIN_TERNARY_PACKED_KERNEL", "0").strip().lower() in {"1", "true", "yes"}
        force_dense = os.environ.get("TRAIN_TERNARY_DENSE_KERNEL", "0").strip().lower() in {"1", "true", "yes"}
        if force_packed or force_dense:
            raise
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
    build_dir = ROOT / "tmp_cuda_extensions" / "packed_ternary"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name="packed_ternary_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        build_directory=str(build_dir),
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=bool(int(os.environ.get("TRAIN_TERNARY_PACKED_VERBOSE", "0"))),
    )


def pack_ternary_weight(weight: Tensor, group_size: int) -> tuple[Tensor, Tensor]:
    ext = _load_extension()
    packed, scales = ext.pack_ternary_weight(weight.contiguous(), int(group_size))
    return packed, scales


class _DenseTernaryWeightFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, group_size: int) -> Tensor:
        ext = _load_extension()
        ctx.input_shape = tuple(weight.shape)
        return ext.materialize_ternary_weight(weight.contiguous(), int(group_size))

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        return grad_out.reshape(ctx.input_shape), None


def dense_ternary_weight(weight: Tensor, group_size: int, work_dtype: torch.dtype | None = None) -> Tensor:
    if work_dtype is not None and weight.dtype != work_dtype:
        weight = weight.to(dtype=work_dtype)
    return _DenseTernaryWeightFn.apply(weight, int(group_size))


class _PackedTernaryLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, packed: Tensor, scales: Tensor, group_size: int) -> Tensor:
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        ext = _load_extension()
        out_flat = ext.packed_ternary_matmul(x_flat, packed, scales, int(weight.shape[1]), int(group_size))
        ctx.save_for_backward(x_flat, packed, scales)
        ctx.group_size = int(group_size)
        ctx.input_shape = tuple(x.shape)
        ctx.weight_shape = tuple(weight.shape)
        return out_flat.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_flat, packed, scales = ctx.saved_tensors
        group_size = int(ctx.group_size)
        input_shape = tuple(ctx.input_shape)
        weight_shape = tuple(ctx.weight_shape)
        grad_out_flat = grad_out.reshape(-1, weight_shape[0]).contiguous()
        ext = _load_extension()
        grad_x_flat = ext.packed_ternary_input_grad(
            grad_out_flat,
            packed,
            scales,
            int(weight_shape[1]),
            group_size,
        )
        grad_weight = grad_out_flat.transpose(0, 1).matmul(x_flat)
        return grad_x_flat.reshape(input_shape), grad_weight, None, None, None


def packed_ternary_linear(
    x: Tensor,
    weight: Tensor,
    packed: Tensor,
    scales: Tensor,
    group_size: int,
) -> Tensor:
    return _PackedTernaryLinearFn.apply(x, weight, packed, scales, int(group_size))
