// ppmd_cuda.hpp — CUDA runtime probe declarations for the _ppmd_cuda backend.
//
// All functions are host-callable.  Implementations are in cuda_probe.cu.
// Safe defaults are returned if CUDA is unavailable or any call fails.

#pragma once

#include <string>

namespace ppmd_cuda {

// Returns true if at least one CUDA device is available.
bool cuda_available();

// Number of visible CUDA devices (0 if none or on error).
int cuda_device_count();

// cudaRuntimeGetVersion result (e.g. 12080 for CUDA 12.8); 0 on error.
int cuda_runtime_version();

// cudaDriverGetVersion result; 0 on error.
int cuda_driver_version();

// Human-readable device name from cudaGetDeviceProperties.
// Returns "" if device index is out of range or any call fails.
std::string cuda_device_name(int device);

// Compute capability major version; 0 on error.
int cuda_compute_capability_major(int device);

// Compute capability minor version; 0 on error.
int cuda_compute_capability_minor(int device);

}  // namespace ppmd_cuda
