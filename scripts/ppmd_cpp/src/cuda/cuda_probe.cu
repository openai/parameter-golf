// cuda_probe.cu — CUDA runtime probe implementations for ppmd_cuda namespace.
//
// All functions are host-callable only (no __global__ kernels).
// Each CUDA API call is wrapped with an error check; on non-success, safe
// defaults are returned.  No exceptions are thrown across the CUDA boundary.

#include <cuda_runtime.h>

#include <string>

#include "ppmd_cuda.hpp"

namespace ppmd_cuda {

bool cuda_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
    return count >= 1;
}

int cuda_device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

int cuda_runtime_version() {
    int ver = 0;
    if (cudaRuntimeGetVersion(&ver) != cudaSuccess) return 0;
    return ver;
}

int cuda_driver_version() {
    int ver = 0;
    if (cudaDriverGetVersion(&ver) != cudaSuccess) return 0;
    return ver;
}

std::string cuda_device_name(int device) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return "";
    if (device < 0 || device >= count) return "";
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "";
    return std::string(prop.name);
}

int cuda_compute_capability_major(int device) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    if (device < 0 || device >= count) return 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
    return prop.major;
}

int cuda_compute_capability_minor(int device) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    if (device < 0 || device >= count) return 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
    return prop.minor;
}

}  // namespace ppmd_cuda
