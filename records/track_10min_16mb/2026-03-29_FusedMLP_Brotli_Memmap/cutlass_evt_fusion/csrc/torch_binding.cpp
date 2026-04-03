// PyTorch C++ extension: CUTLASS EVT fused GEMM * elementwise multiply
// dpre = (go @ down_w.T) * act_grad
// Pass down_w directly (K, N) — NOT down_w.T.contiguous()

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void launch_gemm_mul(
    void const*, void const*, void const*, void*, int, int, int, cudaStream_t);

at::Tensor gemm_mul(at::Tensor go, at::Tensor down_w, at::Tensor act_grad) {
    TORCH_CHECK(go.is_cuda() && go.is_contiguous());
    TORCH_CHECK(down_w.is_cuda() && down_w.is_contiguous());
    TORCH_CHECK(act_grad.is_cuda() && act_grad.is_contiguous());
    TORCH_CHECK(go.scalar_type() == at::kBFloat16);
    TORCH_CHECK(down_w.scalar_type() == at::kBFloat16);
    TORCH_CHECK(act_grad.scalar_type() == at::kBFloat16);

    int M = go.size(0);
    int K = go.size(1);
    int N = down_w.size(1);  // down_w is (K, N) row-major

    TORCH_CHECK(down_w.size(0) == K,
        "K mismatch: go has K=", K, " but down_w has size(0)=", down_w.size(0));
    TORCH_CHECK(act_grad.size(0) == M && act_grad.size(1) == N,
        "act_grad shape must be (M, N), got (", act_grad.size(0), ", ", act_grad.size(1), ")");

    at::Tensor dpre = at::empty({M, N}, go.options());

    launch_gemm_mul(
        go.data_ptr(), down_w.data_ptr(), act_grad.data_ptr(), dpre.data_ptr(),
        M, N, K,
        at::cuda::getCurrentCUDAStream());

    return dpre;
}

TORCH_LIBRARY(cutlass_evt, m) {
    m.def("gemm_mul(Tensor go, Tensor down_w, Tensor act_grad) -> Tensor");
}

TORCH_LIBRARY_IMPL(cutlass_evt, CUDA, m) {
    m.impl("gemm_mul", &gemm_mul);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
