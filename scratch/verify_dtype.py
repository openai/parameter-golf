import torch
import triton_kernels
from triton_kernels import TritonFWHTFn, TritonScanFn

def test_dtype_preservation():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Triton dtype test.")
        return

    # 1. FWHT Dtype Test
    B, T, D = 1, 64, 128
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16).requires_grad_(True)
    H = torch.randn(T, T, device='cuda', dtype=torch.float32) # Matrix is FP32
    
    y = TritonFWHTFn.apply(x, H)
    assert y.dtype == torch.bfloat16
    
    y.sum().backward()
    assert x.grad.dtype == torch.bfloat16, f"FWHT grad dtype mismatch: expected bfloat16, got {x.grad.dtype}"
    print("FWHT Autograd Dtype: PASS")

    # 2. Scan Dtype Test
    B, T, S = 1, 32, 64
    bv = torch.randn(B, T, S, device='cuda', dtype=torch.bfloat16).requires_grad_(True)
    d = torch.sigmoid(torch.randn(B, T, S, device='cuda', dtype=torch.bfloat16)).requires_grad_(True)
    
    y = TritonScanFn.apply(bv, d)
    assert y.dtype == torch.bfloat16
    
    y.sum().backward()
    assert bv.grad.dtype == torch.bfloat16, f"Scan grad_bv dtype mismatch: expected bfloat16, got {bv.grad.dtype}"
    assert d.grad.dtype == torch.bfloat16, f"Scan grad_d dtype mismatch: expected bfloat16, got {d.grad.dtype}"
    print("Scan Autograd Dtype: PASS")

if __name__ == "__main__":
    test_dtype_preservation()
