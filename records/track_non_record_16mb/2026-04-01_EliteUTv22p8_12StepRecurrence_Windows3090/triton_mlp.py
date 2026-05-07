import torch
import torch._dynamo
import triton
import triton.language as tl

# print("[triton_mlp] V2 Robust Loaded (Fused Forward + Fused Backward)")

# ---------------------------------------------------------------------------
# FORWARD KERNEL: Fused MatMul + LeakyReLU^2
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_relu2_fwd_kernel(
    a_ptr, b_ptr, c_ptr, y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_a = (offs_am[:, None] < M) & (k * BLOCK_SIZE_K + offs_k[None, :] < K)
        mask_b = (k * BLOCK_SIZE_K + offs_k[:, None] < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.bfloat16)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # LeakyReLU(0.1)^2 activation
    # f(x) = (x if x > 0 else 0.1x)^2
    # Unifying with train_gpt.py slope = 0.1
    leaky_relu = tl.where(accumulator > 0.0, accumulator, 0.1 * accumulator)
    y2 = leaky_relu * leaky_relu
    
    # Grid locations for storage
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Store pre-activation for backward pass
    y_out_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
    tl.store(y_out_ptrs, accumulator.to(tl.bfloat16), mask=mask)

    # Store final activation
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, y2.to(tl.bfloat16), mask=mask)

# ---------------------------------------------------------------------------
# BACKWARD KERNEL: dX = (dLoss * 2 * relu(Y)) @ W.T
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64,  'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def fused_relu2_bwd_dx_kernel(
    grad_out_ptr, y_ptr, w_ptr, dx_ptr,
    M, K, N,
    stride_gom, stride_gon,
    stride_ym, stride_yn,
    stride_wk, stride_wn,
    stride_dxm, stride_dxk,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bk = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    go_ptrs = grad_out_ptr + (offs_am[:, None] * stride_gom + offs_n[None, :] * stride_gon)
    y_ptrs = y_ptr + (offs_am[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    w_ptrs = w_ptr + (offs_bk[None, :] * stride_wk + offs_n[:, None] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        mask_gn = (offs_am[:, None] < M) & (n * BLOCK_SIZE_N + offs_n[None, :] < N)
        mask_wn = (n * BLOCK_SIZE_N + offs_n[:, None] < N) & (offs_bk[None, :] < K)
        
        go = tl.load(go_ptrs, mask=mask_gn, other=0.0).to(tl.float32)
        y = tl.load(y_ptrs, mask=mask_gn, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask_wn, other=0.0).to(tl.float32)
        
        # Compute derivative: dy = f'(y) * grad_output
        # f'(y) = 2*y if y > 0 else 0.5*y
        dy = tl.where(y > 0.0, 2.0 * y * go, 0.02 * y * go)
        
        accumulator = tl.dot(dy.to(tl.bfloat16), w.to(tl.bfloat16), accumulator)
        go_ptrs += BLOCK_SIZE_N * stride_gon
        y_ptrs += BLOCK_SIZE_N * stride_yn
        w_ptrs += BLOCK_SIZE_N * stride_wn

    offs_dxm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_dxk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    out_ptrs = dx_ptr + offs_dxm[:, None] * stride_dxm + offs_dxk[None, :] * stride_dxk
    mask = (offs_dxm[:, None] < M) & (offs_dxk[None, :] < K)
    tl.store(out_ptrs, accumulator.to(tl.bfloat16), mask=mask)

# ---------------------------------------------------------------------------
# BACKWARD KERNEL: dW = X.T @ (dLoss * 2 * relu(Y))
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_K': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_M': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_relu2_bwd_dw_kernel(
    x_ptr, grad_out_ptr, y_ptr, dw_ptr,
    M, N, K,
    stride_xk, stride_xm,
    stride_gom, stride_gon,
    stride_ym, stride_yn,
    stride_dwk, stride_dwn,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_K * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_k = group_id * GROUP_SIZE_K
    group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE_K)
    pid_k = first_pid_k + (pid % group_size_k)
    pid_n = (pid % num_pid_in_group) // group_size_k

    offs_ck = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    x_ptrs = x_ptr + (offs_ck[:, None] * stride_xk + offs_m[None, :] * stride_xm)
    go_ptrs = grad_out_ptr + (offs_m[:, None] * stride_gom + offs_cn[None, :] * stride_gon)
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_cn[None, :] * stride_yn)

    accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        mask_xm = (offs_ck[:, None] < K) & (m * BLOCK_SIZE_M + offs_m[None, :] < M)
        mask_gm = (m * BLOCK_SIZE_M + offs_m[:, None] < M) & (offs_cn[None, :] < N)
        
        x = tl.load(x_ptrs, mask=mask_xm, other=0.0).to(tl.bfloat16)
        go = tl.load(go_ptrs, mask=mask_gm, other=0.0).to(tl.float32)
        y = tl.load(y_ptrs, mask=mask_gm, other=0.0).to(tl.float32)
        
        # Compute derivative: dy = f'(y) * grad_output
        # f'(y) = 2*y if y > 0 else 0.5*y
        dy = tl.where(y > 0.0, 2.0 * y * go, 0.02 * y * go)
        
        # Matrix multiply: grad_w += x.T @ dy
        accumulator = tl.dot(x, dy.to(tl.bfloat16), accumulator)
        x_ptrs += BLOCK_SIZE_M * stride_xm
        go_ptrs += BLOCK_SIZE_M * stride_gom
        y_ptrs += BLOCK_SIZE_M * stride_ym

    offs_dwk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_dwn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = dw_ptr + offs_dwk[:, None] * stride_dwk + offs_dwn[None, :] * stride_dwn
    mask = (offs_dwk[:, None] < K) & (offs_dwn[None, :] < N)
    tl.store(out_ptrs, accumulator.to(tl.bfloat16), mask=mask)

# ---------------------------------------------------------------------------
# AUTOGRAD FUNCTION
# ---------------------------------------------------------------------------
class FusedReLU2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        # Flatten arbitrary shapes [B, T, C] -> [B*T, C]
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])
        
        M, K = x.shape
        M, K = x.shape
        K_w, N = w.shape
        output = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
        y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16) # Pre-activation buffer
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        fused_relu2_fwd_kernel[grid](
            x, w, output, y,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            output.stride(0), output.stride(1),
            y.stride(0), y.stride(1),
        )
        
        # We save y directly (no redundant matmul here!)
        ctx.save_for_backward(x, w, y)
        ctx.orig_shape = orig_shape
        
        # Restore original leading dimensions
        if len(orig_shape) > 2:
            return output.reshape(*orig_shape[:-1], N)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w, y = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        
        # Flatten grad_output if needed
        N_dim = w.shape[1]
        grad_output = grad_output.reshape(-1, N_dim)
        
        M, K = x.shape
        K_w, N = w.shape
        
        # Compute dX
        grad_x = torch.empty_like(x)
        grid_dx = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']),)
        fused_relu2_bwd_dx_kernel[grid_dx](
            grad_output, y, w, grad_x,
            M, K, N,
            grad_output.stride(0), grad_output.stride(1),
            y.stride(0), y.stride(1),
            w.stride(0), w.stride(1),
            grad_x.stride(0), grad_x.stride(1),
        )
        
        # Compute dW
        grad_w = torch.empty_like(w)
        grid_dw = lambda META: (triton.cdiv(K, META['BLOCK_SIZE_K']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        fused_relu2_bwd_dw_kernel[grid_dw](
            x, grad_output, y, grad_w,
            M, N, K,
            x.stride(1), x.stride(0), # x.T stride
            grad_output.stride(0), grad_output.stride(1),
            y.stride(0), y.stride(1),
            grad_w.stride(0), grad_w.stride(1),
        )
        
        if len(orig_shape) > 2:
            return grad_x.reshape(orig_shape), grad_w
        return grad_x, grad_w

@torch._dynamo.disable
def fused_relu2(x, w):
    return FusedReLU2.apply(x, w)
