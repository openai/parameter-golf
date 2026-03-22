import torch                                                                   
import torch.nn as nn                                                          
import torch.nn.functional as F                                                
import triton                                                                  
import triton.language as tl                                                   
                                                                               
                                                                               
@triton.autotune(                                                              
    configs=[                                                                  
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M'
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M'
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M'
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M'
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M'
    ],                                                                         
    key=['M', 'K'],                                                            
)                                                                              
@triton.jit                                                                    
def fused_qkv_gemm_kernel_nomask(                                              
    a_ptr, wq_ptr, wk_ptr, wv_ptr, c_ptr,                                      
    M, K: tl.constexpr, Nq: tl.constexpr, Nk: tl.constexpr, N: tl.constexpr,   
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,       
    GROUP_M: tl.constexpr,                                                     
):                                                                             
    pid = tl.program_id(0)                                                     
                                                                               
    grid_m = tl.cdiv(M, BLOCK_M)                                               
    grid_n = tl.cdiv(N, BLOCK_N)                                               
                                                                               
    # Swizzled scheduling for better L2 cache reuse across shared sequence weig
    num_pid_in_group = GROUP_M * grid_n                                        
    group_id = pid // num_pid_in_group                                         
    first_pid_m = group_id * GROUP_M                                           
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)                   
                                                                               
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)            
    pid_n = (pid % num_pid_in_group) // group_size_m                           
                                                                               
    n_start = pid_n * BLOCK_N                                                  
                                                                               
    # Zero-cost dynamic pointer routing completely bypassing host weight concat
    if n_start < Nq:                                                           
        w_ptr = wq_ptr                                                         
        n_local_start = n_start                                                
    elif n_start < Nq + Nk:                                                    
        w_ptr = wk_ptr                                                         
        n_local_start = n_start - Nq                                           
    else:                                                                      
        w_ptr = wv_ptr                                                         
        n_local_start = n_start - (Nq + Nk)                                    
                                                                               
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)                           
    offs_n_local = n_local_start + tl.arange(0, BLOCK_N)                       
    offs_k = tl.arange(0, BLOCK_K)                                             
                                                                               
    # Hardcoded continuous strides enforce dense loads and massively reduce reg
    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])                   
    w_ptrs = w_ptr + (offs_n_local[:, None] * K + offs_k[None, :])             
                                                                               
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)                       
                                                                               
    # Totally unrolled, bounds-check-free inner loop for absolute maximum compu
    for k in range(0, K, BLOCK_K):                                             
        a = tl.load(a_ptrs)                                                    
                                                                               
        # Vectorized loading of contiguous HBM memory lines transposes perfectl
        w = tl.load(w_ptrs)                                                    
        b = tl.trans(w)                                                        
                                                                               
        acc += tl.dot(a, b)                                                    
                                                                               
        a_ptrs += BLOCK_K                                                      
        w_ptrs += BLOCK_K                                                      
                                                                               
    offs_n_out = n_start + tl.arange(0, BLOCK_N)                               
    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n_out[None, :])               
                                                                               
    tl.store(c_ptrs, acc.to(tl.bfloat16))                                      
                                                                               
                                                                               
@triton.autotune(                                                              
    configs=[                                                                  
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M'
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M'
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M'
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M'
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M'
    ],                                                                         
    key=['M', 'K'],                                                            
)                                                                              
@triton.jit                                                                    
def fused_qkv_gemm_kernel_mmask(                                               
    a_ptr, wq_ptr, wk_ptr, wv_ptr, c_ptr,                                      
    M, K: tl.constexpr, Nq: tl.constexpr, Nk: tl.constexpr, N: tl.constexpr,   
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,       
    GROUP_M: tl.constexpr,                                                     
):                                                                             
    pid = tl.program_id(0)                                                     
                                                                               
    grid_m = tl.cdiv(M, BLOCK_M)                                               
    grid_n = tl.cdiv(N, BLOCK_N)                                               
                                                                               
    num_pid_in_group = GROUP_M * grid_n                                        
    group_id = pid // num_pid_in_group                                         
    first_pid_m = group_id * GROUP_M                                           
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)                   
                                                                               
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)            
    pid_n = (pid % num_pid_in_group) // group_size_m                           
                                                                               
    n_start = pid_n * BLOCK_N                                                  
                                                                               
    if n_start < Nq:                                                           
        w_ptr = wq_ptr                                                         
        n_local_start = n_start                                                
    elif n_start < Nq + Nk:                                                    
        w_ptr = wk_ptr                                                         
        n_local_start = n_start - Nq                                           
    else:                                                                      
        w_ptr = wv_ptr                                                         
        n_local_start = n_start - (Nq + Nk)                                    
                                                                               
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)                           
    offs_n_local = n_local_start + tl.arange(0, BLOCK_N)                       
    offs_k = tl.arange(0, BLOCK_K)                                             
                                                                               
    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])                   
    w_ptrs = w_ptr + (offs_n_local[:, None] * K + offs_k[None, :])             
                                                                               
    m_mask = offs_m < M                                                        
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)                       
                                                                               
    for k in range(0, K, BLOCK_K):                                             
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)                   
                                                                               
        w = tl.load(w_ptrs)                                                    
        b = tl.trans(w)                                                        
                                                                               
        acc += tl.dot(a, b)                                                    
                                                                               
        a_ptrs += BLOCK_K                                                      
        w_ptrs += BLOCK_K                                                      
                                                                               
    offs_n_out = n_start + tl.arange(0, BLOCK_N)                               
    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n_out[None, :])               
                                                                               
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])                
                                                                               
                                                                               
@triton.autotune(                                                              
    configs=[                                                                  
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M'
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M'
    ],                                                                         
    key=['M', 'K'],                                                            
)                                                                              
@triton.jit                                                                    
def fused_qkv_fallback_kernel(                                                 
    a_ptr, b_ptr, c_ptr,                                                       
    M, K, N,                                                                   
    stride_am, stride_ak,                                                      
    stride_bn, stride_bk,                                                      
    stride_cm, stride_cn,                                                      
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,       
    GROUP_M: tl.constexpr,                                                     
):                                                                             
    pid = tl.program_id(0)                                                     
    grid_m = tl.cdiv(M, BLOCK_M)                                               
    grid_n = tl.cdiv(N, BLOCK_N)                                               
                                                                               
    num_pid_in_group = GROUP_M * grid_n                                        
    group_id = pid // num_pid_in_group                                         
    first_pid_m = group_id * GROUP_M                                           
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)                   
                                                                               
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)            
    pid_n = (pid % num_pid_in_group) // group_size_m                           
                                                                               
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)                           
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)                           
    offs_k = tl.arange(0, BLOCK_K)                                             
                                                                               
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
                                                                               
    m_mask = offs_m < M                                                        
    n_mask = offs_n < N                                                        
                                                                               
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)                       
                                                                               
    for k in range(0, K, BLOCK_K):                                             
        k_mask = (k + offs_k) < K                                              
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0) 
                                                                               
        b_load = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=
        b = tl.trans(b_load)                                                   
                                                                               
        acc += tl.dot(a, b)                                                    
                                                                               
        a_ptrs += BLOCK_K * stride_ak                                          
        b_ptrs += BLOCK_K * stride_bk                                          
                                                                               
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :
                                                                               
                                                                               
class ModelNew(nn.Module):                                                     
    """                                                                        
    Ultra-optimized Fused RMSNorm + Q/K/V linear projections for GQA attention.
    """                                                                        
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):           
        super(ModelNew, self).__init__()                                       
        self.dim = dim                                                         
        self.num_heads = num_heads                                             
        self.num_kv_heads = num_kv_heads                                       
        self.head_dim = dim // num_heads                                       
        self.kv_dim = num_kv_heads * self.head_dim                             
                                                                               
        self.w_q = nn.Parameter(torch.randn(dim, dim, dtype=torch.bfloat16))   
        self.w_k = nn.Parameter(torch.randn(self.kv_dim, dim, dtype=torch.bfloa
        self.w_v = nn.Parameter(torch.randn(self.kv_dim, dim, dtype=torch.bfloa
                                                                               
    def forward(self, x: torch.Tensor) -> torch.Tensor:                        
        # Retain PyTorch F.rms_norm for optimal mathematical precision parity s
        n = F.rms_norm(x, (self.dim,))                                         
                                                                               
        B, S, K = n.shape                                                      
        M = B * S                                                              
                                                                               
        Nq = self.dim                                                          
        Nk = self.kv_dim                                                       
        N = Nq + 2 * Nk                                                        
                                                                               
        n_2d = n.contiguous().view(M, K)                                       
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)       
                                                                               
        # Fast path 1: Structurally drop all inner bounds checks dynamically el
        if M % 256 == 0 and Nq % 256 == 0 and Nk % 256 == 0 and K % 128 == 0:  
            def grid(META):                                                    
                return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['
                                                                               
            fused_qkv_gemm_kernel_nomask[grid](                                
                n_2d, self.w_q, self.w_k, self.w_v, out,                       
                M, K, Nq, Nk, N                                                
            )                                                                  
        # Fast path 2: Retain dynamic pointer routing while keeping a single M-
        elif Nq % 256 == 0 and Nk % 256 == 0 and K % 128 == 0:                 
            def grid(META):                                                    
                return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['
                                                                               
            fused_qkv_gemm_kernel_mmask[grid](                                 
                n_2d, self.w_q, self.w_k, self.w_v, out,                       
                M, K, Nq, Nk, N                                                
            )                                                                  
        else:                                                                  
            # Reliable structural fallback safely catches any absolutely arbitr
            w_qkv = torch.cat([self.w_q, self.w_k, self.w_v], dim=0)           
            def grid_fallback(META):                                           
                return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['
            fused_qkv_fallback_kernel[grid_fallback](                          
                n_2d, w_qkv, out,                                              
                M, K, N,                                                       
                n_2d.stride(0), n_2d.stride(1),                                
                w_qkv.stride(0), w_qkv.stride(1),                              
                out.stride(0), out.stride(1)                                   
            )                                                                  
                                                                               
        return out.view(B, S, N)                                               
                                                                               

── Kernel #105 (1a433b8f) ──
  Kernel time:      0.194 ms
  Reference eager:  0.314 ms
  torch.compile:    0.286 ms
  vs eager:         1.62x faster
  vs torch.compile: 1.48x faster
