import torch                                                                   
import torch.nn as nn                                                          
import torch.nn.functional as F                                                
import triton                                                                  
import triton.language as tl                                                   
                                                                               
@triton.autotune(                                                              
    configs=[                                                                  
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K'
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K'
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K'
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K'
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K'
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K'
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K'
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K'
    ],                                                                         
    key=['M', 'N', 'K'],                                                       
)                                                                              
@triton.jit                                                                    
def fused_relu_sq_gemm_kernel_persist_opt(                                     
    a_ptr, w_ptr, c_ptr,                                                       
    M, N, K,                                                                   
    stride_am, stride_ak,                                                      
    stride_wn, stride_wk,                                                      
    stride_cm, stride_cn,                                                      
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,          
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.co
    GROUP_SIZE_M: tl.constexpr,                                                
):                                                                             
    # Persistent CTA execution model with grid-stride loops to prevent wave-tai
    pid = tl.program_id(axis=0)                                                
    num_programs = tl.num_programs(axis=0)                                     
                                                                               
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)                                       
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)                                       
    num_pid_in_group = GROUP_SIZE_M * num_pid_n                                
    total_tiles = num_pid_m * num_pid_n                                        
                                                                               
    for tile_id in range(pid, total_tiles, num_programs):                      
        # Grouped M layout mapping enforces tight L2 cache reuse among contiguo
        group_id = tile_id // num_pid_in_group                                 
        first_pid_m = group_id * GROUP_SIZE_M                                  
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)              
        pid_m = first_pid_m + (tile_id % group_size_m)                         
        pid_n = (tile_id % num_pid_in_group) // group_size_m                   
                                                                               
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)             
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)             
        offs_k = tl.arange(0, BLOCK_SIZE_K)                                    
                                                                               
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * strid
        # Zero-cost hardware transpose via flipped stride mapping (reads perfec
        w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * strid
                                                                               
        if not EVEN_M:                                                         
            a_mask_m = offs_m[:, None] < M                                     
        if not EVEN_N:                                                         
            w_mask_n = offs_n[None, :] < N                                     
                                                                               
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)         
                                                                               
        # Unrolled and predicate-pruned inner loops for natively divisible boun
        for k_iter in range(0, tl.cdiv(K, BLOCK_SIZE_K)):                      
            if EVEN_K:                                                         
                if EVEN_M:                                                     
                    a = tl.load(a_ptrs)                                        
                else:                                                          
                    a = tl.load(a_ptrs, mask=a_mask_m, other=0.0)              
                                                                               
                if EVEN_N:                                                     
                    w = tl.load(w_ptrs)                                        
                else:                                                          
                    w = tl.load(w_ptrs, mask=w_mask_n, other=0.0)              
            else:                                                              
                k_mask = (k_iter * BLOCK_SIZE_K + offs_k) < K                  
                if EVEN_M:                                                     
                    a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)       
                else:                                                          
                    a = tl.load(a_ptrs, mask=a_mask_m & k_mask[None, :], other=
                                                                               
                if EVEN_N:                                                     
                    w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)       
                else:                                                          
                    w = tl.load(w_ptrs, mask=k_mask[:, None] & w_mask_n, other=
                                                                               
            # ReLU and Squaring strictly modeled matching PyTorch FP32 semantic
            a_f32 = a.to(tl.float32)                                           
            a_f32 = tl.maximum(a_f32, 0.0)                                     
            a_bf16 = (a_f32 * a_f32).to(tl.bfloat16)                           
                                                                               
            # Executes fully optimized bfloat16 hardware tensor core matrix mul
            acc += tl.dot(a_bf16, w)                                           
                                                                               
            a_ptrs += BLOCK_SIZE_K * stride_ak                                 
            w_ptrs += BLOCK_SIZE_K * stride_wk                                 
                                                                               
        c = acc.to(tl.bfloat16)                                                
                                                                               
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)            
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)            
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * str
                                                                               
        if EVEN_M and EVEN_N:                                                  
            tl.store(c_ptrs, c)                                                
        elif EVEN_M:                                                           
            tl.store(c_ptrs, c, mask=offs_cn[None, :] < N)                     
        elif EVEN_N:                                                           
            tl.store(c_ptrs, c, mask=offs_cm[:, None] < M)                     
        else:                                                                  
            tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :]
                                                                               
                                                                               
class ModelNew(nn.Module):                                                     
    def __init__(self, dim: int, hidden: int):                                 
        super(ModelNew, self).__init__()                                       
        self.fc = nn.Linear(dim, hidden, bias=False)                           
        self.proj = nn.Linear(hidden, dim, bias=False)                         
        self.fc.weight.data = self.fc.weight.data.to(torch.bfloat16)           
        self.proj.weight.data = self.proj.weight.data.to(torch.bfloat16)       
        self._num_sms = None                                                   
                                                                               
    def forward(self, x: torch.Tensor) -> torch.Tensor:                        
        B, S, D = x.shape                                                      
        x2d = x.reshape(-1, D)                                                 
                                                                               
        # First projection stays mapped via hardware-optimized dynamic cuBLAS e
        h_pre = F.linear(x2d, self.fc.weight)                                  
                                                                               
        M, K = h_pre.shape                                                     
        N = self.proj.weight.shape[0]                                          
                                                                               
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)              
                                                                               
        # Dynamically hoists bounds-checking overhead by analyzing perfect modu
        EVEN_M = (M % 256 == 0)                                                
        EVEN_N = (N % 256 == 0)                                                
        EVEN_K = (K % 128 == 0)                                                
                                                                               
        if self._num_sms is None:                                              
            self._num_sms = torch.cuda.get_device_properties(x.device).multi_pr
                                                                               
        def grid(meta):                                                        
            tiles = triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta[
            # Ensures optimal latency hiding by actively preventing launch over
            return (min(tiles, self._num_sms * 4),)                            
                                                                               
        fused_relu_sq_gemm_kernel_persist_opt[grid](                           
            h_pre, self.proj.weight, out,                                      
            M, N, K,                                                           
            h_pre.stride(0), h_pre.stride(1),                                  
            self.proj.weight.stride(0), self.proj.weight.stride(1),            
            out.stride(0), out.stride(1),                                      
            EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_K=EVEN_K,                       
        )                                                                      
                                                                               
        return out.view(B, S, N)                                               
                                                                               

── Kernel #83 (69f12a2e) ──
  Kernel time:      0.096 ms
  Reference eager:  0.158 ms
  torch.compile:    0.121 ms
  vs eager:         1.65x faster
  vs torch.compile: 1.26x faster
