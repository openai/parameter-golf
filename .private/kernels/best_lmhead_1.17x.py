import torch                                                                   
import torch.nn as nn                                                          
import triton                                                                  
import triton.language as tl                                                   
                                                                               
                                                                               
@triton.jit                                                                    
def fused_linear_softcap_ce_kernel(                                            
    X_ptr,          # bf16 [N, D]                                              
    W_ptr,          # bf16 [V, D]                                              
    T_ptr,          # int64 [N]                                                
    Loss_ptr,       # fp32 [N]                                                 
    softcap,        # scalar                                                   
    inv_softcap,    # scalar (1/softcap)                                       
    N_rows,         # N                                                        
    stride_xn, stride_xd,                                                      
    stride_wv, stride_wd,                                                      
    D: tl.constexpr,                                                           
    V: tl.constexpr,                                                           
    BLOCK_M: tl.constexpr,                                                     
    BLOCK_N: tl.constexpr,                                                     
    BLOCK_K: tl.constexpr,                                                     
):                                                                             
    pid_m = tl.program_id(0)                                                   
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)                               
    rm_mask = rm < N_rows                                                      
                                                                               
    # Load targets safely; use -100 for ignored elements natively              
    targets = tl.load(T_ptr + rm, mask=rm_mask, other=-100).to(tl.int32)       
                                                                               
    # Initialize Log-Sum-Exp running state and target accumulator entirely insi
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)                  
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                                
    target_val = tl.zeros([BLOCK_M], dtype=tl.float32)                         
                                                                               
    # Pre-compute initial pointers to avoid index math inside loops            
    k_offs_base = tl.arange(0, BLOCK_K)                                        
    x_ptrs_base = X_ptr + rm[:, None] * stride_xn + k_offs_base[None, :] * stri
                                                                               
    v_offs_base = tl.arange(0, BLOCK_N)                                        
    # Load W as [BLOCK_N, BLOCK_K] to perfectly coalesce global memory reads al
    w_ptrs_v_base = W_ptr + v_offs_base[:, None] * stride_wv + k_offs_base[None
                                                                               
    for v_start in range(0, V, BLOCK_N):                                       
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)                   
        v_offs = v_start + v_offs_base                                         
        v_mask = v_offs < V                                                    
                                                                               
        x_ptrs = x_ptrs_base                                                   
        w_ptrs = w_ptrs_v_base                                                 
                                                                               
        for k_start in range(0, D, BLOCK_K):                                   
            k_mask = (k_start + k_offs_base) < D                               
                                                                               
            # Execute bounds-checked masked memory loads                       
            x = tl.load(x_ptrs, mask=(rm_mask[:, None] & k_mask[None, :]), othe
            w = tl.load(w_ptrs, mask=(v_mask[:, None] & k_mask[None, :]), other
                                                                               
            # Accumulate Tensor Core dot product (transpose w on-the-fly for op
            acc = tl.dot(x, tl.trans(w), acc)                                  
                                                                               
            # Zero-overhead pointer advancement safely isolated from index calc
            x_ptrs += BLOCK_K * stride_xd                                      
            w_ptrs += BLOCK_K * stride_wd                                      
                                                                               
        # Advance W base pointer for the next vocabulary tile                  
        w_ptrs_v_base += BLOCK_N * stride_wv                                   
                                                                               
        # Exact PyTorch bf16 softcap numeric parity                            
        logits_bf16 = acc.to(tl.bfloat16)                                      
        scaled_f32 = logits_bf16.to(tl.float32) * inv_softcap                  
        scaled_bf16 = scaled_f32.to(tl.bfloat16)                               
                                                                               
        # Inline numerically stable single-exponential tanh for reduced instruc
        val_f32 = scaled_bf16.to(tl.float32)                                   
        abs_x = tl.abs(val_f32)                                                
        e = tl.exp(-2.0 * abs_x)                                               
        t = (1.0 - e) / (1.0 + e)                                              
        tanh_fp32 = tl.where(val_f32 >= 0.0, t, -t)                            
                                                                               
        tanh_bf16 = tanh_fp32.to(tl.bfloat16)                                  
        softcapped_bf16 = (tanh_bf16.to(tl.float32) * softcap).to(tl.bfloat16) 
        logits_fp32 = softcapped_bf16.to(tl.float32)                           
                                                                               
        # Strictly mask elements falling outside the vocabulary boundaries     
        logits_fp32 = tl.where(v_mask[None, :], logits_fp32, -float('inf'))    
                                                                               
        # Online streaming log-sum-exp folding safely into running accumulators
        m_new = tl.maximum(m_i, tl.max(logits_fp32, axis=1))                   
        alpha = tl.exp(m_i - m_new)                                            
        l_i = tl.fma(l_i, alpha, tl.sum(tl.exp(logits_fp32 - m_new[:, None]), a
        m_i = m_new                                                            
                                                                               
        # Compute dynamic target logits contribution securely via predicate sel
        is_target = targets[:, None] == v_offs[None, :]                        
        target_val += tl.sum(tl.where(is_target, logits_fp32, 0.0), axis=1)    
                                                                               
    # Combine the fully reduced online formula                                 
    loss = -target_val + m_i + tl.log(l_i)                                     
    # Nullify explicitly ignored target token loss penalties                   
    loss = tl.where(targets == -100, 0.0, loss)                                
                                                                               
    tl.store(Loss_ptr + rm, loss, mask=rm_mask)                                
                                                                               
                                                                               
def triton_fused_linear_softcap_ce(x: torch.Tensor, weight: torch.Tensor, targe
    x = x.contiguous()                                                         
    weight = weight.contiguous()                                               
    targets = targets.contiguous()                                             
                                                                               
    N_rows, D = x.shape                                                        
    V, D_w = weight.shape                                                      
    assert D == D_w                                                            
                                                                               
    out = torch.empty(N_rows, dtype=torch.float32, device=x.device)            
                                                                               
    # Tuned hyperparameters: shrinking BLOCK_N shrinks Shared Memory requiremen
    # accommodating 2 blocks per SM on GPUs like the A100 to maximize throughpu
    BLOCK_M = 128                                                              
    BLOCK_N = 64                                                               
    BLOCK_K = 64                                                               
                                                                               
    grid = (triton.cdiv(N_rows, BLOCK_M),)                                     
                                                                               
    fused_linear_softcap_ce_kernel[grid](                                      
        x, weight, targets, out,                                               
        softcap, float(1.0 / softcap),                                         
        N_rows,                                                                
        x.stride(0), x.stride(1),                                              
        weight.stride(0), weight.stride(1),                                    
        D=D, V=V,                                                              
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,                     
        num_warps=8, num_stages=3,                                             
    )                                                                          
    return out                                                                 
                                                                               
                                                                               
class ModelNew(nn.Module):                                                     
    def __init__(self, dim: int, vocab_size: int, softcap: float):             
        super(ModelNew, self).__init__()                                       
        self.dim = dim                                                         
        self.vocab_size = vocab_size                                           
        self.softcap = softcap                                                 
        self.weight = nn.Parameter(torch.randn(vocab_size, dim, dtype=torch.bfl
                                                                               
    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: 
        bsz, sl, dim = x.shape                                                 
        x_flat = x.reshape(-1, dim)                                            
        targets_flat = targets.reshape(-1)                                     
                                                                               
        loss_flat = triton_fused_linear_softcap_ce(x_flat, self.weight, targets
        return loss_flat.reshape(bsz, sl)                                      
                                                                               

── Kernel #69 (39b631e8) ──
  Kernel time:      0.260 ms
  Reference eager:  0.787 ms
  torch.compile:    0.303 ms
  vs eager:         3.03x faster
  vs torch.compile: 1.17x faster
