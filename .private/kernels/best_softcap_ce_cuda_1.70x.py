import torch                                                                   
import torch.nn as nn                                                          
import torch.nn.functional as F                                                
from torch.utils.cpp_extension import load_inline                              
                                                                               
cuda_source = r"""                                                             
#include <torch/extension.h>                                                   
#include <cuda_runtime.h>                                                      
#include <cuda_bf16.h>                                                         
                                                                               
__device__ __forceinline__ float apply_softcap_bf16(__nv_bfloat16 val, float so
    float f_val = __bfloat162float(val);                                       
    __nv_bfloat16 v1 = __float2bfloat16(f_val * inv_softcap);                  
    __nv_bfloat16 v2 = __float2bfloat16(tanhf(__bfloat162float(v1)));          
    __nv_bfloat16 v3 = __float2bfloat16(__bfloat162float(v2) * softcap);       
    return __bfloat162float(v3);                                               
}                                                                              
                                                                               
union Int2Bfloat162 {                                                          
    int i32;                                                                   
    __nv_bfloat162 bf162;                                                      
};                                                                             
                                                                               
__global__ __launch_bounds__(256, 8)                                           
void softcap_ce_kernel_optimized(                                              
    const __nv_bfloat16* __restrict__ logits,                                  
    const int64_t* __restrict__ targets,                                       
    float* __restrict__ losses,                                                
    float softcap,                                                             
    float inv_softcap,                                                         
    int vocab_size,                                                            
    int num_rows                                                               
) {                                                                            
    const int warps_per_block = blockDim.x >> 5;                               
    const int warp_idx = threadIdx.x >> 5;                                     
    const int lane_id  = threadIdx.x & 31;                                     
                                                                               
    int start_row = blockIdx.x * warps_per_block;                              
    int grid_stride = gridDim.x * warps_per_block;                             
                                                                               
    __shared__ float smem_losses[8];                                           
                                                                               
    for (int block_row = start_row; block_row < num_rows; block_row += grid_str
        int row = block_row + warp_idx;                                        
        bool valid = row < num_rows;                                           
                                                                               
        float m = -1e20f;                                                      
        float s = 0.0f;                                                        
        float loss = 0.0f;                                                     
                                                                               
        if (valid) {                                                           
            const int target = (int)targets[row];                              
            const __nv_bfloat16* row_logits = logits + (size_t)row * vocab_size
                                                                               
            const bool can_vectorize = (vocab_size % 8 == 0) && ((reinterpret_c
                                                                               
            if (can_vectorize) {                                               
                #pragma unroll 2                                               
                for (int i = lane_id * 8; i < vocab_size; i += 256) {          
                    int4 vec = *reinterpret_cast<const int4*>(row_logits + i); 
                                                                               
                    Int2Bfloat162 u0, u1, u2, u3;                              
                    u0.i32 = vec.x; u1.i32 = vec.y; u2.i32 = vec.z; u3.i32 = ve
                                                                               
                    float2 v0 = __bfloat1622float2(u0.bf162);                  
                    float2 v1 = __bfloat1622float2(u1.bf162);                  
                    float2 v2 = __bfloat1622float2(u2.bf162);                  
                    float2 v3 = __bfloat1622float2(u3.bf162);                  
                                                                               
                    v0.x *= inv_softcap; v0.y *= inv_softcap;                  
                    v1.x *= inv_softcap; v1.y *= inv_softcap;                  
                    v2.x *= inv_softcap; v2.y *= inv_softcap;                  
                    v3.x *= inv_softcap; v3.y *= inv_softcap;                  
                                                                               
                    u0.bf162 = __float22bfloat162_rn(v0);                      
                    u1.bf162 = __float22bfloat162_rn(v1);                      
                    u2.bf162 = __float22bfloat162_rn(v2);                      
                    u3.bf162 = __float22bfloat162_rn(v3);                      
                                                                               
                    v0 = __bfloat1622float2(u0.bf162);                         
                    v1 = __bfloat1622float2(u1.bf162);                         
                    v2 = __bfloat1622float2(u2.bf162);                         
                    v3 = __bfloat1622float2(u3.bf162);                         
                                                                               
                    v0.x = tanhf(v0.x); v0.y = tanhf(v0.y);                    
                    v1.x = tanhf(v1.x); v1.y = tanhf(v1.y);                    
                    v2.x = tanhf(v2.x); v2.y = tanhf(v2.y);                    
                    v3.x = tanhf(v3.x); v3.y = tanhf(v3.y);                    
                                                                               
                    u0.bf162 = __float22bfloat162_rn(v0);                      
                    u1.bf162 = __float22bfloat162_rn(v1);                      
                    u2.bf162 = __float22bfloat162_rn(v2);                      
                    u3.bf162 = __float22bfloat162_rn(v3);                      
                                                                               
                    v0 = __bfloat1622float2(u0.bf162);                         
                    v1 = __bfloat1622float2(u1.bf162);                         
                    v2 = __bfloat1622float2(u2.bf162);                         
                    v3 = __bfloat1622float2(u3.bf162);                         
                                                                               
                    v0.x *= softcap; v0.y *= softcap;                          
                    v1.x *= softcap; v1.y *= softcap;                          
                    v2.x *= softcap; v2.y *= softcap;                          
                    v3.x *= softcap; v3.y *= softcap;                          
                                                                               
                    u0.bf162 = __float22bfloat162_rn(v0);                      
                    u1.bf162 = __float22bfloat162_rn(v1);                      
                    u2.bf162 = __float22bfloat162_rn(v2);                      
                    u3.bf162 = __float22bfloat162_rn(v3);                      
                                                                               
                    v0 = __bfloat1622float2(u0.bf162);                         
                    v1 = __bfloat1622float2(u1.bf162);                         
                    v2 = __bfloat1622float2(u2.bf162);                         
                    v3 = __bfloat1622float2(u3.bf162);                         
                                                                               
                    float m0 = fmaxf(v0.x, v0.y);                              
                    float m1 = fmaxf(v1.x, v1.y);                              
                    float m2 = fmaxf(v2.x, v2.y);                              
                    float m3 = fmaxf(v3.x, v3.y);                              
                                                                               
                    float m01 = fmaxf(m0, m1);                                 
                    float m23 = fmaxf(m2, m3);                                 
                                                                               
                    float local_m = fmaxf(m01, m23);                           
                                                                               
                    float e0x = __expf(v0.x - local_m);                        
                    float e0y = __expf(v0.y - local_m);                        
                    float e1x = __expf(v1.x - local_m);                        
                    float e1y = __expf(v1.y - local_m);                        
                    float e2x = __expf(v2.x - local_m);                        
                    float e2y = __expf(v2.y - local_m);                        
                    float e3x = __expf(v3.x - local_m);                        
                    float e3y = __expf(v3.y - local_m);                        
                                                                               
                    float local_s = (e0x + e0y) + (e1x + e1y) + (e2x + e2y) + (
                                                                               
                    float d = local_m - m;                                     
                    float e = __expf(-fabsf(d));                               
                    bool ge = (d >= 0.0f);                                     
                    float fma_a = ge ? s : local_s;                            
                    float fma_c = ge ? local_s : s;                            
                    s = fmaf(fma_a, e, fma_c);                                 
                    m = ge ? local_m : m;                                      
                }                                                              
            } else {                                                           
                #pragma unroll 4                                               
                for (int idx = lane_id; idx < vocab_size; idx += 32) {         
                    float val = apply_softcap_bf16(row_logits[idx], softcap, in
                                                                               
                    float d = val - m;                                         
                    float e = __expf(-fabsf(d));                               
                    bool ge = (d >= 0.0f);                                     
                    float fma_a = ge ? s : 1.0f;                               
                    float fma_c = ge ? 1.0f : s;                               
                    s = fmaf(fma_a, e, fma_c);                                 
                    m = ge ? val : m;                                          
                }                                                              
            }                                                                  
                                                                               
            #pragma unroll                                                     
            for (int offset = 16; offset > 0; offset >>= 1) {                  
                float m2 = __shfl_down_sync(0xffffffff, m, offset);            
                float s2 = __shfl_down_sync(0xffffffff, s, offset);            
                                                                               
                float d = m - m2;                                              
                bool ge = (d >= 0.0f);                                         
                float e = __expf(-fabsf(d));                                   
                                                                               
                float fma_a = ge ? s2 : s;                                     
                float fma_c = ge ? s : s2;                                     
                                                                               
                s = fmaf(fma_a, e, fma_c);                                     
                m = ge ? m : m2;                                               
            }                                                                  
                                                                               
            if (lane_id == 0) {                                                
                float tval = apply_softcap_bf16(row_logits[target], softcap, in
                loss = __logf(s) + m - tval;                                   
            }                                                                  
        }                                                                      
                                                                               
        if (valid && lane_id == 0) {                                           
            smem_losses[warp_idx] = loss;                                      
        }                                                                      
                                                                               
        __syncthreads();                                                       
                                                                               
        if (warp_idx == 0 && lane_id < warps_per_block) {                      
            int out_row = block_row + lane_id;                                 
            if (out_row < num_rows) {                                          
                losses[out_row] = smem_losses[lane_id];                        
            }                                                                  
        }                                                                      
                                                                               
        __syncthreads();                                                       
    }                                                                          
}                                                                              
                                                                               
torch::Tensor fused_softcap_ce_cuda(torch::Tensor logits, torch::Tensor targets
    int bsz = logits.size(0);                                                  
    int sl = logits.size(1);                                                   
    int V = logits.size(2);                                                    
    int num_rows = bsz * sl;                                                   
                                                                               
    auto losses = torch::empty({bsz, sl}, torch::dtype(torch::kFloat32).device(
                                                                               
    const int block_size = 256;                                                
    const int warps_per_block = block_size / 32;                               
    int num_blocks = (num_rows + warps_per_block - 1) / warps_per_block;       
                                                                               
    float inv_softcap = 1.0f / softcap;                                        
                                                                               
    softcap_ce_kernel_optimized<<<num_blocks, block_size>>>(                   
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>())
        targets.data_ptr<int64_t>(),                                           
        losses.data_ptr<float>(),                                              
        softcap,                                                               
        inv_softcap,                                                           
        V,                                                                     
        num_rows                                                               
    );                                                                         
                                                                               
    return losses;                                                             
}                                                                              
"""                                                                            
                                                                               
cpp_source = r"""                                                              
torch::Tensor fused_softcap_ce_cuda(torch::Tensor logits, torch::Tensor targets
"""                                                                            
                                                                               
_fused_softcap_ce_opt = load_inline(                                           
    name="fused_softcap_ce_opt",                                               
    cpp_sources=cpp_source,                                                    
    cuda_sources=cuda_source,                                                  
    functions=["fused_softcap_ce_cuda"],                                       
    verbose=False,                                                             
    extra_cflags=["-O3"],                                                      
    extra_cuda_cflags=["-O3", "-use_fast_math", "-Xptxas=-dlcm=cg"]            
)                                                                              
                                                                               
class ModelNew(nn.Module):                                                     
    def __init__(self, dim: int, vocab_size: int, softcap: float):             
        super(ModelNew, self).__init__()                                       
        self.dim = dim                                                         
        self.vocab_size = vocab_size                                           
        self.softcap = softcap                                                 
        self.weight = nn.Parameter(torch.randn(vocab_size, dim, dtype=torch.bfl
                                                                               
    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: 
        logits = F.linear(x, self.weight)                                      
        return _fused_softcap_ce_opt.fused_softcap_ce_cuda(                    
            logits.contiguous(),                                               
            targets.contiguous(),                                              
            float(self.softcap)                                                
        )                                                                      
                                                                               

── Kernel #90 (de648301) ──
  Kernel time:      0.177 ms
  Reference eager:  0.786 ms
  torch.compile:    0.301 ms
  vs eager:         4.44x faster
  vs torch.compile: 1.70x faster
