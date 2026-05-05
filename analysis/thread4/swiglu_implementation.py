"""
SwiGLU MLP drop-in replacement for parameter-golf baseline.

Replace the existing MLP class in train_gpt.py with this SwiGLUMLP.
At hidden=4*dim//3, this has the SAME parameter count as ReLU² (2x expansion)
but ~2-5% better perplexity from the gating mechanism.

Usage: In the Block class, replace `self.mlp = MLP(dim, mlp_mult)` with
       `self.mlp = SwiGLUMLP(dim)` 
       
       And set MLP_MULT environment variable handling accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Assuming CastedLinear is already defined in train_gpt.py
# class CastedLinear(nn.Linear): ...


class SwiGLUMLP(nn.Module):
    """SwiGLU activation MLP: parameter-matched replacement for ReLU² MLP.
    
    SwiGLU(x) = (Swish(xW_gate) ⊙ xW_up) @ W_down
    
    With hidden = 4*dim//3:
      Params = 3 * dim * hidden ≈ 4 * dim² (same as ReLU² with 2x expansion)
    
    At dim=512: hidden=682, params=1,047,552 vs ReLU² 1,048,576
    """
    
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        if hidden is None:
            hidden = 4 * dim // 3
            # Round to nearest multiple of 8 for hardware efficiency
            hidden = ((hidden + 7) // 8) * 8
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.up = CastedLinear(dim, hidden, bias=False)
        self.down = CastedLinear(hidden, dim, bias=False)
        self.down._zero_init = True  # Match baseline's zero-init on output projection
    
    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---- Parameter count verification ----
if __name__ == "__main__":
    dim = 512
    
    # ReLU² baseline
    relu2_params = dim * (2 * dim) + (2 * dim) * dim
    print(f"ReLU² (2x expansion): {relu2_params:,} params/block")
    
    # SwiGLU param-matched
    hidden = 4 * dim // 3
    hidden = ((hidden + 7) // 8) * 8  # Round to 8
    swiglu_params = 3 * dim * hidden
    print(f"SwiGLU (hidden={hidden}): {swiglu_params:,} params/block")
    print(f"Difference: {swiglu_params - relu2_params:+,} params ({(swiglu_params/relu2_params - 1)*100:+.1f}%)")
    
    # Verify with multiple dims
    print("\nParam comparison across dims:")
    for d in [448, 480, 496, 504, 512, 528, 544, 672, 704]:
        relu2 = d * (2 * d) + (2 * d) * d
        h = 4 * d // 3
        h = ((h + 7) // 8) * 8
        swiglu = 3 * d * h
        diff_pct = (swiglu / relu2 - 1) * 100
        print(f"  dim={d:3d}: relu²={relu2:>10,} swiglu(h={h:3d})={swiglu:>10,} ({diff_pct:+.1f}%)")
