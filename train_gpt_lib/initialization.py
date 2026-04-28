import torch
from torch import nn
import math

def apply_initialization(model: nn.Module, scheme: str, num_layers: int):
    """
    Applies various initialization schemes to the model.
    """
    if scheme == "default":
        return # Already handled by model's _init_weights
        
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
                continue
                
            if scheme == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif scheme == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif scheme == "kaiming_uniform_mlp":
                if "mlp" in name:
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                else:
                    nn.init.xavier_uniform_(module.weight)
            elif scheme.startswith("trunc_normal"):
                std = 0.02 if "02" in scheme else 0.01
                nn.init.trunc_normal_(module.weight, std=std)
            elif scheme == "gpt2_like":
                nn.init.trunc_normal_(module.weight, std=0.02)
                if "proj" in name:
                    nn.init.zeros_(module.weight)
            elif scheme == "orthogonal_attn":
                if "attn" in name:
                    nn.init.orthogonal_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)
            elif scheme == "depth_scaled":
                nn.init.xavier_uniform_(module.weight)
                module.weight.data.mul_(1.0 / math.sqrt(2 * num_layers))
            elif scheme == "small_attn_in":
                nn.init.xavier_uniform_(module.weight)
                if "c_q" in name or "c_k" in name or "c_v" in name or "fc" in name:
                    module.weight.data.mul_(0.5)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    if scheme == "depth_scaled" or scheme == "small_attn_in":
        # Ensure tok_emb is still initialized
        if hasattr(model, "tok_emb"):
            nn.init.normal_(model.tok_emb.weight, mean=0.0, std=0.02)
