import torch
import torch.nn as nn
from torch.nn import functional as F

class DEQBlock(nn.Module):
    def __init__(self, d_model, num_heads, expansion=2.0, num_iterations=30):
        super().__init__()
        self.num_iterations = num_iterations
        self.d_model = d_model
        
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * expansion)),
            nn.GELU(),
            nn.Linear(int(d_model * expansion), d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, z, x):
        attn_out, _ = self.attn(self.norm1(z), self.norm1(z), self.norm1(z))
        mlp_out = self.mlp(self.norm2(z))
        return x + attn_out + mlp_out
    
    def fixed_point_solve(self, x, max_iters=None):
        if max_iters is None:
            max_iters = self.num_iterations
        
        z = x.clone()
        for i in range(max_iters):
            z_next = self.forward(z, x)
            error = torch.norm(z_next - z) / (torch.norm(z) + 1e-6)
            z = z_next
            if error < 1e-3:
                break
        return z


class DeepEquilibriumModel(nn.Module):
    def __init__(self, d_model=512, num_heads=8, vocab_size=1024, expansion=3.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)
        
        self.deq_block = DEQBlock(d_model, num_heads, expansion, num_iterations=35)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x = x + self.pos_embed[:, :seq_len, :]
        z = self.deq_block.fixed_point_solve(x)
        z = self.norm(z)
        logits = self.lm_head(z)
        return logits
