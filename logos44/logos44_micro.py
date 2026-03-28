import torch
import torch.nn as nn
import math

class Logos44Field(nn.Module):
    def __init__(self, vocab_size=512, dim=256, rank=32, depth=44, n_signals=128):
        super().__init__()
        self.depth = depth
        self.embed = nn.Embedding(vocab_size, dim)
        self.phase_enc = nn.Parameter(torch.randn(512, dim) * 0.1)
        self.norm = nn.LayerNorm(dim)
        
        # Toroidal Bottleneck
        self.project = nn.Linear(dim, rank * 2, bias=False)
        self.angular = nn.Parameter(torch.ones(rank) * 0.5)
        self.up_proj = nn.Linear(rank, dim, bias=False)
        
        # CDMA Field signals
        self.signals = nn.Parameter(torch.randn(n_signals, dim) * 0.02)
        self.register_buffer('codes', torch.randn(n_signals, rank))
        self.key_to_code = nn.Linear(rank, rank, bias=False)
        
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.phase_enc[:T, :].unsqueeze(0)
        for _ in range(self.depth):
            res = h
            h = self.norm(h)
            # Toroidal Projection
            proj = self.project(h)
            key = torch.sin(proj[..., :32] * self.angular) * torch.cos(proj[..., 32:] * self.angular)
            # Field Decoding
            code = torch.softmax(self.key_to_code(key) @ self.codes.T * 4.0, dim=-1)
            field_sig = code @ self.signals
            h = torch.sigmoid(self.gate_proj(torch.cat([field_sig + self.up_proj(key), res], dim=-1)))
        return torch.matmul(self.out_norm(h), self.embed.weight.t())