"""LOGOS-44 Tiny v4 — CUDA version for RunPod / Parameter Golf.
GQA + SwiGLU + stochastic depth + weight tying + dropout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class GQAttention(nn.Module):
    def __init__(self, dim, n_heads=8, n_kv_heads=4, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.scale = self.head_dim ** -0.5
        self.rotary = RotaryEmbedding(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        cos, sin = self.rotary(T, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Repeat KV heads
        k = k.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
        k = k.reshape(B, T, self.n_heads, self.head_dim)
        v = v.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
        v = v.reshape(B, T, self.n_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(out))

class SwiGLU(nn.Module):
    def __init__(self, dim, mult=2.66, dropout=0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class LogosBlock(nn.Module):
    def __init__(self, dim, n_heads=8, n_kv_heads=4, dropout=0.0, layer_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GQAttention(dim, n_heads, n_kv_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dropout=dropout)
        self.layer_drop = layer_drop

    def forward(self, x):
        if self.training and self.layer_drop > 0 and torch.rand(1).item() < self.layer_drop:
            return x
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Logos44Tiny(nn.Module):
    def __init__(self, vocab_size=8192, dim=384, n_layers=14,
                 n_heads=8, n_kv_heads=4, dropout=0.05, max_layer_drop=0.1):
        super().__init__()
        self.dim = dim
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            LogosBlock(dim, n_heads, n_kv_heads, dropout,
                       layer_drop=max_layer_drop * i / max(n_layers - 1, 1))
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        x = self.emb_drop(self.tok_emb(x) * (self.dim ** 0.5))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
