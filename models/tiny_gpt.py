"""Standard (non-shared) byte-level GPT baseline."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, d_model, bias=False)
        self.gate = nn.Linear(d_model, mlp_dim, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.gate(x)) * self.fc1(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 3.0,
                 parallel_residual: bool = False):
        super().__init__()
        self.parallel_residual = parallel_residual
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio))

    def forward(self, x):
        """Returns x + residual (full residual connection included)."""
        if self.parallel_residual:
            h = self.ln1(x)
            x = x + self.attn(h) + self.mlp(self.ln2(x))
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x

    def residual(self, x):
        """Returns only the residual delta (no skip connection)."""
        if self.parallel_residual:
            h = self.ln1(x)
            return self.attn(h) + self.mlp(self.ln2(x))
        else:
            attn_out = self.attn(self.ln1(x))
            mlp_out = self.mlp(self.ln2(x + attn_out))
            return attn_out + mlp_out


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 512,
                 n_layers: int = 10, n_heads: int = 8,
                 mlp_ratio: float = 3.0, max_seq_len: int = 1024,
                 tie_embeddings: bool = True,
                 parallel_residual: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, parallel_residual)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Seq len {T} > max {self.max_seq_len}"

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_unique_params(self):
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total
