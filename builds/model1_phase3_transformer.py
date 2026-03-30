from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def ternarize_ste(weight: Tensor, threshold: float = 0.05) -> Tensor:
    scale = weight.abs().mean().clamp_min(1e-6)
    ternary = torch.where(
        weight > threshold * scale,
        torch.ones_like(weight),
        torch.where(weight < -threshold * scale, -torch.ones_like(weight), torch.zeros_like(weight)),
    )
    return weight + (ternary - weight).detach()


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        q_weight = ternarize_ste(self.weight)
        return F.linear(x, q_weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight.to(x.dtype)


class GQAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must divide num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_repeat = num_heads // num_kv_heads
        self.q_proj = BitLinear(dim, dim, bias=False)
        self.k_proj = BitLinear(dim, self.head_dim * num_kv_heads, bias=False)
        self.v_proj = BitLinear(dim, self.head_dim * num_kv_heads, bias=False)
        self.o_proj = BitLinear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
        att = att + mask
        probs = F.softmax(att, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.up = BitLinear(dim, hidden_dim, bias=False)
        self.gate = BitLinear(dim, hidden_dim, bias=False)
        self.down = BitLinear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int = 4):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = GQAAttention(dim, num_heads, num_kv_heads)
        self.ffn = FeedForward(dim, dim * mlp_mult)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ResidualBitNetTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,
        model_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        num_kv_heads: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.residual_proj = nn.Linear(vocab_size, model_dim, bias=False)
        self.blocks = nn.ModuleList(
            [ResidualBlock(model_dim, num_heads=num_heads, num_kv_heads=num_kv_heads) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(model_dim)
        self.head = BitLinear(model_dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, residual_distribution: Tensor | None = None) -> Tensor:
        x = self.embed(input_ids)
        if residual_distribution is not None:
            x = x + self.residual_proj(residual_distribution.to(x.dtype))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def verify_forward() -> dict[str, object]:
    torch.manual_seed(0)
    model = ResidualBitNetTransformer()
    input_ids = torch.randint(0, 1024, (2, 16))
    residual = torch.randn(2, 16, 1024)
    logits = model(input_ids, residual)
    ternary = ternarize_ste(model.blocks[0].attn.q_proj.weight).unique(sorted=True).tolist()
    if logits.shape != (2, 16, 1024):
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if any(v not in (-1.0, 0.0, 1.0) for v in ternary):
        raise AssertionError(f"Non-ternary values observed: {ternary}")
    return {"logits_shape": list(logits.shape), "ternary_values": ternary[:3]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the Model 1 residual transformer.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_forward()
    print(json.dumps(result))
    if not args.verify_only:
        print("ResidualBitNetTransformer initialized successfully.")


if __name__ == "__main__":
    main()
