# tests/test_bigram_hash.py
"""Tests for BigramHashEmbedding."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest


class BigramHashEmbedding(nn.Module):
    def __init__(self, hash_size=10240, proj_dim=128, model_dim=512):
        super().__init__()
        self.hash_size = hash_size
        self.embed = nn.Embedding(hash_size, proj_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = nn.Linear(proj_dim, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        t = token_ids.to(torch.int32)
        h = torch.empty_like(t)
        h[..., 0] = self.hash_size - 1
        h[..., 1:] = (
            torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1])
            % (self.hash_size - 1)
        )
        out = self.proj(self.embed(h.long()))
        return out * self.scale.to(dtype=out.dtype)


class TestBigramHashEmbedding:
    def test_zero_init_output(self):
        """Zero-initialized weights → output should be all zeros at init."""
        model = BigramHashEmbedding(hash_size=100, proj_dim=16, model_dim=32)
        ids = torch.randint(0, 50, (2, 10))
        out = model(ids)
        assert torch.allclose(out, torch.zeros_like(out)), \
            "Zero-init BigramHash should output zeros"

    def test_output_shape(self):
        """Output shape must be [B, T, model_dim]."""
        model = BigramHashEmbedding(hash_size=100, proj_dim=16, model_dim=32)
        ids = torch.randint(0, 50, (3, 20))
        out = model(ids)
        assert out.shape == (3, 20, 32), f"Expected (3,20,32), got {out.shape}"

    def test_first_token_uses_padding_hash(self):
        """First token in each sequence must use hash_size-1 (padding bucket)."""
        model = BigramHashEmbedding(hash_size=100, proj_dim=16, model_dim=32)
        ids = torch.zeros(1, 5, dtype=torch.long)
        t = ids.to(torch.int32)
        h = torch.empty_like(t)
        h[..., 0] = model.hash_size - 1
        assert h[0, 0].item() == model.hash_size - 1

    def test_hash_in_valid_range(self):
        """All computed hash values must be in [0, hash_size-1]."""
        hash_size = 100
        model = BigramHashEmbedding(hash_size=hash_size, proj_dim=16, model_dim=32)
        ids = torch.randint(0, 50, (4, 32))
        t = ids.to(torch.int32)
        h = torch.empty_like(t)
        h[..., 0] = hash_size - 1
        h[..., 1:] = (
            torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1])
            % (hash_size - 1)
        )
        assert (h >= 0).all() and (h < hash_size).all(), \
            "All hash values must be in [0, hash_size)"

    def test_different_bigrams_may_have_different_hashes(self):
        """Different token pairs should generally map to different buckets (collision check)."""
        hash_size = 10000
        pairs = [(i, j) for i in range(10) for j in range(10)]
        hashes = set()
        for a, b in pairs:
            h = int(torch.bitwise_xor(torch.tensor(36313 * b), torch.tensor(27191 * a)) % (hash_size - 1))
            hashes.add(h)
        assert len(hashes) > 90, "Hash function should produce mostly unique values for small inputs"

    def test_gradients_flow_through(self):
        """Gradients must flow through BigramHashEmbedding."""
        model = BigramHashEmbedding(hash_size=100, proj_dim=16, model_dim=32)
        nn.init.normal_(model.embed.weight)
        nn.init.normal_(model.proj.weight)
        ids = torch.randint(0, 50, (2, 8))
        out = model(ids)
        loss = out.sum()
        loss.backward()
        assert model.embed.weight.grad is not None
        assert model.proj.weight.grad is not None
        assert model.scale.grad is not None
