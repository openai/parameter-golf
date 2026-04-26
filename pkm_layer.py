"""
Product Key Memory (PKM) — Lample et al., 2019.
"A 12-layer transformer + PKM outperforms a 24-layer transformer at 2x speed."

Core idea: separate COMPUTATION (transformer) from STORAGE (memory lookup).
Most bytes are predictable from common patterns — store them as retrievable
memory slots rather than encoding them implicitly in transformer weights.

Cartesian-product key decomposition gives O(sqrt(N)) lookup over N slots.
At 16MB: ~8MB PKM values + ~8MB base transformer.

Academic basis:
- Lample et al. (2019): Product Key Memory
- Meta (2024): Memory Layers at Scale
- Veness et al. (2021): Gated Linear Networks (DeepMind)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ProductKeyMemory(nn.Module):
    """
    Product Key Memory with Cartesian sub-key decomposition.
    N_values = n_subkeys^2 total slots, but only O(sqrt(N)) lookup cost.
    """
    def __init__(self, d_model, n_subkeys=64, d_key=32, top_k=8):
        super().__init__()
        self.n_subkeys = n_subkeys
        self.n_values = n_subkeys * n_subkeys  # 4096 slots
        self.top_k = top_k
        self.d_model = d_model

        # Two sets of sub-keys for Cartesian product
        self.keys_a = nn.Parameter(torch.randn(n_subkeys, d_key) * 0.02)
        self.keys_b = nn.Parameter(torch.randn(n_subkeys, d_key) * 0.02)

        # Value table — this is the "memory"
        self.values = nn.Parameter(torch.randn(self.n_values, d_model) * 0.01)

        # Query projection: input -> 2 sub-queries
        self.query_proj = nn.Linear(d_model, 2 * d_key, bias=False)

        # Output gate: learn when to use memory vs pass through
        self.gate = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.gate.bias, -2.0)  # Start with gate mostly closed

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        returns: x + gate * memory_output
        """
        B, T, D = x.shape

        # Project to sub-queries
        q = self.query_proj(x)  # [B, T, 2*d_key]
        qa, qb = q.chunk(2, dim=-1)  # each [B, T, d_key]

        # Score against sub-keys
        scores_a = torch.matmul(qa, self.keys_a.T)  # [B, T, n_subkeys]
        scores_b = torch.matmul(qb, self.keys_b.T)  # [B, T, n_subkeys]

        # Top-k sub-keys per side
        tk = min(self.top_k, self.n_subkeys)
        topk_a = scores_a.topk(tk, dim=-1)  # values, indices: [B, T, tk]
        topk_b = scores_b.topk(tk, dim=-1)

        # Cartesian product of top-k: tk^2 candidate slots
        # Indices into flat value table
        idx_a = topk_a.indices.unsqueeze(-1)  # [B, T, tk, 1]
        idx_b = topk_b.indices.unsqueeze(-2)  # [B, T, 1, tk]
        flat_idx = (idx_a * self.n_subkeys + idx_b).reshape(B, T, -1)  # [B, T, tk^2]

        # Scores: sum of sub-scores
        combined_scores = (topk_a.values.unsqueeze(-1) + topk_b.values.unsqueeze(-2)).reshape(B, T, -1)

        # Take final top-k from tk^2 candidates
        final_tk = min(self.top_k, flat_idx.shape[-1])
        final_topk = combined_scores.topk(final_tk, dim=-1)
        final_idx = flat_idx.gather(-1, final_topk.indices)  # [B, T, final_tk]

        # Softmax weights over selected slots
        weights = F.softmax(final_topk.values, dim=-1)  # [B, T, final_tk]

        # Lookup values
        vals = self.values[final_idx]  # [B, T, final_tk, d_model]

        # Weighted sum
        mem_out = (weights.unsqueeze(-1) * vals).sum(-2)  # [B, T, d_model]

        # Gated residual
        g = torch.sigmoid(self.gate(x))  # [B, T, 1]
        return x + g * mem_out

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def memory_mb(self, dtype_bytes=1):
        """Estimate memory footprint in MB at given dtype."""
        val_bytes = self.n_values * self.d_model * dtype_bytes
        key_bytes = 2 * self.n_subkeys * self.keys_a.shape[1] * 2  # fp16
        other = sum(p.numel() * 2 for p in [self.query_proj.weight, self.gate.weight, self.gate.bias])
        return (val_bytes + key_bytes + other) / 1024 / 1024


# Test it
if __name__ == "__main__":
    d = 512
    pkm = ProductKeyMemory(d_model=d, n_subkeys=64, d_key=32, top_k=8)
    x = torch.randn(2, 128, d)
    out = pkm(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {pkm.param_count():,}")
    print(f"Memory slots: {pkm.n_values:,}")
    print(f"Est. size (int8): {pkm.memory_mb(1):.1f} MB")
    print(f"Est. size (fp16): {pkm.memory_mb(2):.1f} MB")

    # Verify gate starts mostly closed
    g = torch.sigmoid(pkm.gate(x))
    print(f"Initial gate mean: {g.mean():.4f} (should be ~0.12, mostly closed)")
