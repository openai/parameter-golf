"""Weight-sharing / recurrence GPT variant.

Uses N unique transformer blocks iterated K times (effective depth = N*K).
Per-loop learned scale factors and optional parallel residual connections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tiny_gpt import RMSNorm, TransformerBlock


class TinyGPTShared(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 512,
                 n_unique_layers: int = 5, n_loops: int = 4,
                 n_heads: int = 8, mlp_ratio: float = 3.0,
                 max_seq_len: int = 1024, tie_embeddings: bool = True,
                 parallel_residual: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_unique_layers = n_unique_layers
        self.n_loops = n_loops

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, parallel_residual)
            for _ in range(n_unique_layers)
        ])

        # Per-loop learned scale factors (initialized to 1.0)
        self.loop_scales = nn.Parameter(torch.ones(n_loops))

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
        # Reset loop_scales to 1.0 after generic init
        nn.init.ones_(self.loop_scales)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Seq len {T} > max {self.max_seq_len}"

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for loop_idx in range(self.n_loops):
            scale = self.loop_scales[loop_idx]
            for block in self.blocks:
                # block.residual() returns only the delta, no skip connection
                x = x + scale * block.residual(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @property
    def effective_depth(self):
        return self.n_unique_layers * self.n_loops

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
