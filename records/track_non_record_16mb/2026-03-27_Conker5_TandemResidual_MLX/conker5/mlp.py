from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int):
        super().__init__()
        self.layers = []
        prev = in_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev, hidden_dim))
            prev = hidden_dim
        self.out = nn.Linear(prev, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = nn.gelu(layer(x))
        return self.out(x)
