"""Reference MLP activation for parent P0 (decompressed train_gpt.py).

The wrapped trainer's MLP forward matches (see README "LeakyReLU(0.5)^2")::

    self.proj(F.leaky_relu(self.fc(x), negative_slope=.5).square())

Use ``LEAKY_RELU_SLOPE=0`` for ReLU² baseline A/B; ``0.5`` matches the published record.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mlp_post_activation(fc_x: torch.Tensor, leaky_relu_slope: float) -> torch.Tensor:
    """Apply squared activation after the MLP up-projection (pre down-proj)."""
    if leaky_relu_slope <= 0:
        return F.relu(fc_x).square()
    return F.leaky_relu(fc_x, negative_slope=leaky_relu_slope).square()
