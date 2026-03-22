import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Fused ReLU² MLP: x + proj(relu(fc(x))²)

    The MLP with ReLU² activation is the single most expensive op per block
    (3x expansion = 512->1536->512). Fusing relu + square + second matmul
    avoids materializing the 1536-dim intermediate in HBM.

    Called 11 times per forward pass (11 layers), and again 11 times in
    backward during TTT. This is the highest-throughput kernel target.
    """

    def __init__(self, dim: int, hidden: int):
        super(Model, self).__init__()
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        self.fc.weight.data = self.fc.weight.data.to(torch.bfloat16)
        self.proj.weight.data = self.proj.weight.data.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim] input (bfloat16)
        Returns:
            out: [batch, seq_len, dim] MLP output (bfloat16)
        """
        h = F.relu(self.fc(x))
        return self.proj(h * h)


BATCH = 8
SEQ_LEN = 2048
DIM = 512
HIDDEN = 1536


def get_inputs():
    x = torch.randn(BATCH, SEQ_LEN, DIM, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [DIM, HIDDEN]
