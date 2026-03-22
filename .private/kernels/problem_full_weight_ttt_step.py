import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Full-weight SGD TTT step: forward + CE loss + backward + SGD update.

    FarnsworthEngine's TTT adapts the entire model to validation data using
    SGD with momentum (lr=0.002, momentum=0.9, 3 epochs). The bottleneck is
    the per-step forward+backward on chunks of validation data.

    This problem represents one transformer block's forward+backward for
    the TTT adaptation step. Fusing the forward, loss computation, and
    weight update into fewer kernel launches could save significant time
    (current budget: 43s for TTT on 8xH100).

    We model the MLP portion since it's the largest compute (3x expansion).
    """

    def __init__(self, dim: int, hidden: int):
        super(Model, self).__init__()
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        # Cast to bf16 like the training script
        self.fc.weight.data = self.fc.weight.data.to(torch.bfloat16)
        self.proj.weight.data = self.proj.weight.data.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through ReLU² MLP + residual.

        Args:
            x: [batch, seq_len, dim] input (bfloat16)

        Returns:
            out: [batch, seq_len, dim] output with residual (bfloat16)
        """
        h = F.relu(self.fc(x))
        return x + self.proj(h * h)


BATCH = 8
SEQ_LEN = 2048
DIM = 512
HIDDEN = 1536  # MLP 3x


def get_inputs():
    x = torch.randn(BATCH, SEQ_LEN, DIM, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [DIM, HIDDEN]
