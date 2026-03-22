import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Batched LoRA forward pass with independent weights per batch element.

    For test-time training (TTT), each document in the batch has its own
    rank-8 LoRA adapter. The forward computes:
      delta = x @ A^T @ B^T    per batch element independently

    Where A is [bsz, rank, in_features] and B is [bsz, out_features, rank].
    This is a batched small-rank matmul (rank=8) that is heavily memory-bound
    because the intermediate tensor [bsz, seq_len, rank] is tiny.

    We need this for Q projection (512->512), V projection (512->256),
    and LM head (512->1024). The LM head variant is the largest.
    """

    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super(Model, self).__init__()
        self.bsz = bsz
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.A = nn.Parameter(torch.randn(bsz, rank, in_features, dtype=torch.bfloat16))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [bsz, seq_len, in_features] input (bfloat16)

        Returns:
            delta: [bsz, seq_len, out_features] LoRA output (bfloat16)
        """
        # x @ A^T -> [bsz, seq_len, rank]
        # result @ B^T -> [bsz, seq_len, out_features]
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)


# LM head variant (largest of the three LoRA targets)
BSZ = 64
SEQ_LEN = 1024
IN_FEATURES = 512
OUT_FEATURES = 1024  # vocab size
RANK = 8


def get_inputs():
    x = torch.randn(BSZ, SEQ_LEN, IN_FEATURES, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [BSZ, IN_FEATURES, OUT_FEATURES, RANK]

