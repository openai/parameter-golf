import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Fused LM head projection + logit softcap + cross-entropy loss.

    In the parameter-golf transformer, the final step computes:
      logits = softcap * tanh(x @ W^T / softcap)   (tied embedding weight)
      loss = CE(logits, targets, reduction='none')   (per-token losses for TTT)

    The intermediate logits tensor is [batch, seq_len, vocab] which is large
    relative to this tiny model. Fusing avoids materializing it in HBM.

    This is the eval bottleneck in test-time training (TTT) where we need
    per-token losses for thousands of document chunks.
    """

    def __init__(self, dim: int, vocab_size: int, softcap: float):
        super(Model, self).__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.softcap = softcap
        self.weight = nn.Parameter(torch.randn(vocab_size, dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim] final hidden states (bfloat16)
            targets: [batch, seq_len] target token ids (int64)

        Returns:
            per_token_loss: [batch, seq_len] CE loss per position (float32)
        """
        bsz, sl, dim = x.shape
        # Project to vocab
        logits = F.linear(x, self.weight)  # [bsz, sl, vocab]
        # Softcap
        logits = self.softcap * torch.tanh(logits / self.softcap)
        # Per-token CE loss
        loss = F.cross_entropy(
            logits.float().reshape(-1, self.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape(bsz, sl)
        return loss


# Problem dimensions matching parameter-golf model
BATCH = 64  # TTT batch size
SEQ_LEN = 1024  # eval sequence length
DIM = 512  # model dimension
VOCAB = 1024  # vocabulary size
SOFTCAP = 30.0


def get_inputs():
    x = torch.randn(BATCH, SEQ_LEN, DIM, dtype=torch.bfloat16)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), dtype=torch.int64)
    return [x, targets]


def get_init_inputs():
    return [DIM, VOCAB, SOFTCAP]
