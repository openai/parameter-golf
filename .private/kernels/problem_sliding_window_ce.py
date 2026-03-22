import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Sliding window cross-entropy scoring with softcap.

    During eval, we compute logits = softcap * tanh(x @ W.T / softcap)
    then CE loss per token. With sliding window (stride=64, seq=2048),
    this is called thousands of times. Fusing the projection + softcap +
    CE into one kernel avoids the large [batch, seq, vocab] intermediate.

    Eval budget: 86s for sliding window on 8xH100. Even small speedups
    compound over thousands of windows.
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
            per_token_loss: [batch, seq_len] CE loss (float32)
        """
        logits = F.linear(x, self.weight)
        logits = self.softcap * torch.tanh(logits / self.softcap)
        bsz, sl, V = logits.shape
        return F.cross_entropy(
            logits.float().reshape(-1, V), targets.reshape(-1), reduction="none"
        ).reshape(bsz, sl)


BATCH = 32
SEQ_LEN = 2048
DIM = 512
VOCAB = 1024
SOFTCAP = 30.0


def get_inputs():
    x = torch.randn(BATCH, SEQ_LEN, DIM, dtype=torch.bfloat16)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), dtype=torch.int64)
    return [x, targets]


def get_init_inputs():
    return [DIM, VOCAB, SOFTCAP]
