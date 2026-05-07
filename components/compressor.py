from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn


class CheatSheetCompressor(nn.Module):
    """
    Takes final hidden states and compresses them into 8 summary vectors.

    These 8 vectors represent compressed memory from a long-context pass.

    Architecture:
    - Input: [batch, seq_len, d_model]
    - Mean pool over seq_len -> [batch, d_model]
    - Linear -> [batch, 8 * d_model]
    - Reshape -> [batch, 8, d_model]
    - LayerNorm on last dimension
    - Output: [batch, 8, d_model]
    """

    def __init__(self, d_model: int, num_vectors: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_vectors = num_vectors
        self.proj = nn.Linear(d_model, num_vectors * d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states: Tensor) -> Tensor:
        pooled = hidden_states.mean(dim=1)
        out = self.proj(pooled).reshape(-1, self.num_vectors, self.d_model)
        return self.norm(out)


class CheatSheetBuffer:
    """
    Stores and manages a finalized global cheat sheet tensor.

    After phase 1:
    - Run compressor on multiple training documents
    - Average vectors across documents
    - Keep a fixed [8, d_model] buffer
    - Buffer is not trainable but is persisted with the artifact
    """

    def __init__(self, d_model: int = 512, num_vectors: int = 8) -> None:
        self.d_model = d_model
        self.num_vectors = num_vectors
        self._buffer = torch.zeros(num_vectors, d_model, dtype=torch.float32)

    def finalize(
        self,
        model: nn.Module,
        compressor: CheatSheetCompressor,
        dataloader: Iterable[Tensor],
        device: torch.device | str,
        n_docs: int = 512,
    ) -> Tensor:
        """
        Finalize buffer by averaging compressor outputs over n_docs batches.

        This method assumes each dataloader item is already a hidden-state tensor
        of shape [batch, seq_len, d_model], or that `model` returns hidden states
        in this shape when called with that item.
        """
        model.eval()
        compressor.eval()
        dev = torch.device(device)
        acc = torch.zeros(self.num_vectors, self.d_model, device=dev, dtype=torch.float32)
        count = 0
        with torch.no_grad():
            for item in dataloader:
                if count >= n_docs:
                    break
                if not isinstance(item, torch.Tensor):
                    item = torch.as_tensor(item)
                x = item.to(dev)
                hidden_states = x if x.ndim == 3 else model(x)
                cheat = compressor(hidden_states)
                acc += cheat.mean(dim=0).to(dtype=torch.float32)
                count += 1
        if count == 0:
            raise ValueError("No samples seen while finalizing cheat sheet buffer.")
        self._buffer = (acc / float(count)).detach().cpu()
        return self._buffer

    def get(self) -> Tensor:
        """Return fixed cheat sheet tensor [8, d_model]."""
        return self._buffer

    def save(self, path: str | Path) -> None:
        """Save buffer tensor to disk."""
        torch.save(self._buffer, Path(path))

    def load(self, path: str | Path) -> Tensor:
        """Load buffer tensor from disk and return it."""
        loaded = torch.load(Path(path), map_location="cpu")
        if not isinstance(loaded, torch.Tensor):
            raise TypeError("Loaded cheat sheet buffer is not a tensor.")
        if loaded.shape != (self.num_vectors, self.d_model):
            raise ValueError(
                f"Expected shape {(self.num_vectors, self.d_model)}, got {tuple(loaded.shape)}"
            )
        self._buffer = loaded.detach().cpu().to(dtype=torch.float32)
        return self._buffer

    def size_bytes(self) -> int:
        """Return raw tensor memory size in bytes."""
        return int(self._buffer.numel() * self._buffer.element_size())


if __name__ == "__main__":
    torch.manual_seed(0)
    bsz, seq_len, d_model = 2, 2048, 512
    compressor = CheatSheetCompressor(d_model=d_model, num_vectors=8)
    dummy_hidden = torch.randn(bsz, seq_len, d_model)
    out = compressor(dummy_hidden)
    assert out.shape == (bsz, 8, d_model), f"Unexpected compressor output shape: {out.shape}"

    buffer = CheatSheetBuffer(d_model=d_model, num_vectors=8)
    finalized = buffer.finalize(
        model=nn.Identity(),
        compressor=compressor,
        dataloader=[dummy_hidden for _ in range(4)],
        device="cpu",
        n_docs=4,
    )
    assert finalized.shape == (8, d_model), f"Unexpected buffer shape: {finalized.shape}"
    print("CheatSheetCompressor and CheatSheetBuffer shape checks passed.")
