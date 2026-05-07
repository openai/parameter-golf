from __future__ import annotations

import torch
from torch import Tensor


class PrefixInjector:
    """
    Prepends fixed cheat-sheet vectors to token embeddings during phase 2.
    """

    def __init__(self, prefix_len: int = 8) -> None:
        self.prefix_len = prefix_len

    def inject(self, embeddings: Tensor, cheat_sheet: Tensor, attention_mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """
        Concatenate prefix to embeddings and expand optional attention mask.
        """
        assert embeddings.ndim == 3, f"embeddings must be [B,T,D], got {tuple(embeddings.shape)}"
        assert cheat_sheet.ndim == 2, f"cheat_sheet must be [P,D], got {tuple(cheat_sheet.shape)}"
        bsz, seq_len, d_model = embeddings.shape
        p_len, c_dim = cheat_sheet.shape
        assert p_len == self.prefix_len, f"expected prefix_len={self.prefix_len}, got {p_len}"
        assert c_dim == d_model, f"cheat_sheet dim {c_dim} != embeddings dim {d_model}"

        fixed_prefix = cheat_sheet.detach().to(device=embeddings.device, dtype=embeddings.dtype)
        expanded_prefix = fixed_prefix.unsqueeze(0).expand(bsz, -1, -1)
        out_embeddings = torch.cat([expanded_prefix, embeddings], dim=1)
        assert out_embeddings.shape == (bsz, seq_len + self.prefix_len, d_model)

        if attention_mask is None:
            return out_embeddings, None

        assert attention_mask.ndim == 4, f"attention_mask must be [B,1,T,T], got {tuple(attention_mask.shape)}"
        assert attention_mask.shape[0] in (1, bsz), "attention_mask batch must be 1 or B"
        assert attention_mask.shape[-1] == seq_len and attention_mask.shape[-2] == seq_len

        full_t = seq_len + self.prefix_len
        idx_i = torch.arange(full_t, device=embeddings.device).view(full_t, 1)
        idx_j = torch.arange(full_t, device=embeddings.device).view(1, full_t)
        causal = idx_j <= idx_i
        prefix_ok = idx_i < self.prefix_len
        allowed = causal | prefix_ok
        neg = torch.tensor(float("-inf"), device=embeddings.device, dtype=attention_mask.dtype)
        z = torch.zeros((), device=embeddings.device, dtype=attention_mask.dtype)
        new_mask_2d = torch.where(allowed, z, neg)
        new_mask = new_mask_2d.view(1, 1, full_t, full_t).expand(bsz, -1, -1, -1)
        return out_embeddings, new_mask

    def remove_prefix(self, hidden_states: Tensor) -> Tensor:
        """
        Remove prepended prefix positions from hidden states.
        """
        assert hidden_states.ndim == 3, f"hidden_states must be [B,T,D], got {tuple(hidden_states.shape)}"
        assert hidden_states.shape[1] >= self.prefix_len, "hidden_states shorter than prefix_len"
        return hidden_states[:, self.prefix_len :, :]

    def verify_no_gradient_leak(self, cheat_sheet: Tensor) -> None:
        """
        Ensure cheat sheet is fixed during phase 2.
        """
        if cheat_sheet.requires_grad:
            raise RuntimeError("Cheat sheet vectors must have requires_grad=False during phase 2.")


if __name__ == "__main__":
    torch.manual_seed(0)
    bsz, seq_len, d_model, prefix_len = 2, 512, 64, 8
    injector = PrefixInjector(prefix_len=prefix_len)
    embeddings = torch.randn(bsz, seq_len, d_model)
    cheat_sheet = torch.randn(prefix_len, d_model, requires_grad=False)

    attn = torch.zeros(bsz, 1, seq_len, seq_len)
    out_e, out_m = injector.inject(embeddings, cheat_sheet, attn)
    assert out_e.shape == (bsz, seq_len + prefix_len, d_model)
    assert out_m is not None and out_m.shape == (bsz, 1, seq_len + prefix_len, seq_len + prefix_len)

    hidden = torch.randn(bsz, seq_len + prefix_len, d_model)
    stripped = injector.remove_prefix(hidden)
    assert stripped.shape == (bsz, seq_len, d_model)
    injector.verify_no_gradient_leak(cheat_sheet)
    print("PrefixInjector shape/gradient checks passed.")
