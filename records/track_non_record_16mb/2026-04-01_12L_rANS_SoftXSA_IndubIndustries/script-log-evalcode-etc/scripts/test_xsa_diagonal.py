"""Tests for XSA diagonal Triton kernel.

Verifies that the Triton kernel matches the naive full-attention recompute.
"""
import torch
import torch.nn.functional as F
import pytest


def _naive_xsa(q, k, v, sdpa_out, alpha):
    """Naive XSA via full attention matrix — ground truth reference."""
    B, num_heads, S, D = q.shape
    _, num_kv_heads, _, _ = k.shape
    heads_per_kv = num_heads // num_kv_heads

    # Expand K/V for GQA
    k_exp = k.repeat_interleave(heads_per_kv, dim=1)
    v_exp = v.repeat_interleave(heads_per_kv, dim=1)

    scale = 1.0 / (D ** 0.5)
    attn = torch.matmul(q.float(), k_exp.float().transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
    attn.masked_fill_(mask[None, None], float('-inf'))
    attn = F.softmax(attn, dim=-1)

    # Extract diagonal: attn_ii for each (B, H, i)
    diag = attn.diagonal(dim1=-2, dim2=-1)  # [B, H, S]
    self_val = diag.unsqueeze(-1) * v_exp.float()  # [B, H, S, D]

    result = sdpa_out.float() - alpha.float()[None, :, None, None] * self_val
    return result.to(sdpa_out.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestXSADiagonal:

    def test_basic_correctness(self):
        """Small tensor, alpha=1.0, check kernel matches naive."""
        from kernels.xsa_diagonal import xsa_subtract_self

        B, H, S, D = 2, 8, 128, 64
        KVH = 4
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        alpha = torch.ones(H, device='cuda', dtype=torch.bfloat16)

        sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        kernel_out = xsa_subtract_self(q, k, v, sdpa_out, alpha)
        naive_out = _naive_xsa(q, k, v, sdpa_out, alpha)

        max_diff = (kernel_out.float() - naive_out.float()).abs().max().item()
        print(f"basic_correctness max_diff: {max_diff:.6f}")
        assert max_diff < 0.05, f"Max diff {max_diff} exceeds bf16 tolerance"

    def test_fractional_alpha(self):
        """Alpha < 1 (soft XSA typical range)."""
        from kernels.xsa_diagonal import xsa_subtract_self

        B, H, S, D = 1, 8, 64, 64
        KVH = 4
        torch.manual_seed(123)

        q = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        alpha = torch.full((H,), 0.88, device='cuda', dtype=torch.bfloat16)

        sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        kernel_out = xsa_subtract_self(q, k, v, sdpa_out, alpha)
        naive_out = _naive_xsa(q, k, v, sdpa_out, alpha)

        max_diff = (kernel_out.float() - naive_out.float()).abs().max().item()
        print(f"fractional_alpha max_diff: {max_diff:.6f}")
        assert max_diff < 0.05, f"Max diff {max_diff} exceeds bf16 tolerance"

    def test_no_gqa(self):
        """num_heads == num_kv_heads (MHA, no GQA)."""
        from kernels.xsa_diagonal import xsa_subtract_self

        B, H, S, D = 1, 4, 64, 64
        KVH = 4  # same as H
        torch.manual_seed(7)

        q = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        alpha = torch.ones(H, device='cuda', dtype=torch.bfloat16)

        sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        kernel_out = xsa_subtract_self(q, k, v, sdpa_out, alpha)
        naive_out = _naive_xsa(q, k, v, sdpa_out, alpha)

        max_diff = (kernel_out.float() - naive_out.float()).abs().max().item()
        print(f"no_gqa max_diff: {max_diff:.6f}")
        assert max_diff < 0.05, f"Max diff {max_diff} exceeds bf16 tolerance"

    def test_zero_alpha(self):
        """Alpha=0 should return SDPA output unchanged."""
        from kernels.xsa_diagonal import xsa_subtract_self

        B, H, S, D = 1, 8, 32, 64
        KVH = 4
        torch.manual_seed(99)

        q = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        alpha = torch.zeros(H, device='cuda', dtype=torch.bfloat16)

        sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        kernel_out = xsa_subtract_self(q, k, v, sdpa_out, alpha)

        max_diff = (kernel_out.float() - sdpa_out.float()).abs().max().item()
        print(f"zero_alpha max_diff: {max_diff:.6f}")
        assert max_diff == 0.0, f"Alpha=0 should be identity, got max_diff={max_diff}"

    def test_per_head_alpha(self):
        """Different alpha per head."""
        from kernels.xsa_diagonal import xsa_subtract_self

        B, H, S, D = 1, 8, 64, 64
        KVH = 4
        torch.manual_seed(55)

        q = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        alpha = torch.tensor([0.0, 0.5, 0.88, 1.0, 1.2, 0.3, 0.95, 0.1],
                             device='cuda', dtype=torch.bfloat16)

        sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        kernel_out = xsa_subtract_self(q, k, v, sdpa_out, alpha)
        naive_out = _naive_xsa(q, k, v, sdpa_out, alpha)

        max_diff = (kernel_out.float() - naive_out.float()).abs().max().item()
        print(f"per_head_alpha max_diff: {max_diff:.6f}")
        assert max_diff < 0.05, f"Max diff {max_diff} exceeds bf16 tolerance"

    def test_seq_len_1(self):
        """Edge case: sequence length 1 (only self-attention possible)."""
        from kernels.xsa_diagonal import xsa_subtract_self

        B, H, S, D = 1, 4, 1, 64
        KVH = 2
        torch.manual_seed(0)

        q = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, KVH, S, D, device='cuda', dtype=torch.bfloat16)
        alpha = torch.ones(H, device='cuda', dtype=torch.bfloat16)

        sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        kernel_out = xsa_subtract_self(q, k, v, sdpa_out, alpha)
        naive_out = _naive_xsa(q, k, v, sdpa_out, alpha)

        max_diff = (kernel_out.float() - naive_out.float()).abs().max().item()
        print(f"seq_len_1 max_diff: {max_diff:.6f}")
        assert max_diff < 0.01, f"Max diff {max_diff} exceeds tolerance for S=1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
