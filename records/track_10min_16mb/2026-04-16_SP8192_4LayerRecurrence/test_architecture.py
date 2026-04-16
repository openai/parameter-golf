"""
Smoke-test for the 4-layer depth recurrence architecture.

Runs on CPU with standard PyTorch (no flash_attn, no CUDA, no data).
Verifies that:
  1. Virtual layer sequences match the expected encoder/decoder paths
  2. Forward pass produces correct output shapes
  3. Gradients flow cleanly through all blocks (including looped blocks)
  4. SOTA (loop_end=5) and this submission (loop_end=6) both pass

Usage:
    python test_architecture.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.ones(num_heads))

    def forward(self, x):
        B, T, D = x.shape
        hd = self.head_dim
        q = F.rms_norm(self.c_q(x).reshape(B, T, self.num_heads, hd), (hd,))
        k = F.rms_norm(self.c_k(x).reshape(B, T, self.num_kv_heads, hd), (hd,))
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, hd)
        q = q * self.q_gain[None, None, :, None]
        grp = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(grp, dim=2)
        v = v.repeat_interleave(grp, dim=2)
        y = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
            is_causal=True,
        ).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, layer_idx):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = Attention(dim, num_heads, num_kv_heads)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1)
        self.parallel = False

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0] * x + mix[1] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            return x_in + self.attn_scale.to(x_in.dtype) * attn_out + self.mlp_scale.to(x_in.dtype) * mlp_out
        x2 = x_in + self.attn_scale.to(x_in.dtype) * attn_out
        return x2 + self.mlp_scale.to(x2.dtype) * self.mlp(self.mlp_norm(x2) * self.ln_scale_factor)


class GPTRecurrent(nn.Module):
    def __init__(self, vocab_size=256, num_layers=11, dim=64,
                 num_heads=8, num_kv_heads=4, mlp_mult=4,
                 loop_start=3, loop_end=5, num_loops=2,
                 parallel_residual_start=7):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, num_kv_heads, mlp_mult, i)
            for i in range(num_layers)
        ])
        for i in range(num_layers):
            self.blocks[i].parallel = (i >= parallel_residual_start)

        loop_seg = list(range(loop_start, loop_end + 1))
        all_idx = list(range(loop_start))
        for _ in range(num_loops + 1):
            all_idx.extend(loop_seg)
        all_idx.extend(range(loop_end + 1, num_layers))

        num_enc = len(all_idx) // 2
        self.encoder_indices = all_idx[:num_enc]
        self.decoder_indices = all_idx[num_enc:]
        self.num_skip = min(len(self.encoder_indices), len(self.decoder_indices))

        self.skip_weights = nn.Parameter(torch.ones(self.num_skip, dim))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip, dim))
        self.tok_emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.tok_emb.weight, std=0.005)

    def forward(self, input_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x0 = x
        skips = []
        for i in self.encoder_indices:
            x = self.blocks[i](x, x0)
            skips.append(x)
        for skip_idx, i in enumerate(self.decoder_indices):
            if skip_idx < self.num_skip and skips:
                sk = self.skip_weights[skip_idx].to(x.dtype) * skips.pop()
                g = torch.sigmoid(self.skip_gates[skip_idx].to(x.dtype))
                x = torch.lerp(sk, x, g)
            x = self.blocks[i](x, x0)
        return F.linear(x, self.tok_emb.weight)


def run_test(name, loop_end, expected_virtual_layers, expected_skips):
    print(f"\n{'='*60}")
    print(f"Config: {name}")

    model = GPTRecurrent(loop_end=loop_end)
    vl = len(model.encoder_indices) + len(model.decoder_indices)
    assert vl == expected_virtual_layers, f"Expected {expected_virtual_layers} virtual layers, got {vl}"
    assert model.num_skip == expected_skips, f"Expected {expected_skips} skips, got {model.num_skip}"

    print(f"  encoder: {model.encoder_indices}")
    print(f"  decoder: {model.decoder_indices}")
    print(f"  virtual_layers={vl}, skips={model.num_skip}")

    # Forward pass
    B, T = 2, 32
    ids = torch.randint(0, 256, (B, T))
    logits = model(ids)
    assert logits.shape == (B, T, 256), f"Bad output shape: {logits.shape}"
    assert not logits.isnan().any(), "NaN in forward pass"
    print(f"  forward pass: OK  shape={logits.shape}")

    # Gradient flow
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1))
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    has_nan = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.isnan().any()]
    assert not no_grad, f"Missing gradients: {no_grad}"
    assert not has_nan, f"NaN gradients: {has_nan}"
    print(f"  gradient flow: OK  loss={loss.item():.4f}")

    # Verify blocks in looped section receive gradients (key check for recurrence)
    looped = set(model.encoder_indices + model.decoder_indices)
    for block_idx in range(11):
        if block_idx in looped:
            for p in model.blocks[block_idx].parameters():
                assert p.grad is not None and not p.grad.isnan().any()
    print(f"  all looped blocks have clean gradients: OK")


if __name__ == "__main__":
    run_test("SOTA (loop_end=5)", loop_end=5, expected_virtual_layers=17, expected_skips=8)
    run_test("This PR (loop_end=6)", loop_end=6, expected_virtual_layers=19, expected_skips=9)
    print(f"\n{'='*60}")
    print("All architecture tests PASSED.")
