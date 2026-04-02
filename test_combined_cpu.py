#!/usr/bin/env python3
"""
CPU E2E test: Scylla base + Parallel Residuals + Mini Depth Recurrence

Tests that the two new architectural features (from PR #1204) integrate
cleanly with the Scylla base (from PR #1143). Validates:
  1. Parallel residuals: attn/MLP write to separate residual lanes
  2. Mini depth recurrence: layers 4,5 repeated with untied MLPs
  3. Forward pass produces valid logits
  4. Backward pass computes gradients
  5. Parameter count stays within budget expectations
  6. Recurrent+parallel overlap works correctly
"""

import math
import sys
import time
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ============================================================================
# Minimal reproduction of Scylla base architecture components
# ============================================================================

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = rope_dims if rope_dims > 0 else dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0
        self._cos_cache = None
        self._sin_cache = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len <= self._cache_len and self._cos_cache is not None:
            return
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)
        self._cache_len = seq_len

    def forward(self, x: Tensor):
        seq_len = x.size(-2)
        self._build_cache(seq_len, x.device, x.dtype)
        cos = self._cos_cache[:seq_len]
        sin = self._sin_cache[:seq_len]
        d = cos.shape[-1]
        x_rope, x_pass = x[..., :d], x[..., d:]
        x1 = x_rope[..., : d // 2]
        x2 = x_rope[..., d // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rope = x_rope * cos + rotated * sin
        return torch.cat((x_rope, x_pass), dim=-1) if x_pass.numel() > 0 else x_rope


class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class SmearGate(nn.Module):
    """[W4 fix] Caches self.fc(x) instead of computing twice."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.fc.weight)

    def forward(self, x: Tensor) -> Tensor:
        g = self.fc(x)
        return x + g * torch.sigmoid(g)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rotary = Rotary(self.head_dim, base=rope_base, rope_dims=16)
        self.rope_dims = 16
        self.qk_gain = nn.Parameter(torch.full((1, num_heads, 1, 1), qk_gain_init))
        self.use_xsa = False

    def forward(self, x_normed, q_w, k_w, v_w, out_w, v_embed=None, v0=None):
        B, T, C = x_normed.shape
        q = F.linear(x_normed, q_w).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = F.linear(x_normed, k_w).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = F.linear(x_normed, v_w).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q)
        k = self.rotary(k)
        q = q * self.qk_gain

        # Expand KV for GQA
        reps = self.num_heads // self.num_kv_heads
        if reps > 1:
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        return F.linear(attn, out_w), v[:, :self.num_kv_heads]


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.dim = dim
        self.mlp_dim = int(mlp_mult * dim)

    def forward(self, x_normed, up_w, down_w):
        x = F.linear(x_normed, up_w)
        # LeakyReLU(0.5)^2
        x = F.leaky_relu(x, negative_slope=0.5)
        x = x * x
        return F.linear(x, down_w)


class BigramHashEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, model_dim, bias=False)
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        bigram_ids = input_ids[:, :-1] * 32 + input_ids[:, 1:]  # simplified hash
        bigram_ids = bigram_ids % self.vocab_size
        bigram_ids = torch.cat([torch.zeros_like(input_ids[:, :1]), bigram_ids], dim=1)
        return self.proj(self.embed(bigram_ids))


# ============================================================================
# Parallel Residuals (from PR #1204 / modded-nanogpt #230)
# ============================================================================

class Block(nn.Module):
    """Transformer block with optional parallel residual routing."""

    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                 qk_gain_init, layer_idx=0, ln_scale=False, parallel=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = parallel

        if parallel:
            # [W1 fix] Separate resid_mix for MLP lane in parallel mode
            self.resid_mix_mlp = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
            # 4 learned routing scalars: attn_to_attn, attn_to_mlp, mlp_to_attn, mlp_to_mlp
            self.route = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))

    def forward(self, x_attn, x_mlp, x0, q_w, k_w, v_w, out_w, up_w, down_w):
        if not self.parallel:
            # Standard sequential path
            x = x_attn  # x_attn == x_mlp in non-parallel mode
            mix = self.resid_mix.to(dtype=x.dtype)
            x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            attn_out, raw_v = self.attn(self.attn_norm(x_in) * self.ln_scale_factor,
                                        q_w, k_w, v_w, out_w)
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * \
                    self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
            return x_out, x_out, raw_v
        else:
            # Parallel residual path
            r = self.route.to(dtype=x_attn.dtype)

            # Attn reads from attn lane (uses resid_mix)
            mix_attn = self.resid_mix.to(dtype=x_attn.dtype)
            x_in_attn = mix_attn[0][None, None, :] * x_attn + mix_attn[1][None, None, :] * x0
            attn_out, raw_v = self.attn(self.attn_norm(x_in_attn) * self.ln_scale_factor,
                                        q_w, k_w, v_w, out_w)
            attn_delta = self.attn_scale.to(dtype=x_attn.dtype)[None, None, :] * attn_out

            # [W1 fix] MLP reads from mlp lane (uses separate resid_mix_mlp)
            mix_mlp = self.resid_mix_mlp.to(dtype=x_mlp.dtype)
            x_in_mlp = mix_mlp[0][None, None, :] * x_mlp + mix_mlp[1][None, None, :] * x0
            mlp_delta = self.mlp_scale.to(dtype=x_mlp.dtype)[None, None, :] * \
                        self.mlp(self.mlp_norm(x_in_mlp) * self.ln_scale_factor, up_w, down_w)

            # Cross-write: each sublayer writes to both lanes
            x_attn_out = x_attn + r[0] * attn_delta + r[2] * mlp_delta
            x_mlp_out = x_mlp + r[1] * attn_delta + r[3] * mlp_delta

            return x_attn_out, x_mlp_out, raw_v


# ============================================================================
# GPT with Parallel Residuals + Mini Depth Recurrence
# ============================================================================

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool = True,
        logit_softcap: float = 30.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        ln_scale: bool = True,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 4,
        # Parallel residuals
        parallel_start_layer: int = 0,  # 0 = disabled, 7 = from layer 7 onwards
        # Mini depth recurrence
        recur_layers: str = "",          # e.g. "4,5" — which layers to repeat
        recur_untie_mlp: bool = True,    # untie MLP weights for repeated pass
    ):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.logit_softcap = logit_softcap
        self.tie_embeddings = tie_embeddings
        self.parallel_start_layer = parallel_start_layer
        # [W3] Layer 5 recurrence intentionally happens post-skip (in decoder).
        # This places recurrence around the U-Net hinge (layers 4,5) per PR #1204.
        self.recur_layer_ids = [int(x) for x in recur_layers.split(",") if x.strip()] if recur_layers else []
        self.recur_untie_mlp = recur_untie_mlp

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)

        # U-Net encoder/decoder split
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # [W2] Parallel must not start inside encoder — skip connections only store attn lane
        assert parallel_start_layer == 0 or parallel_start_layer >= self.num_encoder_layers, \
            f"parallel_start_layer={parallel_start_layer} is inside encoder (layers 0-{self.num_encoder_layers-1}); " \
            f"skip connections only store attn lane, so parallel must start at encoder boundary or later"

        # Parameter banks
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)
        self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))

        # Blocks — mark parallel layers
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            is_parallel = parallel_start_layer > 0 and i >= parallel_start_layer
            self.blocks.append(Block(
                model_dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                qk_gain_init, layer_idx=i, ln_scale=ln_scale, parallel=is_parallel,
            ))

        # XSA on last N layers
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True

        # Mini depth recurrence: untied MLP banks for repeated layers
        if self.recur_layer_ids and recur_untie_mlp:
            self.recur_mlp_up = nn.ParameterDict()
            self.recur_mlp_down = nn.ParameterDict()
            for lid in self.recur_layer_ids:
                self.recur_mlp_up[str(lid)] = nn.Parameter(torch.empty(mlp_dim, model_dim))
                self.recur_mlp_down[str(lid)] = nn.Parameter(torch.empty(model_dim, mlp_dim))
                nn.init.orthogonal_(self.recur_mlp_up[str(lid)])
                nn.init.zeros_(self.recur_mlp_down[str(lid)])
        else:
            self.recur_mlp_up = None
            self.recur_mlp_down = None

        # [I3] Learnable lane merge (only when parallel is enabled)
        if parallel_start_layer > 0:
            self.lane_merge = nn.Parameter(torch.tensor(0.5))
        else:
            self.lane_merge = None

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else nn.Linear(model_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[n + i])
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.qo_bank.data[n + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)

    def _run_block(self, block, i, x_attn, x_mlp, x0, up_w=None, down_w=None):
        """Run a single block with the correct bank weights."""
        n = self.num_layers
        if up_w is None:
            up_w = self.mlp_up_bank[i]
        if down_w is None:
            down_w = self.mlp_down_bank[i]
        return block(
            x_attn, x_mlp, x0,
            self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
            self.qo_bank[n + i], up_w, down_w,
        )

    # [W5 fix] Shared encoder/decoder loop — eliminates duplication between forward and forward_logits
    def _run_layers(self, input_ids: Tensor) -> Tensor:
        """Run embeddings + encoder + decoder + lane merge + final norm. Returns hidden state."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        x_attn = x
        x_mlp = x
        skips: list[Tensor] = []

        # ENCODER
        for i in range(self.num_encoder_layers):
            x_attn, x_mlp, _ = self._run_block(self.blocks[i], i, x_attn, x_mlp, x0)
            if i in self.recur_layer_ids:
                up_w = self.recur_mlp_up[str(i)] if self.recur_mlp_up else self.mlp_up_bank[i]
                down_w = self.recur_mlp_down[str(i)] if self.recur_mlp_down else self.mlp_down_bank[i]
                x_attn, x_mlp, _ = self._run_block(self.blocks[i], i, x_attn, x_mlp, x0,
                                                     up_w=up_w, down_w=down_w)
            skips.append(x_attn)

        # DECODER
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                skip_val = self.skip_weights[i].to(dtype=x_attn.dtype)[None, None, :] * skips.pop()
                x_attn = x_attn + skip_val
                if self.blocks[bi].parallel:
                    x_mlp = x_mlp + skip_val

            x_attn, x_mlp, _ = self._run_block(self.blocks[bi], bi, x_attn, x_mlp, x0)

            # [W3] Depth recurrence in decoder — layer 5 post-skip is intentional per PR #1204
            if bi in self.recur_layer_ids:
                up_w = self.recur_mlp_up[str(bi)] if self.recur_mlp_up else self.mlp_up_bank[bi]
                down_w = self.recur_mlp_down[str(bi)] if self.recur_mlp_down else self.mlp_down_bank[bi]
                x_attn, x_mlp, _ = self._run_block(self.blocks[bi], bi, x_attn, x_mlp, x0,
                                                     up_w=up_w, down_w=down_w)

        # [I3] Learnable lane merge instead of hardcoded 0.5/0.5 average
        if self.lane_merge is not None:
            m = self.lane_merge.to(dtype=x_attn.dtype)
            x = m * x_attn + (1.0 - m) * x_mlp
        else:
            x = x_attn  # x_attn == x_mlp when no parallel

        return self.final_norm(x)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._run_layers(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1), reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits without computing loss — for eval."""
        x = self._run_layers(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# ============================================================================
# [C2 fix] Mixed quantization budget calculator
# ============================================================================

def compute_mixed_quant_budget(model: GPT) -> dict:
    """Estimate artifact size using mixed quantization (INT4 MLP, INT6 attn, INT8 rest).

    Based on PR #1105 mixed quantization approach and our sensitivity scan showing
    MLP banks are least sensitive to low-bit quantization.
    """
    int4_params = 0  # MLP banks — least sensitive
    int6_params = 0  # Attention banks
    int8_params = 0  # Everything else (embeds, scales, routes, etc.)

    for name, param in model.named_parameters():
        n = param.numel()
        if any(k in name for k in ['mlp_up_bank', 'mlp_down_bank', 'recur_mlp_up', 'recur_mlp_down']):
            int4_params += n
        elif any(k in name for k in ['qo_bank', 'kv_bank']):
            int6_params += n
        else:
            int8_params += n

    raw_bytes = int4_params * 0.5 + int6_params * 0.75 + int8_params * 1.0
    brotli_bytes = raw_bytes * 0.85
    code_overhead = 100_000
    total = int(brotli_bytes) + code_overhead

    return {
        'int4_params': int4_params,
        'int6_params': int6_params,
        'int8_params': int8_params,
        'total_params': int4_params + int6_params + int8_params,
        'raw_bytes': raw_bytes,
        'brotli_bytes': brotli_bytes,
        'total_artifact': total,
        'fits_16mb': total < 16_000_000,
    }


# ============================================================================
# TESTS
# ============================================================================

def test_baseline():
    """Test: Scylla base without new features (regression check)."""
    print("=" * 60)
    print("  TEST 1: Baseline (no parallel, no recurrence)")
    print("=" * 60)

    model = GPT(
        vocab_size=1024, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3, ln_scale=True, bigram_vocab_size=2048,
        bigram_dim=128, xsa_last_n=4,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    input_ids = torch.randint(0, 1024, (2, 64))
    target_ids = torch.randint(0, 1024, (2, 64))
    loss = model(input_ids, target_ids)
    loss.backward()
    print(f"  Forward loss: {loss.item():.4f}")
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Grad norm: {grad_norm:.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert grad_norm > 0, "Gradients should flow"

    logits = model.forward_logits(input_ids)
    assert logits.shape == (2, 64, 1024), f"Expected (2, 64, 1024), got {logits.shape}"
    print("  ✓ Baseline pass\n")
    return n_params


def test_parallel_residuals():
    """Test: Parallel residuals from layer 7 onwards."""
    print("=" * 60)
    print("  TEST 2: Parallel Residuals (from layer 7)")
    print("=" * 60)

    model = GPT(
        vocab_size=1024, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3, ln_scale=True, bigram_vocab_size=2048,
        bigram_dim=128, xsa_last_n=4,
        parallel_start_layer=7,
    )
    n_params = sum(p.numel() for p in model.parameters())

    # Verify parallel blocks are correctly marked
    parallel_count = sum(1 for b in model.blocks if b.parallel)
    expected_parallel = 11 - 7  # layers 7,8,9,10
    assert parallel_count == expected_parallel, f"Expected {expected_parallel} parallel blocks, got {parallel_count}"
    print(f"  Params: {n_params:,} ({parallel_count} parallel blocks)")

    # Check route and resid_mix_mlp params exist
    for i in range(7, 11):
        assert hasattr(model.blocks[i], 'route'), f"Block {i} missing route param"
        assert model.blocks[i].route.shape == (4,), f"Block {i} route wrong shape"
        assert hasattr(model.blocks[i], 'resid_mix_mlp'), f"Block {i} missing resid_mix_mlp"

    # Check learnable lane merge exists
    assert model.lane_merge is not None, "lane_merge should exist when parallel enabled"

    input_ids = torch.randint(0, 1024, (2, 64))
    target_ids = torch.randint(0, 1024, (2, 64))
    loss = model(input_ids, target_ids)
    loss.backward()
    print(f"  Forward loss: {loss.item():.4f}")

    # Check route gradients exist (may be zero at step 0 due to zero-init projections)
    for i in range(7, 11):
        assert model.blocks[i].route.grad is not None, f"Block {i} route has no grad"

    # [C1 fix] Route grads are zero at step 0 (downstream projections are zero-init).
    # Verify they become nonzero after 1 optimizer step.
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.step()
    opt.zero_grad()
    loss2 = model(input_ids, target_ids)
    loss2.backward()
    for i in range(7, 11):
        grad_sum = model.blocks[i].route.grad.abs().sum().item()
        assert grad_sum > 0, f"Block {i} route grad still zero after 1 step"
    print(f"  Route grads nonzero after 1 step: ✓")

    logits = model.forward_logits(input_ids)
    assert logits.shape == (2, 64, 1024)
    print("  ✓ Parallel residuals pass\n")
    return n_params


def test_depth_recurrence():
    """Test: Mini depth recurrence on layers 4,5."""
    print("=" * 60)
    print("  TEST 3: Mini Depth Recurrence (layers 4,5)")
    print("=" * 60)

    model = GPT(
        vocab_size=1024, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3, ln_scale=True, bigram_vocab_size=2048,
        bigram_dim=128, xsa_last_n=4,
        recur_layers="4,5", recur_untie_mlp=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Verify untied MLP params exist
    assert model.recur_mlp_up is not None
    assert "4" in model.recur_mlp_up
    assert "5" in model.recur_mlp_up
    recur_params = sum(p.numel() for p in model.recur_mlp_up.values()) + \
                   sum(p.numel() for p in model.recur_mlp_down.values())
    print(f"  Recurrence extra params: {recur_params:,} (untied MLPs for layers 4,5)")

    input_ids = torch.randint(0, 1024, (2, 64))
    target_ids = torch.randint(0, 1024, (2, 64))
    loss = model(input_ids, target_ids)
    loss.backward()
    print(f"  Forward loss: {loss.item():.4f}")

    # Verify untied MLP grads flow
    for lid in ["4", "5"]:
        assert model.recur_mlp_up[lid].grad is not None, f"recur_mlp_up[{lid}] no grad"
        assert model.recur_mlp_down[lid].grad is not None, f"recur_mlp_down[{lid}] no grad"
    print(f"  Recur MLP grads: ✓")

    logits = model.forward_logits(input_ids)
    assert logits.shape == (2, 64, 1024)
    print("  ✓ Depth recurrence pass\n")
    return n_params


def test_combined():
    """Test: All features combined — the actual submission config."""
    print("=" * 60)
    print("  TEST 4: COMBINED (parallel + recurrence + all features)")
    print("=" * 60)

    model = GPT(
        vocab_size=1024, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3, ln_scale=True,
        bigram_vocab_size=2048, bigram_dim=128, xsa_last_n=4,
        parallel_start_layer=7,
        recur_layers="4,5", recur_untie_mlp=True,
    )
    n_params = sum(p.numel() for p in model.parameters())

    # Virtual layer count: 11 physical + 2 recurrence = 13 virtual
    virtual_layers = 11 + len(model.recur_layer_ids)
    parallel_count = sum(1 for b in model.blocks if b.parallel)
    print(f"  Params: {n_params:,}")
    print(f"  Architecture: {11}L physical → {virtual_layers}L virtual")
    print(f"  Parallel blocks: {parallel_count} (layers 7-10)")
    print(f"  Recurrent blocks: {len(model.recur_layer_ids)} (layers 4,5, untied MLP)")

    input_ids = torch.randint(0, 1024, (2, 64))
    target_ids = torch.randint(0, 1024, (2, 64))

    # Multiple forward passes to check stability
    losses = []
    for step in range(3):
        model.zero_grad()
        loss = model(input_ids, target_ids)
        loss.backward()
        losses.append(loss.item())

    print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
    assert all(l > 0 for l in losses), "All losses should be positive"
    assert abs(losses[0] - losses[1]) < 1e-5, "Same input should give same loss"

    # forward_logits path
    logits = model.forward_logits(input_ids)
    assert logits.shape == (2, 64, 1024)

    # [C2 fix] Mixed quantization budget estimate
    budget = compute_mixed_quant_budget(model)
    print(f"  Budget breakdown:")
    print(f"    INT4 (MLP):  {budget['int4_params']:>10,} params → {budget['int4_params']*0.5/1e6:.1f}MB")
    print(f"    INT6 (attn): {budget['int6_params']:>10,} params → {budget['int6_params']*0.75/1e6:.1f}MB")
    print(f"    INT8 (rest): {budget['int8_params']:>10,} params → {budget['int8_params']*1.0/1e6:.1f}MB")
    print(f"    Raw: {budget['raw_bytes']/1e6:.1f}MB → Brotli: {budget['brotli_bytes']/1e6:.1f}MB → Total: {budget['total_artifact']/1e6:.1f}MB")
    print(f"    16MB budget: {'✓ fits' if budget['fits_16mb'] else '✗ OVER BUDGET'}")
    assert budget['fits_16mb'], f"Artifact {budget['total_artifact']:,} bytes exceeds 16MB budget"

    print("  ✓ Combined pass\n")
    return n_params


def test_gradient_consistency():
    """Test: Gradients through parallel+recurrence match expectations."""
    print("=" * 60)
    print("  TEST 5: Gradient Consistency Check")
    print("=" * 60)

    model = GPT(
        vocab_size=1024, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3, ln_scale=True,
        bigram_vocab_size=2048, bigram_dim=128, xsa_last_n=4,
        parallel_start_layer=7, recur_layers="4,5", recur_untie_mlp=True,
    )

    input_ids = torch.randint(0, 1024, (2, 64))
    target_ids = torch.randint(0, 1024, (2, 64))
    loss = model(input_ids, target_ids)
    loss.backward()

    # Bank params that are NOT zero-initialized should have gradients.
    # Note: Out proj (qo_bank[n:]) and mlp_down_bank are zero-init by design,
    # which means kv_bank and mlp_up_bank have zero grad at step 0 — this is
    # expected "zero init residual" behavior, not a bug.
    banks_with_grad = ['qo_bank', 'mlp_down_bank']
    banks_zero_init = ['kv_bank', 'mlp_up_bank']
    for name in banks_with_grad:
        param = getattr(model, name)
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient should be nonzero"
        print(f"  {name}: grad norm = {param.grad.norm().item():.6f} ✓")
    for name in banks_zero_init:
        param = getattr(model, name)
        assert param.grad is not None, f"{name} has no gradient"
        print(f"  {name}: grad norm = {param.grad.norm().item():.6f} (zero-init expected)")

    # Simulate 1 optimizer step then re-check — gradients should flow after projections become non-zero
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.step()
    opt.zero_grad()
    loss2 = model(input_ids, target_ids)
    loss2.backward()
    for name in banks_zero_init:
        param = getattr(model, name)
        grad_sum = param.grad.abs().sum().item()
        assert grad_sum > 0, f"{name} still zero after 1 step — real bug"
        print(f"  {name}: grad norm = {param.grad.norm().item():.6f} ✓ (after 1 step)")

    # Skip weights should have gradients
    assert model.skip_weights.grad is not None
    print(f"  skip_weights: grad norm = {model.skip_weights.grad.norm().item():.6f}")

    # Route params should have nonzero gradients (after 1 step)
    for i in range(7, 11):
        r = model.blocks[i].route
        assert r.grad is not None, f"Block {i} route no gradient"
        assert r.grad.abs().sum() > 0, f"Block {i} route grad is zero after 1 step"

    # Lane merge should have gradient
    assert model.lane_merge is not None and model.lane_merge.grad is not None
    print(f"  lane_merge: grad = {model.lane_merge.grad.item():.6f}")

    print(f"  Route params: all have nonzero gradients ✓")
    print("  ✓ Gradient consistency pass\n")


def test_recurrent_and_parallel():
    """[I2] Test: A layer that is both recurrent and parallel works correctly."""
    print("=" * 60)
    print("  TEST 6: Recurrent + Parallel overlap (layer 7)")
    print("=" * 60)

    model = GPT(
        vocab_size=1024, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3, ln_scale=True,
        bigram_vocab_size=2048, bigram_dim=128, xsa_last_n=4,
        parallel_start_layer=7,
        recur_layers="4,5,7", recur_untie_mlp=True,  # layer 7 is both parallel AND recurrent
    )

    # Verify layer 7 is both parallel and recurrent
    assert model.blocks[7].parallel, "Layer 7 should be parallel"
    assert 7 in model.recur_layer_ids, "Layer 7 should be recurrent"
    assert "7" in model.recur_mlp_up, "Layer 7 should have untied MLP"
    print(f"  Layer 7: parallel=True, recurrent=True, untied_mlp=True")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    input_ids = torch.randint(0, 1024, (2, 64))
    target_ids = torch.randint(0, 1024, (2, 64))
    loss = model(input_ids, target_ids)
    loss.backward()
    print(f"  Forward loss: {loss.item():.4f}")

    # Gradients should flow through both the parallel routing and the recurrence
    assert model.blocks[7].route.grad is not None, "Route grad missing"
    assert model.recur_mlp_up["7"].grad is not None, "Recur MLP up grad missing"
    assert model.recur_mlp_down["7"].grad is not None, "Recur MLP down grad missing"

    # forward_logits should also work
    logits = model.forward_logits(input_ids)
    assert logits.shape == (2, 64, 1024)
    print("  ✓ Recurrent + Parallel pass\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  COMBINED STACK CPU E2E TEST")
    print("  Scylla base + Parallel Residuals + Depth Recurrence")
    print("  Audit findings: C1, C2, W1-W5, I1-I3 all addressed")
    print("=" * 60 + "\n")

    t0 = time.time()

    baseline_params = test_baseline()
    parallel_params = test_parallel_residuals()
    recur_params = test_depth_recurrence()
    combined_params = test_combined()
    test_gradient_consistency()
    test_recurrent_and_parallel()

    elapsed = time.time() - t0

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Baseline params:     {baseline_params:>12,}")
    print(f"  + Parallel residuals:{parallel_params:>12,}  (delta: +{parallel_params - baseline_params:,})")
    print(f"  + Depth recurrence:  {recur_params:>12,}  (delta: +{recur_params - baseline_params:,})")
    print(f"  Combined:            {combined_params:>12,}  (delta: +{combined_params - baseline_params:,})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"\n  ALL 6 TESTS PASSED ✓\n")
