"""
Parameter Golf Autoresearch Training Script — Single-GPU, Single-File.
Adapted from the Parameter Golf SOTA submissions for autonomous experimentation.

This is the ONLY file the AI agent modifies. Everything is fair game:
architecture, optimizer, hyperparameters, training loop, batch size, model size.

Usage: python train_pgolf.py
       python train_pgolf.py --smoke-test
"""

import argparse
import gc
import io
import math
import os
import time
import zlib
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_pgolf import (
    TIME_BUDGET,
    TRAIN_SEQ_LEN,
    VOCAB_SIZE,
    MAX_ARTIFACT_BYTES,
    DATA_PATH,
    TOKENIZER_PATH,
    load_validation_tokens,
    get_tokenizer,
    build_sentencepiece_luts,
    DistributedTokenLoader,
    eval_val,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    measure_artifact_size,
)

# ---------------------------------------------------------------------------
# Hyperparameters (EDIT THESE — this is what the agent tunes)
# ---------------------------------------------------------------------------

# Model architecture
NUM_LAYERS = 8
MODEL_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4           # GQA: fewer KV heads than Q heads
MLP_MULT = 2               # MLP hidden = MODEL_DIM * MLP_MULT
TIE_EMBEDDINGS = True
ROPE_BASE = 10000
LOGIT_SOFTCAP = 30.0

# Training
SEQ_LEN = 1024             # training sequence length
TRAIN_BATCH_TOKENS = 524288  # total tokens per optimizer step
DEVICE_BATCH_SIZE = 64       # micro-batch size (8 for 8GB VRAM laptop GPU)
WARMUP_STEPS = 0
WARMDOWN_ITERS = 3000       # LR warmdown iterations (wallclock-based)

# Optimizer
MATRIX_LR = 0.02            # Muon LR for matrix params
SCALAR_LR = 0.04            # Adam LR for scalars
EMBED_LR = 0.06             # Adam LR for embeddings
MUON_MOMENTUM = 0.95
MUON_MOMENTUM_WARMUP_STEPS = 1500
MUON_MOMENTUM_WARMUP_START = 0.85
WEIGHT_DECAY = 0.0          # 0.04 for best results but keep 0 as baseline
ADAM_BETAS = (0.90, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP_NORM = 1.0

# Evaluation
EVAL_STRIDE = 0             # 0 = standard eval, 64 = sliding window
EVAL_BATCH_SEQS = 4         # batch size for sliding window eval


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x.float(), self.scale.shape, self.scale, 1e-6).to(x.dtype)


class CastedLinear(nn.Linear):
    """Linear layer that keeps weights in fp32 for optimizer quality, casts for forward."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, :, None, :], persistent=False)

    def forward(self, x):
        d = x.shape[-1] // 2
        cos = self.cos[:, :x.shape[1]].to(x.dtype)
        sin = self.sin[:, :x.shape[1]].to(x.dtype)
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base=10000):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        self.c_q = CastedLinear(dim, num_heads * self.head_dim)
        self.c_k = CastedLinear(dim, num_kv_heads * self.head_dim)
        self.c_v = CastedLinear(dim, num_kv_heads * self.head_dim)
        self.c_proj = CastedLinear(num_heads * self.head_dim, dim)
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q.transpose(1, 2)).transpose(1, 2)
        k = self.rotary(k.transpose(1, 2)).transpose(1, 2)

        # QK-norm
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            enable_gqa=(self.num_kv_heads < self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        hidden = int(dim * mult)
        self.c_fc = CastedLinear(dim, hidden)
        self.c_proj = CastedLinear(hidden, dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU²
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, logit_softcap, rope_base):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.logit_softcap = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base)
            for _ in range(num_layers)
        ])
        self.norm_f = RMSNorm(model_dim)

        if tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = CastedLinear(model_dim, vocab_size)

        # U-Net style skip connections
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        # Residual stream mixing
        self.resid_mix = nn.Parameter(torch.ones(num_layers, 2))

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.tok_emb(idx)
        x0 = x

        # Encoder (first half)
        skip_connections = []
        for i in range(self.num_encoder_layers):
            mix = torch.softmax(self.resid_mix[i], dim=0)
            x = mix[0] * x + mix[1] * x0
            x = self.blocks[i](x)
            skip_connections.append(x)

        # Decoder (second half, with skip connections)
        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            mix = torch.softmax(self.resid_mix[layer_idx], dim=0)
            x = mix[0] * x + mix[1] * x0
            if i < len(skip_connections):
                skip_idx = len(skip_connections) - 1 - i
                x = x + self.skip_weights[i] * skip_connections[skip_idx]
            x = self.blocks[layer_idx](x)

        x = self.norm_f(x)

        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.tok_emb.weight.to(x.dtype))

        if self.logit_softcap > 0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        return logits


# ---------------------------------------------------------------------------
# Optimizer: Muon (for matrix params) + Adam (for everything else)
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to compute the matrix sign / polar factor."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + 1e-7)
    if G.size(-2) > G.size(-1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D matrix parameters."""
    def __init__(self, params, lr=0.02, momentum=0.95, backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = buf.clone()

                # Newton-Schulz orthogonalization
                g_ns = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])

                # Scale: preserve norm ratio
                scale = (g.norm() / (g_ns.norm() + 1e-8)).clamp(max=10.0)
                p.add_(g_ns, alpha=-lr * scale)


# ---------------------------------------------------------------------------
# Training Script
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Quick 30s test run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")

    # Detect GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_vram_gb:.1f} GB VRAM)")
    else:
        print("WARNING: No CUDA device. Training will be very slow.")

    # Load tokenizer and data
    sp = get_tokenizer()
    val_tokens = load_validation_tokens(seq_len=SEQ_LEN)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, VOCAB_SIZE, device
    )
    print(f"Validation tokens: {val_tokens.numel():,}")

    # Build model
    model = GPT(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        mlp_mult=MLP_MULT,
        tie_embeddings=TIE_EMBEDDINGS,
        logit_softcap=LOGIT_SOFTCAP,
        rope_base=ROPE_BASE,
    ).to(device)

    # Keep linear weights in fp32, everything else in bf16
    model = model.bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Architecture: {NUM_LAYERS}L, dim={MODEL_DIM}, heads={NUM_HEADS}, "
          f"kv_heads={NUM_KV_HEADS}, mlp={MLP_MULT}x")

    # Setup optimizer
    matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim == 2]
    scalar_params = [p for n, p in model.blocks.named_parameters() if p.ndim < 2]
    scalar_params.extend([model.skip_weights, model.resid_mix])

    optimizer_muon = Muon(
        matrix_params, lr=MATRIX_LR, momentum=MUON_MOMENTUM,
    )
    embed_params = [model.tok_emb.weight]
    if model.lm_head is not None:
        embed_params.append(model.lm_head.weight)

    optimizer_adam = torch.optim.Adam([
        {"params": embed_params, "lr": EMBED_LR},
        {"params": scalar_params, "lr": SCALAR_LR},
    ], betas=ADAM_BETAS, eps=ADAM_EPS)

    for group in optimizer_muon.param_groups + optimizer_adam.param_groups:
        group["base_lr"] = group["lr"]

    # Data loader
    train_loader = DistributedTokenLoader("fineweb_train_*.bin", 0, 1, device)

    # Compute gradient accumulation
    tokens_per_micro = DEVICE_BATCH_SIZE * SEQ_LEN
    grad_accum_steps = max(1, TRAIN_BATCH_TOKENS // tokens_per_micro)
    grad_scale = 1.0 / grad_accum_steps

    print(f"Batch: {TRAIN_BATCH_TOKENS:,} tokens, micro={DEVICE_BATCH_SIZE}x{SEQ_LEN}, "
          f"accum={grad_accum_steps}")

    # Time budget
    target_seconds = 30 if args.smoke_test else TIME_BUDGET
    print(f"Time budget: {target_seconds}s")

    # LR schedule (wallclock-based warmdown)
    def lr_mul(step, elapsed_ms):
        if WARMDOWN_ITERS <= 0:
            return 1.0
        max_ms = target_seconds * 1000.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = WARMDOWN_ITERS * step_ms
        remaining_ms = max(max_ms - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            return remaining_ms / max(warmdown_ms, 1e-9)
        return 1.0

    # --- TRAINING LOOP ---
    model.train()
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    max_wallclock_ms = target_seconds * 1000.0

    step = 0
    best_train_loss = float("inf")

    while True:
        last_step = (stop_after_step is not None and step >= stop_after_step)

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Muon momentum warmup
        frac = min(step / max(MUON_MOMENTUM_WARMUP_STEPS, 1), 1.0)
        muon_momentum = (1 - frac) * MUON_MOMENTUM_WARMUP_START + frac * MUON_MOMENTUM
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
            group["lr"] = group["base_lr"] * scale

        for group in optimizer_adam.param_groups:
            group["lr"] = group["base_lr"] * scale

        # Zero gradients
        optimizer_muon.zero_grad(set_to_none=True)
        optimizer_adam.zero_grad(set_to_none=True)

        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(TRAIN_BATCH_TOKENS, SEQ_LEN, grad_accum_steps)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer_muon.step()
        optimizer_adam.step()

        step += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        training_time_ms += dt * 1000.0
        t0 = time.perf_counter()

        # Logging
        tl = train_loss.item()
        if step <= 5 or step % 50 == 0:
            pct = 100.0 * training_time_ms / max_wallclock_ms
            remaining = max(0, (max_wallclock_ms - training_time_ms) / 1000.0)
            print(f"step {step:05d} ({pct:.1f}%) | loss: {tl:.4f} | "
                  f"lr_mul: {scale:.3f} | dt: {dt*1000:.0f}ms | "
                  f"remaining: {remaining:.0f}s")

        # GC management
        if step == 1:
            gc.collect()
            gc.disable()

        # Check wallclock
        if training_time_ms >= max_wallclock_ms:
            if stop_after_step is None:
                stop_after_step = step
                print(f"Wallclock reached at step {step}. Stopping.")

    print(f"\nTraining complete: {step} steps, {training_time_ms/1000:.1f}s")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB")

    # --- EVALUATION ---
    model.eval()
    torch.cuda.synchronize()
    t_eval = time.perf_counter()

    val_loss, val_bpb = eval_val(
        model, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        seq_len=SEQ_LEN, eval_stride=EVAL_STRIDE, eval_batch_seqs=EVAL_BATCH_SEQS,
    )

    torch.cuda.synchronize()
    eval_time = time.perf_counter() - t_eval

    # --- ARTIFACT SIZE CHECK ---
    # Get the base model (unwrap compile if needed)
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    artifact_bytes, quant_stats = measure_artifact_size(base_model, code_path=__file__)
    artifact_ok = artifact_bytes <= MAX_ARTIFACT_BYTES

    # --- QUANTIZED EVAL (round-trip validation) ---
    sd = base_model.state_dict()
    quant_obj, _ = quantize_state_dict_int8(sd)
    deq_sd = dequantize_state_dict_int8(quant_obj)
    base_model.load_state_dict(deq_sd, strict=True)

    q_val_loss, q_val_bpb = eval_val(
        model, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        seq_len=SEQ_LEN, eval_stride=EVAL_STRIDE, eval_batch_seqs=EVAL_BATCH_SEQS,
    )

    quant_gap = q_val_bpb - val_bpb
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # --- FINAL OUTPUT (autoresearch format) ---
    print("---")
    print(f"val_bpb:          {q_val_bpb:.6f}")
    print(f"val_bpb_prequant: {val_bpb:.6f}")
    print(f"quant_gap:        {quant_gap:.6f}")
    print(f"artifact_bytes:   {artifact_bytes}")
    print(f"artifact_ok:      {artifact_ok}")
    print(f"training_seconds: {training_time_ms / 1000:.1f}")
    print(f"eval_seconds:     {eval_time:.1f}")
    print(f"total_seconds:    {time.perf_counter() - (t0 - training_time_ms/1000):.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_params / 1e6:.1f}")
    print(f"depth:            {NUM_LAYERS}")

    if not artifact_ok:
        print(f"WARNING: Artifact size {artifact_bytes:,} exceeds 16MB limit!")


if __name__ == "__main__":
    main()
