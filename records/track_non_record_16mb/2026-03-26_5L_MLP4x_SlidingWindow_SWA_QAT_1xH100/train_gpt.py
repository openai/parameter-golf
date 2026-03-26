"""
Parameter Golf Autoresearch Training Script — Single-GPU, Single-File.
Includes: BigramHash, SmearGate, Orthogonal Init, SWA, QAT, torch.compile.

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

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
# Hyperparameters
# ---------------------------------------------------------------------------

# Model architecture
NUM_LAYERS = 5
MODEL_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 4
TIE_EMBEDDINGS = True
ROPE_BASE = 10000
LOGIT_SOFTCAP = 30.0
BIGRAM_BUCKETS = 4096
BIGRAM_DIM = 128

# Training
SEQ_LEN = 1024
TRAIN_BATCH_TOKENS = 131072
DEVICE_BATCH_SIZE = 128
WARMDOWN_FRAC = 0.5        # last 50% of wall time is warmdown (fraction-based, no step dependence)

# Optimizer
MATRIX_LR = 0.03
SCALAR_LR = 0.04
EMBED_LR = 0.06
MUON_MOMENTUM = 0.99
MUON_MOMENTUM_WARMUP_STEPS = 500
MUON_MOMENTUM_WARMUP_START = 0.92
WEIGHT_DECAY = 0.04
ADAM_BETAS = (0.90, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP_NORM = 1.0

# SWA
SWA_ENABLED = True
SWA_START_FRAC = 0.4       # collect when lr_mul drops below this
SWA_EVERY = 50             # collect every N steps
SWA_MIN_STEP = 500         # don't start SWA before this step

# QAT
QAT_ENABLED = True

# Evaluation
EVAL_STRIDE = 64
EVAL_BATCH_SEQS = 64


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
    """fp32 weights for optimizer quality, bf16 for compute. Optional int8 QAT."""
    _qat: bool = False
    _zero_init: bool = False

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        w = self.weight
        if self._qat and self.training and w.ndim == 2:
            w_f = w.float()
            amax = w_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = amax / 127.0
            q = (w_f / scale).round().clamp(-127, 127)
            w_q = q * scale
            w = w + (w_q - w_f).detach()  # STE
        return F.linear(x, w.to(x.dtype))


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


class BigramHash(nn.Module):
    """Hash-table embedding for (prev_token, cur_token) bigrams."""
    def __init__(self, num_buckets, hash_dim, model_dim):
        super().__init__()
        self.num_buckets = num_buckets
        self.table = nn.Embedding(num_buckets, hash_dim)
        self.proj = CastedLinear(hash_dim, model_dim, bias=False)
        self.proj._zero_init = True
        nn.init.normal_(self.table.weight, std=0.01)

    def forward(self, input_ids):
        bsz, seqlen = input_ids.shape
        prev_ids = torch.cat([
            torch.zeros(bsz, 1, dtype=input_ids.dtype, device=input_ids.device),
            input_ids[:, :-1]
        ], dim=1)
        h = ((prev_ids.long() * 92821 + input_ids.long()) % self.num_buckets).long()
        return self.proj(self.table(h))


class SmearGate(nn.Module):
    """Per-dim gate blending each token with the previous token."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), 3.0, dtype=torch.float32))

    def forward(self, x):
        g = torch.sigmoid(self.gate).to(dtype=x.dtype)
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return g * x + (1.0 - g) * x_prev


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
        self.c_proj._zero_init = True
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.rotary(q.transpose(1, 2)).transpose(1, 2)
        k = self.rotary(k.transpose(1, 2)).transpose(1, 2)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
            enable_gqa=(self.num_kv_heads < self.num_heads))
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        hidden = int(dim * mult)
        self.c_fc = CastedLinear(dim, hidden)
        self.c_proj = CastedLinear(hidden, dim)
        self.c_proj._zero_init = True

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
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
                 mlp_mult, tie_embeddings, logit_softcap, rope_base,
                 bigram_buckets=0, bigram_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.logit_softcap = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = SmearGate(model_dim)
        self.bigram_hash = BigramHash(bigram_buckets, bigram_dim, model_dim) if bigram_buckets > 0 else None

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base)
            for _ in range(num_layers)
        ])
        self.norm_f = RMSNorm(model_dim)
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.resid_mix = nn.Parameter(torch.ones(num_layers, 2))

        # Orthogonal init for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.orthogonal_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.tok_emb(idx)
        x = self.smear_gate(x)
        if self.bigram_hash is not None:
            x = x + self.bigram_hash(idx)
        x0 = x

        skip_connections = []
        for i in range(self.num_encoder_layers):
            mix = torch.softmax(self.resid_mix[i], dim=0)
            x = mix[0] * x + mix[1] * x0
            x = self.blocks[i](x)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            mix = torch.softmax(self.resid_mix[layer_idx], dim=0)
            x = mix[0] * x + mix[1] * x0
            if i < len(skip_connections):
                x = x + self.skip_weights[i] * skip_connections[len(skip_connections) - 1 - i]
            x = self.blocks[layer_idx](x)

        x = self.norm_f(x)
        logits = self.lm_head(x) if self.lm_head is not None else F.linear(x, self.tok_emb.weight.to(x.dtype))

        if self.logit_softcap > 0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits


# ---------------------------------------------------------------------------
# Optimizer: Muon + Adam
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G, steps=5):
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
                g_ns = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                scale = (g.norm() / (g_ns.norm() + 1e-8)).clamp(max=10.0)
                p.add_(g_ns, alpha=-lr * scale)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3) if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_vram_gb:.1f} GB VRAM)")

    sp = get_tokenizer()
    val_tokens = load_validation_tokens(seq_len=SEQ_LEN)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, VOCAB_SIZE, device)
    print(f"Validation tokens: {val_tokens.numel():,}")

    model = GPT(
        vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, model_dim=MODEL_DIM,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, mlp_mult=MLP_MULT,
        tie_embeddings=TIE_EMBEDDINGS, logit_softcap=LOGIT_SOFTCAP,
        rope_base=ROPE_BASE, bigram_buckets=BIGRAM_BUCKETS, bigram_dim=BIGRAM_DIM,
    ).to(device)

    model = model.bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()

    if QAT_ENABLED:
        for m in model.modules():
            if isinstance(m, CastedLinear):
                m._qat = True

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Architecture: {NUM_LAYERS}L dim={MODEL_DIM} MLP={MLP_MULT}x bigram={BIGRAM_BUCKETS} QAT={QAT_ENABLED} SWA={SWA_ENABLED}")

    model = torch.compile(model)
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Optimizer setup
    matrix_params = [p for n, p in base_model.blocks.named_parameters() if p.ndim == 2]
    scalar_params = [p for n, p in base_model.blocks.named_parameters() if p.ndim < 2]
    scalar_params.extend([base_model.skip_weights, base_model.resid_mix, base_model.smear_gate.gate])
    if base_model.bigram_hash is not None:
        matrix_params.append(base_model.bigram_hash.proj.weight)

    optimizer_muon = Muon(matrix_params, lr=MATRIX_LR, momentum=MUON_MOMENTUM)

    embed_params = [base_model.tok_emb.weight]
    if base_model.lm_head is not None:
        embed_params.append(base_model.lm_head.weight)
    if base_model.bigram_hash is not None:
        embed_params.append(base_model.bigram_hash.table.weight)

    optimizer_adam = torch.optim.Adam([
        {"params": embed_params, "lr": EMBED_LR},
        {"params": scalar_params, "lr": SCALAR_LR},
    ], betas=ADAM_BETAS, eps=ADAM_EPS)

    for group in optimizer_muon.param_groups + optimizer_adam.param_groups:
        group["base_lr"] = group["lr"]

    train_loader = DistributedTokenLoader("fineweb_train_*.bin", 0, 1, device)
    tokens_per_micro = DEVICE_BATCH_SIZE * SEQ_LEN
    grad_accum_steps = max(1, TRAIN_BATCH_TOKENS // tokens_per_micro)
    grad_scale = 1.0 / grad_accum_steps

    print(f"Batch: {TRAIN_BATCH_TOKENS:,} tokens, micro={DEVICE_BATCH_SIZE}x{SEQ_LEN}, accum={grad_accum_steps}")

    target_seconds = 30 if args.smoke_test else TIME_BUDGET
    max_wallclock_ms = target_seconds * 1000.0
    print(f"Time budget: {target_seconds}s")

    # Fraction-based LR schedule — no dependence on step timing
    def lr_mul(elapsed_ms):
        progress = elapsed_ms / max_wallclock_ms  # 0 to 1
        warmdown_start = 1.0 - WARMDOWN_FRAC
        if progress >= warmdown_start:
            return max(0.0, 1.0 - (progress - warmdown_start) / WARMDOWN_FRAC)
        return 1.0

    # SWA state
    swa_state = None
    swa_count = 0

    # --- TRAINING LOOP ---
    model.train()
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        if stop_after_step is not None and step >= stop_after_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(elapsed_ms)

        frac = min(step / max(MUON_MOMENTUM_WARMUP_STEPS, 1), 1.0)
        muon_momentum = (1 - frac) * MUON_MOMENTUM_WARMUP_START + frac * MUON_MOMENTUM
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
            group["lr"] = group["base_lr"] * scale
        for group in optimizer_adam.param_groups:
            group["lr"] = group["base_lr"] * scale

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

        if WEIGHT_DECAY > 0:
            for group in optimizer_muon.param_groups:
                for p in group["params"]:
                    p.data.mul_(1 - group["lr"] * WEIGHT_DECAY)

        # SWA: collect during warmdown, after model has trained enough
        if SWA_ENABLED and scale < SWA_START_FRAC and step >= SWA_MIN_STEP and step % SWA_EVERY == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                print(f"  swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1

        step += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        training_time_ms += dt * 1000.0
        t0 = time.perf_counter()

        tl = train_loss.item()
        if step <= 5 or step % 50 == 0:
            pct = 100.0 * training_time_ms / max_wallclock_ms
            remaining = max(0, (max_wallclock_ms - training_time_ms) / 1000.0)
            print(f"step {step:05d} ({pct:.1f}%) | loss: {tl:.4f} | lr_mul: {scale:.3f} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s")

        if step == 1:
            gc.collect()
            gc.disable()

        if training_time_ms >= max_wallclock_ms:
            if stop_after_step is None:
                stop_after_step = step
                print(f"Wallclock reached at step {step}. Stopping.")

    print(f"\nTraining complete: {step} steps, {training_time_ms/1000:.1f}s")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB")

    # --- Apply SWA ---
    if SWA_ENABLED and swa_state is not None and swa_count > 1:
        print(f"SWA: applying average of {swa_count} checkpoints")
        current_state = base_model.state_dict()
        avg_state = {
            name: (tensor / swa_count).to(dtype=current_state[name].dtype)
            for name, tensor in swa_state.items()
        }
        base_model.load_state_dict(avg_state, strict=True)

    # --- EVALUATION ---
    model.eval()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m._qat = False

    torch.cuda.synchronize()
    t_eval = time.perf_counter()
    val_loss, val_bpb = eval_val(model, device, val_tokens, base_bytes_lut,
        has_leading_space_lut, is_boundary_token_lut, seq_len=SEQ_LEN,
        eval_stride=EVAL_STRIDE, eval_batch_seqs=EVAL_BATCH_SEQS)
    torch.cuda.synchronize()
    eval_time = time.perf_counter() - t_eval

    # --- ARTIFACT SIZE CHECK ---
    artifact_bytes, quant_stats = measure_artifact_size(base_model, code_path=__file__)
    artifact_ok = artifact_bytes <= MAX_ARTIFACT_BYTES

    # --- QUANTIZED EVAL (round-trip) ---
    sd = base_model.state_dict()
    quant_obj, _ = quantize_state_dict_int8(sd)
    deq_sd = dequantize_state_dict_int8(quant_obj)
    base_model.load_state_dict(deq_sd, strict=True)

    q_val_loss, q_val_bpb = eval_val(model, device, val_tokens, base_bytes_lut,
        has_leading_space_lut, is_boundary_token_lut, seq_len=SEQ_LEN,
        eval_stride=EVAL_STRIDE, eval_batch_seqs=EVAL_BATCH_SEQS)

    quant_gap = q_val_bpb - val_bpb
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

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
    print(f"swa_checkpoints:  {swa_count}")

    if not artifact_ok:
        print(f"WARNING: Artifact size {artifact_bytes:,} exceeds 16MB limit!")


if __name__ == "__main__":
    main()
