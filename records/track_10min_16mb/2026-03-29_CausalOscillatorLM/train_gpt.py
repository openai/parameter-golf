"""
Causal Oscillator LM — physics-native language model for Parameter Golf.

Architecture: Token → impulse → causal convolution with damped oscillator
impulse response → oscillator state → causal attention → predict next token.

The only learnable transform: H(ω) = 1/(ω₀² - ω² + 2iγω)

Each token drives a bank of damped oscillators. The impulse response creates
temporal context through physics — recent tokens ring loudly, distant tokens
decay. Attention handles long-range dependencies on top.

Source: https://github.com/rolandnsharp/resonance
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 2000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 5000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    n_oscillators = int(os.environ.get("N_OSCILLATORS", 192))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_heads = int(os.environ.get("NUM_HEADS", 16))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    lr = float(os.environ.get("LR", 0.003))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.01))
    grad_clip = float(os.environ.get("GRAD_CLIP", 1.0))

# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(filepath):
    header = np.fromfile(filepath, dtype=np.int32, count=256)
    assert header[0] == 20240520
    assert header[1] in (1, 7)
    ntok = header[2]
    dtype = np.uint16 if header[1] == 1 else np.uint32
    tokens = np.fromfile(filepath, dtype=dtype, offset=256 * 4, count=ntok)
    return torch.from_numpy(tokens.astype(np.int32))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

class ShardedDataLoader:
    def __init__(self, pattern, seq_len, rank, world_size):
        self.files = sorted(glob.glob(pattern))
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.current_shard = 0
        self.current_pos = 0
        self._load_shard()

    def _load_shard(self):
        self.tokens = load_data_shard(Path(self.files[self.current_shard]))

    def next_batch(self, batch_size):
        B, T = batch_size, self.seq_len
        if self.current_pos + B * T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.files)
            self.current_pos = 0
            self._load_shard()
        buf = self.tokens[self.current_pos:self.current_pos + B * T + 1]
        x = buf[:-1].reshape(B, T)
        y = buf[1:].reshape(B, T)
        self.current_pos += B * T * self.world_size
        return x, y

# -----------------------------
# BPB EVALUATION
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def eval_val(args, model, rank, world_size, device, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    local_batch = max(1, args.val_batch_size // (world_size * seq_len))
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, local_batch):
            batch_end = min(batch_start + local_batch, seq_end)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y).detach()
            n = float(y.numel())
            loss_sum += loss.to(torch.float64) * n
            token_count += n
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    model.train()
    return val_loss, bits_per_token * tokens_per_byte

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95, backend_steps=5, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
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
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)

# -----------------------------
# QUANTIZATION
# -----------------------------

PHYSICS_PARAMS = ('log_omega', 'damping_pre')

def quantize_state_dict_int8(state_dict):
    quantized = {}
    for name, tensor in state_dict.items():
        if any(p in name for p in PHYSICS_PARAMS):
            quantized[name] = tensor.half()
        elif tensor.is_floating_point() and tensor.numel() > 1:
            scale = tensor.abs().max() / 127.0
            quantized[name] = (tensor / scale).round().to(torch.int8)
            quantized[name + ".__scale__"] = scale.float()
        else:
            quantized[name] = tensor
    return quantized

def dequantize_state_dict(quantized):
    state_dict = {}
    scale_suffix = ".__scale__"
    scales = {k: v for k, v in quantized.items() if k.endswith(scale_suffix)}
    for name, tensor in quantized.items():
        if name.endswith(scale_suffix):
            continue
        scale_key = name + scale_suffix
        if scale_key in scales:
            state_dict[name] = tensor.float() * scales[scale_key]
        else:
            state_dict[name] = tensor.float()
    return state_dict

# -----------------------------
# MODEL
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale

class SineGate(nn.Module):
    def forward(self, x):
        return x * torch.sin(x)

class CausalAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, T, _ = x.shape
        H, d = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q = q.view(B, T, H, d).transpose(1, 2)
        k = k.view(B, T, H, d).transpose(1, 2)
        v = v.view(B, T, H, d).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).contiguous().view(B, T, -1))

class OscillatorLayer(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), SineGate(), nn.Linear(dim * 2, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class CausalOscillatorLM(nn.Module):
    """
    Token → impulse → causal convolution with damped impulse response →
    oscillator state → attention → predict next token.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        n_osc = args.n_oscillators
        state_dim = n_osc * 2
        self.token_drive = nn.Embedding(args.vocab_size, n_osc)
        nn.init.normal_(self.token_drive.weight, std=0.1)
        self.log_omega = nn.Parameter(torch.linspace(math.log(0.1), math.log(math.pi), n_osc))
        self.damping_pre = nn.Parameter(torch.zeros(n_osc))
        self.layers = nn.ModuleList([OscillatorLayer(state_dim, args.num_heads) for _ in range(args.num_layers)])
        self.out_norm = RMSNorm(state_dim)
        self.out_proj = nn.Linear(state_dim, args.vocab_size)

    @property
    def omega(self):
        return self.log_omega.exp()
    @property
    def damping(self):
        return 0.05 + 0.9 * torch.sigmoid(self.damping_pre)

    def impulse_response(self, T, device):
        omega = self.omega
        gamma = self.damping * omega
        omega_d = torch.sqrt((omega ** 2 - gamma ** 2).clamp(min=1e-6))
        n = torch.arange(T, device=device, dtype=torch.float32)
        decay = torch.exp(-gamma.unsqueeze(1) * n.unsqueeze(0))
        sin_wd = torch.sin(omega_d.unsqueeze(1) * n.unsqueeze(0))
        cos_wd = torch.cos(omega_d.unsqueeze(1) * n.unsqueeze(0))
        h_pos = decay * sin_wd / (omega_d.unsqueeze(1) + 1e-8)
        h_vel = decay * (cos_wd - (gamma / (omega_d + 1e-8)).unsqueeze(1) * sin_wd)
        return h_pos, h_vel

    def encode(self, token_ids):
        B, T = token_ids.shape
        F_drive = self.token_drive(token_ids).transpose(1, 2)
        h_pos, h_vel = self.impulse_response(T, token_ids.device)
        fft_len = 2 * T
        F_fft = torch.fft.rfft(F_drive, n=fft_len, dim=-1)
        hp_fft = torch.fft.rfft(h_pos, n=fft_len, dim=-1)
        hv_fft = torch.fft.rfft(h_vel, n=fft_len, dim=-1)
        pos = torch.fft.irfft(F_fft * hp_fft.unsqueeze(0), n=fft_len, dim=-1)[..., :T].transpose(1, 2)
        vel = torch.fft.irfft(F_fft * hv_fft.unsqueeze(0), n=fft_len, dim=-1)[..., :T].transpose(1, 2)
        return torch.cat([pos, vel], dim=-1)

    def forward(self, input_ids, target_ids):
        x = self.encode(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.out_proj(self.out_norm(x))
        if self.args.logit_softcap > 0:
            logits = self.args.logit_softcap * torch.tanh(logits / self.args.logit_softcap)
        return F.cross_entropy(logits.view(-1, self.args.vocab_size), target_ids.view(-1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# -----------------------------
# MAIN
# -----------------------------

def main():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_master = rank == 0
    args = Hyperparameters()
    torch.manual_seed(args.seed + rank)

    if is_master:
        print(f"Causal Oscillator LM — H(ω) = 1/(ω₀² - ω² + 2iγω)")
        print(f"Device: {device}, World size: {world_size}")

    sp = spm.SentencePieceProcessor(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    train_loader = ShardedDataLoader(args.train_files, args.train_seq_len, rank, world_size)

    model = CausalOscillatorLM(args).to(device)
    if is_master:
        print(f"Parameters: {model.count_params():,}")
        print(f"Oscillators: {args.n_oscillators}, Layers: {args.num_layers}, Heads: {args.num_heads}")

    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
        raw_model = model.module

    # Optimizer: Muon for 2D, AdamW for physics params
    muon_params, adam_params = [], []
    for n, p in raw_model.named_parameters():
        if p.ndim == 2 and p.shape[0] >= 8 and p.shape[1] >= 8:
            muon_params.append(p)
        else:
            adam_params.append(p)
    if is_master:
        print(f"Muon: {sum(p.numel() for p in muon_params):,}, Adam: {sum(p.numel() for p in adam_params):,}")

    optimizer = torch.optim.AdamW(adam_params, lr=args.lr, weight_decay=args.weight_decay)
    muon_opt = Muon(muon_params, lr=args.lr * 4, momentum=0.95) if muon_params else None

    seq_len = args.train_seq_len
    local_batch_tokens = args.train_batch_tokens // world_size
    local_batch_seqs = local_batch_tokens // seq_len
    grad_accum_steps = max(1, local_batch_seqs // 64)
    micro_batch = local_batch_seqs // grad_accum_steps

    if is_master:
        print(f"Batch: {local_batch_seqs} seqs, {grad_accum_steps} accum, micro={micro_batch}")
        print(f"LR: {args.lr}\n")

    best_bpb = float('inf')
    t0 = time.time()

    for step in range(args.iterations):
        elapsed = time.time() - t0
        if elapsed > args.max_wallclock_seconds:
            if is_master: print(f"Wallclock limit at step {step}")
            break

        if step < args.warmup_steps:
            s = step / max(1, args.warmup_steps)
        elif step >= args.iterations - args.warmdown_iters:
            progress = (step - (args.iterations - args.warmdown_iters)) / args.warmdown_iters
            s = 0.5 * (1 + math.cos(math.pi * progress))
        else:
            s = 1.0
        for pg in optimizer.param_groups: pg['lr'] = args.lr * s
        if muon_opt:
            for pg in muon_opt.param_groups: pg['lr'] = args.lr * 4 * s

        model.train()
        optimizer.zero_grad()
        if muon_opt: muon_opt.zero_grad()
        train_loss_sum = 0.0

        for micro in range(grad_accum_steps):
            x, y = train_loader.next_batch(micro_batch)
            x = x.to(device=device, dtype=torch.int64)
            y = y.to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y) / grad_accum_steps
            loss.backward()
            train_loss_sum += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if muon_opt: muon_opt.step()

        if is_master and step % args.train_log_every == 0:
            bpb = train_loss_sum / math.log(2.0) * grad_accum_steps
            print(f"step {step:5d}  loss {train_loss_sum * grad_accum_steps:.4f}  bpb {bpb:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}  t {elapsed:.0f}s")

        if step % args.val_loss_every == 0 or step == args.iterations - 1:
            val_loss, val_bpb = eval_val(args, raw_model, rank, world_size, device, val_tokens,
                                          base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            if is_master:
                marker = ""
                if val_bpb < best_bpb:
                    best_bpb = val_bpb
                    marker = " * (saved)"
                    torch.save(raw_model.state_dict(), "best_model.pt")
                print(f"  VAL  loss {val_loss:.4f}  bpb {val_bpb:.4f}{marker}")

    if is_master:
        print(f"\n=== Final ===")
        if os.path.exists("best_model.pt"):
            raw_model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True))
            print(f"Loaded best (val_bpb={best_bpb:.4f})")

        torch.save(raw_model.state_dict(), "final_model.pt")
        q_sd = quantize_state_dict_int8(raw_model.state_dict())
        buf = io.BytesIO()
        torch.save(q_sd, buf)
        compressed = zlib.compress(buf.getvalue(), level=9)
        with open("final_model.int8.ptz", "wb") as f:
            f.write(compressed)

        code_bytes = Path(__file__).read_bytes()
        total = len(compressed) + len(code_bytes)
        print(f"Code: {len(code_bytes):,} bytes")
        print(f"Model: {len(compressed):,} bytes")
        print(f"Total: {total:,} bytes (limit: 16,000,000)")
        print(f"Under limit: {total < 16_000_000}")

        with open("final_model.int8.ptz", "rb") as f:
            loaded = torch.load(io.BytesIO(zlib.decompress(f.read())), map_location=device, weights_only=True)
        raw_model.load_state_dict(dequantize_state_dict(loaded))
        val_loss, val_bpb = eval_val(args, raw_model, rank, world_size, device, val_tokens,
                                      base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        print(f"Round-trip val_bpb: {val_bpb:.4f}")
        print(f"Best val_bpb: {best_bpb:.4f}")
        print(f"Parameters: {raw_model.count_params():,}")

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
