import os
import math
import time
import glob
import pickle
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import sentencepiece as spm

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.environ.get("BASE_DIR", "/kaggle/working/parameter-golf")
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(BASE_DIR, "data/datasets/fineweb10B_sp1024"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", os.path.join(BASE_DIR, "data/tokenizers/fineweb_1024_bpe.model"))

TRAIN_GLOB = os.path.join(DATA_PATH, "fineweb_train_*.bin")
VAL_GLOB = os.path.join(DATA_PATH, "fineweb_val_*.bin")

VOCAB_SIZE = 1024
SEQ_LEN = 1024
BATCH_SIZE = 4
MAX_WALLCLOCK_SECONDS = 600
LOG_EVERY = 100
VAL_EVERY = 200
VAL_SEQS_PER_RANK = 64
GRAD_CLIP = 1.0

EMA_ENABLED = True
EMA_DECAY = 0.997

OUT_PATH = "/kaggle/working/hash1024_probe_model.bin"

FUSE_DIM = 448
A_LAYERS = 8
A_DIM = 384
A_HEADS = 6
A_MLP_MULT = 3

B_LAYERS = 5
B_DIM = 320
B_HEADS = 5
B_MLP_MULT = 3

FRO_LR = 8e-4
ADAM_LR = 1.5e-3
WARMUP_STEPS = 40
DECAY_START = 250

EXPORT_PRUNE_THRESHOLD = 0.0025
EXPORT_INT6_KEYS = ["to_a.weight", "to_b.weight", "from_a.weight", "from_b.weight"]
EXPORT_INT8_KEYS = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight", "fc1.weight", "fc2.weight", "lm_head.weight"]

RADIAL_BITS = 10
RADIAL_ALPHA = 0.02
RADIAL_TOKEN_GAIN_INIT = 0.01

HASH_BUCKETS = 1024
HASH_GAIN_INIT = 0.02

# ============================================================
# DISTRIBUTED
# ============================================================
def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, local_rank, world_size, device

# ============================================================
# DATA
# ============================================================
def load_data_shard(file: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def load_training_tokens(pattern: str, seq_len: int) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise RuntimeError(f"No training files found for {pattern}")
    print(f"Loading train shard: {files[0]}")
    tokens = load_data_shard(files[0]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1].long()

def load_validation_tokens(pattern: str, seq_len: int) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise RuntimeError(f"No validation files found for {pattern}")
    print(f"Loading {len(files)} val shards...")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1].long()

# ============================================================
# BPB LUTS
# ============================================================
def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
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
        if piece.startswith(" "):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))

    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

# ============================================================
# MODEL
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(x.dtype)

def weight_quant(w):
    scale = w.abs().mean().clamp(min=1e-5)
    return (torch.sign(w) * scale).detach() + (w - w.detach())

class BitLinear(nn.Linear):
    def forward(self, x):
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        return F.linear(x, weight_quant(self.weight), self.bias)

class BitAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = BitLinear(d_model, d_model, bias=False)
        self.k_proj = BitLinear(d_model, d_model, bias=False)
        self.v_proj = BitLinear(d_model, d_model, bias=False)
        self.out_proj = BitLinear(d_model, d_model, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(y)

class BranchBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_mult):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = BitAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.fc1 = BitLinear(d_model, d_model * mlp_mult, bias=False)
        self.fc2 = BitLinear(d_model * mlp_mult, d_model, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x

class RadialTokenFeatures(nn.Module):
    def __init__(self, n_bits=10, alpha=0.02):
        super().__init__()
        phi = (1 + 5 ** 0.5) / 2
        angles = torch.linspace(0, 2 * math.pi, n_bits + 1)[:n_bits]
        radii = torch.pow(phi, torch.arange(n_bits).float()) * alpha
        self.register_buffer("angles", angles)
        self.register_buffer("radii", radii)
        self.register_buffer("bit_indices", torch.arange(n_bits))

    def forward(self, token_ids: torch.Tensor):
        bits = (token_ids.unsqueeze(-1).long() >> self.bit_indices) & 1
        bits = bits.to(self.radii.dtype)
        re = torch.sum(bits * self.radii * torch.cos(self.angles), dim=-1)
        im = torch.sum(bits * self.radii * torch.sin(self.angles), dim=-1)
        mag = torch.sqrt(re ** 2 + im ** 2 + 1e-8)
        phase = torch.atan2(im, re + 1e-8)
        return torch.stack([re, im, mag, phase], dim=-1)

def make_bigram_hash(input_ids: torch.Tensor, buckets: int) -> torch.Tensor:
    prev = torch.roll(input_ids, shifts=1, dims=1)
    prev[:, 0] = 0
    h = (prev * 131 + input_ids) % buckets
    return h.long()

class DualArchitectureRadialHashBridgeFP(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, FUSE_DIM)

        self.radial_token = RadialTokenFeatures(RADIAL_BITS, RADIAL_ALPHA)
        self.radial_token_proj = nn.Linear(4, FUSE_DIM, bias=False)
        self.radial_token_gain = nn.Parameter(torch.tensor(RADIAL_TOKEN_GAIN_INIT))

        self.hash_emb = nn.Embedding(HASH_BUCKETS, FUSE_DIM)
        self.hash_gain = nn.Parameter(torch.tensor(HASH_GAIN_INIT))

        self.to_a = nn.Linear(FUSE_DIM, A_DIM, bias=False)
        self.to_b = nn.Linear(FUSE_DIM, B_DIM, bias=False)
        self.branch_a = nn.ModuleList([BranchBlock(A_DIM, A_HEADS, A_MLP_MULT) for _ in range(A_LAYERS)])
        self.branch_b = nn.ModuleList([BranchBlock(B_DIM, B_HEADS, B_MLP_MULT) for _ in range(B_LAYERS)])
        self.from_a = nn.Linear(A_DIM, FUSE_DIM, bias=False)
        self.from_b = nn.Linear(B_DIM, FUSE_DIM, bias=False)
        self.fuse_norm = RMSNorm(FUSE_DIM)
        self.lm_head = nn.Linear(FUSE_DIM, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, target_ids=None):
        x = self.tok_emb(input_ids)

        rt = self.radial_token(input_ids)
        x = x + self.radial_token_gain * self.radial_token_proj(rt)

        bh = make_bigram_hash(input_ids, HASH_BUCKETS)
        x = x + self.hash_gain * self.hash_emb(bh)

        xa = self.to_a(x)
        xb = self.to_b(x)

        for blk in self.branch_a:
            xa = blk(xa)
        for blk in self.branch_b:
            xb = blk(xb)

        x = self.from_a(xa) + self.from_b(xb)
        x = self.fuse_norm(x)
        logits = self.lm_head(x.reshape(-1, x.size(-1)))

        if target_ids is not None:
            return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")
        return logits

# ============================================================
# OPTIM
# ============================================================
def soft_matrix_shape(update: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if update.dim() != 2:
        return update
    u = update.float()
    row_rms = torch.sqrt(u.pow(2).mean(dim=1, keepdim=True) + eps)
    u = 0.5 * u + 0.5 * (u / row_rms)
    col_rms = torch.sqrt(u.pow(2).mean(dim=0, keepdim=True) + eps)
    u = 0.5 * u + 0.5 * (u / col_rms)
    return u.to(update.dtype)

class FROStable(torch.optim.Optimizer):
    def __init__(self, params, lr=8e-4, beta1=0.9, beta2=0.999, eps=1e-8,
                 scales=(0.1, 0.01, 0.001), alpha=0.10, gamma=0.60,
                 rt_min=0.05, rt_max=1.0, warmup_steps=10):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        scales=scales, alpha=alpha, gamma=gamma,
                        rt_min=rt_min, rt_max=rt_max, warmup_steps=warmup_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        rt_values = []
        for group in self.param_groups:
            beta1, beta2 = group["beta1"], group["beta2"]
            eps = group["eps"]
            alpha = group["alpha"]
            gamma = group["gamma"]
            rt_min = group["rt_min"]
            rt_max = group["rt_max"]
            warmup_steps = group["warmup_steps"]
            scales = group["scales"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    if p.dim() == 2:
                        state["exp_avg_sq"] = torch.zeros(p.size(0), 1, device=p.device, dtype=p.dtype)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["mu"] = [torch.zeros(1, device=p.device, dtype=torch.float32) for _ in scales]
                    state["s2"] = [torch.zeros(1, device=p.device, dtype=torch.float32) for _ in scales]

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                mu = state["mu"]
                s2 = state["s2"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                bias_correction1 = 1 - beta1 ** state["step"]

                g_flat = grad.float().reshape(-1)
                m_flat = exp_avg.float().reshape(-1)
                gnorm = g_flat.norm()
                mnorm = m_flat.norm()
                if gnorm.item() == 0.0 or mnorm.item() == 0.0:
                    rho_t = torch.tensor(0.0, device=p.device, dtype=torch.float32)
                else:
                    rho_t = torch.dot(g_flat, m_flat) / (gnorm * mnorm + eps)
                    rho_t = rho_t.clamp(-1.0, 1.0)

                for k, lam in enumerate(scales):
                    mu[k].mul_(1 - lam).add_(rho_t, alpha=lam)
                    s2[k].mul_(1 - lam).add_(rho_t * rho_t, alpha=lam)

                log_sum = 0.0
                K = len(scales)
                for k in range(K):
                    rk = (mu[k] * mu[k]) / (s2[k] + eps)
                    rk = rk.clamp(rt_min, rt_max)
                    log_sum = log_sum + torch.log(rk + eps)

                Rt = torch.exp(log_sum / K).clamp(rt_min, rt_max)
                rt_values.append(float(Rt))

                warm = min(1.0, state["step"] / float(warmup_steps))
                Rt_eff = (1.0 - warm) * torch.tensor(1.0, device=p.device) + warm * Rt

                if p.dim() == 2:
                    grad_sq = grad.pow(2).mean(dim=1, keepdim=True)
                    exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)
                else:
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                base_update = exp_avg / denom
                shaped_update = soft_matrix_shape(base_update, eps=eps) if p.dim() == 2 else base_update
                adaptive_factor = alpha + (1.0 - alpha) * gamma * Rt_eff
                step_size = group["lr"] * adaptive_factor / bias_correction1
                p.add_(shaped_update, alpha=-float(step_size))
        return rt_values

def build_param_groups(model):
    fro_params, adam_params = [], []
    adam_prefixes = [
        "tok_emb", "to_a", "to_b", "from_a", "from_b", "fuse_norm", "lm_head",
        "radial_token_proj", "radial_token_gain", "hash_emb", "hash_gain"
    ]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in adam_prefixes):
            adam_params.append(p)
        else:
            fro_params.append(p)
    return fro_params, adam_params

def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr

def lr_mult(step, total_steps_guess=3000):
    if step < WARMUP_STEPS:
        return max(0.1, (step + 1) / float(WARMUP_STEPS))
    if step < DECAY_START:
        return 1.0
    progress = min(1.0, (step - DECAY_START) / max(1.0, total_steps_guess - DECAY_START))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

# ============================================================
# EMA
# ============================================================
@torch.no_grad()
def ema_update(ema_model, live_model, decay):
    live = live_model.module if hasattr(live_model, "module") else live_model
    for ema_p, live_p in zip(ema_model.parameters(), live.parameters()):
        ema_p.data.mul_(decay).add_(live_p.data, alpha=1.0 - decay)
    for ema_b, live_b in zip(ema_model.buffers(), live.buffers()):
        ema_b.copy_(live_b)

# ============================================================
# EVAL
# ============================================================
@torch.no_grad()
def eval_val_subset(model, device, val_tokens, base_bytes_lut, has_space_lut, boundary_lut, rank, world_size):
    was_training = model.training
    model.eval()

    seq_len = SEQ_LEN
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seqs_per_rank = min(VAL_SEQS_PER_RANK, max(1, total_seqs // world_size))
    start_seq = rank * seqs_per_rank
    end_seq = min(start_seq + seqs_per_rank, total_seqs)

    local_loss_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    local_token_count = torch.tensor(0.0, device=device, dtype=torch.float64)
    local_byte_count = torch.tensor(0.0, device=device, dtype=torch.float64)

    for i in range(start_seq, end_seq):
        raw_start = i * seq_len
        raw_end = (i + 1) * seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device)

        x = local[:-1].unsqueeze(0)
        y = local[1:].unsqueeze(0)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            batch_loss = model(x, y)

        batch_loss = batch_loss.to(torch.float64)
        batch_token_count = float(y.numel())

        local_loss_sum += batch_loss * batch_token_count
        local_token_count += batch_token_count

        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        t_bytes = base_bytes_lut[tgt_ids].clone()
        t_bytes += (has_space_lut[tgt_ids] & ~boundary_lut[prev_ids]).to(dtype=torch.int16)
        local_byte_count += t_bytes.to(torch.float64).sum()

    metrics = torch.stack([local_loss_sum, local_token_count, local_byte_count])
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    global_loss_sum, global_token_count, global_byte_count = metrics[0], metrics[1], metrics[2]
    val_loss = global_loss_sum / (global_token_count + 1e-10)
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = global_token_count / (global_byte_count + 1e-10)
    val_bpb = bits_per_token * tokens_per_byte

    if was_training:
        model.train()

    return float(val_loss.item()), float(val_bpb.item())

# ============================================================
# EXPORT
# ============================================================
def prune_small_values(t: torch.Tensor, thr: float):
    out = t.clone()
    out[out.abs() < thr] = 0
    return out

def pack_int6_tensor(x: torch.Tensor):
    return x.to(torch.int8)

def quantize_tensor_int8(v: torch.Tensor):
    scale = v.abs().mean().clamp(min=1e-5)
    q = torch.clamp(torch.round(v / scale * 127.0), -127, 127).to(torch.int8)
    return (q.cpu(), float(scale), "int8")

def quantize_tensor_int6(v: torch.Tensor):
    scale = v.abs().mean().clamp(min=1e-5)
    q = torch.clamp(torch.round(v / scale * 31.0), -31, 31)
    q = pack_int6_tensor(q)
    return (q.cpu(), float(scale), "int6")

def quantize_state_for_export(state_dict):
    q_state = {}
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            q_state[k] = v
            continue
        if not v.is_floating_point():
            q_state[k] = v.detach().cpu()
            continue

        vv = v.detach().cpu()
        if vv.dim() >= 2:
            vv = prune_small_values(vv, EXPORT_PRUNE_THRESHOLD)

        if any(name in k for name in EXPORT_INT6_KEYS):
            q_state[k] = quantize_tensor_int6(vv)
        elif any(name in k for name in EXPORT_INT8_KEYS):
            q_state[k] = quantize_tensor_int8(vv)
        else:
            q_state[k] = vv.to(torch.float16)

    return q_state

@torch.no_grad()
def artifact_audit(model, out_path):
    model = model.cpu().eval()
    q_state = quantize_state_for_export(model.state_dict())
    raw_bytes = pickle.dumps(q_state, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(raw_bytes, level=9)
    with open(out_path, "wb") as f:
        f.write(compressed)

    model_bytes = os.path.getsize(out_path)
    code_bytes = Path(__file__).read_bytes()
    total_bytes = model_bytes + len(code_bytes)
    params = sum(p.numel() for p in model.parameters())

    print("\n=== MIXED EXPORT ARTIFACT AUDIT ===")
    print(f"Parameters:        {params:,}")
    print(f"Source Code:       {len(code_bytes):,} bytes")
    print(f"Compressed model:  {model_bytes:,} bytes")
    print(f"Total artifact:    {total_bytes:,} bytes")
    print(f"Headroom:          {16_000_000 - total_bytes:,} bytes")
    print("PASS ✅" if total_bytes <= 16_000_000 else "FAIL ❌")

# ============================================================
# MAIN
# ============================================================
def main():
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.cuda.manual_seed_all(1337)

    rank, local_rank, world_size, device = setup_distributed()

    train_tokens = load_training_tokens(TRAIN_GLOB, SEQ_LEN)
    val_tokens = load_validation_tokens(VAL_GLOB, SEQ_LEN)

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    base_bytes, has_space, boundary = build_sentencepiece_luts(sp, VOCAB_SIZE, device)

    if rank == 0:
        print(f"Loaded train tokens: {train_tokens.numel():,}")
        print(f"Loaded val tokens:   {val_tokens.numel():,}")

    live_model = DualArchitectureRadialHashBridgeFP().to(device)
    fro_params, adam_params = build_param_groups(live_model)

    if rank == 0:
        print(f"FRO params:  {sum(p.numel() for p in fro_params):,}")
        print(f"Adam params: {sum(p.numel() for p in adam_params):,}")

    ema_model = DualArchitectureRadialHashBridgeFP().to(device)
    ema_model.load_state_dict(live_model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    live_model = nn.parallel.DistributedDataParallel(live_model, device_ids=[local_rank])

    fro_opt = FROStable(fro_params, lr=FRO_LR)
    adam_opt = torch.optim.AdamW(adam_params, lr=ADAM_LR, betas=(0.9, 0.95), weight_decay=0.01)

    torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(1337 + rank)

    start = time.time()
    step = 0
    last_loss = None
    best_val = float("inf")

    while time.time() - start < MAX_WALLCLOCK_SECONDS:
        chunk_size = BATCH_SIZE * SEQ_LEN
        total_available = max(1, (train_tokens.numel() - chunk_size * world_size - 1))
        base_start = int(rng.integers(0, total_available))
        start_token = (base_start + rank * chunk_size) % total_available
        chunk = train_tokens[start_token : start_token + chunk_size + 1]

        x = chunk[:-1].reshape(BATCH_SIZE, SEQ_LEN).to(device, non_blocking=True)
        y = chunk[1:].reshape(BATCH_SIZE, SEQ_LEN).to(device, non_blocking=True)

        fro_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)

        mult = lr_mult(step)
        set_lr(fro_opt, FRO_LR * mult)
        set_lr(adam_opt, ADAM_LR * mult)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = live_model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(live_model.parameters(), GRAD_CLIP)

        rt_values = fro_opt.step()
        adam_opt.step()
        last_loss = loss.detach()

        if EMA_ENABLED:
            ema_update(ema_model, live_model, EMA_DECAY)

        if step % LOG_EVERY == 0:
            loss_tensor = torch.tensor([float(last_loss)], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            mean_loss = loss_tensor.item() / world_size

            toks = (step + 1) * BATCH_SIZE * SEQ_LEN * world_size
            elapsed = time.time() - start
            toks_per_sec = toks / max(elapsed, 1e-6)
            mean_rt = sum(rt_values) / max(len(rt_values), 1) if len(rt_values) else 0.0
            max_mem = torch.cuda.max_memory_allocated(device) / 1e9

            if rank == 0:
                model_ref = live_model.module if hasattr(live_model, "module") else live_model
                rg = float(model_ref.radial_token_gain.detach().cpu())
                hg = float(model_ref.hash_gain.detach().cpu())
                print(
                    f"step {step:04d} | time {elapsed:.1f}s | "
                    f"train_loss {mean_loss:.4f} | tok/s {toks_per_sec:.0f} | "
                    f"mean_Rt {mean_rt:.4f} | lr_mult {mult:.3f} | "
                    f"radial_gain {rg:.5f} | hash_gain {hg:.5f} | max_mem {max_mem:.2f} GB"
                )

        if step > 0 and step % VAL_EVERY == 0:
            val_loss, val_bpb = eval_val_subset(
                ema_model, device, val_tokens, base_bytes, has_space, boundary, rank, world_size
            )
            if rank == 0:
                best_val = min(best_val, val_bpb)
                print(f"VALIDATION-EMA | step {step:04d} | val_loss {val_loss:.4f} | val_bpb {val_bpb:.4f} | best {best_val:.4f}")

        step += 1

    if rank == 0:
        elapsed = time.time() - start
        print(f"\nFinished at step {step} in {elapsed:.1f}s")
        print(f"Final train loss: {float(last_loss):.4f}")
        print(f"Best observed val_bpb: {best_val:.4f}")
        artifact_audit(ema_model, OUT_PATH)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
