# ============================================================
# H100 FINAL CHAMPION - 1.3379 BPB
# q4 @ bs48
# strategia:
#   - train fino a (600 - reserve)
#   - validazione finale EMA obbligatoria
# config:
#   batch_size      = 48
#   residual_beta   = 0.36
#   ema_decay       = 0.996
#   decay_start     = 1000
#   fro_gamma       = 0.66
#   warmup_steps    = 400
# ============================================================

import os
import gc
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
import sentencepiece as spm

# ------------------------------------------------------------
# H100 / CUDA
# ------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

assert torch.cuda.is_available(), "CUDA non disponibile"
DEVICE = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR = os.environ.get("BASE_DIR", "/workspace/parameter-golf")
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(BASE_DIR, "data/datasets/fineweb10B_sp1024"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", os.path.join(BASE_DIR, "data/tokenizers/fineweb_1024_bpe.model"))

TRAIN_GLOB = os.path.join(DATA_PATH, "fineweb_train_*.bin")
VAL_GLOB = os.path.join(DATA_PATH, "fineweb_val_*.bin")

OUT_PATH = "/workspace/final_q4_bs48.bin"

VOCAB_SIZE = 1024
SEQ_LEN = 1024
BATCH_SIZE = 48

MAX_WALLCLOCK_SECONDS = 600.0
FINAL_VAL_RESERVE_SECONDS = 18.0
TRAIN_BUDGET_SECONDS = MAX_WALLCLOCK_SECONDS - FINAL_VAL_RESERVE_SECONDS

LOG_EVERY = 50
VAL_EVERY = 800
VAL_SEQS = 12
FINAL_VAL_SEQS = 12

GRAD_CLIP = 1.8
EMA_ENABLED = True
EMA_DECAY = 0.996
LOSS_EMA_DECAY = 0.98
WEIGHT_DECAY = 0.01

# ------------------------------------------------------------
# ARCH
# ------------------------------------------------------------
FUSE_DIM = 448

A_LAYERS = 8
A_DIM = 384
A_HEADS = 6
A_MLP_MULT = 3

B_LAYERS = 5
B_DIM = 320
B_HEADS = 5
B_MLP_MULT = 3

# ------------------------------------------------------------
# FUSION
# ------------------------------------------------------------
FUSION_MODE = "residual_b"
RESIDUAL_BETA = 0.36

# ------------------------------------------------------------
# OPTIM
# ------------------------------------------------------------
FRO_LR = 9.0e-4
ADAMW_LR = 1.40e-3

WARMUP_STEPS = 400
DECAY_START = 1000
TOTAL_STEPS_GUESS = 4200

FRO_ALPHA = 0.12
FRO_GAMMA = 0.66

# ------------------------------------------------------------
# RADIAL / HASH
# ------------------------------------------------------------
RADIAL_BITS = 10
RADIAL_ALPHA = 0.02
RADIAL_GAIN_INIT = -0.01
RADIAL_GAIN_MIN = -0.075
RADIAL_GAIN_MAX = -0.002

HASH_BUCKETS = 1024
HASH_GAIN_INIT = 0.02
HASH_GAIN_MAX = 1.05

# ------------------------------------------------------------
# DATA / BPB
# ------------------------------------------------------------
def load_data_shard(file: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def load_training_tokens(pattern, max_shards=None):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise RuntimeError(f"No train shards for {pattern}")
    if max_shards: files = files[:max_shards]
    chunks = [load_data_shard(f) for f in files]
    tokens = torch.cat(chunks).contiguous()
    usable = ((tokens.numel() - 1) // SEQ_LEN) * SEQ_LEN
    return tokens[: usable + 1].long()

def load_validation_tokens(pattern):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise RuntimeError(f"No val shards for {pattern}")
    chunks = [load_data_shard(f) for f in files]
    tokens = torch.cat(chunks).contiguous()
    usable = ((tokens.numel() - 1) // SEQ_LEN) * SEQ_LEN
    return tokens[: usable + 1].long()

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
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

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
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
        if x.dtype != self.weight.dtype: x = x.to(self.weight.dtype)
        return F.linear(x, weight_quant(self.weight), self.bias)

class BitAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
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
        self.norm1 = RMSNorm(d_model); self.attn = BitAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model); self.fc1 = BitLinear(d_model, d_model * mlp_mult, bias=False)
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
        self.register_buffer("angles", angles); self.register_buffer("radii", radii)
        self.register_buffer("bit_indices", torch.arange(n_bits))
    def forward(self, token_ids):
        bits = (token_ids.unsqueeze(-1).long() >> self.bit_indices) & 1
        bits = bits.to(self.radii.dtype)
        re = torch.sum(bits * self.radii * torch.cos(self.angles), dim=-1)
        im = torch.sum(bits * self.radii * torch.sin(self.angles), dim=-1)
        mag = torch.sqrt(re**2 + im**2 + 1e-8)
        phase = torch.atan2(im, re + 1e-8)
        return torch.stack([re, im, mag, phase], dim=-1)

def make_bigram_hash(input_ids, buckets):
    prev = torch.roll(input_ids, shifts=1, dims=1); prev[:, 0] = 0
    return ((prev * 131 + input_ids) % buckets).long()

class DualArchitectureRadialHashDisciplined(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, FUSE_DIM)
        self.radial_token = RadialTokenFeatures(RADIAL_BITS, RADIAL_ALPHA)
        self.radial_token_proj = nn.Linear(4, FUSE_DIM, bias=False)
        self.radial_gain = nn.Parameter(torch.tensor(RADIAL_GAIN_INIT))
        self.hash_emb = nn.Embedding(HASH_BUCKETS, FUSE_DIM)
        self.hash_gain = nn.Parameter(torch.tensor(HASH_GAIN_INIT))
        self.to_a = nn.Linear(FUSE_DIM, A_DIM, bias=False); self.to_b = nn.Linear(FUSE_DIM, B_DIM, bias=False)
        self.branch_a = nn.ModuleList([BranchBlock(A_DIM, A_HEADS, A_MLP_MULT) for _ in range(A_LAYERS)])
        self.branch_b = nn.ModuleList([BranchBlock(B_DIM, B_HEADS, B_MLP_MULT) for _ in range(B_LAYERS)])
        self.from_a = nn.Linear(A_DIM, FUSE_DIM, bias=False); self.from_b = nn.Linear(B_DIM, FUSE_DIM, bias=False)
        self.fuse_norm = RMSNorm(FUSE_DIM); self.lm_head = nn.Linear(FUSE_DIM, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.tok_emb.weight
    def forward(self, input_ids, target_ids=None):
        x = self.tok_emb(input_ids)
        rt = self.radial_token(input_ids); x = x + self.radial_gain * self.radial_token_proj(rt)
        bh = make_bigram_hash(input_ids, HASH_BUCKETS); x = x + self.hash_gain * self.hash_emb(bh)
        xa, xb = self.to_a(x), self.to_b(x)
        for blk in self.branch_a: xa = blk(xa)
        for blk in self.branch_b: xb = blk(xb)
        fa, fb = self.from_a(xa), self.from_b(xb)
        fused = self.fuse_norm(fa + RESIDUAL_BETA * fb if FUSION_MODE == "residual_b" else fa + fb)
        logits = self.lm_head(fused.reshape(-1, fused.size(-1)))
        if target_ids is not None: return F.cross_entropy(logits.float(), target_ids.reshape(-1))
        return logits

# ------------------------------------------------------------
# OPTIM / EXPORT
# ------------------------------------------------------------
def soft_matrix_shape(u, eps=1e-8):
    if u.dim() != 2: return u
    row_rms = torch.sqrt(u.pow(2).mean(dim=1, keepdim=True) + eps)
    u = 0.5 * u + 0.5 * (u / row_rms)
    col_rms = torch.sqrt(u.pow(2).mean(dim=0, keepdim=True) + eps)
    return 0.5 * u + 0.5 * (u / col_rms)

class FROStable(torch.optim.Optimizer):
    def __init__(self, params, lr=8e-4, beta1=0.9, beta2=0.999, eps=1e-8, scales=(0.1, 0.01, 0.001), alpha=0.12, gamma=0.66, rt_min=0.05, rt_max=1.0, warmup_steps=10):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, scales=scales, alpha=alpha, gamma=gamma, rt_min=rt_min, rt_max=rt_max, warmup_steps=warmup_steps)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad, state = p.grad, self.state[p]
                if len(state) == 0:
                    state["step"], state["exp_avg"] = 0, torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros(p.size(0), 1, device=p.device) if p.dim() == 2 else torch.zeros_like(p)
                    state["mu"] = [torch.zeros(1, device=p.device) for _ in group["scales"]]
                    state["s2"] = [torch.zeros(1, device=p.device) for _ in group["scales"]]
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2, eps = group["beta1"], group["beta2"], group["eps"]
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                g_flat, m_flat = grad.float().view(-1), exp_avg.float().view(-1)
                gn, mn = g_flat.norm(), m_flat.norm()
                rho = torch.dot(g_flat, m_flat) / (gn * mn + eps) if gn > 0 and mn > 0 else torch.zeros(1, device=p.device)
                rho = rho.clamp(-1.0, 1.0)
                log_sum = 0.0
                for k, lam in enumerate(group["scales"]):
                    state["mu"][k].mul_(1-lam).add_(rho, alpha=lam)
                    state["s2"][k].mul_(1-lam).add_(rho**2, alpha=lam)
                    rk = (state["mu"][k]**2 / (state["s2"][k] + eps)).clamp(group["rt_min"], group["rt_max"])
                    log_sum += torch.log(rk + eps)
                Rt = torch.exp(log_sum / len(group["scales"])).clamp(group["rt_min"], group["rt_max"])
                Rt_eff = (1.0 - min(1, state["step"]/group["warmup_steps"])) + min(1, state["step"]/group["warmup_steps"]) * Rt
                if p.dim() == 2: exp_avg_sq.mul_(beta2).add_(grad.pow(2).mean(dim=1, keepdim=True), alpha=1-beta2)
                else: exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1-beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                shaped = soft_matrix_shape(exp_avg / denom, eps) if p.dim() == 2 else exp_avg / denom
                p.add_(shaped, alpha=-float(group["lr"] * (group["alpha"] + (1-group["alpha"])*group["gamma"]*Rt_eff) / (1-beta1**state["step"])))

@torch.no_grad()
def artifact_audit(model_gpu, out_path):
    import pickle, zlib
    def quant(v, scale_mult):
        s = v.abs().mean().clamp(min=1e-5)
        return (torch.clamp(torch.round(v/s*scale_mult), -scale_mult, scale_mult).to(torch.int8), float(s), f"int{int(math.log2(scale_mult+1)+1)}")
    q_state = {}
    for k, v in model_gpu.state_dict().items():
        vv = v.detach().cpu(); 
        if vv.dim() >= 2: vv[vv.abs() < EXPORT_PRUNE_THRESHOLD] = 0
        if any(x in k for x in EXPORT_INT6_KEYS): q_state[k] = quant(vv, 31)
        elif any(x in k for x in EXPORT_INT8_KEYS): q_state[k] = quant(vv, 127)
        else: q_state[k] = vv.to(torch.float16)
    comp = zlib.compress(pickle.dumps(q_state, protocol=4), level=9)
    with open(out_path, "wb") as f: f.write(comp)
    m_bytes = os.path.getsize(out_path); c_bytes = len(Path(__file__).read_bytes())
    print(f"\n=== AUDIT ===\nModel: {m_bytes:,}\nCode:  {c_bytes:,}\nTotal: {m_bytes+c_bytes:,}\nPASS ✅" if m_bytes+c_bytes <= 16_000_000 else "FAIL ❌")

def main():
    torch.manual_seed(1337)
    model = DualArchitectureRadialHashDisciplined().to(DEVICE)
    ema_m = DualArchitectureRadialHashDisciplined().to(DEVICE); ema_m.load_state_dict(model.state_dict())
    fro_p = [p for n, p in model.named_parameters() if not any(x in n for x in ["tok_emb", "to_", "from_", "norm", "head", "proj", "gain", "hash"])]
    adam_p = [p for n, p in model.named_parameters() if p not in fro_p]
    fro_opt = FROStable(fro_p, lr=FRO_LR, alpha=FRO_ALPHA, gamma=FRO_GAMMA, warmup_steps=WARMUP_STEPS)
    adam_opt = torch.optim.AdamW(adam_p, lr=ADAMW_LR, fused=True)
    train_t = load_training_tokens(TRAIN_GLOB, 32); val_t = load_validation_tokens(VAL_GLOB)
    luts = build_sentencepiece_luts(spm.SentencePieceProcessor(model_file=TOKENIZER_PATH), VOCAB_SIZE, DEVICE)
    start_time = time.time(); step = 0; rng = np.random.default_rng(1337)
    while time.time() - start_time < TRAIN_BUDGET_SECONDS:
        mult = (step+1)/WARMUP_STEPS if step < WARMUP_STEPS else (0.5*(1+math.cos(math.pi*min(1,(step-DECAY_START)/(TOTAL_STEPS_GUESS-DECAY_START)))) if step > DECAY_START else 1.0)
        for g in fro_opt.param_groups: g["lr"] = FRO_LR * mult
        for g in adam_opt.param_groups: g["lr"] = ADAMW_LR * mult
        idx = rng.integers(0, train_t.numel() - BATCH_SIZE*SEQ_LEN - 1)
        chunk = train_t[idx : idx + BATCH_SIZE*SEQ_LEN + 1].to(DEVICE)
        x, y = chunk[:-1].view(BATCH_SIZE, SEQ_LEN), chunk[1:].view(BATCH_SIZE, SEQ_LEN)
        fro_opt.zero_grad(); adam_opt.zero_grad()
        with torch.autocast("cuda", dtype=AMP_DTYPE): loss = model(x, y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        fro_opt.step(); adam_opt.step()
        with torch.no_grad():
            model.radial_gain.clamp_(RADIAL_GAIN_MIN, RADIAL_GAIN_MAX); model.hash_gain.clamp_(0, HASH_GAIN_MAX)
            for ep, lp in zip(ema_m.parameters(), model.parameters()): ep.mul_(EMA_DECAY).add_(lp, alpha=1-EMA_DECAY)
        if step % LOG_EVERY == 0: print(f"step {step:04d} | loss {loss.item():.4f} | time {time.time()-start_time:.1f}s")
        step += 1
    artifact_audit(ema_m, OUT_PATH)

if __name__ == "__main__": main()
