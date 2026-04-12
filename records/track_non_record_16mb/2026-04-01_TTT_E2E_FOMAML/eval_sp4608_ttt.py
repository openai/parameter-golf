"""Naive TTT eval on sp4608 8xH100 model.

Loads the bf16 checkpoint, adds prime MLPs, runs score-first TTT.
Model: 11L, 512d, 8 heads, 4 KV heads, MLP 4.0x, vocab 4608.
No BigramHash, no SmearGate, no VE. Has skip_gates (sigmoid lerp).
"""
from __future__ import annotations
import glob
import math
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    from flash_attn import flash_attn_func

# ── Config ──
DATA_PATH      = os.environ.get("DATA_PATH", "./data_sp4608/datasets/fineweb10B_sp4608")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data_sp4608/tokenizers/fineweb_4608_bpe.model")
CHECKPOINT     = os.environ.get("CHECKPOINT", "sp4608_model.pt")
SEED           = int(os.environ.get("SEED", 1337))

VOCAB_SIZE     = 4608
NUM_LAYERS     = 11
MODEL_DIM      = 512
NUM_HEADS      = 8
NUM_KV_HEADS   = 4
MLP_MULT       = 4.0
LOGIT_SOFTCAP  = 30.0
ROPE_BASE      = 10000.0
QK_GAIN_INIT   = 1.5
ROPE_DIMS      = 16
XSA_LAST_N     = 11
SEQ_LEN        = 2048

PRIME_RANK     = int(os.environ.get("PRIME_RANK", 256))
PRIME_LAYERS   = [int(x) for x in os.environ.get("PRIME_LAYERS", "0,1,2,3,4,5,6,7,8,9,10").split(",")]
TTT_LR         = float(os.environ.get("TTT_LR", 0.1))
TTT_CHUNK      = int(os.environ.get("TTT_CHUNK", 1024))

# ── Data ──
_HEADER_INTS = 256
_HEADER_DTYPE = np.dtype("<i4")
_TOKEN_DTYPE = np.dtype("<u2")
_HEADER_BYTES = _HEADER_INTS * _HEADER_DTYPE.itemsize
_MMAP_CACHE: dict[str, np.memmap] = {}

def load_data_shard(file):
    key = str(file)
    if key not in _MMAP_CACHE:
        header = np.fromfile(file, dtype=_HEADER_DTYPE, count=_HEADER_INTS)
        n = int(header[2])
        _MMAP_CACHE[key] = np.memmap(file, mode="r", dtype=_TOKEN_DTYPE,
                                      offset=_HEADER_BYTES, shape=(n,))
    return torch.from_numpy(_MMAP_CACHE[key])

def load_validation_tokens(pattern, seq_len):
    files = sorted(glob.glob(pattern))
    tokens = torch.cat([load_data_shard(Path(p)) for p in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

def build_sentencepiece_luts(sp, vocab_size, device):
    table_size = max(int(sp.vocab_size()), vocab_size)
    base_bytes = np.zeros(table_size, dtype=np.int16)
    has_space = np.zeros(table_size, dtype=np.bool_)
    is_boundary = np.ones(table_size, dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes, dtype=torch.int16, device=device),
            torch.tensor(has_space, dtype=torch.bool, device=device),
            torch.tensor(is_boundary, dtype=torch.bool, device=device))

# ── Model ──

class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2).float() / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = None

    def forward(self, seq_len, device, dtype):
        if self._cache is None or self._cache[0] != seq_len or self._cache[1].device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cache = (seq_len, freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :])
        return self._cache[1].to(dtype), self._cache[2].to(dtype)

def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        xr, xp = x[..., :rope_dims], x[..., rope_dims:]
        h = rope_dims // 2
        x1, x2 = xr[..., :h], xr[..., h:]
        xr = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((xr, xp), dim=-1)
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        y_g = y.reshape(B, T, Hkv, H // Hkv, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x, q_w, k_w, v_w, out_w, v_embed=None):
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        return F.linear(y.reshape(bsz, seqlen, dim), out_w.to(x.dtype))

class MLP(nn.Module):
    def forward(self, x, up_w, down_w):
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
        return F.linear(x.square(), down_w.to(x.dtype))

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                 qk_gain_init, layer_idx=0, ln_scale=True):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP()
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                prime_up=None, prime_down=None, prime_norm=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, q_w, k_w, v_w, out_w)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        # Prime MLP (before main MLP)
        if prime_up is not None and prime_down is not None:
            h = prime_norm(x_out) if prime_norm is not None else F.rms_norm(x_out, (x_out.size(-1),))
            h = F.leaky_relu(F.linear(h, prime_up.to(x_out.dtype)), negative_slope=0.5).square()
            x_out = x_out + F.linear(h, prime_down.to(x_out.dtype))
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * \
                self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out


class GPT_SP4608(nn.Module):
    def __init__(self, prime_rank=256, prime_layers=None):
        super().__init__()
        dim = MODEL_DIM
        n = NUM_LAYERS
        self.num_layers = n
        self.logit_softcap = LOGIT_SOFTCAP
        self.tok_emb = nn.Embedding(VOCAB_SIZE, dim)
        self.num_encoder_layers = n // 2  # 5
        self.num_decoder_layers = n - self.num_encoder_layers  # 6
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)  # 5
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, dim, dtype=torch.float32))

        head_dim = dim // NUM_HEADS
        kv_dim = NUM_KV_HEADS * head_dim
        mlp_dim = int(MLP_MULT * dim)
        self.qo_bank = nn.Parameter(torch.empty(2 * n, dim, dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * n, kv_dim, dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(n, mlp_dim, dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(n, dim, mlp_dim))

        self.blocks = nn.ModuleList([
            Block(dim, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, ROPE_BASE,
                  QK_GAIN_INIT, layer_idx=i, ln_scale=True)
            for i in range(n)
        ])
        head_dim = dim // NUM_HEADS
        for block in self.blocks:
            block.attn.rope_dims = ROPE_DIMS
            block.attn.rotary = Rotary(head_dim, base=ROPE_BASE, train_seq_len=1024, rope_dims=ROPE_DIMS)
        for i in range(max(0, n - XSA_LAST_N), n):
            self.blocks[i].attn.use_xsa = True

        self.final_norm = RMSNorm()

        # Prime MLPs
        self.prime_layers = prime_layers or []
        self.prime_norms = nn.ModuleDict()
        self.prime_ups = nn.ParameterDict()
        self.prime_downs = nn.ParameterDict()
        for li in self.prime_layers:
            self.prime_norms[str(li)] = RMSNorm()
            self.prime_ups[str(li)] = nn.Parameter(torch.empty(prime_rank, dim))
            self.prime_downs[str(li)] = nn.Parameter(torch.zeros(dim, prime_rank))
            nn.init.orthogonal_(self.prime_ups[str(li)])

    def prime_named_params(self):
        for li in self.prime_layers:
            yield f"prime_up_{li}", self.prime_ups[str(li)]
            yield f"prime_down_{li}", self.prime_downs[str(li)]

    def forward_logits(self, input_ids, prime_overrides=None):
        n = self.num_layers
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            prime_up, prime_down, prime_norm = None, None, None
            if i in self.prime_layers:
                si = str(i)
                if prime_overrides:
                    prime_up = prime_overrides[f"prime_up_{i}"]
                    prime_down = prime_overrides[f"prime_down_{i}"]
                else:
                    prime_up = self.prime_ups[si]
                    prime_down = self.prime_downs[si]
                prime_norm = self.prime_norms[si]
            x = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                prime_up=prime_up, prime_down=prime_down, prime_norm=prime_norm)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
                scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = torch.lerp(scaled_skip, x, g)
            prime_up, prime_down, prime_norm = None, None, None
            if bi in self.prime_layers:
                si = str(bi)
                if prime_overrides:
                    prime_up = prime_overrides[f"prime_up_{bi}"]
                    prime_down = prime_overrides[f"prime_down_{bi}"]
                else:
                    prime_up = self.prime_ups[si]
                    prime_down = self.prime_downs[si]
                prime_norm = self.prime_norms[si]
            x = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                prime_up=prime_up, prime_down=prime_down, prime_norm=prime_norm)
        x = self.final_norm(x)
        logits_proj = F.linear(x, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids, prime_overrides=None):
        logits = self.forward_logits(input_ids, prime_overrides=prime_overrides)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1), reduction="mean")


# ── TTT Eval ──

def reset_primes(model):
    with torch.no_grad():
        for n, p in model.prime_named_params():
            if "down" in n:
                p.zero_()
            else:
                p.data = torch.nn.init.orthogonal_(
                    torch.empty_like(p, dtype=torch.float32)).to(p.dtype)

def eval_ttt(model, val_tokens, device, ttt_lr, chunk_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
             max_chunks=0, momentum=0.0):
    total_tokens = val_tokens.numel() - 1
    num_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens
    if max_chunks > 0:
        num_chunks = min(num_chunks, max_chunks)

    prime_params = [p for _, p in model.prime_named_params()]
    optimizer = torch.optim.SGD(prime_params, lr=ttt_lr, momentum=momentum) if ttt_lr > 0 else None

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    model.eval()
    for ci in range(num_chunks):
        chunk_start = ci * chunk_tokens
        chunk_end = min((ci + 1) * chunk_tokens, total_tokens)
        chunk_len = chunk_end - chunk_start
        if chunk_len < 2:
            continue

        chunk_data = val_tokens[chunk_start:chunk_end + 1].to(device=device, dtype=torch.int64)
        num_seqs = chunk_len // SEQ_LEN
        if num_seqs == 0:
            x = chunk_data[:-1].unsqueeze(0)
            y = chunk_data[1:].unsqueeze(0)
        else:
            x = chunk_data[:num_seqs * SEQ_LEN].reshape(num_seqs, SEQ_LEN)
            y = chunk_data[1:num_seqs * SEQ_LEN + 1].reshape(num_seqs, SEQ_LEN)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  y.reshape(-1), reduction="none")
            loss_sum += nll.to(torch.float64).sum()
            token_count += float(y.numel())
            tgt = y.reshape(-1)
            prev = x.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            byte_count += tb.sum()

        if optimizer is not None and ci < num_chunks - 1:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()
            optimizer.step()

        if ci % 1000 == 0 or ci == num_chunks - 1:
            elapsed = time.perf_counter() - t0
            bpb = (loss_sum.item() / max(token_count.item(), 1)) / math.log(2.0) * \
                  (token_count.item() / max(byte_count.item(), 1))
            print(f"  [{ci+1}/{num_chunks}] bpb={bpb:.6f} t={elapsed:.0f}s")

    val_bpb = (loss_sum / token_count).item() / math.log(2.0) * \
              (token_count.item() / byte_count.item())
    return val_bpb


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    val_tokens = load_validation_tokens(os.path.join(DATA_PATH, "fineweb_val_*.bin"), SEQ_LEN)
    luts = build_sentencepiece_luts(sp, VOCAB_SIZE, device)
    print(f"val tokens: {val_tokens.numel() - 1}")

    model = GPT_SP4608(prime_rank=PRIME_RANK, prime_layers=PRIME_LAYERS).to(device).bfloat16()

    # Load checkpoint (base model keys only)
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model_sd = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
    model.load_state_dict(model_sd)
    print(f"loaded {loaded}/{len(sd)} keys from {CHECKPOINT}")
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"prime params: {sum(p.numel() for _, p in model.prime_named_params()):,}")

    for p in model.parameters():
        p.requires_grad_(False)
    for _, p in model.prime_named_params():
        p.requires_grad_(True)

    # Baseline
    print("\n=== Baseline (no TTT) ===")
    reset_primes(model)
    bl = eval_ttt(model, val_tokens, device, 0.0, TTT_CHUNK, *luts, max_chunks=5000)
    print(f"baseline: {bl:.6f}")

    # LR sweep (5K chunks)
    results = {}
    for lr in [0.03, 0.1, 0.3, 1.0]:
        print(f"\n=== TTT lr={lr} (5K chunks) ===")
        reset_primes(model)
        bpb = eval_ttt(model, val_tokens, device, lr, TTT_CHUNK, *luts, max_chunks=5000)
        results[lr] = bpb
        print(f"lr={lr}: {bpb:.6f} ({bpb - bl:+.6f})")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY (5K chunks, baseline={bl:.6f})")
    print(f"{'='*50}")
    for lr, bpb in sorted(results.items()):
        print(f"  lr={lr:5.2f}: {bpb:.6f} ({bpb - bl:+.6f})")

    # Full eval on best
    best_lr = min(results, key=results.get)
    if results[best_lr] < bl - 0.0001:
        print(f"\n=== Full eval lr={best_lr} ===")
        reset_primes(model)
        full_bpb = eval_ttt(model, val_tokens, device, best_lr, TTT_CHUNK, *luts)
        print(f"FULL: {full_bpb:.6f} (baseline: {bl:.6f}, delta: {full_bpb - bl:+.6f})")


if __name__ == "__main__":
    main()
