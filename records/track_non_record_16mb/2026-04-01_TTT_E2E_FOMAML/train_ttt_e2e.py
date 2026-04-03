"""TTT-E2E: Test-Time Training with End-to-End Meta-Learning for Parameter Golf.

Two-phase training on base PR 1105 model:
  Phase 1: Load pretrained checkpoint (standard training already done)
  Phase 2: FOMAML meta-fine-tuning with prime MLPs on last 3 blocks

Eval: Score-first TTT with prime MLP adaptation on val shard.

Usage:
  python -u train_ttt_e2e.py              # Phase 2 + eval
  PHASE1_STEPS=5000 python -u train_ttt_e2e.py  # Full Phase 1 + Phase 2 + eval
"""
from __future__ import annotations
import glob
import math
import os
import time
import uuid
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

# ── Config ──────────────────────────────────────────────────────────────────

DATA_PATH     = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
CHECKPOINT     = os.environ.get("CHECKPOINT", "final_model.pt")
SEED           = int(os.environ.get("SEED", 1337))

# Model (must match PR 1105)
VOCAB_SIZE     = int(os.environ.get("VOCAB_SIZE", 1024))
NUM_LAYERS     = int(os.environ.get("NUM_LAYERS", 11))
MODEL_DIM      = int(os.environ.get("MODEL_DIM", 512))
NUM_HEADS      = int(os.environ.get("NUM_HEADS", 8))
NUM_KV_HEADS   = int(os.environ.get("NUM_KV_HEADS", 4))
MLP_MULT       = float(os.environ.get("MLP_MULT", 3.0))
LOGIT_SOFTCAP  = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
ROPE_BASE      = float(os.environ.get("ROPE_BASE", 10000.0))
QK_GAIN_INIT   = float(os.environ.get("QK_GAIN_INIT", 1.5))
ROPE_DIMS      = int(os.environ.get("ROPE_DIMS", 16))
XSA_LAST_N     = int(os.environ.get("XSA_LAST_N", 11))
BIGRAM_VOCAB   = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
BIGRAM_DIM     = int(os.environ.get("BIGRAM_DIM", 128))
VE_ENABLED     = bool(int(os.environ.get("VE_ENABLED", "1")))
VE_DIM         = int(os.environ.get("VE_DIM", 128))
VE_LAYERS      = os.environ.get("VE_LAYERS", "9,10")

# Prime MLP
PRIME_RANK     = int(os.environ.get("PRIME_RANK", 256))
PRIME_LAYERS   = [int(x) for x in os.environ.get("PRIME_LAYERS", "8,9,10").split(",")]

# Phase 1 (skip if CHECKPOINT exists)
PHASE1_STEPS   = int(os.environ.get("PHASE1_STEPS", 0))
PHASE1_LR      = float(os.environ.get("PHASE1_LR", 0.025))

# Phase 2: FOMAML
PHASE2_STEPS   = int(os.environ.get("PHASE2_STEPS", 1500))
PHASE2_OUTER_LR = float(os.environ.get("PHASE2_OUTER_LR", 0.003))
PHASE2_INNER_LR = float(os.environ.get("PHASE2_INNER_LR", 0.01))
PHASE2_INNER_K = int(os.environ.get("PHASE2_INNER_K", 1))
SEQ_LEN        = int(os.environ.get("SEQ_LEN", 2048))
BATCH_SEQS     = int(os.environ.get("BATCH_SEQS", 4))  # sequences per micro-batch

# Eval TTT
TTT_LR         = float(os.environ.get("TTT_LR", 0.01))
TTT_CHUNK      = int(os.environ.get("TTT_CHUNK", 1024))  # tokens per TTT mini-batch

# ── Data loading ────────────────────────────────────────────────────────────

_HEADER_INTS = 256
_HEADER_DTYPE = np.dtype("<i4")
_TOKEN_DTYPE = np.dtype("<u2")
_HEADER_BYTES = _HEADER_INTS * _HEADER_DTYPE.itemsize
_MMAP_CACHE: dict[str, np.memmap] = {}

def _read_num_tokens(file: Path) -> int:
    header = np.fromfile(file, dtype=_HEADER_DTYPE, count=_HEADER_INTS)
    return int(header[2])

def load_data_shard(file: Path) -> Tensor:
    key = str(file)
    if key not in _MMAP_CACHE:
        n = _read_num_tokens(file)
        _MMAP_CACHE[key] = np.memmap(file, mode="r", dtype=_TOKEN_DTYPE,
                                      offset=_HEADER_BYTES, shape=(n,))
    return torch.from_numpy(_MMAP_CACHE[key])

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
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

# ── Model (PR 1105 architecture + prime MLPs) ──────────────────────────────

class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, rope_dims: int = 0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2).float() / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: tuple[int, Tensor, Tensor] | None = None

    def forward(self, seq_len: int, device, dtype):
        if self._cache is None or self._cache[0] != seq_len or self._cache[1].device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
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

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def forward(self, token_ids):
        t = token_ids.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        h = self.embed(out.long())
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, ve_dim, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids):
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        # No stored weights — they come from banks
    def forward(self, x, up_w, down_w):
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
        return F.linear(x.square(), down_w.to(x.dtype))

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                 qk_gain_init, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                v_embed=None, prime_up=None, prime_down=None, prime_norm=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor,
                             q_w, k_w, v_w, out_w, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        # Prime MLP (if present): runs BEFORE main MLP, own norm + residual
        if prime_up is not None and prime_down is not None:
            h = prime_norm(x_out) if prime_norm is not None else F.rms_norm(x_out, (x_out.size(-1),))
            h = F.leaky_relu(F.linear(h, prime_up.to(x_out.dtype)), negative_slope=0.5).square()
            x_out = x_out + F.linear(h, prime_down.to(x_out.dtype))
        # Main MLP
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * \
                self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out


class GPT_TTT(nn.Module):
    """PR 1105 GPT with prime MLPs for TTT-E2E."""

    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, logit_softcap, rope_base, qk_gain_init,
                 bigram_vocab_size=0, bigram_dim=128, xsa_last_n=0,
                 rope_dims=0, ln_scale=True,
                 ve_enabled=False, ve_dim=128, ve_layers="9,10",
                 prime_rank=256, prime_layers=None):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Parameter banks (same as PR 1105)
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)
        self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                  qk_gain_init, layer_idx=i, ln_scale=ln_scale)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, rope_dims=rope_dims)
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True

        # Value embeddings
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim_ve = num_kv_heads * head_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()
        self.final_norm = RMSNorm()
        self.lm_head = None  # tied embeddings

        # ── Prime MLPs (TTT-E2E) ──
        self.prime_layers = prime_layers or []
        self.prime_rank = prime_rank
        self.prime_norms = nn.ModuleDict()
        self.prime_ups = nn.ParameterDict()
        self.prime_downs = nn.ParameterDict()
        for li in self.prime_layers:
            self.prime_norms[str(li)] = RMSNorm()
            self.prime_ups[str(li)] = nn.Parameter(torch.empty(prime_rank, model_dim))
            self.prime_downs[str(li)] = nn.Parameter(torch.zeros(model_dim, prime_rank))
            nn.init.orthogonal_(self.prime_ups[str(li)])

    def prime_named_params(self):
        """Yield (name, param) for all prime MLP parameters (for FOMAML)."""
        for li in self.prime_layers:
            yield f"prime_up_{li}", self.prime_ups[str(li)]
            yield f"prime_down_{li}", self.prime_downs[str(li)]

    def _get_ve(self, layer_idx, input_ids, ve_cache):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve']
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    def forward_logits(self, input_ids, prime_overrides=None):
        """Forward pass returning logits. prime_overrides: dict of name->tensor for FOMAML."""
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips = []
        ve_cache = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            # Get prime weights (override or model's own)
            prime_up, prime_down, prime_norm = None, None, None
            if i in self.prime_layers:
                si = str(i)
                if prime_overrides is not None:
                    prime_up = prime_overrides[f"prime_up_{i}"]
                    prime_down = prime_overrides[f"prime_down_{i}"]
                else:
                    prime_up = self.prime_ups[si]
                    prime_down = self.prime_downs[si]
                prime_norm = self.prime_norms[si]
            x = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                v_embed=ve, prime_up=prime_up, prime_down=prime_down, prime_norm=prime_norm)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            prime_up, prime_down, prime_norm = None, None, None
            if bi in self.prime_layers:
                si = str(bi)
                if prime_overrides is not None:
                    prime_up = prime_overrides[f"prime_up_{bi}"]
                    prime_down = prime_overrides[f"prime_down_{bi}"]
                else:
                    prime_up = self.prime_ups[si]
                    prime_down = self.prime_downs[si]
                prime_norm = self.prime_norms[si]
            x = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, prime_up=prime_up, prime_down=prime_down, prime_norm=prime_norm)
        x = self.final_norm(x)
        logits_proj = F.linear(x, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids, prime_overrides=None):
        logits = self.forward_logits(input_ids, prime_overrides=prime_overrides)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1), reduction="mean")


# ── Phase 2: FOMAML ────────────────────────────────────────────────────────

def fomaml_step(model, x_inner, y_inner, x_outer, y_outer, inner_lr, K=1):
    """One FOMAML meta-training step. Returns outer loss value."""
    # 1. Detach prime MLP weights (break gradient through update)
    adapted = {}
    for name, p in model.prime_named_params():
        adapted[name] = p.detach().clone().requires_grad_(True)

    # 2. Inner loop: K steps of SGD on adapted weights
    for _k in range(K):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inner_loss = model(x_inner, y_inner, prime_overrides=adapted)
        grads = torch.autograd.grad(inner_loss, list(adapted.values()))
        adapted = {n: p - inner_lr * g
                   for (n, p), g in zip(adapted.items(), grads)}

    # Mark adapted tensors so backward() populates their .grad
    for v in adapted.values():
        v.retain_grad()

    # 3. Outer loss with adapted weights (base model gets normal gradients)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outer_loss = model(x_outer, y_outer, prime_overrides=adapted)
    outer_loss.backward()

    # 4. FOMAML: copy adapted param gradients to prime init params
    for name, p in model.prime_named_params():
        g = adapted[name].grad
        if g is not None:
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

    return outer_loss.item()


# ── Eval: Score-first TTT ──────────────────────────────────────────────────

def eval_ttt(model, val_tokens, device, sp, ttt_lr, chunk_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """Score-first TTT eval: score each chunk, then update prime MLPs."""
    seq_len = SEQ_LEN
    total_tokens = val_tokens.numel() - 1

    # Collect prime params for SGD
    prime_params = [p for _, p in model.prime_named_params()]
    optimizer = torch.optim.SGD(prime_params, lr=ttt_lr)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    num_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens
    t0 = time.perf_counter()

    model.eval()
    for ci in range(num_chunks):
        chunk_start = ci * chunk_tokens
        chunk_end = min((ci + 1) * chunk_tokens, total_tokens)
        chunk_len = chunk_end - chunk_start
        if chunk_len < 2:
            continue

        # Get chunk data
        chunk_data = val_tokens[chunk_start:chunk_end + 1].to(device=device, dtype=torch.int64)
        # Process in seq_len windows
        num_seqs = chunk_len // seq_len
        if num_seqs == 0:
            # Handle tail
            x = chunk_data[:-1].unsqueeze(0)
            y = chunk_data[1:].unsqueeze(0)
        else:
            x = chunk_data[:num_seqs * seq_len].reshape(num_seqs, seq_len)
            y = chunk_data[1:num_seqs * seq_len + 1].reshape(num_seqs, seq_len)

        # ── Phase 1: SCORE (no grad, record BPB) ──
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

        # ── Phase 2: TRAIN prime MLPs on scored chunk ──
        if ci < num_chunks - 1:  # don't train on last chunk
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                train_loss = model(x, y)
            train_loss.backward()
            optimizer.step()

        if ci % 500 == 0 or ci == num_chunks - 1:
            elapsed = time.perf_counter() - t0
            running_loss = loss_sum.item() / max(token_count.item(), 1)
            running_bpb = running_loss / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1))
            print(f"  ttt [{ci+1}/{num_chunks}] bpb={running_bpb:.6f} time={elapsed:.1f}s")

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    print(f"ttt:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
          f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb


# ── Eval: baseline (no TTT) ────────────────────────────────────────────────

def eval_baseline(model, val_tokens, device, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """Standard eval without TTT for comparison."""
    seq_len = SEQ_LEN
    total_tokens = val_tokens.numel() - 1
    num_seqs = total_tokens // seq_len

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    batch = 16
    with torch.no_grad():
        for si in range(0, num_seqs, batch):
            ei = min(si + batch, num_seqs)
            bsz = ei - si
            raw = val_tokens[si * seq_len:(ei * seq_len) + 1].to(device=device, dtype=torch.int64)
            x = raw[:-1].reshape(bsz, seq_len)
            y = raw[1:].reshape(bsz, seq_len)
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

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    print(f"baseline: val_loss={val_loss:.6f} val_bpb={val_bpb:.6f}")
    return val_loss, val_bpb


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load tokenizer and data
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    val_tokens = load_validation_tokens(
        os.path.join(DATA_PATH, "fineweb_val_*.bin"), SEQ_LEN)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, VOCAB_SIZE, device)
    print(f"val tokens: {val_tokens.numel() - 1}")

    # Build model
    model = GPT_TTT(
        vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, model_dim=MODEL_DIM,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, mlp_mult=MLP_MULT,
        logit_softcap=LOGIT_SOFTCAP, rope_base=ROPE_BASE, qk_gain_init=QK_GAIN_INIT,
        bigram_vocab_size=BIGRAM_VOCAB, bigram_dim=BIGRAM_DIM,
        xsa_last_n=XSA_LAST_N, rope_dims=ROPE_DIMS, ln_scale=True,
        ve_enabled=VE_ENABLED, ve_dim=VE_DIM, ve_layers=VE_LAYERS,
        prime_rank=PRIME_RANK, prime_layers=PRIME_LAYERS,
    ).to(device).bfloat16()

    total_params = sum(p.numel() for p in model.parameters())
    prime_params_count = sum(p.numel() for _, p in model.prime_named_params())
    prime_norm_count = sum(p.numel() for p in model.prime_norms.parameters())
    print(f"model params: {total_params} (prime MLP: {prime_params_count + prime_norm_count})")

    # Load Phase 1 checkpoint
    if os.path.exists(CHECKPOINT):
        print(f"loading checkpoint: {CHECKPOINT}")
        sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        # Load matching keys (ignore prime MLP keys not in checkpoint)
        model_sd = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded += 1
        model.load_state_dict(model_sd)
        print(f"loaded {loaded}/{len(sd)} keys from checkpoint")
    else:
        print(f"WARNING: no checkpoint at {CHECKPOINT}, training from scratch")

    # ── Baseline eval (before Phase 2) ──
    print("\n=== Baseline eval (no TTT, no meta-training) ===")
    eval_baseline(model, val_tokens, device,
                  base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    # ── Phase 2: FOMAML meta-fine-tuning ──
    if PHASE2_STEPS > 0:
        print(f"\n=== Phase 2: FOMAML ({PHASE2_STEPS} steps, inner_lr={PHASE2_INNER_LR}, K={PHASE2_INNER_K}) ===")

        # Load train data
        train_files = sorted(glob.glob(os.path.join(DATA_PATH, "fineweb_train_*.bin")))
        train_tokens = torch.cat([load_data_shard(Path(f)) for f in train_files]).contiguous()
        print(f"train tokens: {train_tokens.numel()}")

        # Freeze base model, only train prime MLP params
        for n, p in model.named_parameters():
            if "prime_" not in n:
                p.requires_grad_(False)

        prime_opt_params = list(model.prime_ups.parameters()) + \
                           list(model.prime_downs.parameters()) + \
                           list(model.prime_norms.parameters())

        optimizer = torch.optim.AdamW(prime_opt_params, lr=PHASE2_OUTER_LR, weight_decay=0.0)

        t0 = time.perf_counter()
        total_train = train_tokens.numel() - 1
        model.train()

        for step in range(PHASE2_STEPS):
            # Sample random mini-batch (need 2x for inner + outer)
            needed = BATCH_SEQS * 2 * SEQ_LEN + 1
            offset = torch.randint(0, total_train - needed, (1,)).item()
            data = train_tokens[offset:offset + needed].to(device=device, dtype=torch.int64)

            half = BATCH_SEQS * SEQ_LEN
            x_inner = data[:half].reshape(BATCH_SEQS, SEQ_LEN)
            y_inner = data[1:half + 1].reshape(BATCH_SEQS, SEQ_LEN)
            x_outer = data[half:2 * half].reshape(BATCH_SEQS, SEQ_LEN)
            y_outer = data[half + 1:2 * half + 1].reshape(BATCH_SEQS, SEQ_LEN)

            optimizer.zero_grad(set_to_none=True)
            loss_val = fomaml_step(model, x_inner, y_inner, x_outer, y_outer,
                                   inner_lr=PHASE2_INNER_LR, K=PHASE2_INNER_K)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 100 == 0 or step == PHASE2_STEPS - 1:
                elapsed = time.perf_counter() - t0
                ms_per_step = 1000.0 * elapsed / (step + 1)
                print(f"  phase2 [{step+1}/{PHASE2_STEPS}] loss={loss_val:.4f} "
                      f"ms/step={ms_per_step:.0f} elapsed={elapsed:.0f}s")

        print(f"Phase 2 done in {time.perf_counter() - t0:.0f}s")

        # Re-enable all gradients
        for p in model.parameters():
            p.requires_grad_(True)

        # Save Phase 2 checkpoint
        torch.save(model.state_dict(), "ttt_e2e_model.pt")
        print("saved ttt_e2e_model.pt")

        # Eval after meta-training (no TTT yet)
        print("\n=== Post-Phase2 eval (no TTT) ===")
        eval_baseline(model, val_tokens, device,
                      base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    # ── Eval with score-first TTT ──
    print(f"\n=== TTT eval (lr={TTT_LR}, chunk={TTT_CHUNK}) ===")
    # Reset prime MLPs to meta-learned init before TTT
    if os.path.exists("ttt_e2e_model.pt"):
        sd = torch.load("ttt_e2e_model.pt", map_location="cpu", weights_only=True)
        prime_keys = [k for k in sd if "prime_" in k]
        for k in prime_keys:
            model.state_dict()[k].copy_(sd[k])
        print(f"reset prime MLPs from ttt_e2e_model.pt ({len(prime_keys)} keys)")

    # Only prime MLP params get gradients during TTT eval
    for p in model.parameters():
        p.requires_grad_(False)
    for _, p in model.prime_named_params():
        p.requires_grad_(True)

    eval_ttt(model, val_tokens, device, sp, TTT_LR, TTT_CHUNK,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)


if __name__ == "__main__":
    main()
