"""
Estimate artifact size for different layer counts without training or GPU.
Usage:
    python estimate_artifact_size.py              # test 12, 13, 14, 15 layers
    NUM_LAYERS=14 python estimate_artifact_size.py
"""
import io, os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── config (mirrors train_gpt_v2.py defaults) ──────────────────────────────
VOCAB_SIZE       = int(os.environ.get("VOCAB_SIZE",  1024))
MODEL_DIM        = int(os.environ.get("MODEL_DIM",   512))
NUM_HEADS        = int(os.environ.get("NUM_HEADS",   8))
NUM_KV_HEADS     = int(os.environ.get("NUM_KV_HEADS", 4))
MLP_MULT         = float(os.environ.get("MLP_MULT",  3.0))
BIGRAM_VOCAB     = int(os.environ.get("BIGRAM_VOCAB_SIZE", 10240))
BIGRAM_DIM       = int(os.environ.get("BIGRAM_DIM",  128))
XSA_LAST_N       = int(os.environ.get("XSA_LAST_N",  4))
ROPE_DIMS        = int(os.environ.get("ROPE_DIMS",   16))
LN_SCALE         = bool(int(os.environ.get("LN_SCALE", 1)))
TIE_EMBEDDINGS   = bool(int(os.environ.get("TIE_EMBEDDINGS", 1)))
LOGIT_SOFTCAP    = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
QK_GAIN_INIT     = float(os.environ.get("QK_GAIN_INIT", 1.5))
ROPE_BASE        = float(os.environ.get("ROPE_BASE",  10000.0))
TIED_EMBED_INIT  = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

CONTROL_PATTERNS = "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,bigram.scale".split(",")
FP16_KEEP        = ["tok_emb"]  # not blocks.11.attn.c_k — layer index changes; keep tok_emb only for safety

ZSTD_LEVEL = int(os.environ.get("ZSTD_LEVEL", 22))

# ── quantization helpers (copied verbatim from train_gpt_v2.py) ────────────
INT8_CLIP_Q = 0.9999984

def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
        return q, scale.to(torch.float16)
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(max(clip_abs / 127.0, 1e-9), dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8)
    return q, scale

def quantize_intN(t, clip_range=31):
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / clip_range).clamp_min(1e-12).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -(clip_range+1), clip_range).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / clip_range, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -(clip_range+1), clip_range).to(torch.int8)
    return q, scale

def _cat(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if "bigram" in name: return "bigram"
    if ".attn." in name: return "attn"
    return "other"

def mixed_quantize_int6(sd):
    result = {}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        cat = _cat(name)
        if not t.is_floating_point() or t.numel() <= 8192:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            continue
        if any(p in name for p in CONTROL_PATTERNS):
            result[name] = t.float()
            continue
        if any(p in name for p in FP16_KEEP):
            result[name] = t.to(torch.float16)
            continue
        if cat in {"mlp", "attn", "bigram"}:
            clip = 7 if cat == "mlp" else 15
            q, s = quantize_intN(t, clip)
            result[name + ".q"] = q
            result[name + ".scale"] = s
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
    return result

def serialize_and_compress(quant_result):
    buf = io.BytesIO()
    torch.save(quant_result, buf)
    raw = buf.getvalue()
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL, threads=-1)
        return cctx.compress(raw)
    except ImportError:
        import zlib
        return zlib.compress(raw, 9)

# ── minimal model (structure only, no forward needed) ─────────────────────

class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),))

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=0, ln_scale=False, layer_idx=0):
        super().__init__()
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.use_xsa = False

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        h = int(dim * mlp_mult)
        self.gate = nn.Linear(dim, h, bias=False)
        self.up   = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)
        self.mlp_scale = nn.Parameter(torch.ones(1))

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, rope_dims=0, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims, ln_scale, layer_idx)
        self.mlp  = MLP(dim, mlp_mult)
        self.resid_mix = nn.Parameter(torch.zeros(dim))
        if ln_scale:
            s = 1.0 / math.sqrt(layer_idx + 1)
            self.attn_scale_ln = nn.Parameter(torch.full((1,), s))
            self.mlp_scale_ln  = nn.Parameter(torch.full((1,), s))

class BigramHashEmbedding(nn.Module):
    def __init__(self, vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, bigram_dim)
        self.proj  = nn.Linear(bigram_dim, model_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(1))

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.smear = nn.Parameter(torch.zeros(dim))

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 bigram_vocab, bigram_dim, xsa_last_n, rope_dims, ln_scale, tie_embeddings,
                 logit_softcap, qk_gain_init, rope_base, tied_embed_init):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram  = BigramHashEmbedding(bigram_vocab, bigram_dim, model_dim) if bigram_vocab > 0 else None
        num_enc = num_layers // 2
        num_skip = min(num_enc, num_layers - num_enc)
        self.skip_weights = nn.Parameter(torch.ones(num_skip, model_dim))
        self.smear  = SmearGate(model_dim)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  rope_dims=rope_dims, layer_idx=i, ln_scale=ln_scale)
            for i in range(num_layers)
        ])
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else nn.Linear(model_dim, vocab_size, bias=False)

def estimate(num_layers):
    model = GPT(
        vocab_size=VOCAB_SIZE, num_layers=num_layers, model_dim=MODEL_DIM,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, mlp_mult=MLP_MULT,
        bigram_vocab=BIGRAM_VOCAB, bigram_dim=BIGRAM_DIM, xsa_last_n=XSA_LAST_N,
        rope_dims=ROPE_DIMS, ln_scale=LN_SCALE, tie_embeddings=TIE_EMBEDDINGS,
        logit_softcap=LOGIT_SOFTCAP, qk_gain_init=QK_GAIN_INIT, rope_base=ROPE_BASE,
        tied_embed_init=TIED_EMBED_INIT,
    )
    n_params = sum(p.numel() for p in model.parameters())
    sd = model.state_dict()
    quant = mixed_quantize_int6(sd)
    compressed = serialize_and_compress(quant)
    size_bytes = len(compressed)
    under = size_bytes < 16_000_000
    flag = "✅" if under else "❌ OVER LIMIT"
    print(f"  layers={num_layers:2d}  params={n_params/1e6:.2f}M  "
          f"artifact={size_bytes:,} bytes ({size_bytes/1e6:.3f} MB)  {flag}")
    return size_bytes

if __name__ == "__main__":
    layer_counts = [int(x) for x in os.environ.get("NUM_LAYERS", "").split(",")] if os.environ.get("NUM_LAYERS") else [12, 13, 14, 15]
    print(f"Artifact size estimates (vocab={VOCAB_SIZE}, dim={MODEL_DIM}, heads={NUM_HEADS}/{NUM_KV_HEADS}, mlp={MLP_MULT}x)")
    print(f"Compressor: zstd-{ZSTD_LEVEL}   Limit: 16,000,000 bytes")
    print()
    for L in layer_counts:
        estimate(L)
