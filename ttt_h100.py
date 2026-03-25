"""
H100 TTT experiment script.
Runs after training to find optimal TTT configuration.

Usage:
  python ttt_h100.py --model logs/<RUN_ID>_model.int8.ptz

Tests (on full validation set, with held-out split):
  1. No TTT (baseline)
  2. SGD TTT (PR #398: 20ep, lr=0.008)
  3. AdamW TTT (10ep, lr=0.0005)
  4. AdamW TTT + Cosine LR
  5. Dynamic Eval (lr=0.001)
  6. AdamW TTT + Dynamic Eval
  7. SGD TTT + Dynamic Eval
  8. TENT→TTT (norm recalib then full TTT)
"""

import argparse
import copy
import glob as glob_mod
import math
import pickle
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# ── Model ──

class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        self.seq_len_cached = 0
    def forward(self, seq_len):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.inv_freq.device)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached, self.sin_cached

def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos = cos[:x.shape[-3]].unsqueeze(0).unsqueeze(2)
    sin = sin[:x.shape[-3]].unsqueeze(0).unsqueeze(2)
    return torch.cat([x1*cos + x2*sin, x1*(-sin) + x2*cos], dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.c_v = CastedLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))
    def forward(self, x, x0):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_kv_heads, self.head_dim)
        cos, sin = self.rotary(T)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        q = apply_rotary_emb(F.rms_norm(q, (self.head_dim,)), cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        k = apply_rotary_emb(F.rms_norm(k, (self.head_dim,)), cos, sin)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=(self.num_heads != self.num_kv_heads))
        return self.proj(y.transpose(1,2).reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, mlp_mult * dim, bias=False)
        self.proj = CastedLinear(mlp_mult * dim, dim, bias=False)
    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn = Attention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.zeros(2, dim))
    def forward(self, x, x0):
        mix = torch.sigmoid(self.resid_mix).to(dtype=x.dtype)
        h = mix[0]*x + mix[1]*x0
        h = F.rms_norm(h, (h.size(-1),)) * self.attn_scale.to(dtype=h.dtype)
        x = x + self.attn(h, x0)
        xn = F.rms_norm(x, (x.size(-1),)) * self.mlp_scale.to(dtype=x.dtype)
        return x + self.mlp(xn)

class GPT(nn.Module):
    def __init__(self, vocab_size=1024, num_layers=9, model_dim=512,
                 num_heads=8, num_kv_heads=4, mlp_mult=2,
                 tie_embeddings=True, logit_softcap=30.0,
                 rope_base=10000.0, qk_gain_init=1.5):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

    def forward_logits(self, input_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None,None,:] * skips.pop()
            x = self.blocks[bi](x, x0)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               target_ids.reshape(-1), reduction="mean")


# ── Data / Dequant ──

def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=int(header[2]), offset=256*4).astype(np.uint16, copy=False))

def _to_tensor(x):
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x

def load_quantized_model(ptz_path):
    obj = pickle.loads(zlib.decompress(open(ptz_path, "rb").read()))
    out = {}
    qmeta = obj.get("qmeta", {})
    for name, q in obj["quantized"].items():
        q = _to_tensor(q); dtype = getattr(torch, obj["dtypes"][name])
        s = _to_tensor(obj["scales"][name]).to(torch.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1]*(q.ndim-1)))).to(dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(dtype)
    for name, t in obj["passthrough"].items():
        t = _to_tensor(t).detach().to("cpu")
        orig = obj.get("passthrough_orig_dtypes", {}).get(name)
        if isinstance(orig, str): t = t.to(getattr(torch, orig))
        out[name] = t
    return out

def infer_config(sd):
    nl = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    dim = sd["tok_emb.weight"].shape[1]
    nh = sd["blocks.0.attn.q_gain"].shape[0]
    nkv = sd["blocks.0.attn.c_k.weight"].shape[0] // (dim // nh)
    mm = sd["blocks.0.mlp.fc.weight"].shape[0] // dim
    return {"vocab_size": sd["tok_emb.weight"].shape[0], "num_layers": nl,
            "model_dim": dim, "num_heads": nh, "num_kv_heads": nkv, "mlp_mult": mm}

def fresh_model(sd, cfg):
    m = GPT(**cfg).to(DEVICE).to(DTYPE)
    m.load_state_dict(sd, strict=False)
    return m


# ── Eval ──

@torch.no_grad()
def eval_bpb(model, tokens, seq_len=1024, batch_size=16):
    model.eval()
    usable = ((tokens.numel()-1)//seq_len)*seq_len
    tokens = tokens[:usable+1]
    total_loss = total_tokens = 0
    for start in range(0, usable, batch_size*seq_len):
        end = min(start + batch_size*seq_len+1, usable+1)
        local = tokens[start:end].to(DEVICE, torch.int64)
        n = (local.numel()-1)//seq_len
        if n == 0: break
        x = local[:n*seq_len].reshape(n, seq_len)
        y = local[1:n*seq_len+1].reshape(n, seq_len)
        with torch.amp.autocast('cuda', dtype=DTYPE):
            loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    model.train()
    return (total_loss / total_tokens) / math.log(2)


# ── TTT Methods ──

def _make_batches(tokens, seq_len, batch_size=16):
    usable = ((tokens.numel()-1)//seq_len)*seq_len
    t = tokens[:usable+1].to(DEVICE, torch.int64)
    num_seqs = usable // seq_len
    batches = []
    for i in range(0, num_seqs, batch_size):
        be = min(i+batch_size, num_seqs)
        s, e = i*seq_len, be*seq_len+1
        local = t[s:e]; n = (local.numel()-1)//seq_len
        if n == 0: break
        batches.append((local[:n*seq_len].reshape(n, seq_len),
                        local[1:n*seq_len+1].reshape(n, seq_len)))
    return batches


def ttt_sgd(model, tokens, epochs=20, lr=0.008, seq_len=1024):
    """PR #398 style SGD TTT."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    batches = _make_batches(tokens, seq_len)
    for ep in range(epochs):
        for x, y in batches:
            with torch.amp.autocast('cuda', dtype=DTYPE):
                loss = model(x, y)
            opt.zero_grad(); loss.backward(); opt.step()


def ttt_adamw(model, tokens, epochs=10, lr=0.0005, wd=0.01, seq_len=1024, cosine=False):
    """AdamW TTT with optional cosine schedule."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    batches = _make_batches(tokens, seq_len)
    total_steps = epochs * len(batches)

    if cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    step = 0
    for ep in range(epochs):
        for x, y in batches:
            with torch.amp.autocast('cuda', dtype=DTYPE):
                loss = model(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if cosine:
                scheduler.step()
            step += 1


def eval_dynamic(model, tokens, lr=0.001, seq_len=1024):
    """Dynamic evaluation — adapt weights during scoring."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    usable = ((tokens.numel()-1)//seq_len)*seq_len
    tokens = tokens[:usable+1]
    total_loss = total_tokens = 0
    for start in range(0, usable, seq_len):
        end = min(start + seq_len+1, usable+1)
        local = tokens[start:end].to(DEVICE, torch.int64)
        if local.numel() < 2: break
        n = local.numel()-1
        x = local[:n].unsqueeze(0)
        y = local[1:n+1].unsqueeze(0)
        with torch.amp.autocast('cuda', dtype=DTYPE):
            loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        opt.zero_grad(); loss.backward(); opt.step()
    return (total_loss / total_tokens) / math.log(2)


def get_norm_params(model):
    params = []
    for name, p in model.named_parameters():
        if any(k in name for k in ["attn_scale", "mlp_scale", "q_gain", "skip_weights"]):
            params.append(p)
    return params


def tent_norm_recalib(model, tokens, epochs=30, lr=0.01, seq_len=1024):
    """TENT-style norm recalibration before TTT."""
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    norm_params = get_norm_params(model)
    for p in norm_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(norm_params, lr=lr)
    batches = _make_batches(tokens, seq_len)
    for ep in range(epochs):
        for x, y in batches:
            with torch.amp.autocast('cuda', dtype=DTYPE):
                loss = model(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--val-dir", type=str,
                        default="./data/datasets/fineweb10B_sp1024/")
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    model_path = args.model
    if "*" in model_path:
        matches = sorted(glob_mod.glob(model_path))
        model_path = matches[0]

    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    sd = load_quantized_model(model_path)
    cfg = infer_config(sd)
    print(f"Config: {cfg}")

    # Load ALL validation shards
    val_files = sorted(Path(args.val_dir).glob("fineweb_val_*.bin"))
    all_tokens = torch.cat([load_data_shard(f) for f in val_files])
    print(f"Total validation tokens: {all_tokens.numel():,}")

    # 80/20 split for overfitting detection
    split = int(all_tokens.numel() * 0.8)
    ttt_tokens = all_tokens[:split]
    eval_tokens = all_tokens[split:]
    print(f"TTT: {ttt_tokens.numel():,}, Eval: {eval_tokens.numel():,}\n")

    results = []

    def run(name, setup_fn, eval_fn=None):
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        model = fresh_model(sd, cfg)
        t0 = time.time()
        if setup_fn:
            setup_fn(model)
        if eval_fn:
            bpb_full = eval_fn(model, all_tokens)
            bpb_held = eval_fn(model, eval_tokens)
        else:
            bpb_full = eval_bpb(model, all_tokens)
            bpb_held = eval_bpb(model, eval_tokens)
        elapsed = time.time() - t0
        print(f"  full={bpb_full:.4f} held={bpb_held:.4f} time={elapsed:.0f}s")
        results.append((name, bpb_full, bpb_held, elapsed))
        del model
        torch.cuda.empty_cache()

    # 1. Baseline
    run("Baseline (no TTT)", None)
    bpb_base_full = results[0][1]
    bpb_base_held = results[0][2]

    # 2. SGD TTT (PR #398: 20ep lr=0.008)
    run("SGD 20ep lr=0.008",
        lambda m: ttt_sgd(m, ttt_tokens, epochs=20, lr=0.008))

    # 3. AdamW TTT 10ep
    run("AdamW 10ep lr=0.0005",
        lambda m: ttt_adamw(m, ttt_tokens, epochs=10, lr=0.0005))

    # 4. AdamW + Cosine LR
    run("AdamW 10ep cosine",
        lambda m: ttt_adamw(m, ttt_tokens, epochs=10, lr=0.0005, cosine=True))

    # 5. Dynamic Eval only
    run("Dynamic Eval lr=0.001", None,
        eval_fn=lambda m, t: eval_dynamic(m, t, lr=0.001))

    # 6. AdamW + Dynamic Eval
    run("AdamW 10ep + DynEval",
        lambda m: ttt_adamw(m, ttt_tokens, epochs=10, lr=0.0005),
        eval_fn=lambda m, t: eval_dynamic(m, t, lr=0.001))

    # 7. SGD + Dynamic Eval
    run("SGD 20ep + DynEval",
        lambda m: ttt_sgd(m, ttt_tokens, epochs=20, lr=0.008),
        eval_fn=lambda m, t: eval_dynamic(m, t, lr=0.001))

    # 8. Norm recalib → AdamW TTT
    def tent_then_adamw(m):
        tent_norm_recalib(m, ttt_tokens, epochs=30, lr=0.01)
        ttt_adamw(m, ttt_tokens, epochs=10, lr=0.0005)
    run("TENT→AdamW 10ep", tent_then_adamw)

    # 9. Norm recalib → SGD TTT
    def tent_then_sgd(m):
        tent_norm_recalib(m, ttt_tokens, epochs=30, lr=0.01)
        ttt_sgd(m, ttt_tokens, epochs=20, lr=0.008)
    run("TENT→SGD 20ep", tent_then_sgd)

    # ── Summary ──
    print("\n" + "=" * 80)
    print(f"{'Method':<30} {'Full':>8} {'F.Δ':>8} {'Held':>8} {'H.Δ':>8} {'Time':>6}")
    print("-" * 72)
    results.sort(key=lambda r: r[2])  # sort by held-out
    for name, bf, bh, t in results:
        fd = bf - bpb_base_full
        hd = bh - bpb_base_held
        print(f"{name:<30} {bf:>8.4f} {fd:>+8.4f} {bh:>8.4f} {hd:>+8.4f} {t:>5.0f}s")

    best = results[0]
    print(f"\nBest (held-out): {best[0]} → {best[2]:.4f} BPB")
    print(f"\nTotal experiments: {len(results)}")


if __name__ == "__main__":
    main()
