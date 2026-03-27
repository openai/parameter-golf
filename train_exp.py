"""
Lightweight experiment runner for Parameter Golf.
Trains only — skips post-training quantization and eval for fast iteration.
Just compare train_loss at the final step.
"""
from __future__ import annotations
import os, sys, time, math, glob, random, copy
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ── Hyperparameters ──
class HP:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", "exp")
    seed = int(os.environ.get("SEED", 42))

    iterations = int(os.environ.get("ITERATIONS", 1000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 5))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 65536))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    log_every = int(os.environ.get("LOG_EVERY", 100))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 16))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 7))
    model_dim = int(os.environ.get("MODEL_DIM", 448))
    num_heads = int(os.environ.get("NUM_HEADS", 7))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.5))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.04))

    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 20480))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    # Experimental flags
    use_swiglu = bool(int(os.environ.get("USE_SWIGLU", "0")))
    depth_recurrence = int(os.environ.get("DEPTH_RECURRENCE", 0))
    trigram_vocab_size = int(os.environ.get("TRIGRAM_VOCAB_SIZE", 0))

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight", "smear", "bigram.scale")

# ── Muon Optimizer ──
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, backend_steps = group["lr"], group["momentum"], group["backend_steps"]
            total = sum(p.numel() for p in params)
            updates = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for p in params:
                if p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "buf" not in state: state["buf"] = torch.zeros_like(g)
                    buf = state["buf"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum)  # nesterov
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates[curr:curr+p.numel()] = g.reshape(-1)
                curr += p.numel()
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                g = updates[curr:curr+p.numel()].view_as(p).to(p.dtype)
                if wd > 0: p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()

# ── Data Loading ──
def load_shard(f):
    h = np.fromfile(f, dtype="<i4", count=256)
    n = int(h[2])
    return torch.from_numpy(np.fromfile(f, dtype="<u2", count=n, offset=1024).astype(np.uint16))

class TokenStream:
    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern))
        self.idx = 0
        self.tokens = load_shard(self.files[0])
        self.pos = 0
    def take(self, n):
        chunks = []
        rem = n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.idx = (self.idx + 1) % len(self.files)
                self.tokens = load_shard(self.files[self.idx])
                self.pos = 0; continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos:self.pos+k])
            self.pos += k; rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

# ── Model ──
class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def forward(self, x): return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), persistent=False)
        self._cache = (0, None, None)
    def forward(self, seq_len, device, dtype):
        if self._cache[0] != seq_len or self._cache[1] is None:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len, freqs.cos()[None,None,:,:], freqs.sin()[None,None,:,:])
        return self._cache[1].to(dtype), self._cache[2].to(dtype)

def apply_rope(x, cos, sin):
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.nh, self.nkv, self.hd = num_heads, num_kv_heads, dim // num_heads
        kv_dim = num_kv_heads * self.hd
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.hd, base=rope_base)
    def forward(self, x):
        B, T, D = x.shape
        q = self.c_q(x).reshape(B,T,self.nh,self.hd).transpose(1,2)
        k = self.c_k(x).reshape(B,T,self.nkv,self.hd).transpose(1,2)
        v = self.c_v(x).reshape(B,T,self.nkv,self.hd).transpose(1,2)
        q, k = F.rms_norm(q, (self.hd,)), F.rms_norm(k, (self.hd,))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None,:,None,None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.nkv != self.nh))
        return self.proj(y.transpose(1,2).contiguous().reshape(B,T,D))

class MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        h = int(mult * dim)
        self.fc = CastedLinear(dim, h, bias=False)
        self.proj = CastedLinear(h, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())

class SwiGLU_MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        h = int(mult * dim * 2 / 3)  # Adjust for gate
        self.gate = CastedLinear(dim, h, bias=False)
        self.up = CastedLinear(dim, h, bias=False)
        self.proj = CastedLinear(h, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.up(x))

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(x.dtype))[None,None,:]
        x_prev = torch.cat([torch.zeros_like(x[:,:1]), x[:,:-1]], dim=1)
        return (1-g)*x + g*x_prev

class BigramHash(nn.Module):
    def __init__(self, vocab, dim, model_dim):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, dim); nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(dim, model_dim, bias=False) if dim != model_dim else None
        if self.proj: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def forward(self, ids):
        t = ids.to(torch.int32)
        mod = self.vocab - 1
        h = torch.empty_like(t); h[...,0] = mod
        h[...,1:] = torch.bitwise_xor(36313*t[...,1:], 27191*t[...,:-1]) % mod
        out = self.embed(h.long())
        if self.proj: out = self.proj(out)
        return out * self.scale.to(out.dtype)

class TrigramHash(nn.Module):
    """Like BigramHash but uses 3-token context."""
    def __init__(self, vocab, dim, model_dim):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, dim); nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(dim, model_dim, bias=False) if dim != model_dim else None
        if self.proj: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def forward(self, ids):
        t = ids.to(torch.int32)
        mod = self.vocab - 1
        h = torch.empty_like(t); h[...,0] = mod; h[...,1] = mod
        h[...,2:] = (torch.bitwise_xor(torch.bitwise_xor(36313*t[...,2:], 27191*t[...,1:-1]), 15473*t[...,:-2])) % mod
        out = self.embed(h.long())
        if self.proj: out = self.proj(out)
        return out * self.scale.to(out.dtype)

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, use_swiglu=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = Attention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = SwiGLU_MLP(dim, mlp_mult) if use_swiglu else MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
    def forward(self, x, x0):
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None,None,:]*x + mix[1][None,None,:]*x0
        x = x + self.attn_scale.to(x.dtype)[None,None,:] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None,None,:] * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.logit_softcap = args.logit_softcap
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        nn.init.normal_(self.tok_emb.weight, std=args.tied_embed_init_std)
        self.bigram = BigramHash(args.bigram_vocab_size, args.bigram_dim, args.model_dim) if args.bigram_vocab_size > 0 else None
        self.trigram = TrigramHash(args.trigram_vocab_size, args.bigram_dim, args.model_dim) if args.trigram_vocab_size > 0 else None
        self.smear = SmearGate(args.model_dim)

        num_layers = args.num_layers
        if args.depth_recurrence > 0:
            # Share weights: create fewer blocks, repeat them
            actual_blocks = args.depth_recurrence
            self.blocks = nn.ModuleList([Block(args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.rope_base, args.qk_gain_init, args.use_swiglu) for _ in range(actual_blocks)])
            self.block_schedule = list(range(actual_blocks)) * (num_layers // actual_blocks)
            num_layers = len(self.block_schedule)
        else:
            self.blocks = nn.ModuleList([Block(args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.rope_base, args.qk_gain_init, args.use_swiglu) for _ in range(num_layers)])
            self.block_schedule = list(range(num_layers))

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, args.model_dim, dtype=torch.float32))
        self.final_norm = RMSNorm()
        self._init_weights()

    def _init_weights(self):
        n = len(self.block_schedule)
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if getattr(mod, "_zero_init", False): nn.init.zeros_(mod.weight)
                elif mod.weight.ndim == 2 and min(mod.weight.shape) >= 64:
                    nn.init.orthogonal_(mod.weight, gain=1.0)
                    if ".proj" in name:
                        with torch.no_grad(): mod.weight.mul_(1/math.sqrt(2*n))

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None: x = x + self.bigram(input_ids)
        if self.trigram is not None: x = x + self.trigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x; skips = []
        schedule = self.block_schedule
        for i in range(self.num_encoder_layers):
            x = self.blocks[schedule[i]](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None,None,:] * skips.pop()
            x = self.blocks[schedule[self.num_encoder_layers + i]](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = self.logit_softcap * torch.tanh(F.linear(x, self.tok_emb.weight) / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

# ── Training ──
def main():
    args = HP()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = GPT(args).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear): m.float()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if (p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model_params:{n_params} run_id:{args.run_id}")

    # Optimizer setup
    block_params = list(model.blocks.named_parameters())
    matrix_params = [p for n,p in block_params if p.ndim==2 and not any(pat in n for pat in CONTROL_PATTERNS)]
    scalar_params = [p for n,p in block_params if p.ndim<2 or any(pat in n for pat in CONTROL_PATTERNS)]
    scalar_params.append(model.skip_weights)
    scalar_params.append(model.smear.gate)
    if model.bigram: scalar_params.append(model.bigram.scale)
    if model.trigram: scalar_params.append(model.trigram.scale)

    tok_groups = [{"params": [model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}]
    if model.bigram:
        tok_groups.append({"params": [model.bigram.embed.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr})
        if model.bigram.proj: matrix_params.append(model.bigram.proj.weight)
    if model.trigram:
        tok_groups.append({"params": [model.trigram.embed.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr})
        if model.trigram.proj: matrix_params.append(model.trigram.proj.weight)

    opt_tok = torch.optim.AdamW(tok_groups, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=0.04)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar]

    grad_accum = 8
    grad_scale = 1.0 / grad_accum
    stream = TokenStream(args.train_files)

    model.train()
    for step in range(1, args.iterations + 1):
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(grad_accum):
            local_tokens = args.train_batch_tokens // grad_accum
            chunk = stream.take(local_tokens + 1).to(torch.int64)
            x = chunk[:-1].reshape(-1, args.train_seq_len).to(device)
            y = chunk[1:].reshape(-1, args.train_seq_len).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            total_loss += loss.item()
            (loss * grad_scale).backward()
        total_loss /= grad_accum

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for g in opt_muon.param_groups:
            g["momentum"] = (1-frac)*args.muon_momentum_warmup_start + frac*args.muon_momentum

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()

        if step % args.log_every == 0 or step <= 10 or step == args.iterations:
            print(f"step:{step}/{args.iterations} train_loss:{total_loss:.4f}", flush=True)

    print(f"FINAL step:{args.iterations} train_loss:{total_loss:.4f} params:{n_params}")

if __name__ == "__main__":
    main()
