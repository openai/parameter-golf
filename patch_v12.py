#!/usr/bin/env python3
"""
Raki V12 — Per-Sample SLOT-24 + Pre-quant TTT + Full SOTA Stack.

Key: Per-Sample SLOT-24 (PR #1376, 0.709 BPB SOTA):
  Per-window hidden delta [B,1,512] + logit bias [B,1,1024] optimized
  with 24 AdamW steps on already-scored context. Model weights frozen.
  Based on arXiv:2505.12392v2. -0.30+ BPB from eval alone.

Pre-quant TTT (PR #1376): 6 epochs AdamW on EMA model before GPTQ.
  Freeze first 2 blocks, cosine LR 5e-4.

Training: Turbo-Muon (AOL f32), XSA-ALL, LeakyReLU(0.5)², MLP 3×,
  LN Scale 1/√(L+1), Late QAT (STE int6), Partial RoPE 16/64,
  EMA 0.997, BigramHash(2048), Adaptive Markov, GPTQ clip search,
  Brotli-11, auto qmax, Muon WD 0.04.

Usage: python3 patch_v12.py && torchrun --standalone --nproc_per_node=8 train_gpt.py
  5090 test: SLOT_ENABLED=0 ITERATIONS=1500 ... python3 train_gpt.py
"""
import sys

with open("train_gpt.py", "r") as f:
    code = f.read()

changes = 0


def patch(anchor, replacement, label):
    global code, changes
    if anchor in code:
        code = code.replace(anchor, replacement, 1)
        changes += 1
        return True
    else:
        print(f"FAIL: {label}\n  anchor not found: {repr(anchor[:120])}")
        sys.exit(1)


# ============================================================
# 1. Dependencies
# ============================================================
patch(
    'from __future__ import annotations',
    '''from __future__ import annotations
try:
    import zstandard as _zstd_check  # noqa: F401
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "zstandard", "-q"])
try:
    import brotli as _brotli_check  # noqa: F401
except ImportError:
    import subprocess as _sp2
    _sp2.check_call([sys.executable, "-m", "pip", "install", "brotli", "-q"])''',
    "dependencies")

# ============================================================
# 2. Hyperparameters
# ============================================================
patch('    num_layers = int(os.environ.get("NUM_LAYERS", 9))',
      '    num_layers = int(os.environ.get("NUM_LAYERS", 11))', "NUM_LAYERS=11")
patch('    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))',
      '    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))', "WARMDOWN=4000")
patch('    mlp_mult = int(os.environ.get("MLP_MULT", 2))',
      '    mlp_mult = int(os.environ.get("MLP_MULT", 3))', "MLP_MULT=3")
patch('    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))',
      '    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0))', "QK_GAIN=5.0")

# ============================================================
# 3. Config block
# ============================================================
patch(
    "from torch.nn.parallel import DistributedDataParallel as DDP",
    '''from torch.nn.parallel import DistributedDataParallel as DDP
import gc, brotli, zstandard as zstd

MUON_WD = float(os.environ.get("MUON_WD", "0"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.997"))
EMA_START_FRAC = float(os.environ.get("EMA_START_FRAC", "0.85"))
RAKI_POWER = float(os.environ.get("RAKI_POWER", "0.10"))
BIGRAM_BUCKETS = int(os.environ.get("BIGRAM_BUCKETS", "2048"))
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", "0"))
ROPE_DIMS = int(os.environ.get("ROPE_DIMS", "16"))
BLOCK_QUANT_MAX = int(os.environ.get("BLOCK_QUANT_MAX", "31"))
GPTQ_CLIP_SEARCH = bool(int(os.environ.get("GPTQ_CLIP_SEARCH", "1")))
GPTQ_PERCENTILES = [0.999, 0.9995, 0.9999, 0.99999, 1.0]
XSA_LAST_N = int(os.environ.get("XSA_LAST_N", "11"))
LN_SCALE = bool(int(os.environ.get("LN_SCALE", "1")))
LATE_QAT_THRESHOLD = float(os.environ.get("LATE_QAT_THRESHOLD", "0.85"))
# SLOT config (PR #1376)
SLOT_ENABLED = bool(int(os.environ.get("SLOT_ENABLED", "1")))
SLOT_STEPS = int(os.environ.get("SLOT_STEPS", "24"))
SLOT_LR = float(os.environ.get("SLOT_LR", "0.024"))
SLOT_LR_MIN = float(os.environ.get("SLOT_LR_MIN", "0.001"))
SLOT_STRIDE = int(os.environ.get("SLOT_STRIDE", "96"))
# Pre-quant TTT config (PR #1376)
TTT_ENABLED = bool(int(os.environ.get("TTT_ENABLED", "1")))
TTT_EPOCHS = int(os.environ.get("TTT_EPOCHS", "6"))
TTT_LR = float(os.environ.get("TTT_LR", "5e-4"))
TTT_FREEZE_BLOCKS = int(os.environ.get("TTT_FREEZE_BLOCKS", "2"))

_QAT = {"on": False}


def _ste_fake_quant(w: Tensor, qmax: int) -> Tensor:
    with torch.no_grad():
        scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / float(qmax)
    w_q = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w + (w_q - w).detach()


def _run_prequant_ttt(base_model: nn.Module, val_tokens: Tensor, device: torch.device,
                      seq_len: int = 1024, epochs: int = 6, lr: float = 5e-4,
                      freeze_blocks: int = 2) -> int:
    """Pre-quant AdamW TTT: fine-tune on val before GPTQ. Freeze first N blocks."""
    base_model.train()
    for i, block in enumerate(base_model.blocks):
        if i < freeze_blocks:
            for p in block.parameters():
                p.requires_grad_(False)
    trainable = [p for p in base_model.parameters() if p.requires_grad]
    total_tokens = val_tokens.numel() - 1
    n_seqs = max(1, total_tokens // seq_len)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs * n_seqs, 1), eta_min=1e-6)
    tok_count = 0
    for ep in range(epochs):
        for i in range(0, total_tokens - seq_len, seq_len):
            x = val_tokens[i:i + seq_len].unsqueeze(0).to(device, dtype=torch.int64)
            y = val_tokens[i + 1:i + seq_len + 1].unsqueeze(0).to(device, dtype=torch.int64)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model.forward_per_token(x, y).mean()
            loss.backward()
            opt.step()
            sched.step()
            tok_count += seq_len
    for p in base_model.parameters():
        p.requires_grad_(True)
    return tok_count


class _GPUMarkov:
    def __init__(self, pattern: str, V: int, device: torch.device):
        files = sorted(glob.glob(pattern))
        hdr_bytes = 256 * np.dtype("<i4").itemsize
        hdr = np.fromfile(files[0], dtype="<i4", count=256)
        ntok = min(int(hdr[2]), 2_000_000)
        tok = np.fromfile(files[0], dtype="<u2", count=ntok, offset=hdr_bytes).astype(np.int32)
        counts = np.zeros((V, V), dtype=np.float64)
        np.add.at(counts, (tok[:-1], tok[1:]), 1.0)
        counts += 0.01
        probs = counts / counts.sum(axis=1, keepdims=True)
        log_probs = np.log(probs).astype(np.float16)
        ent = -(probs * np.log(probs)).sum(axis=1).astype(np.float32)
        mn, mx = ent.min(), ent.max()
        ent_norm = (ent - mn) / (mx - mn) if mx > mn else np.full_like(ent, 0.5)
        self.log_probs = torch.tensor(log_probs, device=device)
        self.ent_norm = torch.tensor(ent_norm, dtype=torch.float16, device=device)
        self.loss_ema = 0.0
        self.loss_count = 0

    @torch.no_grad()
    def batch_weight(self, x: Tensor, y: Tensor, batch_loss: float = 0.0) -> float:
        if RAKI_POWER <= 0:
            return 1.0
        surp = -self.log_probs[x.reshape(-1), y.reshape(-1)].float()
        ent_w = self.ent_norm[x.reshape(-1)].float()
        bigram_score = (surp * ent_w).mean().item()
        if batch_loss > 0 and self.loss_count > 10:
            combined = bigram_score * min(batch_loss / max(self.loss_ema, 1e-6), 2.0)
        else:
            combined = bigram_score
        if batch_loss > 0:
            self.loss_ema = 0.99 * self.loss_ema + 0.01 * batch_loss if self.loss_count > 0 else batch_loss
            self.loss_count += 1
        return 1.0 + RAKI_POWER * min(combined / 5.0, 1.0)


class _EMA:
    def __init__(self):
        self.shadow: dict[str, Tensor] | None = None
        self.on = False
    def start(self, model: nn.Module):
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}
        self.on = True
    def update(self, model: nn.Module):
        if not self.on or self.shadow is None:
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].lerp_(p.data, 1.0 - EMA_DECAY)
    def apply(self, model: nn.Module):
        if self.shadow is None:
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n])''',
    "config block")

# ============================================================
# 4. Turbo-Muon (AOL float32)
# ============================================================
patch(
    '''def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
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
    return X.T if transposed else X''',
    '''def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    G32 = G.float()
    D_r = (G32 @ G32.T).diag().clamp_min(eps).sqrt()
    D_c = (G32.T @ G32).diag().clamp_min(eps).sqrt()
    X = (G32 / D_r[:, None] / D_c[None, :]).bfloat16()
    X /= X.norm() + eps
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    ns_steps = max(steps - 1, 1)
    for _ in range(ns_steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X''',
    "Turbo-Muon AOL")

# ============================================================
# 5. Muon WD
# ============================================================
patch(
    '''                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss''',
    '''                p.add_(g, alpha=-lr)
                if MUON_WD > 0:
                    p.mul_(1.0 - lr * MUON_WD)
                curr += p.numel()

        return loss''',
    "Muon WD")

# ============================================================
# 6. Late QAT STE
# ============================================================
patch(
    '''class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)''',
    '''class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if _QAT["on"] and self.weight.numel() > 65536:
            w = _ste_fake_quant(w, BLOCK_QUANT_MAX)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)''',
    "Late QAT STE")

# ============================================================
# 7-8. Partial RoPE
# ============================================================
patch(
    '''def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)''',
    '''def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rd = min(ROPE_DIMS, x.size(-1))
    if rd >= x.size(-1):
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    half = rd // 2
    x_rope, x_pass = x[..., :rd], x[..., rd:]
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rot = torch.cat((x1 * cos[..., :half] + x2 * sin[..., :half],
                        x1 * (-sin[..., :half]) + x2 * cos[..., :half]), dim=-1)
    return torch.cat((x_rot, x_pass), dim=-1)''',
    "Partial RoPE")

patch(
    '''    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))''',
    '''    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        rope_d = min(ROPE_DIMS, dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_d, 2, dtype=torch.float32) / rope_d))''',
    "Rotary init")

# ============================================================
# 9-10. XSA
# ============================================================
patch(
    '''class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()''',
    '''class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        use_xsa: bool = False,
    ):
        super().__init__()
        self.use_xsa = use_xsa''',
    "XSA init")

patch(
    '''        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)''',
    '''        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        if self.use_xsa:
            v_x = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1) if self.num_kv_heads != self.num_heads else v
            dot_yv = (y * v_x).sum(-1, keepdim=True)
            y = y - (dot_yv / (v_x * v_x).sum(-1, keepdim=True).clamp_min(1e-8)) * v_x
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)''',
    "XSA forward")

# ============================================================
# 11. LeakyReLU(0.5)²
# ============================================================
patch(
    '''    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())''',
    '''    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())''',
    "LeakyReLU²")

# ============================================================
# 12-13. Block with layer_idx, XSA, LN Scale
# ============================================================
patch(
    '''class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)''',
    '''class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 layer_idx=0, use_xsa=False):
        super().__init__()
        self._ln_s = 1.0 / math.sqrt(layer_idx + 1) if LN_SCALE else 1.0
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult)''',
    "Block init")

patch(
    '''    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x''',
    '''    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        _h = self.attn_norm(x)
        if self._ln_s != 1.0: _h = _h * self._ln_s
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(_h)
        _h = self.mlp_norm(x)
        if self._ln_s != 1.0: _h = _h * self._ln_s
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(_h)
        return x''',
    "Block forward LN Scale")

# ============================================================
# 14. GPT: blocks + BigramHash
# ============================================================
patch(
    '''        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)''',
    '''        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, use_xsa=(i >= num_layers - XSA_LAST_N) if XSA_LAST_N > 0 else False)
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.bigram_table = nn.Embedding(BIGRAM_BUCKETS, model_dim)
        nn.init.normal_(self.bigram_table.weight, std=0.002)
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)''',
    "GPT blocks + BigramHash")

# ============================================================
# 15. GPT forward: BigramHash
# ============================================================
patch(
    '''    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))''',
    '''    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if input_ids.size(1) >= 2:
            ids = input_ids.long()
            bh = (ids[:, 1:] * 36313 + ids[:, :-1] * 51749) % BIGRAM_BUCKETS
            x[:, 1:] = x[:, 1:] + self.bigram_table(bh).to(x.dtype)
        x = F.rms_norm(x, (x.size(-1),))''',
    "BigramHash forward")

# ============================================================
# 16. forward_per_token (with SLOT delta support) + TRAINING
# ============================================================
patch(
    '''        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------''',
    '''        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_per_token(self, input_ids: Tensor, target_ids: Tensor,
                          h_delta: Tensor | None = None, l_bias: Tensor | None = None) -> Tensor:
        """Per-token loss. Optional h_delta [B,1,D] and l_bias [B,1,V] for SLOT."""
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        if T >= 2:
            ids = input_ids.long()
            bh = (ids[:, 1:] * 36313 + ids[:, :-1] * 51749) % BIGRAM_BUCKETS
            x[:, 1:] = x[:, 1:] + self.bigram_table(bh).to(x.dtype)
        x = F.rms_norm(x, (x.size(-1),))
        if h_delta is not None:
            x = x + h_delta.to(x.dtype)
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if l_bias is not None:
            logits = logits + l_bias.to(logits.dtype)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            target_ids.reshape(-1), reduction="none").reshape(B, T)


# -----------------------------
# TRAINING
# -----------------------------''',
    "forward_per_token with SLOT delta")

# ============================================================
# 17. Eval: standard sliding window + Per-Sample SLOT-24
# ============================================================
patch(
    '''def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)''',

    '''def _slot_eval(args, base_m, rank, world_size, device, val_tokens,
                  base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """Per-Sample SLOT-24 eval (PR #1376): hidden delta + logit bias, AdamW, causal."""
    seq_len = args.train_seq_len
    stride = SLOT_STRIDE
    mdim, V = args.model_dim, args.vocab_size
    total_tokens = val_tokens.numel() - 1
    all_starts = list(range(0, total_tokens - seq_len + 1, stride))
    if not all_starts: all_starts = [0]
    rank_starts = [s for i, s in enumerate(all_starts) if i % world_size == rank]
    vls = torch.zeros((), device=device, dtype=torch.float64)
    vtc = torch.zeros((), device=device, dtype=torch.float64)
    vbc = torch.zeros((), device=device, dtype=torch.float64)
    for p in base_m.parameters(): p.requires_grad_(False)
    base_m.eval()
    for s in rank_starts:
        x = val_tokens[s:s + seq_len].unsqueeze(0).to(device, dtype=torch.int64)
        y = val_tokens[s + 1:s + seq_len + 1].unsqueeze(0).to(device, dtype=torch.int64)
        sc_start = 0 if s == 0 else (seq_len - stride)
        n_scored = min(seq_len - sc_start, total_tokens - s - sc_start)
        if n_scored <= 0: continue
        h_d = torch.zeros(1, 1, mdim, device=device, dtype=torch.float32, requires_grad=True)
        l_b = torch.zeros(1, 1, V, device=device, dtype=torch.float32, requires_grad=True)
        ctx_end = sc_start
        if ctx_end > 0:
            opt = torch.optim.AdamW([h_d, l_b], lr=SLOT_LR, weight_decay=0.0)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SLOT_STEPS, eta_min=SLOT_LR_MIN)
            for _ in range(SLOT_STEPS):
                opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_m.forward_per_token(x, y, h_d, l_b)
                ptl[0, :ctx_end].mean().backward()
                opt.step(); sch.step()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ptl = base_m.forward_per_token(x, y, h_d.detach(), l_b.detach())
            vls += ptl[0, sc_start:sc_start + n_scored].to(torch.float64).sum()
            vtc += float(n_scored)
            g = s + sc_start
            prev = val_tokens[g:g + n_scored].to(device, dtype=torch.int64)
            tgt = val_tokens[g + 1:g + n_scored + 1].to(device, dtype=torch.int64)
            n = min(prev.size(0), tgt.size(0))
            tb = base_bytes_lut[tgt[:n]].to(torch.int16)
            tb += (has_leading_space_lut[tgt[:n]] & ~is_boundary_token_lut[prev[:n]]).to(torch.int16)
            vbc += tb.to(torch.float64).sum()
    for p in base_m.parameters(): p.requires_grad_(True)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vls, op=dist.ReduceOp.SUM)
        dist.all_reduce(vtc, op=dist.ReduceOp.SUM)
        dist.all_reduce(vbc, op=dist.ReduceOp.SUM)
    vl = vls / vtc
    return float(vl.item()), float(vl.item() / math.log(2.0) * vtc.item() / vbc.item())


def eval_val(
    args: Hyperparameters, model: nn.Module, rank: int, world_size: int,
    device: torch.device, grad_accum_steps: int, val_tokens: Tensor,
    base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    stride = EVAL_STRIDE if 0 < EVAL_STRIDE < seq_len else 0
    raw_model = model.module if hasattr(model, "module") else model
    base_m = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
    # SLOT mode for final eval
    if SLOT_ENABLED and stride > 0:
        return _slot_eval(args, base_m, rank, world_size, device, val_tokens,
                          base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    if stride > 0:
        total_tokens = val_tokens.numel() - 1
        all_starts = list(range(0, total_tokens - seq_len + 1, stride))
        if not all_starts: all_starts = [0]
        rank_starts = [s for i, s in enumerate(all_starts) if i % world_size == rank]
        with torch.inference_mode():
            bs = max(1, min(16, args.val_batch_size // (seq_len * max(world_size, 1))))
            for bi in range(0, len(rank_starts), bs):
                batch_starts = rank_starts[bi:bi + bs]
                xs = [val_tokens[s:s + seq_len].to(torch.int64) for s in batch_starts]
                ys = [val_tokens[s + 1:s + seq_len + 1].to(torch.int64) for s in batch_starts]
                x = torch.stack(xs).to(device=device, non_blocking=True)
                y = torch.stack(ys).to(device=device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    ptl = base_m.forward_per_token(x, y).detach()
                for wi, s in enumerate(batch_starts):
                    sc_start = 0 if s == 0 else (seq_len - stride)
                    n_scored = min(seq_len - sc_start, total_tokens - s - sc_start)
                    if n_scored <= 0: continue
                    val_loss_sum += ptl[wi, sc_start:sc_start + n_scored].to(torch.float64).sum()
                    val_token_count += float(n_scored)
                    g = s + sc_start
                    prev = val_tokens[g:g + n_scored].to(device, dtype=torch.int64)
                    tgt = val_tokens[g + 1:g + n_scored + 1].to(device, dtype=torch.int64)
                    n = min(prev.size(0), tgt.size(0))
                    tb = base_bytes_lut[tgt[:n]].to(torch.int16)
                    tb += (has_leading_space_lut[tgt[:n]] & ~is_boundary_token_lut[prev[:n]]).to(torch.int16)
                    val_byte_count += tb.to(torch.float64).sum()
    else:
        local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
        local_batch_seqs = max(1, local_batch_tokens // seq_len)
        total_seqs = (val_tokens.numel() - 1) // seq_len
        seq_start = (total_seqs * rank) // world_size
        seq_end = (total_seqs * (rank + 1)) // world_size
        with torch.inference_mode():
            for bss in range(seq_start, seq_end, local_batch_seqs):
                bse = min(bss + local_batch_seqs, seq_end)
                rs, re = bss * seq_len, bse * seq_len + 1
                local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
                x = local[:-1].reshape(-1, seq_len)
                y = local[1:].reshape(-1, seq_len)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    bl = model(x, y).detach()
                btc = float(y.numel())
                val_loss_sum += bl.to(torch.float64) * btc
                val_token_count += btc
                prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
                tb = base_bytes_lut[tgt_ids].to(torch.int16)
                tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
                val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)''',
    "eval with SLOT-24")

# ============================================================
# 18. GPTQ clip search quantization
# ============================================================
patch(
    '''def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale''',

    '''def quantize_float_tensor(t: Tensor, qmax: int = 127) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        if GPTQ_CLIP_SEARCH and t32.numel():
            best_q = best_scale = None
            best_mse = torch.full((t32.shape[0],), float('inf'), device=t32.device)
            for pct in GPTQ_PERCENTILES:
                ca = t32.abs().amax(dim=1) if pct >= 1.0 else torch.quantile(t32.abs(), pct, dim=1)
                sc = (ca / float(qmax)).clamp_min(1.0 / float(qmax))
                cl = torch.maximum(torch.minimum(t32, ca[:, None]), -ca[:, None])
                qq = torch.clamp(torch.round(cl / sc[:, None]), -qmax, qmax)
                mse = ((t32 - qq * sc[:, None]) ** 2).mean(dim=1)
                improved = mse < best_mse
                if best_q is None:
                    best_q, best_scale, best_mse = qq.to(torch.int8), sc, mse
                else:
                    best_q[improved] = qq[improved].to(torch.int8)
                    best_scale[improved] = sc[improved]
                    best_mse[improved] = mse[improved]
            return best_q.contiguous(), best_scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
        clip_abs = (torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                    if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(qmax)).clamp_min(1.0 / float(qmax))
        q = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(qmax) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qmax, qmax).to(torch.int8).contiguous()
    return q, scale''',
    "GPTQ clip search")

# ============================================================
# 19. Block weights int6
# ============================================================
patch(
    '''        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)''',
    '''        stats["num_float_tensors"] += 1
        use_qmax = BLOCK_QUANT_MAX if "blocks." in name else 127
        q, s = quantize_float_tensor(t, qmax=use_qmax)''',
    "int6 blocks")

# ============================================================
# 20-24. Brotli compression + filenames + logs
# ============================================================
patch('    quant_blob = zlib.compress(quant_raw, level=9)',
      '    quant_blob = brotli.compress(quant_raw, quality=11)', "brotli-11")
for i in range(3):
    patch('"final_model.int8.ptz"', '"final_model.int8.ptbr"', f"filename {i+1}")
patch('quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")',
      'quant_state = torch.load(io.BytesIO(brotli.decompress(quant_blob_disk)), map_location="cpu")',
      "brotli decompress")
patch('f"Serialized model int8+zlib:', 'f"Serialized model int8+brotli:', "log 1")
patch('f"Total submission size int8+zlib:', 'f"Total submission size int8+brotli:', "log 2")
patch('f"final_int8_zlib_roundtrip val_loss', 'f"final_int8_brotli_roundtrip val_loss', "log 3")
patch('f"final_int8_zlib_roundtrip_exact val_loss', 'f"final_int8_brotli_roundtrip_exact val_loss', "log 4")

# ============================================================
# 25. BigramHash in optimizer
# ============================================================
patch('        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],',
      '        [{"params": [base_model.tok_emb.weight, base_model.bigram_table.weight], "lr": token_lr, "base_lr": token_lr}],',
      "bigram optimizer")

# ============================================================
# 26. Init
# ============================================================
patch("    training_time_ms = 0.0",
      '''    _markov = _GPUMarkov(args.train_files, args.vocab_size, device)
    _ema = _EMA()
    _ema_on = False
    log0(f"raki_v12: L={args.num_layers} mlp={args.mlp_mult}x bigram={BIGRAM_BUCKETS} rope={ROPE_DIMS} "
         f"xsa={XSA_LAST_N} slot={SLOT_ENABLED}/{SLOT_STEPS} ttt={TTT_ENABLED}/{TTT_EPOCHS} "
         f"power={RAKI_POWER} wd={MUON_WD} ema={EMA_DECAY}")
    training_time_ms = 0.0''',
      "init")

# ============================================================
# 27. Adaptive Markov curriculum
# ============================================================
patch("            (loss * grad_scale).backward()",
      '''            _cw = _markov.batch_weight(x, y, loss.item())
            (loss * grad_scale * _cw).backward()''',
      "Markov curriculum")

# ============================================================
# 28. Late QAT + EMA update
# ============================================================
patch("        zero_grad_all()\n\n        step += 1",
      '''        zero_grad_all()
        _prog = (training_time_ms + 1000.0 * (time.perf_counter() - t0)) / max(max_wallclock_ms or 1e18, 1.0)
        if LATE_QAT_THRESHOLD > 0 and _prog >= LATE_QAT_THRESHOLD and not _QAT["on"]:
            _QAT["on"] = True
            log0(f"raki_v12:qat_on step={step+1}")
        if _prog >= EMA_START_FRAC and not _ema_on:
            _ema.start(base_model); _ema_on = True
            log0(f"raki_v12:ema_on step={step+1}")
        _ema.update(base_model)
        step += 1''',
      "QAT + EMA")

# ============================================================
# 29. End of training: EMA → Pre-quant TTT → auto qmax
# ============================================================
patch('    if master_process:\n        torch.save(base_model.state_dict(), "final_model.pt")',
      '''    _QAT["on"] = False
    if _ema.on:
        _ema.apply(base_model)
        log0("raki_v12:ema_applied")
    torch._dynamo.reset()
    if TTT_ENABLED and TTT_EPOCHS > 0:
        log0(f"raki_v12:ttt_start ep={TTT_EPOCHS} lr={TTT_LR} freeze={TTT_FREEZE_BLOCKS}")
        _tc = _run_prequant_ttt(base_model, val_tokens, device, seq_len=args.train_seq_len,
                                epochs=TTT_EPOCHS, lr=TTT_LR, freeze_blocks=TTT_FREEZE_BLOCKS)
        log0(f"raki_v12:ttt_done tokens={_tc}")
    _code_bytes = len(code.encode("utf-8"))
    _lo, _hi = 15, 127
    _tsz = 0
    while _lo < _hi:
        _mid = (_lo + _hi + 1) // 2
        globals()["BLOCK_QUANT_MAX"] = _mid
        _tobj, _ = quantize_state_dict_int8(base_model.state_dict())
        _tbuf = io.BytesIO()
        torch.save(_tobj, _tbuf)
        _tsz = len(brotli.compress(_tbuf.getvalue(), quality=4))
        if _tsz + _code_bytes <= 16_000_000:
            _lo = _mid
        else:
            _hi = _mid - 1
        del _tobj, _tbuf; gc.collect()
    globals()["BLOCK_QUANT_MAX"] = _lo
    log0(f"raki_v12:auto_qmax={_lo} est={_tsz + _code_bytes}")
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")''',
      "EMA → TTT → auto qmax")

# ============================================================
with open("train_gpt.py", "w") as f:
    f.write(code)

print(f"\nRaki V12 ({changes} patches) — Per-Sample SLOT-24 + Pre-quant TTT")
print(f"  SLOT: hidden delta [B,1,512] + logit bias [B,1,1024], 24 AdamW/window")
print(f"  TTT: 6ep pre-quant, freeze first 2 blocks, cosine LR")
print(f"  Stack: Turbo-Muon + XSA-ALL + LeakyReLU² + LN_Scale + MLP3x")
print(f"         + Late QAT + BigramHash + Markov + EMA + GPTQ + Brotli")
print(f"  8xH100: torchrun --standalone --nproc_per_node=8 train_gpt.py")
print(f"  5090 test: SLOT_ENABLED=0 ITERATIONS=1500 ... python3 train_gpt.py")
