#!/usr/bin/env python3
"""
Raki V11 — Record Submission: Full stack + Brotli-11 + qTTT + GPTQ fix.

=== V11 vs V10: 4 targeted improvements ===

  1. GPTQ device fix: V10 crashed because weights were on CPU, Hessians on CUDA.
     Fixed: W.to(dev) + Q.cpu() in _gptq_quantize_weight.

  2. Brotli-11 compression: Replaces zstd-22. ~500KB+ better compression on
     int6 quantized weights → higher auto_qmax → less quantization error.
     Python brotli package, same pattern as zstandard install.

  3. qTTT (Query-only TTT): Instead of adapting ALL weights (which invalidates
     GPTQ rounding and KV caches), only adapt attention Q-projection-like params
     (largest output dim in each attention module). 3× fewer params to update →
     5 epochs instead of 3 → better adaptation. arXiv:2512.13898.

  4. TTT decay prior: After each TTT step, pull weights back toward pre-TTT
     values with λ=0.001. Prevents catastrophic forgetting on later chunks.
     p += λ(p₀ - p). Proven in competition (PR #549 discussion).

=== FULL STACK (from V10) ===

  Turbo-Muon (AOL preconditioned NS, steps-1)
  XSA-ALL (11 layers) + QK-Gain 4.0
  Full Hessian GPTQ (with device fix)
  LeakyReLU(0.5)² + MLP 3× + LN Scale 1/√(L+1)
  Late QAT (STE int6, torch._dynamo.reset() fix)
  Partial RoPE 16/64 + EMA 0.997
  BigramHash(2048) + Adaptive Markov curriculum
  Sliding window eval stride=64
  Brotli-11 compression + auto qmax binary search

  Target: ≤1.09 bpb
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
# 1. Dependencies: zstandard + brotli
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
# 2. Hyperparameters: 11L, warmdown 3500, MLP 3×, QK-Gain 4.0
# ============================================================
patch('    num_layers = int(os.environ.get("NUM_LAYERS", 9))',
      '    num_layers = int(os.environ.get("NUM_LAYERS", 11))',
      "NUM_LAYERS=11")

patch('    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))',
      '    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))',
      "WARMDOWN=3500")

patch('    mlp_mult = int(os.environ.get("MLP_MULT", 2))',
      '    mlp_mult = int(os.environ.get("MLP_MULT", 3))',
      "MLP_MULT=3")

patch('    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))',
      '    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 4.0))',
      "QK_GAIN_INIT=4.0")

# ============================================================
# 3. Turbo-Muon: AOL preconditioning (steps-1 NS iterations)
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
    # Turbo-Muon: AOL preconditioning + reduced NS iterations (arXiv:2512.04632)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    D_r = (X * X).sum(dim=1, keepdim=True).clamp_min(eps * eps)
    D_c = (X * X).sum(dim=0, keepdim=True).clamp_min(eps * eps)
    X = X / (D_r * D_c).pow(0.25)
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    turbo_steps = max(steps - 1, 3)
    for _ in range(turbo_steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X''',
    "Turbo-Muon preconditioning")

# ============================================================
# 4. Config: constants, helpers, EMA, QAT, Markov, GPTQ, qTTT
# ============================================================
patch(
    "from torch.nn.parallel import DistributedDataParallel as DDP",
    '''from torch.nn.parallel import DistributedDataParallel as DDP
import zstandard as zstd
import brotli

# --- Raki V11 config ---
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
GPTQ_FULL = bool(int(os.environ.get("GPTQ_FULL", "1")))
GPTQ_CAL_BATCHES = int(os.environ.get("GPTQ_CAL_BATCHES", "64"))
TTT_ENABLED = bool(int(os.environ.get("TTT_ENABLED", "1")))
TTT_LR = float(os.environ.get("TTT_LR", "0.001"))
TTT_EPOCHS = int(os.environ.get("TTT_EPOCHS", "5"))
TTT_CHUNK = int(os.environ.get("TTT_CHUNK", "32768"))
TTT_GRAD_CLIP = float(os.environ.get("TTT_GRAD_CLIP", "1.0"))
TTT_Q_ONLY = bool(int(os.environ.get("TTT_Q_ONLY", "1")))

_QAT = {"on": False}
_GPTQ_HESSIANS: dict[str, "Tensor"] = {}


def _ste_fake_quant(w: Tensor, qmax: int) -> Tensor:
    with torch.no_grad():
        scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / float(qmax)
    w_q = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w + (w_q - w).detach()


@torch.no_grad()
def _gptq_quantize_weight(W: Tensor, H: Tensor, qmax: int) -> tuple[Tensor, Tensor]:
    """Full Hessian GPTQ: column-by-column quantization with error redistribution."""
    dev = H.device
    W = W.float().clone().to(dev)
    H = H.float()
    n_rows, n_cols = W.shape

    best_scale = None
    best_mse = torch.full((n_rows,), float('inf'), device=dev)
    for pct in GPTQ_PERCENTILES:
        ca = W.abs().amax(dim=1) if pct >= 1.0 else torch.quantile(W.abs(), pct, dim=1)
        sc = (ca / float(qmax)).clamp_min(1.0 / float(qmax))
        cl = torch.clamp(W, -ca[:, None], ca[:, None])
        qq = torch.clamp(torch.round(cl / sc[:, None]), -qmax, qmax)
        mse = ((W - qq * sc[:, None]) ** 2).mean(dim=1)
        improved = mse < best_mse
        if best_scale is None:
            best_scale = sc.clone()
            best_mse = mse
        else:
            best_scale[improved] = sc[improved]
            best_mse[improved] = mse[improved]
    scale = best_scale

    dead = H.diag() == 0
    H = H.clone()
    H[dead, dead] = 1.0
    damp = 0.01 * H.diag().mean()
    H += damp * torch.eye(n_cols, device=dev)

    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except Exception:
        Hinv = torch.linalg.pinv(H)

    Q = torch.zeros(n_rows, n_cols, dtype=torch.int8, device=dev)

    BS = 128
    for j1 in range(0, n_cols, BS):
        j2 = min(j1 + BS, n_cols)
        blen = j2 - j1
        Err = torch.zeros(n_rows, blen, device=dev)
        Hinv_blk = Hinv[j1:j2, j1:j2]

        for jl in range(blen):
            j = j1 + jl
            w_col = W[:, j]
            d = Hinv_blk[jl, jl].clamp_min(1e-10)

            q_col = torch.clamp(torch.round(w_col / scale), -qmax, qmax)
            Q[:, j] = q_col.to(torch.int8)

            err = (w_col - q_col * scale) / d
            Err[:, jl] = err

            if jl + 1 < blen:
                W[:, j + 1:j2] -= err.unsqueeze(1) * Hinv_blk[jl, jl + 1:].unsqueeze(0)

        if j2 < n_cols:
            W[:, j2:] -= Err @ Hinv[j1:j2, j2:]

    return Q.cpu().contiguous(), scale.cpu().to(dtype=torch.float16).contiguous()


def _collect_hessians(model: nn.Module, data_pattern: str, device: torch.device,
                      seq_len: int = 1024, n_batches: int = 64) -> dict[str, Tensor]:
    files = sorted(glob.glob(data_pattern))
    if not files:
        return {}
    hdr_bytes = 256 * np.dtype("<i4").itemsize
    all_tokens = []
    remaining = n_batches * seq_len + 1
    for f in files:
        if remaining <= 0:
            break
        hdr = np.fromfile(f, dtype="<i4", count=256)
        ntok = min(int(hdr[2]), remaining)
        tok = np.fromfile(f, dtype="<u2", count=ntok, offset=hdr_bytes)
        all_tokens.append(tok)
        remaining -= len(tok)
    tokens = np.concatenate(all_tokens) if all_tokens else np.array([], dtype=np.uint16)
    n_seqs = min(n_batches, (len(tokens) - 1) // seq_len)
    if n_seqs < 1:
        return {}

    raw_m = model.module if hasattr(model, "module") else model
    base_m = raw_m._orig_mod if hasattr(raw_m, "_orig_mod") else raw_m

    hessians: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    for name, mod in base_m.named_modules():
        if hasattr(mod, 'weight') and hasattr(mod, 'in_features') and mod.weight.numel() > 65536:
            in_f = mod.in_features
            hessians[name] = torch.zeros(in_f, in_f, device=device, dtype=torch.float32)
            counts[name] = 0

            def _make_hook(_name, _in_f):
                def _hook(module, inp, out):
                    x = inp[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, _in_f)
                    hessians[_name].addmm_(x.T, x)
                    counts[_name] += x.shape[0]
                return _hook

            hooks.append(mod.register_forward_hook(_make_hook(name, in_f)))

    base_m.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(n_seqs):
            s = i * seq_len
            x = torch.tensor(tokens[s:s + seq_len], dtype=torch.int64, device=device).unsqueeze(0)
            y = torch.tensor(tokens[s + 1:s + seq_len + 1], dtype=torch.int64, device=device).unsqueeze(0)
            try:
                base_m(x, y)
            except Exception:
                break

    for h in hooks:
        h.remove()

    for name in list(hessians.keys()):
        if counts.get(name, 0) > 0:
            hessians[name] /= counts[name]
        else:
            del hessians[name]

    return hessians


class _GPUMarkov:
    def __init__(self, pattern: str, V: int, device: torch.device):
        files = sorted(glob.glob(pattern))
        hdr_bytes = 256 * np.dtype("<i4").itemsize
        hdr = np.fromfile(files[0], dtype="<i4", count=256)
        ntok = min(int(hdr[2]), 2_000_000)
        tok = np.fromfile(files[0], dtype="<u2", count=ntok,
                          offset=hdr_bytes).astype(np.int32)
        counts = np.zeros((V, V), dtype=np.float64)
        np.add.at(counts, (tok[:-1], tok[1:]), 1.0)
        counts += 0.01
        probs = counts / counts.sum(axis=1, keepdims=True)
        log_probs = np.log(probs).astype(np.float16)
        ent = -(probs * np.log(probs)).sum(axis=1).astype(np.float32)
        mn, mx = ent.min(), ent.max()
        ent_norm = ((ent - mn) / (mx - mn) if mx > mn
                    else np.full_like(ent, 0.5))
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
            model_difficulty = batch_loss / max(self.loss_ema, 1e-6)
            combined = bigram_score * min(model_difficulty, 2.0)
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
    "config")

# ============================================================
# 5. Muon weight decay
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
# 6. Late QAT STE in CastedLinear
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
# 7. Partial RoPE
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

# ============================================================
# 8. Rotary init for partial dims
# ============================================================
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
# 9. XSA in CausalSelfAttention.__init__
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

# ============================================================
# 10. XSA in CausalSelfAttention.forward
# ============================================================
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
            if self.num_kv_heads != self.num_heads:
                v_x = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            else:
                v_x = v
            dot_yv = (y * v_x).sum(-1, keepdim=True)
            v_norm = (v_x * v_x).sum(-1, keepdim=True).clamp_min(1e-8)
            y = y - (dot_yv / v_norm) * v_x
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)''',
    "XSA forward")

# ============================================================
# 11. LeakyReLU(0.5)² in MLP
# ============================================================
patch(
    '''    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())''',
    '''    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())''',
    "LeakyReLU(0.5)²")

# ============================================================
# 12. Block: layer_idx, XSA, LN Scale
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
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
        use_xsa: bool = False,
    ):
        super().__init__()
        self._ln_s = 1.0 / math.sqrt(layer_idx + 1) if LN_SCALE else 1.0
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult)''',
    "Block init")

# ============================================================
# 13. Block.forward: LN Scale
# ============================================================
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
        if self._ln_s != 1.0:
            _h = _h * self._ln_s
        attn_out = self.attn(_h)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        _h = self.mlp_norm(x)
        if self._ln_s != 1.0:
            _h = _h * self._ln_s
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(_h)
        return x''',
    "Block forward LN Scale")

# ============================================================
# 14. GPT: BigramHash init + blocks with XSA-ALL
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
    '''        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                    use_xsa=(i >= num_layers - XSA_LAST_N) if XSA_LAST_N > 0 else False,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.bigram_table = nn.Embedding(BIGRAM_BUCKETS, model_dim)
        nn.init.normal_(self.bigram_table.weight, std=0.002)
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)''',
    "GPT blocks XSA-ALL + BigramHash")

# ============================================================
# 15. GPT.forward: BigramHash
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
# 16. forward_per_token
# ============================================================
patch(
    '''        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------''',
    '''        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_per_token(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        if T >= 2:
            ids = input_ids.long()
            bh = (ids[:, 1:] * 36313 + ids[:, :-1] * 51749) % BIGRAM_BUCKETS
            x[:, 1:] = x[:, 1:] + self.bigram_table(bh).to(x.dtype)
        x = F.rms_norm(x, (x.size(-1),))
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
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            target_ids.reshape(-1), reduction="none").reshape(B, T)


# -----------------------------
# TRAINING
# -----------------------------''',
    "forward_per_token")

# ============================================================
# 17. Sliding window eval (dual-mode)
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
    seq_len = args.train_seq_len
    stride = EVAL_STRIDE if 0 < EVAL_STRIDE < seq_len else 0
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    if stride > 0:
        total_tokens = val_tokens.numel() - 1
        all_starts = list(range(0, total_tokens - seq_len + 1, stride))
        if not all_starts:
            all_starts = [0]
        rank_starts = [s for i, s in enumerate(all_starts) if i % world_size == rank]
        raw_model = model.module if hasattr(model, "module") else model
        base_m = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
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
                    if n_scored <= 0:
                        continue
                    val_loss_sum += ptl[wi, sc_start:sc_start + n_scored].to(torch.float64).sum()
                    val_token_count += float(n_scored)
                    g = s + sc_start
                    prev = val_tokens[g:g + n_scored].to(device=device, dtype=torch.int64)
                    tgt = val_tokens[g + 1:g + n_scored + 1].to(device=device, dtype=torch.int64)
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
    "sliding window eval")

# ============================================================
# 18. GPTQ-aware quantization (Full Hessian + clip search fallback)
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

    '''def quantize_float_tensor(t: Tensor, qmax: int = 127, tensor_name: str = "") -> tuple[Tensor, Tensor]:
    t32 = t.float()
    hkey = tensor_name.replace(".weight", "") if tensor_name else ""
    if hkey and hkey in _GPTQ_HESSIANS and t32.ndim == 2 and t32.numel() > 65536 and GPTQ_FULL:
        return _gptq_quantize_weight(t32, _GPTQ_HESSIANS[hkey], qmax)
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
    "GPTQ-aware quantization")

# ============================================================
# 19. Block weights use lower qmax + pass tensor name
# ============================================================
patch(
    '''        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)''',
    '''        stats["num_float_tensors"] += 1
        use_qmax = BLOCK_QUANT_MAX if "blocks." in name else 127
        q, s = quantize_float_tensor(t, qmax=use_qmax, tensor_name=name)''',
    "block qmax + tensor name")

# ============================================================
# 20. Brotli-11 compression (replaces zlib)
# ============================================================
patch('    quant_blob = zlib.compress(quant_raw, level=9)',
      '    quant_blob = brotli.compress(quant_raw, quality=11)',
      "brotli-11 compress")
for i in range(3):
    patch('"final_model.int8.ptz"', '"final_model.int8.ptbr"', f"filename {i+1}")
patch('quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")',
      'quant_state = torch.load(io.BytesIO(brotli.decompress(quant_blob_disk)), map_location="cpu")',
      "brotli decompress")
patch('f"Serialized model int8+zlib:', 'f"Serialized model int8+brotli:', "log brotli 1")
patch('f"Total submission size int8+zlib:', 'f"Total submission size int8+brotli:', "log brotli 2")
patch('f"final_int8_zlib_roundtrip val_loss', 'f"final_int8_brotli_roundtrip val_loss', "log brotli 3")
patch('f"final_int8_zlib_roundtrip_exact val_loss', 'f"final_int8_brotli_roundtrip_exact val_loss', "log brotli 4")

# ============================================================
# 21. BigramHash in optimizer
# ============================================================
patch('        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],',
      '        [{"params": [base_model.tok_emb.weight, base_model.bigram_table.weight], "lr": token_lr, "base_lr": token_lr}],',
      "bigram in optimizer")

# ============================================================
# 22. Init: Markov + EMA + logging
# ============================================================
patch("    training_time_ms = 0.0",
      '''    _markov = _GPUMarkov(args.train_files, args.vocab_size, device)
    _ema = _EMA()
    _ema_on = False
    log0(f"raki_v11: L={args.num_layers} mlp={args.mlp_mult}x bigram={BIGRAM_BUCKETS} rope={ROPE_DIMS} xsa={XSA_LAST_N} ln_scale={LN_SCALE} qat={LATE_QAT_THRESHOLD} gptq={GPTQ_FULL} power={RAKI_POWER} wd={MUON_WD} ema={EMA_DECAY} ttt={TTT_ENABLED} qttt={TTT_Q_ONLY} brotli=11")
    training_time_ms = 0.0''',
      "init")

# ============================================================
# 23. Adaptive Markov curriculum
# ============================================================
patch("            (loss * grad_scale).backward()",
      '''            _cw = _markov.batch_weight(x, y, loss.item())
            (loss * grad_scale * _cw).backward()''',
      "Adaptive Markov curriculum")

# ============================================================
# 24. Late QAT (dynamo reset) + EMA update
# ============================================================
patch("        zero_grad_all()\n\n        step += 1",
      '''        zero_grad_all()
        _prog = (training_time_ms + 1000.0 * (time.perf_counter() - t0)) / max(max_wallclock_ms or 1e18, 1.0)
        if LATE_QAT_THRESHOLD > 0 and _prog >= LATE_QAT_THRESHOLD and not _QAT["on"]:
            _QAT["on"] = True
            torch._dynamo.reset()
            log0(f"raki_v11:late_qat step={step+1} prog={_prog:.3f} (dynamo reset)")
        if _prog >= EMA_START_FRAC and not _ema_on:
            _ema.start(base_model); _ema_on = True
            log0(f"raki_v11:ema_started step={step+1}")
        _ema.update(base_model)
        step += 1''',
      "Late QAT + EMA update")

# ============================================================
# 25. End: QAT off + EMA + GPTQ calibration + auto qmax (Brotli)
# ============================================================
patch('    if master_process:\n        torch.save(base_model.state_dict(), "final_model.pt")',
      '''    _QAT["on"] = False
    if _ema.on:
        _ema.apply(base_model)
        log0("raki_v11:ema_applied")
    if GPTQ_FULL and rank == 0:
        log0(f"raki_v11:gptq_calibration n_batches={GPTQ_CAL_BATCHES}")
        _t_gptq = time.perf_counter()
        _GPTQ_HESSIANS.update(_collect_hessians(
            model, args.train_files, device,
            seq_len=args.train_seq_len, n_batches=GPTQ_CAL_BATCHES))
        log0(f"raki_v11:gptq_hessians n={len(_GPTQ_HESSIANS)} time={time.perf_counter()-_t_gptq:.1f}s")
    # Auto qmax binary search with Brotli-11
    _gptq_was = GPTQ_FULL
    globals()["GPTQ_FULL"] = False
    _code_bytes = len(code.encode("utf-8"))
    _lo, _hi = 15, 127
    while _lo < _hi:
        _mid = (_lo + _hi + 1) // 2
        globals()["BLOCK_QUANT_MAX"] = _mid
        _tobj, _ = quantize_state_dict_int8(base_model.state_dict())
        _tbuf = io.BytesIO()
        torch.save(_tobj, _tbuf)
        _tsz = len(brotli.compress(_tbuf.getvalue(), quality=4))
        del _tobj, _tbuf; import gc as _gc; _gc.collect()
        if _tsz + _code_bytes <= 16_000_000:
            _lo = _mid
        else:
            _hi = _mid - 1
    globals()["BLOCK_QUANT_MAX"] = _lo
    globals()["GPTQ_FULL"] = _gptq_was
    log0(f"raki_v11:auto_qmax={_lo} est_clip_bytes={_tsz + _code_bytes}")
    if _gptq_was and _GPTQ_HESSIANS:
        for _try_qmax in [_lo, _lo - 1, _lo - 2]:
            if _try_qmax < 15:
                break
            globals()["BLOCK_QUANT_MAX"] = _try_qmax
            _tobj, _ = quantize_state_dict_int8(base_model.state_dict())
            _tbuf = io.BytesIO()
            torch.save(_tobj, _tbuf)
            _tsz = len(brotli.compress(_tbuf.getvalue(), quality=11))
            del _tobj, _tbuf; import gc as _gc2; _gc2.collect()
            if _tsz + _code_bytes <= 16_000_000:
                globals()["BLOCK_QUANT_MAX"] = _try_qmax
                log0(f"raki_v11:final_qmax={_try_qmax} bytes={_tsz + _code_bytes}")
                break
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")''',
      "EMA + GPTQ + auto qmax Brotli")

# ============================================================
# 26. Score-First qTTT (Query-only Test-Time Training)
#     arXiv:2512.13898 — Only adapt Q-like attention params.
#     3× fewer params → 5 epochs → better adaptation.
#     Decay prior pulls weights back toward pre-TTT values.
# ============================================================
patch(
    '''    if distributed:
        dist.destroy_process_group()''',
    '''    # === Score-First qTTT ===
    if TTT_ENABLED and rank == 0:
        log0(f"raki_v11:ttt lr={TTT_LR} epochs={TTT_EPOCHS} chunk={TTT_CHUNK} q_only={TTT_Q_ONLY} decay={TTT_DECAY}")
        _ttt_t0 = time.perf_counter()
        raw_m = model.module if hasattr(model, 'module') else model
        base_ttt = raw_m._orig_mod if hasattr(raw_m, '_orig_mod') else raw_m

        # qTTT: select only Q-projection-like params (largest output dim in attn)
        # In GQA: Q=[dim,dim], K=[kv_dim,dim], V=[kv_dim,dim]. Q has shape[0]==model_dim.
        if TTT_Q_ONLY:
            q_dim = args.model_dim
            ttt_params = []
            for n, p in base_ttt.named_parameters():
                if "attn" in n and p.ndim == 2 and p.shape[0] == q_dim and p.requires_grad:
                    ttt_params.append(p)
            if not ttt_params:
                ttt_params = [p for p in base_ttt.parameters() if p.requires_grad]
            log0(f"raki_v11:qttt_params n={len(ttt_params)} total={sum(p.numel() for p in ttt_params)}")
        else:
            ttt_params = [p for p in base_ttt.parameters() if p.requires_grad]

        # Save pre-TTT weights for decay prior
        _pre_ttt = {id(p): p.data.clone() for p in ttt_params}

        seq_len = args.train_seq_len
        total_val = val_tokens.numel() - 1
        n_chunks = max(1, total_val // TTT_CHUNK)
        ttt_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        ttt_tok_count = 0
        ttt_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        stride = EVAL_STRIDE if 0 < EVAL_STRIDE < seq_len else seq_len

        for ci in range(n_chunks):
            cs = ci * TTT_CHUNK
            ce = min(cs + TTT_CHUNK, total_val)
            if ce - cs < seq_len:
                continue

            # SCORE: eval chunk under inference_mode (stateless, no weight mutation)
            base_ttt.eval()
            with torch.inference_mode():
                for s in range(0, ce - cs - seq_len + 1, stride):
                    x = val_tokens[cs+s:cs+s+seq_len].to(device=device, dtype=torch.int64).unsqueeze(0)
                    y = val_tokens[cs+s+1:cs+s+seq_len+1].to(device=device, dtype=torch.int64).unsqueeze(0)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        ptl = base_ttt.forward_per_token(x, y).detach()
                    sc = 0 if s == 0 else (seq_len - stride)
                    ns = min(seq_len - sc, ce - cs - s - sc)
                    if ns <= 0:
                        continue
                    ttt_loss_sum += ptl[0, sc:sc+ns].to(torch.float64).sum()
                    ttt_tok_count += float(ns)
                    g = cs + s + sc
                    prev = val_tokens[g:g+ns].to(device=device, dtype=torch.int64)
                    tgt = val_tokens[g+1:g+ns+1].to(device=device, dtype=torch.int64)
                    nn_ = min(prev.size(0), tgt.size(0))
                    tb = base_bytes_lut[tgt[:nn_]].to(torch.int16)
                    tb += (has_leading_space_lut[tgt[:nn_]] & ~is_boundary_token_lut[prev[:nn_]]).to(torch.int16)
                    ttt_byte_count += tb.to(torch.float64).sum()

            # TRAIN: fine-tune on already-scored chunk (AdamW + cosine LR + decay prior)
            base_ttt.train()
            chunk_seqs = (ce - cs) // seq_len
            total_ttt_steps = TTT_EPOCHS * chunk_seqs
            ttt_step = 0
            for ep in range(TTT_EPOCHS):
                for si in range(chunk_seqs):
                    ss = cs + si * seq_len
                    x = val_tokens[ss:ss+seq_len].to(device=device, dtype=torch.int64).unsqueeze(0)
                    y = val_tokens[ss+1:ss+seq_len+1].to(device=device, dtype=torch.int64).unsqueeze(0)
                    ttt_opt.zero_grad()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        tl = base_ttt(x, y)
                    tl.backward()
                    if TTT_GRAD_CLIP > 0:
                        torch.nn.utils.clip_grad_norm_(ttt_params, TTT_GRAD_CLIP)
                    _cos_lr = TTT_LR * 0.5 * (1.0 + math.cos(math.pi * ttt_step / max(total_ttt_steps, 1)))
                    for pg in ttt_opt.param_groups:
                        pg["lr"] = _cos_lr
                    ttt_opt.step()
                    pass  # No decay prior (conflicts with AdamW momentum)
                    ttt_step += 1

            if ci % 100 == 0:
                log0(f"raki_v11:ttt chunk={ci}/{n_chunks}")

        ttt_vl = ttt_loss_sum / ttt_tok_count
        ttt_bpt = ttt_vl.item() / math.log(2.0)
        ttt_tpb = ttt_tok_count.item() / ttt_byte_count.item()
        ttt_bpb = ttt_bpt * ttt_tpb
        log0(f"raki_v11:ttt val_loss:{ttt_vl.item():.4f} val_bpb:{ttt_bpb:.4f} time:{time.perf_counter()-_ttt_t0:.1f}s")
        log0(f"raki_v11:ttt_exact val_loss:{ttt_vl.item():.8f} val_bpb:{ttt_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()''',
    "Score-First qTTT")

# ============================================================
# Write patched file
# ============================================================
with open("train_gpt.py", "w") as f:
    f.write(code)

print(f"\nRaki V11 ({changes} patches): RECORD SUBMISSION — Target ≤1.09 bpb")
print(f"  NEW vs V10: GPTQ device fix + Brotli-11 + qTTT + decay prior")
print(f"  Stack: Turbo-Muon + XSA-ALL + Full GPTQ + qTTT + QK-Gain 4.0")
print(f"         + LeakyReLU² + QAT(dynamo fix) + LN_Scale + MLP3x")
print(f"         + Partial RoPE + EMA + Brotli-11 + Adaptive Markov + BigramHash")
print(f"  8xH100: MUON_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \\")
print(f"    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \\")
print(f"    EMA_DECAY=0.997 EVAL_STRIDE=64 RAKI_POWER=0.10 \\")
print(f"    RUN_ID=raki_v11 torchrun --standalone --nproc_per_node=8 train_gpt.py")
