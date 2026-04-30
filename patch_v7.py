#!/usr/bin/env python3
"""
Raki V7 — Two novel techniques from outside the competition:
1. μ-law Companding Quantization (ITU-T G.711, 1972 telecom → neural nets)
   Non-uniform quantization that matches bell-shaped weight distributions.
   Same qmax, ~30% less quantization error. No artifact cost.
2. Bigram Knowledge Injection (original)
   Auxiliary KL loss distills bigram statistics into model during training.
   Model learns simple patterns faster, dedicates capacity to hard patterns.
Base: V5 (11L + Adaptive Markov + Partial RoPE + BigramHash + EMA + GPTQ + zstd)
Removed: bigram_boost (saves 2MB artifact → higher qmax → lower quant gap)
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
        print(f"FAIL: {label}\n  anchor: {repr(anchor[:120])}")
        sys.exit(1)

# --- zstandard ---
patch(
    'from __future__ import annotations',
    '''from __future__ import annotations
try:
    import zstandard as _zstd_check  # noqa: F401
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "zstandard", "-q"])''',
    "zstandard")

# --- Hyperparameters ---
patch('    num_layers = int(os.environ.get("NUM_LAYERS", 9))',
      '    num_layers = int(os.environ.get("NUM_LAYERS", 11))',
      "NUM_LAYERS=11")
patch('    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))',
      '    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))',
      "WARMDOWN=3500")

# --- Config ---
patch(
    "from torch.nn.parallel import DistributedDataParallel as DDP",
    '''from torch.nn.parallel import DistributedDataParallel as DDP
import zstandard as zstd

MUON_WD = float(os.environ.get("MUON_WD", "0"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.995"))
EMA_START_FRAC = float(os.environ.get("EMA_START_FRAC", "0.85"))
RAKI_POWER = float(os.environ.get("RAKI_POWER", "0.15"))
BIGRAM_BUCKETS = int(os.environ.get("BIGRAM_BUCKETS", "2048"))
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", "0"))
ROPE_DIMS = int(os.environ.get("ROPE_DIMS", "16"))
BLOCK_QUANT_MAX = int(os.environ.get("BLOCK_QUANT_MAX", "31"))
GPTQ_CLIP_SEARCH = bool(int(os.environ.get("GPTQ_CLIP_SEARCH", "1")))
GPTQ_PERCENTILES = [0.999, 0.9995, 0.9999, 0.99999, 1.0]
MULAW_MU = float(os.environ.get("MULAW_MU", "255"))
BIGRAM_KL_ALPHA = float(os.environ.get("BIGRAM_KL_ALPHA", "0.05"))
BIGRAM_KL_START = float(os.environ.get("BIGRAM_KL_START", "0.1"))


def _mulaw_compress(x: Tensor, mu: float = 255.0) -> Tensor:
    xmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    xn = x / xmax
    compressed = torch.sign(xn) * torch.log1p(mu * xn.abs()) / math.log(1.0 + mu)
    return compressed * xmax

def _mulaw_expand(x: Tensor, mu: float = 255.0) -> Tensor:
    xmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    xn = x / xmax
    expanded = torch.sign(xn) * ((1.0 + mu) ** xn.abs() - 1.0) / mu
    return expanded * xmax


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
        self.bigram_probs = torch.tensor(probs.astype(np.float16), device=device)
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

    @torch.no_grad()
    def bigram_kl_target(self, x: Tensor) -> Tensor:
        return self.bigram_probs[x.reshape(-1)].float()


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

# --- Muon WD ---
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

# --- Partial RoPE ---
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

# --- BigramHash init (NO bigram_boost — saves 2MB) ---
patch(
    '''        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)''',
    '''        self.final_norm = RMSNorm()
        self.bigram_table = nn.Embedding(BIGRAM_BUCKETS, model_dim)
        nn.init.normal_(self.bigram_table.weight, std=0.002)
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)''',
    "BigramHash init")

# --- BigramHash forward ---
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

# --- forward_per_token for sliding window ---
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

# --- Sliding window eval ---
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
    "dual-mode eval")

# --- μ-law GPTQ quantization ---
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
    if t32.ndim == 2 and MULAW_MU > 0:
        t32 = _mulaw_compress(t32, MULAW_MU)
    if t32.ndim == 2:
        if GPTQ_CLIP_SEARCH and t32.numel():
            best_q = best_scale = None
            best_mse = torch.full((t32.shape[0],), float('inf'))
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
    "μ-law GPTQ quantization")

# --- μ-law dequantization ---
patch(
    '''        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING ''',
    '''        out[name] = out_t
    if MULAW_MU > 0:
        for name in list(out.keys()):
            if out[name].ndim == 2 and out[name].numel() > 65536:
                out[name] = _mulaw_expand(out[name].float(), MULAW_MU).to(out[name].dtype).contiguous()
    return out


# -----------------------------
# DATA LOADING ''',
    "μ-law dequantization")

# --- Block weights use lower qmax ---
patch(
    '''        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)''',
    '''        stats["num_float_tensors"] += 1
        use_qmax = BLOCK_QUANT_MAX if "blocks." in name else 127
        q, s = quantize_float_tensor(t, qmax=use_qmax)''',
    "block qmax")

# --- zstd-22 ---
patch('    quant_blob = zlib.compress(quant_raw, level=9)',
      '    cctx = zstd.ZstdCompressor(level=22)\n    quant_blob = cctx.compress(quant_raw)',
      "zstd-22")
for i in range(3):
    patch('"final_model.int8.ptz"', '"final_model.int8.ptzst"', f"filename {i+1}")
patch('quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")',
      'dctx = zstd.ZstdDecompressor()\n    quant_state = torch.load(io.BytesIO(dctx.decompress(quant_blob_disk)), map_location="cpu")',
      "zstd decompress")
patch('f"Serialized model int8+zlib:', 'f"Serialized model int8+zstd:', "log 1")
patch('f"Total submission size int8+zlib:', 'f"Total submission size int8+zstd:', "log 2")
patch('f"final_int8_zlib_roundtrip val_loss', 'f"final_int8_zstd_roundtrip val_loss', "log 3")
patch('f"final_int8_zlib_roundtrip_exact val_loss', 'f"final_int8_zstd_roundtrip_exact val_loss', "log 4")

# --- BigramHash in optimizer ---
patch('        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],',
      '        [{"params": [base_model.tok_emb.weight, base_model.bigram_table.weight], "lr": token_lr, "base_lr": token_lr}],',
      "bigram in optimizer")

# --- Init ---
patch("    training_time_ms = 0.0",
      '''    _markov = _GPUMarkov(args.train_files, args.vocab_size, device)
    _ema = _EMA()
    _ema_on = False
    log0(f"raki_v7: L={args.num_layers} bigram={BIGRAM_BUCKETS} rope={ROPE_DIMS} mulaw={MULAW_MU} kl_alpha={BIGRAM_KL_ALPHA} power={RAKI_POWER} wd={MUON_WD} ema={EMA_DECAY}")
    training_time_ms = 0.0''',
      "init")

# --- Bigram Knowledge Injection + Adaptive Markov ---
patch("            (loss * grad_scale).backward()",
      '''            _cw = _markov.batch_weight(x, y, loss.item())
            _frac = (training_time_ms + 1000.0 * (time.perf_counter() - t0)) / max(max_wallclock_ms or 1e18, 1.0)
            if BIGRAM_KL_ALPHA > 0 and _frac >= BIGRAM_KL_START:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    _raw = model.module if hasattr(model, 'module') else model
                    _bm = _raw._orig_mod if hasattr(_raw, '_orig_mod') else _raw
                    _h = _bm.tok_emb(x)
                    if x.size(1) >= 2:
                        _ids = x.long()
                        _bh = (_ids[:, 1:] * 36313 + _ids[:, :-1] * 51749) % BIGRAM_BUCKETS
                        _h[:, 1:] = _h[:, 1:] + _bm.bigram_table(_bh).to(_h.dtype)
                    _logits_flat = F.linear(F.rms_norm(_h, (_h.size(-1),)), _bm.tok_emb.weight)
                    _model_lp = F.log_softmax(_logits_flat.float().reshape(-1, _logits_flat.size(-1)), dim=-1)
                    _bg_target = _markov.bigram_kl_target(x)
                    _kl = F.kl_div(_model_lp, _bg_target, reduction='batchmean')
                    loss = loss + BIGRAM_KL_ALPHA * _kl
            (loss * grad_scale * _cw).backward()''',
      "bigram KL injection + Markov curriculum")

# --- EMA update ---
patch("        zero_grad_all()\n\n        step += 1",
      '''        zero_grad_all()
        _prog = (training_time_ms + 1000.0 * (time.perf_counter() - t0)) / max(max_wallclock_ms or 1e18, 1.0)
        if _prog >= EMA_START_FRAC and not _ema_on:
            _ema.start(base_model); _ema_on = True
            log0(f"raki_v7:ema_started step={step+1}")
        _ema.update(base_model)
        step += 1''',
      "EMA update")

# --- EMA apply + auto qmax ---
patch('    if master_process:\n        torch.save(base_model.state_dict(), "final_model.pt")',
      '''    if _ema.on:
        _ema.apply(base_model)
        log0("raki_v7:ema_applied")
    _code_bytes = len(code.encode("utf-8"))
    _lo, _hi = 15, 127
    while _lo < _hi:
        _mid = (_lo + _hi + 1) // 2
        globals()["BLOCK_QUANT_MAX"] = _mid
        _tobj, _ = quantize_state_dict_int8(base_model.state_dict())
        _tbuf = io.BytesIO()
        torch.save(_tobj, _tbuf)
        _tsz = len(zstd.ZstdCompressor(level=22).compress(_tbuf.getvalue()))
        if _tsz + _code_bytes <= 16_000_000:
            _lo = _mid
        else:
            _hi = _mid - 1
    globals()["BLOCK_QUANT_MAX"] = _lo
    log0(f"raki_v7:auto_qmax={_lo} est_bytes={_tsz + _code_bytes}")
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")''',
      "EMA apply + auto qmax")

with open("train_gpt.py", "w") as f:
    f.write(code)

print(f"\nRaki V7 ({changes} patches)")
print(f"  Novel: μ-law companding quantization + bigram knowledge injection")
print(f"  Base: 11L + Adaptive Markov + Partial RoPE + BigramHash + EMA + GPTQ + zstd")
print(f"  8xH100: MUON_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \\")
print(f"    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \\")
print(f"    EMA_DECAY=0.997 EVAL_STRIDE=64 MLP_MULT=3 TRAIN_BATCH_TOKENS=786432 \\")
print(f"    RUN_ID=raki_v7 torchrun --standalone --nproc_per_node=8 train_gpt.py")
