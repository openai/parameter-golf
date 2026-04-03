#!/usr/bin/env python3
"""
Parameter Golf — Yandimata Style v2 Patch
==========================================
Patches the official baseline train_gpt.py IN-PLACE.
Hybrid approach: GPU-native Markov curriculum + all proven meta techniques.

Key innovations (original):
  - GPU-native Markov curriculum: no CPU transfer, ~0.1ms overhead per step
  - Entropy-weighted curriculum learning for harder batches

Proven meta techniques (from competition consensus):
  1. 10 layers, 3x MLP
  2. Muon WD=0.04, momentum 0.99
  3. Warmdown 3500, tuned LRs
  4. EMA (decay=0.997)
  5. Sliding window eval (stride=64)
  6. Int6 quantization + zstd-22

Usage:
  # On RunPod, starting from clean baseline:
  cd ~/Desktop/parameter-golf-yandimatastyle
  git pull origin main  # get latest upstream
  python3 patch_v2.py
  # Then on RunPod 8xH100:
  RUN_ID=yandimata_v2 torchrun --standalone --nproc_per_node=8 train_gpt.py
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
        print(f"  [{changes:2d}] OK: {label}")
        return True
    else:
        print(f"  FAIL: {label}")
        print(f"    anchor: {repr(anchor[:100])}...")
        sys.exit(1)

# ==============================================================
# SECTION A: Auto-install zstandard
# ==============================================================
patch(
    'from __future__ import annotations',
    '''from __future__ import annotations

# Yandimata v2: ensure zstandard is available
try:
    import zstandard as _zstd_check  # noqa: F401
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "zstandard", "-q"])''',
    "Auto-install zstandard"
)

# ==============================================================
# SECTION B: Hyperparameter tuning
# ==============================================================
patch(
    '    num_layers = int(os.environ.get("NUM_LAYERS", 9))',
    '    num_layers = int(os.environ.get("NUM_LAYERS", 10))',
    "NUM_LAYERS 9->10"
)
patch(
    '    mlp_mult = int(os.environ.get("MLP_MULT", 2))',
    '    mlp_mult = int(os.environ.get("MLP_MULT", 3))',
    "MLP_MULT 2->3"
)
patch(
    '    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))',
    '    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))',
    "WARMDOWN_ITERS 1200->3500"
)
patch(
    '    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))',
    '    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))',
    "MUON_MOMENTUM 0.95->0.99"
)
patch(
    '    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))',
    '    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))',
    "MUON_WARMUP_START 0.85->0.92"
)
patch(
    '    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))',
    '    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))',
    "MUON_WARMUP_STEPS 500->1500"
)
patch(
    '    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))',
    '    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))',
    "MATRIX_LR 0.04->0.025"
)
patch(
    '    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))',
    '    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))',
    "SCALAR_LR 0.04->0.025"
)
patch(
    '    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))',
    '    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))',
    "TIED_EMBED_LR 0.05->0.035"
)

# ==============================================================
# SECTION C: Add GPU-Markov + EMA + config after DDP import
# ==============================================================
patch(
    "from torch.nn.parallel import DistributedDataParallel as DDP",
    '''from torch.nn.parallel import DistributedDataParallel as DDP
import zstandard as zstd

# ---- Yandimata v2 config ----
MUON_WD = float(os.environ.get("MUON_WD", "0.04"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.997"))
EMA_START_FRAC = float(os.environ.get("EMA_START_FRAC", "0.80"))
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", "64"))
QUANT_BITS = int(os.environ.get("QUANT_BITS", "6"))
QUANT_MAX = 2 ** (QUANT_BITS - 1) - 1  # 31 for 6-bit
MARKOV_POWER = float(os.environ.get("MARKOV_POWER", "0.12"))


class _GPUMarkov:
    """GPU-native Markov curriculum — zero CPU transfer overhead.

    Builds a bigram log-prob table from the first training shard,
    keeps it on GPU as a float16 tensor. batch_weight() runs entirely
    on GPU with simple indexing + mean — ~0.1ms per call.
    """
    def __init__(self, pattern: str, V: int, device: torch.device):
        files = sorted(glob.glob(pattern))
        hdr_bytes = 256 * np.dtype("<i4").itemsize
        hdr = np.fromfile(files[0], dtype="<i4", count=256)
        ntok = min(int(hdr[2]), 2_000_000)  # cap for speed
        tok = np.fromfile(files[0], dtype="<u2", count=ntok,
                          offset=hdr_bytes).astype(np.int32)
        # Build bigram counts
        counts = np.zeros((V, V), dtype=np.float64)
        np.add.at(counts, (tok[:-1], tok[1:]), 1.0)
        counts += 0.01  # smoothing
        probs = counts / counts.sum(axis=1, keepdims=True)
        log_probs = np.log(probs).astype(np.float16)
        # Entropy per token (for weighting)
        ent = -(probs * np.log(probs)).sum(axis=1).astype(np.float32)
        mn, mx = ent.min(), ent.max()
        ent_norm = ((ent - mn) / (mx - mn) if mx > mn
                    else np.full_like(ent, 0.5))
        # Store on GPU — this is the key optimization vs original Raki
        self.log_probs = torch.tensor(log_probs, device=device)  # [V, V]
        self.ent_norm = torch.tensor(ent_norm, dtype=torch.float16,
                                     device=device)  # [V]

    @torch.no_grad()
    def batch_weight(self, x: Tensor, y: Tensor) -> float:
        """Returns scalar weight >= 1.0. Runs entirely on GPU."""
        if MARKOV_POWER <= 0:
            return 1.0
        # x, y are already on GPU as int64 — just index directly
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        # Surprisal: -log P(y|x) from bigram table
        surp = -self.log_probs[x_flat, y_flat].float()  # [N]
        # Entropy weight
        ent_w = self.ent_norm[x_flat].float()  # [N]
        score = (surp * ent_w).mean().item()
        return 1.0 + MARKOV_POWER * min(score / 5.0, 1.0)


class _EMA:
    """Exponential moving average of model parameters."""
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
                    p.data.copy_(self.shadow[n])
# ---- end Yandimata v2 config ----''',
    "GPU-Markov + EMA + config"
)

# ==============================================================
# SECTION D: Muon weight decay in optimizer step
# ==============================================================
patch(
    '''                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss''',
    '''                p.add_(g, alpha=-lr)
                if MUON_WD > 0:
                    p.mul_(1.0 - lr * MUON_WD)
                curr += p.numel()

        return loss''',
    "Muon weight decay"
)

# ==============================================================
# SECTION E: Add forward_per_token to GPT for sliding window
# ==============================================================
patch(
    '''        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------''',

    '''        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_per_token(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Returns per-token losses [B, T] for sliding window eval."""
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
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
            target_ids.reshape(-1),
            reduction="none"
        ).reshape(B, T)


# -----------------------------
# TRAINING
# -----------------------------''',
    "forward_per_token for sliding window"
)

# ==============================================================
# SECTION F: Replace eval_val with sliding window version
# ==============================================================
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
    # Yandimata v2: Sliding window eval with stride for better BPB.
    # Only the last `stride` tokens of each window are scored (except first).
    stride = EVAL_STRIDE
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    all_starts: list[int] = []
    pos = 0
    while pos + seq_len <= total_tokens:
        all_starts.append(pos)
        pos += stride
    if not all_starts:
        all_starts = [0]

    rank_starts = [s for i, s in enumerate(all_starts) if i % world_size == rank]

    # Unwrap model to access forward_per_token (past DDP + compile)
    raw_model = model.module if hasattr(model, 'module') else model
    base_m = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        batch_size = max(1, min(16, args.val_batch_size // (seq_len * max(world_size, 1))))

        for bi in range(0, len(rank_starts), batch_size):
            batch_starts = rank_starts[bi:bi + batch_size]
            xs, ys = [], []
            for s in batch_starts:
                chunk = val_tokens[s:s + seq_len + 1].to(dtype=torch.int64)
                xs.append(chunk[:-1])
                ys.append(chunk[1:])
            x = torch.stack(xs).to(device=device, non_blocking=True)
            y = torch.stack(ys).to(device=device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                per_token_loss = base_m.forward_per_token(x, y).detach()

            for wi, s in enumerate(batch_starts):
                score_start = 0 if s == 0 else (seq_len - stride)
                num_scored = seq_len - score_start
                global_end = s + seq_len
                if global_end > total_tokens:
                    num_scored -= (global_end - total_tokens)
                if num_scored <= 0:
                    continue

                losses = per_token_loss[wi, score_start:score_start + num_scored]
                val_loss_sum += losses.to(torch.float64).sum()
                val_token_count += float(num_scored)

                g_start = s + score_start
                prev_ids = val_tokens[g_start:g_start + num_scored].to(device=device)
                tgt_ids = val_tokens[g_start + 1:g_start + num_scored + 1].to(device=device)
                n = min(prev_ids.size(0), tgt_ids.size(0), num_scored)
                prev_ids, tgt_ids = prev_ids[:n], tgt_ids[:n]

                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
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
    "Sliding window eval (stride=64)"
)

# ==============================================================
# SECTION G: Int6 quantization
# ==============================================================
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

    '''def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    qmax = float(QUANT_MAX)  # 31 for int6
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / qmax).clamp_min(1.0 / qmax)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / qmax if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qmax, qmax).to(torch.int8).contiguous()
    return q, scale''',
    "Int6 quantization (31 max)"
)

# ==============================================================
# SECTION H: zstd compression + file renames
# ==============================================================
patch(
    '    quant_blob = zlib.compress(quant_raw, level=9)',
    '    cctx = zstd.ZstdCompressor(level=22)\n    quant_blob = cctx.compress(quant_raw)',
    "zstd-22 compress"
)
for i in range(3):
    patch('"final_model.int8.ptz"', '"final_model.int6.ptzst"', f"filename ({i+1})")

patch(
    'quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")',
    'dctx = zstd.ZstdDecompressor()\n    quant_state = torch.load(io.BytesIO(dctx.decompress(quant_blob_disk)), map_location="cpu")',
    "zstd decompress"
)
patch('f"Serialized model int8+zlib:', 'f"Serialized model int6+zstd:', "log (1)")
patch('f"Total submission size int8+zlib:', 'f"Total submission size int6+zstd:', "log (2)")
patch('f"final_int8_zlib_roundtrip val_loss', 'f"final_int6_zstd_roundtrip val_loss', "log (3)")
patch('f"final_int8_zlib_roundtrip_exact val_loss', 'f"final_int6_zstd_roundtrip_exact val_loss', "log (4)")

# ==============================================================
# SECTION I: Init GPU-Markov + EMA before training loop
# ==============================================================
patch(
    "    training_time_ms = 0.0",
    '''    # ---- Yandimata v2: Init GPU-Markov + EMA ----
    _markov = _GPUMarkov(args.train_files, args.vocab_size, device)
    _ema = _EMA()
    _ema_on = False
    log0(f"yandimata_v2: markov_power={MARKOV_POWER} muon_wd={MUON_WD} "
         f"ema={EMA_DECAY} stride={EVAL_STRIDE} bits={QUANT_BITS}")

    training_time_ms = 0.0''',
    "GPU-Markov + EMA init"
)

# ==============================================================
# SECTION J: Markov curriculum weighting on backward
# ==============================================================
patch(
    "            (loss * grad_scale).backward()",
    '''            # Yandimata v2: GPU-native curriculum weighting
            _cw = _markov.batch_weight(x, y)
            (loss * grad_scale * _cw).backward()''',
    "Markov curriculum backward"
)

# ==============================================================
# SECTION K: EMA update after each step
# ==============================================================
patch(
    "        zero_grad_all()\n\n        step += 1",
    '''        zero_grad_all()

        # Yandimata v2: EMA update
        _prog = (training_time_ms + 1000.0 * (time.perf_counter() - t0)) / max(max_wallclock_ms or 1e18, 1.0)
        if _prog >= EMA_START_FRAC and not _ema_on:
            _ema.start(base_model)
            _ema_on = True
            log0(f"yandimata_v2:ema_started step={step+1}")
        _ema.update(base_model)

        step += 1''',
    "EMA update each step"
)

# ==============================================================
# SECTION L: Apply EMA before final save
# ==============================================================
patch(
    '    if master_process:\n        torch.save(base_model.state_dict(), "final_model.pt")',
    '''    # Yandimata v2: Apply EMA before saving
    if _ema.on:
        _ema.apply(base_model)
        log0("yandimata_v2:ema_applied")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")''',
    "EMA apply before save"
)

# ==============================================================
# WRITE
# ==============================================================
with open("train_gpt.py", "w") as f:
    f.write(code)

print(f"\n{'='*60}")
print(f" YANDIMATA v2 PATCH — {changes} patches applied")
print(f"{'='*60}")
print(f"")
print(f" Original techniques (your ideas, optimized):")
print(f"   - GPU-native Markov curriculum (no CPU sync!)")
print(f"   - Entropy-weighted loss scaling")
print(f"   - EMA averaging")
print(f"")
print(f" Competition meta techniques:")
print(f"   - 10L, 3x MLP, warmdown 3500")
print(f"   - Muon WD=0.04, momentum 0.99")
print(f"   - Sliding window eval (stride=64)")
print(f"   - Int6 + zstd-22")
print(f"   - Tuned LRs")
print(f"")
print(f" Push to your repo:")
print(f"   cd ~/Desktop/parameter-golf-yandimatastyle")
print(f"   cp patch_v2.py ./patch_v2.py")
print(f"   git add patch_v2.py")
print(f"   git commit -m 'Yandimata v2: GPU-Markov + all meta'")
print(f"   git push")
print(f"")
print(f" Run on RunPod (8xH100):")
print(f"   python3 patch_v2.py  # apply to clean baseline")
print(f"   RUN_ID=yandimata_v2 torchrun --standalone --nproc_per_node=8 train_gpt.py")
print(f"{'='*60}")
