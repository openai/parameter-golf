#!/usr/bin/env python3
"""
Raki V2 — V1 proven core + safe improvements only.
10 layers (default arch), Markov curriculum, EMA, Muon WD, GPTQ clip search, zstd-22.
NO recurrence, NO trigram, NO sliding window eval, NO LR changes.

Usage:
  python3 patch_v2.py
  RUN_ID=raki_v2 torchrun --standalone --nproc_per_node=8 train_gpt.py
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

# --- zstandard auto-install ---
patch(
    'from __future__ import annotations',
    '''from __future__ import annotations
try:
    import zstandard as _zstd_check  # noqa: F401
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "zstandard", "-q"])''',
    "zstandard")

# --- Hyperparameters: only NUM_LAYERS and WARMDOWN (proven in V1) ---
patch('    num_layers = int(os.environ.get("NUM_LAYERS", 9))',
      '    num_layers = int(os.environ.get("NUM_LAYERS", 10))',
      "NUM_LAYERS=10")
patch('    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))',
      '    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))',
      "WARMDOWN=3500")

# --- Config + GPU Markov + EMA classes ---
patch(
    "from torch.nn.parallel import DistributedDataParallel as DDP",
    '''from torch.nn.parallel import DistributedDataParallel as DDP
import zstandard as zstd

MUON_WD = float(os.environ.get("MUON_WD", "0.04"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.995"))
EMA_START_FRAC = float(os.environ.get("EMA_START_FRAC", "0.85"))
RAKI_POWER = float(os.environ.get("RAKI_POWER", "0.15"))
GPTQ_CLIP_SEARCH = bool(int(os.environ.get("GPTQ_CLIP_SEARCH", "1")))
GPTQ_PERCENTILES = [0.999, 0.9995, 0.9999, 0.99999, 1.0]


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

    @torch.no_grad()
    def batch_weight(self, x: Tensor, y: Tensor) -> float:
        if RAKI_POWER <= 0:
            return 1.0
        surp = -self.log_probs[x.reshape(-1), y.reshape(-1)].float()
        ent_w = self.ent_norm[x.reshape(-1)].float()
        score = (surp * ent_w).mean().item()
        return 1.0 + RAKI_POWER * min(score / 5.0, 1.0)


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
    "config + GPU-Markov + EMA")

# --- Muon weight decay (V1 tried via env var but never implemented!) ---
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

# --- GPTQ-lite clip search (better INT8 quantization) ---
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
    if t32.ndim == 2:
        if not GPTQ_CLIP_SEARCH or not t32.numel():
            clip_abs = (torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                        if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32))
            clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
            scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
            return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
        best_q = best_scale = None
        best_mse = torch.full((t32.shape[0],), float('inf'))
        for pct in GPTQ_PERCENTILES:
            ca = t32.abs().amax(dim=1) if pct >= 1.0 else torch.quantile(t32.abs(), pct, dim=1)
            sc = (ca / 127.0).clamp_min(1.0 / 127.0)
            cl = torch.maximum(torch.minimum(t32, ca[:, None]), -ca[:, None])
            qq = torch.clamp(torch.round(cl / sc[:, None]), -127, 127)
            mse = ((t32 - qq * sc[:, None]) ** 2).mean(dim=1)
            improved = mse < best_mse
            if best_q is None:
                best_q, best_scale, best_mse = qq.to(torch.int8), sc, mse
            else:
                best_q[improved] = qq[improved].to(torch.int8)
                best_scale[improved] = sc[improved]
                best_mse[improved] = mse[improved]
        return best_q.contiguous(), best_scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale''',
    "GPTQ-lite clip search")

# --- zstd-22 compression ---
patch('    quant_blob = zlib.compress(quant_raw, level=9)',
      '    cctx = zstd.ZstdCompressor(level=22)\n    quant_blob = cctx.compress(quant_raw)',
      "zstd-22")
for i in range(3):
    patch('"final_model.int8.ptz"', '"final_model.int8.ptzst"', f"filename {i+1}")
patch('quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")',
      'dctx = zstd.ZstdDecompressor()\n    quant_state = torch.load(io.BytesIO(dctx.decompress(quant_blob_disk)), map_location="cpu")',
      "zstd decompress")
patch('f"Serialized model int8+zlib:', 'f"Serialized model int8+zstd:', "log zstd 1")
patch('f"Total submission size int8+zlib:', 'f"Total submission size int8+zstd:', "log zstd 2")
patch('f"final_int8_zlib_roundtrip val_loss', 'f"final_int8_zstd_roundtrip val_loss', "log zstd 3")
patch('f"final_int8_zlib_roundtrip_exact val_loss', 'f"final_int8_zstd_roundtrip_exact val_loss', "log zstd 4")

# --- Init Markov + EMA before training loop ---
patch("    training_time_ms = 0.0",
      '''    _markov = _GPUMarkov(args.train_files, args.vocab_size, device)
    _ema = _EMA()
    _ema_on = False
    log0(f"raki_v2: layers={args.num_layers} power={RAKI_POWER} wd={MUON_WD} ema={EMA_DECAY} gptq={GPTQ_CLIP_SEARCH}")
    training_time_ms = 0.0''',
      "init Markov + EMA")

# --- Markov curriculum weighting ---
patch("            (loss * grad_scale).backward()",
      '''            _cw = _markov.batch_weight(x, y)
            (loss * grad_scale * _cw).backward()''',
      "Markov curriculum")

# --- EMA update after optimizer step ---
patch("        zero_grad_all()\n\n        step += 1",
      '''        zero_grad_all()
        _prog = (training_time_ms + 1000.0 * (time.perf_counter() - t0)) / max(max_wallclock_ms or 1e18, 1.0)
        if _prog >= EMA_START_FRAC and not _ema_on:
            _ema.start(base_model); _ema_on = True
            log0(f"raki_v2:ema_started step={step+1}")
        _ema.update(base_model)
        step += 1''',
      "EMA update")

# --- EMA apply before save ---
patch('    if master_process:\n        torch.save(base_model.state_dict(), "final_model.pt")',
      '''    if _ema.on:
        _ema.apply(base_model)
        log0("raki_v2:ema_applied")
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")''',
      "EMA apply")

with open("train_gpt.py", "w") as f:
    f.write(code)

print(f"\nRaki V2 applied ({changes} patches): 10L + Markov + EMA + Muon WD + GPTQ + zstd")
print(f"Run: RUN_ID=raki_v2 torchrun --standalone --nproc_per_node=1 train_gpt.py")
