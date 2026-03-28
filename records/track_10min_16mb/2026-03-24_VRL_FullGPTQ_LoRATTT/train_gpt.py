"""Parameter Golf v43: v42 + Batched LoRA TTT — Atharva Date (ADIITJ).
11L + VRL + LeakyReLU(0.5)² + Full GPTQ + EMA(0.997) + Tight SWA + Late QAT@0.15
+ XSA-all(11) + Partial RoPE(16/64) + LN Scale + VE128(9,10) + SmearGate
+ BigramHash(2048) + Prune(2%) + Batched LoRA TTT (rank=8, 2 ep, cosine LR).
VRL from arxiv:2410.17897, LoRA TTT from PR #77 (samacqua) / PR #512 (MatoTeziTanka).
Base: PR #414 (signalrush 1.1228) + PR #569 (gowtham0992 VRL+FullGPTQ+LeakyReLU² 1.1175).
Attribution: signalrush, gowtham0992, MatoTeziTanka, LoquiAuris.
"""
from __future__ import annotations
import copy, glob, io, json, math, os, random, struct, subprocess, sys, time, uuid, zlib
try:
    import zstandard as zstd; HAS_ZSTD = True
except ImportError: HAS_ZSTD = False
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch, torch.distributed as dist, torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func as _fa3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

# ── HYPERPARAMETERS ──
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 256))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_ns_steps = int(os.environ.get("MUON_NS_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    smear_enabled = bool(int(os.environ.get("SMEAR_ENABLED", "1")))
    backout_enabled = bool(int(os.environ.get("BACKOUT_ENABLED", "0")))
    backout_init = float(os.environ.get("BACKOUT_INIT", 0.2))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_interval = int(os.environ.get("SWA_INTERVAL", 50))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    # GPTQ calibration
    gptq_calib_batches = int(os.environ.get("GPTQ_CALIB_BATCHES", 256))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))
    # QAT-export alignment: STE clip percentile matches GPTQ export
    qat_clip_pct = float(os.environ.get("QAT_CLIP_PCT", 0.9995))
    prune_pct = float(os.environ.get("PRUNE_PCT", 0.02))  # post-quant magnitude pruning
    # LoRA TTT hyperparameters
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", "8"))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", "0.01"))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", "2"))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", "256"))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", "2048"))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", "64"))
    ttt_min_doc_len = int(os.environ.get("TTT_MIN_DOC_LEN", "256"))

# ── SIMPLE MUON (Newton-Schulz5) ──
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16(); X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, ns_steps, wd=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, wd=wd, nesterov=nesterov))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, ns_steps = group["lr"], group["momentum"], group["ns_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad; state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]; buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("wd", 0.0); curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0: p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr); curr += p.numel()
        return loss

# ── TOKENIZER-AGNOSTIC EVALUATION ──
def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size()); table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary_token_np[tid] = False
        if sp.is_byte(tid): base_bytes_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): has_leading_space_np[tid] = True; piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0: raise ValueError(f"Val too short for seq_len={seq_len}")
    return tokens[:usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, eval_seq_len=0):
    seq_len = eval_seq_len if eval_seq_len > 0 else args.train_seq_len
    local_batch_seqs = args.val_batch_size // (world_size * grad_accum_steps) // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size; seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            local = val_tokens[bss*seq_len:(bse*seq_len)+1].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            val_loss_sum += batch_loss.to(torch.float64) * float(y.numel())
            val_token_count += float(y.numel())
            tb = base_bytes_lut[y.reshape(-1)].to(dtype=torch.int16)
            tb += (has_leading_space_lut[y.reshape(-1)] & ~is_boundary_token_lut[x.reshape(-1)]).to(dtype=torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [val_loss_sum, val_token_count, val_byte_count]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0); tpb = val_token_count.item() / val_byte_count.item()
    model.train(); return float(val_loss.item()), float(bpt * tpb)

# ── QUANTIZATION: Full GPTQ (Hessian-aware) + QAT-export alignment ──
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,backout_lambda,bigram.scale,ve_layer_scales,ve_shared.scale,vrl_alphas".split(",") if p)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_PER_ROW_SCALE_DTYPE = torch.float16

def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if "bigram" in name: return "bigram"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    if "ve_shared" in name: return "ve"
    return "other"

def quantize_int6_gptq(weight, hessian=None, clip_range=31, block_size=128):
    """Full GPTQ: Hessian-aware int6 quantization with Cholesky error compensation.
    Based on the reference implementation from IST-DASLab/gptq (ICLR 2023).
    If hessian is None, falls back to GPTQ-lite (percentile search)."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _quantize_int6_percentile(t32, clip_range)
    rows, cols = t32.shape
    H = hessian.float().clone()
    # Kill dead columns
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    # Add damping
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp
    # Column reordering by descending activation (actorder — most important first)
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    # Compute Hessian inverse via Cholesky
    try:
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        # Cholesky failed — fall back to GPTQ-lite
        return _quantize_int6_percentile(t32, clip_range)
    # Determine per-row scale via percentile search on ORIGINAL weights
    best_q = None; best_scale = None; best_err = float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        # GPTQ block-wise quantization with Cholesky error compensation
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            # Propagate block error to remaining columns
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        # Evaluate reconstruction error (element-wise, on permuted weights)
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    # Undo column permutation
    best_q = best_q[:, inv_perm]
    return best_q, best_scale

def _quantize_int6_percentile(t32, clip_range=31):
    """Fallback: GPTQ-lite percentile search (for 1D or no-Hessian cases)."""
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale

def quantize_float_tensor(t):
    """Standard int8 quantization for embeddings."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_q = 99.99984 / 100.0
        clip_abs = torch.quantile(t32.abs(), clip_q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_q = 99.99984 / 100.0
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_mixed(state_dict, hessians=None):
    """Mixed int6/int8 quantization. Uses Full GPTQ when Hessian data available."""
    result, meta = {}, {}
    int6_cats = {"mlp", "attn", "bigram", "ve"}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"; continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float(); meta[name] = "passthrough_ctrl"; continue
        if cat in int6_cats and t.ndim >= 1:
            H = hessians.get(name) if hessians else None
            q, s = quantize_int6_gptq(t, hessian=H)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = {"type": "int6"}; continue
        q, s = quantize_float_tensor(t)
        result[name + ".q"] = q; result[name + ".scale"] = s
        meta[name] = {"type": "int8"}
    return result, meta

def dequantize_state_dict_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        orig_dtype = orig.dtype
        if isinstance(info, str) and info.startswith("passthrough"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16): t = t.to(orig_dtype)
            out[name] = t; continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

# ── DATA LOADING ──
def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ValueError(f"Bad header: {file}")
    num_tokens = int(header[2])
    if file.stat().st_size != header_bytes + num_tokens * np.dtype("<u2").itemsize: raise ValueError(f"Size mismatch: {file}")
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes).astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files); self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n):
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail); chunks.append(self.tokens[self.pos:self.pos+k]); self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device; self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        per_rank_span = global_tokens // (self.world_size * grad_accum_steps) + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span; local = chunk[start:start+per_rank_span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ── TRANSFORMER MODULES ──
class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    _qat_clip_pct: float = 0.9995  # v41: QAT-export alignment — match STE to GPTQ export
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                # v41: Use matched clip percentile for QAT-export alignment
                row_clip = torch.quantile(w32.abs(), CastedLinear._qat_clip_pct, dim=1)
                scale = (row_clip / 31.0).clamp_min(1.0 / 31.0)  # int6: clip_range=31
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -31, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()  # STE: straight-through estimator
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim; self.base = base; self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0; self._cos_cached = self._sin_cached = None
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else: inv_freq = self.inv_freq.to(device)
            freqs = torch.outer(torch.arange(seq_len, device=device, dtype=inv_freq.dtype), inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]; self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads; self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False); self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False); self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False
    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x, v_embed=None, v_residual=None, q_lora=None, v_lora=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        if q_lora is not None: q = q + q_lora(x)  # v43: LoRA delta on Q
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None: v = v + v_embed
        if v_residual is not None: v = v + v_residual  # v42: VRL — add first layer's V
        if v_lora is not None: v = v + v_lora(x)  # v43: LoRA delta on V
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if HAS_FA3:
            y = _fa3_func(q, k, v, causal=True)
            if isinstance(y, tuple): y = y[0]
        else:
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=None, is_causal=True,
                                               enable_gqa=(self.num_kv_heads != self.num_heads)).transpose(1, 2)
        if self.use_xsa: y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, int(mlp_mult * dim), bias=False)
        self.proj = CastedLinear(int(mlp_mult * dim), dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x):
        # v41: LeakyReLU(0.5)² — preserves negative gradient flow, doubles effective MLP capacity
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
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
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32); mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t); out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, ve_dim, kv_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, kv_dim, bias=False) if ve_dim != kv_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids):
        h = self.embed(token_ids)
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(self, x, x0, v_embed=None, v_residual=None, q_lora=None, v_lora=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed, v_residual=v_residual, q_lora=q_lora, v_lora=v_lora)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, smear_enabled=True, backout_enabled=True, backout_init=0.2,
                 bigram_vocab_size=0, bigram_dim=128, xsa_last_n=0,
                 rope_dims=0, ln_scale=False,
                 ve_enabled=False, ve_dim=128, ve_layers="9,10"):
        super().__init__()
        self.tie_embeddings, self.tied_embed_init_std = tie_embeddings, tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.smear_enabled, self.backout_enabled, self.num_layers = smear_enabled, backout_enabled, num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim) if smear_enabled else None
        self.backout_lambda = nn.Parameter(backout_init * torch.ones(1)) if backout_enabled else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, ln_scale=ln_scale)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        kv_dim = num_kv_heads * (model_dim // num_heads)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None; self.ve_layer_scales = nn.ParameterList()
        self.final_norm = RMSNorm()
        # v42: VRL — per-layer alpha for value residual from layer 0
        self.vrl_enabled = num_layers > 1
        if self.vrl_enabled:
            self.vrl_alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) for _ in range(num_layers - 1)
            ])
        else:
            self.vrl_alphas = nn.ParameterList()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        self._init_weights()
    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        nl = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False): nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad(): module.weight.mul_(1.0 / math.sqrt(2 * nl))
        for i, block in enumerate(self.blocks):
            with torch.no_grad():
                phase = torch.sigmoid(torch.tensor(3.0 * (i / max(nl-1, 1) - 0.5)))
                block.resid_mix.data[0] = phase * torch.ones(block.resid_mix.shape[1])
                block.resid_mix.data[1] = (1-phase) * torch.ones(block.resid_mix.shape[1])
    def _get_ve(self, layer_idx, input_ids, ve_cache):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices: return None
        if 've' not in ve_cache: ve_cache['ve'] = self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_cache['ve'] * self.ve_layer_scales[ve_idx].to(dtype=ve_cache['ve'].dtype)
    def _run_layers(self, x, x0, input_ids, lora=None):
        skips, backout_layer, x_backout = [], self.num_layers // 2, None
        ve_cache = {}
        # v42: VRL — precompute layer 0's V projection
        # At layer 0, x == x0, so x_in = mix[0]*x0 + mix[1]*x0
        v0_raw = None
        if self.vrl_enabled:
            blk0 = self.blocks[0]
            mix0 = blk0.resid_mix.to(dtype=x0.dtype)
            x_in0 = mix0[0][None, None, :] * x0 + mix0[1][None, None, :] * x0
            v0_raw = blk0.attn.c_v(blk0.attn_norm(x_in0) * blk0.ln_scale_factor)
        vrl_idx = 0
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            v_res = None
            if i > 0 and v0_raw is not None:
                alpha = torch.sigmoid(self.vrl_alphas[vrl_idx].to(dtype=x.dtype))
                v_res = alpha * v0_raw
                vrl_idx += 1
            q_lora = lora.q_loras[i] if lora is not None else None
            v_lora_fn = lora.v_loras[i] if lora is not None else None
            x = self.blocks[i](x, x0, v_embed=ve, v_residual=v_res, q_lora=q_lora, v_lora=v_lora_fn); skips.append(x)
            if i == backout_layer: x_backout = x
        for i in range(self.num_decoder_layers):
            li = self.num_encoder_layers + i
            if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(li, input_ids, ve_cache)
            v_res = None
            if v0_raw is not None:
                alpha = torch.sigmoid(self.vrl_alphas[vrl_idx].to(dtype=x.dtype))
                v_res = alpha * v0_raw
                vrl_idx += 1
            q_lora = lora.q_loras[li] if lora is not None else None
            v_lora_fn = lora.v_loras[li] if lora is not None else None
            x = self.blocks[li](x, x0, v_embed=ve, v_residual=v_res, q_lora=q_lora, v_lora=v_lora_fn)
            if li == backout_layer and x_backout is None: x_backout = x
        if self.backout_lambda is not None and x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return x
    def _embed(self, input_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None: x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (self.tok_emb.weight.shape[1],))
        if self.smear is not None: x = self.smear(x)
        return x
    def forward(self, input_ids, target_ids, lora=None):
        x0 = self._embed(input_ids); x = self._run_layers(x0, x0, input_ids, lora=lora)
        x_normed = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x_normed, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_normed)
        if lora is not None:
            logits_proj = logits_proj + lora.lm_head_lora(x_normed)
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                target_ids.reshape(-1), reduction="none").reshape(input_ids.shape)
        logits = self.logit_softcap * torch.tanh(logits_proj.reshape(-1, logits_proj.size(-1)) / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")
    def forward_logits(self, input_ids):
        x0 = self._embed(input_ids); x = self.final_norm(self._run_layers(x0, x0, input_ids))
        logits = F.linear(x, self.tok_emb.weight.to(x.dtype)) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

# ── GPTQ CALIBRATION: Collect Hessian H = X^T X per linear layer ──
def collect_hessians(base_model, train_loader, args, device, grad_accum_steps, num_batches=256):
    """Run calibration batches through the model, collecting H = X^T X for each CastedLinear."""
    hessians = {}  # param_name -> H matrix (cols x cols)
    hooks = []
    param_to_name = {}
    for name, module in base_model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            param_to_name[id(module)] = param_name
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')
            def make_hook(mod_id, pname, ncols):
                count = [0]
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])  # (B*T, D)
                    # Accumulate H = X^T X on CPU to save GPU memory
                    xtx = (x.T @ x).cpu()
                    hessians[pname] += xtx
                    count[0] += x.shape[0]
                return hook_fn
            h = module.register_forward_hook(make_hook(id(module), param_name, cols))
            hooks.append(h)
    base_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(num_batches):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            _ = base_model(x, y)
    for h in hooks: h.remove()
    # Normalize and add damping
    for name in hessians:
        H = hessians[name]
        H /= num_batches  # average
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    base_model.train()
    return hessians

# ── SLIDING WINDOW EVAL ──

# ── BATCHED LORA TTT ─────────────────────────────────────────────────────────
BOS_ID = 1  # SentencePiece BOS token ID

class BatchedLinearLoRA(nn.Module):
    """Per-batch-element LoRA adapter. Delta = x @ A^T @ B^T."""
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)
    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()

class BatchedTTTLoRA(nn.Module):
    """All LoRA adapters for one batch: LM head + Q/V per block."""
    def __init__(self, bsz: int, model, rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.blocks:
            q_out = block.attn.c_q.weight.shape[0]
            v_out = block.attn.c_v.weight.shape[0]
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, q_out, rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, v_out, rank))
    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()

def _reset_ttt_optimizer(opt):
    for group in opt.param_groups:
        for p in group["params"]:
            s = opt.state.get(p)
            if not s: continue
            s["exp_avg"].zero_()
            s["exp_avg_sq"].zero_()
            s["step"].fill_(0)

def _build_ttt_optimizer(lora, args):
    return torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr,
                            betas=(args.beta1, args.beta2), eps=1e-10)

def _find_docs(all_tokens: Tensor):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].cpu().numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) + 1 if i + 1 < len(bos_positions) else all_tokens.numel()
        if end - start >= 2:
            docs.append((start, end - start))
    return docs

def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len

def eval_val_ttt_lora(args, base_model, rank, world_size, device, val_tokens,
                      base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """Batched LoRA TTT evaluation: per-document adaptation, sharded across GPUs.
    Score-before-train on every chunk, every epoch (backward-looking — fully legal).
    Cosine LR decay across epochs: lr * 0.5 * (1 + cos(pi * ep / n_epochs))."""
    docs = _find_docs(val_tokens)
    rank_docs = docs[(len(docs) * rank) // world_size: (len(docs) * (rank + 1)) // world_size]
    short_docs = [d for d in rank_docs if d[1] < args.ttt_min_doc_len]
    long_docs = [d for d in rank_docs if d[1] >= args.ttt_min_doc_len]
    master = rank == 0
    if master:
        print(f"lora_ttt: {len(docs)} total docs, rank0: {len(long_docs)} long + {len(short_docs)} short, batch={args.ttt_batch_seqs}")

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    # Score short docs without TTT adaptation
    t0 = time.perf_counter()
    with torch.no_grad():
        for ds, dl in short_docs:
            x = val_tokens[ds: ds + dl - 1].to(device=device, dtype=torch.int64).unsqueeze(0)
            y = val_tokens[ds + 1: ds + dl].to(device=device, dtype=torch.int64).unsqueeze(0)
            n = dl - 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mean_loss = base_model(x, y)
            loss_sum += mean_loss.to(torch.float64) * n
            token_count += n
            tgt = y.reshape(-1); px = x.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[px]).to(torch.float64)
            byte_sum += tb.sum()
    if master:
        print(f"lora_ttt: short_docs time={time.perf_counter()-t0:.1f}s tokens={int(token_count.item())}")

    # Sort long docs by chunk count for batch efficiency
    long_docs.sort(key=lambda d: (d[1] - 2) // args.ttt_chunk_size)
    batch_size = args.ttt_batch_seqs
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len
    n_epochs = args.ttt_epochs
    base_lr = args.ttt_lora_lr

    lora = BatchedTTTLoRA(batch_size, base_model, args.ttt_lora_rank).to(device)
    opt = _build_ttt_optimizer(lora, args)

    t1 = time.perf_counter()
    for bi in range(0, len(long_docs), batch_size):
        batch = long_docs[bi: bi + batch_size]
        bsz = len(batch)
        if bsz == batch_size:
            cur_lora, cur_opt = lora, opt
            cur_lora.reset(); _reset_ttt_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, args.ttt_lora_rank).to(device)
            cur_opt = _build_ttt_optimizer(cur_lora, args)
        pred_lens = [dl - 1 for _, dl in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)
        for epoch in range(n_epochs):
            # Cosine LR decay
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * epoch / n_epochs))
            for pg in cur_opt.param_groups: pg["lr"] = base_lr * lr_scale
            for ci in range(max_nc):
                active = [ci < nc for nc in num_chunks]
                ws_ref, wl_ref, _, _ = _compute_chunk_window(ci, (ci+1)*chunk_size, ci+1, chunk_size, eval_seq_len)
                x = torch.zeros(bsz, wl_ref, dtype=torch.int64, device=device)
                y = torch.zeros(bsz, wl_ref, dtype=torch.int64, device=device)
                doc_info = []
                for b in range(bsz):
                    if not active[b]:
                        doc_info.append((0, 0)); continue
                    ds, dl = batch[b]
                    ws, wl, co, cl = _compute_chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                    toks = val_tokens[ds + ws: ds + ws + wl + 1].to(dtype=torch.int64, device=device)
                    x[b, :wl] = toks[:-1]; y[b, :wl] = toks[1:]
                    doc_info.append((co, cl))
                needs_train = any(ci < nc - 1 for nc in num_chunks)
                if needs_train:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        ptl = base_model(x, y, lora=cur_lora)
                else:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        ptl = base_model(x, y, lora=cur_lora)
                # Score on final epoch only
                if epoch == n_epochs - 1:
                    with torch.no_grad():
                        for b in range(bsz):
                            if not active[b]: continue
                            co, cl = doc_info[b]
                            loss_sum += ptl[b, co: co + cl].to(torch.float64).sum()
                            token_count += cl
                            tgt = y[b, co: co + cl]; px = x[b, co: co + cl]
                            tb = base_bytes_lut[tgt].to(torch.float64)
                            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[px]).to(torch.float64)
                            byte_sum += tb.sum()
                if needs_train:
                    train_loss = torch.zeros(bsz, device=device)
                    for b in range(bsz):
                        if ci >= num_chunks[b] - 1: continue
                        co, cl = doc_info[b]
                        if cl > 0: train_loss[b] = ptl[b, co: co + cl].mean()
                    cur_opt.zero_grad()
                    train_loss.sum().backward()
                    cur_opt.step()
        if master and (bi // batch_size + 1) % 5 == 0:
            elapsed = time.perf_counter() - t1
            done = bi + bsz
            print(f"lora_ttt: batch {bi // batch_size + 1}/{(len(long_docs) + batch_size - 1) // batch_size} time={elapsed:.1f}s avg_loss={loss_sum.item() / max(token_count.item(), 1):.4f}")

    if master:
        print(f"lora_ttt: long_docs time={time.perf_counter()-t1:.1f}s docs={len(long_docs)}")

    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = (loss_sum / (byte_sum * math.log(2))).item()
    return val_loss, val_bpb

def eval_val_sliding(logits_fn, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     seq_len, stride, eval_batch_seqs=256):
    total = val_tokens.numel() - 1; windows, p = [], 0
    while p + seq_len <= total:
        s = 0 if p == 0 else (seq_len - stride); windows.append((p, s)); p += stride
    n = len(windows); per_rank = (n + world_size - 1) // world_size
    my_windows = windows[rank*per_rank:min((rank+1)*per_rank, n)]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for i in range(0, len(my_windows), eval_batch_seqs):
            batch = my_windows[i:i+eval_batch_seqs]; bs = len(batch)
            x_list = [val_tokens[w:w+seq_len] for w, _ in batch]
            y_list = [val_tokens[w+1:w+seq_len+1] for w, _ in batch]
            pad = eval_batch_seqs - bs
            if pad > 0: x_list.extend([x_list[-1]]*pad); y_list.extend([y_list[-1]]*pad)
            x = torch.stack(x_list).to(device=device, dtype=torch.int64)
            y = torch.stack(y_list).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): logits = logits_fn(x)
            for b in range(bs):
                s = batch[b][1]; sl, st = logits[b, s:], y[b, s:]
                loss_sum += F.cross_entropy(sl.float(), st, reduction="sum").to(torch.float64)
                ns = st.numel(); tok_count += ns
                prev, tgt = x[b, s:s+ns], st
                tb = base_bytes_lut[tgt].to(torch.int16)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [loss_sum, tok_count, byte_count]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (loss_sum / tok_count).item()
    return vl, vl / math.log(2.0) * (tok_count.item() / byte_count.item())

# ── MAIN ──
def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8"); args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0")); world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or 8 % world_size != 0: raise ValueError(f"Bad WORLD_SIZE={world_size}")
    grad_accum_steps = 8 // world_size; grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device); dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if not HAS_FA3:
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    logfile = None
    if master_process: os.makedirs("logs", exist_ok=True); logfile = f"logs/{args.run_id}.txt"; print(logfile)
    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)
    log0(code, console=False); log0("=" * 100, console=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_tokens:{val_tokens.numel()-1}")
    CastedLinear._qat_enabled = False
    CastedLinear._qat_clip_pct = args.qat_clip_pct  # v41: QAT-export alignment
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        smear_enabled=args.smear_enabled, backout_enabled=args.backout_enabled, backout_init=args.backout_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if not bool(int(os.environ.get("TORCH_COMPILE_DISABLE", "0"))) else base_model
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    # Optimizer setup
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named_params if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)
    if base_model.smear is not None: scalar_params.append(base_model.smear.gate)
    if base_model.backout_lambda is not None: scalar_params.append(base_model.backout_lambda)
    if base_model.bigram is not None: scalar_params.append(base_model.bigram.scale)
    if base_model.ve_shared is not None:
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales: scalar_params.append(s)
    # v42: VRL alphas
    if base_model.vrl_enabled:
        for a in base_model.vrl_alphas: scalar_params.append(a)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_param_groups = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_param_groups.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None: matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_param_groups.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None: matrix_params.append(base_model.ve_shared.proj.weight)
    optimizer_tok = torch.optim.AdamW(tok_param_groups, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, ns_steps=args.muon_ns_steps, wd=args.muon_wd)
    for group in optimizer_muon.param_groups: group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                          betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
                                           betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    xsa_layers = [i for i in range(args.num_layers) if i >= args.num_layers - args.xsa_last_n] if args.xsa_last_n > 0 else []
    log0(f"model_params:{n_params}"); log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"v42: 11L LeakyReLU(0.5)² Late-QAT@{args.late_qat_threshold} int6-all FullGPTQ EMA({args.ema_decay}) TightSWA XSA-all({args.xsa_last_n}) PartialRoPE({args.rope_dims}/64) LNScale VE128 SmearGate BigramHash({args.bigram_vocab_size}) QATalign({args.qat_clip_pct}) VRL Prune({args.prune_pct}) RawBinary")
    log0(f"XSA:last_{args.xsa_last_n} layers:{xsa_layers}")
    log0(f"FA3:{HAS_FA3} SWA:{args.swa_enabled} warmdown:{args.warmdown_iters} adam_wd:{args.adam_wd}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if step >= ws else 1.0
        step_ms = elapsed_ms / max(step, 1); wd_ms = args.warmdown_iters * step_ms
        rem_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    # WARMUP
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (ws+1) % 10 == 0: log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True): opt.load_state_dict(state)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    # MAIN TRAINING LOOP
    training_time_ms, stop_after_step = 0.0, None
    swa_state, swa_count = None, 0
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                              val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all(); train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): loss = model(x, y)
            train_loss += loss.detach(); (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for group in optimizer_muon.param_groups:
            group["momentum"] = (1-frac)*args.muon_momentum_warmup_start + frac*args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all()
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)
        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_interval == 0:
            if swa_state is None:
                swa_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
                swa_count = 1; log0(f"swa:start step:{step}")
            else:
                for n, t in base_model.state_dict().items(): swa_state[n] += t.detach().cpu()
                swa_count += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rct = torch.tensor(int(reached_cap), device=device); dist.all_reduce(rct, op=dist.ReduceOp.MAX); reached_cap = bool(rct.item())
        if stop_after_step is None and reached_cap: stop_after_step = step
    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")

    # Apply EMA weights
    log0("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)

    # v41: GPTQ calibration — collect Hessians AFTER applying EMA weights
    log0(f"gptq:calibrating with {args.gptq_calib_batches} batches...")
    calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    hessians = collect_hessians(base_model, calib_loader, args, device, grad_accum_steps,
                                num_batches=args.gptq_calib_batches)
    # Map module names to state_dict names for Hessian lookup
    hessian_map = {}
    for name, module in base_model.named_modules():
        if isinstance(module, CastedLinear):
            sd_name = name + ".weight"
            h_name = name + ".weight"
            if h_name in hessians:
                hessian_map[sd_name] = hessians[h_name]
    log0(f"gptq:collected hessians for {len(hessian_map)} layers")

    # QUANTIZE + SAVE (raw binary serialization)
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    code_bytes = len(code.encode("utf-8")); size_limit = 16_000_000
    quant_result, quant_meta = quantize_state_dict_mixed(sd_cpu, hessians=hessian_map)
    # Post-quant magnitude pruning: zero out smallest int6 weights for better compression
    if args.prune_pct > 0:
        all_int6_vals = []
        for name, info in quant_meta.items():
            if isinstance(info, dict) and info.get("type") == "int6":
                qname = name + ".q"
                if qname in quant_result:
                    all_int6_vals.append(quant_result[qname].flatten().abs().float())
        if all_int6_vals:
            all_vals = torch.cat(all_int6_vals)
            k = max(1, int(args.prune_pct * all_vals.numel()))
            threshold = all_vals.kthvalue(k).values.item()
            pruned_count = 0
            for name, info in quant_meta.items():
                if isinstance(info, dict) and info.get("type") == "int6":
                    qname = name + ".q"
                    if qname in quant_result:
                        mask = quant_result[qname].abs() <= int(threshold)
                        pruned_count += mask.sum().item()
                        quant_result[qname][mask] = 0
            total_int6 = sum(quant_result[n + ".q"].numel() for n, i in quant_meta.items() if isinstance(i, dict) and i.get("type") == "int6" and n + ".q" in quant_result)
            log0(f"prune:zeroed {pruned_count}/{total_int6} int6 weights ({100*pruned_count/max(total_int6,1):.1f}%) threshold={threshold:.0f}")
    meta_json = json.dumps(quant_meta).encode("utf-8")
    parts = [struct.pack("<I", len(meta_json)), meta_json]
    tensor_order = sorted(quant_result.keys())
    for tname in tensor_order:
        t = quant_result[tname]
        name_bytes = tname.encode("utf-8")
        dtype_map = {torch.int8: 0, torch.float16: 1, torch.float32: 2, torch.bfloat16: 3}
        dt = dtype_map.get(t.dtype, 2)
        t_np = t.contiguous().numpy() if t.dtype != torch.bfloat16 else t.contiguous().view(torch.uint16).numpy()
        raw = t_np.tobytes()
        parts.append(struct.pack("<H", len(name_bytes)))
        parts.append(name_bytes)
        parts.append(struct.pack("<BB", dt, t.ndim))
        for d in t.shape: parts.append(struct.pack("<I", d))
        parts.append(raw)
    quant_raw = b"".join(parts)
    if HAS_ZSTD:
        model_blob = zstd.ZstdCompressor(level=22).compress(quant_raw); comp_name = "zstd22"
    else:
        model_blob = zlib.compress(quant_raw, level=9); comp_name = "zlib9"
    model_bytes = len(model_blob); total_size = code_bytes + model_bytes
    log0(f"model:{model_bytes} code:{code_bytes} total:{total_size} ({total_size/1e6:.2f} MB)")
    if total_size > size_limit: log0(f"WARNING: Total size {total_size} exceeds 16MB limit by {total_size - size_limit} bytes!")
    else: log0(f"Size OK: {total_size/1e6:.2f} MB")
    if master_process:
        with open("final_model.int6.ptz", "wb") as f: f.write(model_blob)
    if distributed: dist.barrier()
    # ROUNDTRIP DEQUANTIZE
    with open("final_model.int6.ptz", "rb") as f: model_blob_loaded = f.read()
    if HAS_ZSTD: raw_data = zstd.ZstdDecompressor().decompress(model_blob_loaded)
    else: raw_data = zlib.decompress(model_blob_loaded)
    offset = 0
    meta_len = struct.unpack_from("<I", raw_data, offset)[0]; offset += 4
    loaded_meta = json.loads(raw_data[offset:offset+meta_len].decode("utf-8")); offset += meta_len
    dtype_rmap = {0: (torch.int8, np.int8), 1: (torch.float16, np.float16), 2: (torch.float32, np.float32), 3: (torch.bfloat16, np.uint16)}
    loaded_result = {}
    while offset < len(raw_data):
        name_len = struct.unpack_from("<H", raw_data, offset)[0]; offset += 2
        tname = raw_data[offset:offset+name_len].decode("utf-8"); offset += name_len
        dt, ndim = struct.unpack_from("<BB", raw_data, offset); offset += 2
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack_from("<I", raw_data, offset)[0]); offset += 4
        torch_dt, np_dt = dtype_rmap[dt]
        numel = 1
        for s in shape: numel *= s
        nbytes = numel * np.dtype(np_dt).itemsize
        arr = np.frombuffer(raw_data, dtype=np_dt, count=numel, offset=offset).copy()
        offset += nbytes
        t = torch.from_numpy(arr).reshape(shape)
        if torch_dt == torch.bfloat16: t = t.view(torch.bfloat16)
        loaded_result[tname] = t
    deq_state = dequantize_state_dict_mixed(loaded_result, loaded_meta, sd_cpu)
    base_model.load_state_dict(deq_state, strict=True)
    eval_sl = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_tokens_eval = load_validation_tokens(args.val_files, eval_sl) if eval_sl != args.train_seq_len else val_tokens
    raw_logits_fn = torch.compile(base_model.forward_logits, dynamic=False) if not bool(int(os.environ.get("TORCH_COMPILE_DISABLE", "0"))) else base_model.forward_logits
    warmup_x = torch.zeros(args.eval_batch_seqs, eval_sl, dtype=torch.int64, device=device)
    base_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16): _ = raw_logits_fn(warmup_x)
    torch.cuda.synchronize(); t_eval = time.perf_counter()
    q_vl, q_vb = eval_val_sliding(raw_logits_fn, rank, world_size, device,
        val_tokens_eval, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_sl, args.eval_stride, eval_batch_seqs=args.eval_batch_seqs)
    torch.cuda.synchronize(); eval_time = time.perf_counter() - t_eval
    log0(f"final_int6_zstd_roundtrip val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f} eval_time:{eval_time*1000:.0f}ms")
    log0(f"final_int6_zstd_roundtrip_exact val_loss:{q_vl:.8f} val_bpb:{q_vb:.8f}")
    # v43: LoRA TTT evaluation
    if args.ttt_enabled:
        log0(f"ttt:starting lora_ttt rank={args.ttt_lora_rank} lr={args.ttt_lora_lr} epochs={args.ttt_epochs} chunk={args.ttt_chunk_size} cosine_lr=True")
        t_ttt = time.perf_counter()
        ttt_vl, ttt_vb = eval_val_ttt_lora(
            args, base_model, rank, world_size, device, val_tokens_eval,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        ttt_time = time.perf_counter() - t_ttt
        log0(f"final_ttt_lora val_loss:{ttt_vl:.4f} val_bpb:{ttt_vb:.4f} eval_time:{ttt_time*1000:.0f}ms lora_rank:{args.ttt_lora_rank} chunk_size:{args.ttt_chunk_size} epochs:{args.ttt_epochs}")
        log0(f"final_ttt_lora_exact val_loss:{ttt_vl:.8f} val_bpb:{ttt_vb:.8f}")
        log0(f"ttt_gain: {q_vb - ttt_vb:.6f} BPB gain over int6 (int6:{q_vb:.6f} ttt:{ttt_vb:.6f})")
    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
