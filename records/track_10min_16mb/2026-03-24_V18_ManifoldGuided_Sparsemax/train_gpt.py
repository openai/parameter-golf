"""
V18 Manifold-Guided Architecture for Parameter Golf
====================================================
Submission for OpenAI Parameter Golf challenge.
Architecture: Physics-simulated token manifold + geodesic message passing + manifold-guided attention.
"""

#best on runpod

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch._inductor.config
torch._inductor.config.max_autotune = False
torch._inductor.config.coordinate_descent_tuning = False
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP



import os



# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 50000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 65_536))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 64))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # V18 model shape
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    model_dim = int(os.environ.get("MODEL_DIM", 500))
    n_hops = int(os.environ.get("N_HOPS", 4))
    k_nb = int(os.environ.get("K_NB", 12))
    n_attn_heads = int(os.environ.get("N_ATTN_HEADS", 2))
    geo_mixer_expansion = int(os.environ.get("GEO_MIXER_EXPANSION", 2))
    quant_levels = int(os.environ.get("QUANT_LEVELS", 256))

    # Manifold build
    manifold_shards = int(os.environ.get("MANIFOLD_SHARDS", 80))
    physics_steps = int(os.environ.get("PHYSICS_STEPS", 5000))
    physics_dim = int(os.environ.get("PHYSICS_DIM", 300))
    hessian_modes = int(os.environ.get("HESSIAN_MODES", 256))

    # Optimizer
    lr = float(os.environ.get("LR", 4e-3))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.01))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.5))


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def eval_val(
    args, model, rank, world_size, device, grad_accum_steps,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
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
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION (from baseline)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = ()
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict) -> Tensor:
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor, quant_levels: int = 256) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Adaptive clipping: try multiple percentiles per row, pick lowest MSE
        clip_percentiles = [0.999, 0.9995, 0.9999, 0.99999, 1.0]
        best_q = None
        best_scale = None
        best_mse = None
        for pct in clip_percentiles:
            if pct >= 1.0:
                ca = t32.abs().amax(dim=1)
            else:
                ca = torch.quantile(t32.abs(), pct, dim=1)
            ca = ca.clamp_min(1e-8)
            s = (ca / 127.0).clamp_min(1.0 / 127.0)
            clipped = torch.maximum(torch.minimum(t32, ca[:, None]), -ca[:, None])
            q = torch.clamp(torch.round(clipped / s[:, None]), -127, 127)
            if quant_levels < 256:
                step = 256.0 / quant_levels
                q = torch.round(q / step) * step
                q = q.clamp(-128, 127)
            # Reconstruction MSE per row
            recon = q * s[:, None]
            mse = ((t32 - recon) ** 2).mean(dim=1)
            if best_mse is None:
                best_mse = mse
                best_q = q
                best_scale = s
            else:
                # Pick per-row winner
                better = mse < best_mse
                best_mse = torch.where(better, mse, best_mse)
                best_q = torch.where(better.unsqueeze(1), q, best_q)
                best_scale = torch.where(better, s, best_scale)
        return best_q.to(torch.int8).contiguous(), best_scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(t32.abs().max().item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127)
    if quant_levels < 256:
        step = 256.0 / quant_levels
        q = torch.round(q / step) * step
        q = q.clamp(-128, 127)
    return q.to(torch.int8).contiguous(), scale

def quantize_state_dict_int8(state_dict, quant_levels=256):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes = {}
    qmeta = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t, quant_levels=quant_levels)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING (from baseline)
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        local_tokens = (local_tokens // seq_len) * seq_len  # round down to multiple of seq_len
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# DISTRIBUTED MANIFOLD BUILD
# -----------------------------

def build_manifold_distributed(args, rank, world_size, device, log0):
    """Build token manifold using all GPUs for parallel shard processing."""
    V = args.vocab_size
    t_total = time.time()

    # Split shards across ranks for parallel co-occurrence accumulation
    train_files = sorted(glob.glob(args.train_files))
    n_shards = min(args.manifold_shards, len(train_files))
    my_shards = [train_files[i] for i in range(rank, n_shards, world_size)]
    log0(f"  Rank {rank}: processing {len(my_shards)}/{n_shards} shards for manifold")

    # Accumulate stats on GPU using torch for speed
    cooc = torch.zeros((V, V), dtype=torch.float64, device=device)
    left_ctx = torch.zeros((V, V), dtype=torch.float64, device=device)
    right_ctx = torch.zeros((V, V), dtype=torch.float64, device=device)
    succ = torch.zeros((V, V), dtype=torch.float64, device=device)
    fwd_cooc = torch.zeros((V, V), dtype=torch.float64, device=device)
    left_bigram = torch.zeros((V, V), dtype=torch.float64, device=device)
    right_bigram = torch.zeros((V, V), dtype=torch.float64, device=device)

    for si, fpath in enumerate(my_shards):
        enc = load_data_shard(Path(fpath)).to(torch.int64).to(device)

        # Force 1: Springs (co-occurrence)
        for off in range(1, 11):
            w = 1.0 / off
            s, d = enc[:-off], enc[off:]
            pairs = s * V + d
            counts = torch.bincount(pairs, minlength=V*V).reshape(V, V).to(torch.float64)
            cooc += w * (counts + counts.T)

        # Force 2: Torsion
        for off in range(1, 6):
            w = 1.0 / off
            max_i = len(enc) - 2 * off
            if max_i <= 0:
                continue
            s = enc[off:off+max_i]
            ld = enc[:max_i]
            rd = enc[2*off:2*off+max_i]
            left_ctx += w * torch.bincount(s*V+ld, minlength=V*V).reshape(V,V).to(torch.float64)
            right_ctx += w * torch.bincount(s*V+rd, minlength=V*V).reshape(V,V).to(torch.float64)

        # Force 3: Successor counts
        torch.add(succ, torch.bincount(enc[:-1]*V + enc[1:], minlength=V*V).reshape(V,V).to(torch.float64), out=succ)

        # Force 4: Directed springs
        for off in range(1, 6):
            w = 1.0 / off
            s, d = enc[:-off], enc[off:]
            fwd_cooc += w * torch.bincount(s*V+d, minlength=V*V).reshape(V,V).to(torch.float64)

        # Force 5: Syntactic bigrams
        s_ = enc[1:]
        ld_ = enc[:-1]
        left_bigram += torch.bincount(s_*V+ld_, minlength=V*V).reshape(V,V).to(torch.float64)
        right_bigram += torch.bincount(ld_*V+s_, minlength=V*V).reshape(V,V).to(torch.float64)

        del enc
        if si % 5 == 0:
            log0(f"    rank {rank} shard {si+1}/{len(my_shards)} ({time.time()-t_total:.0f}s)")

    # All-reduce across ranks
    log0(f"  All-reducing manifold stats...")
    for t in [cooc, left_ctx, right_ctx, succ, fwd_cooc, left_bigram, right_bigram]:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    log0(f"  Stats accumulated ({time.time()-t_total:.0f}s)")

    # Force 1: finalize springs (on CPU for scipy compatibility later)
    cooc_f = cooc.float()
    rs = cooc_f.sum(1, keepdim=True) + 1e-8
    cs = cooc_f.sum(0, keepdim=True) + 1e-8
    tot = cooc_f.sum() + 1e-8
    ppmi = torch.clamp(torch.log(cooc_f * tot / (rs * cs) + 1e-8), min=0)
    ppmi.fill_diagonal_(0)
    ppmi_cpu = ppmi.cpu().numpy()
    rows, cols = np.where(np.triu(ppmi_cpu > 0.5, k=1))
    spring_w = np.sqrt(ppmi_cpu[rows, cols]).astype(np.float32)
    del cooc, cooc_f, ppmi, ppmi_cpu
    log0(f"  Force 1: {len(rows):,} spring pairs")

    # Force 2: torsion → directional coords
    left_f = left_ctx.float()
    right_f = right_ctx.float()
    left_norm = left_f / (left_f.sum(1, keepdim=True) + 1e-8)
    right_norm = right_f / (right_f.sum(1, keepdim=True) + 1e-8)
    asymmetry = torch.sqrt(((left_norm - right_norm)**2).sum(1)).cpu().numpy()
    lr_diff_t = (left_norm - right_norm).float()
    U_dir_full, S_dir_full, _ = torch.linalg.svd(lr_diff_t, full_matrices=False)
    U_dir_t, S_dir_t = U_dir_full[:, :32], S_dir_full[:32]
    directional_coords = (U_dir_t * S_dir_t.unsqueeze(0)).cpu().numpy().astype(np.float32)
    # Fix SVD sign ambiguity — make largest element in each column positive
    for i in range(directional_coords.shape[1]):
        if directional_coords[np.argmax(np.abs(directional_coords[:, i])), i] < 0:
            directional_coords[:, i] *= -1
    del left_ctx, right_ctx, left_f, right_f, left_norm, right_norm, lr_diff_t
    log0(f"  Force 2: torsion done")

    # Force 3: entropic mass
    succ_f = succ.float()
    succ_norm = succ_f / (succ_f.sum(1, keepdim=True) + 1e-8)
    succ_entropy = -(succ_norm * torch.log(succ_norm + 1e-8)).sum(1).cpu().numpy()
    no_succ = succ_f.sum(1).cpu().numpy() < 10
    succ_entropy[no_succ] = succ_entropy.max()
    entropic_mass = 1.0 / (succ_entropy + 1.0)
    entropic_mass = (entropic_mass / entropic_mass.max()).astype(np.float32)
    del succ, succ_f, succ_norm
    log0(f"  Force 3: entropic mass done")

    # Force 4: directed springs
    fwd_f = fwd_cooc.float()
    fwd_norm_ = fwd_f / (fwd_f.sum(1, keepdim=True) + 1e-8)
    bwd_norm_ = fwd_f.T / (fwd_f.T.sum(1, keepdim=True) + 1e-8)
    dir_strength = torch.abs(fwd_norm_ - bwd_norm_)
    dir_strength = (dir_strength + dir_strength.T) / 2
    dir_strength.fill_diagonal_(0)
    ds_cpu = dir_strength.cpu().numpy()
    dir_threshold = np.percentile(ds_cpu[ds_cpu > 0], 90)
    dir_rows, dir_cols = np.where(np.triu(ds_cpu > dir_threshold, k=1))
    dir_w_vals = ds_cpu[dir_rows, dir_cols].astype(np.float32)
    del fwd_cooc, fwd_f, fwd_norm_, bwd_norm_, dir_strength, ds_cpu
    log0(f"  Force 4: {len(dir_rows):,} directed pairs")

    # Force 5: syntactic coords
    lb_f = left_bigram.float()
    rb_f = right_bigram.float()
    lb_f /= (lb_f.sum(1, keepdim=True) + 1e-8)
    rb_f /= (rb_f.sum(1, keepdim=True) + 1e-8)
    ctx_dist_t = torch.cat([lb_f, rb_f], dim=1).float()
    del left_bigram, right_bigram, lb_f, rb_f
    U_syn_full, S_syn_full, _ = torch.linalg.svd(ctx_dist_t, full_matrices=False)
    U_syn_t, S_syn_t = U_syn_full[:, :32], S_syn_full[:32]
    syntactic_coords = (U_syn_t * S_syn_t.unsqueeze(0)).cpu().numpy().astype(np.float32)
    # Fix SVD sign ambiguity — make largest element in each column positive
    for i in range(syntactic_coords.shape[1]):
        if syntactic_coords[np.argmax(np.abs(syntactic_coords[:, i])), i] < 0:
            syntactic_coords[:, i] *= -1
    del ctx_dist_t
    log0(f"  Force 5: syntactic done")

    # Physics simulation (runs on rank 0's GPU, broadcast result)
    torch.cuda.empty_cache()
    log0(f"\n  Physics simulation ({args.physics_dim}D, {args.physics_steps} steps)...")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    pos = torch.nn.Parameter(torch.randn(V, args.physics_dim, device=device) * 0.1)
    opt_sim = torch.optim.Adam([pos], lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sim, args.physics_steps, eta_min=0.0005)

    src_t = torch.tensor(np.concatenate([rows, cols]), dtype=torch.long, device=device)
    dst_t = torch.tensor(np.concatenate([cols, rows]), dtype=torch.long, device=device)
    sw_t = torch.tensor(np.concatenate([spring_w, spring_w]), dtype=torch.float32, device=device)
    mass_t = torch.tensor(entropic_mass, dtype=torch.float32, device=device)
    asym_t = torch.tensor(asymmetry, dtype=torch.float32, device=device)
    dsrc_t = torch.tensor(np.concatenate([dir_rows, dir_cols]), dtype=torch.long, device=device)
    ddst_t = torch.tensor(np.concatenate([dir_cols, dir_rows]), dtype=torch.long, device=device)
    dw_t = torch.tensor(np.concatenate([dir_w_vals, dir_w_vals]), dtype=torch.float32, device=device)
    n_rep = min(80000, V*(V-1)//2)
    sf = (V*(V-1)//2) / n_rep
    # CPU RNG for deterministic physics across different GPU hardware
    phys_rng = torch.Generator()
    phys_rng.manual_seed(12372)
    t0 = time.time()

    for step in range(args.physics_steps):
        opt_sim.zero_grad()
        n_ss = min(200000, len(src_t))
        si = torch.randint(0, len(src_t), (n_ss,), generator=phys_rng).to(device)
        d = pos[src_t[si]] - pos[dst_t[si]]
        E_spring = (len(src_t)/n_ss) * torch.sum(sw_t[si] * torch.sum(d**2, dim=1))

        ri = torch.randint(0, V, (n_rep,), generator=phys_rng).to(device)
        rj = torch.randint(0, V-1, (n_rep,), generator=phys_rng).to(device)
        rj = rj + (rj >= ri).long()
        E_rep = sf * torch.sum(1.0 / torch.norm(pos[ri]-pos[rj], dim=1).clamp(min=1e-4))

        ai_idx = torch.where(asym_t > asym_t.median())[0]
        n_ap = min(2000, len(ai_idx)*(len(ai_idx)-1)//2)
        if n_ap > 0 and len(ai_idx) > 1:
            ai = ai_idx[torch.randint(0, len(ai_idx), (n_ap,), generator=phys_rng).to(device)]
            aj = ai_idx[torch.randint(0, len(ai_idx), (n_ap,), generator=phys_rng).to(device)]
            mk = ai != aj; ai, aj = ai[mk], aj[mk]
            E_torsion = 0.5 * torch.sum(
                asym_t[ai]*asym_t[aj] / torch.norm(pos[ai]-pos[aj], dim=1).clamp(min=1e-4)
            ) if len(ai) > 0 else torch.tensor(0.0, device=device)
        else:
            E_torsion = torch.tensor(0.0, device=device)

        gi = torch.randint(0, V, (n_rep,), generator=phys_rng).to(device)
        gj = torch.randint(0, V-1, (n_rep,), generator=phys_rng).to(device)
        gj = gj + (gj >= gi).long()
        E_grav = -sf * 0.1 * torch.sum(
            mass_t[gi]*mass_t[gj] / torch.norm(pos[gi]-pos[gj], dim=1).clamp(min=1e-4))

        if len(dsrc_t) > 0:
            n_ds = min(100000, len(dsrc_t))
            di = torch.randint(0, len(dsrc_t), (n_ds,), generator=phys_rng).to(device)
            dd = pos[dsrc_t[di]] - pos[ddst_t[di]]
            E_dir = 0.3 * (len(dsrc_t)/n_ds) * torch.sum(dw_t[di] * torch.sum(dd**2, dim=1))
        else:
            E_dir = torch.tensor(0.0, device=device)

        (E_spring + E_rep + E_torsion + E_grav + E_dir).backward()
        torch.nn.utils.clip_grad_norm_([pos], 10.0)
        opt_sim.step(); sched.step()
        if step % 1000 == 0:
            log0(f"    physics step {step} ({time.time()-t0:.0f}s)")

    positions = pos.detach().cpu().numpy()
    del pos, opt_sim, sched
    torch.cuda.empty_cache()
    log0(f"  Physics done ({time.time()-t0:.0f}s)")

    # Hessian eigendecomposition
    log0(f"  Computing Hessian...")
    coupling = np.zeros((V, V), dtype=np.float32)
    for k in range(len(rows)):
        w = float(spring_w[k])
        coupling[rows[k], cols[k]] += 2 * w
        coupling[cols[k], rows[k]] += 2 * w
    coupling += np.outer(entropic_mass, entropic_mass) * 0.1
    for k in range(len(dir_rows)):
        v = float(dir_w_vals[k]) * 0.3
        coupling[dir_rows[k], dir_cols[k]] += v
        coupling[dir_cols[k], dir_rows[k]] += v
    chunk = 256
    for i in range(0, V, chunk):
        ie = min(i+chunk, V)
        diff = positions[i:ie, None, :] - positions[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        d = np.maximum(d, 1e-8)
        coupling[i:ie] += (1.0 / (d**3)).astype(np.float32)
    np.fill_diagonal(coupling, 0)

    coupling_t = torch.from_numpy(coupling.astype(np.float64))
    evals_all, evecs_all = torch.linalg.eigh(coupling_t)
    idx_ = torch.argsort(evals_all, descending=True)[:args.hessian_modes]
    evals = evals_all[idx_].numpy()
    evecs = evecs_all[:, idx_].numpy()
    # Fix eigh sign ambiguity — make largest element in each column positive
    for i in range(evecs.shape[1]):
        if evecs[np.argmax(np.abs(evecs[:, i])), i] < 0:
            evecs[:, i] *= -1
    hessian_coords = (evecs * np.sqrt(np.abs(evals))[None, :]).astype(np.float32)
    log0(f"  Hessian: {hessian_coords.shape}, eigenvalues: {evals[0]:.2f} → {evals[-1]:.2f}")

    dir_scale = 0.5 * np.std(hessian_coords) / (np.std(directional_coords) + 1e-8)
    syn_scale = 0.3 * np.std(hessian_coords) / (np.std(syntactic_coords) + 1e-8)
    combined = np.concatenate([
        hessian_coords,
        directional_coords[:, -32:] * dir_scale,
        syntactic_coords[:, -32:] * syn_scale,
    ], axis=1).astype(np.float32)

    log0(f"  Manifold ready: {combined.shape} ({time.time()-t_total:.0f}s total)")
    return combined


# -----------------------------
# V18 ARCHITECTURE
# -----------------------------

def sparsemax(z, dim=-1):
    """Sparsemax: sparse alternative to softmax. Produces exact zeros for low scores."""
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    # Reshape k for broadcasting
    shape = [1] * z.ndim
    shape[dim] = -1
    k = k.view(shape)
    z_check = (1 + k * z_sorted > cumsum).float()
    k_z = z_check.sum(dim=dim, keepdim=True)
    tau = (cumsum.gather(dim, (k_z - 1).long()) - 1) / k_z
    return torch.clamp(z - tau, min=0)


def parallel_transport(h, v_prev, v_curr, D, blend=0.2):
    prev_mag = v_prev.norm(dim=-1, keepdim=True)
    curr_mag = v_curr.norm(dim=-1, keepdim=True)
    has_movement = ((prev_mag > 0.05) & (curr_mag > 0.05)).float()
    v_prev_n = F.normalize(v_prev, dim=-1)
    v_curr_n = F.normalize(v_curr, dim=-1)
    alignment = (v_prev_n * v_curr_n).sum(-1, keepdim=True).clamp(-1, 1)
    NS_ = v_prev_n.shape[-1]
    if NS_ < D:
        pad = torch.zeros(*v_prev_n.shape[:-1], D-NS_, device=h.device)
        v_prev_h = torch.cat([v_prev_n, pad], dim=-1)
        v_curr_h = torch.cat([v_curr_n, pad], dim=-1)
    else:
        v_prev_h = v_prev_n[..., :D]
        v_curr_h = v_curr_n[..., :D]
    diff = F.normalize(v_prev_h - v_curr_h, dim=-1)
    h_ref1 = h - 2*(h*diff).sum(-1, keepdim=True)*diff
    sum_ = F.normalize(v_prev_h + v_curr_h, dim=-1)
    h_transp = h_ref1 - 2*(h_ref1*sum_).sum(-1, keepdim=True)*sum_
    t_weight = (1.0 - alignment) / 2.0 * has_movement
    return (1 - blend*t_weight)*h + blend*t_weight*h_transp


class EntropyGeoRouter(nn.Module):
    def __init__(self, d, ns, k, vocab_size):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.vel_proj = nn.Linear(ns, d, bias=False)
        self.edge_fn = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 1))
        self.probe = nn.Linear(d, d, bias=False)
        self.geo_gate = nn.Linear(d, 1)
        nn.init.constant_(self.geo_gate.weight, 0.0)
        nn.init.constant_(self.geo_gate.bias, -4.0)

    def forward(self, messages, hidden, x_ids, scn, mask,
                static_nb, geo_velocity, ctx_conf, current_entropy):
        B, S, Dh = messages.shape
        K = min(self.k, S)

        # Soft geo routing: softmax over all positions instead of hard topk
        endpoint = F.normalize(scn + 0.4 * geo_velocity, dim=-1)
        endpoint_sim = torch.bmm(endpoint, scn.transpose(1,2))
        endpoint_sim = endpoint_sim.masked_fill(mask.unsqueeze(0), -1e4)
        endpoint_sim.diagonal(dim1=1, dim2=2).fill_(-1e4)
        H = current_entropy.detach().unsqueeze(-1)
        H_norm = H / (math.log(self.vocab_size) + 1e-8)
        vel_h = self.vel_proj(geo_velocity)
        h_aimed = F.normalize(hidden + 0.3 * vel_h, dim=-1)
        # Combine endpoint similarity with hidden-message similarity
        h_msg_sim = torch.bmm(h_aimed, messages.transpose(1, 2))  # B,S,S
        combined_geo = (endpoint_sim * H_norm + 0.5 * h_msg_sim) * 5.0  # temperature scaling for sparsemax sharpness
        combined_geo = combined_geo.masked_fill(mask.unsqueeze(0), -1e4)
        combined_geo.diagonal(dim1=1, dim2=2).fill_(-1e4)
        ew_geo = sparsemax(combined_geo, dim=-1)  # sparse continuous routing — exact zeros for distant positions
        geo_agg = torch.bmm(ew_geo, messages)  # B,S,D

        nb_vocab = static_nb.long()[x_ids]
        match = (x_ids.unsqueeze(1).unsqueeze(1) == nb_vocab.unsqueeze(3)).float()
        causal_cov = torch.tril(torch.ones(S, S, device=x_ids.device))
        match = match * causal_cov.unsqueeze(0).unsqueeze(2)
        coverage = match.sum(-1).clamp(max=1).mean(-1, keepdim=True)
        nb_msgs_s = torch.einsum('bskp,bpd->bskd', match, messages)
        nb_scn_s = torch.einsum('bskp,bpd->bskd', match, scn)
        sim_s = (nb_scn_s * scn.unsqueeze(2)).sum(-1)
        ew_s = F.softmax(self.edge_fn(sim_s.unsqueeze(-1)).squeeze(-1), dim=-1)
        static_agg = (nb_msgs_s * ew_s.unsqueeze(-1)).sum(2)

        my_mode = scn.argmax(-1)
        same_mode = (my_mode.unsqueeze(2) == my_mode.unsqueeze(1)).float()
        same_mode = same_mode.masked_fill(mask.unsqueeze(0), 0)
        same_mode.diagonal(dim1=1, dim2=2).fill_(0)
        scn_sim = torch.bmm(scn, scn.transpose(1,2)) * same_mode
        has_nb = same_mode.sum(-1, keepdim=True) > 0
        pos_seq = torch.arange(S, device=scn.device).float()
        pos_sim = -torch.abs(pos_seq.unsqueeze(0) - pos_seq.unsqueeze(1)).unsqueeze(0)
        pos_sim = pos_sim.masked_fill(mask.unsqueeze(0), -1e4)
        pos_sim.diagonal(dim1=1, dim2=2).fill_(-1e4)
        # Soft local routing: softmax over all positions instead of hard topk
        routing_sim = torch.where(has_nb, scn_sim, pos_sim * 0.01)
        del scn_sim, pos_sim, same_mode
        routing_sim = routing_sim.masked_fill(mask.unsqueeze(0), -1e4)
        routing_sim.diagonal(dim1=1, dim2=2).fill_(-1e4)
        ew_l = sparsemax(routing_sim * 5.0, dim=-1)  # temperature scaling for sparsemax sharpness
        local_agg = torch.bmm(ew_l, messages)  # B,S,D
        static_final = coverage * static_agg + (1-coverage) * local_agg

        raw_gate = torch.sigmoid(self.geo_gate(hidden))
        geo_weight = raw_gate * ctx_conf
        agg = geo_weight * geo_agg + (1-geo_weight) * static_final
        probe = F.normalize(self.probe(hidden), dim=-1)
        agg_n = F.normalize(agg, dim=-1)
        rel = torch.sigmoid((agg_n * probe).sum(-1, keepdim=True))
        return agg * rel


class MessageFn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, h):
        return self.net(h)


class HopCell(nn.Module):
    def __init__(self, d, ns):
        super().__init__()
        self.struct_mod = nn.Sequential(nn.Linear(ns, d), nn.GELU(), nn.Linear(d, d*3))
        self.update = nn.Sequential(nn.Linear(d*2, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, state, spectral, inc, message_fn):
        mod = self.struct_mod(spectral)
        scale, shift, gate = mod.chunk(3, dim=-1)
        gate = torch.sigmoid(gate)
        upd = self.update(torch.cat([state, inc], dim=-1))
        new_s = self.norm(state + gate*(scale*upd+shift))
        return new_s, message_fn(new_s)


class GeodesicMixer(nn.Module):
    def __init__(self, d, ns, expansion=2):
        super().__init__()
        d_inner = d * expansion
        self.up = nn.Linear(d, d_inner, bias=False)
        self.down = nn.Linear(d_inner, d, bias=False)
        self.spec_gate = nn.Sequential(nn.Linear(ns, d_inner), nn.Sigmoid())
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.spec_gate[0].weight)
        nn.init.zeros_(self.spec_gate[0].bias)

    def forward(self, h, spectral_coords):
        gate = self.spec_gate(spectral_coords)
        up = F.gelu(self.up(h))
        return h + self.down(up * gate)


class ManifoldGuidedAttention(nn.Module):
    def __init__(self, d, ns, n_heads=2, spectral_coords=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(d + ns + ns, d, bias=False)
        self.k = nn.Linear(d + ns, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        nn.init.eye_(self.v.weight)
        nn.init.eye_(self.out.weight)
        self.out.weight.data *= 0.01  # tiny nonzero — let attention contribute from step 0
        # Q/K manifold init: deterministic SVD of spectral coords
        if spectral_coords is not None:
            sc_t = torch.tensor(spectral_coords, dtype=torch.float32)
            U, S, Vt = torch.linalg.svd(sc_t, full_matrices=False)
            shared_proj = torch.zeros(d, ns)
            k_dims = min(d, ns)
            shared_proj[:k_dims, :] = Vt[:k_dims, :] * (0.1 / (k_dims ** 0.5))
            with torch.no_grad():
                self.q.weight.data[:, d:d+ns] = shared_proj
                self.k.weight.data[:, d:d+ns] = shared_proj
        self.manifold_gate_logit = nn.Parameter(torch.tensor(-5.0))   # start OFF, learn to use manifold
        self.nav_gate_logit = nn.Parameter(torch.tensor(-5.0))       # start OFF, learn to use navigation

    def forward(self, h, scn, geo_velocity, mask):
        B, S, D = h.shape
        q_in = torch.cat([h, scn, geo_velocity], dim=-1)
        q = self.q(q_in).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k_in = torch.cat([h, scn], dim=-1)
        k = self.k(k_in).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        manifold_w = 0.05 + 0.95 * torch.sigmoid(self.manifold_gate_logit)
        proximity = torch.bmm(scn, scn.transpose(1, 2))
        scores = scores + manifold_w * proximity.unsqueeze(1)
        nav_w = 0.05 + 0.95 * torch.sigmoid(self.nav_gate_logit)
        endpoint = F.normalize(scn + 0.4 * geo_velocity, dim=-1)
        endpoint_sim = torch.bmm(endpoint, scn.transpose(1, 2))
        navigation = proximity * endpoint_sim
        scores = scores + nav_w * navigation.unsqueeze(1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1), -1e4)
        scores = scores.clamp(-50, 50)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.norm(h + self.out(out))


class ProbeConvV18(nn.Module):
    def __init__(self, spectral_coords, precomputed_neighbors, args):
        super().__init__()
        D = args.model_dim
        NS = spectral_coords.shape[1]
        V = args.vocab_size
        N_HOPS = args.n_hops
        SEQ_LEN = args.train_seq_len

        self.D = D
        self.N_HOPS = N_HOPS
        self.V = V

        self.register_buffer('sc', torch.tensor(spectral_coords, dtype=torch.float32))
        self.register_buffer('static_nb', precomputed_neighbors)

        self.init_proj = nn.Sequential(nn.Linear(NS, D), nn.GELU(), nn.Linear(D, D), nn.LayerNorm(D))
        self.content_emb = nn.Embedding(V, D)
        self.fuse = nn.Linear(D*2, D)
        self.pos_emb = nn.Embedding(SEQ_LEN, D)

        self.message_fn = MessageFn(D)
        self.hop_cells = nn.ModuleList([HopCell(D, NS) for _ in range(N_HOPS)])
        self.hop_routers = nn.ModuleList([EntropyGeoRouter(D, NS, args.k_nb, V) for _ in range(N_HOPS)])
        self.hop_norms = nn.ModuleList([nn.LayerNorm(D) for _ in range(N_HOPS)])

        self.output = nn.Linear(D, V)
        self.manifold_attn = ManifoldGuidedAttention(D, NS, n_heads=args.n_attn_heads, spectral_coords=spectral_coords)
        self.hop_emb = nn.Embedding(N_HOPS, D)
        self.h_to_spec = nn.Linear(D, NS, bias=False)
        nn.init.zeros_(self.h_to_spec.weight)

        self.geo_mixers = nn.ModuleList([GeodesicMixer(D, NS, expansion=args.geo_mixer_expansion) for _ in range(N_HOPS)])
        self.hop_post_norms = nn.ModuleList([nn.LayerNorm(D) for _ in range(N_HOPS)])

        self.hop_residual_gate = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        nn.init.constant_(self.hop_residual_gate[2].bias, -3.0)  # suppress residual early, let hops dominate

        self.register_buffer('causal_mask', torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool())

    def compute_entropy(self, h):
        with torch.no_grad():
            logits = self.output(h)
            log_p = F.log_softmax(logits, dim=-1)
            p = log_p.exp()
            H = -(p * log_p).sum(-1)
        return H

    def forward(self, input_ids, target_ids):
        B, S = input_ids.shape
        x = input_ids
        sc = self.sc[x]
        scn = F.normalize(self.sc, dim=-1)[x]
        h = self.fuse(torch.cat([self.init_proj(sc), self.content_emb(x)], dim=-1))
        h = h + self.pos_emb(torch.arange(S, device=x.device)).unsqueeze(0)
        mask = self.causal_mask[:S, :S]

        pos = torch.arange(S, device=x.device).float()
        tau = 4.0
        rec_w = torch.exp(-(pos.unsqueeze(0) - pos.unsqueeze(1)) / tau)
        rec_w = rec_w.tril(-1)
        rec_w = rec_w / (rec_w.sum(-1, keepdim=True) + 1e-8)
        ctx_scn = torch.bmm(rec_w.unsqueeze(0).expand(B,-1,-1), scn)
        geo_velocity = F.normalize(ctx_scn - scn, dim=-1)
        geo_velocity = geo_velocity * (pos > 2).float().view(1,S,1)
        ctx_conf = (pos.clamp(max=16) / 16.0).view(1,S,1).expand(B,-1,-1)

        h_init = h

        prev_velocity = None
        for hop in range(self.N_HOPS):
            H = self.compute_entropy(h)
            if prev_velocity is not None:
                h = parallel_transport(h, prev_velocity, geo_velocity, self.D, blend=0.2)
            _, out = self.hop_cells[hop](h, sc, torch.zeros_like(h), self.message_fn)
            inc = self.hop_routers[hop](
                out, h, x, scn, mask,
                self.static_nb, geo_velocity, ctx_conf, H)
            h, _ = self.hop_cells[hop](h, sc, inc, self.message_fn)
            h = self.hop_norms[hop](h)
            h_spec = self.h_to_spec(h)
            prev_velocity = geo_velocity  # PT fix: save BEFORE h_to_spec updates velocity
            dynamic_scn = F.normalize(scn + h_spec, dim=-1)
            ctx_dyn = torch.bmm(rec_w.unsqueeze(0).expand(B,-1,-1), dynamic_scn)
            geo_velocity = F.normalize(ctx_dyn - dynamic_scn, dim=-1)
            geo_velocity = geo_velocity * (pos > 2).float().view(1,S,1)
            h_attn = h + self.hop_emb(torch.tensor(hop, device=x.device))
            h = self.manifold_attn(h_attn, scn, geo_velocity, mask)
            h_before_mix = h
            h_mixed = self.geo_mixers[hop](h, sc)
            delta = h_mixed - h_before_mix
            cap = 0.2 * h_before_mix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scale = (cap / delta_norm).clamp(max=1.0)
            h = h_before_mix + delta * scale
            h = self.hop_post_norms[hop](h)

        h_spec_final = self.h_to_spec(h)
        h_spec_init = self.h_to_spec(h_init)
        displacement = (h_spec_final - h_spec_init).norm(dim=-1, keepdim=True)
        gate = 0.05 + 0.95 * self.hop_residual_gate(displacement)
        h = h + gate * h_init

        logits = self.output(h).reshape(-1, self.V)
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    try:
        code = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        code = "best_pt_fix"
    args = Hyperparameters()

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 1
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)

    # Tokenizer + validation
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Build manifold (distributed across all GPUs)
    log0(f"\n=== Building Manifold (distributed, {world_size} GPUs) ===")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    spectral_coords = build_manifold_distributed(args, rank, world_size, device, log0)
    NS = spectral_coords.shape[1]

    # Precompute KNN neighbors
    scn_f = spectral_coords / (np.linalg.norm(spectral_coords, axis=1, keepdims=True) + 1e-8)
    sim = scn_f @ scn_f.T
    np.fill_diagonal(sim, -1)
    precomputed_neighbors = torch.tensor(
        np.argsort(-sim, axis=1)[:, :args.k_nb].astype(np.int32), dtype=torch.int32)
    log0(f"  KNN: {precomputed_neighbors.shape}")

    # Model
    log0(f"\n=== Model Setup ===")
    base_model = ProbeConvV18(spectral_coords, precomputed_neighbors, args).to(device)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"  V18 params: {n_params:,}")
    log0(f"  D={args.model_dim} NS={NS} hops={args.n_hops}")

    # Deterministic torch.compile: disable autotuning to prevent non-deterministic kernel selection
    compiled_model = torch.compile(base_model, dynamic=False)
    model = compiled_model

    # Broadcast all params from rank 0 to ensure identical init
    if distributed:
        for p in base_model.parameters():
            dist.broadcast(p.data, src=0)
        for name, buf in base_model.named_buffers():
            if buf.is_floating_point():
                dist.broadcast(buf, src=0)

    # Optimizer — plain AdamW (delta cap + gate floors handle spike prevention)
    opt = torch.optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Data loader — each GPU loads full batch size independently (different data)
    # Rank 0 needs full 65K for hop gradients (same as single-GPU Colab)
    # Rank 1 sees different 65K, contributes averaged gradients for non-hop params only
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    train_loader.world_size = 1  # each GPU loads full batch, not batch/world_size
    # Advance rank 1's stream so GPUs read different data
    if rank > 0:
        train_loader.stream.take(rank * 1000000)  # skip ahead so ranks see different shards

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_schedule(step, elapsed_ms):
        """Cosine decay to 10%, hold, then warmdown to 0."""
        # Phase 1: warmup (steps 0-100)
        warmup = min(step / 100, 1.0)
        # Phase 2: cosine decay from 1.0 to 0.1 over steps 0-3400
        if step < 3400:
            cosine = 0.5 * (1.0 + math.cos(math.pi * step / 3400))
            decay = 0.1 + 0.9 * cosine  # decays from 1.0 → 0.1
        # Phase 3: hold at 10% for steps 3400-5500
        elif step < 5500:
            decay = 0.1
        # Phase 4: linear warmdown from 10% → 0 over steps 5500-6100
        else:
            decay = 0.1 * max(1.0 - (step - 5500) / 600.0, 0.0)
        return warmup * decay

    # Warmup (torch.compile warmup)
    if args.warmup_steps > 0:
        initial_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_opt = copy.deepcopy(opt.state_dict())
        model.train()
        for ws in range(args.warmup_steps):
            opt.zero_grad(set_to_none=True)
            for micro in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            opt.step()
            if ws % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"  warmup step {ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_state, strict=True)
        opt.load_state_dict(initial_opt)
        opt.zero_grad(set_to_none=True)
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        train_loader.world_size = 1
        if rank > 0:
            train_loader.stream.take(rank * 1000000)

    # Main training loop
    log0(f"\n=== Training ===")
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_schedule(step, elapsed_ms)
        for group in opt.param_groups:
            group["lr"] = args.lr * scale

        opt.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for micro in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Selective gradient strategy:
        # Hop-chain params use rank 0's LOCAL gradient — averaging kills hop specialization.
        # Everything else uses averaged gradient — more data helps embeddings/output.
        if distributed:
            for name, param in base_model.named_parameters():
                if param.grad is None:
                    continue
                is_hop_param = any(k in name for k in [
                    'hop_cells', 'hop_routers', 'geo_mixers', 'manifold_attn',
                    'message_fn', 'h_to_spec', 'hop_norms', 'hop_post_norms',
                    'hop_emb', 'hop_residual_gate'
                ])
                if not is_hop_param:
                    # Average gradient across GPUs for non-hop params
                    dist.reduce(param.grad, dst=0, op=dist.ReduceOp.SUM)
                    if rank == 0:
                        param.grad.div_(world_size)

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        if not distributed or rank == 0:
            opt.step()

        # EMA weight accumulation for smoother quantization
        # Start after hop params freeze (~step 3000) to capture stable weight averages
        if step >= 3000:
            if not hasattr(base_model, '_ema_state'):
                base_model._ema_state = {n: p.data.clone() for n, p in base_model.named_parameters()}
                base_model._ema_count = 1
            else:
                decay = 0.999
                for n, p in base_model.named_parameters():
                    base_model._ema_state[n].mul_(decay).add_(p.data, alpha=1 - decay)
                base_model._ema_count += 1

        if distributed:
            for param in base_model.parameters():
                dist.broadcast(param.data, src=0)

        # Track best model after step 3000 — save EMA snapshot (smoother = better quantization)
        if step >= 3000 and (not distributed or rank == 0):
            current_loss = train_loss.item()
            if not hasattr(base_model, '_best_loss') or current_loss < base_model._best_loss:
                base_model._best_loss = current_loss
                base_model._best_step = step
                # Save EMA weights at this point (smoother than raw weights, quantizes better)
                if hasattr(base_model, '_ema_state'):
                    base_model._best_state = {n: v.clone() for n, v in base_model._ema_state.items()}
                else:
                    base_model._best_state = {n: p.data.clone() for n, p in base_model.named_parameters()}
                log0(f"  NEW BEST step:{step} loss:{current_loss:.4f} (saved EMA snapshot)")

        step += 1
        # Diagnostics every 200 steps — outside compiled forward
        if step % 200 == 0:
            with torch.no_grad():
                gb = [base_model.hop_routers[i].geo_gate.bias.item() for i in range(4)]
                gw = [base_model.hop_routers[i].geo_gate.weight.norm().item() for i in range(4)]
                mg = base_model.manifold_attn.manifold_gate_logit.item()
                ng = base_model.manifold_attn.nav_gate_logit.item()
                spec_w = base_model.h_to_spec.weight.norm().item()
                spec_max = base_model.h_to_spec.weight.abs().max().item()
                mix_down = [base_model.geo_mixers[i].down.weight.norm().item() for i in range(4)]
                mix_up = [base_model.geo_mixers[i].up.weight.norm().item() for i in range(4)]
                mix_sg = [base_model.geo_mixers[i].spec_gate[0].weight.norm().item() for i in range(4)]
                attn_out = base_model.manifold_attn.out.weight.norm().item()
                attn_v = base_model.manifold_attn.v.weight.norm().item()
                cell_norms = [base_model.hop_cells[i].update[0].weight.norm().item() for i in range(4)]
                msg_w = base_model.message_fn.net[0].weight.norm().item()
                resid_b = base_model.hop_residual_gate[2].bias.item()
                hop_emb_n = base_model.hop_emb.weight.norm().item()
            log0(
                f"  DIAG s{step} "
                f"gb:[{gb[0]:.2f},{gb[1]:.2f},{gb[2]:.2f},{gb[3]:.2f}] "
                f"gw:[{gw[0]:.2f},{gw[1]:.2f},{gw[2]:.2f},{gw[3]:.2f}] "
                f"mg:{mg:.2f} ng:{ng:.2f} "
                f"spec:{spec_w:.1f} smax:{spec_max:.3f} "
                f"mix_d:[{mix_down[0]:.1f},{mix_down[1]:.1f},{mix_down[2]:.1f},{mix_down[3]:.1f}] "
                f"mix_u:[{mix_up[0]:.1f},{mix_up[1]:.1f},{mix_up[2]:.1f},{mix_up[3]:.1f}] "
                f"mix_sg:[{mix_sg[0]:.2f},{mix_sg[1]:.2f},{mix_sg[2]:.2f},{mix_sg[3]:.2f}] "
                f"attn_o:{attn_out:.2f} attn_v:{attn_v:.2f} "
                f"cell:[{cell_norms[0]:.1f},{cell_norms[1]:.1f},{cell_norms[2]:.1f},{cell_norms[3]:.1f}] "
                f"msg:{msg_w:.1f} rb:{resid_b:.2f} hemb:{hop_emb_n:.1f}"
            )
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            cap_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # Serialization + roundtrip validation
    # Priority: best model (lowest loss after step 5000) > EMA > final weights
    if hasattr(base_model, '_best_state'):
        log0(f"  Using BEST model for serialization (step:{base_model._best_step} loss:{base_model._best_loss:.4f})")
        with torch.no_grad():
            for n, p in base_model.named_parameters():
                p.data.copy_(base_model._best_state[n])
    elif hasattr(base_model, '_ema_state'):
        with torch.no_grad():
            for n, p in base_model.named_parameters():
                p.data.copy_(base_model._ema_state[n])
        log0(f"  Using EMA weights for serialization (decay=0.999, {base_model._ema_count} steps)")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict(), quant_levels=args.quant_levels)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()



