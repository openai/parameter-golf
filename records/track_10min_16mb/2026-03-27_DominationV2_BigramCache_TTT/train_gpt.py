"""
Domination V2: Full frontier technique stack for parameter golf.

Adds XSA (Exclusive Self Attention, arXiv:2603.09078) on last N layers,
EMA weight averaging (replacing SWA), and TTT (test-time training) at eval.
Built on V1: 11L MLP-3x, per-dim SmearGate, BigramHash, OrthoInit, Muon WD.
"""

from __future__ import annotations

import bisect
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
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

BOS_ID = 1

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
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
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_lr = float(os.environ.get("TTT_LR", 1e-4))
    cache_enabled = bool(int(os.environ.get("CACHE_ENABLED", "1")))
    cache_alpha = float(os.environ.get("CACHE_ALPHA", 0.20))
    cache_tau = float(os.environ.get("CACHE_TAU", 8.0))
    cache_entropy_power = float(os.environ.get("CACHE_ENTROPY_POWER", 1.0))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.04))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 4096))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.5))
    swa_every = int(os.environ.get("SWA_EVERY", 50))

# -----------------------------
# MUON OPTIMIZER WITH WEIGHT DECAY
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
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
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                     nesterov=nesterov, weight_decay=weight_decay))
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
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0: p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id): base_bytes_np[token_id] = 1; continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"): has_leading_space_np[token_id] = True; piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0: raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]

def find_docs(all_tokens: Tensor) -> list[tuple[int, int]]:
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].tolist()
    docs = []
    for i, start in enumerate(bos_positions):
        end = bos_positions[i + 1] + 1 if i + 1 < len(bos_positions) else int(all_tokens.numel())
        if end - start >= 2: docs.append((int(start), int(end - start)))
    return docs

def shard_docs_by_tokens(docs: list[tuple[int, int]], rank: int, world_size: int) -> list[tuple[int, int]]:
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} out of range for world_size={world_size}")
    if not docs:
        return []
    prefix = [0]
    for _, doc_len in docs:
        prefix.append(prefix[-1] + max(doc_len - 1, 0))
    total_pred_tokens = prefix[-1]
    bounds = [0]
    prev = 0
    for shard_idx in range(1, world_size):
        target = (total_pred_tokens * shard_idx) / world_size
        lo = bisect.bisect_left(prefix, target, lo=prev, hi=len(prefix))
        candidates = [idx for idx in (lo - 1, lo) if prev <= idx <= len(docs)]
        if not candidates:
            cut = prev
        else:
            cut = min(candidates, key=lambda idx: (abs(prefix[idx] - target), -idx))
        bounds.append(cut)
        prev = cut
    bounds.append(len(docs))
    return docs[bounds[rank]:bounds[rank + 1]]

def iter_flat_score_windows(total_tokens: int, seq_len: int, stride: int):
    if total_tokens <= 0:
        return
    first_end = min(seq_len, total_tokens)
    yield 0, first_end, 0, first_end
    scored_end = first_end
    while scored_end < total_tokens:
        next_end = min(scored_end + stride, total_tokens)
        ws = max(0, next_end - seq_len)
        yield ws, next_end - ws, scored_end, next_end
        scored_end = next_end

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(f"VAL_BATCH_SIZE too small for world={world_size} accum={grad_accum_steps} seq={args.train_seq_len}")
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    vls = torch.zeros((), device=device, dtype=torch.float64)
    vtc = torch.zeros((), device=device, dtype=torch.float64)
    vbc = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(seq_start, seq_end, local_batch_seqs):
            be = min(bs + local_batch_seqs, seq_end)
            rs, re = bs * args.train_seq_len, be * args.train_seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                bl = model(x, y).detach()
            bc = float(y.numel())
            vls += bl.to(torch.float64) * bc; vtc += bc
            pi, ti = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[ti].to(dtype=torch.int16)
            tb += (has_leading_space_lut[ti] & ~is_boundary_token_lut[pi]).to(dtype=torch.int16)
            vbc += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vls, op=dist.ReduceOp.SUM); dist.all_reduce(vtc, op=dist.ReduceOp.SUM); dist.all_reduce(vbc, op=dist.ReduceOp.SUM)
    vl = vls / vtc; bpt = vl.item() / math.log(2.0); tpb = vtc.item() / vbc.item()
    model.train(); return float(vl.item()), float(bpt * tpb)

def eval_val_sliding(args, base_model, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride, batch_seqs=32):
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    windows = list(iter_flat_score_windows(total_tokens, seq_len, stride))
    total_windows = len(windows)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = windows[my_s:my_e]
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            bw = my_windows[bi:bi + batch_seqs]; bsz = len(bw)
            xb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            score_starts = []
            for i, (ws, wl, score_start, _) in enumerate(bw):
                wlens.append(wl); score_starts.append(score_start)
                ch = val_tokens[ws:ws + wl + 1].to(dtype=torch.int64, device=device)
                xb[i, :wl] = ch[:-1]; yb[i, :wl] = ch[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(xb)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none").reshape(bsz, seq_len)
            for i, (ws, _, _, _) in enumerate(bw):
                wl = wlens[i]; s = score_starts[i] - ws
                sn = nll[i, s:wl].to(torch.float64); ls += sn.sum(); tc += float(wl - s)
                tgt, prev = yb[i, s:wl], xb[i, s:wl]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                bc += tb.sum()
            if rank == 0 and (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(my_windows)); pct = done / len(my_windows) * 100
                rb = 0.0
                if tc.item() > 0: rl = (ls / tc).item(); rb = rl / math.log(2.0) * (tc.item() / bc.item())
                print(f"  sliding_eval [{pct:5.1f}%] {done}/{len(my_windows)} windows running_bpb={rb:.6f}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ls, op=dist.ReduceOp.SUM); dist.all_reduce(tc, op=dist.ReduceOp.SUM); dist.all_reduce(bc, op=dist.ReduceOp.SUM)
    vl = (ls / tc).item(); bpt = vl / math.log(2.0); tpb = tc.item() / bc.item()
    base_model.train(); return vl, bpt * tpb

def eval_val_sliding_cache(args, base_model, rank, world_size, device, val_tokens, docs,
                           base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                           stride, batch_seqs=32):
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    local_docs = shard_docs_by_tokens(docs, rank, world_size)
    loss_sum = 0.0; token_count = 0.0; byte_count = 0.0
    if not local_docs:
        ls = torch.tensor(loss_sum, device=device, dtype=torch.float64)
        tc = torch.tensor(token_count, device=device, dtype=torch.float64)
        bc = torch.tensor(byte_count, device=device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(ls, op=dist.ReduceOp.SUM); dist.all_reduce(tc, op=dist.ReduceOp.SUM); dist.all_reduce(bc, op=dist.ReduceOp.SUM)
        vl = (ls / tc).item(); bpt = vl / math.log(2.0); tpb = tc.item() / bc.item()
        base_model.train(); return vl, bpt * tpb

    local_start = local_docs[0][0]
    local_end = local_docs[-1][0] + local_docs[-1][1] - 1
    local_windows = []
    for ws, wl, score_start, score_end in iter_flat_score_windows(total_tokens, seq_len, stride):
        if score_end <= local_start: continue
        if score_start >= local_end: break
        ss = max(score_start, local_start); se = min(score_end, local_end)
        if ss < se: local_windows.append((ws, wl, ss, se))

    cache_active = args.cache_enabled
    base_model.eval()
    base_bytes_np = base_bytes_lut.cpu().numpy().astype(np.int64)
    has_space_np = has_leading_space_lut.cpu().numpy().astype(np.bool_)
    is_boundary_np = is_boundary_token_lut.cpu().numpy().astype(np.bool_)
    max_entropy = math.log(base_bytes_np.shape[0])
    max_count = np.iinfo(np.uint16).max
    row_totals = np.zeros(base_bytes_np.shape[0], dtype=np.int32)
    counts = np.zeros((base_bytes_np.shape[0], base_bytes_np.shape[0]), dtype=np.uint16)
    touched_pairs: list[tuple[int, int]] = []
    touched_rows: list[int] = []
    doc_idx = 0
    doc_end = local_docs[doc_idx][0] + local_docs[doc_idx][1] - 1

    def reset_doc_cache():
        for p, y in touched_pairs: counts[p, y] = 0
        for p in touched_rows: row_totals[p] = 0
        touched_pairs.clear(); touched_rows.clear()

    with torch.inference_mode():
        for bi in range(0, len(local_windows), batch_seqs):
            batch = local_windows[bi:bi + batch_seqs]; bsz = len(batch)
            xb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            for i, (ws, wl, _, _) in enumerate(batch):
                ch = val_tokens[ws:ws + wl + 1].to(dtype=torch.int64, device=device)
                xb[i, :wl] = ch[:-1]; yb[i, :wl] = ch[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(xb)

            if cache_active:
                log_probs = F.log_softmax(logits.float(), dim=-1)
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            else:
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none").reshape(bsz, seq_len)

            for i, (ws, _, score_start, score_end) in enumerate(batch):
                lo = score_start - ws; hi = score_end - ws
                if cache_active:
                    tgt = yb[i, lo:hi]
                    prev_np = xb[i, lo:hi].detach().cpu().numpy().astype(np.uint16, copy=True)
                    tgt_np = tgt.detach().cpu().numpy().astype(np.uint16, copy=True)
                    lp_np = log_probs[i, lo:hi].gather(1, tgt[:, None]).squeeze(1).detach().cpu().numpy().astype(np.float32, copy=True)
                    ent_np = entropy[i, lo:hi].detach().cpu().numpy().astype(np.float32, copy=True)
                    base_prob_np = np.exp(lp_np.astype(np.float64, copy=False))
                    pos = score_start; off = 0
                    while pos < score_end:
                        seg_end = min(score_end, doc_end)
                        take = seg_end - pos
                        ps = prev_np[off:off + take]
                        ys = tgt_np[off:off + take]
                        probs = base_prob_np[off:off + take]
                        ents = ent_np[off:off + take]
                        for j in range(take):
                            p = int(ps[j]); y = int(ys[j])
                            total = int(row_totals[p]); seen = int(counts[p, y]); cache_p = (seen / total) if total > 0 else 0.0
                            eff_alpha = args.cache_alpha * (total / (total + args.cache_tau)) if total > 0 else 0.0
                            eff_alpha *= max(float(ents[j]) / max_entropy, 0.0) ** args.cache_entropy_power
                            mix_p = (1.0 - eff_alpha) * float(probs[j]) + eff_alpha * cache_p
                            loss_sum += -math.log(max(mix_p, 1e-30))
                            if total == 0: touched_rows.append(p)
                            if seen == 0: touched_pairs.append((p, y))
                            counts[p, y] = min(seen + 1, max_count)
                            row_totals[p] = total + 1
                        tb = base_bytes_np[ys]
                        tb = tb + np.logical_and(has_space_np[ys], np.logical_not(is_boundary_np[ps])).astype(np.int64)
                        byte_count += float(tb.sum()); token_count += int(take)
                        pos = seg_end; off += take
                        if pos >= doc_end:
                            reset_doc_cache(); doc_idx += 1
                            if doc_idx < len(local_docs):
                                doc_end = local_docs[doc_idx][0] + local_docs[doc_idx][1] - 1
                else:
                    sn = nll[i, lo:hi]; loss_sum += float(sn.sum().item()); token_count += float(hi - lo)
                    prev, tgt = xb[i, lo:hi], yb[i, lo:hi]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += float(tb.sum().item())
            if rank == 0 and (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(local_windows)); pct = done / len(local_windows) * 100
                rb = (loss_sum / math.log(2.0)) / byte_count if byte_count > 0 else 0.0
                print(f"  cache_eval [{pct:5.1f}%] {done}/{len(local_windows)} windows running_bpb={rb:.6f}", flush=True)

    ls = torch.tensor(loss_sum, device=device, dtype=torch.float64)
    tc = torch.tensor(token_count, device=device, dtype=torch.float64)
    bc = torch.tensor(byte_count, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ls, op=dist.ReduceOp.SUM); dist.all_reduce(tc, op=dist.ReduceOp.SUM); dist.all_reduce(bc, op=dist.ReduceOp.SUM)
    vl = (ls / tc).item(); bpt = vl / math.log(2.0); tpb = tc.item() / bc.item()
    base_model.train(); return vl, bpt * tpb

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(p for p in os.environ.get("CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,bigram.scale").split(",") if p)

INT6_QUANT_RANGE = 31
MIXED_QUANT_INT6_CATS = frozenset(
    c.strip() for c in os.environ.get("MIXED_QUANT_INT6_CATS", "mlp,attn").split(",") if c.strip()
)
STE_QAT_ENABLED = bool(int(os.environ.get("STE_QAT_ENABLED", "0")))
STE_QAT_RANGE = int(os.environ.get("STE_QAT_RANGE", INT6_QUANT_RANGE))

def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    return "other"

def _get_ste_range_for_param(name):
    return STE_QAT_RANGE

def quantize_int6_per_row(t):
    t32 = t.float()
    if t32.ndim == 2:
        rm = t32.abs().amax(dim=1)
        sc = (rm / 31.0).clamp_min(1.0 / 31.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / sc.float()[:, None]), -32, 31).to(torch.int8)
        return q, sc
    am = t32.abs().max().item()
    sc = torch.tensor(am / 31.0 if am > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / sc.float()), -32, 31).to(torch.int8)
    return q, sc

def quantize_int8_per_row(t):
    t32 = t.float()
    if t32.ndim == 2:
        rm = t32.abs().amax(dim=1)
        sc = (rm / 127.0).clamp_min(1e-8).to(torch.float16)
        q = torch.clamp(torch.round(t32 / sc.float()[:, None]), -127, 127).to(torch.int8)
        return q, sc
    am = t32.abs().max().item()
    sc = torch.tensor(am / 127.0 if am > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / sc.float()), -127, 127).to(torch.int8)
    return q, sc

def mixed_quantize(state_dict, int6_cats):
    result, meta = {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_int8_per_row(t)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta

def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig.dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig.dtype)
            out[name] = t; continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig.dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig.dtype)
    return out

# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file):
    hb = 256 * np.dtype("<i4").itemsize; header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ValueError(f"Bad shard header: {file}")
    nt = int(header[2])
    if file.stat().st_size != hb + nt * 2: raise ValueError(f"Shard size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=nt, offset=hb)
    if tokens_np.size != nt: raise ValueError(f"Short read: {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0
    def _advance_file(self): self.file_idx = (self.file_idx + 1) % len(self.files); self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n):
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail); chunks.append(self.tokens[self.pos:self.pos + k]); self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device, self.stream = rank, world_size, device, TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        lt = global_tokens // (self.world_size * grad_accum_steps); prs = lt + 1
        chunk = self.stream.take(prs * self.world_size); s = self.rank * prs
        local = chunk[s:s + prs].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if self.training and STE_QAT_ENABLED and w.ndim == 2:
            with torch.no_grad():
                w32 = w.float(); rm = w32.abs().amax(dim=1).clamp_min(1e-8)
                sc = rm / 31.0; wc = torch.clamp(w32, -rm[:, None], rm[:, None])
                wq = (torch.round(wc / sc[:, None]) * sc[:, None]).to(x.dtype)
            w = w + (wq - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), persistent=False)
        self._seq_len_cached = 0; self._cos_cached = None; self._sin_cached = None
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]; self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin):
    h = x.size(-1) // 2; x1, x2 = x[..., :h], x[..., h:]
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
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous()
        if self.use_xsa:
            v_bthd = v.transpose(1, 2).contiguous()
            y = self._xsa_efficient(y, v_bthd)
        return self.proj(y.reshape(bsz, seqlen, dim))

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__(); hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False); self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x): return self.proj(torch.relu(self.fc(x)).square())

class SmearGate(nn.Module):
    """Per-dimension SmearGate (from PR #194): each dim has its own blend ratio."""
    def __init__(self, dim):
        super().__init__(); self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__(); self.bvs = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim); nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def forward(self, token_ids):
        t = token_ids.to(torch.int32); mod = self.bvs - 1
        out = torch.empty_like(t); out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        h = self.embed(out.long())
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init,
                 bigram_vocab_size=0, bigram_dim=128, xsa_last_n=0):
        super().__init__()
        self.tie_embeddings, self.tied_embed_init_std, self.logit_softcap = tie_embeddings, tied_embed_init_std, logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init) for _ in range(num_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings: nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        nl = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False): nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad(): module.weight.mul_(1.0 / math.sqrt(2 * nl))

    def _run_blocks(self, x, x0):
        skips = []
        for i in range(self.num_encoder_layers): x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return x

    def _embed(self, input_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None: x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        return self.smear(x)

    def _logits(self, x):
        if self.tie_embeddings: lp = F.linear(x, self.tok_emb.weight)
        else: lp = self.lm_head(x)
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        x0 = self._embed(input_ids)
        x = self.final_norm(self._run_blocks(x0, x0)).reshape(-1, x0.size(-1))
        return F.cross_entropy(self._logits(x).float(), target_ids.reshape(-1), reduction="mean")

    def forward_logits(self, input_ids):
        x0 = self._embed(input_ids)
        return self._logits(self.final_norm(self._run_blocks(x0, x0)))

# -----------------------------
# TRAINING
# -----------------------------

def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8"); args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size; grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device); dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
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
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:tokens:{val_tokens.numel() - 1}")

    base_model = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
                     num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
                     tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
                     logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
                     bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
                     xsa_last_n=args.xsa_last_n).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear): module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named_params if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)
    for p in base_model.smear.parameters(): scalar_params.append(p)
    if base_model.bigram is not None:
        for n, p in base_model.bigram.named_parameters():
            if p.ndim == 2 and p.shape[0] >= 64 and p.shape[1] >= 64: matrix_params.append(p)
            else: scalar_params.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
                                      betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for group in optimizer_muon.param_groups: group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                         betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizers.insert(1, torch.optim.AdamW([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
                                               betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay, fused=True))

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} swa:{args.swa_enabled} compression:{'zstd-22' if HAS_ZSTD else 'zlib-9'}")
    log0(f"bigram_vocab:{args.bigram_vocab_size} bigram_dim:{args.bigram_dim} grad_clip:{args.grad_clip_norm} muon_wd:{args.muon_wd}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} warmdown:{args.warmdown_iters} seed:{args.seed}")
    log0(f"ste_qat:{STE_QAT_ENABLED} ste_range:{STE_QAT_RANGE} int6_cats:{MIXED_QUANT_INT6_CATS}")
    log0(f"xsa_last_n:{args.xsa_last_n} ema:{args.ema_enabled} ema_decay:{args.ema_decay} ttt:{args.ttt_enabled} ttt_epochs:{args.ttt_epochs}")
    log0(f"cache_enabled:{args.cache_enabled} cache_alpha:{args.cache_alpha} cache_tau:{args.cache_tau} cache_entropy_power:{args.cache_entropy_power}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1); wms = args.warmdown_iters * step_ms
        rms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

    if args.warmup_steps > 0:
        init_sd = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opts = [copy.deepcopy(o.state_dict()) for o in optimizers]; model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): wl = model(x, y)
                (wl * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (ws + 1) % 10 == 0: log0(f"warmup_step:{ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opts, strict=True): o.load_state_dict(s)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0; stop_after_step = None; swa_state = None; swa_count = 0
    ema_state = None
    if args.ema_enabled:
        ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations: log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0); scale = lr_mul(step, elapsed_ms)
        if args.swa_enabled and not args.ema_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
            current = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
            if swa_state is None:
                swa_state = current; swa_count = 1
            else:
                inv = 1.0 / (swa_count + 1); keep = 1.0 - inv
                for k, t in current.items():
                    if torch.is_floating_point(swa_state[k]): swa_state[k].mul_(keep).add_(t, alpha=inv)
                    else: swa_state[k] = t
                swa_count += 1
        zero_grad_all(); train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): loss = model(x, y)
            train_loss += loss.detach(); (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for group in optimizer_muon.param_groups: group["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        if ema_state is not None:
            d = args.ema_decay
            with torch.no_grad():
                for n, t in base_model.state_dict().items():
                    ema_state[n].mul_(d).add_(t.detach().float(), alpha=1.0 - d)
        zero_grad_all(); step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rct = torch.tensor(int(reached_cap), device=device); dist.all_reduce(rct, op=dist.ReduceOp.MAX); reached_cap = bool(rct.item())
        if stop_after_step is None and reached_cap: stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    if ema_state is not None:
        log0("ema: applying EMA weights")
        avg_sd = {n: t.to(dtype=base_model.state_dict()[n].dtype) for n, t in ema_state.items()}
        base_model.load_state_dict(avg_sd, strict=True); del ema_state, avg_sd
    elif swa_state is not None:
        log0(f"swa: averaging {swa_count} checkpoints")
        base_model.load_state_dict(swa_state, strict=True); del swa_state

    export_sd = base_model.state_dict()
    if master_process:
        torch.save(export_sd, "final_model.pt")
        log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes  Code: {len(code.encode('utf-8'))} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    quant_result, quant_meta = mixed_quantize(sd_cpu, MIXED_QUANT_INT6_CATS)
    qbuf = io.BytesIO(); torch.save({"w": quant_result, "m": quant_meta}, qbuf); qraw = qbuf.getvalue()
    if HAS_ZSTD: qblob = zstd.ZstdCompressor(level=22).compress(qraw); cl = "zstd-22"
    else: qblob = zlib.compress(qraw, level=9); cl = "zlib-9"
    if master_process:
        with open("final_model.int8.ptz", "wb") as f: f.write(qblob)
        qfb = len(qblob); cb = len(code.encode("utf-8"))
        log0(f"final_int8_zlib_roundtrip compressed_model_bytes:{qfb} code_bytes:{cb} total_artifact_bytes:{qfb + cb}")
        log0(f"Serialized {cl}: {qfb} bytes  Total: {qfb + cb} bytes")
    if distributed: dist.barrier()
    with open("final_model.int8.ptz", "rb") as f: qbd = f.read()
    rd = zstd.ZstdDecompressor().decompress(qbd) if HAS_ZSTD else zlib.decompress(qbd)
    qs = torch.load(io.BytesIO(rd), map_location="cpu")
    base_model.load_state_dict(dequantize_mixed(qs["w"], qs["m"], sd_cpu), strict=True)

    if args.ttt_enabled and args.ttt_epochs > 0:
        log0(f"ttt: starting {args.ttt_epochs} epochs of test-time training (lr={args.ttt_lr})")
        torch.cuda.synchronize(); ttt_start = time.perf_counter()
        base_model.train()
        ttt_params = [p for p in base_model.parameters() if p.requires_grad]
        ttt_opt = torch.optim.SGD(ttt_params, lr=args.ttt_lr)
        ttt_seq_len = args.train_seq_len
        total_val = val_tokens.numel() - 1
        total_seqs = total_val // ttt_seq_len
        my_start = (total_seqs * rank) // world_size
        my_end = (total_seqs * (rank + 1)) // world_size
        for ttt_ep in range(args.ttt_epochs):
            ttt_loss_sum = 0.0; ttt_count = 0
            for si in range(my_start, my_end, 4):
                se = min(si + 4, my_end); bsz = se - si
                rs = si * ttt_seq_len; re = se * ttt_seq_len + 1
                local = val_tokens[rs:re].to(device=device, dtype=torch.int64)
                x = local[:-1].reshape(bsz, ttt_seq_len)
                y = local[1:].reshape(bsz, ttt_seq_len)
                ttt_opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = base_model(x, y)
                loss.backward()
                ttt_opt.step()
                ttt_loss_sum += loss.item() * bsz; ttt_count += bsz
            if rank == 0:
                log0(f"  ttt_epoch:{ttt_ep+1}/{args.ttt_epochs} loss:{ttt_loss_sum/max(ttt_count,1):.4f}")
        torch.cuda.synchronize()
        log0(f"ttt: done in {time.perf_counter() - ttt_start:.1f}s")

    torch.cuda.synchronize(); tqe = time.perf_counter()
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        if args.cache_enabled:
            docs = find_docs(val_tokens)
            log0(f"final_eval_mode:sliding_window_cache stride:{args.eval_stride} batch_seqs:{args.eval_batch_seqs} docs:{len(docs)}")
            qvl, qvb = eval_val_sliding_cache(args, base_model, rank, world_size, device, val_tokens, docs, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, stride=args.eval_stride, batch_seqs=args.eval_batch_seqs)
        else:
            log0(f"final_eval_mode:sliding_window stride:{args.eval_stride} batch_seqs:{args.eval_batch_seqs}")
            qvl, qvb = eval_val_sliding(args, base_model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, stride=args.eval_stride, batch_seqs=args.eval_batch_seqs)
    else:
        qvl, qvb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} eval_time:{1000.0 * (time.perf_counter() - tqe):.0f}ms")
    log0(f"final_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")
    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
