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
try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 0))
    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY", 0))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
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
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    bigramhash_buckets = int(os.environ.get("BIGRAMHASH_BUCKETS", 4096))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    smeargate = bool(int(os.environ.get("SMEARGATE", "1")))
    unet_skips = bool(int(os.environ.get("UNET_SKIPS", "1")))
    int6_qat = bool(int(os.environ.get("INT6_QAT", "1")))
    rope_partial_dims = int(os.environ.get("ROPE_PARTIAL_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    xsa_layers = int(os.environ.get("XSA_LAYERS", 4))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 0))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    recurrence_repeats = int(os.environ.get("RECURRENCE_REPEATS", 0))
    recurrence_unique_head = int(os.environ.get("RECURRENCE_UNIQUE_HEAD", 2))
    recurrence_unique_tail = int(os.environ.get("RECURRENCE_UNIQUE_TAIL", 2))
    moe_num_experts = int(os.environ.get("MOE_NUM_EXPERTS", 0))
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
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
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss
def int6_ste(w: Tensor) -> Tensor:
    if w.ndim == 2:
        abs_max = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        abs_max = w.abs().amax().clamp_min(1e-8)
    scale = abs_max / 31.0
    w_q = (w / scale).round().clamp(-31, 31) * scale
    return w + (w_q - w).detach()
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
    max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 0))
    if max_tokens > 0 and tokens.numel() > max_tokens:
        tokens = tokens[:max_tokens].contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
def eval_val(
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
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte
def eval_val_sliding_ttt(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    log0=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    log0(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
         f"total_windows={len(window_starts)} stride={stride} "
         f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} "
         f"freeze_blocks={args.ttt_freeze_blocks}")
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = False
        for bi in frozen_block_ids:
            if f"blocks.{bi}." in name:
                freeze = True
                break
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)
    log0(f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)} "
         f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]
        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()
        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
         f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb
_INT6_MODE = bool(int(os.environ.get("INT6_QAT", "1")))
_INT3_MODE = bool(int(os.environ.get("INT3_QUANT", "0")))  # Ternary quantization
if _INT3_MODE:
    _QUANT_MAX_VAL = 3  # Ternary: -3, -2, -1, 0, 1, 2, 3
    _QUANT_BITS = 3
elif _INT6_MODE:
    _QUANT_MAX_VAL = 31
    _QUANT_BITS = 6
else:
    _QUANT_MAX_VAL = 127
    _QUANT_BITS = 8
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())
def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t
_GPTQ_LITE_PERCENTILES = [0.9999, 0.99995, 0.99999, 0.999995, 1.0]
def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    qmax = _QUANT_MAX_VAL
    if t32.ndim == 2 and t32.numel() > 0:
        abs_vals = t32.abs()
        best_q = None
        best_scale = None
        best_mse = None
        for pct in _GPTQ_LITE_PERCENTILES:
            if pct >= 1.0:
                clip_abs = abs_vals.amax(dim=1)
            else:
                clip_abs = torch.quantile(abs_vals, pct, dim=1)
            clip_abs = clip_abs.clamp_min(1e-8)
            scale = (clip_abs / qmax).clamp_min(1.0 / qmax)
            clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
            q = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax)
            recon = q * scale[:, None]
            mse = (t32 - recon).square().mean(dim=1)
            if best_mse is None:
                best_mse = mse
                best_q = q
                best_scale = scale
            else:
                improved = mse < best_mse
                if improved.any():
                    best_mse = torch.where(improved, mse, best_mse)
                    best_q = torch.where(improved[:, None], q, best_q)
                    best_scale = torch.where(improved, scale, best_scale)
        return best_q.to(torch.int8).contiguous(), best_scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    if t32.ndim == 2:
        return torch.empty_like(t32, dtype=torch.int8), torch.empty((t32.shape[0],), dtype=INT8_PER_ROW_SCALE_DTYPE)
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / qmax if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qmax, qmax).to(torch.int8).contiguous()
    return q, scale
def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
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
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": f"int{_QUANT_BITS}_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats
def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
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
        chunks: list[Tensor] = []
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
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _int6_qat: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._int6_qat and self.training:
            w = int6_ste(w)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class BigramHash(nn.Module):
    def __init__(self, num_buckets: int, dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, dim)
        nn.init.zeros_(self.emb.weight)
    def forward(self, input_ids: Tensor) -> Tensor:
        prev = F.pad(input_ids[:, :-1], (1, 0), value=0)
        bigram_hash = (prev * 31 + input_ids) % self.num_buckets
        return self.emb(bigram_hash)
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        positions = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        smooth = torch.cumsum(x, dim=1) / positions
        return g * x + (1 - g) * smooth
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_partial_dims: int = 0,
        use_xsa: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        rope_dim = rope_partial_dims if 0 < rope_partial_dims < self.head_dim else self.head_dim
        self.rope_dim = rope_dim
        self.rotary = Rotary(rope_dim, base=rope_base)
        self.use_xsa = use_xsa
    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        if self.rope_dim < self.head_dim:
            q_rope, q_pass = q[..., :self.rope_dim], q[..., self.rope_dim:]
            k_rope, k_pass = k[..., :self.rope_dim], k[..., self.rope_dim:]
            q_rope = apply_rotary_emb(q_rope, cos, sin)
            k_rope = apply_rotary_emb(k_rope, cos, sin)
            q = torch.cat([q_rope, q_pass], dim=-1)
            k = torch.cat([k_rope, k_pass], dim=-1)
        else:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        if self.use_xsa:
            if self.num_kv_heads != self.num_heads:
                repeats = self.num_heads // self.num_kv_heads
                v_expanded = v.repeat_interleave(repeats, dim=1)
            else:
                v_expanded = v
            y = y - v_expanded
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())
class MoEMLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        expert_hidden = (mlp_mult * dim) // num_experts
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                "fc": CastedLinear(dim, expert_hidden, bias=False),
                "proj": CastedLinear(expert_hidden, dim, bias=False),
            })
            for _ in range(num_experts)
        ])
        for e in self.experts:
            e["proj"]._zero_init = True
        self.router = nn.Linear(dim, num_experts, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        logits = self.router(x.detach())
        weights = torch.softmax(logits, dim=-1)
        indices = weights.argmax(dim=-1)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            if not mask.any():
                continue
            tokens = x[mask]
            h = F.leaky_relu(expert["fc"](tokens), negative_slope=0.5)
            h = expert["proj"](h.square())
            out[mask] = h
        return out
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        use_smeargate: bool = False,
        rope_partial_dims: int = 0,
        use_xsa: bool = False,
        ln_scale_factor: float = 1.0,
        moe_num_experts: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            rope_partial_dims=rope_partial_dims,
            use_xsa=use_xsa,
        )
        self.mlp = MoEMLP(dim, mlp_mult, moe_num_experts) if moe_num_experts > 1 else MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.smear = SmearGate(dim) if use_smeargate else None
        self.ln_scale_factor = ln_scale_factor
    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed = self.attn_norm(x)
        if self.ln_scale_factor != 1.0:
            normed = normed * self.ln_scale_factor
        attn_out = self.attn(normed)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_normed = self.mlp_norm(x)
        if self.ln_scale_factor != 1.0:
            mlp_normed = mlp_normed * self.ln_scale_factor
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_normed)
        if self.smear is not None:
            x = self.smear(x)
        return x
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        bigramhash_buckets: int = 0,
        use_smeargate: bool = False,
        unet_skips: bool = True,
        rope_partial_dims: int = 0,
        ln_scale: bool = False,
        xsa_layers: int = 0,
        recurrence_repeats: int = 0,
        recurrence_unique_head: int = 2,
        recurrence_unique_tail: int = 2,
        moe_num_experts: int = 0,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram_hash = BigramHash(bigramhash_buckets, model_dim) if bigramhash_buckets > 0 else None
        self.recurrence_repeats = recurrence_repeats
        if recurrence_repeats > 0:
            self.num_encoder_layers = 0
            self.num_decoder_layers = 0
            self.num_skip_weights = 0
            self.skip_weights = nn.Parameter(torch.zeros(0, dtype=torch.float32))
            n_head = recurrence_unique_head
            n_tail = recurrence_unique_tail
            effective_layers = n_head + recurrence_repeats + n_tail
            def make_block(layer_idx: int, eff_total: int) -> Block:
                xsa_start_idx = eff_total - xsa_layers if xsa_layers > 0 else eff_total
                return Block(
                    model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                    use_smeargate=use_smeargate,
                    rope_partial_dims=rope_partial_dims,
                    use_xsa=(layer_idx >= xsa_start_idx),
                    ln_scale_factor=(1.0 / math.sqrt(layer_idx + 1)) if ln_scale else 1.0,
                    moe_num_experts=moe_num_experts,
                )
            self.head_blocks = nn.ModuleList([make_block(i, effective_layers) for i in range(n_head)])
            self.shared_block = make_block(n_head, effective_layers)
            self.recurrence_gate = nn.Parameter(torch.full((recurrence_repeats,), 0.5, dtype=torch.float32))
            self.tail_blocks = nn.ModuleList([
                make_block(n_head + recurrence_repeats + i, effective_layers) for i in range(n_tail)
            ])
            self.blocks = nn.ModuleList()
        else:
            if unet_skips:
                self.num_encoder_layers = num_layers // 2
                self.num_decoder_layers = num_layers - self.num_encoder_layers
                self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
            else:
                self.num_encoder_layers = num_layers
                self.num_decoder_layers = 0
                self.num_skip_weights = 0
            self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
            xsa_start = num_layers - xsa_layers if xsa_layers > 0 else num_layers
            self.blocks = nn.ModuleList(
                [
                    Block(
                        model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                        use_smeargate=use_smeargate,
                        rope_partial_dims=rope_partial_dims,
                        use_xsa=(i >= xsa_start),
                        ln_scale_factor=(1.0 / math.sqrt(i + 1)) if ln_scale else 1.0,
                        moe_num_experts=moe_num_experts,
                    )
                    for i in range(num_layers)
                ]
            )
            self.head_blocks = nn.ModuleList()
            self.shared_block = None
            self.recurrence_gate = None
            self.tail_blocks = nn.ModuleList()
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram_hash is not None:
            x = x + self.bigram_hash(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        if self.recurrence_repeats > 0:
            for block in self.head_blocks:
                x = block(x, x0)
            for r in range(self.recurrence_repeats):
                gate = torch.sigmoid(self.recurrence_gate[r]).to(dtype=x.dtype)
                residual = x
                x = self.shared_block(x, x0)
                x = gate * x + (1.0 - gate) * residual
            for block in self.tail_blocks:
                x = block(x, x0)
        else:
            skips: list[Tensor] = []
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if getattr(self, '_return_logits', False):
            return logits.float().reshape(input_ids.shape[0], input_ids.shape[1], -1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram_hash is not None:
            x = x + self.bigram_hash(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        if self.recurrence_repeats > 0:
            for block in self.head_blocks:
                x = block(x, x0)
            for r in range(self.recurrence_repeats):
                gate = torch.sigmoid(self.recurrence_gate[r]).to(dtype=x.dtype)
                residual = x
                x = self.shared_block(x, x0)
                x = gate * x + (1.0 - gate) * residual
            for block in self.tail_blocks:
                x = block(x, x0)
        else:
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
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if not int(os.environ.get("TORCHDYNAMO_DISABLE", "0")):
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    _use_flash = sys.platform != "win32"
    enable_cudnn_sdp(False)
    enable_flash_sdp(_use_flash)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(not _use_flash)
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    CastedLinear._int6_qat = args.int6_qat
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigramhash_buckets=args.bigramhash_buckets,
        use_smeargate=args.smeargate,
        unet_skips=args.unet_skips,
        rope_partial_dims=args.rope_partial_dims,
        ln_scale=args.ln_scale,
        xsa_layers=args.xsa_layers,
        recurrence_repeats=args.recurrence_repeats,
        recurrence_unique_head=args.recurrence_unique_head,
        recurrence_unique_tail=args.recurrence_unique_tail,
        moe_num_experts=args.moe_num_experts,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if int(os.environ.get("TORCHDYNAMO_DISABLE", "0")):
        compiled_model = base_model
    else:
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    if args.recurrence_repeats > 0:
        block_named_params = (
            list(base_model.head_blocks.named_parameters())
            + list(base_model.shared_block.named_parameters())
            + list(base_model.tail_blocks.named_parameters())
        )
        if base_model.recurrence_gate is not None:
            block_named_params.append(("recurrence_gate", base_model.recurrence_gate))
    else:
        block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    embed_params: list[nn.Parameter] = [base_model.tok_emb.weight]
    if base_model.bigram_hash is not None:
        embed_params.append(base_model.bigram_hash.emb.weight)
    optimizer_tok = torch.optim.Adam(
        [{"params": embed_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"sdp_backends:cudnn=False flash={_use_flash} mem_efficient=False math={not _use_flash}")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    log0(
        f"meta_baseline: int6_qat:{args.int6_qat} bigramhash:{args.bigramhash_buckets} "
        f"smeargate:{args.smeargate} ema_decay:{args.ema_decay} unet_skips:{args.unet_skips} "
        f"warmdown_iters:{args.warmdown_iters} compression:{'zstd-22' if _HAS_ZSTD else 'zlib-9'}"
    )
    log0(
        f"strong_gains: leaky_relu_sq:True rope_partial_dims:{args.rope_partial_dims} "
        f"ln_scale:{args.ln_scale} xsa_layers:{args.xsa_layers}"
    )
    if args.recurrence_repeats > 0:
        eff = args.recurrence_unique_head + args.recurrence_repeats + args.recurrence_unique_tail
        log0(
            f"depth_recurrence: head={args.recurrence_unique_head} repeats={args.recurrence_repeats} "
            f"tail={args.recurrence_unique_tail} effective_layers={eff}"
        )
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    ema_state: dict[str, Tensor] | None = None
    if args.ema_decay > 0:
        ema_state = {name: param.data.clone() for name, param in base_model.named_parameters()}
        log0(f"ema:enabled decay={args.ema_decay}")
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            ema_backup: dict[str, Tensor] | None = None
            if ema_state is not None:
                ema_backup = {}
                with torch.no_grad():
                    for name, p in base_model.named_parameters():
                        ema_backup[name] = p.data.clone()
                        p.data.copy_(ema_state[name])
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            if ema_backup is not None:
                with torch.no_grad():
                    for name, p in base_model.named_parameters():
                        p.data.copy_(ema_backup[name])
                del ema_backup
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        if ema_state is not None:
            with torch.no_grad():
                decay = args.ema_decay
                for name, p in base_model.named_parameters():
                    ema_state[name].mul_(decay).add_(p.data, alpha=1.0 - decay)
        step += 1
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0 and master_process:
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"ckpt_step{step}.pt"
            ckpt_data = {"step": step, "model": base_model.state_dict()}
            if ema_state is not None:
                ckpt_data["ema"] = {k: v.cpu() for k, v in ema_state.items()}
            torch.save(ckpt_data, ckpt_path)
            ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"), key=lambda p: p.stat().st_mtime)
            for old in ckpts[:-2]:
                old.unlink(missing_ok=True)
            log0(f"checkpoint:saved {ckpt_path} (kept {min(len(ckpts), 2)} latest)")
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    if ema_state is not None:
        with torch.no_grad():
            for name, p in base_model.named_parameters():
                p.data.copy_(ema_state[name])
        log0("ema:applied EMA weights for export")
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")
    quant_label = f"int{_QUANT_BITS}"
    compress_label = "zstd" if _HAS_ZSTD else "zlib"
    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    if _HAS_ZSTD:
        compressor = zstd.ZstdCompressor(level=22)
        quant_blob = compressor.compress(quant_raw)
    else:
        quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    artifact_name = f"final_model.{quant_label}.ptz"
    if master_process:
        with open(artifact_name, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(artifact_name)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model {quant_label}+{compress_label}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size {quant_label}+{compress_label}: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open(artifact_name, "rb") as f:
        quant_blob_disk = f.read()
    if _HAS_ZSTD:
        decompressor = zstd.ZstdDecompressor()
        quant_state = torch.load(io.BytesIO(decompressor.decompress(quant_blob_disk)), map_location="cpu")
    else:
        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_{quant_label}_{compress_label}_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_{quant_label}_{compress_label}_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
        )
        torch.cuda.synchronize()
        log0(
            f"final_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride if args.eval_stride > 0 else 64,
            log0=log0,
        )
        torch.cuda.synchronize()
        log0(
            f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
    if int(os.environ.get("NGRAM_EVAL", "0")):
        ngram_max_order = int(os.environ.get("NGRAM_MAX_ORDER", "9"))
        log0(f"ngram_eval:starting max_order={ngram_max_order}")
        torch.cuda.synchronize()
        t_ngram = time.perf_counter()
        try:
            from ngram_eval import eval_val_ngram
            ng_val_loss, ng_val_bpb = eval_val_ngram(
                args, base_model, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                max_order=ngram_max_order, log_fn=log0,
            )
            torch.cuda.synchronize()
            log0(
                f"ngram_eval_result val_loss:{ng_val_loss:.4f} val_bpb:{ng_val_bpb:.4f} "
                f"eval_time:{1000.0 * (time.perf_counter() - t_ngram):.0f}ms"
            )
            log0(f"ngram_eval_result_exact val_loss:{ng_val_loss:.8f} val_bpb:{ng_val_bpb:.8f}")
        except Exception as e:
            log0(f"ngram_eval:FAILED {e}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
