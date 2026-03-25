"""Training infrastructure for recurrent bestbase experiments.

Contains: Hyperparameters, Muon optimizer, data loading, evaluation
helpers, tokenizer utilities, and quantization export.  Mirrors the
current-best record utilities with additions for recurrence.
"""
from __future__ import annotations

import argparse
import copy
import glob
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from model_recurrent_bestbase import (
    CastedLinear, RecurrentGPT, CONTROL_TENSOR_NAME_PATTERNS,
    restore_low_dim_params_to_fp32,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH",
                                     "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
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
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get(
        "MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get(
        "MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 1536))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "6,7")
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    # recurrence-specific
    num_stem_layers = int(os.environ.get("NUM_STEM_LAYERS", 3))
    num_core_layers = int(os.environ.get("NUM_CORE_LAYERS", 2))
    num_tail_layers = int(os.environ.get("NUM_TAIL_LAYERS", 3))
    num_passes = int(os.environ.get("NUM_PASSES", 3))
    core_quant_bits = int(os.environ.get("CORE_QUANT_BITS", 6))
    core_quant_enabled = bool(int(os.environ.get("CORE_QUANT_ENABLED", "1")))


# ---------------------------------------------------------------------------
# CLI argument parser (overlaid on top of env-based Hyperparameters)
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("recurrence")
    g.add_argument("--recurrent-layer-range", type=str, default=None,
                   help="start,end (e.g. 3,8)")
    g.add_argument("--shared-core-layers", type=int, default=None)
    g.add_argument("--num-passes", type=int, default=None)
    g.add_argument("--quant-bits", type=int, default=None)
    g.add_argument("--quant-mode", type=str, default="per_row",
                   choices=["per_row", "per_tensor"])

    g = parser.add_argument_group("feedback")
    g.add_argument("--feedback-rank", type=int, default=2)
    g.add_argument("--feedback-mode", type=str, default="diagonal",
                   choices=["identity", "diagonal", "low_rank"])
    g.add_argument("--per-pass-feedback", action="store_true")
    g.add_argument("--affine-junction", action="store_true")

    g = parser.add_argument_group("stability")
    g.add_argument("--clip-hidden", action="store_true")
    g.add_argument("--clip-value", type=float, default=10.0)
    g.add_argument("--residual-scale-init", type=float, default=1.0)
    g.add_argument("--jacobian-proxy-weight", type=float, default=0.0)

    g = parser.add_argument_group("ttt")
    g.add_argument("--ttt-regime", type=str, default="tail_only",
                   choices=["tail_only", "tail_plus_stem",
                            "all_unique_layers", "all_layers",
                            "all_layers_with_recurrent_lr_scale"])
    g.add_argument("--ttt-recurrent-lr-scale", type=float, default=0.1)

    g = parser.add_argument_group("precision")
    g.add_argument("--leave-embeddings-fp16", action="store_true")
    g.add_argument("--leave-head-fp16", action="store_true")


def apply_cli_overrides(args: Hyperparameters,
                        cli: argparse.Namespace) -> Hyperparameters:
    if cli.recurrent_layer_range is not None:
        s, e = cli.recurrent_layer_range.split(",")
        args.num_stem_layers = int(s)
        total = args.num_stem_layers + args.num_core_layers + args.num_tail_layers
        args.num_tail_layers = total - int(e)
    if cli.shared_core_layers is not None:
        args.num_core_layers = cli.shared_core_layers
    if cli.num_passes is not None:
        args.num_passes = cli.num_passes
    if cli.quant_bits is not None:
        args.core_quant_bits = cli.quant_bits
    return args


# ---------------------------------------------------------------------------
# Newton-Schulz + Muon
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5,
                                eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


class Muon(torch.optim.Optimizer):
    """Parallel Muon: reduce-scatter → local NS5 → all-gather."""

    def __init__(self, params, lr: float, momentum: float,
                 backend_steps: int, nesterov: bool = True,
                 weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay))
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    "p": p, "B": B,
                    "padded_grad": torch.zeros(padded_B, *tail,
                                               device=dev, dtype=torch.bfloat16),
                    "shard": torch.zeros(shard_B, *tail,
                                         device=dev, dtype=torch.bfloat16),
                    "shard_mom": torch.zeros(shard_B, *tail,
                                             device=dev, dtype=torch.bfloat16),
                    "full_update": torch.zeros(padded_B, *tail,
                                               device=dev, dtype=torch.bfloat16),
                    "scale": max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m["p"].numel())
        self._built = True

    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m["p"]
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m["padded_grad"]
            pg[:m["B"]].copy_(p.grad.bfloat16())
            if pg.shape[0] > m["B"]:
                pg[m["B"]:].zero_()
            fut = dist.reduce_scatter_tensor(
                m["shard"], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self._built:
            self._build()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            prev_ag_handle = None
            prev_m = None
            sharded = self._distributed and hasattr(self, "_rs_futures")
            for i, m in enumerate(self._bank_meta):
                p = m["p"]
                if p.grad is None:
                    continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m["p"]
                    upd = prev_m["full_update"][:prev_m["B"]]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m["shard"]
                    buf = m["shard_mom"]
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(
                    update, steps=backend_steps)
                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m["full_update"], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m["scale"])
            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m["p"]
                upd = prev_m["full_update"][:prev_m["B"]]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
            if hasattr(self, "_rs_futures"):
                del self._rs_futures
        return loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens,
                            offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for {pattern}")
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
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int,
                 device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int,
                   grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start:start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return (x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True))


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if (sp.is_control(token_id) or sp.is_unknown(token_id)
                or sp.is_unused(token_id)):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
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
        raise FileNotFoundError(f"No files for {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[:usable + 1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int, world_size: int, device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor, has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y).detach()
            btc = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * btc
            val_token_count += btc
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (has_leading_space_lut[tgt_ids]
                   & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (val_loss_sum, val_token_count, val_byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)


def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int, world_size: int, device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor, has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
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
    compiled_logits = torch.compile(base_model.forward_logits,
                                    dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64,
                                  device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64,
                                  device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64,
                                                   device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
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
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt]
                       & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bpt * tpb


# ---------------------------------------------------------------------------
# Quantization export (int6 + lzma)
# ---------------------------------------------------------------------------

INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                    if t32.numel() else
                    torch.empty((t32.shape[0],), dtype=torch.float32))
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]),
                        -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = (float(torch.quantile(t32.abs().flatten(),
                INT8_CLIP_Q).item()) if t32.numel() else 0.0)
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0,
                         dtype=torch.float32)
    q = torch.clamp(
        torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale),
        -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_int6_per_row(t: Tensor, clip_range: int = 31
                          ) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            row_clip = (torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0
                        else t32.abs().amax(dim=1))
            s = (row_clip / clip_range).clamp_min(
                1.0 / clip_range).to(torch.float16)
            q = torch.clamp(
                torch.round(t32 / s.float()[:, None]),
                -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0,
                         dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()),
                    -clip_range, clip_range).to(torch.int8)
    return q, scale


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name:
        return "attn"
    return "other"


def _unbank_state_dict(sd: dict[str, Tensor],
                       num_unique: int) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    n = num_unique
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"layer.{i}.attn.c_q.weight"] = tensor[i]
                out[f"layer.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"layer.{i}.attn.c_k.weight"] = tensor[i]
                out[f"layer.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"layer.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"layer.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out


def _rebank_state_dict(sd: dict[str, Tensor], num_unique: int,
                       template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    n = num_unique
    qo = [None] * (2 * n)
    kv = [None] * (2 * n)
    up = [None] * n
    down = [None] * n
    consumed: set[str] = set()
    for i in range(n):
        for key, arr, idx in [
            (f"layer.{i}.attn.c_q.weight", qo, i),
            (f"layer.{i}.attn.proj.weight", qo, n + i),
            (f"layer.{i}.attn.c_k.weight", kv, i),
            (f"layer.{i}.attn.c_v.weight", kv, n + i),
            (f"layer.{i}.mlp.fc.weight", up, i),
            (f"layer.{i}.mlp.proj.weight", down, i),
        ]:
            if key in sd:
                arr[idx] = sd[key]
                consumed.add(key)
    out["qo_bank"] = torch.stack(qo).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up).to(
        dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down).to(
        dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out


def mixed_quantize_int6(state_dict: dict[str, Tensor],
                        int6_cats: set[str]):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = (t.to(torch.float16)
                            if t.is_floating_point() else t)
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def dequantize_mixed_int6(result: dict[str, Tensor],
                          meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        od = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and od in (torch.float32, torch.bfloat16):
                t = t.to(od)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float()
                         * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
                         ).to(od)
        else:
            out[name] = (q.float() * float(s.item())).to(od)
    return out


def export_and_roundtrip(
    base_model: RecurrentGPT,
    args: Hyperparameters,
    log0,
    device: torch.device,
    rank: int,
    world_size: int,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    feedback_module=None,
    stabilizer=None,
):
    """Quantize, export, reload, and evaluate roundtrip quality."""
    master_process = rank == 0
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    # remove feedback / stability from export state_dict
    export_sd = {k: v for k, v in sd_cpu.items()
                 if not k.startswith("_feedback") and not k.startswith("_stab")}

    unbanked = _unbank_state_dict(export_sd, base_model.num_unique)
    quant_result, quant_meta = mixed_quantize_int6(unbanked, {"mlp", "attn"})

    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = lzma.compress(quant_buf.getvalue(), preset=6)

    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        code_bytes = len(Path(__file__).read_text("utf-8").encode("utf-8"))
        log0(f"Serialized model int6+lzma: {len(quant_blob)} bytes")
        log0(f"Total submission size int6+lzma: {len(quant_blob) + code_bytes} bytes")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(lzma.decompress(quant_blob_disk)), map_location="cpu")
    deq_unbanked = dequantize_mixed_int6(
        quant_state["w"], quant_state["m"], unbanked)
    deq_sd = _rebank_state_dict(deq_unbanked, base_model.num_unique, export_sd)

    eval_model = RecurrentGPT(
        vocab_size=args.vocab_size, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim, rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
        ve_layers=args.ve_layers, xsa_last_n=args.xsa_last_n,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
        num_stem_layers=args.num_stem_layers,
        num_core_layers=args.num_core_layers,
        num_tail_layers=args.num_tail_layers,
        num_passes=args.num_passes,
        core_quant_bits=args.core_quant_bits,
        core_quant_enabled=False,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_sd, strict=False)

    effective_eval_seq_len = (args.eval_seq_len
                              if args.eval_seq_len > 0 else args.train_seq_len)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_loss, q_bpb = eval_val(
        args, eval_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut,
        is_boundary_token_lut, eval_seq_len=effective_eval_seq_len)
    torch.cuda.synchronize()
    log0(f"final_int6_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} "
         f"eval_time:{1000*(time.perf_counter()-t_qeval):.0f}ms")

    if args.eval_stride > 0 and args.eval_stride < effective_eval_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut,
            is_boundary_token_lut, stride=args.eval_stride,
            eval_seq_len=effective_eval_seq_len)
        torch.cuda.synchronize()
        log0(f"final_int6_sliding val_loss:{sw_loss:.4f} "
             f"val_bpb:{sw_bpb:.4f} stride:{args.eval_stride} "
             f"eval_time:{1000*(time.perf_counter()-t_slide):.0f}ms")

    return eval_model


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(args: Hyperparameters,
                device: torch.device) -> RecurrentGPT:
    model = RecurrentGPT(
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        xsa_last_n=args.xsa_last_n,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
        num_stem_layers=args.num_stem_layers,
        num_core_layers=args.num_core_layers,
        num_tail_layers=args.num_tail_layers,
        num_passes=args.num_passes,
        core_quant_bits=args.core_quant_bits,
        core_quant_enabled=args.core_quant_enabled,
    ).to(device).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    return model


def build_optimizers(
    base_model: RecurrentGPT,
    args: Hyperparameters,
    extra_scalar_params: list[nn.Parameter] | None = None,
) -> tuple[list[torch.optim.Optimizer], list[nn.Parameter]]:
    """Return (list_of_optimizers, replicated_params)."""
    matrix_params = [
        base_model.qo_bank, base_model.kv_bank,
        base_model.mlp_up_bank, base_model.mlp_down_bank,
    ]
    all_blocks = base_model.all_blocks()
    block_named_params = []
    for blk in all_blocks:
        block_named_params.extend(blk.named_parameters())

    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)

    token_lr = (args.tied_embed_lr if args.tie_embeddings else args.embed_lr)
    tok_groups = [{"params": [base_model.tok_emb.weight],
                   "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_groups.append({"params": [base_model.bigram.embed.weight],
                           "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            scalar_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_groups.append({"params": [base_model.ve_shared.embed.weight],
                           "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            scalar_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)

    if extra_scalar_params:
        scalar_params.extend(extra_scalar_params)

    optimizer_tok = torch.optim.AdamW(
        tok_groups, betas=(args.beta1, args.beta2),
        eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr,
          "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)

    replicated_params: list[nn.Parameter] = []
    for pg in optimizer_tok.param_groups:
        replicated_params.extend(pg["params"])
    replicated_params.extend(scalar_params)

    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    optimizer_head = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight],
              "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        replicated_params.append(base_model.lm_head.weight)
        optimizers.append(optimizer_head)

    return optimizers, replicated_params
