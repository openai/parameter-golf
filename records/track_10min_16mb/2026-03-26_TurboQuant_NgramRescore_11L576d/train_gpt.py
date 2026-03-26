from __future__ import annotations
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
import zlib
from pathlib import Path
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False
    flash_attn_3_func = None
import struct
from typing import Dict, Tuple, Optional

# =============================================================================
# TurboQuant: Rotation-based Lloyd-Max quantization (2/3/4-bit)
# Replaces int6/int8 per-row quantization with lower MSE at fewer bits.
# =============================================================================
CODEBOOK_2BIT = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104])
CODEBOOK_3BIT = torch.tensor([-2.1519, -1.3439, -0.7560, -0.2451,
                                0.2451,  0.7560,  1.3439,  2.1519])
CODEBOOK_4BIT = torch.tensor([-2.7333, -2.0698, -1.5417, -1.0833,
                               -0.6568, -0.3388, -0.1062,  0.1062,
                                0.3388,  0.6568,  1.0833,  1.5417,
                                2.0698,  2.7333])
_TURBO_CODEBOOKS = {2: CODEBOOK_2BIT, 3: CODEBOOK_3BIT, 4: CODEBOOK_4BIT}

_turbo_rotation_cache: Dict[Tuple[int, int], Tensor] = {}

@torch.compiler.disable
def _turbo_get_codebook(bits: int, dim: int, device='cpu') -> Tensor:
    return _TURBO_CODEBOOKS[bits].to(device=device) / math.sqrt(dim)

@torch.compiler.disable
def _turbo_get_rotation(dim: int, seed: int = 42, device='cpu') -> Tensor:
    key = (dim, seed)
    if key not in _turbo_rotation_cache or _turbo_rotation_cache[key].device != torch.device(device):
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
        G = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
        Q, R = torch.linalg.qr(G)
        Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
        _turbo_rotation_cache[key] = Q.float().to(device)
    return _turbo_rotation_cache[key]

class _TurboQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, rotation, codebook):
        norms = weight.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        w_unit = weight / norms
        w_rot = w_unit @ rotation.T
        dists = (w_rot.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
        w_rot_q = codebook[dists.argmin(dim=-1)]
        return w_rot_q @ rotation * norms
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def turbo_ste(weight: Tensor, rotation: Tensor, codebook: Tensor) -> Tensor:
    return _TurboQuantSTE.apply(weight, rotation, codebook)

_turbo_cb_cache: Dict[Tuple[int, int, str], Tensor] = {}

@torch.compiler.disable
def _turbo_cached_cb(bits: int, dim: int, device) -> Tensor:
    key = (bits, dim, str(device))
    if key not in _turbo_cb_cache:
        _turbo_cb_cache[key] = _turbo_get_codebook(bits, dim, device)
    return _turbo_cb_cache[key]

class TurboQuantScheduler:
    """Progressive quantization: 4-bit -> 3-bit -> 2-bit during warmdown."""
    def __init__(self):
        self.enabled = False
        self.bits = 4
    def update(self, warmdown_scale: float):
        if warmdown_scale > 0.5:
            self.enabled = False
            self.bits = 4
        elif warmdown_scale > 0.3:
            self.enabled = True
            self.bits = 4
        elif warmdown_scale > 0.15:
            self.enabled = True
            self.bits = 3
        else:
            self.enabled = True
            self.bits = 2

_turbo_scheduler = TurboQuantScheduler()
_turbo_qat_enabled = False

# TurboQuant control tensor patterns (kept in FP32/FP16)
_TURBO_CONTROL_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix",
    "resid_mixes", "q_gain", "skip_weight", "skip_weights", "smear",
    "dtg_gate", "ve_layer_scales", "ve_shared.scale", "attn_gate", "vr_lambda",
)

def _turbo_bits_for_param(name: str) -> int:
    """Assign bit-width per component type."""
    if "mlp_up" in name:
        return 2  # MLP up: high redundancy
    elif "mlp_down" in name:
        return 3  # MLP down: needs precision
    elif "qo_bank" in name or "kv_bank" in name:
        return 3  # Attention: precision-critical
    elif "tok_emb" in name or "embed" in name:
        return 4  # Embeddings: quality-critical
    else:
        return 3  # Default

def turbo_serialize(state_dict: Dict[str, Tensor], seed: int = 42) -> Tuple[Dict, Dict]:
    """Quantize state dict with TurboQuant rotation codebooks."""
    quantized = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            quantized[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "p"
            continue
        if any(p in name for p in _TURBO_CONTROL_PATTERNS):
            quantized[name] = t.float()
            meta[name] = "c"
            continue
        bits = _turbo_bits_for_param(name)
        if t.ndim == 3:
            B, M, N = t.shape
            rot = _turbo_get_rotation(N, seed)
            cb = _turbo_get_codebook(bits, N)
            all_idx, all_norms = [], []
            for b in range(B):
                w = t[b].float()
                norms = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                w_rot = (w / norms) @ rot.T
                idx = (w_rot.unsqueeze(-1) - cb.view(1, 1, -1)).abs().argmin(dim=-1)
                all_idx.append(idx.to(torch.uint8))
                all_norms.append(norms.to(torch.float16))
            quantized[name + ".q"] = torch.stack(all_idx)
            quantized[name + ".s"] = torch.stack(all_norms)
            meta[name] = {"b": bits, "d": N}
        elif t.ndim == 2:
            N = t.shape[-1]
            rot = _turbo_get_rotation(N, seed)
            cb = _turbo_get_codebook(bits, N)
            w = t.float()
            norms = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            w_rot = (w / norms) @ rot.T
            idx = (w_rot.unsqueeze(-1) - cb.view(1, 1, -1)).abs().argmin(dim=-1)
            quantized[name + ".q"] = idx.to(torch.uint8)
            quantized[name + ".s"] = norms.to(torch.float16)
            meta[name] = {"b": bits, "d": N}
        else:
            quantized[name] = t.to(torch.float16)
            meta[name] = "p"
    return quantized, meta

def turbo_deserialize(quantized: Dict, meta: Dict,
                       template: Dict[str, Tensor], seed: int = 42) -> Dict[str, Tensor]:
    """Dequantize TurboQuant state dict."""
    out = {}
    for name, orig in template.items():
        info = meta.get(name)
        if info is None:
            continue
        dtype = orig.dtype
        if info in ("p", "c"):
            t = quantized[name]
            out[name] = t.to(dtype) if t.dtype != dtype else t
            continue
        if isinstance(info, dict):
            bits, dim = info["b"], info["d"]
            rot = _turbo_get_rotation(dim, seed)
            cb = _turbo_get_codebook(bits, dim)
            indices = quantized[name + ".q"]
            norms = quantized[name + ".s"]
            if indices.ndim == 3:
                B = indices.shape[0]
                slices = []
                for b in range(B):
                    y_hat = cb[indices[b].long()]
                    slices.append(y_hat @ rot * norms[b].float())
                out[name] = torch.stack(slices).to(dtype)
            else:
                y_hat = cb[indices.long()]
                out[name] = (y_hat @ rot * norms.float()).to(dtype)
    return out

def turbo_compress_model(state_dict: Dict[str, Tensor], seed: int = 42) -> bytes:
    """Full pipeline: TurboQuant quantize -> torch.save -> LZMA compress."""
    quantized, meta = turbo_serialize(state_dict, seed)
    buf = io.BytesIO()
    torch.save({"w": quantized, "m": meta, "s": seed}, buf)
    return lzma.compress(buf.getvalue(), preset=6)

def turbo_decompress_model(blob: bytes, template: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Full pipeline: LZMA decompress -> torch.load -> TurboQuant dequantize."""
    data = torch.load(io.BytesIO(lzma.decompress(blob)), map_location="cpu", weights_only=False)
    return turbo_deserialize(data["w"], data["m"], template, data["s"])

# =============================================================================
# End TurboQuant
# =============================================================================

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_tokens_limit = int(os.environ.get("VAL_TOKENS_LIMIT", 0))
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
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 576))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.5))
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
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    lawa_enabled = bool(int(os.environ.get("LAWA_ENABLED", "0")))
    lawa_k = int(os.environ.get("LAWA_K", 10))
    lawa_freq = int(os.environ.get("LAWA_FREQ", 100))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))
    activation_mode = os.environ.get("ACTIVATION_MODE", "leaky_relu_sq")
    activation_neg_slope = float(os.environ.get("ACTIVATION_NEG_SLOPE", 0.5))
    asymmetric_square_init = float(os.environ.get("ASYMMETRIC_SQUARE_INIT", 0.25))
    gated_square_beta_init = float(os.environ.get("GATED_SQUARE_BETA_INIT", 1.0))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    # N-gram eval cache
    ngram_enabled = bool(int(os.environ.get("NGRAM_ENABLED", "1")))
    ngram_min_order = int(os.environ.get("NGRAM_MIN_ORDER", 2))
    ngram_max_order = int(os.environ.get("NGRAM_MAX_ORDER", 12))
    ngram_num_buckets = int(os.environ.get("NGRAM_NUM_BUCKETS", 16_777_216))  # 16M
    ngram_chunk_size = int(os.environ.get("NGRAM_CHUNK_SIZE", 512))
    ngram_alpha_min = float(os.environ.get("NGRAM_ALPHA_MIN", 0.05))
    ngram_alpha_max = float(os.environ.get("NGRAM_ALPHA_MAX", 0.70))
    ngram_entropy_center = float(os.environ.get("NGRAM_ENTROPY_CENTER", 3.0))
    ngram_entropy_scale = float(os.environ.get("NGRAM_ENTROPY_SCALE", 2.0))
    ngram_min_count = int(os.environ.get("NGRAM_MIN_COUNT", 2))
    ngram_mode = os.environ.get("NGRAM_MODE", "two_pass")  # "single_pass" or "two_pass"
    ngram_eval_chunk_tokens = int(os.environ.get("NGRAM_EVAL_CHUNK_TOKENS", 262144))
    # Complementary training
    complement_enabled = bool(int(os.environ.get("COMPLEMENT_ENABLED", "0")))
    complement_alpha = float(os.environ.get("COMPLEMENT_ALPHA", 0.5))

# --- Batched Newton-Schulz orthogonalization ---

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Batched Newton-Schulz orthogonalization. G: (B,M,N) or (M,N)."""
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

# --- Parallel Muon optimizer ---

class Muon(torch.optim.Optimizer):
    """Parallel Muon: post-backward reduce-scatter -> local NS5 -> all-gather.

    No DDP for bank params. After backward, this optimizer:
    1. Launches async reduce-scatter for all banks (biggest first)
    2. Returns control so Adam can step on small params while RS is in-flight
    3. Waits for each RS, runs local NS5 on the shard, launches async all-gather
    4. Each all-gather overlaps with next bank's NS5
    """
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
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
                    'p': p,
                    'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        # Sort by size descending -- launch biggest reduce-scatters first
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        """Phase 1: launch async reduce-scatter for all banks. Call right after backward."""
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']:
                pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        """Phase 3: wait for RS, local NS5, all-gather. Call AFTER Adam steps."""
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

            sharded = self._distributed and hasattr(self, '_rs_futures')

            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None:
                    continue

                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m['shard']
                    buf = m['shard_mom']
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                update = zeropower_via_newtonschulz5(update, steps=backend_steps)

                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])

            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

            if hasattr(self, '_rs_futures'):
                del self._rs_futures

        return loss

# --- Tokenizer evaluation helpers ---

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
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )
def load_validation_tokens(pattern: str, seq_len: int, token_limit: int = 0) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    if token_limit > 0:
        tokens = tokens[: min(tokens.numel(), token_limit + 1)]
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
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
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
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
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

# --- Quantization helpers ---

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda",
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
def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
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
        "__quant_format__": "int8_clean_per_row_v1",
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

# --- Data loading ---

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

# --- Transformer modules ---

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
@torch.compiler.disable
def _turbo_qat_forward(w: Tensor, x_dtype, bits: int, device) -> Tensor:
    """TurboQuant STE — runs outside torch.compile to avoid dynamo issues."""
    rotation = _turbo_get_rotation(w.shape[1], seed=42, device=device)
    codebook = _turbo_cached_cb(bits, w.shape[1], device)
    with torch.no_grad():
        w_q = turbo_ste(w.float(), rotation, codebook).to(x_dtype)
    return w + (w_q - w).detach()

class CastedLinear(nn.Linear):
    _qat_enabled: bool = False  # Legacy flag (unused with TurboQuant)
    def forward(self, x: Tensor) -> Tensor:
        global _turbo_qat_enabled, _turbo_scheduler
        w = self.weight.to(x.dtype)
        if _turbo_qat_enabled and _turbo_scheduler.enabled and self.training and w.ndim == 2:
            w = _turbo_qat_forward(w, x.dtype, _turbo_scheduler.bits, w.device)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
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
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
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
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        gated_attention: bool = False,
        value_residual: bool = False,
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
        # No CastedLinear -- weights come from banks
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False  # set by GPT.__init__ for deep layers only
        # Gated attention and value residual (non-banked small params)
        self.gated_attention = gated_attention
        if gated_attention:
            self.attn_gate = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)
        self.value_residual = value_residual
        if value_residual:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Efficient XSA: subtract self-value projection via GQA-aware reshape (no repeat_interleave).
        y: [B, T, H, D], v: [B, T, Hkv, D]. H must be divisible by Hkv."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
        vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D] -- broadcast ready
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, v_embed: Tensor | None = None, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v if self.value_residual else None
        if self.value_residual and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if _HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            # SDP fallback: expand KV heads to match Q heads for compatibility
            qt = q.transpose(1, 2)  # (B, H_q, T, D)
            kt = k.transpose(1, 2)  # (B, H_kv, T, D)
            vt = v.transpose(1, 2)
            if kt.shape[1] != qt.shape[1]:
                rep = qt.shape[1] // kt.shape[1]
                kt = kt.repeat_interleave(rep, dim=1)
                vt = vt.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True).transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.gated_attention:
            # gate shape: (bsz, seqlen, num_heads) -> (bsz, seqlen, num_heads, 1) for B,T,H,D layout
            gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
            y = y * gate
        y = y.reshape(bsz, seqlen, dim)
        return F.linear(y, out_w.to(x.dtype)), raw_v

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers.
    Each table maps vocab tokens to a low-dim embedding, projected to model_dim."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_mult: int,
        activation_mode: str = "leaky_relu_sq",
        activation_neg_slope: float = 0.5,
        asymmetric_square_init: float = 0.25,
        gated_square_beta_init: float = 1.0,
    ):
        super().__init__()
        # No CastedLinear -- weights come from banks
        self.activation_mode = activation_mode
        self.activation_neg_slope = activation_neg_slope
        if activation_mode == "asymmetric_square":
            self.neg_sq_scale = nn.Parameter(torch.tensor(asymmetric_square_init, dtype=torch.float32))
        else:
            self.neg_sq_scale = None
        if activation_mode == "gated_square":
            self.gated_square_beta = nn.Parameter(torch.tensor(gated_square_beta_init, dtype=torch.float32))
        else:
            self.gated_square_beta = None
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        u = F.linear(x, up_w.to(x.dtype))
        if self.activation_mode == "leaky_relu_sq":
            h = F.leaky_relu(u, negative_slope=self.activation_neg_slope).square()
        elif self.activation_mode == "asymmetric_square":
            neg_sq_scale = self.neg_sq_scale.to(dtype=u.dtype).clamp(0.0, 4.0)
            h = F.relu(u).square() + neg_sq_scale * F.relu(-u).square()
        elif self.activation_mode == "gated_square":
            beta = self.gated_square_beta.to(dtype=u.dtype).clamp(0.0, 8.0)
            h = u.square() * torch.sigmoid(beta * u)
        elif self.activation_mode == "sign_preserving_square":
            h = u * u.abs()
        else:
            raise ValueError(f"Unknown ACTIVATION_MODE={self.activation_mode}")
        return F.linear(h, down_w.to(x.dtype))

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        gated_attention: bool = False,
        value_residual: bool = False,
        activation_mode: str = "leaky_relu_sq",
        activation_neg_slope: float = 0.5,
        asymmetric_square_init: float = 0.25,
        gated_square_beta_init: float = 1.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        gated_attention=gated_attention, value_residual=value_residual)
        self.mlp = MLP(
            dim,
            mlp_mult,
            activation_mode=activation_mode,
            activation_neg_slope=activation_neg_slope,
            asymmetric_square_init=asymmetric_square_init,
            gated_square_beta_init=gated_square_beta_init,
        )
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None
    def forward(self, x: Tensor, x0: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, up_w: Tensor, down_w: Tensor, v_embed: Tensor | None = None, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, raw_v = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, q_w, k_w, v_w, out_w, v_embed=v_embed, v0=v0)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out, raw_v

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
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.1,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
        gated_attention: bool = False,
        value_residual: bool = False,
        activation_mode: str = "leaky_relu_sq",
        activation_neg_slope: float = 0.5,
        asymmetric_square_init: float = 0.25,
        gated_square_beta_init: float = 1.0,
    ):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)  # kv_dim for value projection
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.value_residual = value_residual
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        # Parameter banks: contiguous 3D tensors for batched optimizer
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)
        self.num_layers = num_layers
        self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                    ln_scale=ln_scale,
                    dtg=dtg,
                    gated_attention=gated_attention,
                    value_residual=value_residual,
                    activation_mode=activation_mode,
                    activation_neg_slope=activation_neg_slope,
                    asymmetric_square_init=asymmetric_square_init,
                    gated_square_beta_init=gated_square_beta_init,
                )
                for i in range(num_layers)
            ]
        )
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim_ve = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()  # keep empty for compat
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
        )
        for head in self.mtp_heads:
            head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        # Init banks: orthogonal, with proj layers scaled down and out/down zero-init
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)        # Q
            nn.init.zeros_(self.qo_bank.data[n + i])                    # Out (zero init)
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)        # K
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)    # V
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)    # MLP up
            nn.init.zeros_(self.mlp_down_bank.data[i])                  # MLP down (zero init)
            # Scale proj layers (out_proj and mlp_down are "proj" layers)
            self.qo_bank.data[n + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)
        # Init remaining nn.Linear modules (bigram proj, mtp heads, lm_head)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        """Get value embedding for a specific layer using shared table + per-layer scale."""
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x, _ = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, v0=v0)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_targets = target_ids[:, k + 1 :].reshape(-1)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x, _ = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, v0=v0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

# --- Sliding window evaluation ---

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
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
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
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=False)
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
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, log0=print,
) -> tuple[float, float]:
    """Legal score-first TTT (PR #461 recipe): score each chunk with sliding windows,
    then train on it. Every token scored BEFORE any update that could use it."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    # Pre-compute all window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]

    # Assign each window to a chunk based on the first token it scores
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

    # Freeze first N blocks
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

        # --- Phase 1: SCORE this chunk's windows (inference_mode) ---
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

        # --- Phase 2: TRAIN on this chunk (already scored = legal) ---
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


# === N-GRAM EVAL CACHE + TWO-PASS RESCORE ===

_NGRAM_PRIMES = np.array([
    36313, 27191, 51647, 81929, 131071, 174763, 233017, 283721,
    347237, 411527, 479909, 557927, 646333, 746773, 862319, 992353,
], dtype=np.int64)

# Per-order multipliers: orders 2-3 suppressed, 4 near-neutral, 5-12 boosted
_ORDER_MULTS = np.array([
    0.30, 0.30, 0.97, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
], dtype=np.float32)


class NgramCache:
    """Hash-table n-gram cache with vectorized numpy operations."""

    def __init__(self, min_order: int = 2, max_order: int = 16,
                 num_buckets: int = 16_777_216):
        self.min_order = min_order
        self.max_order = max_order
        self.num_orders = max_order - min_order + 1
        self.num_buckets = num_buckets
        self.bucket_mask = np.int64(num_buckets - 1)
        # Two flat hash tables per order: context counts and full (context+target) counts
        self.ctx_tables = [np.zeros(num_buckets, dtype=np.int32) for _ in range(self.num_orders)]
        self.full_tables = [np.zeros(num_buckets, dtype=np.int32) for _ in range(self.num_orders)]

    def _compute_hashes(self, tokens_np: np.ndarray, start: int, end: int, order_idx: int):
        """Compute context and full hashes for positions [start, end) at given order."""
        n = self.min_order + order_idx
        valid_start = max(start, n - 1)
        N = end - valid_start
        if N <= 0:
            return None, None, valid_start
        # Context hash: XOR of tokens[pos-n+1+k] * primes[k] for k=0..n-2
        h = np.zeros(N, dtype=np.int64)
        for k in range(n - 1):
            offset = valid_start - (n - 1) + k
            h ^= tokens_np[offset:offset + N].astype(np.int64) * _NGRAM_PRIMES[k % len(_NGRAM_PRIMES)]
        ctx_h = h & self.bucket_mask
        # Full hash: context + target token
        target_prime = _NGRAM_PRIMES[min(n - 1, len(_NGRAM_PRIMES) - 1)]
        full_h = (h ^ (tokens_np[valid_start:end].astype(np.int64) * target_prime)) & self.bucket_mask
        return ctx_h, full_h, valid_start

    def _bincount_add(self, table: np.ndarray, indices: np.ndarray):
        """Fast histogram accumulation using np.bincount (much faster than np.add.at)."""
        counts = np.bincount(indices.astype(np.intp), minlength=self.num_buckets)
        table += counts[:self.num_buckets].astype(table.dtype)

    def update_range(self, tokens_np: np.ndarray, start: int, end: int):
        """Add tokens[start:end] to the cache for all orders."""
        for oi in range(self.num_orders):
            ctx_h, full_h, vs = self._compute_hashes(tokens_np, start, end, oi)
            if ctx_h is None:
                continue
            self._bincount_add(self.ctx_tables[oi], ctx_h)
            self._bincount_add(self.full_tables[oi], full_h)

    def build_full(self, tokens_np: np.ndarray):
        """Build complete cache from entire token sequence (vectorized)."""
        for oi in range(self.num_orders):
            ctx_h, full_h, _ = self._compute_hashes(tokens_np, 0, len(tokens_np), oi)
            if ctx_h is None:
                continue
            self._bincount_add(self.ctx_tables[oi], ctx_h)
            self._bincount_add(self.full_tables[oi], full_h)

    def score_range(self, tokens_np: np.ndarray, start: int, end: int,
                    min_count: int = 2):
        """Score tokens[start:end] against the cache.

        Returns:
            ngram_prob: (N,) float32 - n-gram probability for the true target token
            matched_order: (N,) int32 - which order matched (-1 = no match)
        """
        N = end - start
        ngram_prob = np.zeros(N, dtype=np.float32)
        matched_order = np.full(N, -1, dtype=np.int32)
        matched = np.zeros(N, dtype=bool)

        # Backoff from highest to lowest order
        for oi in range(self.num_orders - 1, -1, -1):
            n = self.min_order + oi
            ctx_h, full_h, vs = self._compute_hashes(tokens_np, start, end, oi)
            if ctx_h is None:
                continue
            offset = vs - start
            ctx_counts = self.ctx_tables[oi][ctx_h]
            full_counts = self.full_tables[oi][full_h]
            # Cap full counts to context counts (hash collision mitigation)
            full_counts = np.minimum(full_counts, ctx_counts)
            # Only match when: sufficient context, target has been seen, not already matched
            eligible = (ctx_counts >= min_count) & (full_counts > 0) & ~matched[offset:]
            if not np.any(eligible):
                continue
            prob = full_counts[eligible].astype(np.float32) / np.maximum(ctx_counts[eligible].astype(np.float32), 1.0)
            # Find which positions in the output array to fill
            out_idx = np.where(eligible)[0] + offset
            ngram_prob[out_idx] = prob
            matched_order[out_idx] = n
            matched[out_idx] = True

        return ngram_prob, matched_order


def eval_val_sliding_store(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, log0=print,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Sliding-window eval that stores per-token model_p and entropy.

    Returns: (model_p, entropy, token_bytes, token_targets, val_loss, val_bpb)
    where model_p and entropy are arrays covering this rank's scored tokens,
    and val_loss/val_bpb are the standard (un-blended) metrics.

    Also returns global-offset index arrays for mapping back to token positions.
    """
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    # Pre-allocate per-token storage (we'll trim later)
    # Each token is scored in exactly one window
    model_p_list: list[np.ndarray] = []
    entropy_list: list[np.ndarray] = []
    bytes_list: list[np.ndarray] = []
    position_list: list[np.ndarray] = []  # global target-token positions
    nll_list: list[np.ndarray] = []

    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=False)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end_pos = min(ws + seq_len, total_tokens)
                wlen = end_pos - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end_pos + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)  # (bsz, seq_len, vocab_size)
            # Compute per-token quantities
            logits_f = logits.float()
            log_probs = F.log_softmax(logits_f, dim=-1)  # (bsz, seq_len, V)
            probs = log_probs.exp()
            # NLL for each token
            nll_all = F.cross_entropy(
                logits_f.reshape(-1, logits_f.size(-1)),
                y_batch.reshape(-1), reduction="none"
            ).reshape(bsz, seq_len)
            # Model probability of true token
            mp = probs.gather(2, y_batch.unsqueeze(-1)).squeeze(-1)  # (bsz, seq_len)
            # Entropy of model distribution
            ent = -(probs * log_probs).sum(dim=-1)  # (bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                # Positions are TARGET token indices in val_tokens (ws+j+1 for scored position j)
                positions = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
                position_list.append(positions)
                model_p_list.append(mp[i, s:wlen].cpu().numpy().astype(np.float32))
                entropy_list.append(ent[i, s:wlen].cpu().numpy().astype(np.float32))
                nll_list.append(nll_all[i, s:wlen].cpu().numpy().astype(np.float64))
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                bytes_list.append(tb.cpu().numpy())

    all_positions = np.concatenate(position_list) if position_list else np.array([], dtype=np.int64)
    all_model_p = np.concatenate(model_p_list) if model_p_list else np.array([], dtype=np.float32)
    all_entropy = np.concatenate(entropy_list) if entropy_list else np.array([], dtype=np.float32)
    all_nll = np.concatenate(nll_list) if nll_list else np.array([], dtype=np.float64)
    all_bytes = np.concatenate(bytes_list) if bytes_list else np.array([], dtype=np.float64)


    # Compute standard (un-blended) BPB for this rank
    local_loss_sum = all_nll.sum()
    local_token_count = float(len(all_nll))
    local_byte_count = all_bytes.sum()

    # All-reduce for standard BPB
    loss_sum_t = torch.tensor(local_loss_sum, device=device, dtype=torch.float64)
    token_count_t = torch.tensor(local_token_count, device=device, dtype=torch.float64)
    byte_count_t = torch.tensor(local_byte_count, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count_t, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum_t / token_count_t).item()
    val_bpb = val_loss / math.log(2.0) * (token_count_t.item() / byte_count_t.item())

    base_model.train()
    return all_model_p, all_entropy, all_bytes, all_positions, val_loss, val_bpb


def ngram_rescore(
    args: Hyperparameters,
    tokens_np: np.ndarray,
    cache: NgramCache,
    model_p: np.ndarray,
    entropy: np.ndarray,
    token_bytes: np.ndarray,
    positions: np.ndarray,
    rank: int, world_size: int, device: torch.device,
    log0=print,
) -> tuple[float, float]:
    """Rescore tokens using n-gram cache blended with stored neural model_p.

    This is Pass 2: the cache is already complete.
    Returns: (val_loss, val_bpb)
    """
    N = len(positions)
    if N == 0:
        return 0.0, 0.0

    # Score all of this rank's positions against the full cache
    # We need to score at the GLOBAL token positions
    # The cache.score_range expects contiguous ranges, but our positions may be sparse
    # Instead, we score the full range and index into it
    # Actually, positions are sorted (from sliding windows), so we can score chunks

    # Score the full token range (0 to len(tokens_np)) and pick our positions.
    # Position p in the n-gram means: predict tokens_np[p] given context.
    # positions from sliding-window are target-token indices into val_tokens.
    ngram_prob_all, matched_order_all = cache.score_range(
        tokens_np, 0, len(tokens_np), min_count=args.ngram_min_count
    )

    # Pick our positions (guaranteed in [1, len(tokens_np)-1])
    ngram_prob = ngram_prob_all[positions]
    matched_order = matched_order_all[positions]
    matched = matched_order >= 0

    # Entropy-adaptive alpha with per-order multipliers
    alpha = np.zeros(N, dtype=np.float32)
    if np.any(matched):
        order_idx = (matched_order[matched] - cache.min_order).astype(np.int32)
        centers = args.ngram_entropy_center - 0.25 * order_idx.astype(np.float32)
        sig = 1.0 / (1.0 + np.exp(-args.ngram_entropy_scale * (entropy[matched] - centers)))
        raw_alpha = args.ngram_alpha_min + (args.ngram_alpha_max - args.ngram_alpha_min) * sig
        # Per-order multipliers
        mults = _ORDER_MULTS[np.minimum(order_idx, len(_ORDER_MULTS) - 1)]
        raw_alpha *= mults
        alpha[matched] = np.clip(raw_alpha, 0.0, 0.95)

    # Blend: p_blend = (1 - alpha) * model_p + alpha * ngram_prob
    p_blend = (1.0 - alpha) * model_p + alpha * ngram_prob
    # Clamp to avoid log(0)
    p_blend = np.maximum(p_blend, 1e-10)
    # For unmatched tokens, use model_p directly
    p_blend[~matched] = np.maximum(model_p[~matched], 1e-10)

    # NLL
    nll = -np.log(p_blend).astype(np.float64)

    # Aggregate
    local_loss_sum = nll.sum()
    local_token_count = float(N)
    local_byte_count = token_bytes.sum()

    # All-reduce
    loss_sum_t = torch.tensor(local_loss_sum, device=device, dtype=torch.float64)
    token_count_t = torch.tensor(local_token_count, device=device, dtype=torch.float64)
    byte_count_t = torch.tensor(local_byte_count, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count_t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum_t / token_count_t).item()
    val_bpb = val_loss / math.log(2.0) * (token_count_t.item() / byte_count_t.item())

    n_matched = int(matched.sum())
    log0(f"ngram_rescore: matched={n_matched}/{N} ({100*n_matched/max(N,1):.1f}%) "
         f"mean_alpha={alpha[matched].mean():.3f}" if n_matched > 0 else "ngram_rescore: no matches")

    return val_loss, val_bpb


def eval_ngram_two_pass(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, log0=print,
) -> tuple[float, float]:
    """Two-pass n-gram evaluation.

    Pass 1: Sliding-window neural eval → store per-token model_p and entropy.
    Build:  Complete n-gram cache from all tokens (vectorized).
    Pass 2: Rescore ALL tokens by blending neural model_p with n-gram predictions.
    """
    t0 = time.perf_counter()

    # --- Pass 1: Neural eval with per-token storage ---
    log0(f"ngram_two_pass: starting Pass 1 (sliding-window neural eval)")
    model_p, entropy, token_bytes, positions, pass1_loss, pass1_bpb = eval_val_sliding_store(
        args, base_model, rank, world_size, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, batch_seqs=batch_seqs, log0=log0,
    )
    t_pass1 = time.perf_counter()
    log0(f"ngram_two_pass: Pass 1 done val_bpb={pass1_bpb:.6f} "
         f"tokens_scored={len(positions)} time={t_pass1 - t0:.1f}s")

    # --- Build complete n-gram cache ---
    log0(f"ngram_two_pass: building cache orders={args.ngram_min_order}-{args.ngram_max_order} "
         f"buckets={args.ngram_num_buckets}")
    tokens_np = val_tokens.numpy().astype(np.int16)
    cache = NgramCache(
        min_order=args.ngram_min_order,
        max_order=args.ngram_max_order,
        num_buckets=args.ngram_num_buckets,
    )
    cache.build_full(tokens_np)
    t_cache = time.perf_counter()
    log0(f"ngram_two_pass: cache built in {t_cache - t_pass1:.1f}s")

    # --- Pass 2: N-gram rescore ---
    log0(f"ngram_two_pass: starting Pass 2 (n-gram rescore)")
    val_loss, val_bpb = ngram_rescore(
        args, tokens_np, cache, model_p, entropy, token_bytes, positions,
        rank, world_size, device, log0=log0,
    )
    t_pass2 = time.perf_counter()
    log0(f"ngram_two_pass: Pass 2 done val_bpb={val_bpb:.6f} "
         f"improvement={pass1_bpb - val_bpb:.6f} time={t_pass2 - t_cache:.1f}s")
    log0(f"ngram_two_pass: total time={t_pass2 - t0:.1f}s")

    return val_loss, val_bpb


def eval_ngram_single_pass(
    args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    batch_seqs=32, log0=print,
) -> tuple[float, float]:
    """Single-pass incremental n-gram eval (legally safe — no self-inclusion).

    Processes validation tokens in chunks. For each chunk:
      1. Score chunk tokens with the neural model (simple chunk-based forward).
      2. Score each token against the CURRENT n-gram cache (which does NOT yet
         contain this chunk) — backward-looking only.
      3. Blend neural model_p with n-gram probability using entropy-adaptive alpha.
      4. Accumulate loss, token count, byte count.
      5. Update the cache with this chunk's tokens (score-first guarantee).

    All ranks process the same chunks in the same order, so the cache stays
    identical across ranks. Each rank scores its own subset of tokens within
    each chunk.
    """
    t0 = time.perf_counter()
    seq_len = args.train_seq_len
    chunk_tokens = args.ngram_eval_chunk_tokens
    tokens_np = val_tokens.numpy().astype(np.int16)
    total_tokens = val_tokens.numel() - 1  # -1 because we predict next token

    # Build chunk boundaries (all ranks use the same chunks)
    chunk_starts = list(range(0, total_tokens, chunk_tokens))
    num_chunks = len(chunk_starts)

    log0(f"ngram_single_pass: {num_chunks} chunks of {chunk_tokens} tokens, "
         f"total={total_tokens}, seq_len={seq_len}")

    # Initialize empty cache (builds incrementally)
    cache = NgramCache(
        min_order=args.ngram_min_order,
        max_order=args.ngram_max_order,
        num_buckets=args.ngram_num_buckets,
    )

    # Accumulators
    total_loss_sum = 0.0
    total_token_count = 0.0
    total_byte_count = 0.0
    total_matched = 0
    total_scored = 0
    alpha_sum = 0.0
    alpha_count = 0

    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=False)

    with torch.inference_mode():
        for ci, c_start in enumerate(chunk_starts):
            c_end = min(c_start + chunk_tokens, total_tokens)
            chunk_len = c_end - c_start  # number of target tokens in this chunk

            if chunk_len <= 0:
                continue

            # --- Step 1: Neural model scoring for this chunk ---
            # Target tokens are at positions c_start+1 .. c_end in val_tokens
            # (predicting val_tokens[c_start+1] from context starting at some point)
            # We process in windows of seq_len within the chunk.
            # Each window: input = val_tokens[ws:ws+seq_len], target = val_tokens[ws+1:ws+seq_len+1]
            # We score positions that fall within this chunk only.

            # Build windows covering this chunk's target positions
            # Target position p means predicting val_tokens[p] given val_tokens[..p-1]
            # We need windows whose scored region covers [c_start+1, c_end]
            # A window starting at ws scores targets ws+1..ws+seq_len
            # For coverage of target c_start+1, we need ws <= c_start
            # Use non-overlapping windows within the chunk for simplicity
            windows = []
            ws = c_start
            while ws < c_end:
                w_end = min(ws + seq_len, total_tokens)
                if w_end > ws:
                    windows.append(ws)
                ws += seq_len

            # Distribute windows across ranks
            my_s = (len(windows) * rank) // world_size
            my_e = (len(windows) * (rank + 1)) // world_size
            my_windows = windows[my_s:my_e]

            # Per-token arrays for this rank's portion of the chunk
            chunk_model_p = []
            chunk_entropy = []
            chunk_nll = []
            chunk_bytes = []
            chunk_positions = []  # global target positions

            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end_pos = min(ws + seq_len, total_tokens)
                    wlen = end_pos - ws
                    wlens.append(wlen)
                    chunk_data = val_tokens[ws:end_pos + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_data[:-1]
                    y_batch[i, :wlen] = chunk_data[1:]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)

                logits_f = logits.float()
                log_probs = F.log_softmax(logits_f, dim=-1)
                probs = log_probs.exp()
                nll_all = F.cross_entropy(
                    logits_f.reshape(-1, logits_f.size(-1)),
                    y_batch.reshape(-1), reduction="none"
                ).reshape(bsz, seq_len)
                mp = probs.gather(2, y_batch.unsqueeze(-1)).squeeze(-1)
                ent = -(probs * log_probs).sum(dim=-1)

                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    # Score all positions in this window (no stride overlap handling
                    # needed since we use non-overlapping windows)
                    # Target positions: ws+1 .. ws+wlen (global token indices)
                    positions = np.arange(ws + 1, ws + wlen + 1, dtype=np.int64)

                    # Only keep positions within this chunk's range [c_start+1, c_end]
                    mask = (positions >= c_start + 1) & (positions <= c_end)
                    if not np.any(mask):
                        continue
                    local_idx = np.where(mask)[0]
                    positions = positions[mask]

                    chunk_positions.append(positions)
                    chunk_model_p.append(mp[i, local_idx].cpu().numpy().astype(np.float32))
                    chunk_entropy.append(ent[i, local_idx].cpu().numpy().astype(np.float32))
                    chunk_nll.append(nll_all[i, local_idx].cpu().numpy().astype(np.float64))

                    tgt = y_batch[i, local_idx]
                    prev = x_batch[i, local_idx]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    chunk_bytes.append(tb.cpu().numpy())

            # Concatenate this rank's chunk results
            if chunk_positions:
                all_pos = np.concatenate(chunk_positions)
                all_mp = np.concatenate(chunk_model_p)
                all_ent = np.concatenate(chunk_entropy)
                all_nll = np.concatenate(chunk_nll)
                all_tb = np.concatenate(chunk_bytes)
            else:
                all_pos = np.array([], dtype=np.int64)
                all_mp = np.array([], dtype=np.float32)
                all_ent = np.array([], dtype=np.float32)
                all_nll = np.array([], dtype=np.float64)
                all_tb = np.array([], dtype=np.float64)

            N = len(all_pos)

            # --- Step 2: N-gram scoring from CURRENT cache (before update) ---
            if N > 0 and ci > 0:
                # Score this rank's positions against the cache
                # Use score_range over the full token array with the chunk bounds
                # But score_range returns results indexed from start, so we need
                # to score a contiguous range and pick our positions
                ngram_prob_chunk, matched_order_chunk = cache.score_range(
                    tokens_np, c_start + 1, c_end + 1,
                    min_count=args.ngram_min_count,
                )
                # Map our positions to indices within the score_range output
                # score_range(tokens_np, c_start+1, c_end+1) returns array of
                # length (c_end+1) - (c_start+1) = c_end - c_start = chunk_len
                # Index i corresponds to global position c_start+1+i
                local_idx = (all_pos - (c_start + 1)).astype(np.intp)
                # Bounds check
                valid = (local_idx >= 0) & (local_idx < len(ngram_prob_chunk))
                ngram_prob = np.zeros(N, dtype=np.float32)
                matched_order = np.full(N, -1, dtype=np.int32)
                if np.any(valid):
                    ngram_prob[valid] = ngram_prob_chunk[local_idx[valid]]
                    matched_order[valid] = matched_order_chunk[local_idx[valid]]
            else:
                ngram_prob = np.zeros(N, dtype=np.float32)
                matched_order = np.full(N, -1, dtype=np.int32)

            # --- Step 3: Blend neural + n-gram ---
            if N > 0:
                matched = matched_order >= 0
                alpha = np.zeros(N, dtype=np.float32)
                if np.any(matched):
                    order_idx = (matched_order[matched] - cache.min_order).astype(np.int32)
                    centers = args.ngram_entropy_center - 0.25 * order_idx.astype(np.float32)
                    sig = 1.0 / (1.0 + np.exp(-args.ngram_entropy_scale * (all_ent[matched] - centers)))
                    raw_alpha = args.ngram_alpha_min + (args.ngram_alpha_max - args.ngram_alpha_min) * sig
                    mults = _ORDER_MULTS[np.minimum(order_idx, len(_ORDER_MULTS) - 1)]
                    raw_alpha *= mults
                    alpha[matched] = np.clip(raw_alpha, 0.0, 0.95)

                p_blend = (1.0 - alpha) * all_mp + alpha * ngram_prob
                p_blend = np.maximum(p_blend, 1e-10)
                p_blend[~matched] = np.maximum(all_mp[~matched], 1e-10)

                nll_blend = -np.log(p_blend).astype(np.float64)

                total_loss_sum += nll_blend.sum()
                total_token_count += float(N)
                total_byte_count += all_tb.sum()
                n_matched = int(matched.sum())
                total_matched += n_matched
                total_scored += N
                if n_matched > 0:
                    alpha_sum += float(alpha[matched].sum())
                    alpha_count += n_matched

            # --- Step 5: Update cache with this chunk (ALL ranks, same update) ---
            # Update range: target positions c_start+1 .. c_end, but update_range
            # adds n-grams for tokens[start:end], so we update the chunk range
            cache.update_range(tokens_np, c_start, c_end + 1)

            if ci % max(1, num_chunks // 5) == 0 or ci == num_chunks - 1:
                log0(f"ngram_single_pass: chunk {ci+1}/{num_chunks} "
                     f"scored={total_scored} matched={total_matched}")

    # --- All-reduce across ranks ---
    loss_sum_t = torch.tensor(total_loss_sum, device=device, dtype=torch.float64)
    token_count_t = torch.tensor(total_token_count, device=device, dtype=torch.float64)
    byte_count_t = torch.tensor(total_byte_count, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count_t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum_t / token_count_t).item()
    val_bpb = val_loss / math.log(2.0) * (token_count_t.item() / byte_count_t.item())

    t_total = time.perf_counter() - t0
    mean_alpha = alpha_sum / max(alpha_count, 1)
    log0(f"ngram_single_pass: done val_bpb={val_bpb:.6f} "
         f"matched={total_matched}/{total_scored} ({100*total_matched/max(total_scored,1):.1f}%) "
         f"mean_alpha={mean_alpha:.3f} time={t_total:.1f}s")

    base_model.train()
    return val_loss, val_bpb


# === COMPLEMENTARY TRAINING ===

class TrainBigramTracker:
    """Tracks bigram statistics from training data for complementary loss weighting."""

    def __init__(self, vocab_size: int, device: torch.device):
        # bigram_counts[prev_token, target_token] = count
        self.counts = torch.zeros(vocab_size, vocab_size, device=device, dtype=torch.float32)
        self.row_totals = torch.zeros(vocab_size, device=device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor):
        """Update bigram counts. x: context tokens, y: target tokens."""
        prev = x.reshape(-1)
        tgt = y.reshape(-1)
        idx = prev.long() * self.counts.shape[1] + tgt.long()
        self.counts.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        self.row_totals.scatter_add_(0, prev.long(), torch.ones(prev.shape[0], device=prev.device, dtype=torch.float32))

    @torch.no_grad()
    def get_weights(self, x: Tensor, y: Tensor, alpha: float = 0.5) -> Tensor:
        """Compute per-token loss weights: downweight tokens predictable by bigrams."""
        prev = x.reshape(-1)
        tgt = y.reshape(-1)
        totals = self.row_totals[prev.long()]
        counts = self.counts[prev.long(), tgt.long()]
        ngram_prob = counts / totals.clamp(min=1.0)
        weights = (1.0 - alpha * ngram_prob).clamp(min=0.1)
        return weights.reshape(y.shape)


# --- GPTQ-lite int6 quantization ---

def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
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

def _unbank_state_dict(sd: dict[str, Tensor], num_layers: int) -> dict[str, Tensor]:
    """Convert 3D bank tensors into individual 2D tensors with standard names."""
    out: dict[str, Tensor] = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out

def _rebank_state_dict(sd: dict[str, Tensor], num_layers: int, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert individual 2D tensors back into 3D bank tensors."""
    out: dict[str, Tensor] = {}
    n = num_layers
    # Reconstruct banks from individual weight keys
    qo_slices = [None] * (2 * n)
    kv_slices = [None] * (2 * n)
    up_slices = [None] * n
    down_slices = [None] * n
    consumed = set()
    for i in range(n):
        qk = f"blocks.{i}.attn.c_q.weight"
        if qk in sd:
            qo_slices[i] = sd[qk]
            consumed.add(qk)
        ok = f"blocks.{i}.attn.proj.weight"
        if ok in sd:
            qo_slices[n + i] = sd[ok]
            consumed.add(ok)
        kk = f"blocks.{i}.attn.c_k.weight"
        if kk in sd:
            kv_slices[i] = sd[kk]
            consumed.add(kk)
        vk = f"blocks.{i}.attn.c_v.weight"
        if vk in sd:
            kv_slices[n + i] = sd[vk]
            consumed.add(vk)
        fk = f"blocks.{i}.mlp.fc.weight"
        if fk in sd:
            up_slices[i] = sd[fk]
            consumed.add(fk)
        dk = f"blocks.{i}.mlp.proj.weight"
        if dk in sd:
            down_slices[i] = sd[dk]
            consumed.add(dk)
    out["qo_bank"] = torch.stack(qo_slices).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv_slices).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up_slices).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down_slices).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out

def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):
    num_layers_total = max(
        (int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
        default=0,
    ) + 1
    late_k_layers = set(range(num_layers_total - 2, num_layers_total))
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
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
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta
def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

# --- Training ---

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    # zeropower_via_newtonschulz5 runs eagerly with bmm -- do NOT compile
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
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
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
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len, args.val_tokens_limit)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    # TurboQuant: progressive QAT replaces legacy int6 STE
    global _turbo_qat_enabled, _turbo_scheduler
    if args.qat_enabled:
        _turbo_qat_enabled = True
        _turbo_scheduler.enabled = True
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
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
        activation_mode=args.activation_mode,
        activation_neg_slope=args.activation_neg_slope,
        asymmetric_square_init=args.asymmetric_square_init,
        gated_square_beta_init=args.gated_square_beta_init,
    ).to(device).bfloat16()
    # Banks stay FP32 (like CastedLinear weights), cast to BF16 in forward
    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    # No DDP -- Parallel Muon handles bank grad communication via reduce-scatter,
    # and non-bank grads are manually all-reduced before Adam steps.
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model = compiled_model
    # Separate compile for forward_logits (used in complementary training)
    compiled_forward_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=False)

    # Optimizer split:
    # - 4 parameter banks -> Muon (batched Newton-Schulz)
    # - token embedding -> Adam
    # - scalars/control tensors -> Adam
    # - bigram proj, mtp heads, VE proj -> Adam (small matrix params not worth banking)
    matrix_params = [
        base_model.qo_bank, base_model.kv_bank,
        base_model.mlp_up_bank, base_model.mlp_down_bank,
    ]
    block_named_params = list(base_model.blocks.named_parameters())
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            scalar_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            scalar_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    # Non-bank params that need manual all-reduce (replicated across GPUs)
    replicated_params = list(optimizer_tok.param_groups[0]["params"])
    for pg in optimizer_tok.param_groups[1:]:
        replicated_params.extend(pg["params"])
    replicated_params.extend(scalar_params)

    optimizer_head = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        replicated_params.append(base_model.lm_head.weight)
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_head is not None:
        optimizers.append(optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params = sum(p.numel() for p in base_model.mtp_heads.parameters())
    log0(f"model_params:{n_params}")
    log0(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"activation_mode:{args.activation_mode} neg_slope:{args.activation_neg_slope} "
        f"asym_init:{args.asymmetric_square_init} gated_beta_init:{args.gated_square_beta_init}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
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
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            # All-reduce all grads for warmup (simple, not optimized)
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    # Complementary training tracker
    bigram_tracker = TrainBigramTracker(args.vocab_size, device) if args.complement_enabled else None
    if bigram_tracker is not None:
        log0(f"complement:enabled alpha={args.complement_alpha}")
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    from collections import deque
    lawa_queue: deque[dict[str, Tensor]] = deque(maxlen=args.lawa_k)
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = 0.997
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
        # TurboQuant progressive QAT: 4-bit -> 3-bit -> 2-bit during warmdown
        _turbo_scheduler.update(scale)
        if _turbo_scheduler.enabled and not _turbo_qat_enabled:
            _turbo_qat_enabled = True
            log0(f"turbo_qat:enabled step:{step} bits:{_turbo_scheduler.bits} scale:{scale:.4f}")
        elif _turbo_qat_enabled and _turbo_scheduler.enabled:
            pass  # bits update handled by scheduler
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                if args.complement_enabled and bigram_tracker is not None:
                    # Complementary training: single forward, weighted CE
                    logits = compiled_forward_logits(x)
                    logits_flat = logits.reshape(-1, logits.size(-1)).float()
                    per_token_nll = F.cross_entropy(logits_flat, y.reshape(-1), reduction="none")
                    comp_weights = bigram_tracker.get_weights(x, y, alpha=args.complement_alpha).reshape(-1)
                    loss = (per_token_nll * comp_weights).sum() / comp_weights.sum()
                    bigram_tracker.update(x, y)
                else:
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
        # === 3-phase overlapped optimizer step ===
        # Phase 1: Launch async reduce-scatter for banks (biggest first)
        optimizer_muon.launch_reduce_scatters()
        # Phase 2: All-reduce non-bank grads + step Adam (while bank RS is in-flight)
        if distributed:
            for p in replicated_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optimizer_tok.step()
        optimizer_scalar.step()
        if optimizer_head is not None:
            optimizer_head.step()
        # Phase 3: Wait for RS, local NS5, all-gather (banks processed last)
        optimizer_muon.step()
        zero_grad_all()
        # EMA update
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        if args.lawa_enabled and step % args.lawa_freq == 0:
            lawa_queue.append({name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()})
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
    # Apply weight averaging
    if args.lawa_enabled and len(lawa_queue) > 1:
        log0(f"lawa:applying LAWA averaging k={len(lawa_queue)}")
        current_state = base_model.state_dict()
        avg_state = {name: torch.zeros(t.shape, dtype=torch.float32, device='cpu') for name, t in current_state.items()}
        for snap in lawa_queue:
            for name in avg_state:
                avg_state[name] += snap[name].float()
        for name in avg_state:
            avg_state[name] /= len(lawa_queue)
            avg_state[name] = avg_state[name].to(dtype=current_state[name].dtype)
        base_model.load_state_dict(avg_state, strict=True)
    else:
        log0("ema:applying EMA weights")
        current_state = base_model.state_dict()
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(avg_state, strict=True)
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )
    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)
    if excluded_mtp > 0:
        log0(f"export_excluding_mtp_params:{excluded_mtp}")
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    # Disable TurboQuant QAT before eval — not needed post-training
    _turbo_qat_enabled = False
    _turbo_scheduler.enabled = False
    log0("turbo_qat:disabled for eval")
    # TurboQuant serialization (replaces int6/int8 pipeline)
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    quant_blob = turbo_compress_model(sd_cpu)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model turbo+lzma: {quant_file_bytes} bytes")
        log0(f"Total submission size turbo+lzma: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    deq_state = turbo_decompress_model(quant_blob_disk, sd_cpu)
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        gated_attention=args.gated_attention, value_residual=args.value_residual,
        activation_mode=args.activation_mode,
        activation_neg_slope=args.activation_neg_slope,
        asymmetric_square_init=args.asymmetric_square_init,
        gated_square_beta_init=args.gated_square_beta_init,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=False)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize()
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
            f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms"
        )
        log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
    # Legal score-first TTT (PR #461 recipe)
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, log0=log0,
        )
        torch.cuda.synchronize()
        log0(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
             f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")
        log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
    # --- N-gram rescore ---
    if args.ngram_enabled:
        ngram_model = eval_model
        torch.cuda.synchronize()
        t_ngram = time.perf_counter()
        if args.ngram_mode == "single_pass":
            log0(f"ngram: using single_pass mode (chunk_tokens={args.ngram_eval_chunk_tokens})")
            ng_val_loss, ng_val_bpb = eval_ngram_single_pass(
                args, ngram_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                batch_seqs=32, log0=log0,
            )
            torch.cuda.synchronize()
            log0(f"ngram_single_pass val_loss:{ng_val_loss:.4f} val_bpb:{ng_val_bpb:.4f} "
                 f"eval_time:{1000.0 * (time.perf_counter() - t_ngram):.0f}ms")
            log0(f"ngram_single_pass_exact val_loss:{ng_val_loss:.8f} val_bpb:{ng_val_bpb:.8f}")
        else:
            log0(f"ngram: using two_pass mode")
            ng_val_loss, ng_val_bpb = eval_ngram_two_pass(
                args, ngram_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=args.eval_stride, log0=log0,
            )
            torch.cuda.synchronize()
            log0(f"ngram_two_pass val_loss:{ng_val_loss:.4f} val_bpb:{ng_val_bpb:.4f} "
                 f"eval_time:{1000.0 * (time.perf_counter() - t_ngram):.0f}ms")
            log0(f"ngram_two_pass_exact val_loss:{ng_val_loss:.8f} val_bpb:{ng_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{ng_val_loss:.8f} val_bpb:{ng_val_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
