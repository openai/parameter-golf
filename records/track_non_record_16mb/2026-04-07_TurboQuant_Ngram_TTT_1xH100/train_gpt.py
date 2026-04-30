import copy
import glob
import io
import json
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
import torch.nn.functional as _F
def flash_attn_3_func(q, k, v, causal=True, **kw):
    """SDPA wrapper replacing FA3 for non-Hopper GPUs or torch.compile compat."""
    return _F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
        is_causal=causal, enable_gqa=(k.size(2) != q.size(2))
    ).transpose(1,2)
_E=os.environ.get
_sync=torch.cuda.synchronize
_now=time.perf_counter
class HP:
    data_path = _E("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = _E("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = _E("RUN_ID", str(uuid.uuid4()))
    seed = int(_E("SEED", 1337))
    val_batch_size = int(_E("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(_E("VAL_LOSS_EVERY", 4000))
    train_log_every = int(_E("TRAIN_LOG_EVERY", 500))
    iterations = int(_E("ITERATIONS", 20000))
    warmdown_iters = int(_E("WARMDOWN_ITERS", 3500))
    warmup_steps = int(_E("WARMUP_STEPS", 20))
    train_batch_tokens = int(_E("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(_E("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(_E("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(_E("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(_E("QK_GAIN_INIT", 1.5))
    vocab_size = int(_E("VOCAB_SIZE", 1024))
    num_layers = int(_E("NUM_LAYERS", 11))
    num_kv_heads = int(_E("NUM_KV_HEADS", 4))
    model_dim = int(_E("MODEL_DIM", 512))
    num_heads = int(_E("NUM_HEADS", 8))
    mlp_mult = float(_E("MLP_MULT", 3.0))
    tie_embeddings = bool(int(_E("TIE_EMBEDDINGS", "1")))
    rope_base = float(_E("ROPE_BASE", 10000.0))
    logit_softcap = float(_E("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(_E("EMBED_LR", 0.6))
    head_lr = float(_E("HEAD_LR", 0.008))
    tied_embed_lr = float(_E("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(_E("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(_E("MATRIX_LR", 0.025))
    scalar_lr = float(_E("SCALAR_LR", 0.025))
    muon_momentum = float(_E("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(_E("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(_E("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(_E("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(_E("BETA1", 0.9))
    beta2 = float(_E("BETA2", 0.95))
    adam_eps = float(_E("ADAM_EPS", 1e-8))
    grad_clip_norm = float(_E("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(_E("EVAL_STRIDE", 64))
    mtp_num_heads = int(_E("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(_E("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(_E("MUON_BETA2", 0.95))
    muon_wd = float(_E("MUON_WD", 0.04))
    adam_wd = float(_E("ADAM_WD", 0.04))
    qat_enabled = bool(int(_E("QAT_ENABLED", "0")))
    bigram_vocab_size = int(_E("BIGRAM_VOCAB_SIZE", 4096))
    bigram_dim = int(_E("BIGRAM_DIM", 128))
    xsa_last_n = int(_E("XSA_LAST_N", 4))
    rope_dims = int(_E("ROPE_DIMS", 16))
    ln_scale = bool(int(_E("LN_SCALE", "1")))
    dtg_enabled = bool(int(_E("DTG_ENABLED", "0")))
    late_qat_threshold = float(_E("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(_E("VE_ENABLED", "1")))
    ve_dim = int(_E("VE_DIM", 128))
    ve_layers = _E("VE_LAYERS", "9,10")
    vrl_enabled = bool(int(_E("VRL_ENABLED", "1")))
    gated_attn_enabled = bool(int(_E("GATED_ATTN_ENABLED", "1")))
    gated_attn_bias_init = float(_E("GATED_ATTN_BIAS_INIT", 4.0))
    wsd_stable_frac = float(_E("WSD_STABLE_FRAC", 0.75))
    qat_trigger_frac = float(_E("QAT_TRIGGER_FRAC", 0.85))
    prog_seq_enabled = bool(int(_E("PROG_SEQ_ENABLED", "0")))
    prog_seq_phase1_len = int(_E("PROG_SEQ_PHASE1_LEN", 512))
    prog_seq_phase2_len = int(_E("PROG_SEQ_PHASE2_LEN", 1024))
    prog_seq_phase3_len = int(_E("PROG_SEQ_PHASE3_LEN", 2048))
    prog_seq_frac1 = float(_E("PROG_SEQ_FRAC1", 0.33))
    prog_seq_frac2 = float(_E("PROG_SEQ_FRAC2", 0.55))
    gptq_n_samples = int(_E("GPTQ_N_SAMPLES", 256))
    gptq_block_size = int(_E("GPTQ_BLOCK_SIZE", 128))
    gptq_percdamp = float(_E("GPTQ_PERCDAMP", 0.01))
    ttt_epochs = int(_E("TTT_EPOCHS", 1))
    ttt_lr = float(_E("TTT_LR", 0.003))
    ttt_chunk_tokens = int(_E("TTT_CHUNK_TOKENS", 65536))
    ttt_freeze_blocks = int(_E("TTT_FREEZE_BLOCKS", 9))
    ttt_grad_clip = float(_E("TTT_GRAD_CLIP", 1.0))
    ttt_temperature = float(_E("TTT_TEMPERATURE", 0.98))
    ttt_momentum = float(_E("TTT_MOMENTUM", 0.9))
    ttt_lora_rank = int(_E("TTT_LORA_RANK", 8))
    ttt_polyak_decay = float(_E("TTT_POLYAK_DECAY", 0.998))
    decoder_lr_mult = float(_E("DECODER_LR_MULT", 2.0))
    ngram_enabled = bool(int(_E("NGRAM_ENABLED", "1")))
    ngram_order = int(_E("NGRAM_EVAL_ORDER", 9))
    ngram_min_order = int(_E("NGRAM_EVAL_MIN_ORDER", 2))
    ngram_alpha = float(_E("NGRAM_EVAL_ALPHA", 0.30))
    ngram_adaptive = bool(int(_E("NGRAM_EVAL_ADAPTIVE", "1")))
    ngram_alpha_min = float(_E("NGRAM_EVAL_ALPHA_MIN", 0.12))
    ngram_alpha_max = float(_E("NGRAM_EVAL_ALPHA_MAX", 0.60))
    ngram_ent_center = float(_E("NGRAM_EVAL_ENTROPY_CENTER", 3.0))
    ngram_ent_scale = float(_E("NGRAM_EVAL_ENTROPY_SCALE", 2.0))
    ngram_min_count = int(_E("NGRAM_EVAL_MIN_COUNT", 2))
    ngram_buckets = int(_E("NGRAM_EVAL_BUCKETS", 4194304))
    ngram_batch_seqs = int(_E("NGRAM_EVAL_BATCH_SEQS", 32))
    ngram_chunk_tokens = int(_E("NGRAM_EVAL_CHUNK_TOKENS", 1_000_000))
    complement_alpha = float(_E("COMPLEMENT_ALPHA", 0.50))
    prune_pct = float(_E("PRUNE_PCT", 0.03))
def _ns5(G , steps = 10, eps = 1e-7):
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
def _ns5b(G , steps = 10, eps = 1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    norms = X.flatten(1).norm(dim=1).unsqueeze(-1).unsqueeze(-1).clamp_min(eps)
    X = X / norms
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = X.bmm(X.transpose(1, 2))
        B = b * A + c * A.bmm(A)
        X = a * X + B.bmm(X)
    if transposed:
        X = X.transpose(1, 2)
    return X
class PM(torch.optim.Optimizer):
    def __init__(self, params, lr , momentum , backend_steps ,
                 nesterov = True, weight_decay = 0.0):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                        nesterov=nesterov, weight_decay=weight_decay)
        param_list = list(params)
        super().__init__(param_list, defaults)
        self._bb()
    def _bb(self):
        self._group_banks: list[dict[tuple[int, int], list[int]]] = []
        for group in self.param_groups:
            banks: dict[tuple[int, int], list[int]] = {}
            for idx, p in enumerate(group["params"]):
                if p.ndim == 2:
                    shape_key = (p.shape[0], p.shape[1])
                    if shape_key not in banks:
                        banks[shape_key] = []
                    banks[shape_key].append(idx)
            self._group_banks.append(banks)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for gi, group in enumerate(self.param_groups):
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            param_updates = {}
            momentum_grads = {}
            for i, p in enumerate(params):
                if i % world_size != rank or p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                momentum_grads[i] = g
            group_banks = self._group_banks[gi] if gi < len(self._group_banks) else {}
            for shape_key, indices in group_banks.items():
                bank_indices = [idx for idx in indices if idx in momentum_grads]
                if not bank_indices:
                    continue
                if len(bank_indices) >= 2:
                    stacked = torch.stack([momentum_grads[idx] for idx in bank_indices])
                    ns_result = _ns5b(stacked, steps=backend_steps)
                    for j, idx in enumerate(bank_indices):
                        p = params[idx]
                        g = ns_result[j]
                        g *= max(1, g.size(0) / g.size(1)) ** 0.5
                        param_updates[idx] = g
                else:
                    idx = bank_indices[0]
                    g = _ns5(momentum_grads[idx], steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    param_updates[idx] = g
            for idx, g in momentum_grads.items():
                if idx not in param_updates:
                    param_updates[idx] = g
            curr = 0
            for i, p in enumerate(params):
                if i in param_updates:
                    updates_flat[curr : curr + p.numel()] = param_updates[i].reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss
def _sp_luts(
    sp: spm.SentencePieceProcessor, vocab_size , device ):
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
def _ld_shard(file ):
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
def _ld_val(pattern , seq_len ):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([_ld_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
class TS:
    def __init__(self, pattern ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = _ld_shard(self.files[0])
        self.pos = 0
    def _af(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = _ld_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n ):
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._af()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)
class DTL:
    def __init__(self, pattern , rank , world_size , device ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TS(pattern)
    def next_batch(self, global_tokens , seq_len , gas ):
        local_tokens = global_tokens // (self.world_size * gas)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
def _ev(
    args ,
    model ,
    rank ,
    world_size ,
    device ,
    gas ,
    vtok ,
    bblut ,
    hlslut ,
    ibtlut ,
    eval_seq_len = None,
):
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * gas)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={gas}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (vtok.numel() - 1) // seq_len
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
            local = vtok[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = bblut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (hlslut[tgt_ids] & ~ibtlut[prev_ids]).to(dtype=torch.int16)
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

# ============================================================
# LoRA Layer for TTT
# ============================================================
class LoRALayer(nn.Module):
    """Low-rank adapter for TTT. Adds A @ B to the base linear's output."""
    def __init__(self, in_features, out_features, rank=8, device=None, dtype=None):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device, dtype=dtype) * (1.0 / math.sqrt(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
    def forward(self, x):
        # x: (..., in_features) -> (..., out_features)
        return F.linear(F.linear(x, self.lora_A), self.lora_B)

# ============================================================
# N-gram hash/score/update helpers
# ============================================================
_NHP = np.array(
    [36313, 27191, 51647, 81929, 131071, 174763, 233021, 283721, 347239],
    dtype=np.uint64,
)

_NGRAM_ENT_CENTERS = {9: 2.5, 8: 2.75, 7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5}

def _ngh(
    val_np ,
    global_j ,
    ctx_width ,
    primes ,
    mask ,
):
    valid = global_j >= ctx_width
    if not valid.any():
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty
    v_idx = np.nonzero(valid)[0]
    jv = global_j[v_idx]
    ctx_hash = np.zeros(len(jv), dtype=np.uint64)
    for k in range(ctx_width):
        tok = val_np[jv - (ctx_width - k)].astype(np.uint64)
        ctx_hash ^= tok * primes[k % len(primes)]
    ctx_key = (ctx_hash & mask).astype(np.int64)
    tgt = val_np[jv].astype(np.uint64)
    full_key = ((ctx_hash ^ (tgt * primes[ctx_width % len(primes)])) & mask).astype(np.int64)
    return v_idx, ctx_key, full_key
def _ngs(
    seg_model_p ,
    seg_logits ,
    val_np ,
    global_j ,
    ctx_tables: dict[int, np.ndarray],
    full_tables: dict[int, np.ndarray],
    *,
    max_order = 9,
    min_order = 2,
    adaptive = True,
    alpha_fixed = 0.30,
    alpha_min = 0.05,
    alpha_max = 0.60,
    ent_center = 3.0,
    ent_scale = 2.0,
    min_count = 2,
    primes = _NHP,
    mask = np.uint64(4_194_304 - 1),
):
    seg_len = len(seg_model_p)
    if adaptive:
        with torch.no_grad():
            log_probs = F.log_softmax(seg_logits.float(), dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()
    _OM = {2: 0.3, 3: 0.3, 4: 1.0}
    p_ng = np.zeros(seg_len, dtype=np.float64)
    ng_matched = np.zeros(seg_len, dtype=np.bool_)
    ng_order = np.zeros(seg_len, dtype=np.int32)
    for n in range(max_order, min_order - 1, -1):
        ctx_width = n - 1
        v_idx, ctx_key, full_key = _ngh(
            val_np, global_j, ctx_width, primes, mask,
        )
        if len(v_idx) == 0:
            continue
        still_need = ~ng_matched[v_idx]
        if not still_need.any():
            continue
        v_idx = v_idx[still_need]
        ctx_key = ctx_key[still_need]
        full_key = full_key[still_need]
        ctx_counts = ctx_tables[n][ctx_key].astype(np.float64)
        full_counts = full_tables[n][full_key].astype(np.float64)
        has_data = ctx_counts >= float(min_count)
        if has_data.any():
            p = np.minimum(full_counts, ctx_counts) / np.maximum(ctx_counts, 1.0)
            p = np.clip(p, 0.0, 1.0)
            hit_idx = v_idx[has_data]
            p_ng[hit_idx] = p[has_data]
            ng_matched[hit_idx] = True
            ng_order[hit_idx] = n
    mixed_p = seg_model_p.copy()
    if ng_matched.any():
        m_idx = np.nonzero(ng_matched)[0]
        matched_orders = ng_order[m_idx]
        if adaptive:
            # Per-order entropy centers with per-order multipliers
            centers = np.array([_NGRAM_ENT_CENTERS.get(o, ent_center) for o in matched_orders], dtype=np.float64)
            sig = 1.0 / (1.0 + np.exp(-ent_scale * (entropy[m_idx] - centers)))
            a = alpha_min + (alpha_max - alpha_min) * sig
            # Apply per-order multipliers on top of per-order centers
            for order_val, mult in _OM.items():
                order_mask = matched_orders == order_val
                a[order_mask] *= mult
            high_order_mask = matched_orders >= 5
            a[high_order_mask] *= 2.0
            a = np.clip(a, 0.0, 0.95)
        else:
            a = np.full(len(m_idx), alpha_fixed)
        mixed_p[m_idx] = (1.0 - a) * mixed_p[m_idx] + a * p_ng[m_idx]
    return -np.log(np.clip(mixed_p, 1e-12, 1.0))
def _ngu(
    val_np ,
    global_j ,
    ctx_tables: dict[int, np.ndarray],
    full_tables: dict[int, np.ndarray],
    *,
    max_order = 9,
    min_order = 2,
    primes = _NHP,
    mask = np.uint64(4_194_304 - 1),
):
    buckets = int(mask) + 1
    for n in range(min_order, max_order + 1):
        ctx_width = n - 1
        v_idx, ctx_key, full_key = _ngh(
            val_np, global_j, ctx_width, primes, mask,
        )
        if len(v_idx) == 0:
            continue
        ctx_tables[n] += np.bincount(ctx_key, minlength=buckets).astype(ctx_tables[n].dtype)
        full_tables[n] += np.bincount(full_key, minlength=buckets).astype(full_tables[n].dtype)

# ============================================================
# Integrated single-pass eval (TTT + N-gram combined)
# ============================================================
def _ev_integrated(
    args ,
    bm ,
    rank ,
    world_size ,
    device ,
    vtok ,
    bblut ,
    hlslut ,
    ibtlut ,
    stride ,
    batch_seqs = 32,
    eval_seq_len = None,
    log_fn=None,
    time_budget_s = 600.0,
):
    """Single-pass evaluation: for each chunk, score with sliding window,
    compute n-gram probabilities, blend, accumulate BPB, update n-gram cache,
    then train TTT (LoRA) on scored tokens."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = vtok.numel() - 1
    chunk_tokens = args.ttt_chunk_tokens
    ttt_epochs = args.ttt_epochs
    ttt_lr = args.ttt_lr
    ttt_grad_clip = args.ttt_grad_clip
    ttt_temperature = args.ttt_temperature
    lora_rank = args.ttt_lora_rank
    polyak_decay = args.ttt_polyak_decay

    # N-gram config
    max_order = args.ngram_order
    min_order = max(args.ngram_min_order, 2)
    adaptive = args.ngram_adaptive
    alpha_fixed = args.ngram_alpha
    alpha_min = args.ngram_alpha_min
    alpha_max = args.ngram_alpha_max
    ent_center = args.ngram_ent_center
    ent_scale = args.ngram_ent_scale
    min_count = args.ngram_min_count
    buckets = args.ngram_buckets
    mask = np.uint64(buckets - 1)

    # Initialize n-gram tables
    ctx_tables = {
        n: np.zeros(buckets, dtype=np.uint32) for n in range(min_order, max_order + 1)
    }
    full_tables = {
        n: np.zeros(buckets, dtype=np.uint32) for n in range(min_order, max_order + 1)
    }
    val_np = vtok.numpy()

    # Build window lists per chunk
    all_window_starts = [
        ws for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= 1
    ]
    num_chunks = max((total_tokens + chunk_tokens - 1) // chunk_tokens, 1)
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in all_window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)

    distributed = dist.is_available() and dist.is_initialized()
    num_layers = len(bm.blocks)

    # ---- Setup LoRA adapters on Q and V of last 2 blocks ----
    lora_layers = {}  # {(block_idx, 'q'|'v'): LoRALayer}
    unfreeze_start = max(num_layers - 2, 0)  # last 2 blocks
    for bi in range(unfreeze_start, num_layers):
        block = bm.blocks[bi]
        q_in = block.attn.c_q.in_features
        q_out = block.attn.c_q.out_features
        v_in = block.attn.c_v.in_features
        v_out = block.attn.c_v.out_features
        lora_q = LoRALayer(q_in, q_out, rank=lora_rank, device=device, dtype=torch.float32)
        lora_v = LoRALayer(v_in, v_out, rank=lora_rank, device=device, dtype=torch.float32)
        lora_layers[(bi, 'q')] = lora_q
        lora_layers[(bi, 'v')] = lora_v

    # Collect all LoRA parameters
    lora_params = []
    for lora in lora_layers.values():
        lora_params.extend(list(lora.parameters()))

    # Polyak EMA state for LoRA params
    polyak_state = {id(p): p.data.clone() for p in lora_params}

    if log_fn:
        total_lora = sum(p.numel() for p in lora_params)
        log_fn(f"integrated_eval: LoRA rank={lora_rank} on Q,V of blocks {unfreeze_start}-{num_layers-1}, "
               f"{total_lora} LoRA params, {num_chunks} chunks, polyak_decay={polyak_decay}, "
               f"ngram order={min_order}-{max_order} adaptive={adaptive} alpha=[{alpha_min},{alpha_max}]")

    # Monkey-patch forward to inject LoRA
    # Save original forward methods
    _orig_attn_fwd = {}
    for bi in range(unfreeze_start, num_layers):
        block = bm.blocks[bi]
        _orig_attn_fwd[bi] = block.attn.forward

    def _make_lora_attn_forward(block_idx, orig_fwd, loras):
        lora_q = loras[(block_idx, 'q')]
        lora_v = loras[(block_idx, 'v')]
        def lora_forward(x, v_embed=None, v0=None, vr_lambda=None):
            attn = bm.blocks[block_idx].attn
            bsz, seqlen, dim = x.shape
            q = attn.c_q(x) + lora_q(x)  # Add LoRA delta to Q
            q = q.reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
            k = attn.c_k(x).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
            v = attn.c_v(x) + lora_v(x)  # Add LoRA delta to V
            if v_embed is not None:
                v = v + v_embed
            v = v.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
            raw_v = v
            if v0 is not None and vr_lambda is not None:
                lam = torch.softmax(vr_lambda, dim=0)
                v = lam[0] * v0 + lam[1] * v
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            cos, sin = attn.rotary(seqlen, x.device, q.dtype)
            q = _arope(q, cos, sin, attn.rope_dims)
            k = _arope(k, cos, sin, attn.rope_dims)
            q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
            y = flash_attn_3_func(q, k, v, causal=True)
            if attn.use_xsa:
                y = attn._xsa(y, v)
            if attn.use_gated_attn:
                gate = torch.sigmoid(attn.attn_gate(x))
                y = y * gate.unsqueeze(-1)
            y = y.reshape(bsz, seqlen, dim)
            return attn.proj(y), raw_v
        return lora_forward

    def _install_lora(loras, use_polyak=False):
        """Install LoRA-augmented forward on attention layers. If use_polyak,
        swap in polyak-averaged LoRA weights for scoring."""
        if use_polyak:
            # Swap in polyak weights
            saved = {}
            for p in lora_params:
                saved[id(p)] = p.data.clone()
                p.data.copy_(polyak_state[id(p)])
        for bi in range(unfreeze_start, num_layers):
            bm.blocks[bi].attn.forward = _make_lora_attn_forward(bi, _orig_attn_fwd[bi], loras)
        if use_polyak:
            return saved
        return None

    def _restore_lora_weights(saved):
        """Restore training weights after polyak scoring."""
        if saved is not None:
            for p in lora_params:
                p.data.copy_(saved[id(p)])

    def _uninstall_lora():
        """Restore original forward methods."""
        for bi in range(unfreeze_start, num_layers):
            bm.blocks[bi].attn.forward = _orig_attn_fwd[bi]

    # Freeze all base model params for TTT
    for p in bm.parameters():
        p.requires_grad_(False)

    # ---- Accumulators ----
    model_loss_sum = 0.0
    ngram_loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0

    t0 = _now()

    # ---- AdamW optimizer for LoRA ----
    ttt_optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": ttt_lr}],
        lr=ttt_lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    for ci in range(num_chunks):
        chunk_ws_all = chunk_windows[ci]
        if not chunk_ws_all:
            # Still need to update ngram cache for this chunk
            chunk_start = ci * chunk_tokens
            chunk_end = min((ci + 1) * chunk_tokens, total_tokens)
            chunk_global_j = np.arange(chunk_start + 1, chunk_end + 1, dtype=np.int64)
            _ngu(val_np, chunk_global_j, ctx_tables, full_tables,
                 max_order=max_order, min_order=min_order, primes=_NHP, mask=mask)
            continue

        # Shard windows across ranks for scoring
        my_s = (len(chunk_ws_all) * rank) // world_size
        my_e = (len(chunk_ws_all) * (rank + 1)) // world_size
        my_windows = chunk_ws_all[my_s:my_e]

        chunk_model_loss = 0.0
        chunk_ngram_loss = 0.0
        chunk_token_count = 0.0
        chunk_byte_count = 0.0

        # ---- SCORE phase: install LoRA with Polyak weights ----
        bm.eval()
        saved_weights = _install_lora(lora_layers, use_polyak=(ci > 0))

        # SLOT config
        slot_enabled = bool(int(_E("SLOT_ENABLED", "1")))
        slot_steps = int(_E("SLOT_STEPS", "16"))
        slot_lr = float(_E("SLOT_LR", "0.005"))

        for bi_w in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi_w:bi_w + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    seg = vtok[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = seg[:-1]
                    y_batch[i, :wlen] = seg[1:]

                if slot_enabled:
                    # SLOT: split forward into hidden + logits, optimize delta
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            hidden = bm.fwd_hidden(x_batch)
                    hidden_det = hidden.detach().float()
                    model_dim = hidden_det.shape[-1]
                    delta = torch.zeros(1, 1, model_dim, device=device, dtype=torch.float32, requires_grad=True)
                    slot_opt = torch.optim.AdamW([delta], lr=slot_lr, weight_decay=1e-8, eps=1e-5)
                    for _ss in range(slot_steps):
                        slot_opt.zero_grad()
                        logits_s = bm.compute_logits(hidden_det + delta)
                        loss_s = F.cross_entropy(
                            logits_s.reshape(-1, logits_s.size(-1)),
                            y_batch.reshape(-1),
                            reduction="mean",
                        )
                        loss_s.backward()
                        slot_opt.step()
                    with torch.no_grad():
                        logits = bm.compute_logits(hidden_det + delta.detach())
                else:
                    with torch.inference_mode():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = bm.fwd_l(x_batch)

                logits_f = logits.float()
                if ttt_temperature != 1.0:
                    logits_f = logits_f / ttt_temperature
                nll = F.cross_entropy(
                    logits_f.reshape(-1, logits_f.size(-1)),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    seg_len_val = wlen - s
                    if seg_len_val <= 0:
                        continue
                    seg_nll = nll[i, s:wlen].to(torch.float64).cpu().numpy()
                    seg_model_p = np.exp(-seg_nll)
                    global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
                    chunk_model_loss += float(seg_nll.sum())
                    # N-gram scoring integrated here
                    mixed_nll = _ngs(
                        seg_model_p,
                        logits_f[i, s:wlen],
                        val_np,
                        global_j,
                        ctx_tables,
                        full_tables,
                        max_order=max_order,
                        min_order=min_order,
                        adaptive=adaptive,
                        alpha_fixed=alpha_fixed,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                        ent_center=ent_center,
                        ent_scale=ent_scale,
                        min_count=min_count,
                        primes=_NHP,
                        mask=mask,
                    )
                    chunk_ngram_loss += float(mixed_nll.sum())
                    chunk_token_count += float(seg_len_val)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = bblut[tgt].to(torch.float64)
                    tb += (hlslut[tgt] & ~ibtlut[prev]).to(torch.float64)
                    chunk_byte_count += float(tb.sum().item())

        # Restore training LoRA weights after polyak scoring
        _restore_lora_weights(saved_weights)

        # Aggregate across ranks
        _cm = torch.tensor(chunk_model_loss, device=device, dtype=torch.float64)
        _cn = torch.tensor(chunk_ngram_loss, device=device, dtype=torch.float64)
        _ct = torch.tensor(chunk_token_count, device=device, dtype=torch.float64)
        _cb = torch.tensor(chunk_byte_count, device=device, dtype=torch.float64)
        if distributed:
            dist.all_reduce(_cm, op=dist.ReduceOp.SUM)
            dist.all_reduce(_cn, op=dist.ReduceOp.SUM)
            dist.all_reduce(_ct, op=dist.ReduceOp.SUM)
            dist.all_reduce(_cb, op=dist.ReduceOp.SUM)
        model_loss_sum += _cm.item()
        ngram_loss_sum += _cn.item()
        token_count += _ct.item()
        byte_count += _cb.item()

        # ---- UPDATE n-gram cache AFTER scoring ----
        chunk_start = ci * chunk_tokens
        chunk_end = min((ci + 1) * chunk_tokens, total_tokens)
        chunk_global_j = np.arange(chunk_start + 1, chunk_end + 1, dtype=np.int64)
        _ngu(
            val_np, chunk_global_j, ctx_tables, full_tables,
            max_order=max_order, min_order=min_order,
            primes=_NHP, mask=mask,
        )

        # ---- TRAIN LoRA on this chunk (score-first: we already scored) ----
        if ci < num_chunks - 1:
            _install_lora(lora_layers, use_polyak=False)  # install with training weights
            chunk_len = chunk_end - chunk_start
            if chunk_len >= seq_len:
                num_seqs = chunk_len // seq_len
                train_tokens = vtok[chunk_start : chunk_start + num_seqs * seq_len + 1]
                bm.train()
                for epoch in range(ttt_epochs):
                    for si in range(0, num_seqs, batch_seqs):
                        batch_end_s = min(si + batch_seqs, num_seqs)
                        actual_bsz = batch_end_s - si
                        t_start = si * seq_len
                        t_end = batch_end_s * seq_len + 1
                        local_tok = train_tokens[t_start:t_end].to(dtype=torch.int64, device=device)
                        x_train = local_tok[:-1].reshape(actual_bsz, seq_len)
                        y_train = local_tok[1:].reshape(actual_bsz, seq_len)
                        ttt_optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits_train = bm.fwd_l(x_train)
                        per_tok_loss = F.cross_entropy(
                            logits_train.float().reshape(-1, logits_train.size(-1)),
                            y_train.reshape(-1),
                            reduction="none",
                        ).reshape(actual_bsz, seq_len)
                        # Byte-weighted TTT loss
                        byte_weights = bblut[y_train].float()
                        byte_weights += (hlslut[y_train] & ~ibtlut[x_train]).float()
                        byte_weights = byte_weights.clamp(min=1.0)
                        loss = (per_tok_loss * byte_weights).sum() / byte_weights.sum()
                        loss.backward()
                        if distributed:
                            for p in lora_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        if ttt_grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(lora_params, ttt_grad_clip)
                        ttt_optimizer.step()
                # Polyak update after training on this chunk
                with torch.no_grad():
                    for p in lora_params:
                        polyak_state[id(p)].mul_(polyak_decay).add_(p.data, alpha=1.0 - polyak_decay)

        if log_fn and (ci + 1) % 5 == 0:
            elapsed = _now() - t0
            cur_model_bpb = (model_loss_sum / max(token_count, 1.0)) / math.log(2.0) * (token_count / max(byte_count, 1.0))
            cur_ngram_bpb = (ngram_loss_sum / max(token_count, 1.0)) / math.log(2.0) * (token_count / max(byte_count, 1.0))
            log_fn(
                f"integrated_eval: chunk {ci + 1}/{num_chunks} "
                f"model_bpb={cur_model_bpb:.6f} ngram_bpb={cur_ngram_bpb:.6f} "
                f"delta={cur_ngram_bpb - cur_model_bpb:.6f} t={elapsed:.0f}s"
            )

        # Per-chunk time guard: abort if eval budget exceeded
        elapsed_eval = _now() - t0
        if time_budget_s > 0 and elapsed_eval > time_budget_s:
            if log_fn:
                log_fn(f"integrated_eval: ABORTING at chunk {ci + 1}/{num_chunks} -- "
                       f"elapsed {elapsed_eval:.0f}s exceeds {time_budget_s:.0f}s budget")
            break

    # Restore original attention forwards
    _uninstall_lora()

    model_val_loss = model_loss_sum / max(token_count, 1.0)
    ngram_val_loss = ngram_loss_sum / max(token_count, 1.0)
    tpb = token_count / max(byte_count, 1.0)
    model_val_bpb = model_val_loss / math.log(2.0) * tpb
    ngram_val_bpb = ngram_val_loss / math.log(2.0) * tpb
    if log_fn:
        elapsed = _now() - t0
        log_fn(
            f"integrated_eval: DONE model_bpb={model_val_bpb:.4f} ngram_bpb={ngram_val_bpb:.4f} "
            f"delta={ngram_val_bpb - model_val_bpb:.4f} elapsed={elapsed:.0f}s"
        )
    return model_val_loss, model_val_bpb, ngram_val_loss, ngram_val_bpb


_CTRL = tuple(
    pattern
    for pattern in _E(
        "_CTRL",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda",
    ).split(",")
    if pattern
)
_PRSD = torch.float16
_CLQ = 0.9999984
def _qf(t ):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), _CLQ, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=_PRSD).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), _CLQ).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale
def _turboquant_encode(t, bits=5):
    """TurboQuant: random rotation + optimal scalar quantizer.
    Based on Google's TurboQuant (arXiv:2504.19874).
    For weight matrices: normalize rows, apply random rotation,
    quantize each coordinate with Lloyd-Max optimal centroids."""
    t32 = t.float()
    if t32.ndim != 2:
        # Fallback to simple quantization for non-matrix tensors
        clip_range = (1 << (bits - 1)) - 1
        amax = t32.abs().max().item()
        scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
        return q, scale
    rows, cols = t32.shape
    # Step 1: Store row norms and normalize
    norms = t32.norm(dim=1, keepdim=True).clamp(min=1e-10)
    t_normed = t32 / norms
    # Step 2: Random rotation via Hadamard-like transform (deterministic)
    # Use a random orthogonal matrix seeded deterministically
    torch.manual_seed(42)
    # For efficiency, use random sign flips + permutation (cheap approximation of random rotation)
    signs = (torch.randint(0, 2, (cols,), device=t32.device) * 2 - 1).float()
    perm = torch.randperm(cols, device=t32.device)
    t_rot = (t_normed * signs[None, :])[:, perm]
    # Step 3: After rotation, coordinates ~= N(0, 1/cols)
    # Scale to standard range and quantize with uniform quantizer
    # The theoretical std is 1/sqrt(cols)
    std_est = 1.0 / (cols ** 0.5)
    clip_range = (1 << (bits - 1)) - 1
    # Clip at ~3.5 sigma (captures 99.95% of mass for Gaussian)
    clip_val = 3.5 * std_est
    scale_per_row = (norms.squeeze(1) * clip_val / clip_range).clamp(min=1e-10).to(torch.float16)
    q = torch.clamp(torch.round(t_rot / (clip_val / clip_range)), -clip_range, clip_range).to(torch.int8)
    # Store inverse permutation and signs for dequantization
    inv_perm = torch.argsort(perm)
    return q, scale_per_row, signs, inv_perm, norms.squeeze(1).to(torch.float16)

def _turboquant_decode(q, scale, signs, inv_perm, norms, bits=5):
    """Dequantize TurboQuant encoded weights."""
    clip_range = (1 << (bits - 1)) - 1
    rows = q.shape[0]
    cols = q.shape[1]
    std_est = 1.0 / (cols ** 0.5)
    clip_val = 3.5 * std_est
    # Dequantize
    t_rot = q.float() * (clip_val / clip_range)
    # Undo permutation and sign flips
    t_normed = t_rot[:, inv_perm] * signs[None, :].float()
    # Restore norms
    return t_normed * norms.float()[:, None]

def _qi5(t , clip_range = 15):
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
def _brs(W , clip_range = 15):
    W32 = W.float()
    nrows = W32.shape[0]
    best_scales = torch.zeros(nrows, dtype=torch.float32, device=W.device)
    best_err = torch.full((nrows,), float('inf'), dtype=torch.float32, device=W.device)
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(W32.abs(), pct, dim=1)
        else:
            row_clip = W32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range)
        q = torch.clamp(torch.round(W32 / s[:, None]), -clip_range, clip_range)
        recon = q * s[:, None]
        row_mse = (W32 - recon).pow(2).mean(dim=1)
        improved = row_mse < best_err
        best_scales[improved] = s[improved]
        best_err[improved] = row_mse[improved]
    return best_scales
def _gptq_qw(
    W ,
    H ,
    clip_range = 15,
    block_size = 128,
    percdamp = 0.01,
):
    W = W.float().clone()
    nrows, ncols = W.shape
    H = H.float().clone()
    row_scale = _brs(W, clip_range)
    diag = torch.diag(H)
    damp = percdamp * diag.mean()
    diag += damp
    H[range(ncols), range(ncols)] = diag
    perm = torch.argsort(torch.diag(H))
    invperm = torch.argsort(perm)
    W = W[:, perm]
    H = H[perm][:, perm]
    # Robust Cholesky with progressive regularization
    diag_mean = torch.diag(H).mean().item() + 1e-6
    Hinv = None
    for reg in [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        try:
            Hreg = H if reg == 0 else H + torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * (reg * diag_mean)
            L = torch.linalg.cholesky(Hreg)
            Hinv = torch.cholesky_inverse(L)
            break
        except (torch._C._LinAlgError, torch.linalg.LinAlgError, RuntimeError):
            if reg >= 1.0:
                Hinv = torch.diag(1.0 / torch.diag(H).clamp_min(1e-10))
                break
    Q = torch.zeros_like(W)
    for i1 in range(0, ncols, block_size):
        i2 = min(i1 + block_size, ncols)
        W_block = W[:, i1:i2].clone()
        Q_block = torch.zeros_like(W_block)
        Err_block = torch.zeros_like(W_block)
        Hinv_block = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / row_scale), -clip_range, clip_range)
            Q_block[:, j] = q_col
            deq_col = q_col * row_scale
            err = (w_col - deq_col) / d.clamp_min(1e-10)
            Err_block[:, j] = err
            if j + 1 < i2 - i1:
                W_block[:, j + 1:] -= err[:, None] * Hinv_block[j, j + 1:][None, :]
        Q[:, i1:i2] = Q_block
        if i2 < ncols:
            W[:, i2:] -= Err_block @ Hinv[i1:i2, i2:]
    Q = Q[:, invperm]
    return Q.to(torch.int8), row_scale.to(torch.float16)
def _gptq_cal(
    model ,
    train_pattern ,
    device ,
    n_samples = 256,
    seq_len = 2048,
):
    hessians = {}
    n_seen = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, CL)):
            if module.weight.ndim == 2 and module.weight.numel() > 65536:
                def make_hook(layer_name , in_features ):
                    def hook_fn(mod, inp, out):
                        x = inp[0]
                        if x.ndim == 3:
                            x = x.reshape(-1, x.size(-1))
                        x = x.float()
                        xtx = x.T @ x
                        if layer_name not in hessians:
                            hessians[layer_name] = torch.zeros(in_features, in_features, device=device)
                            n_seen[layer_name] = 0
                        hessians[layer_name] += xtx
                        n_seen[layer_name] += x.shape[0]
                    return hook_fn
                h = module.register_forward_hook(make_hook(name, module.in_features))
                hooks.append(h)
    stream = TS(train_pattern)
    model.eval()
    samples_run = 0
    with torch.inference_mode():
        while samples_run < n_samples:
            batch_size = min(4, n_samples - samples_run)
            tokens = stream.take(batch_size * (seq_len + 1)).to(dtype=torch.int64, device=device)
            tokens = tokens[:batch_size * (seq_len + 1)].reshape(batch_size, seq_len + 1)
            x = tokens[:, :seq_len]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model.fwd_l(x)
            samples_run += batch_size
    for h in hooks:
        h.remove()
    for name in hessians:
        if n_seen[name] > 0:
            hessians[name] = hessians[name].clone() / n_seen[name]
    model.train()
    return hessians
def _clsp(name ):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def _mq5g(
    state_dict: dict[str, Tensor],
    int5_cats: set[str],
    hessians: dict[str, Tensor],
    block_size = 128,
    percdamp = 0.01,
    prune_pct = 0.0,
):
    result = {}
    meta = {}
    gptq_count = 0
    naive_count = 0
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _clsp(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in _CTRL):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        # Apply magnitude pruning before quantization
        if prune_pct > 0 and t.ndim == 2:
            with torch.no_grad():
                abs_t = t.float().abs()
                threshold = torch.quantile(abs_t.flatten(), prune_pct)
                t = t.clone()
                t[abs_t < threshold] = 0.0
        if cat in int5_cats and t.ndim >= 1:
            h_key = name.rsplit(".", 1)[0] if name.endswith((".weight", ".bias")) else name
            if t.ndim == 2 and h_key in hessians:
                H = hessians[h_key].cpu()
                if H.shape[0] == t.shape[1] and H.shape[1] == t.shape[1]:
                    q, s = _gptq_qw(t, H, clip_range=15, block_size=block_size, percdamp=percdamp)
                    gptq_count += 1
                else:
                    q, s = _qi5(t)
                    naive_count += 1
            else:
                q, s = _qi5(t)
                naive_count += 1
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int5"}
        else:
            q, s = _qf(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta, gptq_count, naive_count
def _dq5(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]):
    out = {}
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
class RN(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x ):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CL(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._qat_enabled = False
        self.register_buffer("_soft_round_alpha", torch.tensor(1.0), persistent=False)
    def forward(self, x ):
        w = self.weight.to(x.dtype)
        if self._qat_enabled and self.training and w.ndim == 2:
            alpha = self._soft_round_alpha.item()
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 15.0).clamp_min(1.0 / 15.0)
                w_scaled = w32 / scale[:, None]
                w_rounded = torch.round(w_scaled)
                diff = w_rounded - w_scaled
                w_soft = w_scaled + (1.0 / (2.0 * alpha)) * torch.tanh(alpha * diff)
                w_soft = torch.clamp(w_soft, -15, 15)
                w_q = (w_soft * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def _sqat(module , enabled , alpha = 1.0):
    for m in module.modules():
        if isinstance(m, CL):
            m._qat_enabled = enabled
            m._soft_round_alpha.fill_(alpha)
def _fp32(module ):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in _CTRL)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim , base = 10000.0, train_seq_len = 1024, rope_dims = 0):
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
    def _ic(self):
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    def forward(self, seq_len , device , dtype: torch.dtype):
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
def _arope(x , cos , sin , rope_dims = 0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
class CSA(nn.Module):
    def __init__(self, dim , num_heads , num_kv_heads , rope_base , qk_gain_init ,
                 gated_attn = False, gated_attn_bias_init = 4.0):
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
        self.c_q = CL(dim, dim, bias=False)
        self.c_k = CL(dim, kv_dim, bias=False)
        self.c_v = CL(dim, kv_dim, bias=False)
        self.proj = CL(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False
        self.use_gated_attn = gated_attn
        if gated_attn:
            self.attn_gate = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, gated_attn_bias_init)
    def _xsa(self, y , v ):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x , v_embed: Tensor | None = None,
                v0 = None, vr_lambda: nn.Parameter | None = None,
                ):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v
        if v0 is not None and vr_lambda is not None:
            lam = torch.softmax(vr_lambda, dim=0)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = _arope(q, cos, sin, self.rope_dims)
        k = _arope(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa(y, v)
        if self.use_gated_attn:
            gate = torch.sigmoid(self.attn_gate(x))
            y = y * gate.unsqueeze(-1)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y), raw_v
class SG(nn.Module):
    def __init__(self, dim ):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x ):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
class BHE(nn.Module):
    def __init__(self, bigram_vocab_size , bigram_dim , model_dim ):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CL(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def _bh(self, tokens ):
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids ):
        h = self.embed(self._bh(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class VEm(nn.Module):
    def __init__(self, vocab_size , ve_dim , model_dim ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CL(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids ):
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class MLP(nn.Module):
    def __init__(self, dim , mlp_mult ):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CL(dim, hidden, bias=False)
        self.proj = CL(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x ):
        x = F.leaky_relu(self.fc(x), negative_slope=0.9)
        return self.proj(x.square())
class Block(nn.Module):
    def __init__(self, dim , num_heads , num_kv_heads , mlp_mult ,
                 rope_base , qk_gain_init , layer_idx = 0,
                 ln_scale = False, dtg = False,
                 gated_attn = False, gated_attn_bias_init = 4.0,
                 vrl_enabled = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RN()
        self.mlp_norm = RN()
        self.attn = CSA(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        gated_attn=gated_attn, gated_attn_bias_init=gated_attn_bias_init)
        self.mlp = MLP(dim, mlp_mult)
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
        if vrl_enabled and layer_idx > 0:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        else:
            self.vr_lambda = None
    def forward(self, x , x0 , v_embed: Tensor | None = None,
                v0 = None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, raw_v = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed,
            v0=v0, vr_lambda=self.vr_lambda,
        )
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out, raw_v
class TNT:
    def __init__(self, vocab_size , device , complement_alpha = 0.5):
        self.V = vocab_size
        self.alpha = complement_alpha
        self.bi_counts = torch.zeros(vocab_size, vocab_size, device=device, dtype=torch.int64)
        self.bi_totals = torch.zeros(vocab_size, device=device, dtype=torch.int64)
    @torch.no_grad()
    def update(self, x , y ):
        xf = x.reshape(-1)
        yf = y.reshape(-1)
        ones = torch.ones(xf.numel(), device=xf.device, dtype=torch.int64)
        self.bi_counts.reshape(-1).scatter_add_(0, (xf * self.V + yf).long(), ones)
        self.bi_totals.scatter_add_(0, xf.long(), ones)
    def get_weights(self, x , y ):
        xf = x.reshape(-1)
        yf = y.reshape(-1)
        total = self.bi_totals[xf.long()].float()
        count = self.bi_counts.reshape(-1)[(xf * self.V + yf).long()].float()
        ngram_prob = count / (total + 1.0)
        return (1.0 - self.alpha * ngram_prob).clamp(min=0.1)
class GPT(nn.Module):
    def __init__(self, vocab_size , num_layers , model_dim , num_heads ,
                 num_kv_heads , mlp_mult , tie_embeddings , tied_embed_init_std ,
                 logit_softcap , rope_base , qk_gain_init ,
                 mtp_num_heads = 0, mtp_loss_weight = 0.1,
                 bigram_vocab_size = 0, bigram_dim = 128, xsa_last_n = 0,
                 rope_dims = 0, ln_scale = False, dtg = False,
                 ve_enabled = False, ve_dim = 128, ve_layers = "9,10",
                 vrl_enabled = False, gated_attn = False, gated_attn_bias_init = 4.0):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)
        self.vrl_enabled = vrl_enabled
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BHE(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SG(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, ln_scale=ln_scale, dtg=dtg,
                  gated_attn=gated_attn, gated_attn_bias_init=gated_attn_bias_init,
                  vrl_enabled=vrl_enabled)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = VEm(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()
        self.final_norm = RN()
        self.lm_head = None if tie_embeddings else CL(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CL(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
        )
        for head in self.mtp_heads:
            head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._iw()
    def _iw(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
    def _get_ve(self, layer_idx , input_ids , ve_cache: dict | None = None):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def _irc(self):
        for block in self.blocks:
            block.attn.rotary._ic()
    def forward(self, input_ids , target_ids ):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips = []
        ve_cache: dict = {}
        v0 = None
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](x, x0, v_embed=ve, v0=v0)
            if i == 0 and self.vrl_enabled:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x, _ = self.blocks[bi](x, x0, v_embed=ve, v0=v0)
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
        if hasattr(self, '_ngram_tracker') and self._ngram_tracker is not None and self.training:
            per_tok_loss = F.cross_entropy(logits.float(), targets, reduction="none")
            weights = self._ngram_tracker.get_weights(input_ids, target_ids)
            main_loss = (per_tok_loss * weights).mean()
        else:
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
    def fwd_l(self, input_ids ):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips = []
        ve_cache: dict = {}
        v0 = None
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](x, x0, v_embed=ve, v0=v0)
            if i == 0 and self.vrl_enabled:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x, _ = self.blocks[bi](x, x0, v_embed=ve, v0=v0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def fwd_hidden(self, input_ids):
        """Forward pass returning hidden states before lm_head projection."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips = []
        ve_cache: dict = {}
        v0 = None
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](x, x0, v_embed=ve, v0=v0)
            if i == 0 and self.vrl_enabled:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x, _ = self.blocks[bi](x, x0, v_embed=ve, v0=v0)
        x = self.final_norm(x)
        return x

    def compute_logits(self, hidden, delta=None):
        """Project hidden states (+ optional SLOT delta) to logits."""
        x = hidden
        if delta is not None:
            x = x + delta
        # Cast to match weight dtype (bfloat16 typically)
        if self.tie_embeddings:
            x = x.to(dtype=self.tok_emb.weight.dtype)
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            x = x.to(dtype=self.lm_head.weight.dtype)
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

def _gsl(args , elapsed_ms , max_wallclock_ms ):
    if not args.prog_seq_enabled:
        return args.train_seq_len
    frac = elapsed_ms / max(max_wallclock_ms, 1e-9)
    if frac < args.prog_seq_frac1:
        return args.prog_seq_phase1_len
    elif frac < args.prog_seq_frac2:
        return args.prog_seq_phase2_len
    else:
        return args.prog_seq_phase3_len
def main():
    global _ns5
    code = Path(__file__).read_text(encoding="utf-8")
    args = HP()
    _ns5 = torch.compile(_ns5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(_E("RANK", "0"))
    world_size = int(_E("WORLD_SIZE", "1"))
    local_rank = int(_E("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so gas stays integral")
    gas = 8 // world_size
    gsc = 1.0 / gas
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    mp = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    logfile = None
    if mp:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
    def log0(msg , console = True):
        if not mp:
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
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    eesl = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, eesl)
    vtok = _ld_val(args.val_files, val_seq_len)
    bblut, hlslut, ibtlut = _sp_luts(sp, args.vocab_size, device)
    log0(f"bpb:sp={args.tokenizer_path}")
    log0(f"tl:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val:{args.val_files} tokens:{vtok.numel() - 1}")
    log0(f"v:opti-ms2 act:lr09sq xsa:last_{args.xsa_last_n} "
         f"qat:sr wd:WSD gptq:fh ttt:lora_polyak_adamw compression:lzma "
         f"optimizer:PM prog_seq:enabled vrl:{args.vrl_enabled} gated_attn:{args.gated_attn_enabled} "
         f"decoder_lr_mult:{args.decoder_lr_mult} ngram:{args.ngram_enabled} ngram_order:{args.ngram_order} "
         f"complement_alpha:{args.complement_alpha} prune_pct:{args.prune_pct}")
    bm = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads, mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        vrl_enabled=args.vrl_enabled, gated_attn=args.gated_attn_enabled,
        gated_attn_bias_init=args.gated_attn_bias_init,
    ).to(device).bfloat16()
    for module in bm.modules():
        if isinstance(module, CL):
            module.float()
    _fp32(bm)
    complement_alpha = args.complement_alpha
    if complement_alpha > 0:
        ngram_tracker = TNT(args.vocab_size, device, complement_alpha=complement_alpha)
        bm._ngram_tracker = ngram_tracker
        log0(f"comp:on alpha={complement_alpha}")
    else:
        bm._ngram_tracker = None
        log0("comp:off")
    cm = torch.compile(bm, dynamic=False, fullgraph=False)
    model = DDP(cm, device_ids=[local_rank], broadcast_buffers=False) if distributed else cm
    ebi = set(range(bm.num_encoder_layers))
    dbi = set(range(bm.num_encoder_layers, len(bm.blocks)))
    emp = []
    dmp = []
    scp = []
    for block_idx, block in enumerate(bm.blocks):
        for name, p in block.named_parameters():
            if p.ndim == 2 and not any(pattern in name for pattern in _CTRL):
                if block_idx in ebi:
                    emp.append(p)
                else:
                    dmp.append(p)
            elif p.ndim < 2 or any(pattern in name for pattern in _CTRL):
                scp.append(p)
    if bm.mtp_num_heads > 0:
        emp.extend([p for p in bm.mtp_heads.parameters() if p.ndim == 2])
    if bm.skip_weights.numel() > 0:
        scp.append(bm.skip_weights)
    scp.append(bm.smear.gate)
    if bm.bigram is not None:
        scp.append(bm.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [bm.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if bm.bigram is not None:
        tok_params.append({"params": [bm.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if bm.bigram.proj is not None:
            emp.append(bm.bigram.proj.weight)
    if bm.ve_shared is not None:
        tok_params.append({"params": [bm.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if bm.ve_shared.proj is not None:
            emp.append(bm.ve_shared.proj.weight)
        scp.append(bm.ve_shared.scale)
        for s in bm.ve_layer_scales:
            scp.append(s)
    opt_t = torch.optim.AdamW(
        tok_params, betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_wd, fused=True,
    )
    encoder_lr = args.matrix_lr
    decoder_lr = args.matrix_lr * args.decoder_lr_mult
    amp2 = emp + dmp
    epi = {id(p) for p in emp}
    opt_m = PM(
        amp2, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd,
    )
    enc_params = [p for p in amp2 if id(p) in epi]
    dec_params = [p for p in amp2 if id(p) not in epi]
    opt_m.param_groups = []
    if enc_params:
        opt_m.add_param_group({
            "params": enc_params, "lr": encoder_lr, "base_lr": encoder_lr,
            "momentum": args.muon_momentum, "backend_steps": args.muon_backend_steps,
            "nesterov": True, "weight_decay": args.muon_wd,
        })
    if dec_params:
        opt_m.add_param_group({
            "params": dec_params, "lr": decoder_lr, "base_lr": decoder_lr,
            "momentum": args.muon_momentum, "backend_steps": args.muon_backend_steps,
            "nesterov": True, "weight_decay": args.muon_wd,
        })
    opt_m._bb()
    opt_s = torch.optim.AdamW(
        [{"params": scp, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_wd, fused=True,
    )
    optimizers = [opt_t, opt_m, opt_s]
    if bm.lm_head is not None:
        opt_h = torch.optim.Adam(
            [{"params": [bm.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, opt_h)
    n_params = sum(p.numel() for p in bm.parameters())
    mtp_params = sum(p.numel() for p in bm.mtp_heads.parameters())
    log0(f"model_params:{n_params}")
    log0(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params}")
    xsa_layers = [i for i, b in enumerate(bm.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"world_size:{world_size} gas:{gas}")
    log0(f"seed:{args.seed}")
    log0(f"lr_schedule:WSD wsd_stable_frac:{args.wsd_stable_frac} decay_shape:cosine qat_trigger_frac:{args.qat_trigger_frac}")
    log0(f"optimizer:PM (NS)")
    if args.prog_seq_enabled:
        log0(f"pseq:on phases={args.prog_seq_phase1_len}/{args.prog_seq_phase2_len}/{args.prog_seq_phase3_len} "
             f"fracs={args.prog_seq_frac1:.2f}/{args.prog_seq_frac2:.2f}")
    log0(f"ttt_config: optimizer=AdamW(LoRA) epochs={args.ttt_epochs} lr={args.ttt_lr} "
         f"lora_rank={args.ttt_lora_rank} polyak_decay={args.ttt_polyak_decay} "
         f"chunk={args.ttt_chunk_tokens} "
         f"grad_clip={args.ttt_grad_clip} temperature={args.ttt_temperature}")
    log0(f"decoder_lr_mult:{args.decoder_lr_mult} encoder_matrix_lr:{encoder_lr} decoder_matrix_lr:{decoder_lr}")
    ga_layers = [i for i, b in enumerate(bm.blocks) if b.attn.use_gated_attn]
    vrl_layers = [i for i, b in enumerate(bm.blocks) if b.vr_lambda is not None]
    log0(f"gated_attn:{'enabled' if args.gated_attn_enabled else 'disabled'} layers:{ga_layers}")
    log0(f"value_residual:{'enabled' if args.vrl_enabled else 'disabled'} layers:{vrl_layers}")
    log0(f"gptq_config: samples={args.gptq_n_samples} block_size={args.gptq_block_size} damp={args.gptq_percdamp}")
    log0(f"prune_pct:{args.prune_pct}")
    tl = DTL(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    if args.max_wallclock_seconds <= 0:
        raise RuntimeError(
            f"FATAL: MAX_WALLCLOCK_SECONDS={args.max_wallclock_seconds} disables the training time cap. "
            f"Competition rules require MAX_WALLCLOCK_SECONDS=600. NEVER set to 0 on paid runs."
        )
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds
    def lr_mul(step , elapsed_ms ):
        if max_wallclock_ms is None:
            total = max(args.iterations, 1)
            stable_end = int(args.wsd_stable_frac * total)
            if step <= stable_end:
                return 1.0
            decay_steps = max(total - stable_end, 1)
            progress = min((step - stable_end) / decay_steps, 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        stable_ms = args.wsd_stable_frac * max_wallclock_ms
        if elapsed_ms <= stable_ms:
            return 1.0
        decay_ms = max(max_wallclock_ms - stable_ms, 1e-9)
        progress = min((elapsed_ms - stable_ms) / decay_ms, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if args.warmup_steps > 0:
        ims = {name: tensor.detach().cpu().clone() for name, tensor in bm.state_dict().items()}
        ios = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        warmup_seq = args.prog_seq_phase1_len if args.prog_seq_enabled else args.train_seq_len
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(gas):
                if distributed:
                    model.require_backward_grad_sync = micro_step == gas - 1
                x, y = tl.next_batch(args.train_batch_tokens, warmup_seq, gas)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * gsc).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        if args.prog_seq_enabled:
            for extra_seq in [args.prog_seq_phase2_len, args.prog_seq_phase3_len]:
                if extra_seq != warmup_seq:
                    zero_grad_all()
                    x, y = tl.next_batch(args.train_batch_tokens, extra_seq, gas)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        loss_w = model(x, y)
                    (loss_w * gsc).backward()
                    for opt in optimizers:
                        opt.step()
                    zero_grad_all()
                    log0(f"warmup:primed seq_len={extra_seq}")
        bm.load_state_dict(ims, strict=True)
        for opt, state in zip(optimizers, ios, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        tl = DTL(args.train_files, rank, world_size, device)
    ema_state = {name: t.detach().float().clone() for name, t in bm.state_dict().items()}
    ema_decay = 0.997
    swa_state = None
    swa_count = 0
    swa_every = 50
    swa_scale_threshold = 0.2
    ttms = 0.0
    stop_after_step = None
    csl = args.prog_seq_phase1_len if args.prog_seq_enabled else args.train_seq_len
    qat_start_ms = None
    _sync()
    t0 = _now()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            _sync()
            ttms += 1000.0 * (_now() - t0)
            val_loss, val_bpb = _ev(
                args, model, rank, world_size, device, gas,
                vtok, bblut, hlslut, ibtlut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{ttms:.0f}ms step_avg:{ttms / max(step, 1):.2f}ms "
                 f"seq_len:{csl}")
            _sync()
            t0 = _now()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"early_stop train_time:{ttms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_ms = ttms + 1000.0 * (_now() - t0)
        if args.prog_seq_enabled and max_wallclock_ms is not None:
            new_seq_len = _gsl(args, elapsed_ms, max_wallclock_ms)
            if new_seq_len != csl:
                log0(f"prog_seq:transition seq_len {csl}->{new_seq_len} step:{step} elapsed_ms:{elapsed_ms:.0f}")
                csl = new_seq_len
                bm._irc()
        scale = lr_mul(step, elapsed_ms)
        is_qat_on = any(isinstance(m, CL) and m._qat_enabled for m in bm.modules())
        if not is_qat_on and args.late_qat_threshold > 0:
            trigger = False
            if max_wallclock_ms is not None:
                if elapsed_ms >= args.qat_trigger_frac * max_wallclock_ms:
                    trigger = True
            else:
                if step >= int(args.qat_trigger_frac * args.iterations):
                    trigger = True
            if trigger:
                _sqat(bm, True, alpha=1.0)
                qat_start_ms = elapsed_ms
                log0(f"late_qat:enabled step:{step} scale:{scale:.4f} elapsed_ms:{elapsed_ms:.0f} trigger:wallclock@{args.qat_trigger_frac}")
        elif is_qat_on and qat_start_ms is not None:
            remaining_ms = max((max_wallclock_ms or float('inf')) - elapsed_ms, 0)
            qat_total_ms = max(elapsed_ms - qat_start_ms, 1e-9)
            if max_wallclock_ms is not None:
                total_qat_ms = max(max_wallclock_ms - qat_start_ms, 1e-9)
                alpha_frac = min(qat_total_ms / total_qat_ms, 1.0)
            else:
                alpha_frac = min(qat_total_ms / max(1000.0 * args.max_wallclock_seconds * (1.0 - args.qat_trigger_frac), 1e-9), 1.0)
            alpha = 1.0 + 15.0 * alpha_frac
            _sqat(bm, True, alpha=alpha)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        micro_batches: list[tuple[Tensor, Tensor]] = []
        for micro_step in range(gas):
            if distributed:
                model.require_backward_grad_sync = micro_step == gas - 1
            x, y = tl.next_batch(args.train_batch_tokens, csl, gas)
            if bm._ngram_tracker is not None:
                micro_batches.append((x.detach(), y.detach()))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * gsc).backward()
        train_loss /= gas
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in opt_m.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(bm.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        if bm._ngram_tracker is not None:
            for mb_x, mb_y in micro_batches:
                bm._ngram_tracker.update(mb_x, mb_y)
        with torch.no_grad():
            for name, t in bm.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        if scale < swa_scale_threshold and step % swa_every == 0:
            with torch.no_grad():
                if swa_state is None:
                    swa_state = {name: t.clone() for name, t in ema_state.items()}
                    swa_count = 1
                else:
                    for name, t in ema_state.items():
                        swa_state[name] += t
                    swa_count += 1
        step += 1
        approx_ttms = ttms + 1000.0 * (_now() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ttms:.0f}ms step_avg:{approx_ttms / step:.2f}ms "
                 f"seq_len:{csl}")
        reached_cap = max_wallclock_ms is not None and approx_ttms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(f"peak_mem: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
         f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
    log0(f"steps:{step}")
    log0(f"swa_n:{swa_count}")
    if swa_state is not None and swa_count > 0:
        log0(f"swa:blending EMA+SWA (0.7*EMA + 0.3*SWA, {swa_count} SWA checkpoints)")
        swa_avg = {name: (swa_state[name] / swa_count) for name in swa_state}
        avg_state = {}
        for name, ema_t in ema_state.items():
            dtype = bm.state_dict()[name].dtype
            avg_state[name] = (0.7 * ema_t + 0.3 * swa_avg[name]).to(dtype=dtype)
    else:
        log0("ema:applying EMA weights (no SWA checkpoints collected)")
        avg_state = {name: t.to(dtype=bm.state_dict()[name].dtype)
                     for name, t in ema_state.items()}
    bm.load_state_dict(avg_state, strict=True)
    _sync()
    t_diag = _now()
    diag_val_loss, diag_val_bpb = _ev(
        args, cm, rank, world_size, device, gas,
        vtok, bblut, hlslut, ibtlut,
    )
    _sync()
    log0(f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
         f"eval_time:{1000.0 * (_now() - t_diag):.0f}ms")
    quant_method = _E("QUANT_METHOD", "gptq")
    if quant_method.startswith("turboquant"):
        # TurboQuant: random rotation + optimal scalar quantizer
        turbo_bits = int(_E("TURBO_BITS", "5"))
        log0(f"turboquant:starting with {turbo_bits} bits per weight")
        full_state_dict = bm.state_dict()
        export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
        sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
        quant_result = {}
        quant_meta = {}
        for name, t in sd_cpu.items():
            if t.ndim == 2 and t.numel() > 64:
                result = _turboquant_encode(t, bits=turbo_bits)
                if len(result) == 5:
                    q, scale, signs, inv_perm, norms = result
                    quant_result[name] = q
                    quant_meta[name] = {"type": f"turbo{turbo_bits}", "scale": scale,
                                        "signs": signs, "inv_perm": inv_perm, "norms": norms}
                else:
                    q, scale = result
                    quant_result[name] = q
                    quant_meta[name] = {"type": f"int{turbo_bits}", "scale": scale}
            else:
                quant_result[name] = _qf(t)[0] if t.ndim >= 1 else t
                quant_meta[name] = {"type": "int8", "scale": _qf(t)[1] if t.ndim >= 1 else None}
        log0(f"turboquant:quantized {len(quant_result)} tensors")
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = lzma.compress(quant_raw, preset=9 | lzma.PRESET_EXTREME)
        best_compressor = "lzma"
        log0(f"turboquant:compressed {len(quant_raw)} -> {len(quant_blob)} bytes")
        if mp:
            with open("final_model.int5.ptz", "wb") as f:
                f.write(b"LZMA")
                f.write(quant_blob)
            quant_file_bytes = 4 + len(quant_blob)
            code_bytes = len(code.encode("utf-8"))
            total_bytes = quant_file_bytes + code_bytes
            log0(f"Serialized model turbo{turbo_bits}+{best_compressor}: {quant_file_bytes} bytes")
            log0(f"Total submission size: {total_bytes} bytes")
        # Skip GPTQ path, go to roundtrip eval
        # Need to reconstruct model for eval
        log0(f"turboquant:reconstructing model for roundtrip eval")
        rebuilt_sd = {}
        for name in quant_result:
            meta = quant_meta[name]
            q = quant_result[name]
            if meta["type"].startswith("turbo"):
                rebuilt_sd[name] = _turboquant_decode(
                    q, meta["scale"], meta["signs"], meta["inv_perm"], meta["norms"],
                    bits=turbo_bits
                )
            elif meta["type"] == "int8" and meta.get("scale") is not None:
                rebuilt_sd[name] = q.float() * meta["scale"].float()
            else:
                rebuilt_sd[name] = q.float() if isinstance(q, torch.Tensor) else q
        bm.load_state_dict({k: v.to(device) for k, v in rebuilt_sd.items()}, strict=False)
    if not quant_method.startswith("turboquant"):
        log0(f"gptq:calibrating with {args.gptq_n_samples} samples...")
        _sync()
        t_cal = _now()
        hessians = _gptq_cal(bm, args.train_files, device, n_samples=args.gptq_n_samples, seq_len=args.train_seq_len)
        _sync()
        log0(f"gptq:calibration done in {1000.0 * (_now() - t_cal):.0f}ms, {len(hessians)} layers")
        full_state_dict = bm.state_dict()
        export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
        excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)
        if excluded_mtp > 0:
            log0(f"mtp_excl:{excluded_mtp}")
        if mp:
            torch.save(export_sd, "final_model.pt")
            model_bytes = os.path.getsize("final_model.pt")
            code_bytes = len(code.encode("utf-8"))
            log0(f"Serialized model: {model_bytes} bytes")
            log0(f"Code size: {code_bytes} bytes")
        sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
        log0(f"gptq:quantizing with block_size={args.gptq_block_size} percdamp={args.gptq_percdamp} prune_pct={args.prune_pct}...")
        _sync()
        t_quant = _now()
        quant_result, quant_meta, gptq_count, naive_count = _mq5g(
            sd_cpu, {"mlp", "attn"}, hessians,
            block_size=args.gptq_block_size, percdamp=args.gptq_percdamp,
            prune_pct=args.prune_pct,
        )
        _sync()
        log0(f"gptq:quantization done in {1000.0 * (_now() - t_quant):.0f}ms (gptq:{gptq_count} naive:{naive_count})")
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = lzma.compress(quant_raw, preset=9 | lzma.PRESET_EXTREME)
        best_compressor = "lzma"
        log0(f"compression:{best_compressor} raw_size:{len(quant_raw)} compressed_size:{len(quant_blob)} "
             f"ratio:{len(quant_raw)/len(quant_blob):.2f}x")
        if mp:
            with open("final_model.int5.ptz", "wb") as f:
                f.write(b"LZMA")
                f.write(quant_blob)
            quant_file_bytes = 4 + len(quant_blob)
            code_bytes = len(code.encode("utf-8"))
            total_bytes = quant_file_bytes + code_bytes
            log0(f"Serialized model int5+{best_compressor}: {quant_file_bytes} bytes")
            log0(f"Total submission size: {total_bytes} bytes")
            if total_bytes > 16_000_000:
                raise RuntimeError(
                    f"FATAL: Total submission size {total_bytes} exceeds 16MB limit! "
                    f"Delta: +{total_bytes - 16_000_000} bytes. "
                    f"Artifact: {quant_file_bytes} bytes, Code: {code_bytes} bytes."
                )
            else:
                log0(f"Size budget OK: {16_000_000 - total_bytes} bytes remaining")
        if distributed:
            dist.barrier()
        with open("final_model.int5.ptz", "rb") as f:
            comp_id = f.read(4)
            quant_blob_disk = f.read()
        if comp_id == b"LZMA":
            raw_bytes = lzma.decompress(quant_blob_disk)
        elif comp_id == b"ZSTD":
            raw_bytes = zstandard.ZstdDecompressor().decompress(quant_blob_disk)
        elif comp_id == b"ZLIB":
            raw_bytes = zlib.decompress(quant_blob_disk)
        else:
            raise ValueError(f"Unknown compressor ID: {comp_id!r}")
        quant_state = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
        deq_state = _dq5(quant_state["w"], quant_state["m"], sd_cpu)
    em = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        vrl_enabled=args.vrl_enabled, gated_attn=args.gated_attn_enabled,
        gated_attn_bias_init=args.gated_attn_bias_init,
    ).to(device).bfloat16()
    for m in em.modules():
        if isinstance(m, CL):
            m.float()
    _fp32(em)
    if quant_method.startswith("turboquant"):
        em.load_state_dict({k: v.to(device) for k, v in rebuilt_sd.items()}, strict=False)
    else:
        em.load_state_dict(deq_state, strict=True)
    ce = torch.compile(em, dynamic=False, fullgraph=False)
    _sync()
    t_eval_start = _now()  # Start of 600s eval budget
    EVAL_BUDGET_S = 600.0
    t_qeval = _now()
    q_val_loss, q_val_bpb = _ev(
        args, ce, rank, world_size, device, gas,
        vtok, bblut, hlslut, ibtlut,
        eval_seq_len=eesl,
    )
    _sync()
    log0(f"final_int5_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0 * (_now() - t_qeval):.0f}ms")
    log0(f"final_int5_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # ============================================================
    # Single integrated eval pass (TTT + N-gram combined)
    # ============================================================
    eval_elapsed = _now() - t_eval_start
    eval_remaining = EVAL_BUDGET_S - eval_elapsed
    log0(f"eval_budget: {eval_elapsed:.0f}s elapsed, {eval_remaining:.0f}s remaining")

    ngram_time_ms = 0.0
    ngram_val_bpb = q_val_bpb
    ngram_val_loss = q_val_loss
    ngram_model_bpb = q_val_bpb
    ngram_model_loss = q_val_loss
    ttt_val_bpb = q_val_bpb
    ttt_val_loss = q_val_loss

    if eval_remaining < 60:
        log0("eval_budget: SKIPPING integrated eval -- insufficient time")
    elif args.ngram_enabled:
        log0("integrated_eval: starting single-pass TTT+N-gram evaluation")
        _sync()
        t_integrated = _now()
        ngram_model_loss, ngram_model_bpb, ngram_val_loss, ngram_val_bpb = _ev_integrated(
            args, em, rank, world_size, device,
            vtok, bblut, hlslut, ibtlut,
            stride=args.eval_stride,
            batch_seqs=args.ngram_batch_seqs,
            eval_seq_len=eesl,
            log_fn=log0,
            time_budget_s=eval_remaining,
        )
        _sync()
        ngram_time_ms = 1000.0 * (_now() - t_integrated)
        ttt_val_bpb = ngram_model_bpb  # model BPB (with TTT LoRA but without n-gram)
        ttt_val_loss = ngram_model_loss
        log0(f"final_integrated model_val_loss:{ngram_model_loss:.4f} model_val_bpb:{ngram_model_bpb:.4f}")
        log0(f"final_integrated ngram_val_loss:{ngram_val_loss:.4f} ngram_val_bpb:{ngram_val_bpb:.4f}")
        log0(f"final_integrated delta_bpb:{ngram_val_bpb - ngram_model_bpb:.4f}")
        log0(f"final_integrated_exact model_bpb:{ngram_model_bpb:.8f} ngram_bpb:{ngram_val_bpb:.8f}")
        log0(f"final_integrated_time:{ngram_time_ms:.0f}ms")
    else:
        log0("ngram:DISABLED (set NGRAM_ENABLED=1 to enable)")

    if mp:
        best_bpb = ngram_val_bpb if args.ngram_enabled else ttt_val_bpb
        submission = {
            "name": "opti-ms2",
            "github_id": _E("GITHUB_ID", "callithyia"),
            "variant": "opti-ms2",
            "description": "Optimised MS2: integrated eval, LoRA TTT, per-order entropy centers, pruning",
            "base": "ms2v2",
            "val_bpb": round(best_bpb, 8),
            "ttt_val_bpb": round(ttt_val_bpb, 8),
            "ngram_model_bpb": round(ngram_model_bpb, 8),
            "ngram_val_bpb": round(ngram_val_bpb, 8),
            "ngram_delta_bpb": round(ngram_val_bpb - ngram_model_bpb, 8) if args.ngram_enabled else 0.0,
            "seed": args.seed,
            "wsd_stable_frac": args.wsd_stable_frac,
            "qat_trigger_frac": args.qat_trigger_frac,
            "prog_seq_enabled": args.prog_seq_enabled,
            "ttt_optimizer": "AdamW(LoRA)",
            "ttt_epochs": args.ttt_epochs,
            "ttt_lr": args.ttt_lr,
            "ttt_lora_rank": args.ttt_lora_rank,
            "ttt_polyak_decay": args.ttt_polyak_decay,
            "ttt_temperature": args.ttt_temperature,
            "vrl_enabled": args.vrl_enabled,
            "gated_attn_enabled": args.gated_attn_enabled,
            "decoder_lr_mult": args.decoder_lr_mult,
            "compression": "lzma",
            "ngram_enabled": args.ngram_enabled,
            "ngram_order": args.ngram_order,
            "ngram_min_order": args.ngram_min_order,
            "ngram_adaptive": args.ngram_adaptive,
            "ngram_alpha_min": args.ngram_alpha_min,
            "ngram_alpha_max": args.ngram_alpha_max,
            "ngram_ent_center": args.ngram_ent_center,
            "ngram_ent_scale": args.ngram_ent_scale,
            "ngram_min_count": args.ngram_min_count,
            "ngram_buckets": args.ngram_buckets,
            "complement_alpha": args.complement_alpha,
            "prune_pct": args.prune_pct,
            "activation": "leaky_relu_0.9_sq",
            "total_steps": step,
            "ttms": round(ttms, 1),
            "integrated_eval_time_ms": round(ngram_time_ms, 1),
        }
        with open("submission.json", "w") as _f:
            json.dump(submission, _f, indent=2)
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
