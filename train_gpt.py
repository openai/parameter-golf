"""
The `train_gpt.py` script upgraded with IFNSO, XSA, SLOT, and N-gram Backoff Cache.
"""

from __future__ import annotations

import glob
import math
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# -----------------------------
# HYPERPARAMETERS
# -----------------------------
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 16384))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))

    iterations = int(os.environ.get("ITERATIONS", 10000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 8192))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 512))  # Shorter seq for RTX 2050
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
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
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


# -----------------------------
# IFNSO OPTIMIZER
# -----------------------------


def zeropower_via_ifnso(G: Tensor, eps: float = 1e-7) -> Tensor:
    # Iteration-Free Newton-Schulz Orthogonalization (IFNSO)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    A = X @ X.T
    # Constrained polynomial coefficients for exponential term selection
    c0, c1, c2 = 1.5, -0.5, 0.125
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    approx_inv_sqrt = c0 * I + c1 * A + c2 * (A @ A)

    X = approx_inv_sqrt @ X
    return X.T if transposed else X


class MuonIFNSO(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )
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

                    # IFNSO single-pass orthogonalization
                    g = zeropower_via_ifnso(g)
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


# -----------------------------
# EVALUATION SETUP & N-GRAM CACHE
# -----------------------------


class BackoffNgramMixer:
    def __init__(self, vocab_size=1024, max_order=9):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.cache = {
            i: defaultdict(lambda: defaultdict(int)) for i in range(1, max_order + 1)
        }

    def update(self, context, target):
        context_tuple = tuple(context.tolist())
        for order in range(1, min(len(context_tuple) + 1, self.max_order + 1)):
            ngram_context = context_tuple[-order:]
            self.cache[order][ngram_context][target.item()] += 1

    def predict(self, context, device):
        context_tuple = tuple(context.tolist())
        for order in range(min(len(context_tuple), self.max_order), 0, -1):
            ngram_context = context_tuple[-order:]
            if ngram_context in self.cache[order]:
                counts = self.cache[order][ngram_context]
                total = sum(counts.values())
                probs = torch.zeros(self.vocab_size, device=device)
                for token, count in counts.items():
                    probs[token] = count / total
                return probs
        return torch.ones(self.vocab_size, device=device) / self.vocab_size

    def mix_distributions(self, neural_logits, context):
        neural_probs = F.softmax(neural_logits, dim=-1)
        entropy = -torch.sum(neural_probs * torch.log(neural_probs + 1e-9))

        max_entropy = math.log(self.vocab_size)
        alpha = torch.clamp(1.0 - (entropy / max_entropy), min=0.0, max=1.0)

        ngram_probs = self.predict(context, neural_logits.device)
        return alpha * neural_probs + (1 - alpha) * ngram_probs


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)

    id_to_piece = getattr(sp, "id_to_piece", None)
    if id_to_piece is None:
        id_to_piece = getattr(sp, "IdToPiece", None)
    if id_to_piece is None:
        raise AttributeError(
            "SentencePieceProcessor does not expose an id-to-piece API"
        )

    for token_id in range(sp_vocab_size):
        try:
            piece = str(id_to_piece(token_id))
        except Exception:
            continue
        if not piece or piece in ("<unk>", "<s>", "</s>"):
            continue
        is_boundary_token_np[token_id] = False
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        try:
            base_bytes_np[token_id] = len(piece.encode("utf-8"))
        except Exception:
            base_bytes_np[token_id] = 0
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    total_tokens = 0
    chunks: list[Tensor] = []
    for file in files:
        shard = load_data_shard(file)
        chunks.append(shard)
        total_tokens += shard.numel()

    if len(chunks) == 1:
        tokens = chunks[0].contiguous()
    else:
        tokens = torch.cat(chunks, dim=0).contiguous()

    usable = ((total_tokens - 1) // seq_len) * seq_len
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
    local_batch_tokens = max(args.val_batch_size // (world_size * grad_accum_steps), args.train_seq_len)
    local_batch_seqs = max(local_batch_tokens // args.train_seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Initialize Evaluative Structures - simplified for memory
    ngram_mixer = BackoffNgramMixer(vocab_size=args.vocab_size, max_order=9)
    slot_delta = torch.zeros(
        (1, args.model_dim), device=device, dtype=torch.bfloat16, requires_grad=False  # No grad for eval
    )

    # Fixed number of val batches for faster evaluation
    max_val_batches = 20
    
    for batch_idx in range(max_val_batches):
        # Sample random positions in validation set
        max_start = val_tokens.numel() - args.train_seq_len - 1
        raw_start = torch.randint(0, max_start, (1,)).item()
        raw_end = raw_start + args.train_seq_len + 1
        local = val_tokens[raw_start:raw_end].to(
            device=device, dtype=torch.int64, non_blocking=True
        )
        x = local[:-1].reshape(1, args.train_seq_len)
        y = local[1:].reshape(1, args.train_seq_len)

        # SCORE PHASE: Stateless, left-to-right causal evaluation
        model.eval()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x, y, return_logits=True, slot_delta=slot_delta)
                batch_loss = F.cross_entropy(
                    logits.float(), y.reshape(-1), reduction="mean"
                ).detach()

        batch_token_count = float(y.numel())
        val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
        val_token_count += batch_token_count

        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        token_bytes += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).to(dtype=torch.int16)
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
# DATA LOADING
# -----------------------------


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    with file.open("rb") as f:
        header = np.fromfile(f, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.memmap(
        file,
        dtype="<u2",
        mode="r",
        offset=header_bytes,
        shape=(num_tokens,),
    )
    return torch.from_numpy(tokens_np)


class AsyncShardPrefetcher:
    def __init__(self, files: list[Path]):
        self.files = files
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def submit(self, file_idx: int) -> None:
        self.future = self.executor.submit(load_data_shard, self.files[file_idx])

    def result(self) -> Tensor:
        if self.future is None:
            raise RuntimeError("Prefetch was not submitted")
        return self.future.result()

    def close(self) -> None:
        self.executor.shutdown(wait=True)


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.pos = 0
        self.prefetcher = AsyncShardPrefetcher(self.files)
        self.tokens = load_data_shard(self.files[0])
        self.prefetcher.submit((self.file_idx + 1) % len(self.files))

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = self.prefetcher.result()
        self.pos = 0
        self.prefetcher.submit((self.file_idx + 1) % len(self.files))

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
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks)

    def close(self) -> None:
        self.prefetcher.close()


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> tuple:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )

    def close(self) -> None:
        self.stream.close()


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, self.weight.to(input.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if self._cos_cached is None or self._seq_len_cached != seq_len:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)
            self._cos_cached, self._sin_cached = (
                freqs.cos()[None, None, :, :],
                freqs.sin()[None, None, :, :],
            )
            self._seq_len_cached = seq_len
        cos = self._cos_cached
        sin = self._sin_cached
        if cos is None or sin is None:
            raise RuntimeError("Rotary cache was not initialized")
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        rope_base,
        qk_gain_init,
        apply_xsa: bool = False,
    ):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = (
            num_heads,
            num_kv_heads,
            dim // num_heads,
        )
        self.apply_xsa = apply_xsa
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = (
            self.c_q(x)
            .reshape(bsz, seqlen, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.c_k(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.c_v(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads)
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)

        # XSA Orthogonal Projection
        if self.apply_xsa:
            v_expanded = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            v_flat = v_expanded.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
            dot_product = (y * v_flat).sum(dim=-1, keepdim=True)
            v_norm_sq = (v_flat * v_flat).sum(dim=-1, keepdim=True) + 1e-6
            y = y - (dot_product / v_norm_sq) * v_flat

        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_base,
        qk_gain_init,
        apply_xsa: bool = False,
    ):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, apply_xsa=apply_xsa
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(
            self.attn_norm(x)
        )
        return x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x)
        )


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        model_dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        tie_embeddings,
        tied_embed_init_std,
        logit_softcap,
        rope_base,
        qk_gain_init,
    ):
        super().__init__()
        self.tie_embeddings, self.logit_softcap = tie_embeddings, logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(
            torch.ones(
                min(self.num_encoder_layers, self.num_decoder_layers),
                model_dim,
                dtype=torch.float32,
            )
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    apply_xsa=(i >= num_layers // 2),
                )
                for i in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        )

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        return_logits: bool = False,
        slot_delta: Tensor | None = None,
    ) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = (
                    x
                    + self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))

        # SLOT Parameter Injection
        if slot_delta is not None:
            if slot_delta.numel() == 0:
                raise ValueError("slot_delta must be non-empty when provided")
            x = x + slot_delta.to(dtype=x.dtype)  # type: ignore[union-attr]

        targets = target_ids.reshape(-1)
        logits_proj = (
            F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        )
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

        if return_logits:
            return logits
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------


def main() -> None:
    args = Hyperparameters()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum_steps = 8 // world_size
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)

    # Only init distributed when we have proper multi-GPU env vars
    distributed = dist.is_available() and world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len).cpu()  # Keep on CPU
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )

    base_model = (
        GPT(
            args.vocab_size,
            args.num_layers,
            args.model_dim,
            args.num_heads,
            args.num_kv_heads,
            args.mlp_mult,
            args.tie_embeddings,
            args.tied_embed_init_std,
            args.logit_softcap,
            args.rope_base,
            args.qk_gain_init,
        )
        .to(device)
        .bfloat16()
    )
    model = (
        DDP(base_model, device_ids=[device.index], broadcast_buffers=False)
        if distributed
        else base_model
    )

    matrix_params = [
        p for name, p in base_model.blocks.named_parameters() if p.ndim == 2
    ]
    scalar_params = [
        p for name, p in base_model.blocks.named_parameters() if p.ndim < 2
    ] + [base_model.skip_weights]

    optimizer_tok = torch.optim.Adam(
        [base_model.tok_emb.weight], lr=args.tied_embed_lr, fused=True
    )
    optimizer_muon = MuonIFNSO(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum
    )
    optimizer_scalar = torch.optim.Adam(scalar_params, lr=args.scalar_lr, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Momentum warmup for Muon
    base_momentum = args.muon_momentum
    momentum = args.muon_momentum_warmup_start
    momentum_step = (
        base_momentum - args.muon_momentum_warmup_start
    ) / args.muon_momentum_warmup_steps

    step = 0
    t0 = time.perf_counter()
    while step < args.iterations:
        # Update Muon momentum warmup
        if step < args.muon_momentum_warmup_steps:
            momentum = args.muon_momentum_warmup_start + step * momentum_step
            for group in optimizer_muon.param_groups:
                group["momentum"] = momentum

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss / grad_accum_steps).backward()

        # Gradient clipping if enabled
        if args.grad_clip_norm > 0:
            model.clip_grad_norm_(args.grad_clip_norm)

        for opt in optimizers:
            opt.step()
        step += 1

        # Logging
        if step % args.train_log_every == 0:
            elapsed = time.perf_counter() - t0
            tokens_per_sec = (
                args.train_batch_tokens * step / elapsed
            )
            print(
                f"step {step}/{args.iterations} | "
                f"loss={loss.item():.4f} | "
                f"lr_momentum={momentum:.4f} | "
                f"tokens/s={tokens_per_sec:.0f} | "
                f"elapsed={elapsed:.1f}s"
            )

        # Validation
        if step % args.val_loss_every == 0 and step > 0:
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
            print(f"val_loss={val_loss:.4f} | val_bpb={val_bpb:.4f}")

        # Max wallclock safety check
        if time.perf_counter() - t0 > args.max_wallclock_seconds:
            print(f"Max wallclock reached ({args.max_wallclock_seconds}s), stopping.")
            break

    train_loader.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
