import io
import glob
import math
import os
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from itertools import accumulate, pairwise
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
import sentencepiece as spm

def _read_code() -> str:
    with open(__file__, "r", encoding="utf-8") as f:
        return f.read()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def zeropower_via_svd(G: Tensor, steps=None) -> Tensor:
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    return U @ Vh


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz iteration to approximate the orthogonal factor of a matrix.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


zeropower_backends = {
    "svd": zeropower_via_svd,
    "newtonschulz5": zeropower_via_newtonschulz5,
}


class Muon(torch.optim.Optimizer):
    """
    SGD-momentum + orthogonalized updates for 2D parameters.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        backend: str = "newtonschulz5",
        backend_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps,
        )
        super().__init__(params, defaults)

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
            zeropower_backend = zeropower_backends[group["backend"]]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr_idx = 0
            for i, p in enumerate(params):
                if i % world_size == rank:
                    g = p.grad
                    if g is not None:
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(g)
                        if group["nesterov"]:
                            g = g.add(buf, alpha=momentum)
                        g = zeropower_backend(g, steps=group["backend_steps"])
                        g *= max(1, g.size(0) / g.size(1)) ** 0.5
                        updates_flat[curr_idx : curr_idx + p.numel()] = g.reshape(-1)
                curr_idx += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in params:
                g = updates_flat[curr_idx : curr_idx + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr_idx += p.numel()

        return loss


# NOTE: BPB accounting must match the tokenizer used to create the token shards.
# `data/fineweb10B` shards are GPT-2 tokenized (see data/cached_fineweb10B.py), while
# matched SentencePiece experiments should set TOKENIZER_KIND=sentencepiece and
# TOKENIZER_PATH to the corresponding `.model`.
TOKENIZER_KIND = os.environ.get("TOKENIZER_KIND", "gpt2").lower()


def _default_sentencepiece_tokenizer_path() -> str:
    # Prefer the local matched-export tokenizer paths that exist in this repo state, while
    # keeping the older path as a fallback for other checkouts.
    candidates = (
        "data/matched_10B/tokenizers/fineweb_1024_bpe.model",
        "data/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model",
        "data/tokenizers/fineweb_1k_bpe.model",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return candidates[0]


# Allow an env override so SentencePiece BPB evals can point at the exact tokenizer.
FIXED_TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", _default_sentencepiece_tokenizer_path())
# Cache key includes tokenizer identity/path + vocab size so multi-tokenizer checks in a
# single process cannot reuse a stale lookup table from a different tokenizer.
_BYTES_LUT_CACHE: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
def bytes_per_token(input_ids: Tensor, target_ids: Tensor) -> Tensor:
    """
    Return per-target UTF-8 byte counts for compression metrics.
    Must return a tensor with the same shape as target_ids.
    """
    if input_ids.shape != target_ids.shape:
        raise ValueError(f"shape mismatch: input_ids={tuple(input_ids.shape)} target_ids={tuple(target_ids.shape)}")

    model_vocab_size = _env_int("VOCAB_SIZE", 50304)
    if TOKENIZER_KIND in {"gpt2", "gpt-2"}:
        tokenizer_cache_key = "gpt2"
    elif TOKENIZER_KIND in {"sentencepiece", "spm", "sp"}:
        tokenizer_cache_key = f"sentencepiece:{Path(FIXED_TOKENIZER_PATH).resolve()}"
    else:
        tokenizer_cache_key = TOKENIZER_KIND

    device_key = str(target_ids.device)
    cache_key = (tokenizer_cache_key, device_key, model_vocab_size)
    cached = _BYTES_LUT_CACHE.get(cache_key)
    if cached is None:
        if TOKENIZER_KIND in {"gpt2", "gpt-2"}:
            try:
                import tiktoken
            except ImportError as exc:  # pragma: no cover - runtime dependency check
                raise ImportError(
                    "tiktoken is required for TOKENIZER_KIND=gpt2 bytes-per-token accounting"
                ) from exc

            enc = tiktoken.get_encoding("gpt2")
            table_size = max(int(enc.n_vocab), model_vocab_size)
            base_bytes = [0] * table_size
            # GPT-2 token byte lengths are intrinsic to each token; no sentencepiece-style
            # boundary/leading-space reconstruction is needed.
            has_leading_space = [False] * table_size
            is_boundary_token = [True] * table_size

            eot_id = int(enc.eot_token)
            for token_id in range(min(int(enc.n_vocab), table_size)):
                if token_id == eot_id:
                    # EOT is a synthetic document boundary marker, not literal text bytes.
                    base_bytes[token_id] = 0
                    is_boundary_token[token_id] = True
                    continue
                piece_bytes = enc.decode_single_token_bytes(token_id)
                base_bytes[token_id] = len(piece_bytes)
                is_boundary_token[token_id] = False
        elif TOKENIZER_KIND in {"sentencepiece", "spm", "sp"}:
            tokenizer_path = Path(FIXED_TOKENIZER_PATH)
            if not tokenizer_path.is_file():
                raise FileNotFoundError(
                    f"Tokenizer not found at {tokenizer_path}. "
                    "Set TOKENIZER_PATH to the SentencePiece model used to build the shards."
                )
            sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
            sp_vocab_size = int(sp.vocab_size())
            # Model vocab can be padded larger than the SentencePiece vocab (for example
            # 50,304 model slots for a 1k BPE tokenizer), so size the LUT by model vocab.
            table_size = max(sp_vocab_size, model_vocab_size)

            base_bytes = [0] * table_size
            has_leading_space = [False] * table_size
            # True means this token acts as a text boundary for a following leading-space token.
            is_boundary_token = [True] * table_size

            for token_id in range(sp_vocab_size):
                is_control = sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id)
                if is_control:
                    base_bytes[token_id] = 0
                    is_boundary_token[token_id] = True
                    continue

                is_boundary_token[token_id] = False
                if sp.is_byte(token_id):
                    base_bytes[token_id] = 1
                    continue

                piece = sp.id_to_piece(token_id)
                leading = piece.startswith("▁")
                if leading:
                    piece = piece[1:]
                has_leading_space[token_id] = leading
                base_bytes[token_id] = len(piece.encode("utf-8"))
        else:
            raise ValueError(
                f"Unsupported TOKENIZER_KIND={TOKENIZER_KIND!r}. "
                "Expected 'gpt2' or 'sentencepiece'."
            )

        cached = (
            torch.tensor(base_bytes, dtype=torch.int16, device=target_ids.device),
            torch.tensor(has_leading_space, dtype=torch.bool, device=target_ids.device),
            torch.tensor(is_boundary_token, dtype=torch.bool, device=target_ids.device),
        )
        _BYTES_LUT_CACHE[cache_key] = cached

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = cached
    flat_prev = input_ids.reshape(-1)
    flat_target = target_ids.reshape(-1)

    out = base_bytes_lut[flat_target].to(dtype=torch.int16)
    needs_space = has_leading_space_lut[flat_target]
    prev_allows_space = ~is_boundary_token_lut[flat_prev]
    out = out + (needs_space & prev_allows_space).to(dtype=torch.int16)
    return out.reshape_as(target_ids)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def quantize_state_dict_int8_per_tensor(state_dict: dict[str, Tensor]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    orig_dtypes: dict[str, str] = {}
    nonfloat: dict[str, Tensor] = {}

    stats = {
        "param_count": 0,
        "num_tensors": 0,
        "num_float_tensors": 0,
        "num_nonfloat_tensors": 0,
        "baseline_tensor_bytes": 0,
        "int8_payload_bytes": 0,
    }

    for name, tensor in state_dict.items():
        t = tensor.detach().to(device="cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += _tensor_nbytes(t)
        if t.is_floating_point():
            stats["num_float_tensors"] += 1
            t32 = t.to(dtype=torch.float32)
            max_abs = float(t32.abs().max().item()) if t32.numel() > 0 else 0.0
            scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
            q = torch.clamp(torch.round(t32 / scale), -127, 127).to(torch.int8).contiguous()
            s = torch.tensor(scale, dtype=torch.float32)
            quantized[name] = q
            scales[name] = s
            orig_dtypes[name] = _dtype_name(t.dtype)
            stats["int8_payload_bytes"] += _tensor_nbytes(q) + _tensor_nbytes(s)
        else:
            stats["num_nonfloat_tensors"] += 1
            nonfloat[name] = t
            stats["int8_payload_bytes"] += _tensor_nbytes(t)

    quant_obj: dict[str, object] = {
        "__quant_format__": "int8_sym_per_tensor_v1",
        "quantized": quantized,
        "scales": scales,
        "orig_dtypes": orig_dtypes,
        "nonfloat": nonfloat,
    }
    return quant_obj, stats


def dequantize_state_dict_int8_per_tensor(quant_obj: dict[str, object]) -> dict[str, Tensor]:
    if quant_obj.get("__quant_format__") != "int8_sym_per_tensor_v1":
        raise ValueError(f"unsupported quant format: {quant_obj.get('__quant_format__')!r}")
    quantized = quant_obj["quantized"]
    scales = quant_obj["scales"]
    orig_dtypes = quant_obj["orig_dtypes"]
    nonfloat = quant_obj["nonfloat"]
    if not isinstance(quantized, dict) or not isinstance(scales, dict) or not isinstance(orig_dtypes, dict):
        raise TypeError("invalid int8 quantized checkpoint object")

    out: dict[str, Tensor] = {}
    for name, q in quantized.items():
        if not torch.is_tensor(q):
            raise TypeError(f"quantized tensor missing for {name}")
        s = scales[name]
        scale = float(s.item()) if torch.is_tensor(s) else float(s)
        dtype_name = orig_dtypes[name]
        t = q.to(dtype=torch.float32) * scale
        out[name] = t.to(dtype=_dtype_from_name(dtype_name)).contiguous()
    for name, t in nonfloat.items():
        if not torch.is_tensor(t):
            raise TypeError(f"nonfloat tensor missing for {name}")
        out[name] = t.detach().to(device="cpu").contiguous()
    return out


def serialize_quantized_int8_zlib(quant_obj: dict[str, object]) -> tuple[bytes, int]:
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw_bytes = buf.getvalue()
    return zlib.compress(raw_bytes, level=9), len(raw_bytes)


def deserialize_quantized_int8_zlib(blob: bytes) -> dict[str, Tensor]:
    raw = zlib.decompress(blob)
    obj = torch.load(io.BytesIO(raw), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"unexpected decoded object type: {type(obj)}")
    return dequantize_state_dict_int8_per_tensor(obj)


@dataclass
class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/fineweb10B/")
    train_files: str = os.path.join(data_path, "fineweb_train_*.bin")
    val_files: str = os.path.join(data_path, "fineweb_val_*.bin")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    val_tokens: int = _env_int("VAL_TOKENS", 10_485_760)
    val_batch_size: int = _env_int("VAL_BATCH_SIZE", 64 * 1024 * 8)
    val_loss_every: int = _env_int("VAL_LOSS_EVERY", 125)
    num_scheduled_iterations: int = _env_int("ITERATIONS", 3000)
    num_extension_iterations: int = _env_int("NUM_EXTENSION_ITERATIONS", 0)
    warmdown_iters: int = _env_int("WARMDOWN_ITERS", 900)
    warmup_steps: int = _env_int("WARMUP_STEPS", _env_int("TIMING_WARMUP_STEPS", 50))
    save_checkpoint: bool = _env_bool("SAVE_CHECKPOINT", True)
    use_amp: bool = _env_bool("USE_AMP", True)
    enable_val_bpb: bool = _env_bool("ENABLE_VAL_BPB", False)
    auto_final_int8_zlib: bool = _env_bool("AUTO_FINAL_INT8_ZLIB", True)

    # Model
    vocab_size: int = _env_int("VOCAB_SIZE", 50304)
    num_layers: int = _env_int("NUM_LAYERS", 12)
    model_dim: int = _env_int("MODEL_DIM", 768)
    num_heads: int = _env_int("NUM_HEADS", 6)
    # 0 means "use full MHA" (NUM_KV_HEADS = NUM_HEADS). Set 1 for MQA, or
    # an exact divisor of NUM_HEADS for GQA.
    num_kv_heads: int = _env_int("NUM_KV_HEADS", 0)
    mlp_mult: int = _env_int("MLP_MULT", 4)
    # 0 disables chunking and is materially faster on current H100 stacks.
    logit_chunk_tokens: int = _env_int("LOGIT_CHUNK_TOKENS", 0)
    logit_softcap: float = _env_float("LOGIT_SOFTCAP", 30.0)
    rope_base: float = _env_float("ROPE_BASE", 10000.0)

    # Optimizer
    base_lr: float = _env_float("BASE_LR", 0.04)
    embed_lr: float = _env_float("EMBED_LR", 0.6)
    head_lr: float = _env_float("HEAD_LR", 0.008)
    matrix_lr: float = _env_float("MATRIX_LR", 0.04)
    scalar_lr: float = _env_float("SCALAR_LR", 0.04)
    muon_momentum: float = _env_float("MUON_MOMENTUM", 0.95)
    muon_nesterov: bool = _env_bool("MUON_NESTEROV", True)
    muon_backend: str = os.environ.get("MUON_BACKEND", "newtonschulz5")
    muon_backend_steps: int = _env_int("MUON_BACKEND_STEPS", 5)
    muon_momentum_warmup_start: float = _env_float("MUON_MOMENTUM_WARMUP_START", 0.85)
    muon_momentum_warmup_steps: int = _env_int("MUON_MOMENTUM_WARMUP_STEPS", 500)
    beta1: float = _env_float("BETA1", 0.9)
    beta2: float = _env_float("BETA2", 0.95)
    eps: float = _env_float("ADAM_EPS", 1e-8)
    grad_clip_norm: float = _env_float("GRAD_CLIP_NORM", 0.0)


@dataclass
class TrainingStage:
    lr_mul: float
    batch_size: int
    train_max_seq_len: int
    duration: float | None = None


class TrainingSchedule:
    def __init__(
        self,
        stages: list[TrainingStage],
        scheduled_iterations: int,
        extension_iterations: int,
        warmdown_iters: int,
    ) -> None:
        self.stages = stages
        self.scheduled_iterations = scheduled_iterations
        self.warmdown_iters = warmdown_iters
        self.total_steps = scheduled_iterations + extension_iterations

        if len(stages) == 1:
            self.boundaries = [(0, self.total_steps)]
        else:
            ends = [0] + [
                round(c * scheduled_iterations) for c in accumulate(s.duration for s in stages[:-1])
            ] + [self.total_steps]
            assert ends[-2] == scheduled_iterations, "Stage durations must sum to 1.0"
            self.boundaries = list(pairwise(ends))

    def lookup(self, step: int) -> tuple[TrainingStage, float]:
        for i, (start, end) in enumerate(self.boundaries):
            if step < end:
                return self.stages[i], (step - start) / max(end - start, 1)
        return self.stages[-1], 1.0

    def get_lr_mul(self, step: int) -> float:
        stage, _ = self.lookup(step)
        lr_mul = stage.lr_mul
        if self.warmdown_iters > 0:
            warmdown_start = max(self.scheduled_iterations - self.warmdown_iters, 0)
            if warmdown_start <= step < self.scheduled_iterations:
                decay_ratio = (self.scheduled_iterations - step) / max(self.warmdown_iters, 1)
                lr_mul *= max(decay_ratio, 0.0)
        return lr_mul


def _load_data_shard(file: Path) -> torch.Tensor:
    header = torch.from_file(str(file), shared=False, size=256, dtype=torch.int32)
    assert header[0] == 20240520, f"magic number mismatch for {file}"
    assert header[1] == 1, f"unsupported shard version for {file}"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, f"token count mismatch for {file}"
    return tokens


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = _load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
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
        return torch.cat(chunks, dim=0)


class DistributedTokenLoader:
    """
    Simple distributed loader:
    - Reads sequential token shards.
    - Splits each global microbatch into disjoint rank-local slices.
    - Packs into fixed-length [B, T] tensors.
    """

    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        assert global_tokens % (self.world_size * grad_accum_steps) == 0, (
            "global_tokens must be divisible by world_size * grad_accum_steps"
        )
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        usable_tokens = (local_tokens // seq_len) * seq_len
        if usable_tokens == 0:
            raise ValueError(f"Sequence length {seq_len} is too long for local token budget {local_tokens}")

        per_rank_span = usable_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return (
            x.to(device=self.device, non_blocking=True),
            y.to(device=self.device, non_blocking=True),
        )


def run_validation_pass(
    model: nn.Module,
    args: Hyperparameters,
    grad_accum_steps: int,
    seq_len: int,
    rank: int,
    world_size: int,
    device: torch.device,
    enable_val_bpb: bool,
    print_fn=None,
) -> tuple[float, float | None]:
    assert args.val_tokens % args.val_batch_size == 0, "VAL_TOKENS must be divisible by VAL_BATCH_SIZE"
    val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
    val_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)
    val_loss = torch.zeros((), device=device)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    val_bpb_enabled = enable_val_bpb

    model.eval()
    with torch.no_grad():
        for _ in range(val_steps):
            x, y = val_loader.next_batch(args.val_batch_size, seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                val_loss += model(x, y).detach()
            if val_bpb_enabled:
                try:
                    bytes_tensor = bytes_per_token(x, y)
                    if bytes_tensor.shape != y.shape:
                        raise ValueError(
                            f"bytes_per_token returned shape {tuple(bytes_tensor.shape)}, "
                            f"expected {tuple(y.shape)}"
                        )
                    val_token_count += float(y.numel())
                    val_byte_count += bytes_tensor.to(dtype=torch.float64).sum()
                except Exception as exc:  # pragma: no cover - runtime fallback
                    val_bpb_enabled = False
                    if print_fn is not None:
                        print_fn(f"val_bpb:disabled error={exc}", True)

    val_loss /= val_steps
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_bpb: float | None = None
    if val_bpb_enabled and val_byte_count.item() > 0:
        bits_per_token = val_loss.item() / math.log(2.0)
        tokens_per_byte = val_token_count.item() / val_byte_count.item()
        val_bpb = bits_per_token * tokens_per_byte
    elif val_bpb_enabled and val_byte_count.item() <= 0 and print_fn is not None:
        print_fn("val_bpb:disabled zero_byte_count", True)

    return float(val_loss.item()), val_bpb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
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
    x1 = x[..., :half]
    x2 = x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float):
        super().__init__()
        assert dim % num_heads == 0, "model_dim must be divisible by num_heads"
        if num_kv_heads <= 0:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("NUM_HEADS must be divisible by NUM_KV_HEADS for GQA/MQA")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_per_kv = num_heads // num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.v_mix = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, v1: Tensor | None) -> tuple[Tensor, Tensor]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if v1 is None:
            v1 = v
        mix = self.v_mix.to(dtype=v.dtype)
        v = (1 - mix) * v + mix * v1
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y), v1


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        # Keep the simple relu^2 nonlinearity.
        x = torch.relu(self.fc(x))
        x = x.square()
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base=rope_base)
        self.mlp = MLP(dim, mlp_mult)
        self.resid_mix = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor, v1: Tensor | None) -> tuple[Tensor, Tensor]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0
        attn_out, v1 = self.attn(self.attn_norm(x), v1)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, v1


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        max_seq_len: int,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads <= 0 else num_kv_heads
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, self.num_kv_heads, mlp_mult, rope_base=rope_base) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(model_dim)
        self.lm_head = CastedLinear(model_dim, vocab_size, bias=False)
        self.lm_head._zero_init = True
        self._init_weights(num_layers)

    def _init_weights(self, num_layers: int) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)

    def _apply_softcap(self, logits: Tensor) -> Tensor:
        if self.logit_softcap <= 0:
            return logits
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        _, seqlen = input_ids.shape
        if seqlen > self.max_seq_len:
            raise ValueError(f"Input sequence length {seqlen} exceeds max_seq_len {self.max_seq_len}")
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None
        skip_connections: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x, v1 = self.blocks[i](x, x0, v1)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            if skip_connections:
                skip = skip_connections.pop()
                skip_weight = self.skip_weights[i].to(dtype=x.dtype)
                x = x + skip_weight * skip
            x, v1 = self.blocks[self.num_encoder_layers + i](x, x0, v1)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        chunk_tokens = self.logit_chunk_tokens

        if chunk_tokens <= 0 or x.size(0) <= chunk_tokens:
            logits = self._apply_softcap(self.lm_head(x))
            return F.cross_entropy(logits.float(), targets, reduction="mean")

        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        for start in range(0, x.size(0), chunk_tokens):
            end = min(start + chunk_tokens, x.size(0))
            logits = self._apply_softcap(self.lm_head(x[start:end]))
            loss_sum += F.cross_entropy(logits.float(), targets[start:end], reduction="sum")
        return loss_sum / x.size(0)


class TrainingManager:
    def __init__(
        self,
        schedule: TrainingSchedule,
        optimizers: list[torch.optim.Optimizer],
        args: Hyperparameters,
    ):
        self.schedule = schedule
        self.optimizers = optimizers
        self.args = args
        stage0 = schedule.stages[0]
        self.batch_size = stage0.batch_size
        self.train_max_seq_len = stage0.train_max_seq_len

    def advance_schedule(self, step: int) -> None:
        stage, _ = self.schedule.lookup(step)
        self.batch_size = stage.batch_size
        self.train_max_seq_len = stage.train_max_seq_len

    def zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step_optimizers(self, step: int, params) -> None:
        lr_mul = self.schedule.get_lr_mul(step)
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_mul

        if self.args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self.args.grad_clip_norm)
        for opt in self.optimizers:
            opt.step()
        self.zero_grad()


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    return distributed, rank, world_size, local_rank, device


def nvidia_smi() -> str:
    import subprocess

    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


def main() -> None:
    code = _read_code()
    args = Hyperparameters()
    distributed, rank, world_size, local_rank, device = init_distributed()
    master_process = rank == 0

    assert 8 % world_size == 0, "world_size must divide 8 to keep grad accumulation parity"
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(True)
    enable_flash_sdp(False)
    enable_mem_efficient_sdp(False)
    # Keep a safe fallback for shapes unsupported by cuDNN SDPA (e.g. head_dim=92).
    enable_math_sdp(True)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def print0(msg: str, console: bool = False) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    print0(code)
    print0("=" * 100)
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.__version__}")
    print0(nvidia_smi())
    print0("=" * 100)

    training_stages = [
        TrainingStage(train_max_seq_len=1024, batch_size=8 * 64 * 1024, lr_mul=1.0),
    ]
    schedule = TrainingSchedule(
        training_stages,
        args.num_scheduled_iterations,
        args.num_extension_iterations,
        warmdown_iters=args.warmdown_iters,
    )

    max_seq_len = max(stage.train_max_seq_len for stage in training_stages)
    raw_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        max_seq_len=max_seq_len,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
    ).to(device).bfloat16()
    for module in raw_model.modules():
        if isinstance(module, CastedLinear):
            module.float()

    raw_model = torch.compile(raw_model, dynamic=False, fullgraph=True)

    model: nn.Module
    if distributed:
        model = DDP(raw_model, device_ids=[local_rank], broadcast_buffers=False)
    else:
        model = raw_model

    block_params = list(raw_model.blocks.parameters())
    matrix_params = [p for p in block_params if p.ndim == 2]
    scalar_params = [p for p in block_params if p.ndim < 2] + [raw_model.skip_weights]
    optimizer_tok = torch.optim.Adam(
        [{"params": [raw_model.tok_emb.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        fused=True,
    )
    optimizer_head = torch.optim.Adam(
        [{"params": [raw_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        nesterov=args.muon_nesterov,
        backend=args.muon_backend,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [
        optimizer_tok,
        optimizer_head,
        optimizer_muon,
        optimizer_scalar,
    ]
    training_manager = TrainingManager(schedule, optimizers, args)

    n_params = sum(p.numel() for p in raw_model.parameters())
    attn_mode = (
        "mha"
        if raw_model.num_kv_heads == raw_model.num_heads
        else ("mqa" if raw_model.num_kv_heads == 1 else "gqa")
    )
    print0(f"model_params:{n_params}", console=True)
    print0(
        f"attention_mode:{attn_mode} num_heads:{raw_model.num_heads} num_kv_heads:{raw_model.num_kv_heads}",
        console=True,
    )
    print0(
        f"world_size:{world_size} grad_accum_steps:{grad_accum_steps} "
        f"compile_model:True compile_dynamic:False compile_fullgraph:True "
        f"use_amp:{args.use_amp}",
        console=True,
    )
    val_bpb_enabled = args.enable_val_bpb
    if val_bpb_enabled:
        msg = f"val_bpb:enabled tokenizer_kind={TOKENIZER_KIND}"
        if TOKENIZER_KIND in {"sentencepiece", "spm", "sp"}:
            msg += f" tokenizer_path={FIXED_TOKENIZER_PATH}"
        print0(msg, console=True)
    else:
        print0("val_bpb:disabled by ENABLE_VAL_BPB=0", console=True)
    print0(
        f"auto_final_int8_zlib:{args.auto_final_int8_zlib} "
        f"(disable with AUTO_FINAL_INT8_ZLIB=0)",
        console=True,
    )

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    warmup_steps = max(args.warmup_steps, 0)
    if warmup_steps > 0:
        print0(f"warmup_steps:{warmup_steps}", console=True)
        training_manager.advance_schedule(0)
        model.train()
        for warmup_step in range(warmup_steps):
            training_manager.zero_grad()
            for micro_step in range(grad_accum_steps):
                if isinstance(model, DDP):
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(
                    training_manager.batch_size,
                    training_manager.train_max_seq_len,
                    grad_accum_steps,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            training_manager.zero_grad()
            if (
                warmup_steps <= 20
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == warmup_steps
            ):
                print0(f"warmup_step:{warmup_step + 1}/{warmup_steps}", console=True)
        training_manager.zero_grad()
        if isinstance(model, DDP):
            model.require_backward_grad_sync = True
        # Start measured training from the beginning of the dataset.
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    train_steps = schedule.total_steps
    for step in range(train_steps + 1):
        last_step = step == train_steps
        training_manager.advance_schedule(step)

        run_eval = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if run_eval:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)

            model.eval()
            assert args.val_tokens % args.val_batch_size == 0, "VAL_TOKENS must be divisible by VAL_BATCH_SIZE"
            val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
            val_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)
            val_loss = torch.zeros((), device=device)
            val_token_count = torch.zeros((), device=device, dtype=torch.float64)
            val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
            with torch.no_grad():
                for _ in range(val_steps):
                    x, y = val_loader.next_batch(
                        args.val_batch_size,
                        training_manager.train_max_seq_len,
                        grad_accum_steps,
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                        val_loss += model(x, y).detach()
                    if val_bpb_enabled:
                        try:
                            bytes_tensor = bytes_per_token(x, y)
                            if bytes_tensor.shape != y.shape:
                                raise ValueError(
                                    f"bytes_per_token returned shape {tuple(bytes_tensor.shape)}, "
                                    f"expected {tuple(y.shape)}"
                                )
                            val_token_count += float(y.numel())
                            val_byte_count += bytes_tensor.to(dtype=torch.float64).sum()
                        except Exception as exc:  # pragma: no cover - runtime fallback
                            val_bpb_enabled = False
                            print0(f"val_bpb:disabled error={exc}", console=True)
            val_loss /= val_steps
            if distributed:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

            val_bpb_str = ""
            if val_bpb_enabled and val_byte_count.item() > 0:
                bits_per_token = val_loss.item() / math.log(2.0)
                tokens_per_byte = val_token_count.item() / val_byte_count.item()
                val_bpb = bits_per_token * tokens_per_byte
                val_bpb_str = f" val_bpb:{val_bpb:.4f}"
            elif val_bpb_enabled and val_byte_count.item() <= 0:
                val_bpb_enabled = False
                print0("val_bpb:disabled zero_byte_count", console=True)
            print0(
                f"step:{step}/{train_steps} val_loss:{val_loss.item():.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
                f"{val_bpb_str}",
                console=True,
            )
            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                ckpt = {"step": step, "code": code, "model": raw_model.state_dict()}
                os.makedirs(f"logs/{args.run_id}", exist_ok=True)
                torch.save(ckpt, f"logs/{args.run_id}/state_step{step:06d}.pt")
            break

        training_manager.zero_grad()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if isinstance(model, DDP):
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                training_manager.batch_size,
                training_manager.train_max_seq_len,
                grad_accum_steps,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                loss = model(x, y)
            train_loss = loss.detach()
            (loss * grad_scale).backward()
        if args.muon_momentum_warmup_steps > 0:
            frac = min(step / args.muon_momentum_warmup_steps, 1.0)
            momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        else:
            momentum = args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = momentum
        training_manager.step_optimizers(step, raw_model.parameters())

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        print0(
            f"step:{step + 1}/{train_steps} train_loss:{train_loss.item():.4f} "
            f"train_time:{approx_training_time_ms:.0f}ms "
            f"step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
            console=True,
        )

    print0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
        console=True,
    )

    if master_process:
        torch.save(raw_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        print0(f"Serialized model: {model_bytes} bytes", console=True)
        print0(f"Code size: {code_bytes} bytes", console=True)
        print0(f"Total submission size: {model_bytes + code_bytes} bytes", console=True)

    if args.auto_final_int8_zlib:
        state_for_quant = raw_model.state_dict()
        quant_obj, quant_stats = quantize_state_dict_int8_per_tensor(state_for_quant)
        quant_blob, quant_serialized_bytes = serialize_quantized_int8_zlib(quant_obj)
        quantized_path = "final_model.int8.ptz"
        if master_process:
            with open(quantized_path, "wb") as f:
                f.write(quant_blob)
            quant_file_bytes = os.path.getsize(quantized_path)
            code_bytes = len(code.encode("utf-8"))
            ratio = (
                quant_stats["baseline_tensor_bytes"] / quant_stats["int8_payload_bytes"]
                if quant_stats["int8_payload_bytes"] > 0
                else 0.0
            )
            print0(
                f"Serialized model int8+zlib: {quant_file_bytes} bytes "
                f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_serialized_bytes} "
                f"payload_ratio:{ratio:.2f}x)",
                console=True,
            )
            print0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes", console=True)

        quant_state = deserialize_quantized_int8_zlib(quant_blob)
        try:
            load_res = raw_model.load_state_dict(quant_state, strict=True)
        except RuntimeError:
            first_key = next(iter(quant_state))
            if isinstance(first_key, str) and first_key.startswith("_orig_mod."):
                stripped = {k.removeprefix("_orig_mod."): v for k, v in quant_state.items()}
                if hasattr(raw_model, "_orig_mod"):
                    load_res = raw_model._orig_mod.load_state_dict(stripped, strict=True)
                else:
                    raise
            else:
                raise
        if master_process:
            print0(
                f"final_int8_zlib_roundtrip load_ok "
                f"missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)}",
                console=True,
            )

        torch.cuda.synchronize()
        t_qeval = time.perf_counter()
        quant_val_loss, quant_val_bpb = run_validation_pass(
            model=model,
            args=args,
            grad_accum_steps=grad_accum_steps,
            seq_len=training_manager.train_max_seq_len,
            rank=rank,
            world_size=world_size,
            device=device,
            enable_val_bpb=args.enable_val_bpb,
            print_fn=print0,
        )
        torch.cuda.synchronize()
        quant_eval_ms = 1000.0 * (time.perf_counter() - t_qeval)
        quant_bpb_str = f" val_bpb:{quant_val_bpb:.4f}" if quant_val_bpb is not None else ""
        print0(
            f"final_int8_zlib_roundtrip val_loss:{quant_val_loss:.4f}"
            f"{quant_bpb_str} eval_time:{quant_eval_ms:.0f}ms",
            console=True,
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
