import glob
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from itertools import accumulate, pairwise
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


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


@dataclass
class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", ".")
    train_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_train_*.bin")
    val_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_val_*.bin")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    val_tokens: int = _env_int("VAL_TOKENS", 10_485_760)
    val_batch_size: int = _env_int("VAL_BATCH_SIZE", 4 * 64 * 1024 * 8)
    val_loss_every: int = _env_int("VAL_LOSS_EVERY", 250)
    num_scheduled_iterations: int = _env_int("NUM_SCHEDULED_ITERATIONS", 1490)
    num_extension_iterations: int = _env_int("NUM_EXTENSION_ITERATIONS", 40)
    save_checkpoint: bool = _env_bool("SAVE_CHECKPOINT", False)
    compile_model: bool = _env_bool("COMPILE_MODEL", False)
    use_amp: bool = _env_bool("USE_AMP", True)

    # Model
    vocab_size: int = _env_int("VOCAB_SIZE", 50257)
    num_layers: int = _env_int("NUM_LAYERS", 11)
    model_dim: int = _env_int("MODEL_DIM", 768)
    num_heads: int = _env_int("NUM_HEADS", 6)
    mlp_mult: int = _env_int("MLP_MULT", 4)
    logit_chunk_tokens: int = _env_int("LOGIT_CHUNK_TOKENS", 32768)
    logit_softcap: float = _env_float("LOGIT_SOFTCAP", 30.0)
    rope_base: float = _env_float("ROPE_BASE", 10000.0)

    # Optimizer
    base_lr: float = _env_float("BASE_LR", 0.008)
    beta1: float = _env_float("BETA1", 0.9)
    beta2: float = _env_float("BETA2", 0.95)
    eps: float = _env_float("ADAM_EPS", 1e-8)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.01)
    grad_clip_norm: float = _env_float("GRAD_CLIP_NORM", 1.0)


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
        cooldown_frac: float = 0.55,
    ) -> None:
        self.stages = stages
        self.scheduled_iterations = scheduled_iterations
        self.cooldown_frac = cooldown_frac
        self.total_steps = scheduled_iterations + extension_iterations

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
        cd_start = int(self.scheduled_iterations * (1 - self.cooldown_frac))
        if step >= cd_start:
            t = min(1.0, (step - cd_start) / max(self.scheduled_iterations - cd_start, 1))
            lr_mul = lr_mul * (1 - t) + 0.15 * t
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
        x = local[:-1].view(-1, seq_len)
        y = local[1:].view(-1, seq_len)
        return (
            x.to(device=self.device, non_blocking=True),
            y.to(device=self.device, non_blocking=True),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, eps=self.eps)


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
    def __init__(self, dim: int, num_heads: int, rope_base: float):
        super().__init__()
        assert dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        # Keep the simple relu^2 nonlinearity.
        x = torch.relu(self.fc(x))
        x = x.square()
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, rope_base: float):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, rope_base=rope_base)
        self.mlp = MLP(dim, mlp_mult)
        self.resid_mix = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
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
        self.skip_weights = nn.Parameter(torch.ones(self.num_encoder_layers))
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, mlp_mult, rope_base=rope_base) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head._zero_init = True
        self._init_weights(num_layers)

    def _init_weights(self, num_layers: int) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        scale = 0.02 / math.sqrt(2 * num_layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=scale)

    def _apply_softcap(self, logits: Tensor) -> Tensor:
        if self.logit_softcap <= 0:
            return logits
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        _, seqlen = input_ids.shape
        if seqlen > self.max_seq_len:
            raise ValueError(f"Input sequence length {seqlen} exceeds max_seq_len {self.max_seq_len}")
        x = self.tok_emb(input_ids)
        x0 = x
        skip_connections: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            if skip_connections:
                skip_idx = len(skip_connections) - 1
                skip_weight = self.skip_weights[skip_idx].to(dtype=x.dtype)
                x = x + skip_weight * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).view(-1, x.size(-1))
        targets = target_ids.view(-1)
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
    def __init__(self, schedule: TrainingSchedule, optimizer: torch.optim.Optimizer, args: Hyperparameters):
        self.schedule = schedule
        self.optimizer = optimizer
        self.args = args
        stage0 = schedule.stages[0]
        self.batch_size = stage0.batch_size
        self.train_max_seq_len = stage0.train_max_seq_len

    def advance_schedule(self, step: int) -> None:
        stage, _ = self.schedule.lookup(step)
        self.batch_size = stage.batch_size
        self.train_max_seq_len = stage.train_max_seq_len

    def step_optimizers(self, step: int, params) -> None:
        step_lr = self.args.base_lr * self.schedule.get_lr_mul(step)
        for group in self.optimizer.param_groups:
            group["lr"] = step_lr
        if self.args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self.args.grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


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
        TrainingStage(duration=1 / 3, train_max_seq_len=896, batch_size=8 * 2048 * 8, lr_mul=1.0),
        TrainingStage(duration=1 / 3, train_max_seq_len=2048, batch_size=16 * 2048 * 8, lr_mul=1.52),
        TrainingStage(duration=1 / 3, train_max_seq_len=2048, batch_size=24 * 2048 * 8, lr_mul=1.73),
        TrainingStage(train_max_seq_len=2048, batch_size=24 * 2048 * 8, lr_mul=1.0),
    ]
    schedule = TrainingSchedule(
        training_stages,
        args.num_scheduled_iterations,
        args.num_extension_iterations,
        cooldown_frac=0.55,
    )

    max_seq_len = max(stage.train_max_seq_len for stage in training_stages)
    raw_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        mlp_mult=args.mlp_mult,
        max_seq_len=max_seq_len,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
    ).to(device)

    if args.compile_model:
        raw_model = torch.compile(raw_model, dynamic=False)

    model: nn.Module
    if distributed:
        model = DDP(raw_model, device_ids=[local_rank], broadcast_buffers=False)
    else:
        model = raw_model

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.base_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    training_manager = TrainingManager(schedule, optimizer, args)

    n_params = sum(p.numel() for p in raw_model.parameters())
    print0(f"model_params:{n_params}", console=True)
    print0(
        f"world_size:{world_size} grad_accum_steps:{grad_accum_steps} "
        f"compile_model:{args.compile_model} use_amp:{args.use_amp}",
        console=True,
    )

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
            with torch.no_grad():
                for _ in range(val_steps):
                    x, y = val_loader.next_batch(
                        args.val_batch_size,
                        training_manager.train_max_seq_len,
                        grad_accum_steps,
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                        val_loss += model(x, y).detach()
            val_loss /= val_steps
            if distributed:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            print0(
                f"step:{step}/{train_steps} val_loss:{val_loss.item():.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms",
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

        optimizer.zero_grad(set_to_none=True)
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
            (loss * grad_scale).backward()
        training_manager.step_optimizers(step, raw_model.parameters())

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        print0(
            f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms "
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

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
