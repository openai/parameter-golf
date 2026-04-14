from __future__ import annotations
import copy
import glob
import io
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
import lzma
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

torch.set_float32_matmul_precision("high")

_FA_VERSION = 0
_fa_func = None
try:
    from flash_attn_interface import flash_attn_func as _fa_func
    _FA_VERSION = 3
except ImportError:
    try:
        from flash_attn import flash_attn_func as _fa_func
        _FA_VERSION = 2
    except ImportError:
        _FA_VERSION = 0
        _fa_func = None

try:
    import triton  # type: ignore  # noqa: F401
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

try:
    import zstandard as zstd  # type: ignore
    _HAS_ZSTD = True
except Exception:
    zstd = None
    _HAS_ZSTD = False

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    njit = None
    _HAS_NUMBA = False


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").exists():
            return candidate
    return start


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_SCRIPT_DIR)
_DEFAULT_DATA_PATH = _REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"
_DEFAULT_TOKENIZER_PATH = _REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"
_DEFAULT_CKPT_DIR = Path("/workspace/checkpoints") if Path("/workspace").exists() else (_REPO_ROOT / "checkpoints")

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", str(_DEFAULT_DATA_PATH))
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", str(_DEFAULT_TOKENIZER_PATH))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 0))
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
    eval_timeout_seconds = float(os.environ.get("EVAL_TIMEOUT_SECONDS", 580.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    compile_model = bool(int(os.environ.get("COMPILE_MODEL", "0" if os.name == "nt" else "1")))
    compile_muon = bool(int(os.environ.get("COMPILE_MUON", "0" if os.name == "nt" else "1")))
    adam_fused = bool(int(os.environ.get("ADAM_FUSED", "0" if os.name == "nt" else "1")))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
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
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 256))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
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
    soft_round_qat = bool(int(os.environ.get("SOFT_ROUND_QAT", "1")))
    soft_round_temp_start = float(os.environ.get("SOFT_ROUND_TEMP_START", 1.0))
    soft_round_temp_end = float(os.environ.get("SOFT_ROUND_TEMP_END", 0.05))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    vrl_enabled = bool(int(os.environ.get("VRL_ENABLED", "1")))
    leaky_relu = bool(int(os.environ.get("LEAKY_RELU", "1")))
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 4))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", os.environ.get("TTT_LR", "1e-4")))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 4))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_every_n_chunks = int(os.environ.get("TTT_EVERY_N_CHUNKS", 1))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", os.environ.get("TTT_BATCH_SEQS", "32")))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", "32"))
    ttt_train_batch_seqs = int(os.environ.get("TTT_TRAIN_BATCH_SEQS", "8"))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    ttt_optimizer = os.environ.get("TTT_OPTIMIZER", "adam")
    ttt_temperature = float(os.environ.get("TTT_TEMPERATURE", 0.98))
    byte_weighted_ttt = bool(int(os.environ.get("BYTE_WEIGHTED_TTT", "1")))
    adaptive_lr = bool(int(os.environ.get("ADAPTIVE_LR", "1")))
    adaptive_lr_max = float(os.environ.get("ADAPTIVE_LR_MAX", 3.0))
    eval_only = bool(int(os.environ.get("EVAL_ONLY", "0")))
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "final_model.pt")
    fast_eval_only = bool(int(os.environ.get("FAST_EVAL_ONLY", "1")))
    ngram_backend = os.environ.get("NGRAM_BACKEND", "numpy").strip().lower()
    ckpt_dir = os.environ.get("CKPT_DIR", str(_DEFAULT_CKPT_DIR))
    ckpt_every_secs = float(os.environ.get("CKPT_EVERY_SECS", 60.0))
    resume_ckpt = bool(int(os.environ.get("RESUME_CKPT", "1")))
    copy_artifact_to_ckpt_dir = bool(int(os.environ.get("COPY_ARTIFACT_TO_CKPT_DIR", "1")))
    mlp_quant_bits = int(os.environ.get("MLP_QUANT_BITS", "5"))
    main_quant_bits = int(os.environ.get("MAIN_QUANT_BITS", "6"))
    artifact_codec = os.environ.get("ARTIFACT_CODEC", "lzma").strip().lower()
    ttt_max_chunks = int(os.environ.get("TTT_MAX_CHUNKS", 0))
    skip_sliding_window = bool(int(os.environ.get("SKIP_SLIDING_WINDOW", "0")))
    use_hedge_mixer = bool(int(os.environ.get("USE_HEDGE_MIXER", "1")))
    mixer_eta = float(os.environ.get("MIXER_ETA", 0.1))
    mixer_min_tokens = int(os.environ.get("MIXER_MIN_TOKENS", 10000))


def maybe_compile(obj, enabled: bool):
    if not enabled or not _HAS_TRITON:
        return obj
    return torch.compile(obj, dynamic=False, fullgraph=True)


def eval_timeout_reached(
    timeout_seconds: float,
    start_time: float,
    device: torch.device,
    collective: bool = False,
) -> bool:
    if timeout_seconds <= 0:
        return False
    timed_out = (time.perf_counter() - start_time) >= timeout_seconds
    if collective and dist.is_available() and dist.is_initialized():
        timeout_tensor = torch.tensor(int(timed_out), device=device)
        dist.all_reduce(timeout_tensor, op=dist.ReduceOp.MAX)
        timed_out = bool(timeout_tensor.item())
    return timed_out


def fused_optimizer_kwargs(enabled: bool) -> dict[str, bool]:
    return {"fused": True} if enabled else {}


def broadcast_bool(flag: bool, device: torch.device) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return flag
    value = torch.tensor(int(flag), device=device)
    dist.broadcast(value, src=0)
    return bool(value.item())


def get_spot_ckpt_dir(args: Hyperparameters) -> Path | None:
    raw = str(args.ckpt_dir).strip()
    return Path(raw) if raw else None


def get_spot_ckpt_path(args: Hyperparameters) -> Path | None:
    ckpt_dir = get_spot_ckpt_dir(args)
    if ckpt_dir is None:
        return None
    return ckpt_dir / f"train_ckpt_seed{args.seed}.pt"


def get_spot_artifact_dir(args: Hyperparameters) -> Path | None:
    ckpt_dir = get_spot_ckpt_dir(args)
    if ckpt_dir is None:
        return None
    return ckpt_dir / f"artifact_seed{args.seed}"


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=True)


def save_training_checkpoint(
    ckpt_path: Path,
    base_model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    train_loader: "DistributedTokenLoader | None",
    tracker: "TrainNgramTracker | None",
    ema_state: dict[str, Tensor],
    swa_state: dict[str, Tensor] | None,
    swa_count: int,
    step: int,
    training_time_ms: float,
    qat_start_step: int,
    master_process: bool,
    distributed: bool,
    device: torch.device,
    log0,
) -> None:
    if distributed:
        dist.barrier()
    if master_process:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model_state = {name: tensor.detach().cpu() for name, tensor in base_model.state_dict().items()}
        ckpt = {
            "step": int(step),
            "training_time_ms": float(training_time_ms),
            "model": model_state,
            "ema_state": {name: tensor.detach().cpu() for name, tensor in ema_state.items()},
            "swa_state": None if swa_state is None else {name: tensor.detach().cpu() for name, tensor in swa_state.items()},
            "swa_count": int(swa_count),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "train_loader": None if train_loader is None else train_loader.state_dict(),
            "ngram_tracker": None if tracker is None else tracker.state_dict(),
            "qat_enabled": bool(CastedLinear._qat_enabled),
            "soft_round_temp": float(CastedLinear._soft_round_temp),
            "qat_start_step": int(qat_start_step),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(device=device),
        }
        tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, ckpt_path)
        log0(f"[spot] ckpt:{ckpt_path} s:{step} tt:{training_time_ms:.0f}ms")
    if distributed:
        dist.barrier()


def copy_final_artifacts_to_ckpt_dir(
    args: Hyperparameters,
    master_process: bool,
    log0,
) -> None:
    if not master_process or not args.copy_artifact_to_ckpt_dir:
        return
    artifact_dir = get_spot_artifact_dir(args)
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name in ("final_model.pt", "final_model.int6.ptz"):
        src = Path(name)
        if src.exists():
            shutil.copy2(src, artifact_dir / src.name)
    log0(f"[spot] artifact:{artifact_dir}")
class BackoffNgramMixer:
    PRIMES = [36313, 27191, 51647, 81929, 131071, 174763, 233017]
    def __init__(self, vocab_size: int, device: torch.device, num_buckets: int = 4_000_000,
                 max_order: int = 7, min_count: int = 2, min_tokens: int = 5000,
                 alpha_base: float = 0.05, alpha_range: float = 0.55, alpha_center: float = 4.0):
        self.V = vocab_size
        self.B = num_buckets
        self.MASK = num_buckets - 1 if (num_buckets & (num_buckets - 1)) == 0 else None
        self.max_order = max_order
        self.min_count = min_count
        self.min_tokens = min_tokens
        self.device = device
        self.tokens_seen = 0
        self.alpha_base = alpha_base
        self.alpha_range = alpha_range
        self.alpha_center = alpha_center
        self.uni_counts = torch.zeros(vocab_size, device=device, dtype=torch.float32)
        self.uni_total = 0.0
        self.ctx_counts = []
        self.full_counts = []
        for _ in range(max_order - 1):
            self.ctx_counts.append(torch.zeros(num_buckets, device=device, dtype=torch.float32))
            self.full_counts.append(torch.zeros(num_buckets, device=device, dtype=torch.float32))
    def _bucket(self, h: Tensor) -> Tensor:
        if self.MASK is not None:
            return h & self.MASK
        return h.abs() % self.B
    def update(self, tokens: Tensor):
        t = tokens.to(self.device).long()
        n = t.numel()
        self.tokens_seen += n
        ones = torch.ones(n, device=self.device, dtype=torch.float32)
        self.uni_counts.scatter_add_(0, t, ones)
        self.uni_total += n
        for order in range(2, self.max_order + 1):
            if n < order:
                continue
            oi = order - 2
            nxt = t[order - 1:]
            ctx_h = t[0:n - order + 1] * self.PRIMES[0]
            for k in range(1, order - 1):
                ctx_h = ctx_h ^ (t[k:n - order + 1 + k] * self.PRIMES[k % len(self.PRIMES)])
            ctx_key = self._bucket(ctx_h)
            full_h = ctx_h ^ (nxt * self.PRIMES[(order - 1) % len(self.PRIMES)])
            full_key = self._bucket(full_h)
            self.ctx_counts[oi].scatter_add_(0, ctx_key, ones[:n - order + 1])
            self.full_counts[oi].scatter_add_(0, full_key, ones[:n - order + 1])
    def score(
        self,
        logits: Tensor,
        x_batch: Tensor,
        y_batch: Tensor,
        temperature: float = 1.0,
        score_starts: list[int] | Tensor | None = None,
        score_lens: list[int] | Tensor | None = None,
    ) -> Tensor:
        bsz, slen, V = logits.shape
        if temperature != 1.0:
            logits = logits / temperature
        log_probs_neural = F.log_softmax(logits.float(), dim=-1)
        neural_p = log_probs_neural.gather(-1, y_batch.unsqueeze(-1)).squeeze(-1).exp()
        neural_nll = -neural_p.clamp(min=1e-12).log()
        if score_starts is None:
            active_mask = torch.ones((bsz, slen), dtype=torch.bool, device=self.device)
        else:
            starts_t = torch.as_tensor(score_starts, device=self.device, dtype=torch.int64).view(-1, 1)
            if score_lens is None:
                ends_t = torch.full_like(starts_t, slen)
            else:
                ends_t = torch.as_tensor(score_lens, device=self.device, dtype=torch.int64).view(-1, 1)
            pos = torch.arange(slen, device=self.device, dtype=torch.int64).view(1, -1)
            active_mask = (pos >= starts_t) & (pos < ends_t)
        if self.tokens_seen < self.min_tokens or not bool(active_mask.any()):
            return neural_nll
        active_rows, active_cols = torch.where(active_mask)
        neural_p_active = neural_p[active_rows, active_cols]
        if self.uni_total > 0:
            ngram_p_active = (self.uni_counts[y_batch[active_rows, active_cols]] + 0.5) / (self.uni_total + 0.5 * V)
        else:
            ngram_p_active = torch.full((active_rows.numel(),), 1.0 / V, device=self.device)
        ngram_hit = torch.zeros(active_rows.numel(), device=self.device, dtype=torch.bool)
        for order in range(self.max_order, 1, -1):
            oi = order - 2
            cw = order - 1
            eligible = (active_cols >= (cw - 1)) & (~ngram_hit)
            if not bool(eligible.any()):
                continue
            rows = active_rows[eligible]
            cols = active_cols[eligible]
            ctx_h = x_batch[rows, cols - (cw - 1)] * self.PRIMES[0]
            for k in range(1, cw):
                ctx_h = ctx_h ^ (x_batch[rows, cols - (cw - 1) + k] * self.PRIMES[k % len(self.PRIMES)])
            ctx_key = self._bucket(ctx_h)
            full_h = ctx_h ^ (y_batch[rows, cols] * self.PRIMES[(order - 1) % len(self.PRIMES)])
            full_key = self._bucket(full_h)
            ctx_c = self.ctx_counts[oi][ctx_key]
            full_c = self.full_counts[oi][full_key]
            valid = ctx_c >= self.min_count
            if bool(valid.any()):
                eligible_idx = torch.where(eligible)[0]
                dst = eligible_idx[valid]
                p = (full_c[valid].clamp(max=ctx_c[valid]) / ctx_c[valid].clamp(min=1)).clamp(0, 1)
                ngram_p_active[dst] = p
                ngram_hit[dst] = True
        probs_neural = log_probs_neural.exp()
        entropy_active = -(probs_neural[active_rows, active_cols] * log_probs_neural[active_rows, active_cols]).sum(dim=-1)
        alpha = self.alpha_base + self.alpha_range * torch.sigmoid(
            2.0 * (entropy_active - self.alpha_center))
        mixed_p = (1.0 - alpha) * neural_p_active + alpha * ngram_p_active
        out_nll = neural_nll.clone()
        out_nll[active_rows, active_cols] = -mixed_p.clamp(min=1e-12).log()
        return out_nll
class TrainNgramTracker:
    def __init__(self, vocab_size: int, device: torch.device, complement_alpha: float = 0.5):
        self.V = vocab_size
        self.alpha = complement_alpha
        self.bi_counts = torch.zeros(vocab_size, vocab_size, device=device, dtype=torch.float32)
        self.bi_totals = torch.zeros(vocab_size, device=device, dtype=torch.float32)
    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor):
        xf = x.reshape(-1)
        yf = y.reshape(-1)
        ones = torch.ones(xf.numel(), device=xf.device, dtype=torch.float32)
        self.bi_counts.reshape(-1).scatter_add_(0, xf * self.V + yf, ones)
        self.bi_totals.scatter_add_(0, xf, ones)
    def get_weights(self, x: Tensor, y: Tensor) -> Tensor:
        xf = x.reshape(-1)
        yf = y.reshape(-1)
        total = self.bi_totals[xf]
        count = self.bi_counts.reshape(-1)[xf * self.V + yf]
        ngram_prob = count / (total + 1)
        return (1.0 - self.alpha * ngram_prob).clamp(min=0.1)
    def state_dict(self) -> dict[str, object]:
        return {
            "alpha": float(self.alpha),
            "bi_counts": self.bi_counts.detach().cpu(),
            "bi_totals": self.bi_totals.detach().cpu(),
        }
    def load_state_dict(self, state: dict[str, object]) -> None:
        self.alpha = float(state.get("alpha", self.alpha))
        bi_counts = state.get("bi_counts")
        bi_totals = state.get("bi_totals")
        if isinstance(bi_counts, torch.Tensor) and bi_counts.shape == self.bi_counts.shape:
            self.bi_counts.copy_(bi_counts.to(self.bi_counts.device, dtype=self.bi_counts.dtype))
        if isinstance(bi_totals, torch.Tensor) and bi_totals.shape == self.bi_totals.shape:
            self.bi_totals.copy_(bi_totals.to(self.bi_totals.device, dtype=self.bi_totals.dtype))
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
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
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
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss
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
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"no files:{pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"val too short for {seq_len}")
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
            "VAL_BATCH_SIZE too small; "
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
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,vrl_scales",
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
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"bad header:{file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"size mismatch:{file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"short read:{file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"no files:{pattern}")
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
    def state_dict(self) -> dict[str, int]:
        return {"file_idx": int(self.file_idx), "pos": int(self.pos)}
    def load_state_dict(self, state: dict[str, int]) -> None:
        self.file_idx = int(state.get("file_idx", 0)) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        pos = int(state.get("pos", 0))
        self.pos = min(max(pos, 0), self.tokens.numel())
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
    def state_dict(self) -> dict[str, object]:
        return {
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "stream": self.stream.state_dict(),
        }
    def load_state_dict(self, state: dict[str, object]) -> None:
        stream_state = state.get("stream")
        if isinstance(stream_state, dict):
            self.stream.load_state_dict(stream_state)
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    _soft_round_qat: bool = True
    _soft_round_temp: float = 1.0
    quant_bits: int = 6

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            clip_val = (1 << (self.quant_bits - 1)) - 1
            if CastedLinear._soft_round_qat:
                w32 = self.weight.float()
                row_max = w32.detach().abs().amax(dim=1)
                scale = (row_max / float(clip_val)).clamp_min(1.0 / float(clip_val))
                w_s = w32 / scale[:, None]
                residual = w_s - w_s.detach().round()
                temp = CastedLinear._soft_round_temp
                w_soft = w_s.detach().round() + 0.5 * torch.tanh(residual / temp)
                w = (w_soft.clamp(-clip_val, clip_val) * scale[:, None]).to(x.dtype)
            else:
                with torch.no_grad():
                    w32 = self.weight.float()
                    row_max = w32.abs().amax(dim=1)
                    scale = (row_max / float(clip_val)).clamp_min(1.0 / float(clip_val))
                    w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -clip_val, clip_val) * scale[:, None]).to(x.dtype)
                w = w + (w_q - w).detach()
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
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim%num_heads!=0")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads%num_kv_heads!=0")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("odd head_dim")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False
        self.gated_attention = gated_attention
        if gated_attention:
            self.attn_gate = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, v_embed: Tensor | None = None, lora=None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        if lora is not None:
            q = q + lora.q_delta(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
            k = k + lora.k_delta(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if _FA_VERSION == 3:
            y = _fa_func(q, k, v, causal=True)
        elif _FA_VERSION == 2:
            y = _fa_func(q.bfloat16(), k.bfloat16(), v.bfloat16(), causal=True)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, enable_gqa=True).transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.gated_attention:
            gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
            y = y * gate
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)
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
    def __init__(self, dim: int, mlp_mult: int, leaky: bool = False):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self._neg_slope = 0.5 if leaky else 0.0
    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), self._neg_slope)
        return self.proj(x.square())
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
        **kwargs,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                         gated_attention=kwargs.get("gated_attention", False))
        self.mlp = MLP(dim, mlp_mult, leaky=kwargs.get("leaky", False))
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
    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None, lora=None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed, lora=lora)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out
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
        vrl_enabled: bool = False,
        leaky_relu: bool = False,
        gated_attention: bool = False,
    ):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)
        if logit_softcap <= 0.0:
            raise ValueError(f"softcap<=0:{logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.vrl_enabled = vrl_enabled
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
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
                    leaky=leaky_relu,
                    gated_attention=gated_attention,
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
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()
        if self.vrl_enabled:
            self.vrl_scales = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, dtype=torch.float32)) for _ in range(num_layers - 1)]
            )
        else:
            self.vrl_scales = nn.ParameterList()
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
    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        if self.vrl_enabled:
            mix0 = self.blocks[0].resid_mix.to(dtype=x0.dtype)
            x_in_0 = mix0[0][None, None, :] * x0 + mix0[1][None, None, :] * x0
            n0 = F.rms_norm(x_in_0, (x_in_0.size(-1),)) * self.blocks[0].ln_scale_factor
            v0_raw = self.blocks[0].attn.c_v(n0)
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            if self.vrl_enabled and i > 0:
                vr = v0_raw * self.vrl_scales[i - 1].to(dtype=v0_raw.dtype)
                v_extra = (ve + vr) if ve is not None else vr
            else:
                v_extra = ve
            x = self.blocks[i](x, x0, v_embed=v_extra)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            if self.vrl_enabled:
                vr = v0_raw * self.vrl_scales[bi - 1].to(dtype=v0_raw.dtype)
                v_extra = (ve + vr) if ve is not None else vr
            else:
                v_extra = ve
            x = self.blocks[bi](x, x0, v_embed=v_extra)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("need lm_head")
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
    def forward_logits(self, input_ids: Tensor, lora_layers=None) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        if self.vrl_enabled:
            mix0 = self.blocks[0].resid_mix.to(dtype=x0.dtype)
            x_in_0 = mix0[0][None, None, :] * x0 + mix0[1][None, None, :] * x0
            n0 = F.rms_norm(x_in_0, (x_in_0.size(-1),)) * self.blocks[0].ln_scale_factor
            v0_raw = self.blocks[0].attn.c_v(n0)
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            if self.vrl_enabled and i > 0:
                vr = v0_raw * self.vrl_scales[i - 1].to(dtype=v0_raw.dtype)
                v_extra = (ve + vr) if ve is not None else vr
            else:
                v_extra = ve
            lora = lora_layers[i] if lora_layers is not None else None
            x = self.blocks[i](x, x0, v_embed=v_extra, lora=lora)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            if self.vrl_enabled:
                vr = v0_raw * self.vrl_scales[bi - 1].to(dtype=v0_raw.dtype)
                v_extra = (ve + vr) if ve is not None else vr
            else:
                v_extra = ve
            lora = lora_layers[bi] if lora_layers is not None else None
            x = self.blocks[bi](x, x0, v_embed=v_extra, lora=lora)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


class AttentionLoRA(nn.Module):
    def __init__(self, model_dim: int, kv_dim: int, rank: int):
        super().__init__()
        self.q_A = nn.Parameter(torch.empty(model_dim, rank))
        self.q_B = nn.Parameter(torch.zeros(rank, model_dim))
        self.k_A = nn.Parameter(torch.empty(model_dim, rank))
        self.k_B = nn.Parameter(torch.zeros(rank, kv_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.q_A.size(0))
        with torch.no_grad():
            self.q_A.uniform_(-bound, bound)
            self.k_A.uniform_(-bound, bound)
            self.q_B.zero_()
            self.k_B.zero_()

    def q_delta(self, x: Tensor) -> Tensor:
        return (x @ self.q_A.to(dtype=x.dtype)) @ self.q_B.to(dtype=x.dtype)

    def k_delta(self, x: Tensor) -> Tensor:
        return (x @ self.k_A.to(dtype=x.dtype)) @ self.k_B.to(dtype=x.dtype)


class TTTLoRAAdapter(nn.Module):
    def __init__(self, model: GPT, rank: int):
        super().__init__()
        kv_dim = model.num_kv_heads * (model.tok_emb.embedding_dim // model.blocks[0].attn.num_heads)
        self.layers = nn.ModuleList(
            [AttentionLoRA(model.tok_emb.embedding_dim, kv_dim, rank) for _ in range(len(model.blocks))]
        )

    def clone_state(self) -> dict[str, Tensor]:
        return {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()}

    def load_cloned_state(self, state: dict[str, Tensor]) -> None:
        self.load_state_dict(state, strict=True)


def iter_document_segments(val_tokens: Tensor, bos_token_id: int) -> list[tuple[int, int]]:
    total = int(val_tokens.numel())
    if total <= 1:
        return []
    starts = [0]
    if bos_token_id >= 0:
        bos_positions = (val_tokens == bos_token_id).nonzero(as_tuple=False).flatten().tolist()
        starts = sorted({0, *[int(pos) for pos in bos_positions if 0 <= int(pos) < total - 1]})
    docs: list[tuple[int, int]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else total
        if end - start > 1:
            docs.append((start, end))
    return docs


def iter_eval_segments(val_tokens: Tensor, bos_token_id: int, reset_per_document: bool) -> list[tuple[int, int]]:
    total = int(val_tokens.numel())
    if total <= 1:
        return []
    if not reset_per_document:
        return [(0, total)]
    return iter_document_segments(val_tokens, bos_token_id)


def build_ttt_chunk_windows(total_tokens: int, seq_len: int, stride: int, chunk_tokens: int) -> list[list[int]]:
    if total_tokens <= 0:
        return []
    window_starts = [
        ws for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0
    ]
    num_chunks = (total_tokens + chunk_tokens - 1) // max(chunk_tokens, 1)
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        scored_start = ws + (0 if ws == 0 else max(wlen - stride, 0))
        ci = min(scored_start // max(chunk_tokens, 1), num_chunks - 1)
        chunk_windows[ci].append(ws)
    return chunk_windows


def build_ttt_optimizer(args: Hyperparameters, params) -> torch.optim.Optimizer:
    if args.ttt_optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.ttt_lora_lr, weight_decay=0.0, betas=(0.9, 0.999))
    if args.ttt_optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.ttt_lora_lr, momentum=args.ttt_momentum)
    return torch.optim.Adam(params, lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)


def get_even_ttt_seq_span(total_seqs: int, rank: int, world_size: int) -> tuple[int, int, int]:
    if world_size <= 1:
        return 0, total_seqs, 0
    # Keep exactly the same number of TTT optimizer steps on every rank.
    usable_total = total_seqs - (total_seqs % world_size)
    per_rank = usable_total // world_size
    start = rank * per_rank
    end = start + per_rank
    dropped = total_seqs - usable_total
    return start, end, dropped


def train_lora_on_chunk(
    args: Hyperparameters,
    base_model: GPT,
    lora: TTTLoRAAdapter,
    chunk_tokens: Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    base_bytes_lut: Tensor,
) -> float:
    if chunk_tokens.numel() <= 1:
        return 0.0
    chunk_tokens = chunk_tokens.to(device=device, dtype=torch.int64)
    seq_len = args.train_seq_len
    num_pred_tokens = int(chunk_tokens.numel()) - 1
    chunk_seqs = num_pred_tokens // seq_len
    if chunk_seqs <= 0:
        return 0.0
    distributed = dist.is_available() and dist.is_initialized()
    if distributed and world_size > 1:
        my_seq_start, my_seq_end, _ = get_even_ttt_seq_span(chunk_seqs, rank, world_size)
    else:
        my_seq_start, my_seq_end = 0, chunk_seqs
    my_chunk_seqs = my_seq_end - my_seq_start
    if my_chunk_seqs <= 0:
        return 0.0
    optimizer = build_ttt_optimizer(args, lora.parameters())
    avg_loss = 0.0
    steps = 0
    base_model.eval()
    for _ in range(args.ttt_epochs):
        for bs in range(0, my_chunk_seqs, args.ttt_train_batch_seqs):
            be = min(bs + args.ttt_train_batch_seqs, my_chunk_seqs)
            seq_start = my_seq_start + bs
            start_tok = seq_start * seq_len
            end_tok = (my_seq_start + be) * seq_len + 1
            local = chunk_tokens[start_tok:end_tok]
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_t = base_model.forward_logits(x, lora_layers=lora.layers)
            if args.byte_weighted_ttt:
                per_tok_nll = F.cross_entropy(
                    logits_t.reshape(-1, logits_t.size(-1)).float(),
                    y.reshape(-1),
                    reduction="none",
                )
                byte_weights = base_bytes_lut[y.reshape(-1)].float()
                byte_weights = byte_weights / byte_weights.mean().clamp(min=1e-6)
                loss = (per_tok_nll * byte_weights).mean()
            else:
                loss = F.cross_entropy(logits_t.reshape(-1, logits_t.size(-1)).float(), y.reshape(-1))
            loss.backward()
            if distributed and world_size > 1:
                for p in lora.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            if args.ttt_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(lora.parameters(), args.ttt_grad_clip)
            optimizer.step()
            avg_loss += float(loss.item())
            steps += 1
    return avg_loss / max(steps, 1)
def eval_val_sliding_ttt(
    args, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, log0=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    reset_per_document = bool(args.ttt_enabled)
    docs = iter_eval_segments(val_tokens, getattr(args, "bos_token_id", -1), reset_per_document)
    if args.ttt_max_chunks > 0:
        capped_docs: list[tuple[int, int]] = []
        remaining = args.ttt_max_chunks
        for doc_start, doc_end in docs:
            doc_chunks = (max(doc_end - doc_start - 1, 0) + args.ttt_chunk_tokens - 1) // max(args.ttt_chunk_tokens, 1)
            if doc_chunks <= 0:
                continue
            capped_docs.append((doc_start, doc_end))
            remaining -= doc_chunks
            if remaining <= 0:
                break
        docs = capped_docs
    total_doc_chunks = sum(
        (max(doc_end - doc_start - 1, 0) + args.ttt_chunk_tokens - 1) // max(args.ttt_chunk_tokens, 1)
        for doc_start, doc_end in docs
    )
    log0(
        f"ttt:lora docs={len(docs)} chunks={total_doc_chunks} ct={args.ttt_chunk_tokens} "
        f"s={stride} lr={args.ttt_lora_lr} ep={args.ttt_epochs} r={args.ttt_lora_rank} "
        f"opt={args.ttt_optimizer} bw={args.byte_weighted_ttt} alr={args.adaptive_lr}({args.adaptive_lr_max}) "
        f"t={args.ttt_temperature} bs={batch_seqs}/{args.ttt_train_batch_seqs}"
    )
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    distributed = dist.is_available() and dist.is_initialized()
    for p in base_model.parameters():
        p.requires_grad_(False)
    lora: TTTLoRAAdapter | None = None
    initial_lora_state: dict[str, Tensor] | None = None
    if args.ttt_enabled and args.ttt_lora_rank > 0 and args.ttt_epochs > 0:
        lora = TTTLoRAAdapter(base_model, args.ttt_lora_rank).to(device)
        initial_lora_state = lora.clone_state()
    mixer: BackoffNgramMixer | None = None
    if args.use_hedge_mixer:
        ngram_order = int(os.environ.get("NGRAM_ORDER", "10"))
        ngram_buckets = int(os.environ.get("NGRAM_BUCKETS", "4194304"))
        alpha_base = float(os.environ.get("ALPHA_BASE", "0.20"))
        alpha_range = float(os.environ.get("ALPHA_RANGE", "0.55"))
        alpha_center = float(os.environ.get("ALPHA_CENTER", "3.0"))
        min_count = int(os.environ.get("MIN_COUNT", "2"))
        mixer = BackoffNgramMixer(args.vocab_size, device, num_buckets=ngram_buckets,
                                   max_order=ngram_order, min_count=min_count,
                                   min_tokens=args.mixer_min_tokens,
                                   alpha_base=alpha_base, alpha_range=alpha_range,
                                   alpha_center=alpha_center)
        mem_mb = ngram_buckets * 4 * 2 * (ngram_order - 1) / 1e6
        log0(f"bo:o={ngram_order} b={ngram_buckets} m={mem_mb:.0f}M a={alpha_base}+{alpha_range}*s(H-{alpha_center}) mc={min_count}")
    if lora is not None:
        def score_forward(input_ids: Tensor) -> Tensor:
            return base_model.forward_logits(input_ids, lora_layers=lora.layers)
        compiled_logits = maybe_compile(score_forward, args.compile_model)
    else:
        compiled_logits = maybe_compile(base_model.forward_logits, args.compile_model)
    t0 = time.perf_counter()
    global_chunk_idx = 0
    timed_out = False
    for doc_idx, (doc_start, doc_end) in enumerate(docs):
        if eval_timeout_reached(args.eval_timeout_seconds, t0, device, collective=distributed):
            timed_out = True
            break
        doc_tokens = val_tokens[doc_start:doc_end]
        if doc_tokens.numel() <= 1:
            continue
        if reset_per_document and lora is not None and initial_lora_state is not None:
            lora.load_cloned_state(initial_lora_state)
        total_doc_tokens = int(doc_tokens.numel()) - 1
        chunk_windows = build_ttt_chunk_windows(total_doc_tokens, seq_len, stride, args.ttt_chunk_tokens)
        for ci, windows in enumerate(chunk_windows):
            if eval_timeout_reached(args.eval_timeout_seconds, t0, device, collective=distributed):
                timed_out = True
                break
            global_chunk_idx += 1
            if not windows:
                continue
            chunk_start = ci * args.ttt_chunk_tokens
            chunk_end = min((ci + 1) * args.ttt_chunk_tokens, total_doc_tokens)
            my_s = (len(windows) * rank) // world_size
            my_e = (len(windows) * (rank + 1)) // world_size
            my_windows = windows[my_s:my_e]
            base_model.eval()
            with torch.inference_mode():
                for bi in range(0, len(my_windows), batch_seqs):
                    batch_ws = my_windows[bi:bi + batch_seqs]
                    bsz = len(batch_ws)
                    if bsz == 0:
                        continue
                    padded_bsz = max(batch_seqs, bsz)
                    batch_start = min(batch_ws)
                    batch_end = max(min(ws + seq_len, total_doc_tokens) for ws in batch_ws)
                    batch_doc = doc_tokens[batch_start:batch_end + 1].to(dtype=torch.int64, device=device)
                    x_batch = torch.zeros(padded_bsz, seq_len, dtype=torch.int64, device=device)
                    y_batch = torch.zeros(padded_bsz, seq_len, dtype=torch.int64, device=device)
                    wlens: list[int] = []
                    for i, ws in enumerate(batch_ws):
                        end = min(ws + seq_len, total_doc_tokens)
                        wlen = end - ws
                        wlens.append(wlen)
                        offset = ws - batch_start
                        local_doc = batch_doc[offset:offset + wlen + 1]
                        x_batch[i, :wlen] = local_doc[:-1]
                        y_batch[i, :wlen] = local_doc[1:]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = compiled_logits(x_batch)[:bsz]
                    x_eval = x_batch[:bsz]
                    y_eval = y_batch[:bsz]
                    score_starts = [0 if ws == 0 else max(wlen - stride, 0) for ws, wlen in zip(batch_ws, wlens)]
                    if mixer is not None and mixer.tokens_seen >= mixer.min_tokens:
                        nll = mixer.score(
                            logits,
                            x_eval,
                            y_eval,
                            args.ttt_temperature,
                            score_starts=score_starts,
                            score_lens=wlens,
                        )
                    else:
                        if args.ttt_temperature != 1.0:
                            logits = logits / args.ttt_temperature
                        nll = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)).float(),
                            y_eval.reshape(-1),
                            reduction="none",
                        ).reshape(bsz, seq_len)
                    for i, ws in enumerate(batch_ws):
                        wlen = wlens[i]
                        s = score_starts[i]
                        scored_nll = nll[i, s:wlen].to(torch.float64)
                        loss_sum += scored_nll.sum()
                        token_count += float(wlen - s)
                        tgt, prev = y_eval[i, s:wlen], x_eval[i, s:wlen]
                        tb = base_bytes_lut[tgt].to(torch.float64)
                        tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                        byte_count += tb.sum()
            if timed_out:
                break
            if mixer is not None:
                if distributed:
                    dist.barrier()
                mixer.update(doc_tokens[chunk_start:chunk_end].to(device))
            is_last_chunk = ci == len(chunk_windows) - 1
            should_ttt = (
                args.ttt_enabled
                and lora is not None
                and not is_last_chunk
                and args.ttt_epochs > 0
                and (ci + 1) % max(args.ttt_every_n_chunks, 1) == 0
            )
            if should_ttt:
                if eval_timeout_reached(args.eval_timeout_seconds, t0, device, collective=distributed):
                    timed_out = True
                    break
                if args.adaptive_lr and len(chunk_windows) > 0:
                    progress = min(ci / max(len(chunk_windows) * 0.3, 1.0), 1.0)
                    lr_mult = 1.0 + (args.adaptive_lr_max - 1.0) * progress
                    effective_lr = args.ttt_lora_lr * lr_mult
                else:
                    effective_lr = args.ttt_lora_lr
                prev_lr = args.ttt_lora_lr
                args.ttt_lora_lr = effective_lr
                train_loss = train_lora_on_chunk(
                    args,
                    base_model,
                    lora,
                    doc_tokens[chunk_start:chunk_end + 1],
                    device,
                    rank,
                    world_size,
                    base_bytes_lut,
                )
                args.ttt_lora_lr = prev_lr
                if rank == 0:
                    log0(
                        f"ttt:doc={doc_idx + 1}/{len(docs)} chunk={ci + 1}/{len(chunk_windows)} "
                        f"nll={train_loss:.4f} lr={effective_lr:.6g}"
                    )
            if rank == 0 and (global_chunk_idx % 10 == 0 or global_chunk_idx == total_doc_chunks):
                elapsed = time.perf_counter() - t0
                rl = loss_sum.item() / max(token_count.item(), 1)
                rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
                log0(f"  tc[{global_chunk_idx}/{total_doc_chunks}]bpb={rbpb:.6f} t={elapsed:.1f}s")
        if timed_out:
            break
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    if timed_out and rank == 0:
        log0(f"eval:timeout hit at {time.perf_counter()-t0:.1f}s during ttt/ngram pass")
    if token_count.item() <= 0 or byte_count.item() <= 0:
        for p in base_model.parameters():
            p.requires_grad_(True)
        base_model.eval()
        return float("inf"), float("inf")
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    log0(f"ttt:vl={val_loss:.6f} bpb={val_bpb:.6f} t={time.perf_counter()-t0:.1f}s")
    return val_loss, val_bpb
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
    log0=print,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    distributed = dist.is_available() and dist.is_initialized()
    local_iters = (len(my_windows) + batch_seqs - 1) // batch_seqs
    max_iters_tensor = torch.tensor(local_iters, device=device, dtype=torch.int64)
    if distributed:
        dist.all_reduce(max_iters_tensor, op=dist.ReduceOp.MAX)
    max_iters = int(max_iters_tensor.item())
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = maybe_compile(base_model.forward_logits, args.compile_model)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for iter_idx in range(max_iters):
            if eval_timeout_reached(args.eval_timeout_seconds, t0, device, collective=distributed):
                if rank == 0:
                    log0(f"eval:timeout hit at {time.perf_counter()-t0:.1f}s during sliding pass")
                break
            bi = iter_idx * batch_seqs
            if bi >= len(my_windows):
                continue
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            if bsz == 0:
                continue
            padded_bsz = max(batch_seqs, bsz)
            batch_start = min(batch_ws)
            batch_end = max(min(ws + seq_len, total_tokens) for ws in batch_ws)
            batch_tokens = val_tokens[batch_start:batch_end + 1].to(dtype=torch.int64, device=device)
            x_batch = torch.zeros(padded_bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(padded_bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                offset = ws - batch_start
                chunk = batch_tokens[offset:offset + wlen + 1]
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)[:bsz]
            y_eval = y_batch[:bsz]
            x_eval = x_batch[:bsz]
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_eval.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_eval[i, s:wlen]
                prev = x_eval[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    if token_count.item() <= 0 or byte_count.item() <= 0:
        base_model.train()
        return float("inf"), float("inf")
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_signed_per_row(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    clip_range = (1 << (bits - 1)) - 1
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


def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    del clip_range
    return quantize_signed_per_row(t, 6)


def quantize_int5_per_row(t: Tensor, clip_range: int = 15) -> tuple[Tensor, Tensor]:
    del clip_range
    return quantize_signed_per_row(t, 5)


def pack_int5_tensor(q: Tensor) -> tuple[Tensor, int]:
    flat = q.detach().to(torch.int16).reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return torch.empty((0,), dtype=torch.uint8), 0
    vals = flat.to(torch.int32) + 16
    if ((vals < 0) | (vals > 31)).any():
        raise ValueError("int5 pack out of range")
    pad = (-n) % 8
    if pad:
        vals = torch.cat([vals, torch.zeros(pad, dtype=torch.int32)], dim=0)
    groups = vals.view(-1, 8)
    b0 = (groups[:, 0] | ((groups[:, 1] & 0x07) << 5)).to(torch.uint8)
    b1 = (((groups[:, 1] >> 3) & 0x03) | (groups[:, 2] << 2) | ((groups[:, 3] & 0x01) << 7)).to(torch.uint8)
    b2 = (((groups[:, 3] >> 1) & 0x0F) | ((groups[:, 4] & 0x0F) << 4)).to(torch.uint8)
    b3 = (((groups[:, 4] >> 4) & 0x01) | (groups[:, 5] << 1) | ((groups[:, 6] & 0x03) << 6)).to(torch.uint8)
    b4 = (((groups[:, 6] >> 2) & 0x07) | (groups[:, 7] << 3)).to(torch.uint8)
    packed = torch.stack((b0, b1, b2, b3, b4), dim=1).reshape(-1).contiguous()
    return packed, n


def pack_int6_tensor(q: Tensor) -> tuple[Tensor, int]:
    flat = q.detach().to(torch.int16).reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return torch.empty((0,), dtype=torch.uint8), 0
    vals = flat.to(torch.int32) + 32
    if ((vals < 0) | (vals > 63)).any():
        raise ValueError("int6 pack out of range")
    pad = (-n) % 4
    if pad:
        vals = torch.cat([vals, torch.zeros(pad, dtype=torch.int32)], dim=0)
    groups = vals.view(-1, 4)
    b0 = (groups[:, 0] | ((groups[:, 1] & 0x03) << 6)).to(torch.uint8)
    b1 = (((groups[:, 1] >> 2) & 0x0F) | ((groups[:, 2] & 0x0F) << 4)).to(torch.uint8)
    b2 = (((groups[:, 2] >> 4) & 0x03) | (groups[:, 3] << 2)).to(torch.uint8)
    packed = torch.stack((b0, b1, b2), dim=1).reshape(-1).contiguous()
    return packed, n


def unpack_int5_tensor(packed: Tensor, numel: int, shape: list[int] | tuple[int, ...]) -> Tensor:
    if numel == 0:
        return torch.empty(shape, dtype=torch.int8)
    raw = packed.detach().to(torch.uint8).reshape(-1)
    if raw.numel() % 5 != 0:
        raise ValueError("bad packed int5 length")
    groups = raw.view(-1, 5).to(torch.int32)
    v0 = groups[:, 0] & 0x1F
    v1 = ((groups[:, 0] >> 5) & 0x07) | ((groups[:, 1] & 0x03) << 3)
    v2 = (groups[:, 1] >> 2) & 0x1F
    v3 = ((groups[:, 1] >> 7) & 0x01) | ((groups[:, 2] & 0x0F) << 1)
    v4 = ((groups[:, 2] >> 4) & 0x0F) | ((groups[:, 3] & 0x01) << 4)
    v5 = (groups[:, 3] >> 1) & 0x1F
    v6 = ((groups[:, 3] >> 6) & 0x03) | ((groups[:, 4] & 0x07) << 2)
    v7 = (groups[:, 4] >> 3) & 0x1F
    vals = torch.stack((v0, v1, v2, v3, v4, v5, v6, v7), dim=1).reshape(-1)[:numel]
    q = (vals - 16).to(torch.int8)
    return q.view(*shape).contiguous()


def unpack_int6_tensor(packed: Tensor, numel: int, shape: list[int] | tuple[int, ...]) -> Tensor:
    if numel == 0:
        return torch.empty(shape, dtype=torch.int8)
    raw = packed.detach().to(torch.uint8).reshape(-1)
    if raw.numel() % 3 != 0:
        raise ValueError("bad packed int6 length")
    groups = raw.view(-1, 3).to(torch.int32)
    v0 = groups[:, 0] & 0x3F
    v1 = ((groups[:, 0] >> 6) & 0x03) | ((groups[:, 1] & 0x0F) << 2)
    v2 = ((groups[:, 1] >> 4) & 0x0F) | ((groups[:, 2] & 0x03) << 4)
    v3 = (groups[:, 2] >> 2) & 0x3F
    vals = torch.stack((v0, v1, v2, v3), dim=1).reshape(-1)[:numel]
    q = (vals - 32).to(torch.int8)
    return q.view(*shape).contiguous()


def pack_signed_tensor(q: Tensor, bits: int) -> tuple[Tensor, int]:
    if bits == 5:
        return pack_int5_tensor(q)
    if bits == 6:
        return pack_int6_tensor(q)
    raise ValueError(f"unsupported pack bits:{bits}")


def unpack_signed_tensor(packed: Tensor, numel: int, shape: list[int] | tuple[int, ...], bits: int) -> Tensor:
    if bits == 5:
        return unpack_int5_tensor(packed, numel, shape)
    if bits == 6:
        return unpack_int6_tensor(packed, numel, shape)
    raise ValueError(f"unsupported unpack bits:{bits}")


def summarize_mixed_quantized_artifact(result: dict[str, Tensor], meta: dict[str, object]) -> dict[str, int]:
    stats = {
        "int5_packed_bytes": 0,
        "int5_scale_bytes": 0,
        "int6_packed_bytes": 0,
        "int6_scale_bytes": 0,
        "int8_q_bytes": 0,
        "int8_scale_bytes": 0,
        "passthrough_bytes": 0,
        "num_int5_tensors": 0,
        "num_int6_tensors": 0,
        "num_int8_tensors": 0,
        "num_passthrough_tensors": 0,
    }
    for name, info in meta.items():
        if info == "passthrough" or info == "passthrough_ctrl" or info == "passthrough_fp16":
            t = result[name]
            stats["passthrough_bytes"] += tensor_nbytes(t)
            stats["num_passthrough_tensors"] += 1
            continue
        if not isinstance(info, dict):
            continue
        q_key = name + ".q"
        s_key = name + ".scale"
        q = result[q_key]
        s = result[s_key]
        if info.get("type") == "int5_packed":
            stats["int5_packed_bytes"] += tensor_nbytes(q)
            stats["int5_scale_bytes"] += tensor_nbytes(s)
            stats["num_int5_tensors"] += 1
        elif info.get("type") == "int6_packed":
            stats["int6_packed_bytes"] += tensor_nbytes(q)
            stats["int6_scale_bytes"] += tensor_nbytes(s)
            stats["num_int6_tensors"] += 1
        elif info.get("type") == "int8":
            stats["int8_q_bytes"] += tensor_nbytes(q)
            stats["int8_scale_bytes"] += tensor_nbytes(s)
            stats["num_int8_tensors"] += 1
    stats["total_payload_bytes"] = (
        stats["int5_packed_bytes"]
        + stats["int5_scale_bytes"]
        + stats["int6_packed_bytes"]
        + stats["int6_scale_bytes"]
        + stats["int8_q_bytes"]
        + stats["int8_scale_bytes"]
        + stats["passthrough_bytes"]
    )
    return stats


_ARTIFACT_MAGIC = b"PGQ1"
_ARTIFACT_CODEC_IDS = {"lzma": 1, "zstd": 2}
_ARTIFACT_CODEC_NAMES = {v: k for k, v in _ARTIFACT_CODEC_IDS.items()}


def compress_artifact_blob(raw: bytes, codec_pref: str = "lzma") -> tuple[bytes, str]:
    codec_pref = (codec_pref or "lzma").strip().lower()
    candidates: list[tuple[str, bytes]] = [
        ("lzma", lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME))
    ]
    if _HAS_ZSTD and codec_pref in ("auto", "zstd"):
        compressor = zstd.ZstdCompressor(level=22)
        candidates.append(("zstd", compressor.compress(raw)))
    if codec_pref == "zstd" and not _HAS_ZSTD:
        raise RuntimeError("ARTIFACT_CODEC=zstd but zstandard is unavailable")
    if codec_pref == "lzma":
        codec_name, payload = candidates[0]
    else:
        codec_name, payload = min(candidates, key=lambda item: len(item[1]))
    header = _ARTIFACT_MAGIC + bytes([_ARTIFACT_CODEC_IDS[codec_name]])
    return header + payload, codec_name


def decompress_artifact_blob(blob: bytes) -> tuple[bytes, str]:
    if blob.startswith(_ARTIFACT_MAGIC) and len(blob) > len(_ARTIFACT_MAGIC):
        codec_id = blob[len(_ARTIFACT_MAGIC)]
        payload = blob[len(_ARTIFACT_MAGIC) + 1 :]
        codec_name = _ARTIFACT_CODEC_NAMES.get(codec_id)
        if codec_name == "lzma":
            return lzma.decompress(payload), codec_name
        if codec_name == "zstd":
            if not _HAS_ZSTD:
                raise RuntimeError("artifact uses zstd but zstandard is unavailable")
            return zstd.ZstdDecompressor().decompress(payload), codec_name
        raise ValueError(f"unknown artifact codec id:{codec_id}")
    return lzma.decompress(blob), "lzma-legacy"


def get_quant_bits_by_cat(args: Hyperparameters) -> dict[str, int]:
    return {
        "mlp": int(args.mlp_quant_bits),
        "attn": int(args.main_quant_bits),
        "embed": int(args.main_quant_bits),
        "other": int(args.main_quant_bits),
    }


def mixed_quantize_int6(state_dict: dict[str, Tensor], quant_bits_by_cat: dict[str, int]):
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
        bits = int(quant_bits_by_cat.get(cat, 0))
        if bits in (5, 6) and t.ndim >= 1:
            q, s = quantize_signed_per_row(t, bits)
            q_packed, q_numel = pack_signed_tensor(q, bits)
            result[name + ".q"] = q_packed
            result[name + ".scale"] = s
            meta[name] = {"type": f"int{bits}_packed", "bits": bits, "shape": list(t.shape), "numel": q_numel}
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
        q_deq = q
        if isinstance(info, dict) and str(info.get("type", "")).endswith("_packed"):
            bits = int(info.get("bits", 6))
            q_deq = unpack_signed_tensor(
                q,
                int(info["numel"]),
                tuple(int(x) for x in info["shape"]),
                bits,
            )
        if s.ndim > 0:
            out[name] = (q_deq.float() * s.float().view(q_deq.shape[0], *([1] * (q_deq.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q_deq.float() * float(s.item())).to(orig_dtype)
    return out
def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if args.compile_muon:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"bad WORLD_SIZE:{world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"8%WORLD_SIZE={world_size}!=0")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist_backend = "nccl" if os.name != "nt" else "gloo"
        if dist_backend == "nccl":
            dist.init_process_group(backend=dist_backend, device_id=device)
        else:
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", "29500")
            dist.init_process_group(
                backend=dist_backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=rank,
                world_size=world_size,
            )
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _gpu_name = torch.cuda.get_device_name(0)
    _is_high_end = "H100" in _gpu_name or "A100" in _gpu_name
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    if _is_high_end:
        enable_cudnn_sdp(True)
        enable_flash_sdp(False)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)
    else:
        enable_cudnn_sdp(True)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(True)
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
    log0("="*60,console=False)
    log0(f"py:{sys.version}",console=False)
    log0(f"pt:{torch.__version__}",console=False)
    log0(subprocess.run(["nvidia-smi"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False).stdout,console=False)
    log0("="*60,console=False)
    log0(f"fa:{_FA_VERSION} gpu:{_gpu_name} he:{_is_high_end}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"need .model:{args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"vocab mismatch:{args.vocab_size}!={int(sp.vocab_size())}"
        )
    args.bos_token_id = int(sp.bos_id())
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = 0 if args.eval_only else len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    if args.val_max_tokens > 0:
        val_tokens = val_tokens[: min(args.val_max_tokens, val_tokens.numel() - 1) + 1].contiguous()
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"bpb:sp={args.tokenizer_path}")
    log0(f"train:{dataset_dir.name} shards:{actual_train_files}")
    log0(f"val:{args.val_files} n:{val_tokens.numel()-1}")
    CastedLinear._qat_enabled = args.qat_enabled
    CastedLinear._soft_round_qat = args.soft_round_qat
    CastedLinear._soft_round_temp = args.soft_round_temp_start
    qat_start_step = 0 if args.qat_enabled else -1
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
        vrl_enabled=args.vrl_enabled,
        leaky_relu=args.leaky_relu,
        gated_attention=args.gated_attention,
    ).to(device).bfloat16()
    for name, module in base_model.named_modules():
        if isinstance(module, CastedLinear):
            if ".mlp." in name:
                module.quant_bits = args.mlp_quant_bits
            else:
                module.quant_bits = args.main_quant_bits
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    complement_alpha = float(os.environ.get("COMPLEMENT_ALPHA", "0.5"))
    if complement_alpha > 0:
        tracker = TrainNgramTracker(args.vocab_size, device, complement_alpha=complement_alpha)
        base_model._ngram_tracker = tracker
        log0(f"compl:{complement_alpha}")
    else:
        base_model._ngram_tracker = None
    if distributed:
        torch._dynamo.config.optimize_ddp = False
    compiled_model = maybe_compile(base_model, args.compile_model)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.mtp_num_heads > 0:
        matrix_params.extend([p for p in base_model.mtp_heads.parameters() if p.ndim == 2])
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
    if base_model.vrl_enabled:
        for s in base_model.vrl_scales:
            scalar_params.append(s)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            matrix_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        **fused_optimizer_kwargs(args.adam_fused),
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
        **fused_optimizer_kwargs(args.adam_fused),
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            **fused_optimizer_kwargs(args.adam_fused),
        )
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params = sum(p.numel() for p in base_model.mtp_heads.parameters())
    log0(f"p:{n_params}")
    log0(
        f"model:{args.num_layers}L d={args.model_dim} mlp={args.mlp_mult} "
        f"h={args.num_heads} kv={args.num_kv_heads} fp16={n_params * 2 / 1e6:.2f}MB "
        f"qmain={args.main_quant_bits} qmlp={args.mlp_quant_bits}"
    )
    log0(f"mtp:{args.mtp_num_heads} w:{args.mtp_loss_weight} p:{mtp_params}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"xsa:{args.xsa_last_n} l:{xsa_layers}")
    log0(f"ws:{world_size} ga:{grad_accum_steps}")
    log0(f"sdp:{_is_high_end}")
    log0(f"attn:h={args.num_heads} kv={args.num_kv_heads}")
    log0(f"vrl:{args.vrl_enabled} lrelu:{args.leaky_relu} ttt:{args.ttt_enabled}")
    log0(f"compile:model={args.compile_model} muon={args.compile_muon} fused={args.adam_fused} triton={_HAS_TRITON}")
    log0(f"eval:stride={args.eval_stride} bs={args.eval_batch_seqs} ttt_bs={args.ttt_batch_seqs}/{args.ttt_train_batch_seqs}")
    log0(f"eval:timeout={args.eval_timeout_seconds:.3f}s")
    log0(f"spot:dir={args.ckpt_dir} every={args.ckpt_every_secs:.1f}s resume={args.resume_ckpt}")
    log0(f"tie:{args.tie_embeddings} elr:{token_lr} hlr:{args.head_lr if base_model.lm_head is not None else 0.0} mlr:{args.matrix_lr} slr:{args.scalar_lr}")
    log0(f"tbt:{args.train_batch_tokens} tsl:{args.train_seq_len} it:{args.iterations} wu:{args.warmup_steps} mws:{args.max_wallclock_seconds:.3f}")
    log0(f"s:{args.seed}")
    train_loader: DistributedTokenLoader | None = None
    if not args.eval_only:
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    spot_ckpt_path = get_spot_ckpt_path(args) if not args.eval_only else None
    if spot_ckpt_path is not None and master_process:
        spot_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if distributed and spot_ckpt_path is not None:
        dist.barrier()
    resume_pending = False
    if spot_ckpt_path is not None and args.resume_ckpt:
        resume_pending = master_process and spot_ckpt_path.exists()
        resume_pending = broadcast_bool(resume_pending, device)
        if resume_pending:
            log0(f"[spot] resume:{spot_ckpt_path}")
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
    if args.warmup_steps > 0 and not args.eval_only and not resume_pending:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                assert train_loader is not None
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"wu:{warmup_step+1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    if args.eval_only:
        log0(f"eval:load {args.checkpoint_path}")
        ckpt_state = torch.load(args.checkpoint_path, map_location="cpu")
        base_model.load_state_dict(ckpt_state, strict=True)
        log0(f"eval:loaded {sum(p.numel() for p in base_model.parameters())}p")
        full_state_dict = base_model.state_dict()
        export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
        sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
        quant_result, quant_meta = mixed_quantize_int6(sd_cpu, get_quant_bits_by_cat(args))
        quant_stats = summarize_mixed_quantized_artifact(quant_result, quant_meta)
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob, quant_codec = compress_artifact_blob(quant_raw, args.artifact_codec)
        if master_process:
            with open("final_model.int6.ptz", "wb") as f:
                f.write(quant_blob)
            log0(f"eval:qsize:{len(quant_blob)}B")
            log0(
                "eval:qdiag "
                f"raw={len(quant_raw)}B codec={quant_codec} blob={len(quant_blob)}B "
                f"ratio={len(quant_blob)/max(len(quant_raw),1):.4f} "
                f"int5q={quant_stats['int5_packed_bytes']}B int5s={quant_stats['int5_scale_bytes']}B "
                f"int6q={quant_stats['int6_packed_bytes']}B int6s={quant_stats['int6_scale_bytes']}B "
                f"int8q={quant_stats['int8_q_bytes']}B int8s={quant_stats['int8_scale_bytes']}B "
                f"pass={quant_stats['passthrough_bytes']}B"
            )
        if distributed:
            dist.barrier()
        with open("final_model.int6.ptz", "rb") as f:
            quant_blob_disk = f.read()
        quant_raw_disk, _ = decompress_artifact_blob(quant_blob_disk)
        quant_state = torch.load(io.BytesIO(quant_raw_disk), map_location="cpu")
        deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
        eval_model = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            mtp_num_heads=0, mtp_loss_weight=0.0,
            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
            xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
            ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
            vrl_enabled=args.vrl_enabled, leaky_relu=args.leaky_relu,
            gated_attention=args.gated_attention,
        ).to(device).bfloat16()
        for name, m in eval_model.named_modules():
            if isinstance(m, CastedLinear):
                if ".mlp." in name:
                    m.quant_bits = args.mlp_quant_bits
                else:
                    m.quant_bits = args.main_quant_bits
                m.float()
        restore_low_dim_params_to_fp32(eval_model)
        eval_model.load_state_dict(deq_state, strict=True)
        sw_seq_len = effective_eval_seq_len
        if not args.skip_sliding_window and args.eval_stride > 0 and args.eval_stride < sw_seq_len:
            torch.cuda.synchronize()
            t_slide = time.perf_counter()
            sw_val_loss, sw_val_bpb = eval_val_sliding(
                args, eval_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=args.eval_stride, batch_seqs=args.eval_batch_seqs, eval_seq_len=sw_seq_len, log0=log0,
            )
            torch.cuda.synchronize()
            log0(f"eval:sw bpb:{sw_val_bpb:.4f} s:{args.eval_stride} t:{1000.0*(time.perf_counter()-t_slide):.0f}ms")
        elif args.skip_sliding_window:
            log0("eval:skip_sw")
        if args.ttt_enabled or args.use_hedge_mixer:
            mode = "ttt_lora" if args.ttt_enabled else "ngram"
            log0(
                f"eval:{mode} lr={args.ttt_lora_lr} ep={args.ttt_epochs} "
                f"c={args.ttt_chunk_tokens} r={args.ttt_lora_rank}"
            )
            torch.cuda.synchronize()
            t_ttt = time.perf_counter()
            ttt_val_loss, ttt_val_bpb = eval_val_sliding_ttt(
                args, eval_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=args.eval_stride, batch_seqs=args.ttt_batch_seqs, log0=log0,
            )
            torch.cuda.synchronize()
            log0(f"eval:{mode} bpb:{ttt_val_bpb:.4f} t:{1000.0*(time.perf_counter()-t_ttt):.0f}ms")
        if distributed:
            dist.destroy_process_group()
        return
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = 0.997
    training_time_ms = 0.0
    stop_after_step: int | None = None
    if resume_pending:
        assert spot_ckpt_path is not None
        ckpt = torch.load(spot_ckpt_path, map_location="cpu", weights_only=False)
        base_model.load_state_dict(ckpt["model"], strict=True)
        optimizer_states = ckpt.get("optimizers", [])
        if len(optimizer_states) != len(optimizers):
            raise ValueError(f"optimizer mismatch:{len(optimizer_states)}!={len(optimizers)}")
        for opt, state in zip(optimizers, optimizer_states, strict=True):
            opt.load_state_dict(state)
            optimizer_to_device(opt, device)
        ema_state_raw = ckpt.get("ema_state")
        if isinstance(ema_state_raw, dict):
            ema_state = {
                name: tensor.to(device=device, dtype=torch.float32)
                for name, tensor in ema_state_raw.items()
                if isinstance(tensor, torch.Tensor)
            }
        swa_state_raw = ckpt.get("swa_state")
        if isinstance(swa_state_raw, dict):
            swa_state = {
                name: tensor.to(dtype=base_model.state_dict()[name].dtype if name in base_model.state_dict() else tensor.dtype)
                for name, tensor in swa_state_raw.items()
                if isinstance(tensor, torch.Tensor)
            }
        else:
            swa_state = None
        swa_count = int(ckpt.get("swa_count", 0))
        training_time_ms = float(ckpt.get("training_time_ms", 0.0))
        step = int(ckpt.get("step", 0))
        CastedLinear._qat_enabled = bool(ckpt.get("qat_enabled", CastedLinear._qat_enabled))
        CastedLinear._soft_round_temp = float(ckpt.get("soft_round_temp", CastedLinear._soft_round_temp))
        qat_start_step = int(ckpt.get("qat_start_step", qat_start_step))
        loader_state = ckpt.get("train_loader")
        if train_loader is not None and isinstance(loader_state, dict):
            train_loader.load_state_dict(loader_state)
        tracker_state = ckpt.get("ngram_tracker")
        if base_model._ngram_tracker is not None and isinstance(tracker_state, dict):
            base_model._ngram_tracker.load_state_dict(tracker_state)
        rng_python = ckpt.get("rng_python")
        if rng_python is not None:
            random.setstate(rng_python)
        rng_numpy = ckpt.get("rng_numpy")
        if rng_numpy is not None:
            np.random.set_state(rng_numpy)
        rng_torch = ckpt.get("rng_torch")
        if isinstance(rng_torch, torch.Tensor):
            torch.set_rng_state(rng_torch)
        rng_cuda = ckpt.get("rng_cuda")
        if isinstance(rng_cuda, torch.Tensor):
            torch.cuda.set_rng_state(rng_cuda, device=device)
        if max_wallclock_ms is not None and training_time_ms >= max_wallclock_ms:
            stop_after_step = step
        log0(f"[spot] resumed s:{step} tt:{training_time_ms:.0f}ms qat:{CastedLinear._qat_enabled}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if not resume_pending:
        step = 0
    last_ckpt_time = time.perf_counter()
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
            log0(f"s:{step}/{args.iterations} vl:{val_loss:.4f} bpb:{val_bpb:.4f} tt:{training_time_ms:.0f}ms sa:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stop tt:{training_time_ms:.0f}ms s:{step}/{args.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            qat_start_step = step
            log0(f"qat:{step} s:{scale:.4f}")
        if CastedLinear._qat_enabled and CastedLinear._soft_round_qat and qat_start_step >= 0:
            qat_total = max(args.iterations - qat_start_step, 1)
            qat_progress = min((step - qat_start_step) / qat_total, 1.0)
            log_start = math.log(args.soft_round_temp_start)
            log_end = math.log(args.soft_round_temp_end)
            CastedLinear._soft_round_temp = math.exp(log_start + qat_progress * (log_end - log_start))
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            assert train_loader is not None
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
            if base_model._ngram_tracker is not None:
                base_model._ngram_tracker.update(x, y)
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
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(f"s:{step}/{args.iterations} tl:{train_loss.item():.4f} tt:{approx_training_time_ms:.0f}ms sa:{approx_training_time_ms/step:.2f}ms")
        should_ckpt = spot_ckpt_path is not None and args.ckpt_every_secs > 0
        if should_ckpt:
            need_ckpt = master_process and (time.perf_counter() - last_ckpt_time) >= args.ckpt_every_secs
            need_ckpt = broadcast_bool(need_ckpt, device)
            if need_ckpt:
                save_training_checkpoint(
                    spot_ckpt_path,
                    base_model,
                    optimizers,
                    train_loader,
                    base_model._ngram_tracker,
                    ema_state,
                    swa_state,
                    swa_count,
                    step,
                    approx_training_time_ms,
                    qat_start_step,
                    master_process,
                    distributed,
                    device,
                    log0,
                )
                last_ckpt_time = time.perf_counter()
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(f"mem:{torch.cuda.max_memory_allocated()//1024//1024}M R:{torch.cuda.max_memory_reserved()//1024//1024}M")
    current_state = base_model.state_dict()
    if args.swa_enabled and swa_state is not None and swa_count > 0:
        log0(f"swa:apply n:{swa_count}")
        avg_state = {
            name: (t / swa_count).to(dtype=current_state[name].dtype)
            for name, t in swa_state.items()
        }
    else:
        log0("ema:apply")
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"diag vl:{diag_val_loss:.4f} bpb:{diag_val_bpb:.4f} t:{1000.0*(time.perf_counter()-t_diag):.0f}ms")
    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)
    if excluded_mtp > 0:
        log0(f"excl_mtp:{excluded_mtp}")
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"model:{model_bytes}B")
        log0(f"code:{code_bytes}B")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, get_quant_bits_by_cat(args))
    quant_stats = summarize_mixed_quantized_artifact(quant_result, quant_meta)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob, quant_codec = compress_artifact_blob(quant_raw, args.artifact_codec)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"q:{quant_file_bytes}B")
        log0(f"total:{quant_file_bytes+code_bytes}B")
        log0(
            "qdiag "
            f"raw={len(quant_raw)}B codec={quant_codec} blob={len(quant_blob)}B "
            f"ratio={len(quant_blob)/max(len(quant_raw),1):.4f} "
            f"int5q={quant_stats['int5_packed_bytes']}B int5s={quant_stats['int5_scale_bytes']}B "
            f"int6q={quant_stats['int6_packed_bytes']}B int6s={quant_stats['int6_scale_bytes']}B "
            f"int8q={quant_stats['int8_q_bytes']}B int8s={quant_stats['int8_scale_bytes']}B "
            f"pass={quant_stats['passthrough_bytes']}B"
        )
        copy_final_artifacts_to_ckpt_dir(args, master_process, log0)
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_raw_disk, _ = decompress_artifact_blob(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(quant_raw_disk), map_location="cpu")
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
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
        vrl_enabled=args.vrl_enabled, leaky_relu=args.leaky_relu,
        gated_attention=args.gated_attention,
    ).to(device).bfloat16()
    for name, m in eval_model.named_modules():
        if isinstance(m, CastedLinear):
            if ".mlp." in name:
                m.quant_bits = args.mlp_quant_bits
            else:
                m.quant_bits = args.main_quant_bits
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = maybe_compile(eval_model, args.compile_model)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(f"q_rt vl:{q_val_loss:.4f} bpb:{q_val_bpb:.4f} t:{1000.0*(time.perf_counter()-t_qeval):.0f}ms")
    log0(f"q_rt_x vl:{q_val_loss:.8f} bpb:{q_val_bpb:.8f}")
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            batch_seqs=args.eval_batch_seqs,
            eval_seq_len=sw_seq_len,
            log0=log0,
        )
        torch.cuda.synchronize()
        log0(f"q_sw vl:{sw_val_loss:.4f} bpb:{sw_val_bpb:.4f} s:{args.eval_stride} t:{1000.0*(time.perf_counter()-t_slide):.0f}ms")
        log0(f"q_sw_x vl:{sw_val_loss:.8f} bpb:{sw_val_bpb:.8f}")
        log0(f"q8_x vl:{sw_val_loss:.8f} bpb:{sw_val_bpb:.8f}")
    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize()
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64,
            batch_seqs=args.eval_batch_seqs,
            eval_seq_len=sw_seq_len,
            log0=log0,
        )
        torch.cuda.synchronize()
        log0(f"q_s64 vl:{sw64_val_loss:.4f} bpb:{sw64_val_bpb:.4f} s:64 t:{1000.0*(time.perf_counter()-t_slide64):.0f}ms")
        log0(f"q_s64_x vl:{sw64_val_loss:.8f} bpb:{sw64_val_bpb:.8f}")
        log0(f"q8_x vl:{sw64_val_loss:.8f} bpb:{sw64_val_bpb:.8f}")
    if args.ttt_enabled or args.use_hedge_mixer:
        metric_name = "ttt" if args.ttt_enabled else "ngram"
        log0(f"{metric_name}:start")
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, batch_seqs=args.ttt_batch_seqs, log0=log0,
        )
        torch.cuda.synchronize()
        log0(f"{metric_name} vl:{ttt_val_loss:.4f} bpb:{ttt_val_bpb:.4f} t:{1000.0*(time.perf_counter()-t_ttt):.0f}ms")
        log0(f"{metric_name}_x vl:{ttt_val_loss:.8f} bpb:{ttt_val_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
