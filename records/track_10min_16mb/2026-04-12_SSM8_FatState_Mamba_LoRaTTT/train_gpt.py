import os
import sys
import io
import time
import math
import zlib
import queue
import threading
import subprocess
import glob
import random
import datetime
import contextlib
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import numpy as np
import sentencepiece as spm

import zstandard
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


TOTAL_SECONDS         = float(os.environ.get("TOTAL_SECONDS", "600.0"))
MAX_EVAL_TIME         = float(os.environ.get("MAX_EVAL_TIME", "600.0"))
COMPILE_MODE          = os.environ.get("COMPILE_MODE", "max-autotune")
ARTIFACT_BUDGET_BYTES = int(os.environ.get("ARTIFACT_BUDGET_BYTES", "16000000"))


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class SSM8Config:
    vocab_size:        int   = 1056
    tie_weights:       bool  = True
    d_model:           int   = 640
    d_inner:           int   = 1280
    d_state:           int   = 34
    d_conv:            int   = 4
    num_layers:        int   = 8
    head_adapter_rank: int   = 16
    dt_min:            float = 0.001
    dt_max:            float = 0.1
    bias:              bool  = False
    conv_bias:         bool  = True
    eps:               float = 1e-6
    qat_threshold:     float = 0.15


def build_config() -> SSM8Config:
    cfg = SSM8Config()
    cfg.d_state           = max(8, int(os.environ.get("D_STATE",           str(cfg.d_state))))
    cfg.head_adapter_rank = max(0, int(os.environ.get("HEAD_ADAPTER_RANK", str(cfg.head_adapter_rank))))
    return cfg


# =============================================================================
# 2. MUON OPTIMIZER
# =============================================================================

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.04):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None or p.ndim != 2:
                    continue
                g = p.grad
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(mu).add_(g)
                g_ortho = zeropower_via_newtonschulz5(buf)
                scale = max(1.0, g_ortho.shape[0] / g_ortho.shape[1]) ** 0.5
                p.data.add_(g_ortho, alpha=-lr * scale)


# =============================================================================
# 3. MODEL
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


# Excluded from torch.compile: Inductor cannot trace selective_scan CUDA kernel.
@torch.compiler.disable
def _run_selective_scan(u, dt, A, B, C, D, z, dt_bias,
                        return_last_state: bool = False):
    # All inputs forced to fp32 to prevent BF16 instability under 8-GPU all-reduce.
    return selective_scan_fn(
        u.float(), dt.float(), A, B.float(), C.float(),
        D=D, z=z.float(),
        delta_bias=dt_bias,
        delta_softplus=True,
        return_last_state=return_last_state,
    )


class SSM8Block(nn.Module):
    def __init__(self, config: SSM8Config):
        super().__init__()
        self.config  = config
        d_model      = config.d_model
        d_inner      = config.d_inner
        d_state      = config.d_state
        d_conv       = config.d_conv
        dt_rank      = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=config.bias)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, bias=config.conv_bias,
                                  kernel_size=d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj   = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=config.bias)

        # Log-uniform dt initialisation for ODE stability.
        dt_vals = torch.exp(
            torch.rand(d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
        inv_dt = dt_vals + torch.log(-torch.expm1(-dt_vals))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.weight.data.normal_(0.0, 0.02)

        # Log-HiPPO A initialisation.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection D, identity initialisation.
        self.D = nn.Parameter(torch.ones(d_inner, dtype=torch.float32))

        # Depth-scaled output projection.
        scaled_std = 0.02 / math.sqrt(2.0 * config.num_layers)
        nn.init.normal_(self.out_proj.weight, 0.0, scaled_std)

    def forward(self, hidden_states: torch.Tensor,
                initial_state: Optional[torch.Tensor] = None,
                return_state: bool = False):
        B, L, _ = hidden_states.shape

        xz   = self.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        x    = x.contiguous()
        z    = z.contiguous()

        conv_w = self.conv1d.weight.squeeze(1)
        x      = causal_conv1d_fn(x, conv_w, self.conv1d.bias, activation="silu")

        x_dbl               = self.x_proj(x.transpose(1, 2))
        dt_raw, B_mat, C_mat = torch.split(
            x_dbl, [self.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        dt    = F.linear(dt_raw, self.dt_proj.weight).transpose(1, 2).contiguous()
        B_mat = B_mat.transpose(1, 2).contiguous()
        C_mat = C_mat.transpose(1, 2).contiguous()

        A_fp32       = -torch.exp(self.A_log.float())
        D_fp32       = self.D.float()
        dt_bias_fp32 = self.dt_proj.bias.float()

        if initial_state is not None:
            try:
                result = _run_selective_scan(
                    x, dt, A_fp32, B_mat, C_mat, D_fp32, z, dt_bias_fp32,
                    return_last_state=return_state)
            except TypeError:
                result      = _run_selective_scan(x, dt, A_fp32, B_mat, C_mat,
                                                   D_fp32, z, dt_bias_fp32,
                                                   return_last_state=False)
                return_state = False
        else:
            result = _run_selective_scan(
                x, dt, A_fp32, B_mat, C_mat, D_fp32, z, dt_bias_fp32,
                return_last_state=return_state)

        if return_state and isinstance(result, tuple):
            y, last_state = result
        else:
            y          = result if not isinstance(result, tuple) else result[0]
            last_state = None

        y   = y.to(hidden_states.dtype)
        out = self.out_proj(y.transpose(1, 2))

        return (out, last_state) if return_state else out


class SSM8Layer(nn.Module):
    def __init__(self, config: SSM8Config):
        super().__init__()
        self.norm  = RMSNorm(config.d_model, config.eps)
        self.block = SSM8Block(config)

    def forward(self, hidden_states: torch.Tensor,
                initial_state: Optional[torch.Tensor] = None,
                return_state: bool = False):
        if return_state:
            out, state = self.block(self.norm(hidden_states),
                                    initial_state=initial_state, return_state=True)
            return hidden_states + out, state
        out = self.block(self.norm(hidden_states),
                         initial_state=initial_state, return_state=False)
        return hidden_states + out


class SSM8Model(nn.Module):
    def __init__(self, config: SSM8Config):
        super().__init__()
        self.config            = config
        self.embedding         = nn.Embedding(config.vocab_size, config.d_model)
        self.logit_bias        = nn.Parameter(torch.zeros(config.vocab_size))
        self.head_adapter_rank = max(0, config.head_adapter_rank)

        if self.head_adapter_rank > 0:
            self.head_adapter_A = nn.Parameter(
                torch.empty(self.head_adapter_rank, config.d_model))
            self.head_adapter_B = nn.Parameter(
                torch.empty(config.vocab_size, self.head_adapter_rank))
            nn.init.normal_(self.head_adapter_A, mean=0.0, std=0.02)
            nn.init.zeros_(self.head_adapter_B)
        else:
            self.register_parameter('head_adapter_A', None)
            self.register_parameter('head_adapter_B', None)

        self.layers     = nn.ModuleList([SSM8Layer(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model, config.eps)

    def forward(self, x: torch.Tensor,
                ssm_states: Optional[List] = None,
                return_states: bool = False,
                use_checkpoint: bool = False) -> Tuple:
        h          = self.embedding(x)
        new_states = [] if return_states else None

        for i, layer in enumerate(self.layers):
            state_in = ssm_states[i] if ssm_states else None

            if use_checkpoint and self.training:
                h = grad_checkpoint(layer, h, use_reentrant=True)
                if return_states:
                    new_states.append(None)
            else:
                result = layer(h, initial_state=state_in, return_state=return_states)
                if return_states:
                    h, state_out = result
                    new_states.append(state_out)
                else:
                    h = result

        h      = self.final_norm(h)
        logits = F.linear(h, self.embedding.weight, self.logit_bias)

        if self.head_adapter_A is not None and self.head_adapter_B is not None:
            adapter_hidden = F.linear(h, self.head_adapter_A)
            logits         = logits + F.linear(adapter_hidden, self.head_adapter_B)

        # Soft logit cap: stabilises int6 quantisation.
        cap    = 30.0
        logits = torch.tanh(logits / cap) * cap

        return (logits, new_states) if return_states else logits


# =============================================================================
# 4. INT6 QUANTISATION
# =============================================================================

INT6_MAX = 31.0
INT8_MAX = 127.0


def fake_quant_int6(p: torch.Tensor) -> torch.Tensor:
    scale = p.float().abs().max(dim=-1, keepdim=True).values / INT6_MAX + 1e-8
    q     = (p.float() / scale).round().clamp(-INT6_MAX, INT6_MAX)
    return (q * scale).to(p.dtype)


def apply_qat_snap(model: nn.Module):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and 'embedding' not in name:
                p.data.copy_(fake_quant_int6(p.data))


def _gptq_lite_int6(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    t_f                      = tensor.float()
    best_q, best_s, best_mse = None, None, float('inf')
    for pct in [0.999, 0.9995, 0.9999, 0.99999, 1.0]:
        clip  = (torch.quantile(t_f.abs(), pct, dim=1, keepdim=True)
                 if pct < 1.0 else t_f.abs().max(dim=1, keepdim=True).values)
        tc    = t_f.clamp(-clip, clip)
        scale = (clip / INT6_MAX).clamp(min=1e-8)
        q     = (tc / scale).round().clamp(-INT6_MAX, INT6_MAX).to(torch.int8)
        mse   = (t_f - q.float() * scale).pow(2).sum().item()
        if mse < best_mse:
            best_mse = mse
            best_q   = q
            best_s   = scale.squeeze(1).bfloat16()
    return best_q, best_s


def _int8_per_row(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    t_f   = tensor.float()
    clip  = t_f.abs().max(dim=1, keepdim=True).values
    scale = (clip / INT8_MAX).clamp(min=1e-8)
    q     = (t_f / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q, scale.squeeze(1).bfloat16()


def save_artifact(model: nn.Module, path: str):
    state_dict  = model.state_dict()
    quantized   = {}
    q_scheme    = {}
    passthrough = {}

    for name, t in state_dict.items():
        cpu = t.cpu()
        if 'embedding' in name:
            q, s            = _int8_per_row(cpu)
            quantized[name] = (q, s)
            q_scheme[name]  = 'int8'
        elif cpu.ndim >= 2 and cpu.numel() > 2048:
            q, s            = _gptq_lite_int6(cpu)
            quantized[name] = (q, s)
            q_scheme[name]  = 'int6'
        else:
            passthrough[name] = cpu.bfloat16()

    artifact = dict(quantized=quantized, q_scheme=q_scheme, passthrough=passthrough)
    buf      = io.BytesIO()
    torch.save(artifact, buf)
    raw        = buf.getvalue()
    compressed = zstandard.ZstdCompressor(level=22).compress(raw)

    with open(path, 'wb') as f:
        f.write(compressed)

    code_bytes  = os.path.getsize(sys.argv[0]) if os.path.exists(sys.argv[0]) else 0
    model_bytes = len(compressed)
    total       = code_bytes + model_bytes
    budget_mb   = ARTIFACT_BUDGET_BYTES / 1e6
    print(f"[ARTIFACT] zstd-22 | code={code_bytes/1e6:.2f}MB "
          f"model={model_bytes/1e6:.2f}MB total={total/1e6:.2f}MB/"
          f"{budget_mb:.2f}MB ({total} / {ARTIFACT_BUDGET_BYTES} bytes)")
    if total > ARTIFACT_BUDGET_BYTES:
        print("[ARTIFACT] WARNING: artifact exceeds budget!")


def load_artifact(model: nn.Module, path: str):
    with open(path, 'rb') as f:
        raw_comp = f.read()
    raw      = zstandard.ZstdDecompressor().decompress(raw_comp)
    artifact = torch.load(io.BytesIO(raw), weights_only=False, map_location='cpu')
    new_sd   = {
        name: (q.float() * s.float().unsqueeze(1)).bfloat16()
        for name, (q, s) in artifact['quantized'].items()
    }
    for name, t in artifact['passthrough'].items():
        new_sd[name] = t
    model.load_state_dict(new_sd, strict=True)


# =============================================================================
# 5. LORA FOR TEST-TIME TRAINING
# =============================================================================

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base               = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.r       = r
        self.scaling = alpha / r
        dev, dt      = base.weight.device, base.weight.dtype
        self.lora_A  = nn.Parameter(torch.empty(r, base.in_features,  device=dev, dtype=dt))
        self.lora_B  = nn.Parameter(torch.empty(base.out_features, r, device=dev, dtype=dt))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAInjector:
    def __init__(self, r: int = 8, alpha: float = 16.0):
        self.r, self.alpha = r, alpha
        self.injected: List[LoRALinear] = []

    def inject(self, model: nn.Module):
        self.injected = []
        for module in model.modules():
            if isinstance(module, SSM8Block):
                for attr in ('in_proj', 'out_proj'):
                    orig = getattr(module, attr)
                    lora = LoRALinear(orig, self.r, self.alpha)
                    setattr(module, attr, lora)
                    self.injected.append(lora)
        print(f"[TTT] LoRA r={self.r} -> {len(self.injected)} projections.")

    def reset(self):
        with torch.no_grad():
            for l in self.injected:
                l.reset_parameters()

    def params(self) -> List[nn.Parameter]:
        return [p for l in self.injected for p in [l.lora_A, l.lora_B]]


# =============================================================================
# 6. DATA LOADING
# =============================================================================

class EntropyFilter:
    def __init__(self, initial_ratio: float = 4.0):
        self.current_ratio  = initial_ratio
        self.stats_accepted = 0
        self.stats_rejected = 0

    def update(self, elapsed_seconds: float, total_seconds: float):
        progress = elapsed_seconds / max(1.0, total_seconds)
        if progress >= 0.80:
            self.current_ratio = 1.8
        elif progress >= 0.50:
            self.current_ratio = 2.0
        elif progress >= 0.20:
            self.current_ratio = 2.5
        else:
            self.current_ratio = 4.0

    def is_valid(self, chunk: np.ndarray) -> bool:
        if chunk.size == 0:
            return False
        raw   = chunk.tobytes()
        ratio = len(raw) / len(zlib.compress(raw, level=1))
        if ratio > self.current_ratio:
            self.stats_rejected += 1
            return False
        self.stats_accepted += 1
        return True

    def yield_pct(self) -> float:
        total = self.stats_accepted + self.stats_rejected
        return 100.0 * self.stats_accepted / total if total else 0.0


class FastGolfDataLoader:
    def __init__(self, file_paths: List[str], batch_size: int, seq_len: int,
                 entropy_filter: EntropyFilter,
                 rank: int = 0, world_size: int = 1, seed: int = 42):
        if not file_paths:
            raise ValueError("No data files provided.")
        self.file_paths     = file_paths.copy()
        self.batch_size     = batch_size
        self.seq_len        = seq_len
        self.entropy_filter = entropy_filter
        self.rank           = rank
        self.world_size     = world_size
        self.chunk_size     = batch_size * (seq_len + 1)

        rng              = random.Random(seed)
        rng.shuffle(self.file_paths)
        self.current_idx = rng.randint(0, len(self.file_paths) - 1)
        self._load_file()

        global_chunk     = world_size * self.chunk_size
        max_off          = max(0, self.total_len - global_chunk)
        aligned_max      = max_off - (max_off % global_chunk)
        self.init_offset = (rng.randint(0, aligned_max // global_chunk)
                            * global_chunk) if aligned_max > 0 else 0

        self.queue      = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self._thread    = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _load_file(self):
        path           = self.file_paths[self.current_idx]
        self.data      = np.memmap(path, dtype=np.uint16, mode='r')
        self.total_len = len(self.data)
        if self.total_len < self.world_size * self.chunk_size:
            raise ValueError(f"File too small: {path}")

    def _worker(self):
        try:
            ptr             = self.rank * self.chunk_size + self.init_offset
            consec_rejected = 0
            while not self.stop_event.is_set():
                if ptr + self.chunk_size > self.total_len:
                    del self.data
                    self.current_idx = (self.current_idx + 1) % len(self.file_paths)
                    self._load_file()
                    ptr = self.rank * self.chunk_size
                    continue
                chunk  = self.data[ptr: ptr + self.chunk_size]
                ptr   += self.world_size * self.chunk_size
                if not self.entropy_filter.is_valid(chunk):
                    consec_rejected += 1
                    if consec_rejected >= 20:
                        consec_rejected = 0
                        self.entropy_filter.stats_rejected -= 1
                        self.entropy_filter.stats_accepted += 1
                    else:
                        continue
                else:
                    consec_rejected = 0
                t = torch.from_numpy(chunk.copy()).view(self.batch_size, self.seq_len + 1)
                self.queue.put(t.pin_memory())
        except Exception as e:
            import traceback
            print(f"[DataLoader rank {self.rank}] FATAL: {e}", flush=True)
            traceback.print_exc()
            self.stop_event.set()

    def get_batch(self, device: torch.device, vocab_size: int = 0):
        if self.stop_event.is_set() and self.queue.empty():
            raise RuntimeError("DataLoader worker died.")
        t = self.queue.get()
        X = t[:, :-1].to(device, dtype=torch.long, non_blocking=True)
        Y = t[:, 1:].to(device,  dtype=torch.long, non_blocking=True)
        if vocab_size > 0:
            X = X.clamp(0, vocab_size - 1)
            Y = Y.clamp(0, vocab_size - 1)
        return X, Y

    def __del__(self):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


# =============================================================================
# 7. TRAINING
# =============================================================================

def _compute_wsd_progress(elapsed: float, total: float, wu: float, st: float) -> float:
    return min(1.0, (elapsed - wu - st) / max(1.0, total - wu - st))


def get_wsd_lr(elapsed: float, total: float) -> float:
    wu = 0.10 * total
    st = 0.70 * total
    if elapsed < wu:
        return elapsed / max(1.0, wu)
    if elapsed < wu + st:
        return 1.0
    prog = _compute_wsd_progress(elapsed, total, wu, st)
    return 0.01 + 0.5 * 0.99 * (1.0 + math.cos(math.pi * prog))


def train():
    start_wall  = time.perf_counter()
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK",       "0"))
    world_size  = int(os.environ.get("WORLD_SIZE",  "1"))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    seed = int(os.environ.get("SEED", "1337"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark          = True
    torch.backends.cuda.matmul.allow_tf32  = True

    DATA_PATH     = os.environ.get("DATA_PATH",  "./data/datasets/fineweb10B_sp1024")
    OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", "./")
    ARTIFACT_FILE = os.path.join(OUTPUT_DIR, "ssm8_artifact.bin")

    train_files = sorted(glob.glob(os.path.join(DATA_PATH, "fineweb_train_*.bin")))
    if not train_files:
        if global_rank == 0:
            print(f"[ERROR] No training shards in {DATA_PATH}")
        dist.destroy_process_group()
        sys.exit(1)

    if global_rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Found {len(train_files)} training shards.")
    dist.barrier()

    SAFE_BUFFER_SEC = 5.0
    EFFECTIVE_TIME  = TOTAL_SECONDS - SAFE_BUFFER_SEC
    if global_rank == 0:
        print(f"[TIMING] Budget: {TOTAL_SECONDS:.0f}s | "
              f"Safe buffer: {SAFE_BUFFER_SEC:.0f}s | "
              f"Effective training: {EFFECTIVE_TIME:.0f}s")

    BATCH_SIZE     = 32
    SEQ_LEN        = 2048
    GRAD_ACCUM     = 1
    USE_CHECKPOINT = True
    LOG_EVERY      = 30.0
    QAT_SNAP_EVERY = 20

    lr_scale_power_default = 0.35 if world_size >= 8 else 0.5
    lr_scale_power         = float(os.environ.get("LR_SCALE_POWER", str(lr_scale_power_default)))
    ws_scale               = float(world_size) ** lr_scale_power

    base_lr_base   = float(os.environ.get("BASE_LR_BASE",   "0.010"))
    embed_lr_base  = float(os.environ.get("EMBED_LR_BASE",  "0.035"))
    scalar_lr_base = float(os.environ.get("SCALAR_LR_BASE", "0.020"))

    max_base_default   = 0.022 if world_size >= 8 else 1.0
    max_embed_default  = 0.060 if world_size >= 8 else 1.0
    max_scalar_default = 0.040 if world_size >= 8 else 1.0
    max_base_lr   = float(os.environ.get("MAX_BASE_LR",   str(max_base_default)))
    max_embed_lr  = float(os.environ.get("MAX_EMBED_LR",  str(max_embed_default)))
    max_scalar_lr = float(os.environ.get("MAX_SCALAR_LR", str(max_scalar_default)))

    BASE_LR   = min(base_lr_base   * ws_scale, max_base_lr)
    EMBED_LR  = min(embed_lr_base  * ws_scale, max_embed_lr)
    SCALAR_LR = min(scalar_lr_base * ws_scale, max_scalar_lr)
    MUON_WD   = 0.04
    ADAM_WD   = 0.04

    grad_clip_default = 0.15 if world_size >= 8 else 0.3
    GRAD_CLIP         = float(os.environ.get("GRAD_CLIP", str(grad_clip_default)))

    MU_START        = float(os.environ.get("MU_START", "0.92"))
    mu_end_default  = 0.95 if world_size >= 8 else 0.99
    MU_END          = float(os.environ.get("MU_END", str(mu_end_default)))

    config = build_config()
    model  = SSM8Model(config).to(device)

    if global_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] Params: {total_params:,} | BF16: {total_params*2/1e6:.1f}MB "
              f"| d_state={config.d_state} head_r={config.head_adapter_rank} "
              f"| Use checkpoint: {USE_CHECKPOINT}")
        print(f"[OPT] ws_scale={ws_scale:.3f} (power={lr_scale_power:.2f}) "
              f"BASE_LR={BASE_LR:.5f} EMBED_LR={EMBED_LR:.5f} "
              f"SCALAR_LR={SCALAR_LR:.5f} GRAD_CLIP={GRAD_CLIP:.2f} "
              f"MU_START={MU_START:.3f} MU_END={MU_END:.3f}")

    model     = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module

    ema_model = SSM8Model(config).to(device)
    ema_model.load_state_dict(raw_model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False
    EMA_DECAY = 0.997

    dist.barrier()
    model = torch.compile(model, mode=COMPILE_MODE)

    entropy_filter = EntropyFilter(initial_ratio=4.0)

    seed_t = torch.zeros(1, dtype=torch.long, device=device)
    if global_rank == 0:
        seed_t[0] = int(time.time()) % 1_000_000
    dist.broadcast(seed_t, src=0)

    dataloader = FastGolfDataLoader(
        file_paths=train_files, batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        entropy_filter=entropy_filter,
        rank=global_rank, world_size=world_size, seed=int(seed_t.item()))

    muon_params, embed_params, scalar_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and 'embedding' not in name:
            muon_params.append(p)
        elif 'embedding' in name:
            embed_params.append(p)
        else:
            scalar_params.append(p)

    opt_muon = Muon(muon_params, lr=BASE_LR, momentum=MU_START, weight_decay=MUON_WD)
    opt_adam = torch.optim.AdamW(
        [{'params': embed_params,  'lr': EMBED_LR},
         {'params': scalar_params, 'lr': SCALAR_LR}],
        betas=(0.9, 0.95), eps=1e-8, weight_decay=ADAM_WD, fused=True)

    # Dummy forward/backward to trigger torch.compile kernel generation.
    dX = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN),
                       device=device, dtype=torch.long)
    dY = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN),
                       device=device, dtype=torch.long)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(dX, use_checkpoint=USE_CHECKPOINT)
        loss   = F.cross_entropy(logits.view(-1, config.vocab_size), dY.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt_muon.step(); opt_adam.step()
    opt_muon.zero_grad(set_to_none=True); opt_adam.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dist.barrier()

    if global_rank == 0:
        print(f"Warmup + compile: {time.perf_counter()-start_wall:.1f}s")

    train_start_time = 0.0
    timer_started    = False
    last_log_time    = 0.0
    step             = 0
    if global_rank == 0:
        print(f"[TIMING] Training budget: {EFFECTIVE_TIME:.0f}s (TOTAL_SECONDS - buffer).")

    while True:
        if timer_started and step % 50 == 0:
            elapsed = time.perf_counter() - train_start_time
            flag    = torch.zeros(1, dtype=torch.int32, device=device)
            if elapsed >= EFFECTIVE_TIME:
                flag[0] = 1
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item():
                if global_rank == 0:
                    print(f"[TIMING] Time limit reached, saving artifact.")
                break

        X, Y = dataloader.get_batch(device, vocab_size=config.vocab_size)

        if not timer_started:
            train_start_time = time.perf_counter()
            timer_started    = True

        elapsed  = time.perf_counter() - train_start_time
        lr_scale = get_wsd_lr(elapsed, EFFECTIVE_TIME)

        # Momentum ramp based on wall-clock time, not steps.
        mu_warmup_seconds = 0.15 * EFFECTIVE_TIME
        mu = MU_START + (MU_END - MU_START) * min(1.0, elapsed / mu_warmup_seconds)

        for g in opt_muon.param_groups:
            g['lr']       = BASE_LR * lr_scale
            g['momentum'] = mu
        opt_adam.param_groups[0]['lr'] = EMBED_LR  * lr_scale
        opt_adam.param_groups[1]['lr'] = SCALAR_LR * lr_scale

        entropy_filter.update(elapsed, TOTAL_SECONDS)

        is_accum = (step + 1) % GRAD_ACCUM != 0
        ctx      = model.no_sync() if is_accum else contextlib.nullcontext()

        with ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits  = model(X, use_checkpoint=USE_CHECKPOINT)
                ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), Y.view(-1))
                if step % 2 == 0:
                    z_loss = 2e-4 * torch.logsumexp(logits, dim=-1).pow(2).mean()
                    loss   = (ce_loss + z_loss) / GRAD_ACCUM
                else:
                    loss = ce_loss / GRAD_ACCUM

            finite_flag = torch.ones(1, dtype=torch.int32, device=device)
            if not torch.isfinite(loss.detach()):
                finite_flag[0] = 0
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if finite_flag.item() == 0:
                if global_rank == 0:
                    print(f"[TRAIN] Non-finite loss at step={step}, stopping.")
                break

            loss.backward()

        if not is_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt_muon.step(); opt_adam.step()
            opt_muon.zero_grad(set_to_none=True); opt_adam.zero_grad(set_to_none=True)

        with torch.no_grad():
            ema_ps = list(ema_model.parameters())
            raw_ps = list(raw_model.parameters())
            torch._foreach_mul_(ema_ps, EMA_DECAY)
            torch._foreach_add_(ema_ps, raw_ps, alpha=1.0 - EMA_DECAY)
            for eb, rb in zip(ema_model.buffers(), raw_model.buffers()):
                eb.data.copy_(rb.data)

        qat_active = lr_scale < config.qat_threshold
        if qat_active and step % QAT_SNAP_EVERY == 0:
            apply_qat_snap(raw_model)

        log_flag = torch.zeros(1, dtype=torch.int32, device=device)
        if global_rank == 0 and elapsed - last_log_time >= LOG_EVERY:
            log_flag[0]   = 1
            last_log_time = elapsed
        dist.broadcast(log_flag, src=0)

        if log_flag.item():
            yt = torch.tensor([entropy_filter.yield_pct()], device=device)
            dist.all_reduce(yt, op=dist.ReduceOp.AVG)
            if global_rank == 0:
                print(f"[STEP {step:05d}] t={elapsed:.0f}s/{EFFECTIVE_TIME:.0f}s "
                      f"loss={loss.item()*GRAD_ACCUM:.4f} "
                      f"lr_scale={lr_scale:.3f} mu={mu:.3f} "
                      f"qat={qat_active} yield={yt.item():.0f}%")
        step += 1

    torch.cuda.synchronize()
    dist.barrier()

    if global_rank == 0:
        print("[SAVE] Quantising EMA model with GPTQ-lite + zstd-22...")
        save_artifact(ema_model, ARTIFACT_FILE)

    dist.barrier()


# =============================================================================
# 8. EVALUATION WITH SCORE-FIRST TTT
# =============================================================================

def evaluate():
    global_rank = int(os.environ.get("RANK",      "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    device      = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if global_rank == 0:
        print(f"\n[EVAL] Running on {world_size} GPU(s)")

    start_time     = time.perf_counter()
    DATA_PATH      = os.environ.get("DATA_PATH",      "./data/datasets/fineweb10B_sp1024")
    OUTPUT_DIR     = os.environ.get("OUTPUT_DIR",      "./")
    ARTIFACT_FILE  = os.path.join(OUTPUT_DIR, "ssm8_artifact.bin")
    TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")

    dist.barrier()

    val_files = sorted(glob.glob(os.path.join(DATA_PATH, "fineweb_val_*.bin")))
    if not val_files:
        if global_rank == 0:
            print("[EVAL] No validation files found.")
        return

    config = build_config()
    model  = SSM8Model(config).to(device)

    artifact_ready = torch.zeros(1, dtype=torch.int32, device=device)
    if global_rank == 0 and os.path.exists(ARTIFACT_FILE):
        artifact_ready[0] = 1
    dist.broadcast(artifact_ready, src=0)

    if artifact_ready.item() != 1:
        raise RuntimeError(f"[EVAL] Missing artifact: {ARTIFACT_FILE}")

    load_artifact(model, ARTIFACT_FILE)
    if global_rank == 0:
        print(f"[EVAL] Loaded artifact from {ARTIFACT_FILE}")

    dist.barrier()
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    lora    = LoRAInjector(r=8, alpha=16.0)
    lora.inject(model)
    ttt_opt = torch.optim.AdamW(lora.params(), lr=3e-3, weight_decay=0.01)
    tau     = nn.Parameter(torch.tensor([1.0], device=device))
    tau_opt = torch.optim.SGD([tau], lr=0.05)

    sp        = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    byte_lens = []
    for i in range(config.vocab_size):
        if i < sp.get_piece_size():
            try:
                byte_lens.append(len(sp.decode_ids([i]).encode('utf-8')))
            except Exception:
                byte_lens.append(0)
        else:
            byte_lens.append(0)
    vocab_bytes     = torch.tensor(byte_lens, dtype=torch.float64, device=device)
    vocab_has_bytes = vocab_bytes > 0

    prior_counts   = torch.ones(config.vocab_size, dtype=torch.float32, device=device)
    PRIOR_STRENGTH = 0.03

    SEQ_LEN        = 2048
    STRIDE         = 256
    LOSS_THRESHOLD = 0.5
    TTT_STEPS      = 3
    EOS_TOKEN_ID   = 0

    val_file   = val_files[0]
    data       = np.memmap(val_file, dtype=np.uint16, mode='r')
    total_len  = len(data)
    base_len   = total_len // world_size
    rank_start = global_rank * base_len
    rank_end   = rank_start + base_len if global_rank < world_size - 1 else total_len

    if global_rank == 0:
        print(f"[EVAL] Validation tokens: {total_len:,}")

    data_slice = data[rank_start:rank_end]
    lora.reset()
    tau.data.fill_(1.0)
    ttt_opt.state.clear()
    tau_opt.state.clear()

    total_loss   = 0.0
    total_tokens = 0
    total_bytes  = 0.0
    eval_step    = 0
    last_report  = 0.0
    ptr          = 0
    ssm_states   = None
    abs_pos      = 0
    prev_X       = None
    prev_Y       = None

    while ptr + STRIDE + 1 <= len(data_slice):
        if eval_step % 20 == 0 and time.perf_counter() - start_time > MAX_EVAL_TIME:
            break

        eval_step  += 1
        chunk_size  = SEQ_LEN if ssm_states is None else STRIDE
        if ptr + chunk_size + 1 > len(data_slice):
            break

        chunk = data_slice[ptr: ptr + chunk_size + 1]
        X     = torch.from_numpy(chunk[:-1].copy()).long().unsqueeze(0).to(device)
        Y     = torch.from_numpy(chunk[1:].copy()).long().unsqueeze(0).to(device)
        X     = X.clamp(0, config.vocab_size - 1)
        Y     = Y.clamp(0, config.vocab_size - 1)

        if prev_X is not None:
            ttt_opt.zero_grad(set_to_none=True)
            tau_opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    lg_ttt = model(prev_X, use_checkpoint=False)
                    l_ttt  = F.cross_entropy(
                        (lg_ttt / tau).view(-1, config.vocab_size), prev_Y.view(-1))
            l_ttt.backward()
            tau_opt.step()
            tau.data.clamp_(0.8, 2.0)
            if l_ttt.item() > LOSS_THRESHOLD:
                ttt_opt.step()
                for _ in range(TTT_STEPS - 1):
                    ttt_opt.zero_grad(set_to_none=True)
                    with torch.enable_grad():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            lg2 = model(prev_X, use_checkpoint=False)
                            l2  = F.cross_entropy(
                                (lg2 / tau).view(-1, config.vocab_size), prev_Y.view(-1))
                    l2.backward()
                    ttt_opt.step()
            ttt_opt.zero_grad(set_to_none=True)
            tau_opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                try:
                    logits, new_states = model(
                        X, ssm_states=ssm_states, return_states=True, use_checkpoint=False)
                    ssm_states = new_states
                except Exception:
                    logits     = model(X, use_checkpoint=False)
                    ssm_states = None

                prior_log     = torch.log(prior_counts / prior_counts.sum())
                scaled_logits = (
                    (logits / tau) + PRIOR_STRENGTH * prior_log.view(1, 1, -1).to(logits.dtype)
                ).view(-1, config.vocab_size)
                flat_targets  = Y.view(-1)
                per_token_nll = F.cross_entropy(scaled_logits, flat_targets, reduction='none')

        token_bytes = vocab_bytes[flat_targets]
        valid_mask  = vocab_has_bytes[flat_targets]
        if valid_mask.any():
            total_loss   += per_token_nll[valid_mask].sum().item()
            total_tokens += valid_mask.sum().item()
            try:
                decoded_text  = sp.decode_ids(flat_targets.detach().cpu().tolist())
                total_bytes  += len(decoded_text.encode('utf-8'))
            except Exception:
                total_bytes  += token_bytes[valid_mask].sum().item()

        if global_rank == 0:
            elapsed_eval = time.perf_counter() - start_time
            if elapsed_eval - last_report >= 60.0:
                print(f"[EVAL] tokens_processed={total_tokens}")
                last_report = elapsed_eval

        prev_X = X
        prev_Y = Y

        with torch.no_grad():
            y_flat = Y.view(-1)
            prior_counts.index_add_(0, y_flat,
                                    torch.ones_like(y_flat, dtype=prior_counts.dtype))

        ptr     += chunk_size
        abs_pos += chunk_size

        if EOS_TOKEN_ID in X[0]:
            ssm_states = None
            abs_pos    = 0
            tau.data.fill_(1.0)
            prev_X = prev_Y = None

        if abs_pos > 4096:
            ssm_states = None
            abs_pos    = 0
            lora.reset()
            tau.data.fill_(1.0)
            prev_X = prev_Y = None

    metrics = torch.tensor([total_loss, total_tokens, total_bytes],
                           device=device, dtype=torch.float64)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    if global_rank == 0:
        g_loss, g_tokens, g_bytes = metrics.tolist()
        if g_bytes > 0:
            bpb = (g_loss / math.log(2)) / g_bytes
            print(f"\n[EVAL RESULT] BPB: {bpb:.4f} | tokens_processed={g_tokens:.0f}")


# =============================================================================
# 9. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if "RANK" not in os.environ:
        nproc = max(1, torch.cuda.device_count())
        if nproc > 1:
            print(f"Auto-launching {nproc}-GPU job via torchrun...")
            try:
                subprocess.run([
                    sys.executable, "-m", "torch.distributed.run",
                    "--standalone", f"--nproc_per_node={nproc}", sys.argv[0]
                ], check=True)
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)
            sys.exit(0)

        os.environ.setdefault("RANK",        "0")
        os.environ.setdefault("LOCAL_RANK",  "0")
        os.environ.setdefault("WORLD_SIZE",  "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(random.randint(20000, 29999)))

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    try:
        train()
        evaluate()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()