"""H-Net Milestone 1 training pilot.

Trains the HNetM1 model (hierarchical byte-level with fixed chunker) on the
cached byte shards (produced by make_byte_shards.py). Uses plain AdamW on all
parameters. Keeps it minimal: no TTT, no EMA, no Muon, no GPTQ. The question
this pilot answers: does the hierarchical architecture train at ~25M params on
FineWeb bytes?

Env knobs:
    DATA_DIR         default /workspace/parameter-golf/data/
    BYTE_DATA_DIR    default ${DATA_DIR}/datasets/fineweb10B_bytes/
    TOKENIZER_PATH   default ${DATA_DIR}/tokenizers/fineweb_8192_bpe.model
                       (used for val-bpb byte-count sanity only, not for input)
    ITERATIONS       default 300
    WARMUP_STEPS     default 10
    BYTE_SEQ_LEN     default 4096    bytes per sample
    CHUNK_STRIDE     default 4       fixed chunker stride
    BATCH_SIZE       default 8       sequences per step
    GRAD_ACCUM       default 1
    LR               default 3e-4
    WD               default 0.01
    MAX_WALLCLOCK_SECONDS  default 0  (0 = no cap)
    RUN_ID           default hnet_m1_pilot
    VAL_EVERY        default 0 (0 = only at end)
    TRAIN_LOG_EVERY  default 20

The training loop logs lines in the same shape as bigbag's, so our phase3
parser works:
    step:N/M train_loss:X train_time:Yms step_avg:Zms tok/s:W

At end: prints per-byte val_loss (nats) and val_bpb.
"""
from __future__ import annotations
import glob
import math
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
DATA_DIR         = os.environ.get("DATA_DIR", "/workspace/parameter-golf/data/")
BYTE_DATA_DIR    = os.environ.get("BYTE_DATA_DIR", os.path.join(DATA_DIR, "datasets", "fineweb10B_bytes"))
TOKENIZER_PATH   = os.environ.get("TOKENIZER_PATH", os.path.join(DATA_DIR, "tokenizers", "fineweb_8192_bpe.model"))
ITERATIONS       = int(os.environ.get("ITERATIONS", 300))
WARMUP_STEPS     = int(os.environ.get("WARMUP_STEPS", 10))
BYTE_SEQ_LEN     = int(os.environ.get("BYTE_SEQ_LEN", 4096))
CHUNK_STRIDE     = int(os.environ.get("CHUNK_STRIDE", 4))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", 8))
GRAD_ACCUM       = int(os.environ.get("GRAD_ACCUM", 1))
LR               = float(os.environ.get("LR", 3e-4))
WD               = float(os.environ.get("WD", 0.01))
MAX_WALLCLOCK    = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0))
RUN_ID           = os.environ.get("RUN_ID", "hnet_m1_pilot")
VAL_EVERY        = int(os.environ.get("VAL_EVERY", 0))
TRAIN_LOG_EVERY  = int(os.environ.get("TRAIN_LOG_EVERY", 20))
SEED             = int(os.environ.get("SEED", 42))
COMPILE          = int(os.environ.get("COMPILE", 1))   # set 0 to skip torch.compile (eager)
WARMDOWN_FRAC    = float(os.environ.get("WARMDOWN_FRAC", 0.3))  # final 30% of steps cosine to 0

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / f"{RUN_ID}.txt"

def log(msg: str, console: bool = True):
    if console:
        print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ---------------------------------------------------------------------------
# Load baseline module for Block/RMSNorm/... class defs.
# Register under sys.modules so torch.compile / dynamo can resolve globals
# referenced from Block.forward (flash_attn_3_func, F, torch, etc.) via a real
# importable module.
# ---------------------------------------------------------------------------
import types as _types
BASELINE_PATH = Path(os.environ.get("BASELINE_PATH", "/workspace/work/train_gpt_baseline.py"))
BASELINE_MOD_NAME = "baseline_ns"
_baseline_mod = _types.ModuleType(BASELINE_MOD_NAME)
_baseline_mod.__file__ = str(BASELINE_PATH)
sys.modules[BASELINE_MOD_NAME] = _baseline_mod
exec(compile(BASELINE_PATH.read_text(), str(BASELINE_PATH), "exec"), _baseline_mod.__dict__)
ns = _baseline_mod.__dict__

# Local import of the HNet model factory
sys.path.insert(0, str(Path(__file__).parent))
from hnet_m1 import build_hnet_m1, count_params  # noqa: E402

# ---------------------------------------------------------------------------
# Byte-shard loader (mirrors baseline load_data_shard but u16 values are bytes)
# ---------------------------------------------------------------------------
SHARD_MAGIC = 20240520

def load_byte_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    assert header.size == 256 and int(header[0]) == SHARD_MAGIC, f"bad header {path}"
    n = int(header[2])
    arr = np.fromfile(path, dtype="<u2", count=n, offset=256 * 4)
    assert arr.size == n
    return arr.astype(np.uint8)   # bytes are in [0,255]; downcast here

class ByteStream:
    def __init__(self, pattern: str):
        self.files = sorted(Path(p) for p in glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"no byte shards at {pattern}")
        self.file_idx = 0
        self.buf = load_byte_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.buf = load_byte_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        pieces = []
        remaining = n
        while remaining > 0:
            avail = self.buf.size - self.pos
            if avail <= 0:
                self._advance(); continue
            k = min(remaining, avail)
            pieces.append(self.buf[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    log(f"[hnet_m1] RUN_ID={RUN_ID} ITERATIONS={ITERATIONS} BYTE_SEQ_LEN={BYTE_SEQ_LEN} "
        f"CHUNK_STRIDE={CHUNK_STRIDE} BATCH_SIZE={BATCH_SIZE} GRAD_ACCUM={GRAD_ACCUM} LR={LR} WD={WD}")
    log(f"[hnet_m1] BYTE_DATA_DIR={BYTE_DATA_DIR}")

    torch.manual_seed(SEED); np.random.seed(SEED)
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build model
    model = build_hnet_m1(ns, byte_seq_len=BYTE_SEQ_LEN, chunk_stride=CHUNK_STRIDE)
    model = model.to(device).to(dtype)
    # Keep small/control parameters in fp32 (baseline convention)
    for name, p in model.named_parameters():
        if p.ndim < 2:
            p.data = p.data.float()

    total_p, nonemb_p = count_params(model)
    log(f"[hnet_m1] params total={total_p:,} non-embedding={nonemb_p:,}")

    # Optimizer: plain AdamW on matrix params; plain AdamW on vector/scalar params at same LR
    matrix_params = [p for p in model.parameters() if p.ndim == 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [
            {"params": matrix_params, "lr": LR, "weight_decay": WD},
            {"params": scalar_params, "lr": LR, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    for g in opt.param_groups: g["base_lr"] = g["lr"]

    # Data
    train_stream = ByteStream(os.path.join(BYTE_DATA_DIR, "fineweb_train_*.bin"))
    val_stream   = ByteStream(os.path.join(BYTE_DATA_DIR, "fineweb_val_*.bin"))

    def next_batch(stream, B, T):
        raw = stream.take(B * (T + 1))
        raw = raw[: B * (T + 1)].reshape(B, T + 1)
        x = torch.from_numpy(raw[:, :-1].astype(np.int64)).to(device, non_blocking=True)
        y = torch.from_numpy(raw[:,  1:].astype(np.int64)).to(device, non_blocking=True)
        return x, y

    # compile (or not)
    if COMPILE:
        log("[hnet_m1] torch.compile enabled")
        compiled = torch.compile(model, dynamic=False)
    else:
        log("[hnet_m1] torch.compile DISABLED (eager mode)")
        compiled = model

    # warmup
    for wstep in range(WARMUP_STEPS):
        x, y = next_batch(train_stream, BATCH_SIZE, BYTE_SEQ_LEN)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            loss = compiled(x, y)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if (wstep + 1) % max(1, WARMUP_STEPS // 4) == 0:
            log(f"warmup_step:{wstep+1}/{WARMUP_STEPS}")

    # restart data stream so warmup tokens don't bias training window
    train_stream = ByteStream(os.path.join(BYTE_DATA_DIR, "fineweb_train_*.bin"))

    log("[hnet_m1] entering main loop")
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    total_tokens = 0

    warmdown_start = int(ITERATIONS * (1.0 - WARMDOWN_FRAC))
    for step in range(1, ITERATIONS + 1):
        # cosine warmdown over the last WARMDOWN_FRAC fraction of steps
        if step > warmdown_start:
            frac = (step - warmdown_start) / max(ITERATIONS - warmdown_start, 1)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * frac))
        else:
            lr_scale = 1.0
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_scale

        x, y = next_batch(train_stream, BATCH_SIZE, BYTE_SEQ_LEN)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            loss = compiled(x, y)
        (loss / GRAD_ACCUM).backward()
        if step % GRAD_ACCUM == 0:
            opt.step(); opt.zero_grad(set_to_none=True)
        total_tokens += BATCH_SIZE * BYTE_SEQ_LEN

        torch.cuda.synchronize()
        t_now = time.perf_counter() - t_start

        if step <= 5 or step % TRAIN_LOG_EVERY == 0 or step == ITERATIONS:
            toks_per_s = total_tokens / max(t_now, 1e-9)
            log(f"{step}/{ITERATIONS} train_loss: {loss.item():.4f} "
                f"train_time: {t_now/60:.1f}m tok/s: {int(toks_per_s)}")

        if MAX_WALLCLOCK > 0 and t_now >= MAX_WALLCLOCK:
            log(f"[hnet_m1] hit MAX_WALLCLOCK_SECONDS={MAX_WALLCLOCK} at step {step}")
            break

    # final validation: per-byte CE across ~1M tokens
    log("[hnet_m1] running final val")
    model.eval()
    val_loss_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    val_tok_sum  = torch.tensor(0.0, device=device, dtype=torch.float64)
    val_batches  = 16
    with torch.inference_mode():
        for _ in range(val_batches):
            x, y = next_batch(val_stream, BATCH_SIZE, BYTE_SEQ_LEN)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                logits = model.forward_logits(x)
            n = x.numel()
            ce = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1), reduction="sum")
            val_loss_sum += ce.double()
            val_tok_sum  += float(n)
    val_nll = (val_loss_sum / val_tok_sum).item()
    val_bpb = val_nll / math.log(2.0)   # per-byte CE is already per-byte; ln -> bits
    log(f"final val_nll: {val_nll:.4f} val_bpb: {val_bpb:.4f}")
    log(f"[hnet_m1] total training tokens: {total_tokens:,}")
    log(f"[hnet_m1] wallclock: {(time.perf_counter()-t_start)/60:.2f}m")


if __name__ == "__main__":
    main()
