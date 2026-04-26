"""
H-Net training script. Single GPU (or CPU). Plain AdamW, no Muon, no compile.

Run:
    python train_hnet.py

Env vars (same convention as train_gpt.py):
    DATA_PATH            ./data/datasets/fineweb10B_byte260
    RUN_ID               smoke
    ITERATIONS           500
    TRAIN_BATCH_TOKENS   8192        bytes per optimizer step
    TRAIN_SEQ_LEN        1024
    VAL_BATCH_SIZE       8192
    VAL_LOSS_EVERY       0           0 = only at end
    VAL_TOKENS_CAP       65536       cap val tokens for fast iteration; 0 = full split
    LR                   3e-4
    WARMUP_STEPS         100
    WEIGHT_DECAY         0.1
    RATIO_LOSS_ALPHA     0.03
    TARGET_RATIO         0.25
    D_ENC, D_MAIN        128, 256
    N_ENC, N_MAIN, N_DEC 3, 6, 3
    N_HEADS              4
    SEED                 1337
"""
from __future__ import annotations

import glob
import math
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp, enable_cudnn_sdp

from hnet_model import HNet


# -----------------------------
# DATA LOADING (byte260 .bin shards)
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"no shards found for pattern: {pattern}")
        self.idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self) -> None:
        self.idx = (self.idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        rem = n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def next_train_batch(
    stream: TokenStream, batch_tokens: int, seq_len: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    raw = stream.take(batch_tokens + 1).to(dtype=torch.int64)
    x = raw[:-1].reshape(-1, seq_len)
    y = raw[1:].reshape(-1, seq_len)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def load_val_tokens(pattern: str, seq_len: int, cap: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"no val shards: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files])
    if cap > 0:
        tokens = tokens[: cap + 1]
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError("val split too short for seq_len")
    return tokens[: usable + 1]


# -----------------------------
# VAL_BPB FOR BYTE260
# -----------------------------
# byte260 vocab (PureByteTokenizer, data/download_hf_docs_and_tokenize.py):
#   0=PAD, 1=BOS, 2=EOS, 3=UNK (specials, 0 byte cost)
#   4..259 = the 256 raw byte values (1 byte each)

_BYTES_PER_TOKEN_LUT: Tensor | None = None

def bytes_per_token_lut(device: torch.device) -> Tensor:
    global _BYTES_PER_TOKEN_LUT
    if _BYTES_PER_TOKEN_LUT is None or _BYTES_PER_TOKEN_LUT.device != device:
        lut = torch.zeros(260, dtype=torch.int16, device=device)
        lut[4:] = 1
        _BYTES_PER_TOKEN_LUT = lut
    return _BYTES_PER_TOKEN_LUT


@torch.inference_mode()
def eval_val_bpb(
    model: HNet,
    val_tokens: Tensor,
    seq_len: int,
    batch_seqs: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_seqs = (val_tokens.numel() - 1) // seq_len
    bpt_lut = bytes_per_token_lut(device)
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    for s in range(0, total_seqs, batch_seqs):
        end = min(s + batch_seqs, total_seqs)
        raw_start = s * seq_len
        raw_end = end * seq_len + 1
        chunk = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ar_loss, _ = model(x, y)
        n = float(y.numel())
        nll_sum += ar_loss.to(torch.float64) * n
        tok_count += n
        byte_count += bpt_lut[y.reshape(-1)].to(torch.float64).sum()

    val_loss = (nll_sum / tok_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = (tok_count / byte_count).item()
    model.train()
    return val_loss, bits_per_token * tokens_per_byte


# -----------------------------
# MAIN
# -----------------------------

def main() -> None:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_byte260")
    train_pattern = os.path.join(data_path, "fineweb_train_*.bin")
    val_pattern = os.path.join(data_path, "fineweb_val_*.bin")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4())[:8])
    seed = int(os.environ.get("SEED", 1337))

    iterations = int(os.environ.get("ITERATIONS", 500))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 8192))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 8192))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_tokens_cap = int(os.environ.get("VAL_TOKENS_CAP", 65536))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))

    lr = float(os.environ.get("LR", 3e-4))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 100))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.1))
    ratio_loss_alpha = float(os.environ.get("RATIO_LOSS_ALPHA", 0.03))
    target_ratio = float(os.environ.get("TARGET_RATIO", 1.0 / 6.0))

    d_enc = int(os.environ.get("D_ENC", 128))
    d_main = int(os.environ.get("D_MAIN", 256))
    n_enc = int(os.environ.get("N_ENC", 3))
    n_main = int(os.environ.get("N_MAIN", 6))
    n_dec = int(os.environ.get("N_DEC", 3))
    n_heads = int(os.environ.get("N_HEADS", 4))

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        # Windows wheels lack flash; fall back to math/cudnn so MHA has a kernel.
        enable_flash_sdp(False)
        enable_mem_efficient_sdp(False)
        enable_cudnn_sdp(True)
        enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"

    def log(msg: str) -> None:
        print(msg)
        with open(logfile, "a", encoding="utf-8") as f:
            print(msg, file=f)

    log(f"run_id:{run_id} device:{device}")
    log(f"data_path:{data_path}")

    train_stream = TokenStream(train_pattern)
    val_tokens = load_val_tokens(val_pattern, train_seq_len, val_tokens_cap)
    log(f"val_tokens:{val_tokens.numel()}")

    model = HNet(
        vocab_size=260, d_enc=d_enc, d_main=d_main,
        n_enc=n_enc, n_main=n_main, n_dec=n_dec, n_heads=n_heads,
        target_ratio=target_ratio,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model_params:{n_params:,}")
    log(f"shape: d_enc={d_enc} d_main={d_main} n_enc={n_enc} n_main={n_main} n_dec={n_dec} heads={n_heads}")
    log(f"target_ratio:{target_ratio} ratio_alpha:{ratio_loss_alpha}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95),
        weight_decay=weight_decay, eps=1e-8,
    )

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, iterations - warmup_steps)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    val_seqs = max(1, val_batch_size // train_seq_len)
    t0 = time.time()
    for step in range(iterations):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        x, y = next_train_batch(train_stream, train_batch_tokens, train_seq_len, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ar_loss, ratio_loss = model(x, y)
        loss = ar_loss + ratio_loss_alpha * ratio_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % train_log_every == 0 or step == 0:
            elapsed = time.time() - t0
            log(
                f"step:{step+1}/{iterations} ar:{ar_loss.item():.4f} "
                f"ratio:{ratio_loss.item():.4f} lr:{lr_at(step):.2e} "
                f"elapsed:{elapsed:.1f}s"
            )

        if val_loss_every > 0 and (step + 1) % val_loss_every == 0:
            vl, vb = eval_val_bpb(model, val_tokens, train_seq_len, val_seqs, device)
            log(f"step:{step+1} val_loss:{vl:.4f} val_bpb:{vb:.4f}")

    vl, vb = eval_val_bpb(model, val_tokens, train_seq_len, val_seqs, device)
    log(f"final step:{iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f}")

    ckpt_path = f"logs/{run_id}_final.pt"
    torch.save({"state_dict": model.state_dict(), "val_bpb": vb, "params": n_params}, ckpt_path)
    log(f"saved:{ckpt_path}")


if __name__ == "__main__":
    main()
