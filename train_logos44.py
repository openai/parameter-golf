"""
LOGOS-44 v4 — Standalone CUDA trainer for RunPod.
Uses the official Parameter Golf data pipeline (SentencePiece + FineWeb shards).

Usage (on RunPod, single GPU):
    cd /workspace/parameter-golf
    python train_logos44.py

Usage (multi-GPU):
    torchrun --nproc_per_node=8 train_logos44.py
"""
import copy
import glob
import io
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from logos44_tiny import Logos44Tiny

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════
DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")

VOCAB_SIZE    = int(os.environ.get("VOCAB_SIZE", 1024))   # must match tokenizer
DIM           = int(os.environ.get("DIM", 384))            # 384 = safe for 16MB
N_LAYERS      = int(os.environ.get("N_LAYERS", 14))        # more depth with smaller dim
N_HEADS       = int(os.environ.get("N_HEADS", 8))
N_KV_HEADS    = int(os.environ.get("N_KV_HEADS", 4))
DROPOUT       = float(os.environ.get("DROPOUT", 0.05))
MAX_LAYER_DROP = float(os.environ.get("MAX_LAYER_DROP", 0.1))

SEQ_LEN       = int(os.environ.get("SEQ_LEN", 1024))
BATCH_TOKENS  = int(os.environ.get("BATCH_TOKENS", 524288))
MAX_STEPS     = int(os.environ.get("MAX_STEPS", 20000))
MAX_SECONDS   = float(os.environ.get("MAX_SECONDS", 600.0))
PEAK_LR       = float(os.environ.get("PEAK_LR", 1e-3))
MIN_LR        = float(os.environ.get("MIN_LR", 1e-5))
WARMUP        = int(os.environ.get("WARMUP", 200))
WEIGHT_DECAY  = float(os.environ.get("WEIGHT_DECAY", 0.1))
GRAD_CLIP     = float(os.environ.get("GRAD_CLIP", 1.0))
EMA_DECAY     = float(os.environ.get("EMA_DECAY", 0.999))
SEED          = int(os.environ.get("SEED", 1337))

VAL_EVERY     = int(os.environ.get("VAL_EVERY", 500))
LOG_EVERY     = int(os.environ.get("LOG_EVERY", 50))

# ═══════════════════════════════════════════
# DATA LOADING (from official repo format)
# ═══════════════════════════════════════════
def load_data_shard(filepath: Path) -> Tensor:
    header = np.fromfile(filepath, dtype=np.int32, count=256)
    assert header[0] == 20240520, f"Bad magic in {filepath}"
    ntok = header[1]
    tokens = np.fromfile(filepath, dtype=np.uint16, offset=256 * 4, count=ntok)
    return torch.from_numpy(tokens.astype(np.int32))

class ShardedDataLoader:
    def __init__(self, pattern, seq_len, batch_tokens, rank=0, world_size=1):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"No files for {pattern}"
        self.seq_len = seq_len
        self.local_batch = batch_tokens // world_size
        self.batch_seqs = self.local_batch // seq_len
        self.rank = rank
        self.world_size = world_size
        self.shard_idx = 0
        self.pos = 0
        self._load_shard()

    def _load_shard(self):
        self.tokens = load_data_shard(Path(self.files[self.shard_idx]))
        self.pos = self.rank * self.batch_seqs * self.seq_len

    def next_batch(self, device):
        B, T = self.batch_seqs, self.seq_len
        end = self.pos + B * T + 1
        if end > len(self.tokens):
            self.shard_idx = (self.shard_idx + 1) % len(self.files)
            self._load_shard()
        buf = self.tokens[self.pos : self.pos + B * T + 1].to(device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += B * T * self.world_size
        return x, y

# ═══════════════════════════════════════════
# BPB EVALUATION (tokenizer-agnostic, from official repo)
# ═══════════════════════════════════════════
def build_byte_lut(tokenizer_path, vocab_size, device):
    """Build bytes-per-token lookup table from SentencePiece model."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    sp_vs = sp.vocab_size()
    table_size = max(sp_vs, vocab_size)
    base_bytes = np.zeros(table_size, dtype=np.float32)
    has_space = np.zeros(table_size, dtype=np.bool_)
    is_boundary = np.ones(table_size, dtype=np.bool_)

    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))

    return (
        torch.tensor(base_bytes, device=device),
        torch.tensor(has_space, device=device),
        torch.tensor(is_boundary, device=device),
    )

@torch.no_grad()
def eval_bpb(model, val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
             seq_len, device, batch_seqs=8):
    model.eval()
    total_seqs = (val_tokens.numel() - 1) // seq_len
    total_nll = 0.0
    total_bytes = 0.0

    for start in range(0, total_seqs, batch_seqs):
        end = min(start + batch_seqs, total_seqs)
        B = end - start
        idxs = torch.arange(start, end)
        x = torch.stack([val_tokens[i * seq_len : i * seq_len + seq_len] for i in idxs]).to(device)
        y = torch.stack([val_tokens[i * seq_len + 1 : i * seq_len + seq_len + 1] for i in idxs]).to(device)

        logits = model(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='none')
        nll = nll.view(B, seq_len)

        token_bytes = base_bytes_lut[y] + has_space_lut[y].float()
        token_bytes = token_bytes * (~is_boundary_lut[y]).float()

        total_nll += nll.sum().item()
        total_bytes += token_bytes.sum().item()

    model.train()
    if total_bytes == 0:
        return float('inf'), float('inf')
    val_loss = total_nll / (total_seqs * seq_len)
    val_bpb = total_nll / total_bytes / math.log(2)
    return val_loss, val_bpb

# ═══════════════════════════════════════════
# LR SCHEDULE
# ═══════════════════════════════════════════
def get_lr(step):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(MAX_STEPS - WARMUP, 1)
    return MIN_LR + (PEAK_LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * progress))

# ═══════════════════════════════════════════
# EXPORT (int8 + zlib for 16MB check)
# ═══════════════════════════════════════════
def export_model(model, path="model_export.bin"):
    buf = io.BytesIO()
    sd = {k: v.to(torch.int8) if v.dtype in (torch.float32, torch.float16, torch.bfloat16)
          else v for k, v in model.state_dict().items()}
    torch.save(sd, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, 9)
    with open(path, "wb") as f:
        f.write(compressed)
    raw_mb = len(raw) / 1024**2
    comp_mb = len(compressed) / 1024**2
    return raw_mb, comp_mb

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED + rank)

    if rank == 0:
        print(f"[LOGOS-44 v4] Device: {device} | World: {world_size}")
        print(f"Vocab: {VOCAB_SIZE} | Dim: {DIM} | Layers: {N_LAYERS} | "
              f"Heads: {N_HEADS}/{N_KV_HEADS}")
        print(f"Dropout: {DROPOUT} | LayerDrop: {MAX_LAYER_DROP} | EMA: {EMA_DECAY}")
        print(f"Batch tokens: {BATCH_TOKENS} | Seq: {SEQ_LEN} | Max steps: {MAX_STEPS}")
        print(f"Max wallclock: {MAX_SECONDS}s\n")

    model = Logos44Tiny(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, N_KV_HEADS,
                        DROPOUT, MAX_LAYER_DROP).to(device)
    ema_model = copy.deepcopy(model)

    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Parametry: {num_params:,}")
        raw_mb, comp_mb = export_model(model, "/tmp/test_export.bin")
        print(f"Artefakt: {raw_mb:.1f}MB raw -> {comp_mb:.1f}MB compressed "
              f"({'OK' if comp_mb < 16 else 'ZA DUZY'})\n")

    if ddp:
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])

    raw_model = model.module if ddp else model

    decay_params = [p for n, p in raw_model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in raw_model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=PEAK_LR, betas=(0.9, 0.95), fused=True)

    train_pattern = os.path.join(DATA_PATH, "fineweb_train_*.bin")
    val_pattern = os.path.join(DATA_PATH, "fineweb_val_*.bin")
    train_loader = ShardedDataLoader(train_pattern, SEQ_LEN, BATCH_TOKENS, rank, world_size)

    val_files = sorted(glob.glob(val_pattern))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files])
    usable = ((val_tokens.numel() - 1) // SEQ_LEN) * SEQ_LEN
    val_tokens = val_tokens[:usable + 1]

    base_bytes_lut, has_space_lut, is_boundary_lut = build_byte_lut(
        TOKENIZER_PATH, VOCAB_SIZE, device)

    if rank == 0:
        print(f"Val tokens: {val_tokens.numel():,}\n")

    model.train()
    best_bpb = float('inf')
    t0 = time.time()

    for step in range(MAX_STEPS):
        elapsed = time.time() - t0
        if elapsed > MAX_SECONDS:
            if rank == 0:
                print(f"\nWallclock limit ({MAX_SECONDS}s) reached at step {step}")
            break

        lr = get_lr(step)
        for g in optimizer.param_groups:
            g['lr'] = lr

        x, y = train_loader.next_batch(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            for ep, p in zip(ema_model.parameters(), raw_model.parameters()):
                ep.lerp_(p, 1 - EMA_DECAY)

        if rank == 0 and (step % LOG_EVERY == 0 or step == MAX_STEPS - 1):
            print(f"Krok {step:05d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | {elapsed:.0f}s")

        if rank == 0 and (step % VAL_EVERY == 0 or step == MAX_STEPS - 1) and step > 0:
            val_loss, val_bpb = eval_bpb(
                ema_model, val_tokens, base_bytes_lut, has_space_lut,
                is_boundary_lut, SEQ_LEN, device)
            marker = ""
            if val_bpb < best_bpb:
                best_bpb = val_bpb
                torch.save(ema_model.state_dict(), "logos44_best.pt")
                marker = " * BEST"
            print(f"  -> Val Loss: {val_loss:.4f} | Val BPB: {val_bpb:.4f}{marker}")

    if rank == 0:
        val_loss, val_bpb = eval_bpb(
            ema_model, val_tokens, base_bytes_lut, has_space_lut,
            is_boundary_lut, SEQ_LEN, device)
        print(f"\n{'='*50}")
        print(f"  Final Val BPB: {val_bpb:.4f}")
        print(f"  Best Val BPB:  {best_bpb:.4f}")
        raw_mb, comp_mb = export_model(ema_model, "logos44_export.bin")
        print(f"  Artefakt: {comp_mb:.1f}MB / 16.0MB limit")
        print(f"  Parametry: {num_params:,}")
        print(f"  Czas: {time.time() - t0:.0f}s")
        print(f"{'='*50}")

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
