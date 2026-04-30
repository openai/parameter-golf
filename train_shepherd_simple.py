#!/usr/bin/env python3
"""Minimal Shepherd training — no competition pipeline, just train and save."""
import sys, os, time, glob, math
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(__file__))
from shepherd_embryo_v0_0002 import ShepherdEmbryo

# Config
DEVICE = "cuda"
DTYPE = torch.bfloat16
VOCAB_SIZE = 1024
SEQ_LEN = 1024
BATCH_SIZE = 8
LR = 3e-4
ITERATIONS = 2000
VAL_EVERY = 500
LOG_EVERY = 50
DATA_PATH = "./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"

def load_shard(path):
    return np.memmap(path, dtype=np.uint16, mode='r')

def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+seq_len+1].astype(np.int64)) for i in ix]).to(device)
    return x, y

@torch.no_grad()
def validate(model, val_data, tokenizer):
    model.eval()
    losses = []
    for _ in range(20):
        x, y = get_batch(val_data, 8, SEQ_LEN, DEVICE)
        loss = model(x, y)
        losses.append(loss.item())
    model.train()
    avg_loss = sum(losses) / len(losses)
    bytes_per_token = sum(len(tokenizer.Encode(tokenizer.IdToPiece(i), out_type=str)) 
                          for i in range(tokenizer.GetPieceSize())) / tokenizer.GetPieceSize()
    # Approximate bpb
    bpb = (avg_loss / math.log(2)) / max(bytes_per_token, 1.0)
    return avg_loss, bpb

def main():
    print("=" * 60)
    print("  SHEPHERD EMBRYO — SIMPLE TRAINING")
    print("=" * 60)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)

    # Load data
    train_files = sorted(glob.glob(os.path.join(DATA_PATH, "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(os.path.join(DATA_PATH, "fineweb_val_*.bin")))
    print(f"  Train shards: {len(train_files)}")
    print(f"  Val shards: {len(val_files)}")
    train_data = load_shard(train_files[0])
    val_data = load_shard(val_files[0])

    # Build model
    model = ShepherdEmbryo(
        vocab_size=VOCAB_SIZE,
        model_dim=384,
        seed_rank=48,
        num_probes=5,
        num_probe_depths=3,
        fold_top_k=2,
        num_core_layers=3,
        num_heads=6,
        num_kv_heads=3,
        mlp_mult=2,
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  LR: {LR}")
    print(f"  Batch: {BATCH_SIZE} x {SEQ_LEN}")
    print("-" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ITERATIONS, eta_min=LR/10)

    model.train()
    t0 = time.time()

    for step in range(1, ITERATIONS + 1):
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)

        with torch.autocast(device_type='cuda', dtype=DTYPE):
            loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0 or step <= 10:
            elapsed = time.time() - t0
            print(f"  step:{step}/{ITERATIONS} loss:{loss.item():.4f} time:{elapsed:.0f}s step_avg:{elapsed/step*1000:.0f}ms")

        if step % VAL_EVERY == 0:
            val_loss, val_bpb = validate(model, val_data, sp)
            print(f"  step:{step}/{ITERATIONS} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # Save
    torch.save(model.state_dict(), "shepherd_model.pt")
    print(f"\n  Saved: shepherd_model.pt")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
