"""
CPU Smoke Test — train the baseline model for a few steps and measure val_bpb.
This gives us REAL numbers to compare against, even without GPU.

Usage: python smoke_test.py [script_name] [num_steps]
  e.g.: python smoke_test.py train_gpt.py 20
"""
import sys
import os
import time
import math
import glob
import numpy as np
from pathlib import Path

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F
import sentencepiece as spm

# ---- Config ----
script_name = sys.argv[1] if len(sys.argv) > 1 else "train_gpt.py"
num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
seq_len = 256  # shorter for CPU speed
batch_tokens = 2048  # tiny batch for CPU

print(f"=== SMOKE TEST: {script_name}, {num_steps} steps, seq_len={seq_len} ===")

# ---- Data loading ----
data_path = "./data/datasets/fineweb10B_sp1024"
tokenizer_path = "./data/tokenizers/fineweb_1024_bpe.model"
vocab_size = 1024

def load_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256*4)
    return torch.from_numpy(tokens.astype(np.uint16))

train_files = sorted(glob.glob(os.path.join(data_path, "fineweb_train_*.bin")))
val_files = sorted(glob.glob(os.path.join(data_path, "fineweb_val_*.bin")))

if not train_files:
    print(f"ERROR: No training data found at {data_path}")
    print("Run: python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1")
    sys.exit(1)

print(f"Train shards: {len(train_files)}, Val shards: {len(val_files)}")

# Load a small chunk of training data
train_tokens = load_shard(Path(train_files[0]))
print(f"Train tokens loaded: {train_tokens.numel():,}")

# Load validation data
val_tokens_list = [load_shard(Path(f)) for f in val_files]
val_tokens = torch.cat(val_tokens_list) if val_files else train_tokens[:10000]
print(f"Val tokens loaded: {val_tokens.numel():,}")

# ---- Load tokenizer for BPB calculation ----
sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

def build_bpb_luts(sp, vocab_size):
    sp_vocab = int(sp.vocab_size())
    table_size = max(sp_vocab, vocab_size)
    base_bytes = np.zeros(table_size, dtype=np.int16)
    has_leading_space = np.zeros(table_size, dtype=np.bool_)
    is_boundary = np.ones(table_size, dtype=np.bool_)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16),
        torch.tensor(has_leading_space, dtype=torch.bool),
        torch.tensor(is_boundary, dtype=torch.bool),
    )

base_bytes_lut, has_leading_space_lut, is_boundary_lut = build_bpb_luts(sp, vocab_size)

# ---- Build model (import from script) ----
# We'll build a small version of the baseline model directly
from torch import nn, Tensor

class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

def apply_rotary(x, cos, sin):
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, num_kv_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.fc = nn.Linear(dim, dim * mult, bias=False)
        self.proj = nn.Linear(dim * mult, dim, bias=False)

    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = RMSNorm()
        self.norm2 = RMSNorm()
        self.attn = Attention(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SmallGPT(nn.Module):
    def __init__(self, vocab_size=1024, dim=512, n_layers=9):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(n_layers)])
        self.norm = RMSNorm()
        self.dim = dim
        nn.init.normal_(self.tok_emb.weight, std=0.005)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = F.linear(x, self.tok_emb.weight)  # tied embeddings
        if targets is not None:
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits

# ---- Create model ----
model = SmallGPT(vocab_size=vocab_size, dim=512, n_layers=9)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

# ---- Evaluate BEFORE training (random init) ----
def evaluate(model, val_tokens, seq_len, max_seqs=50):
    model.eval()
    usable = ((val_tokens.numel() - 1) // seq_len) * seq_len
    tokens = val_tokens[:usable + 1]
    total_seqs = usable // seq_len
    n_seqs = min(total_seqs, max_seqs)

    loss_sum = 0.0
    token_count = 0
    byte_count = 0

    with torch.no_grad():
        for i in range(n_seqs):
            start = i * seq_len
            chunk = tokens[start:start + seq_len + 1].long()
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            loss = model(x, y)
            loss_sum += loss.item() * seq_len
            token_count += seq_len

            prev_ids = chunk[:-1]
            tgt_ids = chunk[1:]
            tbytes = base_bytes_lut[tgt_ids].to(torch.int16)
            tbytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).to(torch.int16)
            byte_count += tbytes.sum().item()

    val_loss = loss_sum / token_count
    bpt = val_loss / math.log(2.0)
    tpb = token_count / byte_count
    val_bpb = bpt * tpb
    model.train()
    return val_loss, val_bpb

print("\n--- Pre-training evaluation ---")
t0 = time.time()
pre_loss, pre_bpb = evaluate(model, val_tokens, seq_len)
print(f"val_loss: {pre_loss:.4f}  val_bpb: {pre_bpb:.4f}  ({time.time()-t0:.1f}s)")

# ---- Train ----
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
model.train()

print(f"\n--- Training {num_steps} steps ---")
pos = 0
for step in range(1, num_steps + 1):
    # Get batch
    if pos + batch_tokens + 1 > train_tokens.numel():
        pos = 0
    chunk = train_tokens[pos:pos + batch_tokens + 1].long()
    pos += batch_tokens
    x = chunk[:-1].reshape(-1, seq_len)
    y = chunk[1:].reshape(-1, seq_len)

    optimizer.zero_grad()
    loss = model(x, y)
    loss.backward()
    optimizer.step()

    if step <= 5 or step % 5 == 0:
        print(f"step {step:3d}/{num_steps}: train_loss={loss.item():.4f}")

# ---- Evaluate AFTER training ----
print("\n--- Post-training evaluation ---")
t0 = time.time()
post_loss, post_bpb = evaluate(model, val_tokens, seq_len)
print(f"val_loss: {post_loss:.4f}  val_bpb: {post_bpb:.4f}  ({time.time()-t0:.1f}s)")

print(f"\n=== RESULTS ===")
print(f"Pre-training:  val_bpb = {pre_bpb:.4f}")
print(f"Post-training: val_bpb = {post_bpb:.4f}")
print(f"Improvement:   {pre_bpb - post_bpb:.4f} BPB")
print(f"Baseline target: 1.2244 BPB")
if post_bpb < 1.2244:
    print(f"*** BEATS BASELINE by {1.2244 - post_bpb:.4f} BPB ***")
else:
    print(f"Still {post_bpb - 1.2244:.4f} BPB above baseline (expected with {num_steps} steps on CPU)")
