import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob, os, math


import os

data_path = "./data/datasets/fineweb10B_sp1024/"
print("Files in dataset folder:", os.listdir(data_path))

import numpy as np
import glob
import os
import torch
# ======================
# LOAD TOKENIZED DATA (Parameter Golf correct)
# ======================
data_dir = "./data/datasets/fineweb10B_sp1024/"

if not os.path.exists(data_dir):
    print("Downloading dataset...")
    os.system("python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1")

train_files = sorted(glob.glob(os.path.join(data_dir, "*")))

if len(train_files) == 0:
    raise ValueError("No training files found")

print("[DATA FILE]", train_files[0])

data = np.memmap(train_files[0], dtype=np.uint16, mode='r')

# convert to torch
data = torch.from_numpy(data.astype(np.int64))

# FIXED vocab (VERY IMPORTANT)
vocab_size = 1024

# split
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import math

# ======================
# CONFIG (OBLIGATOIRE)
# ======================
batch_size = 64        # + rapide (GPU parallèle)
block_size = 128       # ↓ 2x plus rapide

n_embd = 128           # ↓ énorme gain
n_head = 4             # cohérent avec embd
n_layer = 3            # ↓ moitié du modèle

max_iters = 800
eval_interval = 100
eval_iters = 10

dropout = 0.1

lr_max = 1e-4
warmup_steps = 500
weight_decay = 0.1
grad_clip = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
# ======================
# MODEL
# ======================
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        return logits, loss

# ======================
# CREATE MODEL (IMPORTANT si pas déjà fait)
# ======================
model = GPT().to(device)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ======================
# OPTIMIZER
# ======================
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)

# ======================
# LR SCHEDULE (warmup + cosine)
# ======================
def get_lr(step):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / (max_iters - warmup_steps)
    return 0.5 * lr_max * (1 + math.cos(math.pi * progress))

# ======================
# LOSS EVAL
# ======================
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
# ======================
# BATCH FUNCTION (OBLIGATOIRE)
# ======================
def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
# ======================
# TRAIN LOOP
# ======================
loss_history = []

for step in range(max_iters):

    # update lr
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # batch
    xb, yb = get_batch("train")

    # forward
    logits, loss = model(xb, yb)

    # backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    loss_history.append(loss.item())

    # logging (fixed)
    if step % 10 == 0:
        recent = loss_history[-10:]
        avg_loss = sum(recent) / len(recent)
        print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")


# ======================
# TEXT GENERATION
# ======================
@torch.no_grad()
def generate(idx, max_new_tokens=200):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx

# sample
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(generate(context)[0].tolist())

torch.save(model.state_dict(), "model.pt")
