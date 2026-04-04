
# Clone repo
!git clone https://github.com/openai/parameter-golf.git
%cd parameter-golf

# Install deps
!pip install huggingface-hub datasets tqdm sentencepiece torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Use fewer shards if slow (e.g., 5)
!python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, time, random
import zstandard as zstd   # pip install zstandard if needed

# ---------------- CONFIG (16MB ready) ----------------
VOCAB_SIZE = 1024
BLOCK_SIZE = 256
N_EMBED = 384
N_LAYER = 8
N_HEAD = 8
QDMN_LOOPS = 3
STEPS = 12000
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = "/content/parameter-golf/data/datasets/fineweb10B_sp1024/"

print(f"--- QDMN Creative Submission (based on your paper) on {DEVICE} ---")

# ---------------- RoPE ----------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("sin", freqs.sin())
        self.register_buffer("cos", freqs.cos())

    def forward(self, x):
        B, T, H, D = x.shape
        sin = self.sin[:T].unsqueeze(0).unsqueeze(2)
        cos = self.cos[:T].unsqueeze(0).unsqueeze(2)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

# ---------------- ATTENTION ----------------
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

# ---------------- BLOCK ----------------
class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# ---------------- QDMN REFINER (from your paper + fixed residual) ----------------
class QDMNRefiner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.U = nn.Sequential(nn.Linear(dim, dim//2, bias=False), nn.GELU(), nn.Linear(dim//2, dim, bias=False))
        self.D = nn.Sequential(nn.Linear(dim, 2*dim, bias=False), nn.GELU(), nn.Linear(2*dim, dim, bias=False))
        self.B = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.GELU(), nn.Linear(dim, dim, bias=False))
        self.zap = nn.Parameter(torch.zeros(1, 1, dim))
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        h = self.U(self.ln(x))
        h = h + self.D(self.ln(h))
        h = h + self.B(self.ln(h))
        gate = torch.sigmoid(self.zap)
        return residual + (h * gate)

# ---------------- MODEL ----------------
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEAD) for _ in range(N_LAYER)])
        self.qdmn = nn.ModuleList([QDMNRefiner(N_EMBED) for _ in range(QDMN_LOOPS)])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.head = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        x = self.blocks(x)
        for refiner in self.qdmn:
            x = refiner(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        return logits, loss

# ---------------- INIT ----------------
model = GPT().to(DEVICE)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')

# ---------------- DATA & TRAINING ----------------
all_shards = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith("fineweb_train")])

def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+BLOCK_SIZE+1].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

print("Training QDMN Creative Submission...")
start = time.time()
shard_data = np.fromfile(all_shards[0], dtype=np.uint16)

for step in range(STEPS + 1):
    if step % 300 == 0 and step > 0:
        shard = random.choice(all_shards)
        shard_data = np.fromfile(shard, dtype=np.uint16)

    x, y = get_batch(shard_data)

    with torch.amp.autocast(DEVICE.type):
        _, loss = model(x, y)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % 200 == 0:
        print(f"Step {step:5d} | Loss {loss.item():.4f} | Time {(time.time()-start)/60:.1f} min")

print("Training done!")

# ---------------- VALIDATION ----------------
VAL_PATH = os.path.join(DATA_DIR, "fineweb_val_000000.bin")
val_data = np.fromfile(VAL_PATH, dtype=np.uint16)
model.eval()
total_loss = 0.0
count = 0
with torch.no_grad():
    for i in range(200):
        start_idx = random.randint(0, len(val_data) - BLOCK_SIZE - 1)
        x = torch.tensor(val_data[start_idx:start_idx+BLOCK_SIZE].astype(np.int64)).unsqueeze(0).to(DEVICE)
        y = torch.tensor(val_data[start_idx+1:start_idx+BLOCK_SIZE+1].astype(np.int64)).unsqueeze(0).to(DEVICE)
        with torch.amp.autocast(DEVICE.type):
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1

bpb = total_loss / (count * np.log(2))
print("
======================")
print(f"VALID BPB: {bpb:.4f}")
print("======================")

# ---------------- INT6 QUANTIZATION + ZSTD COMPRESSION ----------------
print("
Applying int6 quantization + zstd compression for submission...")

@torch.no_grad()
def quantize_to_int6(model):
    for param in model.parameters():
        if param.requires_grad:
            scale = param.abs().max() / 31.0
            param.data = torch.round(param.data / scale).clamp(-32, 31) * scale
    return model

model = quantize_to_int6(model)

# Save quantized state dict
torch.save(model.state_dict(), "qdmn_submission_int6.pth")

# Compress with high zstd level
with open("qdmn_submission_int6.pth", "rb") as f:
    data = f.read()

cctx = zstd.ZstdCompressor(level=22)
compressed = cctx.compress(data)

with open("qdmn_final_artifact.bin", "wb") as f:
    f.write(compressed)

print(f"Final compressed artifact size: {len(compressed) / (1024*1024):.3f} MB")
print("Submission file saved as 'qdmn_final_artifact.bin'")

# ---------------- DECOMPRESSION + LOAD FUNCTION (required for submission) ----------------
def load_qdmn_from_artifact(artifact_path, model):
    dctx = zstd.ZstdDecompressor()
    with open(artifact_path, "rb") as f:
        decompressed = dctx.decompress(f.read())
    with open("temp.pth", "wb") as f:
        f.write(decompressed)
    state_dict = torch.load("temp.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    return model

print("
You can load the artifact with: load_qdmn_from_artifact('qdmn_final_artifact.bin', model)")
print("Ready for submission!")
