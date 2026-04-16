import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import zlib
import pickle

# ======================
# CONFIG
# ======================
VOCAB_SIZE = 1024
BLOCK_SIZE = 128
EMBED_DIM = 192
N_HEAD = 6
N_LAYERS = 4
BATCH_SIZE = 32
LR = 3e-4
MAX_ITERS = 2000

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# MODEL
# ======================
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embed = nn.Parameter(torch.zeros(1, BLOCK_SIZE, EMBED_DIM))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=N_HEAD,
                batch_first=True
            )
            for _ in range(N_LAYERS)
        ])

        self.ln = nn.LayerNorm(EMBED_DIM)
        self.fc = nn.Linear(EMBED_DIM, VOCAB_SIZE)

        # 🔥 weight tying (VERY IMPORTANT)
        self.fc.weight = self.embed.weight

    def forward(self, x):
        B, T = x.shape

        x = self.embed(x) + self.pos_embed[:, :T, :]

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, src_mask=mask)

        x = self.ln(x)
        return self.fc(x)

# ======================
# DUMMY DATA (REPLACE LATER WITH REAL LOADER)
# ======================
data = torch.randint(0, VOCAB_SIZE, (10000,))
X = []
Y = []

for i in range(len(data) - BLOCK_SIZE):
    X.append(data[i:i+BLOCK_SIZE])
    Y.append(data[i+1:i+BLOCK_SIZE+1])

X = torch.stack(X)
Y = torch.stack(Y)

# ======================
# TRAINING
# ======================
model = TinyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

start_time = time.time()

for step in range(MAX_ITERS):
    idx = torch.randint(0, len(X), (BATCH_SIZE,))
    xb = X[idx].to(device)
    yb = Y[idx].to(device)

    logits = model(xb)
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step}, loss {loss.item()}")

# ======================
# FAKE VALIDATION (replace later)
# ======================
val_loss = loss.item()
val_bpb = val_loss / 0.693  # approx conversion

print("val_loss:", val_loss)
print("val_bpb:", val_bpb)

# ======================
# COMPRESSION
# ======================
compressed = zlib.compress(pickle.dumps(model.state_dict()))
print("compressed_size_bytes:", len(compressed))

torch.save(model.state_dict(), "model.pt")