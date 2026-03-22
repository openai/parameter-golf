"""
Minimal TTT debug: does SmearGate break TTT LoRA adaptation?

Test plan:
1. Create a tiny model WITH SmearGate, train briefly
2. Run TTT-style LoRA adaptation on a few chunks
3. Check if per-token loss improves (TTT working) or degrades (TTT broken)
4. Repeat WITHOUT SmearGate
5. Compare

This runs on CPU, no GPU needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        g = torch.sigmoid(self.gate)[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class TinyModel(nn.Module):
    def __init__(self, vocab=64, dim=32, use_smear=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.smear = SmearGate(dim) if use_smear else nn.Identity()
        self.linear1 = nn.Linear(dim, dim*2, bias=False)
        self.linear2 = nn.Linear(dim*2, dim, bias=False)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.dim = dim
        self.vocab = vocab

    def forward(self, x, targets, lora_head=None):
        h = self.emb(x)
        h = F.rms_norm(h, (self.dim,))
        h = self.smear(h)
        h = self.linear2(F.relu(self.linear1(h)).square())
        logits = self.head(h)
        if lora_head is not None:
            logits = logits + lora_head(h)
        # Per-token loss
        B, S, V = logits.shape
        return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction='none').reshape(B, S)

class BatchedLoRA(nn.Module):
    def __init__(self, bsz, in_f, out_f, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(bsz, rank, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(bsz, out_f, rank))
    def forward(self, x):
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)
    def reset(self):
        with torch.no_grad():
            self.A.normal_(0, 0.01)
            self.B.zero_()

def test_ttt(use_smear, seed=42):
    torch.manual_seed(seed)
    V, D = 64, 32
    model = TinyModel(V, D, use_smear=use_smear)

    # "Train" briefly
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(200):
        x = torch.randint(0, V, (4, 64))
        y = torch.randint(0, V, (4, 64))
        loss = model(x, y).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    train_loss = model(x, y).mean().item()

    # Now do TTT-style eval
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Create a "document" and process in chunks
    doc = torch.randint(0, V, (1, 256))
    chunk_size = 32

    # Score WITHOUT TTT
    with torch.no_grad():
        ptl_no_ttt = model(doc[:, :-1], doc[:, 1:])
    no_ttt_loss = ptl_no_ttt.mean().item()

    # Score WITH TTT (LoRA on head, adapted per-chunk)
    lora = BatchedLoRA(1, D, V, rank=4)
    ttt_opt = torch.optim.Adam(lora.parameters(), lr=0.01)

    ttt_losses = []
    for ci in range(0, 255, chunk_size):
        end = min(ci + chunk_size, 255)
        x_chunk = doc[:, ci:end]
        y_chunk = doc[:, ci+1:end+1]

        # Forward + score
        ptl = model(x_chunk, y_chunk, lora_head=lora)
        chunk_loss = ptl.mean().item()
        ttt_losses.append(chunk_loss)

        # Train LoRA on this chunk (except last)
        if end < 255:
            ttt_opt.zero_grad()
            ptl.mean().backward()
            ttt_opt.step()

    ttt_loss = sum(ttt_losses) / len(ttt_losses)

    smear_label = "WITH SmearGate" if use_smear else "NO SmearGate  "
    delta = ttt_loss - no_ttt_loss
    direction = "IMPROVED" if delta < 0 else "DEGRADED"
    print(f"{smear_label}: train={train_loss:.4f} no_ttt={no_ttt_loss:.4f} ttt={ttt_loss:.4f} delta={delta:+.4f} ({direction})")
    return delta

print("=== TTT SmearGate Debug ===")
print()
deltas_smear = [test_ttt(use_smear=True, seed=s) for s in range(5)]
deltas_nosmear = [test_ttt(use_smear=False, seed=s) for s in range(5)]
print()
print(f"SmearGate avg delta: {sum(deltas_smear)/len(deltas_smear):+.4f}")
print(f"No SmearGate avg delta: {sum(deltas_nosmear)/len(deltas_nosmear):+.4f}")
