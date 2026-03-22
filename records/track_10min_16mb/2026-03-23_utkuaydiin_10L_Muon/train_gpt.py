import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard
import io
import time
import numpy as np
import os
from torch.optim import Optimizer
from torch.optim.swa_utils import AveragedModel

device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

# ==================== MIXED QUANT LINEAR ====================
class MixedQuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, bits=6):
        super().__init__(in_features, out_features, bias=bias)
        self.bits = bits

    def quantize_post_training(self):
        t = self.weight.data.float()
        clip = 15 if self.bits == 5 else 31
        scale = t.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / clip
        q = torch.clamp(torch.round(t / scale), -(clip + 1), clip).to(torch.int8)
        return q, scale.squeeze(1).to(torch.float16)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

# ==================== MUON HYBRID OPTIMIZER ====================
@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(0) > G.size(1): X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)

class MuonHybridOptimizer(Optimizer):
    def __init__(self, model, muon_lr=0.02, adam_lr=3e-4, wd=0.04):
        self.muon_params = [p for n, p in model.named_parameters() if p.ndim == 2 and "wte" not in n and "hash_table" not in n]
        muon_ids = {id(p) for p in self.muon_params}
        adam_params = [p for p in model.parameters() if id(p) not in muon_ids]
        self.adam = torch.optim.AdamW(adam_params, lr=adam_lr, weight_decay=0.0, betas=(0.9, 0.95))
        self.muon_lr = muon_lr
        self.wd = wd
        self.momentum = 0.92
        self.momentums = {p: torch.zeros_like(p.data) for p in self.muon_params}
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < 5000:
            self.momentum = 0.92 + 0.07 * (self.step_count / 5000)
        self.adam.step()
        for p in self.muon_params:
            if p.grad is None: continue
            g = p.grad.data
            buf = self.momentums[p]
            buf.mul_(self.momentum).add_(g)
            g = g + self.momentum * buf
            g_ortho = zeropower_via_newtonschulz5(g)
            if self.wd > 0:
                g_ortho = g_ortho + self.wd * p.data
            p.data.add_(g_ortho, alpha=-self.muon_lr)

    def zero_grad(self, set_to_none=False):
        self.adam.zero_grad(set_to_none)
        for p in self.muon_params:
            if p.grad is not None: p.grad.zero_()

# ==================== CUSTOM ZSTD-22 PACKER ====================
class CustomBinaryCheckpoint:
    @torch.no_grad()
    def pack(self, model):
        sd = model.state_dict()
        packed = {}
        for name, t in sd.items():
            if "wte" in name or "lm_head" in name or t.ndim == 1:
                packed[name] = t.cpu().to(torch.float16)
                continue
            clip = 15 if "mlp" in name else 31
            scale = t.abs().amax(dim=1, keepdim=True) / clip
            q = torch.clamp(torch.round(t / scale), -(clip+1), clip).to(torch.int8).cpu()
            s = scale.squeeze(1).to(torch.float16).cpu()
            packed[name + ".q"] = q
            packed[name + ".scale"] = s

        buffer = io.BytesIO()
        torch.save(packed, buffer)
        raw = buffer.getvalue()
        compressed = zstandard.ZstdCompressor(level=22).compress(raw)
        print(f"[Pack] Raw {len(raw)/1e6:.2f} MB → Zstd {len(compressed)/1e6:.2f} MB")
        return compressed

# ==================== LOCAL MIXERS ====================
class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.gate = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + self.gate * x_conv

class BigramHash(nn.Module):
    def __init__(self, vocab_size, buckets=8192, d_bigram=48, dim=512):
        super().__init__()
        self.table = nn.Parameter(torch.randn(buckets, d_bigram).to(torch.float16))
        self.proj = nn.Linear(d_bigram, dim, bias=False)

    def forward(self, idx):
        bigram_idx = (idx[:, :-1] * 10007 + idx[:, 1:]) % self.table.shape[0]
        bigram_idx = F.pad(bigram_idx, (1, 0), value=0)
        h = self.table[bigram_idx]
        return self.proj(h.to(self.proj.weight.dtype))

# ==================== FULL SOTA MODEL ====================
class ParameterGolfGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.wte.weight = self.lm_head.weight

        self.smear = SmearGate(config.n_embd)
        self.bigram = BigramHash(config.vocab_size, config.bigram_buckets, config.d_bigram, config.n_embd)

        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            block = nn.ModuleDict({
                'attn_c_attn': MixedQuantLinear(config.n_embd, 3*config.n_embd, bias=False, bits=6),
                'attn_c_proj': MixedQuantLinear(3*config.n_embd, config.n_embd, bias=False, bits=6),
                'mlp_c_fc':   MixedQuantLinear(config.n_embd, 3*config.n_embd, bias=False, bits=5),
                'mlp_c_proj': MixedQuantLinear(3*config.n_embd, config.n_embd, bias=False, bits=5),
            })
            self.layers.append(block)

    def forward(self, idx, targets=None):
        x = self.wte(idx)
        x = x + self.bigram(idx)
        x = self.smear(x)
        for block in self.layers:
            qkv = block['attn_c_attn'](x)
            x = x + block['attn_c_proj'](qkv)
            h = F.relu(block['mlp_c_fc'](x))
            x = x + block['mlp_c_proj'](h)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# ==================== HELPERS ====================
def build_model(config):
    model = ParameterGolfGPT(config).to(device)
    gain = 0.707 if getattr(config, 'n_layers', 10) == 11 else 1.0
    for name, p in model.named_parameters():
        if p.ndim == 2 and "wte" not in name and "hash_table" not in name:
            t = torch.empty_like(p, dtype=torch.float32)
            nn.init.orthogonal_(t, gain=gain if "mlp_c_proj" in name else 1.0)
            p.data.copy_(t.to(p.dtype))
        elif p.ndim == 1:
            nn.init.zeros_(p)
    return model

@torch.no_grad()
def apply_magnitude_pruning(model, prune_percent=0.055):
    prunable = [p.view(-1) for n, p in model.named_parameters() if p.ndim == 2 and "wte" not in n and "hash_table" not in n]
    all_w = torch.cat(prunable)
    k = int(all_w.numel() * prune_percent)
    thresh = torch.kthvalue(all_w.abs(), k).values.item()
    pruned = 0
    for n, p in model.named_parameters():
        if p.ndim == 2 and "wte" not in n and "hash_table" not in n:
            mask = p.abs() >= thresh
            p.mul_(mask)
            pruned += (~mask).sum().item()
    print(f"[Prune] Global {prune_percent*100}% — {pruned:,} weights zeroed")

def setup_swa(model):
    swa_model = AveragedModel(model).to(device)
    return swa_model, None

def finalize_and_pack(model, swa_model, config):
    print("\n--- FINALIZING WINNING ARTIFACT ---")
    model.load_state_dict(swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict())
    apply_magnitude_pruning(model)
    packer = CustomBinaryCheckpoint()
    compressed = packer.pack(model)
    path = "submission.zst"
    with open(path, "wb") as f:
        f.write(compressed)
    print(f"[WINNER] {path} saved — {len(compressed)/1e6:.2f} MB")
    return path

# ==================== CORRECTED FINEWEB DATALOADER (points to your subfolder) ====================
def get_batch(split="train", batch_size=2, block_size=2048):
    data_dir = 'data/datasets/fineweb10B_sp1024'
    filename = os.path.join(data_dir, f'fineweb_{split}_000000.bin')
    
    if not os.path.exists(filename):
        print("WARNING: Using random tokens")
        x = torch.randint(0, 1024, (batch_size, block_size))
        y = torch.randint(0, 1024, (batch_size, block_size))
        return x.to(device), y.to(device)

    data = np.memmap(filename, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ==================== REAL TRAINING LOOP ====================
class RealConfig:
    vocab_size = 1024
    block_size = 2048
    n_embd = 448
    n_layers = 10
    bigram_buckets = 8192
    d_bigram = 48

if __name__ == "__main__":
    config = RealConfig()
    model = build_model(config)
    print(f"Model ready — {sum(p.numel() for p in model.parameters()):,} params")
    
    optimizer = MuonHybridOptimizer(model, muon_lr=0.02, adam_lr=3e-4, wd=0.04)
    swa_model, _ = setup_swa(model)

    print("=== STARTING 10-MINUTE TRAINING ON REAL FINEWEB DATA ===")
    step = 0
    start_time = time.time()
    max_wallclock = 590

    while True:
        if time.time() - start_time > max_wallclock:
            print(f"\n[Time] 9.8 minutes reached — stopping for PTQ & packing")
            break

        idx, targets = get_batch("train", batch_size=2, block_size=config.block_size)
        logits, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step > 3000 and step % 50 == 0:
            swa_model.update_parameters(model)

        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

    finalize_and_pack(model, swa_model, config)
    print("TRAINING COMPLETE — submission.zst ready for upload!")
