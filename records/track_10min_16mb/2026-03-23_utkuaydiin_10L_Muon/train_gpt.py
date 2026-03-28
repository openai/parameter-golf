import io
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.swa_utils import AveragedModel

# COPILOT FIX: Optional zstandard with zlib fallback for portability
try:
    import zstandard
except ImportError:
    import zlib

    class _ZlibShim:
        class ZstdCompressor:
            def __init__(self, level=3):
                self.level = 9  # zlib max

            def compress(self, data):
                return zlib.compress(data, self.level)

        class ZstdDecompressor:
            def decompress(self, data):
                return zlib.decompress(data)

    zstandard = _ZlibShim()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True


# ==================== MIXED QUANT LINEAR ====================
class MixedQuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, bits=6):
        super().__init__(in_features, out_features, bias=bias)
        self.bits = bits

    def quantize_post_training(self):
        t = self.weight.data.float()
        clip = 15 if self.bits == 5 else 31
        # COPILOT FIX: Guard against zero-scale rows to avoid NaN
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
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class MuonHybridOptimizer(Optimizer):
    def __init__(self, model, muon_lr=0.02, adam_lr=3e-4, wd=0.04):
        # COPILOT FIX: Corrected bigram.table naming (was hash_table)
        self.muon_params = [
            p
            for n, p in model.named_parameters()
            if p.ndim == 2 and "wte" not in n and "bigram.table" not in n
        ]
        muon_ids = {id(p) for p in self.muon_params}
        adam_params = [p for p in model.parameters() if id(p) not in muon_ids]

        # COPILOT FIX: Proper Optimizer initialization
        defaults = dict(muon_lr=muon_lr, adam_lr=adam_lr, wd=wd)
        super().__init__(model.parameters(), defaults)

        self.adam = torch.optim.AdamW(
            adam_params, lr=adam_lr, weight_decay=0.0, betas=(0.9, 0.95)
        )
        self.muon_lr = muon_lr
        self.wd = wd
        self.step_count = 0

    def step(self):
        self.step_count += 1
        momentum = 0.92
        if self.step_count < 5000:
            momentum = 0.92 + 0.07 * (self.step_count / 5000)

        self.adam.step()
        for p in self.muon_params:
            if p.grad is None:
                continue

            # COPILOT FIX: Store momentum in self.state
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.data)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(p.grad.data)
            g = p.grad.data + momentum * buf
            g_ortho = zeropower_via_newtonschulz5(g)

            if self.wd > 0:
                g_ortho = g_ortho + self.wd * p.data
            p.data.add_(g_ortho, alpha=-self.muon_lr)

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none)
        for p in self.muon_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()


# ==================== PACKER & PRUNER ====================
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
            # COPILOT FIX: Clamp scale to avoid NaN on zero-rows
            scale = t.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / clip
            q = (
                torch.clamp(torch.round(t / scale), -(clip + 1), clip)
                .to(torch.int8)
                .cpu()
            )
            s = scale.squeeze(1).to(torch.float16).cpu()
            packed[name + ".q"] = q
            packed[name + ".scale"] = s

        buffer = io.BytesIO()
        torch.save(packed, buffer)
        return zstandard.ZstdCompressor(level=22).compress(buffer.getvalue())


@torch.no_grad()
def apply_magnitude_pruning(model, prune_percent=0.055):
    # COPILOT FIX: Consistency with bigram.table name
    prunable = [
        p
        for n, p in model.named_parameters()
        if p.ndim == 2 and "wte" not in n and "bigram.table" not in n
    ]
    if not prunable or prune_percent <= 0:
        return

    all_w = torch.cat([p.view(-1) for p in prunable])
    numel = all_w.numel()
    # COPILOT FIX: Clamp k to avoid kthvalue domain errors
    k = max(1, min(int(numel * prune_percent), numel - 1))

    thresh = torch.kthvalue(all_w.abs(), k).values.item()
    pruned = 0
    for p in prunable:
        mask = p.abs() >= thresh
        p.mul_(mask)
        pruned += (~mask).sum().item()
    print(f"[Prune] Global {prune_percent * 100}% — {pruned:,} weights zeroed")


# ==================== ARCHITECTURE ====================
class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
        )
        self.gate = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + self.gate * x_conv


class BigramHash(nn.Module):
    def __init__(self, vocab_size, buckets, d_bigram, dim):
        super().__init__()
        self.table = nn.Parameter(torch.randn(buckets, d_bigram).to(torch.float16))
        self.proj = nn.Linear(d_bigram, dim, bias=False)

    def forward(self, idx):
        bigram_idx = (idx[:, :-1] * 10007 + idx[:, 1:]) % self.table.shape[0]
        bigram_idx = F.pad(bigram_idx, (1, 0), value=0)
        h = self.table[bigram_idx]
        return self.proj(h.to(self.proj.weight.dtype))


class ParameterGolfGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.wte.weight = self.lm_head.weight
        self.smear = SmearGate(config.n_embd)
        self.bigram = BigramHash(
            config.vocab_size, config.bigram_buckets, config.d_bigram, config.n_embd
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn_c_attn": MixedQuantLinear(
                            config.n_embd, 3 * config.n_embd, bits=6
                        ),
                        "attn_c_proj": MixedQuantLinear(
                            3 * config.n_embd, config.n_embd, bits=6
                        ),
                        "mlp_c_fc": MixedQuantLinear(
                            config.n_embd, 3 * config.n_embd, bits=5
                        ),
                        "mlp_c_proj": MixedQuantLinear(
                            3 * config.n_embd, config.n_embd, bits=5
                        ),
                    }
                )
                for _ in range(config.n_layers)
            ]
        )

    def forward(self, idx, targets=None):
        x = self.wte(idx) + self.bigram(idx)
        x = self.smear(x)
        for b in self.layers:
            x = x + b["attn_c_proj"](b["attn_c_attn"](x))
            x = x + b["mlp_c_proj"](F.relu(b["mlp_c_fc"](x)))
        logits = self.lm_head(x)
        return logits, F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        ) if targets is not None else None


# ==================== HELPERS & DATA ====================
def build_model(config):
    model = ParameterGolfGPT(config).to(device)
    for name, p in model.named_parameters():
        if p.ndim == 2 and "wte" not in name and "bigram.table" not in name:
            nn.init.orthogonal_(p)
        elif p.ndim == 1:
            nn.init.zeros_(p)
    return model


def get_batch(split="train", batch_size=2, block_size=2048):
    data_dir = "data/datasets/fineweb10B_sp1024"
    filename = os.path.join(data_dir, f"fineweb_{split}_000000.bin")

    # COPILOT FIX: Fail-fast for data missing (maintainer requirement)
    if not os.path.exists(filename):
        if os.getenv("FINEWEB_ALLOW_RANDOM_TOKENS", "0") == "1":
            x = torch.randint(0, 1024, (batch_size, block_size))
            y = torch.randint(0, 1024, (batch_size, block_size))
            return x.to(device), y.to(device)
        raise FileNotFoundError(
            f"Shard not found: {filename}. Set FINEWEB_ALLOW_RANDOM_TOKENS=1 to test."
        )

    # COPILOT FIX: Skip 1024-byte header
    header_nbytes = 1024
    data = np.memmap(filename, dtype=np.uint16, mode="r", offset=header_nbytes)

    # COPILOT FIX: -1 range guard
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    return x.to(device), y.to(device)


def finalize_and_pack(model, swa_model, config):
    print("\n--- FINALIZING WINNING ARTIFACT ---")

    # COPILOT FIX: SWA safety guard (only load if we actually averaged steps)
    if swa_model is not None and getattr(swa_model, "n_averaged", 0) > 0:
        model.load_state_dict(swa_model.module.state_dict())

    apply_magnitude_pruning(model)
    compressed = CustomBinaryCheckpoint().pack(model)
    with open("submission.zst", "wb") as f:
        f.write(compressed)
    print(f"DONE. Size: {len(compressed) / 1e6:.2f} MB")


# ==================== MAIN ====================
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
    optimizer = MuonHybridOptimizer(model)
    swa_model = AveragedModel(model)

    print("=== STARTING 10-MINUTE TRAINING ===")
    start_time = time.time()
    while time.time() - start_time < 590:
        idx, targets = get_batch()
        logits, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if optimizer.step_count > 3000 and optimizer.step_count % 50 == 0:
            swa_model.update_parameters(model)

        if optimizer.step_count % 50 == 0:
            print(
                f"Step {optimizer.step_count} | Loss: {loss.item():.4f} | Time: {time.time() - start_time:.1f}s"
            )

    finalize_and_pack(model, swa_model, config)
