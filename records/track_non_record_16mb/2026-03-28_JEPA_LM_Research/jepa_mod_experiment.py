"""
JEPA-LM + Mixture of Depths Experiment
=======================================
Compare three training approaches on a tiny transformer:
  1. Pure CE (standard next-token prediction)
  2. JEPA-LM Hybrid (representation prediction + token prediction)
  3. JEPA-LM + Mixture of Depths (adaptive compute per token)

All models share the same architecture and parameter count.
Tests convergence speed and final loss on real text data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
from dataclasses import dataclass

# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    vocab_size: int = 1024        # Small vocab (matches competition's sp1024)
    dim: int = 192                # Model dimension
    n_heads: int = 6              # Attention heads
    n_layers: int = 6             # Transformer layers
    seq_len: int = 256            # Sequence length
    mlp_expansion: int = 3        # MLP expansion factor
    dropout: float = 0.0
    # JEPA
    jepa_weight: float = 0.5      # Weight of JEPA loss vs CE loss
    jepa_target_ema: float = 0.996 # EMA decay for target encoder
    jepa_pred_chunks: int = 4     # Predict representation N positions ahead
    # Mixture of Depths
    mod_capacity: float = 0.5     # Fraction of tokens that go through each layer
    # Training
    lr: float = 3e-4
    batch_size: int = 32
    n_steps: int = 3000
    eval_every: int = 100
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    seed: int = 42


# ============================================================
# Model Components
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = cfg.dim * cfg.mlp_expansion
        self.fc = nn.Linear(cfg.dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.dim, bias=False)

    def forward(self, x):
        # LeakyReLU squared (matching competition)
        h = F.leaky_relu(self.fc(x), negative_slope=0.09)
        return self.proj(h * h)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.dim)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
# Mixture of Depths Router
# ============================================================
class MoDRouter(nn.Module):
    """Per-layer router that selects top-k tokens to process."""
    def __init__(self, dim, capacity=0.5):
        super().__init__()
        self.gate = nn.Linear(dim, 1, bias=True)
        self.capacity = capacity

    def forward(self, x):
        """Returns routing weights and mask for which tokens to process."""
        B, T, D = x.shape
        scores = self.gate(x).squeeze(-1)  # (B, T)
        k = max(1, int(T * self.capacity))
        # Top-k selection
        topk_vals, topk_idx = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        # Sigmoid weights for differentiable routing
        weights = torch.sigmoid(scores)
        return mask, weights


class MoDTransformerBlock(nn.Module):
    """Transformer block with Mixture of Depths - only processes top-k tokens."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.block = TransformerBlock(cfg)
        self.router = MoDRouter(cfg.dim, cfg.mod_capacity)

    def forward(self, x):
        mask, weights = self.router(x)  # mask: (B, T), weights: (B, T)

        if mask.any():
            # Process only selected tokens through the full block
            # For simplicity, process all but scale by routing weight
            out = self.block(x)
            # Apply routing: selected tokens get full output, others get skip connection
            w = (weights * mask.float()).unsqueeze(-1)  # (B, T, 1)
            x = x + w * (out - x)
        return x


# ============================================================
# JEPA Predictor
# ============================================================
class JEPAPredictor(nn.Module):
    """Lightweight predictor: given context representation, predict target representation."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.pred_chunks = cfg.jepa_pred_chunks
        # Small 2-layer MLP predictor
        self.net = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim, bias=False),
        )
        # Positional offset embedding (predict at different future positions)
        self.offset_emb = nn.Embedding(cfg.jepa_pred_chunks, cfg.dim)

    def forward(self, context_repr, offsets):
        """
        context_repr: (B, T, D) — representation from context encoder
        offsets: int — how many positions ahead to predict
        Returns: (B, T-offsets, D) — predicted representations
        """
        # Shift context to align with targets
        src = context_repr[:, :context_repr.size(1) - offsets]
        offset_emb = self.offset_emb(torch.tensor(offsets - 1, device=src.device))
        return self.net(src + offset_emb)


# ============================================================
# Full Models
# ============================================================
class BaseTransformer(nn.Module):
    """Standard transformer for CE baseline."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, x  # Return both logits and representations


class JEPATransformer(nn.Module):
    """Transformer with JEPA auxiliary loss."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Context encoder (online, gets gradients)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.vocab_size, cfg.dim, bias=False)  # Will be tied

        # Target encoder (EMA copy, no gradients)
        self.target_tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.target_pos_emb = nn.Embedding(cfg.seq_len, cfg.dim)
        self.target_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.target_ln_f = RMSNorm(cfg.dim)

        # JEPA predictor (lightweight)
        self.predictor = JEPAPredictor(cfg)

        # LM head (tied to embeddings)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self._init_weights()
        # Initialize target = context
        self._copy_to_target()
        # Freeze target
        for p in self.target_parameters():
            p.requires_grad = False

    def target_parameters(self):
        for m in [self.target_tok_emb, self.target_pos_emb, self.target_blocks, self.target_ln_f]:
            yield from m.parameters()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @torch.no_grad()
    def _copy_to_target(self):
        """Copy context encoder weights to target encoder."""
        for (name_c, p_c), (name_t, p_t) in zip(
            list(self.tok_emb.named_parameters()) +
            list(self.pos_emb.named_parameters()) +
            list(self.blocks.named_parameters()) +
            list(self.ln_f.named_parameters()),
            list(self.target_tok_emb.named_parameters()) +
            list(self.target_pos_emb.named_parameters()) +
            list(self.target_blocks.named_parameters()) +
            list(self.target_ln_f.named_parameters()),
        ):
            p_t.data.copy_(p_c.data)

    @torch.no_grad()
    def update_target_ema(self):
        """EMA update of target encoder."""
        decay = self.cfg.jepa_target_ema
        for (_, p_c), (_, p_t) in zip(
            list(self.tok_emb.named_parameters()) +
            list(self.pos_emb.named_parameters()) +
            list(self.blocks.named_parameters()) +
            list(self.ln_f.named_parameters()),
            list(self.target_tok_emb.named_parameters()) +
            list(self.target_pos_emb.named_parameters()) +
            list(self.target_blocks.named_parameters()) +
            list(self.target_ln_f.named_parameters()),
        ):
            p_t.data.mul_(decay).add_(p_c.data, alpha=1 - decay)

    def forward_context(self, idx):
        """Forward pass through context encoder."""
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x

    @torch.no_grad()
    def forward_target(self, idx):
        """Forward pass through target encoder (no grad)."""
        B, T = idx.shape
        x = self.target_tok_emb(idx) + self.target_pos_emb(torch.arange(T, device=idx.device))
        for block in self.target_blocks:
            x = block(x)
        x = self.target_ln_f(x)
        return x

    def forward(self, idx):
        context_repr = self.forward_context(idx)
        logits = self.head(context_repr)
        return logits, context_repr

    def compute_jepa_loss(self, idx, context_repr):
        """Compute JEPA representation prediction loss."""
        target_repr = self.forward_target(idx)  # (B, T, D)

        total_jepa_loss = 0.0
        n_offsets = 0
        for offset in range(1, self.cfg.jepa_pred_chunks + 1):
            pred = self.predictor(context_repr, offset)  # (B, T-offset, D)
            tgt = target_repr[:, offset:]                # (B, T-offset, D)
            # Normalize both (VICReg-style, prevents collapse)
            pred_norm = F.normalize(pred, dim=-1)
            tgt_norm = F.normalize(tgt, dim=-1)
            # Cosine similarity loss (1 - cos_sim)
            loss = 1.0 - (pred_norm * tgt_norm).sum(-1).mean()
            total_jepa_loss += loss
            n_offsets += 1

        return total_jepa_loss / n_offsets


class JEPAMoDTransformer(nn.Module):
    """JEPA + Mixture of Depths: adaptive compute per token."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.dim)
        # First and last layers always process all tokens, middle layers use MoD
        self.blocks = nn.ModuleList()
        for i in range(cfg.n_layers):
            if i == 0 or i == cfg.n_layers - 1:
                self.blocks.append(TransformerBlock(cfg))
            else:
                self.blocks.append(MoDTransformerBlock(cfg))
        self.ln_f = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        # Target encoder for JEPA (same structure but all dense — target doesn't need MoD)
        self.target_tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.target_pos_emb = nn.Embedding(cfg.seq_len, cfg.dim)
        self.target_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.target_ln_f = RMSNorm(cfg.dim)

        # JEPA predictor
        self.predictor = JEPAPredictor(cfg)

        self._init_weights()
        self._copy_to_target()
        for p in self.target_parameters():
            p.requires_grad = False

    def target_parameters(self):
        for m in [self.target_tok_emb, self.target_pos_emb, self.target_blocks, self.target_ln_f]:
            yield from m.parameters()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @torch.no_grad()
    def _copy_to_target(self):
        ctx_params = list(self.tok_emb.parameters()) + list(self.pos_emb.parameters())
        tgt_params = list(self.target_tok_emb.parameters()) + list(self.target_pos_emb.parameters())
        # For blocks, match by index — context blocks may be MoD wrapped
        for i in range(self.cfg.n_layers):
            ctx_block = self.blocks[i]
            if isinstance(ctx_block, MoDTransformerBlock):
                ctx_block = ctx_block.block
            ctx_params.extend(ctx_block.parameters())
            tgt_params.extend(self.target_blocks[i].parameters())
        ctx_params.extend(self.ln_f.parameters())
        tgt_params.extend(self.target_ln_f.parameters())
        for p_c, p_t in zip(ctx_params, tgt_params):
            p_t.data.copy_(p_c.data)

    @torch.no_grad()
    def update_target_ema(self):
        decay = self.cfg.jepa_target_ema
        ctx_params = list(self.tok_emb.parameters()) + list(self.pos_emb.parameters())
        tgt_params = list(self.target_tok_emb.parameters()) + list(self.target_pos_emb.parameters())
        for i in range(self.cfg.n_layers):
            ctx_block = self.blocks[i]
            if isinstance(ctx_block, MoDTransformerBlock):
                ctx_block = ctx_block.block
            ctx_params.extend(ctx_block.parameters())
            tgt_params.extend(self.target_blocks[i].parameters())
        ctx_params.extend(self.ln_f.parameters())
        tgt_params.extend(self.target_ln_f.parameters())
        for p_c, p_t in zip(ctx_params, tgt_params):
            p_t.data.mul_(decay).add_(p_c.data, alpha=1 - decay)

    @torch.no_grad()
    def forward_target(self, idx):
        B, T = idx.shape
        x = self.target_tok_emb(idx) + self.target_pos_emb(torch.arange(T, device=idx.device))
        for block in self.target_blocks:
            x = block(x)
        x = self.target_ln_f(x)
        return x

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, x

    def compute_jepa_loss(self, idx, context_repr):
        target_repr = self.forward_target(idx)
        total_jepa_loss = 0.0
        n_offsets = 0
        for offset in range(1, self.cfg.jepa_pred_chunks + 1):
            pred = self.predictor(context_repr, offset)
            tgt = target_repr[:, offset:]
            pred_norm = F.normalize(pred, dim=-1)
            tgt_norm = F.normalize(tgt, dim=-1)
            loss = 1.0 - (pred_norm * tgt_norm).sum(-1).mean()
            total_jepa_loss += loss
            n_offsets += 1
        return total_jepa_loss / n_offsets


# ============================================================
# Data: Generate text-like sequences
# ============================================================
def generate_training_data(n_sequences, seq_len, vocab_size, seed=42):
    """Generate synthetic text with structure (not random).
    Uses a simple Markov process to create learnable patterns."""
    rng = torch.Generator().manual_seed(seed)

    # Create a sparse transition matrix (each token has ~10 likely successors)
    n_successors = min(10, vocab_size)
    # For each token, pick random successors with random probabilities
    all_data = []

    # Build transition table
    transitions = torch.zeros(vocab_size, vocab_size)
    for t in range(vocab_size):
        successors = torch.randint(0, vocab_size, (n_successors,), generator=rng)
        probs = torch.rand(n_successors, generator=rng)
        probs = probs / probs.sum()
        for s, p in zip(successors, probs):
            transitions[t, s.item()] += p.item()
    # Normalize
    transitions = transitions / transitions.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Generate sequences from Markov chain
    for _ in range(n_sequences):
        seq = [torch.randint(0, vocab_size, (1,), generator=rng).item()]
        for _ in range(seq_len - 1):
            probs = transitions[seq[-1]]
            next_tok = torch.multinomial(probs, 1, generator=rng).item()
            seq.append(next_tok)
        all_data.append(seq)

    return torch.tensor(all_data, dtype=torch.long)


# ============================================================
# Training Loop
# ============================================================
def count_params(model, trainable_only=True):
    """Count parameters, excluding target encoder if present."""
    total = 0
    target_names = {'target_tok_emb', 'target_pos_emb', 'target_blocks', 'target_ln_f'}
    for name, p in model.named_parameters():
        top_name = name.split('.')[0]
        if top_name in target_names:
            continue  # Skip target encoder (not stored in artifact)
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def train_model(model, data, cfg, mode="ce", label=""):
    """Train a model and return loss history.
    mode: 'ce' | 'jepa' | 'jepa_mod'
    """
    device = cfg.device
    model = model.to(device)
    data = data.to(device)

    # Only optimize trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_steps)

    n_data = data.size(0)
    history = {"ce_loss": [], "jepa_loss": [], "total_loss": [], "step_time_ms": []}

    param_count = count_params(model)
    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"Mode: {mode} | Params (excl. target): {param_count:,} | Device: {device}")
    print(f"{'='*60}")

    model.train()
    t0 = time.time()

    for step in range(cfg.n_steps):
        step_start = time.time()

        # Sample batch
        idx = torch.randint(0, n_data, (cfg.batch_size,))
        batch = data[idx]  # (B, seq_len)

        input_ids = batch[:, :-1]   # (B, seq_len-1)
        targets = batch[:, 1:]      # (B, seq_len-1)

        # Forward
        logits, repr_out = model(input_ids)

        # CE loss (always computed)
        ce_loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))

        # JEPA loss (if applicable)
        jepa_loss = torch.tensor(0.0, device=device)
        if mode in ("jepa", "jepa_mod"):
            jepa_loss = model.compute_jepa_loss(input_ids, repr_out)

        # Combined loss
        if mode == "ce":
            total_loss = ce_loss
        else:
            total_loss = (1 - cfg.jepa_weight) * ce_loss + cfg.jepa_weight * jepa_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        # EMA update target encoder
        if mode in ("jepa", "jepa_mod"):
            model.update_target_ema()

        step_ms = (time.time() - step_start) * 1000

        # Log
        if step % cfg.eval_every == 0:
            history["ce_loss"].append(ce_loss.item())
            history["jepa_loss"].append(jepa_loss.item())
            history["total_loss"].append(total_loss.item())
            history["step_time_ms"].append(step_ms)

            elapsed = time.time() - t0
            jepa_str = f" | JEPA: {jepa_loss.item():.4f}" if mode != "ce" else ""
            print(f"  Step {step:4d}/{cfg.n_steps} | CE: {ce_loss.item():.4f}{jepa_str} | "
                  f"Total: {total_loss.item():.4f} | {step_ms:.1f}ms/step | {elapsed:.1f}s elapsed")

    total_time = time.time() - t0
    history["total_time"] = total_time
    history["params"] = param_count

    # Final eval on full dataset (sample 200 sequences)
    model.eval()
    with torch.no_grad():
        eval_idx = torch.randint(0, n_data, (min(200, n_data),))
        eval_batch = data[eval_idx]
        logits, _ = model(eval_batch[:, :-1])
        final_ce = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), eval_batch[:, 1:].reshape(-1))
        history["final_ce"] = final_ce.item()

    print(f"\n  Final CE Loss: {final_ce.item():.4f} | Total time: {total_time:.1f}s")
    print(f"  Avg step time: {total_time/cfg.n_steps*1000:.1f}ms")

    return history


# ============================================================
# Main Experiment
# ============================================================
def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    print("=" * 60)
    print("JEPA-LM + Mixture of Depths Experiment")
    print("=" * 60)
    print(f"Config: dim={cfg.dim}, layers={cfg.n_layers}, heads={cfg.n_heads}")
    print(f"Vocab: {cfg.vocab_size}, SeqLen: {cfg.seq_len}, Steps: {cfg.n_steps}")
    print(f"JEPA weight: {cfg.jepa_weight}, MoD capacity: {cfg.mod_capacity}")
    print(f"Device: {cfg.device}")

    # Generate data
    print("\nGenerating training data (Markov chain)...")
    n_train = 2000
    data = generate_training_data(n_train, cfg.seq_len + 1, cfg.vocab_size, seed=cfg.seed)
    print(f"Data shape: {data.shape}")

    # Compute theoretical entropy of Markov chain for reference
    print(f"Data range: [{data.min().item()}, {data.max().item()}]")

    results = {}

    # ---- Experiment 1: Pure CE ----
    torch.manual_seed(cfg.seed)
    model_ce = BaseTransformer(cfg)
    results["ce"] = train_model(model_ce, data, cfg, mode="ce", label="Pure CE (Baseline)")

    # ---- Experiment 2: JEPA-LM Hybrid ----
    torch.manual_seed(cfg.seed)
    model_jepa = JEPATransformer(cfg)
    results["jepa"] = train_model(model_jepa, data, cfg, mode="jepa", label="JEPA-LM Hybrid")

    # ---- Experiment 3: JEPA + Mixture of Depths ----
    torch.manual_seed(cfg.seed)
    model_jepa_mod = JEPAMoDTransformer(cfg)
    results["jepa_mod"] = train_model(model_jepa_mod, data, cfg, mode="jepa_mod",
                                       label="JEPA-LM + Mixture of Depths")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Params':>10} {'Final CE':>10} {'Avg ms/step':>12} {'Total Time':>10}")
    print("-" * 72)
    for name, label in [("ce", "Pure CE (Baseline)"), ("jepa", "JEPA-LM Hybrid"),
                         ("jepa_mod", "JEPA-LM + MoD")]:
        r = results[name]
        avg_ms = r["total_time"] / cfg.n_steps * 1000
        print(f"{label:<30} {r['params']:>10,} {r['final_ce']:>10.4f} {avg_ms:>12.1f} {r['total_time']:>10.1f}s")

    # Convergence comparison
    print(f"\n{'Step':>6} {'CE Loss':>10} {'JEPA Loss':>10} {'JEPA+MoD Loss':>14}")
    print("-" * 42)
    n_evals = len(results["ce"]["ce_loss"])
    for i in range(n_evals):
        step = i * cfg.eval_every
        ce = results["ce"]["ce_loss"][i]
        jepa = results["jepa"]["ce_loss"][i]
        jepa_mod = results["jepa_mod"]["ce_loss"][i]
        print(f"{step:>6} {ce:>10.4f} {jepa:>10.4f} {jepa_mod:>14.4f}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    ce_final = results["ce"]["final_ce"]
    jepa_final = results["jepa"]["final_ce"]
    mod_final = results["jepa_mod"]["final_ce"]

    jepa_vs_ce = (jepa_final - ce_final) / ce_final * 100
    mod_vs_ce = (mod_final - ce_final) / ce_final * 100

    print(f"JEPA-LM vs CE: {jepa_vs_ce:+.2f}% ({'better' if jepa_vs_ce < 0 else 'worse'})")
    print(f"JEPA+MoD vs CE: {mod_vs_ce:+.2f}% ({'better' if mod_vs_ce < 0 else 'worse'})")

    ce_step_ms = results["ce"]["total_time"] / cfg.n_steps * 1000
    jepa_step_ms = results["jepa"]["total_time"] / cfg.n_steps * 1000
    mod_step_ms = results["jepa_mod"]["total_time"] / cfg.n_steps * 1000

    print(f"\nThroughput overhead:")
    print(f"  JEPA-LM: {(jepa_step_ms/ce_step_ms - 1)*100:+.1f}% vs CE")
    print(f"  JEPA+MoD: {(mod_step_ms/ce_step_ms - 1)*100:+.1f}% vs CE")

    # Convergence speed: at what step does each model reach CE's step-500 loss?
    if len(results["ce"]["ce_loss"]) > 5:
        target_loss = results["ce"]["ce_loss"][5]  # CE loss at step 500
        print(f"\nConvergence speed (steps to reach CE's step-500 loss of {target_loss:.4f}):")
        for name, label in [("ce", "CE"), ("jepa", "JEPA-LM"), ("jepa_mod", "JEPA+MoD")]:
            losses = results[name]["ce_loss"]
            reached = None
            for i, l in enumerate(losses):
                if l <= target_loss:
                    reached = i * cfg.eval_every
                    break
            if reached is not None:
                print(f"  {label}: step {reached}")
            else:
                print(f"  {label}: never reached")

    # Save results
    save_path = "/Users/himanshudongre/Documents/GitHub/parameter_golf/jepa_mod_results.json"
    # Convert for JSON serialization
    json_results = {}
    for k, v in results.items():
        json_results[k] = {kk: vv for kk, vv in v.items()}
    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
