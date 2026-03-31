"""
JEPA-LM Real Text Validation (Phase 1)
=======================================
Same architecture as jepa_mod_experiment.py but trained on REAL English text
(not synthetic Markov chains). This validates whether the 19.5% CE improvement
we saw on synthetic data holds on natural language.

Downloads a small text corpus, tokenizes with simple byte-level mapping,
and compares CE baseline vs JEPA-LM hybrid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
import urllib.request
import hashlib
from dataclasses import dataclass

# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    vocab_size: int = 1024
    dim: int = 192
    n_heads: int = 6
    n_layers: int = 6
    seq_len: int = 256
    mlp_expansion: int = 3
    dropout: float = 0.0
    # JEPA
    jepa_weight: float = 0.5
    jepa_target_ema: float = 0.996
    jepa_pred_chunks: int = 4
    # Training
    lr: float = 3e-4
    batch_size: int = 32
    n_steps: int = 3000
    eval_every: int = 100
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    seed: int = 42


# ============================================================
# Model Components (identical to jepa_mod_experiment.py)
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
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
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
# JEPA Predictor
# ============================================================
class JEPAPredictor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.pred_chunks = cfg.jepa_pred_chunks
        self.net = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim, bias=False),
        )
        self.offset_emb = nn.Embedding(cfg.jepa_pred_chunks, cfg.dim)

    def forward(self, context_repr, offsets):
        src = context_repr[:, :context_repr.size(1) - offsets]
        offset_emb = self.offset_emb(torch.tensor(offsets - 1, device=src.device))
        return self.net(src + offset_emb)


# ============================================================
# Models
# ============================================================
class BaseTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
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
        return logits, x


class JEPATransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.dim)

        # Target encoder (EMA, no gradients)
        self.target_tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.target_pos_emb = nn.Embedding(cfg.seq_len, cfg.dim)
        self.target_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.target_ln_f = RMSNorm(cfg.dim)

        self.predictor = JEPAPredictor(cfg)

        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

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
            p_t.data.copy_(p_c.data)

    @torch.no_grad()
    def update_target_ema(self):
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
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x

    @torch.no_grad()
    def forward_target(self, idx):
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
# Real Text Data Loading
# ============================================================
def download_text_corpus():
    """Download a real English text corpus for testing."""
    cache_path = "/Users/himanshudongre/Documents/GitHub/parameter_golf/text_corpus.txt"

    if os.path.exists(cache_path):
        print(f"Loading cached corpus from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    # Download several Project Gutenberg books for diverse English text
    urls = [
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",      # Alice in Wonderland
        "https://www.gutenberg.org/cache/epub/84/pg84.txt",      # Frankenstein
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",  # Sherlock Holmes
    ]

    all_text = []
    for url in urls:
        print(f"Downloading {url.split('/')[-1]}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=30)
            text = response.read().decode('utf-8', errors='ignore')
            # Strip Project Gutenberg headers/footers
            start = text.find("*** START OF")
            if start != -1:
                start = text.find("\n", start) + 1
            else:
                start = 0
            end = text.find("*** END OF")
            if end == -1:
                end = len(text)
            all_text.append(text[start:end])
        except Exception as e:
            print(f"  Failed to download: {e}")

    corpus = "\n\n".join(all_text)

    # Cache it
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(corpus)

    print(f"Total corpus: {len(corpus):,} characters")
    return corpus


def tokenize_text(text, vocab_size=1024, seq_len=257):
    """Simple byte-level tokenizer mapping to vocab_size buckets.
    This approximates the competition's sp1024 tokenizer behavior."""
    # Convert text to bytes, hash each byte to vocab_size range
    # Use overlapping byte-pairs for richer tokenization
    tokens = []
    text_bytes = text.encode('utf-8', errors='ignore')

    # Simple: map each byte (0-255) directly, use remaining slots for common bigrams
    # First 256 slots = raw bytes
    # Slots 256-1023 = common byte bigrams (hashed)
    for i in range(len(text_bytes)):
        byte_val = text_bytes[i]
        if i + 1 < len(text_bytes):
            # Check if this bigram should use a higher slot
            bigram = (text_bytes[i] << 8) | text_bytes[i + 1]
            bigram_slot = 256 + (bigram % (vocab_size - 256))
            # Use bigram slot with 30% probability (to mix unigram + bigram)
            if bigram % 3 == 0:
                tokens.append(bigram_slot)
                continue
        tokens.append(byte_val)

    # Pack into sequences
    n_sequences = len(tokens) // seq_len
    tokens = tokens[:n_sequences * seq_len]
    data = torch.tensor(tokens, dtype=torch.long).reshape(n_sequences, seq_len)

    return data


# ============================================================
# Training
# ============================================================
def count_params(model, trainable_only=True):
    total = 0
    target_names = {'target_tok_emb', 'target_pos_emb', 'target_blocks', 'target_ln_f'}
    for name, p in model.named_parameters():
        top_name = name.split('.')[0]
        if top_name in target_names:
            continue
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def train_model(model, train_data, eval_data, cfg, mode="ce", label=""):
    device = cfg.device
    model = model.to(device)
    train_data = train_data.to(device)
    eval_data = eval_data.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_steps)

    n_train = train_data.size(0)
    history = {"ce_loss": [], "eval_ce_loss": [], "jepa_loss": [], "step_time_ms": []}

    param_count = count_params(model)
    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"Mode: {mode} | Params (excl. target): {param_count:,} | Device: {device}")
    print(f"Train sequences: {n_train:,} | Eval sequences: {eval_data.size(0):,}")
    print(f"{'='*60}")

    model.train()
    t0 = time.time()

    for step in range(cfg.n_steps):
        step_start = time.time()

        idx = torch.randint(0, n_train, (cfg.batch_size,))
        batch = train_data[idx]
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        logits, repr_out = model(input_ids)
        ce_loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))

        jepa_loss = torch.tensor(0.0, device=device)
        if mode == "jepa":
            jepa_loss = model.compute_jepa_loss(input_ids, repr_out)

        if mode == "ce":
            total_loss = ce_loss
        else:
            total_loss = (1 - cfg.jepa_weight) * ce_loss + cfg.jepa_weight * jepa_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if mode == "jepa":
            model.update_target_ema()

        step_ms = (time.time() - step_start) * 1000

        if step % cfg.eval_every == 0:
            # Eval on held-out data
            model.eval()
            with torch.no_grad():
                eval_batch = eval_data[:min(200, eval_data.size(0))]
                eval_logits, _ = model(eval_batch[:, :-1])
                eval_ce = F.cross_entropy(
                    eval_logits.reshape(-1, cfg.vocab_size),
                    eval_batch[:, 1:].reshape(-1)
                )
            model.train()

            history["ce_loss"].append(ce_loss.item())
            history["eval_ce_loss"].append(eval_ce.item())
            history["jepa_loss"].append(jepa_loss.item())
            history["step_time_ms"].append(step_ms)

            elapsed = time.time() - t0
            jepa_str = f" | JEPA: {jepa_loss.item():.4f}" if mode != "ce" else ""
            print(f"  Step {step:4d}/{cfg.n_steps} | Train CE: {ce_loss.item():.4f} | "
                  f"Eval CE: {eval_ce.item():.4f}{jepa_str} | "
                  f"{step_ms:.1f}ms/step | {elapsed:.1f}s elapsed")

    total_time = time.time() - t0
    history["total_time"] = total_time
    history["params"] = param_count

    # Final eval
    model.eval()
    with torch.no_grad():
        eval_batch = eval_data[:min(500, eval_data.size(0))]
        logits, _ = model(eval_batch[:, :-1])
        final_ce = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            eval_batch[:, 1:].reshape(-1)
        )
        history["final_eval_ce"] = final_ce.item()

    print(f"\n  Final Eval CE: {final_ce.item():.4f} | Total time: {total_time:.1f}s")
    print(f"  Avg step time: {total_time/cfg.n_steps*1000:.1f}ms")

    return history


# ============================================================
# Main
# ============================================================
def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    print("=" * 60)
    print("JEPA-LM Real Text Validation (Phase 1)")
    print("=" * 60)
    print(f"Config: dim={cfg.dim}, layers={cfg.n_layers}, heads={cfg.n_heads}")
    print(f"Vocab: {cfg.vocab_size}, SeqLen: {cfg.seq_len}, Steps: {cfg.n_steps}")
    print(f"JEPA weight: {cfg.jepa_weight}")
    print(f"Device: {cfg.device}")

    # Load real text
    print("\nLoading real English text...")
    corpus = download_text_corpus()
    print(f"Corpus size: {len(corpus):,} characters")

    # Tokenize
    print("Tokenizing...")
    data = tokenize_text(corpus, cfg.vocab_size, cfg.seq_len + 1)
    print(f"Total sequences: {data.size(0):,}")
    print(f"Token range: [{data.min().item()}, {data.max().item()}]")

    # Train/eval split (90/10)
    n_total = data.size(0)
    n_eval = max(200, n_total // 10)
    perm = torch.randperm(n_total)
    eval_data = data[perm[:n_eval]]
    train_data = data[perm[n_eval:]]
    print(f"Train: {train_data.size(0):,} | Eval: {eval_data.size(0):,}")

    results = {}

    # ---- Experiment 1: Pure CE ----
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Pure CE Baseline")
    print("=" * 60)
    torch.manual_seed(cfg.seed)
    model_ce = BaseTransformer(cfg)
    results["ce"] = train_model(model_ce, train_data, eval_data, cfg,
                                 mode="ce", label="Pure CE (Baseline)")
    del model_ce
    if cfg.device == "mps":
        torch.mps.empty_cache()

    # ---- Experiment 2: JEPA-LM Hybrid ----
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: JEPA-LM Hybrid")
    print("=" * 60)
    torch.manual_seed(cfg.seed)
    model_jepa = JEPATransformer(cfg)
    results["jepa"] = train_model(model_jepa, train_data, eval_data, cfg,
                                   mode="jepa", label="JEPA-LM Hybrid")
    del model_jepa
    if cfg.device == "mps":
        torch.mps.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Real English Text)")
    print("=" * 60)

    ce_r = results["ce"]
    jepa_r = results["jepa"]

    print(f"\n{'Model':<25} {'Params':>10} {'Final Eval CE':>14} {'Avg ms/step':>12} {'Time':>8}")
    print("-" * 72)
    for name, label, r in [("ce", "Pure CE (Baseline)", ce_r),
                            ("jepa", "JEPA-LM Hybrid", jepa_r)]:
        avg_ms = r["total_time"] / cfg.n_steps * 1000
        print(f"{label:<25} {r['params']:>10,} {r['final_eval_ce']:>14.4f} "
              f"{avg_ms:>12.1f} {r['total_time']:>8.1f}s")

    # Key metric
    ce_final = ce_r["final_eval_ce"]
    jepa_final = jepa_r["final_eval_ce"]
    improvement = (jepa_final - ce_final) / ce_final * 100

    print(f"\n*** JEPA-LM vs CE on REAL TEXT: {improvement:+.2f}% ***")
    if improvement < -5:
        print("*** PASS: JEPA improves CE on real text! Proceed to Phase 2. ***")
    elif improvement < 0:
        print("*** MARGINAL: Small improvement. May not hold at scale. ***")
    else:
        print("*** FAIL: JEPA does NOT help on real text. Pivot to depth recurrence. ***")

    # Convergence comparison
    print(f"\n{'Step':>6} {'CE Train':>10} {'CE Eval':>10} {'JEPA Train':>10} {'JEPA Eval':>10}")
    print("-" * 50)
    n_evals = min(len(ce_r["ce_loss"]), len(jepa_r["ce_loss"]))
    for i in range(n_evals):
        step = i * cfg.eval_every
        print(f"{step:>6} {ce_r['ce_loss'][i]:>10.4f} {ce_r['eval_ce_loss'][i]:>10.4f} "
              f"{jepa_r['ce_loss'][i]:>10.4f} {jepa_r['eval_ce_loss'][i]:>10.4f}")

    # Throughput comparison
    ce_ms = ce_r["total_time"] / cfg.n_steps * 1000
    jepa_ms = jepa_r["total_time"] / cfg.n_steps * 1000
    overhead = (jepa_ms / ce_ms - 1) * 100
    print(f"\nThroughput: CE={ce_ms:.1f}ms/step, JEPA={jepa_ms:.1f}ms/step ({overhead:+.1f}% overhead)")

    # Throughput-adjusted comparison (what matters for competition)
    # In competition: 600s budget, each ms/step = fewer steps = higher BPB
    ce_equiv_steps = 600000 / ce_ms
    jepa_equiv_steps = 600000 / jepa_ms
    print(f"\nCompetition-equivalent steps in 600s:")
    print(f"  CE:   {ce_equiv_steps:,.0f} steps")
    print(f"  JEPA: {jepa_equiv_steps:,.0f} steps ({(jepa_equiv_steps/ce_equiv_steps - 1)*100:+.1f}%)")

    # Net value: quality improvement must exceed throughput cost
    # Rough estimate: each 1% fewer steps ≈ 1% quality loss at frontier
    quality_gain_pct = -improvement  # Positive = good
    step_loss_pct = (1 - jepa_equiv_steps / ce_equiv_steps) * 100  # Positive = bad
    net_benefit = quality_gain_pct - step_loss_pct
    print(f"\nNet benefit estimate:")
    print(f"  Quality gain: {quality_gain_pct:+.1f}%")
    print(f"  Step loss:    {step_loss_pct:+.1f}%")
    print(f"  Net:          {net_benefit:+.1f}% ({'POSITIVE — proceed!' if net_benefit > 0 else 'NEGATIVE — needs throughput optimization'})")

    # Save
    save_path = "/Users/himanshudongre/Documents/GitHub/parameter_golf/jepa_real_text_results.json"
    json_results = {}
    for k, v in results.items():
        json_results[k] = {kk: vv for kk, vv in v.items()}
    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
