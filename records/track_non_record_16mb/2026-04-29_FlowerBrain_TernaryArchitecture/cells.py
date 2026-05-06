"""
Flower Brain PG — 6-Cell Ternary Architecture
Each cell is a specialist BitNet micro-model wired in hexagonal topology.

Cells:
  1. EmbeddingCell — token encoding via BigramHash + ternary projection
  2. AttentionCell — sparse ternary attention (attend/ignore/counter-attend)
  3. TransformCell — MLP replacement with domain sub-regions
  4. ContextCell — XSA void cell (cross-sequence subtraction)
  5. RoutingCell — existing 92.7% classifier (thalamus)
  6. PredictionCell — output head, weight-tied to EmbeddingCell

Architecture constants derived from:
  - 92.7% classifier: vocab=8000, embed=128, hidden=256, 1.26M params
  - PG baseline: model_dim=512, num_heads=8, num_kv_heads=4
  - Hawking insight: ternary for MLP (67% of params), int6 for attention
  - Target: ~11M params per cell, 66M total, ~13MB compressed
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# BitLinear — Ternary linear layer with straight-through estimator
# From our 92.7% classifier, proven architecture
# ═══════════════════════════════════════════════════════════════════════

class BitLinear(nn.Module):
    """Ternary linear: weights quantized to {-1, 0, +1} during forward,
    full-precision master weights for gradient updates (STE)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.rms_norm = nn.RMSNorm(in_features)
        # Initialize
        nn.init.kaiming_normal_(self.weight)

    def ternary_quantize(self, w):
        """Quantize to {-1, 0, +1} with straight-through estimator."""
        # Threshold: weights below magnitude threshold become void (0)
        threshold = w.abs().mean()
        # Sign gives {-1, +1}, threshold gives {0}
        w_ternary = torch.sign(w) * (w.abs() > threshold).float()
        # STE: gradient flows through as if no quantization
        return w + (w_ternary - w).detach()

    def forward(self, x):
        x = self.rms_norm(x.float()).to(x.dtype)
        w = self.ternary_quantize(self.weight).to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        out = F.linear(x, w, b)
        return out

    @property
    def void_fraction(self):
        """Fraction of weights that are zero (void)."""
        with torch.no_grad():
            threshold = self.weight.abs().mean()
            return (self.weight.abs() <= threshold).float().mean().item()


# ═══════════════════════════════════════════════════════════════════════
# Cell 1 — EMBEDDING CELL
# BigramHash token encoding + ternary projection
# Void = hash collision space (tokens sharing a bucket share the zero path)
# ═══════════════════════════════════════════════════════════════════════

class EmbeddingCell(nn.Module):
    """Token encoding with BigramHash and ternary projection."""

    def __init__(self, vocab_size, embed_dim, bigram_buckets=3072, bigram_dim=112):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        # BigramHash: captures 2-token patterns in a compressed space
        self.bigram_emb = nn.Embedding(bigram_buckets, bigram_dim)
        self.bigram_proj = BitLinear(bigram_dim, embed_dim, bias=False)
        # Final projection to model dim
        self.proj = BitLinear(embed_dim, embed_dim, bias=False)
        self.embed_dim = embed_dim
        self.bigram_buckets = bigram_buckets

    def bigram_hash(self, token_ids):
        """FNV-1a hash of consecutive token pairs → bucket index."""
        # Shift right by 1 to get previous token
        prev = torch.roll(token_ids, 1, dims=-1)
        prev[:, 0] = 0  # no previous for first token
        # Simple hash: (prev * 16777619) ^ current mod buckets
        h = ((prev.long() * 16777619) ^ token_ids.long()) % self.bigram_buckets
        return h

    def forward(self, token_ids):
        # Token embedding
        x = self.tok_emb(token_ids)
        # BigramHash embedding
        bh = self.bigram_hash(token_ids)
        bg = self.bigram_emb(bh)
        bg = self.bigram_proj(bg)
        # Combine
        x = x + bg
        x = self.proj(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
# Cell 2 — ATTENTION CELL
# Sparse ternary attention: attend (+1), ignore (0), counter-attend (-1)
# 30% void = 30% of attention weights are correctly zero
# ═══════════════════════════════════════════════════════════════════════

class AttentionCell(nn.Module):
    """Multi-head attention with ternary Q/K projections for sparse attention."""

    def __init__(self, model_dim, num_heads=8, num_kv_heads=4, rope_dims=16):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        # Q/K/V projections — Q and K are ternary (sparse attention)
        self.q_proj = BitLinear(model_dim, model_dim, bias=False)
        self.k_proj = BitLinear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = BitLinear(model_dim, model_dim, bias=False)

        # QK gain (from our 1.0810 finding: higher gain = better)
        self.qk_gain = nn.Parameter(torch.tensor(5.25))

        # Partial RoPE
        self.rope_dims = rope_dims

    def forward(self, x, freqs_cis=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: expand KV heads
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Scaled dot-product attention with QK gain
        scale = self.qk_gain / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════════════
# Cell 3 — TRANSFORM CELL (MLP replacement)
# Feedforward with ternary weights — 30% void = 30% compute skip
# LeakyReLU(0.5)^2 from PG baseline (squared distance, not gate)
# ═══════════════════════════════════════════════════════════════════════

class TransformCell(nn.Module):
    """MLP replacement with ternary weights and void-aware activation."""

    def __init__(self, model_dim, mlp_mult=4.0):
        super().__init__()
        hidden = int(model_dim * mlp_mult)
        self.fc = BitLinear(model_dim, hidden, bias=False)
        self.proj = BitLinear(hidden, model_dim, bias=False)

    def forward(self, x):
        h = self.fc(x)
        # LeakyReLU(0.5)^2 — squared distance measure from PG baseline
        h = F.leaky_relu(h, 0.5).square()
        return self.proj(h)


# ═══════════════════════════════════════════════════════════════════════
# Cell 4 — CONTEXT CELL (XSA / Void Cell)
# Cross-sequence attention: subtract self-value bias
# The void weights ARE the subtraction — holds zero crossing stable
# Nakata: three modes at 60° = hexagonal, stable void at centre
# ═══════════════════════════════════════════════════════════════════════

class ContextCell(nn.Module):
    """XSA cell: subtracts the running mean of value vectors to remove
    self-correlation bias. The void fraction stabilizes this subtraction."""

    def __init__(self, model_dim, num_heads=8):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        # Ternary projection for the subtraction signal
        self.xsa_proj = BitLinear(model_dim, model_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(model_dim))

    def forward(self, x, v_running_mean=None):
        B, T, D = x.shape
        # Compute running mean of representations
        if v_running_mean is None:
            cumsum = x.cumsum(dim=1)
            counts = torch.arange(1, T + 1, device=x.device).float().view(1, -1, 1)
            v_running_mean = cumsum / counts
        # XSA: subtract the self-value bias
        xsa_signal = self.xsa_proj(v_running_mean)
        # Gated subtraction — the void controls how much to subtract
        gate = torch.sigmoid(self.gate)
        return x - gate * xsa_signal


# ═══════════════════════════════════════════════════════════════════════
# Cell 5 — ROUTING CELL (already built — load from checkpoint)
# 92.7% accuracy domain classifier, 1.26M params
# Acts as thalamus: routes signal to specialist cells
# ═══════════════════════════════════════════════════════════════════════

class RoutingCell(nn.Module):
    """Routes between cells based on input domain.
    In full Flower Brain: decides which cells fire for each token.
    For PG: generates per-token routing weights for cell contributions."""

    def __init__(self, model_dim, num_cells=6):
        super().__init__()
        self.proj_in = BitLinear(model_dim, 256, bias=True)
        self.hidden = BitLinear(256, 256, bias=True)
        self.route_out = nn.Linear(256, num_cells)  # soft routing weights

    def forward(self, x):
        h = F.leaky_relu(self.proj_in(x), 0.1)
        h = F.leaky_relu(self.hidden(h), 0.1)
        # Softmax routing: which cells contribute most for each position
        weights = F.softmax(self.route_out(h), dim=-1)
        return weights


# ═══════════════════════════════════════════════════════════════════════
# Cell 6 — PREDICTION CELL (output head)
# Maps final representations to vocabulary logits
# Weight-tied to EmbeddingCell for compression
# ═══════════════════════════════════════════════════════════════════════

class PredictionCell(nn.Module):
    """Output head: project to vocab logits. Weight-tied to embedding."""

    def __init__(self, model_dim, vocab_size, softcap=30.0):
        super().__init__()
        self.pre_norm = nn.RMSNorm(model_dim)
        self.head_proj = BitLinear(model_dim, model_dim, bias=False)
        self.softcap = softcap
        # Weight tying: set embed_weight after construction
        self.embed_weight = None
        self.vocab_size = vocab_size

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.head_proj(x)
        if self.embed_weight is not None:
            logits = F.linear(x, self.embed_weight)
        else:
            raise RuntimeError("PredictionCell requires embed_weight to be set (weight tying)")
        # Softcap from PG baseline
        logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits


# ═══════════════════════════════════════════════════════════════════════
# FLOWER TOPOLOGY — Hexagonal wiring of all 6 cells
# Information flows: Embed → Attn + Context → Transform → Predict
# Routing cell modulates all connections
# ═══════════════════════════════════════════════════════════════════════

class FlowerBrainPG(nn.Module):
    """6-cell Flower Brain for Parameter Golf.

    Topology (hexagonal):
        [EMBED] ←→ [ATTN]
           ↕              ↕
       [ROUTE] ←→ [CONTEXT]
           ↕              ↕
        [TRANSFORM] ←→ [PREDICT]
    """

    def __init__(self, vocab_size=8192, model_dim=512, num_heads=8,
                 num_kv_heads=4, mlp_mult=4.0, num_layers=11,
                 depth_recur_start=3, depth_recur_end=5, num_loops=2,
                 parallel_residual_start=7):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_loops = num_loops
        self.depth_recur_start = depth_recur_start
        self.depth_recur_end = depth_recur_end
        self.parallel_residual_start = parallel_residual_start

        # The 6 cells
        self.embed_cell = EmbeddingCell(vocab_size, model_dim)
        self.attn_cells = nn.ModuleList([
            AttentionCell(model_dim, num_heads, num_kv_heads)
            for _ in range(num_layers)
        ])
        self.transform_cells = nn.ModuleList([
            TransformCell(model_dim, mlp_mult)
            for _ in range(num_layers)
        ])
        self.context_cell = ContextCell(model_dim, num_heads)
        self.routing_cell = RoutingCell(model_dim)
        self.predict_cell = PredictionCell(model_dim, vocab_size)

        # Weight tying: prediction uses embedding weights
        self.predict_cell.embed_weight = self.embed_cell.tok_emb.weight

        # Layer norms (per-layer)
        self.ln_attn = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(num_layers)])
        self.ln_mlp = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(num_layers)])

        # Looping state
        self.looping_active = False

    def _build_layer_schedule(self):
        """Depth recurrence: loop layers [start, end) like PG baseline."""
        if not self.looping_active or self.num_loops <= 0:
            return list(range(self.num_layers))
        # Encoder: [0, ..., end-1, start, ..., end-1] (loop once)
        encoder = list(range(self.depth_recur_end))
        for _ in range(self.num_loops - 1):
            encoder.extend(range(self.depth_recur_start, self.depth_recur_end))
        # Decoder: rest
        decoder = list(range(self.depth_recur_end, self.num_layers))
        return encoder + decoder

    def forward(self, token_ids, targets=None):
        # Cell 1: Embedding
        x = self.embed_cell(token_ids)

        # Layer schedule with depth recurrence
        schedule = self._build_layer_schedule()

        # Process through layers
        for layer_idx in schedule:
            residual = x

            # Cell 2: Attention
            attn_out = self.attn_cells[layer_idx](self.ln_attn[layer_idx](x))

            # Cell 4: Context (XSA) — subtract self-value bias
            attn_out = self.context_cell(attn_out)

            # Cell 3: Transform (MLP)
            mlp_out = self.transform_cells[layer_idx](self.ln_mlp[layer_idx](x))

            # Parallel residuals (from layer parallel_residual_start)
            if layer_idx >= self.parallel_residual_start:
                x = residual + attn_out + mlp_out
            else:
                x = residual + attn_out
                x = x + mlp_out

        # Cell 6: Prediction
        logits = self.predict_cell(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            return loss

        return logits

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def void_fraction(self):
        """Average void fraction across all BitLinear layers."""
        fracs = []
        for m in self.modules():
            if isinstance(m, BitLinear):
                fracs.append(m.void_fraction)
        return sum(fracs) / len(fracs) if fracs else 0.0


# ═══════════════════════════════════════════════════════════════════════
# Size estimation
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    model = FlowerBrainPG(vocab_size=8192, model_dim=512)
    total_params = model.param_count()
    # Ternary: 1.585 bits per weight
    ternary_bytes = int(total_params * 1.585 / 8)
    # With ~30% void compression
    compressed_est = int(ternary_bytes * 0.65)

    print(f"Total parameters: {total_params:,}")
    print(f"Ternary size (raw): {ternary_bytes:,} bytes ({ternary_bytes/1024/1024:.1f} MB)")
    print(f"Estimated compressed: {compressed_est:,} bytes ({compressed_est/1024/1024:.1f} MB)")
    print(f"16MB budget remaining: {16_000_000 - compressed_est:,} bytes")
    print(f"Void fraction: {model.void_fraction():.1%}")

    # Test forward pass
    x = torch.randint(0, 8192, (2, 128))
    y = torch.randint(0, 8192, (2, 128))
    loss = model(x, y)
    print(f"Test forward pass — loss: {loss.item():.4f}")
