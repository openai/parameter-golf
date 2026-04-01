"""
Local Test: Knowledge Distillation + QAT int4 (Train Bigger)
=============================================================

Tests the "train big, compress small" strategy at LOCAL SCALE:
  - Teacher: 6L 256d (~5M params) — trains at FP32
  - Student: 4L 192d (~1.5M params) — trains with QAT int4, soft targets
  - Comparison: Direct 4L 192d training at FP32 and QAT int4

This validates the CONCEPT before RunPod:
  1. Does distillation beat direct training at same param budget?
  2. Does QAT int4 maintain quality at higher compression?
  3. Does a larger teacher meaningfully improve student quality?

SCALING TO RUNPOD:
  Teacher: 12L 768d (~150M params) at FP16/FP8
  Student: 12L 576d (~40M params) at QAT int4
  Store student at int4 + NF4 centroids + Brotli = ~16MB

ALSO TESTS: Product Quantization compression
  - Compress 5M teacher to see reconstruction quality
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import json
import os
import urllib.request

VOCAB_SIZE = 1024
SEQ_LEN = 512
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_STEPS = 1500
BATCH_SIZE = 32
LR = 3e-4

print(f"Device: {DEVICE}")
print(f"Local Distillation + QAT Test")
print()

# ============================================================
# Data Loading
# ============================================================
def download_text_corpus():
    cache_path = "/Users/himanshudongre/Documents/GitHub/parameter_golf/text_corpus.txt"
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    urls = [
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    ]
    all_text = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=30)
            text = response.read().decode('utf-8', errors='ignore')
            all_text.append(text)
        except Exception as e:
            print(f"  Failed: {e}")
    combined = "\n\n".join(all_text)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(combined)
    return combined

def tokenize_text(text, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    raw_bytes = text.encode('utf-8')
    tokens = [b % vocab_size for b in raw_bytes]
    n_seq = len(tokens) // (seq_len + 1)
    tokens = tokens[:n_seq * (seq_len + 1)]
    return torch.tensor(tokens, dtype=torch.long).view(n_seq, seq_len + 1)

# ============================================================
# Quantization
# ============================================================
def uniform_quantize_ste(w, n_bits=4):
    """Uniform quantization with STE for QAT."""
    n_levels = 2 ** n_bits
    w_min = w.min(dim=-1, keepdim=True).values
    w_max = w.max(dim=-1, keepdim=True).values
    scale = (w_max - w_min) / (n_levels - 1)
    scale = scale.clamp(min=1e-8)
    w_norm = (w - w_min) / scale
    w_int = w_norm.round().clamp(0, n_levels - 1)
    w_q = w_int * scale + w_min
    return w + (w_q - w).detach()

# ============================================================
# Model Components
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale

class MLP(nn.Module):
    def __init__(self, dim, expansion=2.0, quant_fn=None):
        super().__init__()
        hidden = int(dim * expansion)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.quant_fn = quant_fn
    def forward(self, x):
        if self.quant_fn:
            g = F.linear(x, self.quant_fn(self.gate.weight))
            u = F.linear(x, self.quant_fn(self.up.weight))
            return F.linear(F.gelu(g) * u, self.quant_fn(self.down.weight))
        return self.down(F.gelu(self.gate(x)) * self.up(x))

class Attention(nn.Module):
    def __init__(self, dim, n_heads, rope_dims=16, quant_fn=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.quant_fn = quant_fn
        self.rope_dims = rope_dims
        freqs = 1.0 / (10000.0 ** (torch.arange(0, rope_dims, 2).float() / rope_dims))
        t = torch.arange(SEQ_LEN).float()
        freqs = torch.outer(t, freqs)
        self.register_buffer('cos_cache', freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('sin_cache', freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def _apply_rope(self, x):
        rd = self.rope_dims
        x_rope, x_pass = x[..., :rd], x[..., rd:]
        x1, x2 = x_rope[..., :rd//2], x_rope[..., rd//2:]
        cos = self.cos_cache[:, :, :x.size(2), :]
        sin = self.sin_cache[:, :, :x.size(2), :]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([out, x_pass], dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        if self.quant_fn:
            qkv = F.linear(x, self.quant_fn(self.qkv.weight))
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self._apply_rope(q), self._apply_rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        if self.quant_fn:
            return F.linear(y, self.quant_fn(self.out.weight))
        return self.out(y)

class Block(nn.Module):
    def __init__(self, dim, n_heads, expansion=2.0, quant_fn=None):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, quant_fn=quant_fn)
        self.ln2 = RMSNorm(dim)
        self.mlp = MLP(dim, expansion, quant_fn=quant_fn)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class LM(nn.Module):
    def __init__(self, dim, n_layers, n_heads, expansion=2.0, quant_fn=None):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, dim)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, expansion, quant_fn=quant_fn)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.ln_f(x), self.tok_emb.weight)

# ============================================================
# Training Functions
# ============================================================
def eval_ce(model, eval_seq):
    model.eval()
    with torch.no_grad():
        eb = eval_seq[:100].to(DEVICE)
        logits = model(eb[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), eb[:, 1:].reshape(-1))
    return ce.item()

def train_standard(model, train_seq, eval_seq, steps=TRAIN_STEPS, label=""):
    """Standard training with hard labels."""
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{label}] {n_params:,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    t0 = time.time()
    for step in range(steps + 1):
        if step % 500 == 0:
            ce = eval_ce(model, eval_seq)
            print(f"    Step {step:4d} | CE: {ce:.4f}", flush=True)
            model.train()
        if step >= steps:
            break
        bi = torch.randint(0, train_seq.size(0), (BATCH_SIZE,))
        batch = train_seq[bi].to(DEVICE)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    final_ce = eval_ce(model, eval_seq)
    elapsed = time.time() - t0
    print(f"    Final CE: {final_ce:.4f} ({elapsed:.0f}s)", flush=True)
    return final_ce

def train_distilled(student, teacher, train_seq, eval_seq, steps=1000,
                    temperature=3.0, alpha=0.5, label=""):
    """Knowledge distillation: student learns from teacher's soft targets.

    Loss = α * CE(student, hard_label) + (1-α) * T² * KL(student/T || teacher/T)

    The T² factor compensates for the gradient scaling from temperature.
    """
    student = student.to(DEVICE)
    teacher = teacher.to(DEVICE)
    teacher.eval()

    n_params = sum(p.numel() for p in student.parameters())
    print(f"  [{label}] Student: {n_params:,} params, T={temperature}, α={alpha}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    t0 = time.time()
    for step in range(steps + 1):
        if step % 500 == 0:
            ce = eval_ce(student, eval_seq)
            print(f"    Step {step:4d} | CE: {ce:.4f}", flush=True)
            student.train()
        if step >= steps:
            break

        bi = torch.randint(0, train_seq.size(0), (BATCH_SIZE,))
        batch = train_seq[bi].to(DEVICE)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Student logits
        student_logits = student(inputs)

        # Teacher logits (no grad)
        with torch.no_grad():
            teacher_logits = teacher(inputs)

        # Hard label loss
        hard_loss = F.cross_entropy(
            student_logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        )

        # Soft label loss (KL divergence at temperature T)
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        soft_loss = F.kl_div(
            student_soft.reshape(-1, VOCAB_SIZE),
            teacher_soft.reshape(-1, VOCAB_SIZE),
            reduction='batchmean'
        ) * (temperature ** 2)

        loss = alpha * hard_loss + (1 - alpha) * soft_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    final_ce = eval_ce(student, eval_seq)
    elapsed = time.time() - t0
    print(f"    Final CE: {final_ce:.4f} ({elapsed:.0f}s)", flush=True)
    return final_ce

# ============================================================
# Product Quantization
# ============================================================
def product_quantize_model(model, group_size=8, n_centroids=256):
    """Compress model weights using product quantization.

    Returns: (codebooks, indices, metadata) — everything needed to reconstruct.
    """
    compressed = {}
    total_original = 0
    total_compressed = 0

    for name, param in model.named_parameters():
        w = param.detach().cpu().numpy().flatten()
        total_original += w.nbytes

        if len(w) < group_size:
            compressed[name] = {'type': 'raw', 'data': w}
            total_compressed += w.nbytes
            continue

        # Pad to multiple of group_size
        pad_len = (group_size - len(w) % group_size) % group_size
        if pad_len > 0:
            w = np.concatenate([w, np.zeros(pad_len, dtype=w.dtype)])

        # Reshape into groups
        groups = w.reshape(-1, group_size)  # (N_groups, group_size)
        n_groups = groups.shape[0]

        # K-means clustering (simplified: random init + 10 iterations)
        # Initialize centroids from random data points
        idx = np.random.choice(n_groups, min(n_centroids, n_groups), replace=False)
        centroids = groups[idx].copy()

        for _ in range(10):
            # Assign each group to nearest centroid
            # Compute distances: (N_groups, n_centroids)
            dists = np.sum((groups[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assignments = dists.argmin(axis=1).astype(np.uint8)

            # Update centroids
            for c in range(min(n_centroids, n_groups)):
                mask = assignments == c
                if mask.sum() > 0:
                    centroids[c] = groups[mask].mean(axis=0)

        compressed[name] = {
            'type': 'pq',
            'centroids': centroids,  # (n_centroids, group_size) float32
            'assignments': assignments,  # (n_groups,) uint8
            'original_len': len(param.detach().cpu().numpy().flatten()),
            'shape': list(param.shape),
        }

        # Size: centroids + assignments
        cb_size = centroids.nbytes  # n_centroids * group_size * 4
        idx_size = assignments.nbytes  # n_groups * 1
        total_compressed += cb_size + idx_size

    ratio = total_original / total_compressed
    print(f"  Product Quantization: {total_original/1024:.1f}KB → {total_compressed/1024:.1f}KB ({ratio:.1f}x compression)", flush=True)
    return compressed, total_compressed

def reconstruct_from_pq(compressed):
    """Reconstruct weight tensors from product-quantized data."""
    weights = {}
    for name, data in compressed.items():
        if data['type'] == 'raw':
            weights[name] = torch.tensor(data['data'])
        else:
            centroids = data['centroids']
            assignments = data['assignments']
            original_len = data['original_len']
            shape = data['shape']

            # Reconstruct
            reconstructed = centroids[assignments].flatten()[:original_len]
            weights[name] = torch.tensor(reconstructed).reshape(shape)
    return weights

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Loading data...", flush=True)
    corpus = download_text_corpus()
    all_sequences = tokenize_text(corpus)
    n_train = int(len(all_sequences) * 0.9)
    train_seq = all_sequences[:n_train]
    eval_seq = all_sequences[n_train:]
    print(f"  Train: {train_seq.shape}, Eval: {eval_seq.shape}")

    results = {}

    # ==========================================
    # A: Small model direct training (baseline)
    # ==========================================
    print("\n" + "=" * 70)
    print("A: Small model (4L 192d) — direct training, FP32")
    print("=" * 70)
    torch.manual_seed(42)
    small_fp32 = LM(dim=192, n_layers=4, n_heads=6, expansion=2.0)
    ce_a = train_standard(small_fp32, train_seq, eval_seq, label="small_fp32")
    results["A_small_fp32"] = ce_a

    # ==========================================
    # B: Small model with QAT int4
    # ==========================================
    print("\n" + "=" * 70)
    print("B: Small model (4L 192d) — QAT int4")
    print("=" * 70)
    torch.manual_seed(42)
    small_qat = LM(dim=192, n_layers=4, n_heads=6, expansion=2.0,
                    quant_fn=uniform_quantize_ste)
    ce_b = train_standard(small_qat, train_seq, eval_seq, label="small_qat_int4")
    results["B_small_qat_int4"] = ce_b

    # ==========================================
    # C: Teacher model (bigger) — direct training
    # ==========================================
    print("\n" + "=" * 70)
    print("C: Teacher model (6L 256d) — direct training, FP32")
    print("=" * 70)
    torch.manual_seed(42)
    teacher = LM(dim=256, n_layers=6, n_heads=8, expansion=2.5)
    ce_c = train_standard(teacher, train_seq, eval_seq, steps=TRAIN_STEPS, label="teacher")
    results["C_teacher"] = ce_c

    # ==========================================
    # D: Distilled student — soft targets from teacher
    # ==========================================
    print("\n" + "=" * 70)
    print("D: Distilled student (4L 192d) — from teacher, FP32")
    print("=" * 70)
    for temp in [2.0, 4.0]:
        for alpha in [0.3, 0.5]:
            torch.manual_seed(42)
            student = LM(dim=192, n_layers=4, n_heads=6, expansion=2.0)
            ce_d = train_distilled(student, teacher, train_seq, eval_seq,
                                    steps=TRAIN_STEPS, temperature=temp, alpha=alpha,
                                    label=f"distill_T{temp}_a{alpha}")
            results[f"D_distill_T{temp}_a{alpha}"] = ce_d

    # ==========================================
    # E: Distilled student + QAT int4
    # ==========================================
    print("\n" + "=" * 70)
    print("E: Distilled student (4L 192d) — from teacher, QAT int4")
    print("=" * 70)
    # Use best distillation config from D
    best_d = min((v, k) for k, v in results.items() if k.startswith("D_"))
    best_d_ce, best_d_key = best_d
    print(f"  Best distill config: {best_d_key} (CE={best_d_ce:.4f})")

    # Parse temp and alpha from key
    parts = best_d_key.split("_")
    best_temp = float(parts[2][1:])
    best_alpha = float(parts[3][1:])

    torch.manual_seed(42)
    student_qat = LM(dim=192, n_layers=4, n_heads=6, expansion=2.0,
                      quant_fn=uniform_quantize_ste)
    ce_e = train_distilled(student_qat, teacher, train_seq, eval_seq,
                            steps=TRAIN_STEPS, temperature=best_temp, alpha=best_alpha,
                            label="distill_qat_int4")
    results["E_distill_qat_int4"] = ce_e

    # ==========================================
    # F: Product Quantization test
    # ==========================================
    print("\n" + "=" * 70)
    print("F: Product Quantization of teacher model")
    print("=" * 70)
    compressed, comp_size = product_quantize_model(teacher, group_size=8, n_centroids=256)
    print(f"  Compressed size: {comp_size/1024:.1f}KB")

    # Reconstruct and evaluate
    reconstructed_weights = reconstruct_from_pq(compressed)
    teacher_pq = LM(dim=256, n_layers=6, n_heads=8, expansion=2.5).to(DEVICE)
    teacher_pq_state = teacher_pq.state_dict()
    for name, tensor in reconstructed_weights.items():
        if name in teacher_pq_state:
            teacher_pq_state[name] = tensor
    teacher_pq.load_state_dict(teacher_pq_state)
    ce_f = eval_ce(teacher_pq, eval_seq)
    results["F_teacher_pq"] = ce_f
    pq_deg = (ce_f - ce_c) / ce_c * 100
    print(f"  PQ teacher CE: {ce_f:.4f} (degradation: {pq_deg:+.2f}% vs FP32 teacher)", flush=True)

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  A: Small FP32 (direct):     {results['A_small_fp32']:.4f}")
    print(f"  B: Small QAT int4 (direct):  {results['B_small_qat_int4']:.4f} ({(results['B_small_qat_int4']-results['A_small_fp32'])/results['A_small_fp32']*100:+.2f}% vs A)")
    print(f"  C: Teacher FP32:             {results['C_teacher']:.4f} ({(results['C_teacher']-results['A_small_fp32'])/results['A_small_fp32']*100:+.2f}% vs A)")

    best_d = min((v, k) for k, v in results.items() if k.startswith("D_"))
    print(f"  D: Best distilled FP32:      {best_d[0]:.4f} ({(best_d[0]-results['A_small_fp32'])/results['A_small_fp32']*100:+.2f}% vs A) [{best_d[1]}]")

    print(f"  E: Distilled QAT int4:       {results['E_distill_qat_int4']:.4f} ({(results['E_distill_qat_int4']-results['A_small_fp32'])/results['A_small_fp32']*100:+.2f}% vs A)")
    print(f"  F: Teacher PQ:               {results['F_teacher_pq']:.4f} ({pq_deg:+.2f}% vs teacher FP32)")

    print(f"\n  KEY QUESTIONS:")
    print(f"    Does distillation help?   A={results['A_small_fp32']:.4f} → D={best_d[0]:.4f} ({(best_d[0]-results['A_small_fp32'])/results['A_small_fp32']*100:+.2f}%)")
    print(f"    Does QAT int4 hold?       A={results['A_small_fp32']:.4f} → B={results['B_small_qat_int4']:.4f} ({(results['B_small_qat_int4']-results['A_small_fp32'])/results['A_small_fp32']*100:+.2f}%)")
    print(f"    Distill + QAT?            A={results['A_small_fp32']:.4f} → E={results['E_distill_qat_int4']:.4f} ({(results['E_distill_qat_int4']-results['A_small_fp32'])/results['A_small_fp32']*100:+.2f}%)")
    print(f"    Product Quant quality?    Teacher={results['C_teacher']:.4f} → PQ={results['F_teacher_pq']:.4f} ({pq_deg:+.2f}%)")

    # Save
    results_path = "/Users/himanshudongre/Documents/GitHub/parameter_golf/results_distill_qat.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    print(f"  Finished: {time.strftime('%H:%M:%S')}")
