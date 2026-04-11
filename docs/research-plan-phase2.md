# Research Plan: Phase 2 Experiment Matrix

## Context

### Phase 1 Results (8xH100, 10-min, 1 seed)

| Rank | Experiment | val_bpb | Delta |
|------|-----------|---------|-------|
| 1 | **LeakyReLU(0.5)²** | **1.2333** | -0.0039 |
| 2 | Baseline (relu²) | 1.2372 | — |
| 3 | LeakyReLU(0.5)² + softcap=20 | 1.2406 | +0.0034 |
| 4 | softcap=20 | 1.2426 | +0.0054 |
| 5 | SiLU | 1.2509 | +0.0137 |
| 6 | GELU | 1.2542 | +0.0170 |
| 7 | sin² | 1.2937 | +0.0565 |
| 8 | sin² + softcap=20 | 1.2971 | +0.0599 |

**Key finding**: LeakyReLU(0.5)² is the only activation that beats baseline. sin² hurts significantly. softcap=20 is worse than default 30.

### Current SOTA (as of 2026-04-05)

| Rank | Submission | val_bpb | Key Techniques |
|------|-----------|---------|----------------|
| 1 | AR Self-Gen GPTQ | 1.1147 | LeakyReLU(0.5)², Full Hessian GPTQ int6, XSA-all, BigramHash 3072+trigram, Parameter Banking, Parallel Muon |
| 2 | LeakyReLU² + TTT | 1.1194 | LeakyReLU(0.5)², Legal Score-First TTT, Parallel Muon, Parameter Banking |
| 3 | Ternary U-Net | 1.1570 | 768d/10L U-Net, ternary STE, 4x relu² MLP, Poly5 softcap, NeoMuon, YaRN, 8192 BPE |

**Gap from baseline to SOTA**: 1.2372 → 1.1147 = 0.1225 bpb. Our baseline is missing ~12 techniques the SOTA stacks together.

---

## Prioritized Technique Stack

### Tier 1: Proven, Low-Risk (already in SOTA submissions)

| # | Technique | Expected gain | Implementation |
|---|-----------|--------------|----------------|
| 1a | **LeakyReLU(0.5)²** | -0.002 to -0.003 bpb | 1-line change (confirmed in Phase 1) |
| 1b | **11 layers** (from 9) | ~-0.01 bpb | Env var: NUM_LAYERS=11 |
| 1c | **3x MLP** (from 2x) | ~-0.005 bpb | Env var: MLP_MULT=3 |
| 1d | **BigramHash embedding** | ~-0.005 bpb | Add BigramHashEmbedding class, polynomial XOR hash into 3072 buckets |
| 1e | **XSA (Cross-Sequence Attention)** | ~-0.003 bpb | Gram-Schmidt orthogonalization after flash attention on last N layers |
| 1f | **U-Net skip connections** | ~-0.003 bpb | Encoder-decoder split with learned skip weights |
| 1g | **Parameter Banking** | 0 bpb (speed) | Bank all weights into contiguous 3D tensors for batched Muon NS5 |
| 1h | **Parallel Muon** | 0 bpb (speed) | Reduce-scatter → local NS5 → all-gather pipeline |
| 1i | **GPTQ int6** (full Hessian) | ~-0.005 bpb | Post-training quantization with AR self-generated calibration data |
| 1j | **Value residual / embeddings** | ~-0.002 bpb | Propagate first layer's V to all subsequent layers |

### Tier 2: Novel, Medium-Risk (from research)

| # | Technique | Expected gain | Source |
|---|-----------|--------------|--------|
| 2a | **FAN layer in MLP** | Unknown, -0.01+ if periodic structure helps | FAN (arXiv 2410.02675), FANformer (arXiv 2502.21309) |
| 2b | **DML-inspired dual-path MLP** | Unknown, novel | User's idea + DoubleMLDeep, PODNN, ND-LoRA prior art |
| 2c | **Differential Attention** | -0.005+ (65% token efficiency) | Microsoft Research 2024 |
| 2d | **Multi-Token Prediction** | +12-17% downstream | Meta arXiv 2404.19737 |
| 2e | **Variance-Adaptive Muon (Muon-VS)** | 1.36x sample efficiency | arXiv 2601.14603 |
| 2f | **NeoMuon RMSNorm-before-NS5** | Stability, especially with novel activations | Ternary U-Net submission |
| 2g | **Depth recurrence + LoRA adapters** | Nx depth at same params | Depth Recurrence submission |

### Tier 3: Experimental, High-Risk

| # | Technique | Notes |
|---|-----------|-------|
| 3a | **FAN + SwiGLU combo** | Periodic features gated by SwiGLU — novel architecture |
| 3b | **Ternary QAT (BitNet b1.58)** | 5.7x more effective params in 16MB, but complex |
| 3c | **Wavelet positional encoding** | RoPE replacement, multi-scale (arXiv 2502.02004) |
| 3d | **Token-dependent activation blend** | Mixture of activations, ~1.5K params/layer |

---

## DML-Inspired MLP Design (Deep Dive)

### Concept
Two parallel sub-networks in the MLP, inspired by Double Machine Learning's orthogonalization:
- **Path A ("nuisance")**: Captures predictable/confounding structure  
- **Path B ("target")**: Captures residual causal signal
- **Combiner**: Merges with orthogonality constraint (Barlow Twins loss)

### Prior Art
- **SwiGLU**: Already dual-pathway (`(W1·x) ⊙ swish(W2·x)`) but no orthogonality constraint
- **PODNN (2021)**: Parallel streams + Gram-Schmidt orthogonalization + aggregator
- **ND-LoRA (2025)**: Barlow Twins decorrelation of parallel streams, reduces hallucinations 14.6%
- **Bilinear FFN**: Pure `(W1·x) ⊙ (W2·x)` — closest algebraic analog to DML residual combination

### Proposed Architecture

```python
class DML_MLP(nn.Module):
    """Double ML-inspired MLP with orthogonal dual pathways."""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 2 / 3)  # budget-neutral with SwiGLU
        # Path A: "nuisance" pathway (narrower)
        self.nuisance_up = nn.Linear(dim, hidden_dim // 2, bias=False)
        # Path B: "target" pathway (wider)  
        self.target_up = nn.Linear(dim, hidden_dim, bias=False)
        # Combiner: merges residualized representations
        self.combiner = nn.Linear(hidden_dim + hidden_dim // 2, dim, bias=False)
        
    def forward(self, x):
        # Nuisance pathway: captures predictable structure
        h_n = F.leaky_relu(self.nuisance_up(x), 0.5).square()
        # Target pathway: captures residual signal
        h_t = F.leaky_relu(self.target_up(x), 0.5).square()
        # Orthogonalize: project out nuisance component from target
        # (Gram-Schmidt style, like XSA does for attention)
        h_n_norm = F.normalize(h_n, dim=-1)
        proj = (h_t * h_n_norm).sum(dim=-1, keepdim=True) * h_n_norm
        h_t_orth = h_t - proj
        # Combine
        return self.combiner(torch.cat([h_n, h_t_orth], dim=-1))
```

### Alternative: DML-Gated (SwiGLU-like)

```python
class DML_GatedMLP(nn.Module):
    """DML structure via gating with Barlow Twins regularization."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.W_nuisance = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.W_target = nn.Linear(dim, hidden_dim, bias=False)    # value
        self.W_out = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        gate = F.leaky_relu(self.W_nuisance(x), 0.5).square()  # nuisance
        value = self.W_target(x)                                  # target
        return self.W_out(gate * value)  # multiplicative combination
    
    def barlow_twins_loss(self, x):
        """Training-time regularization for orthogonality."""
        h_n = self.W_nuisance(x)
        h_t = self.W_target(x)
        # Cross-correlation matrix
        h_n = (h_n - h_n.mean(0)) / (h_n.std(0) + 1e-5)
        h_t = (h_t - h_t.mean(0)) / (h_t.std(0) + 1e-5)
        C = (h_n.T @ h_t) / h_n.shape[0]
        # Barlow Twins: minimize off-diagonal
        return (C ** 2).sum() - (C.diag() ** 2).sum()
```

### Parameter Budget (d=512, per layer)

| Design | Matrices | Params |
|--------|----------|--------|
| Standard relu² (mult=2) | 2 × 512×1024 | 1.05M |
| SwiGLU (2/3 scale) | 3 × 512×682 | 1.05M |
| DML-Gated (SwiGLU-like) | 3 × 512×682 | 1.05M |
| DML with concat+combiner | 512×341 + 512×682 + 1023×512 | 1.05M |

**The DML-Gated variant is budget-neutral with SwiGLU** — same 3-matrix structure, but with explicit orthogonality regularization via Barlow Twins loss.

---

## Proposed Phase 2 Experiment Matrix

### Round 1: Stacking Proven Techniques (1 seed each, RTX 5090 screening → H100 validation)
Build incrementally on baseline, adding one proven technique at a time:

| # | Experiment | Changes from baseline |
|---|-----------|----------------------|
| R1-1 | LeakyReLU(0.5)² + 11L + 3x MLP | Activation + architecture |
| R1-2 | R1-1 + BigramHash 3072 | + embedding enrichment |
| R1-3 | R1-2 + XSA (last 4 layers) | + attention improvement |
| R1-4 | R1-3 + U-Net skips | + gradient flow |
| R1-5 | R1-4 + Value residual | + token identity bypass |

Expected: R1-5 should approach ~1.15-1.18 bpb range.

### Round 2: Novel Designs (1 seed each, RTX 5090 screening → H100 validation)

#### 2A: MLP Architecture

| # | Experiment | MLP modification |
|---|-----------|-----------------|
| R2-1 | FAN layer (25% Fourier + 75% LeakyReLU²) | Periodic features in MLP |
| R2-2 | DML-Gated MLP + Barlow Twins loss | Dual-path with orthogonality |
| R2-3 | DML with Gram-Schmidt orthogonalization | Explicit residualization in forward pass |
| R2-4 | FAN + DML-Gated combo | Periodic features + orthogonal dual-path |

#### 2B: Causal Data Augmentation

| # | Experiment | Modification |
|---|-----------|-------------|
| R2-5 | Token dropout (10%) | Drop random input tokens, compact sequence — causal intervention on input |
| R2-6 | Token dropout (10%) + DML-Gated MLP | Test if input-side and architecture-side causal regularization compound |
| R2-7 | Token dropout (10%) + Rho-1 selective loss | Input-side (which context) + output-side (which targets) causal selection |
| R2-8 | Graduated token dropout (20%→0% over training) | Heavy augmentation early (learn robust features), clean data late (fine-tune) |

**Token dropout rationale**: Standard dropout regularizes on the *feature* dimension (don't rely on any single activation). Token dropout regularizes on the *sequence* dimension (don't rely on any single context position). These are orthogonal regularizations — one prevents feature co-adaptation, the other forces robust causal dependency learning.

**Implementation** (no architecture change needed):
```python
def token_dropout(input_ids, target_ids, drop_rate=0.1):
    """Drop random tokens from input, compact sequence.
    This is a causal intervention: do(remove token_i) — forces the model
    to learn which context positions are truly causally necessary."""
    if drop_rate <= 0 or not self.training:
        return input_ids, target_ids
    mask = torch.rand(input_ids.shape, device=input_ids.device) > drop_rate
    mask[:, 0] = True  # always keep first token
    # Compact: keep only non-dropped positions
    # Each sequence may have different length — pad or use variable-length batch
    return input_ids[mask], target_ids[mask]
```

**Efficiency bonus**: 10% token dropout = 10% shorter sequences = ~10% faster per step, so you get MORE training steps in the 10-min window while also regularizing.

**Relationship to other techniques**:
- vs **Dropout**: Orthogonal — dropout is feature-dim, token dropout is sequence-dim
- vs **Rho-1 selective loss**: Complementary — Rho-1 selects which tokens to LEARN FROM (output side), token dropout selects which context to LEARN WITH (input side)
- vs **Attention dropout**: Related but different — attention dropout randomly zeros attention scores, token dropout removes the tokens entirely (stronger intervention)
- vs **BERT masking**: Different objective — BERT masks and predicts the masked token, token dropout removes and doesn't predict (pure regularization)

#### 2B-extra: Corrupted Context Training (Scheduled Sampling variant)

| # | Experiment | Modification |
|---|-----------|-------------|
| R2-11 | Corrupted context (10%) | Replace 10% of ground truth tokens with model's own predictions, train on corrupted context |
| R2-12 | Graduated corruption (0%→20%→0%) | No corruption early (learn basics), ramp up mid-training (learn robustness), clean late (fine-tune) |

**Corrupted context rationale**: Standard training always shows the model perfect context (teacher forcing). At inference, the model sees its own (possibly wrong) predictions — a distribution it's never trained on. Corrupted context training bridges this gap by occasionally replacing ground truth tokens with the model's predictions, forcing it to learn to predict correctly from imperfect context.

**Implementation** (no architecture change, applied during data loading):
```python
def corrupted_context(model, input_ids, corruption_rate=0.1):
    """Replace random ground truth tokens with model's own predictions.
    Bridges the train/inference gap (exposure bias) without expensive
    autoregressive rollout — one forward pass to get predictions, then
    swap tokens and train on the corrupted version."""
    if corruption_rate <= 0 or not model.training:
        return input_ids
    with torch.no_grad():
        logits = model(input_ids)
        predictions = logits.argmax(dim=-1)  # model's best guess per position
    mask = torch.rand(input_ids.shape, device=input_ids.device) < corruption_rate
    mask[:, 0] = False  # keep first token
    corrupted = input_ids.clone()
    corrupted[mask] = predictions[mask]
    return corrupted
```

**Cost**: One extra forward pass per batch (no backward) — roughly 30-40% overhead. In a 10-min window this means ~25% fewer training steps, so the corruption signal must outweigh the step reduction.

**Relationship to other techniques**:
- vs **Token dropout**: Dropout removes tokens entirely (model sees less context). Corruption replaces with plausible-but-wrong tokens (model sees misleading context). Dropout is cheaper (no extra forward pass). Both are causal interventions but on different aspects.
- vs **Multi-Token Prediction**: MTP predicts further ahead from correct context. Corruption trains on incorrect context. Complementary.
- vs **Scheduled Sampling** (Bengio 2015): Same concept but simpler — we don't need to mix sampling strategies, just replace tokens. The "graduated" variant (R2-12) implements the scheduling.
- vs **AR Self-Generation** (SOTA #1): They generate autoregressively for GPTQ calibration post-training. We apply the same idea during training as data augmentation.

#### 2C: Causal Attention & Connectivity

| # | Experiment | Modification |
|---|-----------|-------------|
| R2-9 | Cross-layer Barlow Twins | Decorrelation loss between successive transformer block outputs |
| R2-10 | Learned DAG skip connections | Replace fixed U-Net skips with sparse, prunable layer-to-layer edges |

### Round 3: Quantization & Eval (3 seeds on best R1+R2 config)

| # | Experiment | Addition |
|---|-----------|---------|
| R3-1 | Best config + GPTQ int6 | Post-training quantization |
| R3-2 | R3-1 + sliding window eval (stride=64) | Better evaluation |
| R3-3 | R3-2 + Legal TTT | Test-time training |

### Cost Estimate

**Screening on RTX 5090** (~$0.45/run):
- Round 1: 5 runs × ~12 min ≈ $2.25
- Round 2: 12 runs × ~12 min ≈ $5.40
- **Screening subtotal: ~$7.65**

**Validation on 8xH100** (~$3.59/run):
- Round 3 (winners only): ~3 configs × 3 seeds × ~12 min ≈ $32.30
- **Validation subtotal: ~$32.30**

**Total: ~$40** (down from ~$77 by screening on RTX 5090)

---

## Implementation Order

1. **Immediate**: Build R1 config by porting SOTA techniques from the top submissions' code
2. **Screen R1 on RTX 5090**: Run all 5 R1 experiments, identify which technique stack converges best
3. **Implement R2 variants**: FAN layer, DML-Gated MLP, token dropout, cross-layer Barlow Twins
4. **Screen R2 on RTX 5090**: Run all 10 R2 experiments on best R1 config
5. **H100 validation**: Take top 2-3 configs to 8xH100 for 3-seed statistical significance runs
6. **Submission assembly**: GPTQ int6 + sliding window eval + Legal TTT on best config
