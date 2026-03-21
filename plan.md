# Experiment Plans for Parameter Golf

## Context
- **Goal:** Beat SOTA val_bpb of 1.1748 on 8xH100 in 10 min, model < 16MB
- **Our branch:** `feat/wd-fp16embed-warmdown` — has FP16 embed passthrough, AdamW WD 0.01, warmdown 2500
- **File to modify:** `train_gpt.py` only

---

## Available Changes (Building Blocks)

### From SOTA (Proven)
- **S1:** NUM_LAYERS 9→10
- **S2:** TIED_EMBED_LR 0.05→0.10
- **S3:** Overtone spectral embedding init (SVD power-law)
- **S4:** Phase-transition residual mixing init (sigmoid per layer)
- **S5:** Sliding window eval (stride=64)

### Novel Experiments (Untested)
- **N1:** SwiGLU activation (replace ReLU²) — adds gate projection
- **N2:** Decoupled weight decay in Muon optimizer (0.01)
- **N3:** Label smoothing (0.1) on cross_entropy
- **N4:** Enable gradient clipping (GRAD_CLIP_NORM=1.0)
- **N5:** Cosine LR schedule (replace linear warmdown)
- **N6:** Embedding scaling by √dim
- **N7:** TRAIN_SEQ_LEN=2048 + reduced LRs (MATRIX_LR=0.032, SCALAR_LR=0.032)

---

## Experiment Plans

### Plan 1: Reproduce SOTA Exactly
**Changes:** S1 + S2 + S3 + S4 + S5
**Goal:** Match 1.1748 bpb as our baseline for all future experiments
**What we learn:** Confirms our implementation works correctly
**Expected:** ~1.175 bpb

---

### Plan 2: SOTA + Muon Weight Decay
**Changes:** S1 + S2 + S3 + S4 + S5 + **N2**
**Goal:** Test if real Muon WD beats SOTA (which only has AdamW WD)
**Hypothesis:** Decoupled WD on matrix params → smaller weights → better int8 quantization
**Expected:** ~1.172 bpb (smaller quant gap)

---

### Plan 3: SOTA + Regularization Stack
**Changes:** S1 + S2 + S3 + S4 + S5 + **N2 + N3 + N4**
**Goal:** Stack all regularization: Muon WD + label smoothing + grad clipping
**Hypothesis:** Combined regularization prevents oversharp logits → better generalization + quantization
**Expected:** ~1.168 bpb

---

### Plan 4: SOTA + Cosine LR Schedule
**Changes:** S1 + S2 + S3 + S4 + S5 + **N5**
**Goal:** Test cosine annealing vs linear warmdown
**Hypothesis:** Cosine is standard in LLM training; may converge better in fixed time budget
**Expected:** ~1.172 bpb

---

### Plan 5: SOTA + Longer Sequences
**Changes:** S1 + S2 + S3 + S4 + S5 + **N7**
**Goal:** Train with 2048 seq len (already proven independently to help)
**Hypothesis:** Longer context + sliding window eval is multiplicative
**Trade-off:** Fewer steps (slower per-step), but richer gradients
**Expected:** ~1.165 bpb

---

### Plan 6: SOTA + Longer Sequences + Regularization
**Changes:** S1 + S2 + S3 + S4 + S5 + **N2 + N3 + N7**
**Goal:** Combine the best of Plan 3 and Plan 5
**Hypothesis:** Longer sequences + regularization is the best combo
**Expected:** ~1.160 bpb

---

### Plan 7: SwiGLU Experiment (9 layers to fit params)
**Changes:** S2 + S3 + S4 + S5 + **N1** (with NUM_LAYERS=9 and MLP_MULT=2)
**Goal:** Test SwiGLU activation at 9 layers (extra gate param = can't fit 10 layers)
**Hypothesis:** SwiGLU quality per-param > ReLU², compensating for one fewer layer
**Trade-off:** -1 layer but better MLP — net param count similar
**Expected:** Unknown — could be better or worse

---

### Plan 8: SwiGLU + Reduced MLP Width (10 layers)
**Changes:** S1 + S2 + S3 + S4 + S5 + **N1** (with MLP_MULT adjusted down to fit 10L)
**Goal:** Keep 10 layers but shrink MLP hidden to accommodate gate projection
**Hypothesis:** SwiGLU is more efficient per-param, so smaller hidden dim still wins
**Expected:** Unknown — param budget is tight

---

### Plan 9: Kitchen Sink (Best Combo)
**Changes:** S1 + S2 + S3 + S4 + S5 + **N2 + N3 + N4 + N5 + N7**
**Goal:** Combine all low-risk novel changes at once
**Everything except SwiGLU** (which changes architecture too much to stack)
**Expected:** ~1.155 bpb (ambitious)

---

### Plan 10: Cosine + Longer Seq + Muon WD (Lean Combo)
**Changes:** S1 + S2 + S3 + S4 + S5 + **N2 + N5 + N7**
**Goal:** Three highest-impact novel changes without label smoothing
**Rationale:** Label smoothing may hurt BPB since it explicitly adds entropy to predictions
**Expected:** ~1.162 bpb

---

## Priority Order for Running

| Priority | Plan | Rationale |
|----------|------|-----------|
| 1st | **Plan 1** | Must reproduce SOTA first as baseline |
| 2nd | **Plan 2** | Minimal change, tests Muon WD alone |
| 3rd | **Plan 5** | Tests seq 2048 alone (known to work separately) |
| 4th | **Plan 4** | Tests cosine schedule alone |
| 5th | **Plan 6** | Best combo from individual winners |
| 6th | **Plan 10** | Lean combo if Plan 6 disappoints |
| 7th | **Plan 3** | Tests regularization stack |
| 8th | **Plan 7** | SwiGLU experiment (different architecture) |
| 9th | **Plan 9** | Kitchen sink if nothing else works |
| 10th | **Plan 8** | SwiGLU + 10L (hard to fit) |

---

## Per-Run Protocol
1. Run with `EVAL_STRIDE=64` on 8xH100
2. Check `final_int8_ttt_lora val_bpb` (competition score)
3. Check `final_int8_zlib_roundtrip val_bpb` (quant gap)
4. Check model size < 16MB
5. If promising: run 3 seeds (1337, 42, 7) for statistical significance
6. Record results in this doc

---

## Results Log

| Plan | Seed | val_bpb (TTT) | val_bpb (int8) | Model Size | Steps | Notes |
|------|------|--------------|----------------|------------|-------|-------|
| 1 | 1337 | — | — | — | — | TODO |
| 1 | 42 | — | — | — | — | TODO |
| 1 | 7 | — | — | — | — | TODO |

---

## Code Change Reference

### S1: NUM_LAYERS 10
```python
# Line 64
num_layers = int(os.environ.get("NUM_LAYERS", 10))
```

### S2: TIED_EMBED_LR 0.10
```python
# Line 76
tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.10))
```

### S3: Overtone Spectral Init
```python
# In _init_weights(), after nn.init.normal_():
with torch.no_grad():
    U, S, V = torch.linalg.svd(self.tok_emb.weight.data, full_matrices=False)
    target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
    self.tok_emb.weight.data = (U * target_S[None, :]) @ V
```

### S4: Phase-Transition Residual Mixing
```python
# In _init_weights(), after zero_init loop:
num_layers = len(self.blocks)
for i, block in enumerate(self.blocks):
    with torch.no_grad():
        phase = torch.sigmoid(torch.tensor(3.0 * (i / max(num_layers - 1, 1) - 0.5)))
        block.resid_mix.data[0] = phase * torch.ones(block.resid_mix.shape[1])
        block.resid_mix.data[1] = (1 - phase) * torch.ones(block.resid_mix.shape[1])
```

### S5: Sliding Window Eval
- Add `eval_seq_len` and `eval_stride` hyperparams to Hyperparameters class
- Add `forward_logits()` method to GPT class (returns logits, not loss)
- Add `eval_val_sliding()` function (overlapping window BPB scoring)
- Wire up in main() — use sliding eval when `eval_stride > 0`
- Reference implementation: `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py` lines 767-865

### N1: SwiGLU MLP
```python
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.gate(x)) * self.fc(x))
```

### N2: Muon Weight Decay
```python
# In Muon.__init__, add weight_decay to defaults dict:
dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=0.01)

# In Muon.step(), change the parameter update loop (lines ~169-172):
curr = 0
for p in params:
    g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
    p.mul_(1.0 - lr * group.get("weight_decay", 0.0))  # decoupled WD
    p.add_(g, alpha=-lr)
    curr += p.numel()
```

### N3: Label Smoothing
```python
# In GPT.forward(), line ~752, add label_smoothing parameter:
return F.cross_entropy(
    logits.float().reshape(-1, logits.size(-1)),
    target_ids.reshape(-1),
    reduction="mean",
    label_smoothing=0.1,
)
```

### N4: Gradient Clipping
```python
# Line 87 — change default from 0.0 to 1.0:
grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
```

### N5: Cosine LR Schedule
```python
# Replace lr_mul() function at lines 1167-1176:
def lr_mul(step: int, elapsed_ms: float) -> float:
    if max_wallclock_ms is None:
        return 1.0
    progress = min(elapsed_ms / max_wallclock_ms, 1.0)
    return max(0.5 * (1.0 + math.cos(math.pi * progress)), 0.0)
```

### N6: Embedding Scaling
```python
# In GPT.forward(), line ~723, add sqrt(dim) scaling:
x = self.tok_emb(input_ids) * math.sqrt(self.tok_emb.embedding_dim)
```

### N7: TRAIN_SEQ_LEN 2048 + LR Adjustments
```python
# Line 58:
train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
# Line 78 (reduce 20% for longer seqs):
matrix_lr = float(os.environ.get("MATRIX_LR", 0.032))
# Line 79:
scalar_lr = float(os.environ.get("SCALAR_LR", 0.032))
# Line 76 (if using with S2, reduce from 0.10 to 0.08):
tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.08))
```
