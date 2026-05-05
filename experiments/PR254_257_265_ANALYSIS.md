# Analysis of PRs #254, #257, #265 on openai/parameter-golf
**Date:** 2026-03-20

---

## PR #254 — FarnsworthEngine v1 (timowhite88)
**val_bpb: 1.1303** (mean of 3 seeds: 1.1313)
**Artifact: 15,877,181 bytes (15.88 MB)**
**Steps: 7,248 at 81.5ms/step on 8×H100**

### Architecture & Techniques
| Component | Value |
|-----------|-------|
| Layers | 11 (dim=512, 8 heads, 4 KV heads GQA) |
| MLP | 3× expansion (1536 hidden), ReLU² |
| Quantization | Mixed int6 (MLP+attn) + int8 (embed) + FP16 tied embed |
| Compression | zstd-22 |
| SmearGate | ✅ (512 params) |
| BigramHash | 2048 buckets, dim=128 |
| Init | Orthogonal + muP |
| Optimizer | Muon WD=0.04, momentum=0.99, warmup 1500 steps |
| SWA | 7 checkpoint average during warmdown |
| Attention | FlashAttention 3 |
| RoPE | NTK base=50000 |
| Sequence | Train@2048, eval@2048 |
| **TTT** | **Full-weight SGD on val data (lr=0.002, mom=0.9, 3 epochs, freeze first 2 blocks)** |
| Eval | Sliding window stride=64 |

### TTT Implementation Details (lines 1038-1104)
```python
def ttt_adapt(args, base_model, device, val_tokens, ...):
    # 1. Freeze first 2 blocks for stability
    for i, block in enumerate(base_model.blocks):
        if i < args.ttt_freeze_blocks:
            for p in block.parameters():
                p.requires_grad_(False)
    
    # 2. Full-weight SGD (NOT LoRA) on all unfrozen params
    optimizer = torch.optim.SGD(ttt_params, lr=0.002, momentum=0.9)
    
    # 3. Run 3 epochs over val data, all-reduce grads across 8 GPUs
    for epoch in range(3):
        for batch in val_batches:
            loss = base_model(x, y)
            loss.backward()
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            clip_grad_norm_(ttt_params, 1.0)
            optimizer.step()
```

### TTT Timing
| Phase | Time |
|-------|------|
| Training | 600s (wallclock cap) |
| TTT adaptation | 43s |
| Sliding window eval | 86s |
| **Total eval** | **129s** (well within 600s eval budget) |

### TTT Loss Progression
```
ttt_epoch:1/3 loss:1.9496 time:14.5s
ttt_epoch:2/3 loss:1.9482 time:28.7s  (delta: -0.0014)
ttt_epoch:3/3 loss:1.9475 time:43.0s  (delta: -0.0007)
```
Very small loss change — model barely adapts in 3 epochs.

### TTT Impact Analysis
Comparing PR #254 (with TTT) vs PR #265 (same arch, no TTT):
- **Non-sliding after quant**: 1.1528 (PR254) vs 1.1529 (PR265) = **-0.0001 BPP from TTT**
- **Sliding window**: 1.1303 (PR254) vs 1.1307 (PR265) = **-0.0004 BPP from TTT**
- **TTT cost: 43 seconds for ~0.0004 BPP improvement**

### What's NOVEL vs What We Know
| Technique | Novel? | Notes |
|-----------|--------|-------|
| 11L MLP3x int6 | ❌ | We already have this (exp087, exp091) |
| SmearGate + BigramHash | ❌ | Already have |
| SWA | ❌ | Already have |
| zstd-22 | ❌ | Already have |
| FA3 | ❌ | Already have |
| Full-weight TTT SGD | ⚠️ Partially | We knew about TTT from andrewgcodes PR#4 but hadn't seen it with freeze-first-2-blocks specifically |
| NTK-RoPE base=50000 | ⚠️ | We knew NTK-RoPE exists but hadn't locked onto base=50000 as the value both top PRs use |
| Sliding window stride=64 | ❌ | Already have |

### Worth Implementing?
**TTT: LOW PRIORITY.** Only gives 0.0004 BPP improvement at 43s cost. The model barely learns anything in 3 epochs. The andrewgcodes approach (5 epochs, lr=0.003, all layers unfrozen) was more aggressive and claimed +0.006 BPP, but PR #254's results suggest TTT has diminishing returns on well-trained models.

**NTK-RoPE base=50000: VERIFY.** Both top PRs use this exact value. We should confirm we're using it.

---

## PR #257 — Autoresearch/Modal Sweep (ThomAub)
**Status: CLOSED**
**No body, no results, no submission.**

This is just a hyperparameter sweep done on Modal infrastructure. Not a real submission. **No useful information.**

---

## PR #265 — 11L + Efficient Partial XSA (unnir / vadim borisov)
**val_bpb: 1.1307** (single seed 1337)
**Artifact: 15,892,986 bytes (15.89 MB)**
**Steps: 6,976 at 86ms/step on 8×H100**
**Pre-quant BPB: 1.1437 → Post-quant sliding: 1.1307**

### Architecture & Techniques
| Component | Value |
|-----------|-------|
| Layers | 11 (dim=512, 8 heads, 4 KV heads GQA) |
| MLP | 3× expansion (1536 hidden), ReLU² |
| Quantization | Int8 (with percentile clipping at 99.99984%) — NOT int6 despite README claiming int6 |
| Compression | zstd-22 |
| SmearGate | ✅ |
| BigramHash | 2048 buckets, dim=128 |
| Init | Orthogonal + muP |
| Optimizer | Muon WD=0.04, momentum=0.99, warmup 0.92→0.99/1500 steps |
| SWA | Every 120 steps, 13 checkpoint average |
| Attention | FlashAttention 3 |
| RoPE | NTK-aware (train_seq_len=1024, auto-scales to 2048) |
| **XSA** | **Exclusive Self Attention on last 3 layers (arXiv:2603.09078)** |
| Eval | Sliding window stride=64 |

### ⭐ NOVEL TECHNIQUE: Exclusive Self Attention (XSA)

**Paper:** arXiv:2603.09078 (Shuangfei Zhai, 2026)

**Core Idea:** After computing attention output `y = softmax(QK^T)V`, subtract each token's
self-value projection. This forces the model to learn from CONTEXT rather than self-reference,
as standard attention has a bias toward copying a token's own value vector to its output.

**Implementation (lines 625-636):**
```python
def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
    """Subtract self-value projection via GQA-aware reshape (no repeat_interleave)."""
    B, T, H, D = y.shape
    Hkv = v.size(-2)
    group = H // Hkv
    # FREE reshape into KV head groups — no memory allocation
    y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
    vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D]
    # Project out self-value component
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, T, H, D)
```

**Integration (line 649-652):**
```python
y = flash_attn_3_func(q, k, v, causal=True)
# XSA: subtract self-value projection (deep layers only)
if self.use_xsa:
    y = self._xsa_efficient(y, v)
```

**Key Design Decisions:**
1. **Partial application**: Only applied to last 3 of 11 layers (XSA_LAST_N=3).
   Rationale: the paper shows self-attention bias (cosine similarity between output and
   self-value) *increases* across layers. Deeper layers need it more.
2. **Efficient GQA-aware**: Uses reshape+broadcast instead of repeat_interleave.
   Reduces overhead from ~7ms/step to ~2ms/step.
3. **Zero extra parameters**: Pure algorithmic change, no new weights.
4. **Compatible with FlashAttention**: Runs as post-processing after flash_attn output.

**Wiring in GPT.__init__ (lines 792-795):**
```python
if xsa_last_n > 0:
    for i in range(max(0, num_layers - xsa_last_n), num_layers):
        self.blocks[i].attn.use_xsa = True
```

### XSA Impact
PR #265 claims ~0.002 BPP improvement from XSA at <2ms/step cost.
This is difficult to isolate from the logs since there's no ablation without XSA,
but comparing to other 11L submissions with similar configs suggests it's real.

### SWA Cadence: every 120 steps → 13 checkpoints
Different from our every 50 steps / 24 checkpoints. With only 6976 steps, SWA starts
at step 5520 (warmdown start) and collects every 120 steps = 13 checkpoints.
This is a "quality over quantity" approach — fewer but better-converged checkpoints.

### What's NOVEL vs What We Know
| Technique | Novel? | Notes |
|-----------|--------|-------|
| **XSA (Exclusive Self Attention)** | ✅ **YES** | Completely new technique we've never seen or tried |
| Efficient GQA-aware XSA | ✅ **YES** | Novel implementation avoiding repeat_interleave |
| Partial XSA (last 3 layers only) | ✅ **YES** | Novel selective application strategy |
| SWA every 120 steps | ⚠️ Minor | We use every 50 — this is just different tuning |
| 11L MLP3x | ❌ | Already have |
| SmearGate + BigramHash | ❌ | Already have |
| Int8 + percentile clipping | ❌ | Already have |

### Worth Implementing?
**XSA: YES — HIGH PRIORITY.** 
- +0.002 BPP improvement
- <2ms/step overhead (~2.3% of step time)
- Zero extra parameters, zero artifact size increase
- Simple to implement (~20 lines of code)
- Completely orthogonal to everything else we do
- Can tune XSA_LAST_N (they use 3, could try 2-5)
- Works with torch.compile (fullgraph=True in their code)

---

## Summary: What to Steal

### Tier 1: Definitely implement
| Technique | Source | Expected Impact | Effort |
|-----------|--------|----------------|--------|
| **XSA on last 3 layers** | PR #265 | +0.002 BPP | ~20 lines, 1 hour |

### Tier 2: Verify/tune
| Technique | Source | Expected Impact | Effort |
|-----------|--------|----------------|--------|
| NTK-RoPE base=50000 | Both PRs | Verify we use this | Config check |
| SWA every 120 steps | PR #265 | ±0.0005 BPP | Config change |

### Tier 3: Low priority
| Technique | Source | Expected Impact | Effort |
|-----------|--------|----------------|--------|
| Full-weight TTT SGD | PR #254 | +0.0004 BPP (measured) | ~40 lines, needs 43s eval budget |

### Key Takeaway
**XSA is the only genuinely novel technique worth stealing.** PR #254's TTT gives negligible improvement (0.0004 BPP) on a well-trained 11L model — the model's loss barely changes over 3 SGD epochs, suggesting the val distribution is already well-modeled. Both PRs otherwise use the same stack we already have (11L, 3x MLP, SmearGate, BigramHash, OrthoInit, Muon WD=0.04, SWA, FA3, NTK-RoPE, sliding eval).

The real competitive edge at this point is **not about adding more techniques** but about **squeezing the most from existing ones**: better hyperparameter tuning, better SWA scheduling, and marginal gains from XSA-style algorithmic improvements that cost nothing in parameters or speed.
