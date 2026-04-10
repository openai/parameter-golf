# Parameter Golf — Strategy & Experiment Log

## Best Local Result: 1.803 BPB

**Config:** 2 shared blocks, 256d, MLP 3x, depth 6, per-layer scales, grad clip 1.0
**Data:** 90M tokens (fineweb_xl), **Steps:** 5984 in 9 min, **Params:** 1.45M

BPB trajectory (90M tokens, 256d MLP3x):
| Steps | BPB |
|-------|-----|
| 2000 | 2.144 |
| 4000 | 1.965 |
| 5984 | **1.803** (9-min wallclock cap) |
| 8000 | 1.783 (uncapped run) |

**Key scaling finding:** Tiny models (1-3M params) trained on lots of data with many steps beat larger models with fewer steps. At equal wall time on Apple Silicon, 256d/1.45M params dominates 384d/3M params.

---

## Core Strategy (Revised)

**Don't build recurrence as a standalone. Layer it onto the winning meta stack.**

The binding constraints are: 16MB artifact, 600s training on 8xH100, 600s eval budget.
The current best (1.1483) uses int6+MLP3x+SmearGate+BigramHash+MuonWD+SWA — no recurrence.
Recurrence alone (PR #148) only reached 1.2196. Our edge is recurrence + meta combined.

**Spend freed bytes on precision, not width.** Our local data confirms: wider models
(768d+) had *worse* post-quant BPB despite better pre-quant. Width taxes the training
FLOP budget (fewer steps in 10 min). Instead, spend saved bytes on:
1. Sensitive rows staying 8-bit / fp16 (FTLE-lite allocation)
2. Cheap repeat-specific modulators (repeat embeddings, tiny adapters)
3. Embeddings and control tensors at higher precision

## Build Order

### Phase 1: Port sharing into the winning stack
- Start from the strongest current submission (int6, MLP3x, SmearGate, BigramHash, sliding window eval, zstd-22, SWA, Muon WD)
- Add `NUM_UNIQUE_LAYERS=3` for 9 virtual layers at 512d
- Keep training compute identical to baseline — same speed, same steps, 1/3 params
- Measure: does sharing cost or gain BPB on the full stack?

### Phase 2: Cheap repeat asymmetry
Before touching width, add cheap symmetry breaking so each recurrence cycle does different useful work:
- **Repeat embedding**: small learned vector [num_layers, dim] added to block input at each virtual layer. Gives each cycle a "phase signal."
- **Rank-4 LoRA adapters**: tiny low-rank deltas on Q/V projections per virtual layer. Cost: 4 * dim * num_layers * 2 = ~36K params at 512d.
- **Bounded recurrence control**: replace unconstrained residual with a softmax-gated mixture:
  ```
  c = softmax([carry, anchor_x0, attn, mlp])
  x_next = c0*x + c1*x0 + tau*(c2*attn_out + c3*mlp_out)
  ```
  with tau < 1 or learned bounded scalar per block.

### Phase 3: Stability control + adaptive eval depth
- **Lyapunov delta regularizer**: penalize non-monotone or too-large ||x_{r+1} - x_r|| / ||x_r|| during training. Already validated locally — trained blocks become contractive (δ: 0.025→0.017 over 20 extra cycles).
- **Adaptive eval recurrence**: train 9 virtual layers, evaluate 12-15 when the window is still changing. Per-window halting rule: stop when δ < ε or logit entropy stabilizes.
- **RG diagnostic**: track δ_r and ρ_r = δ_{r+1}/δ_r to decide if extra depth is real or theater.

### Phase 4: FTLE-lite mixed precision
Spend saved bytes on **intelligent precision allocation** rather than uniform int6:
- During last 20-30% of training, compute row-group sensitivity via:
  - EMA of rowwise gradient norm
  - EMA of rowwise parameter path length
  - Variance across SWA snapshots
- Solve bit allocation: hot rows → 8-bit, middle → 6-bit, cold → 4-bit
- This is where sharing cashes out — fewer unique params means more byte slack for precision where it matters.

### Phase 5: Modest width increase
Only after phases 1-4 are validated:
- Try 576d or 640d (NOT 768+)
- Confirm the step-count hit is acceptable
- The priority is always post-quant BPB, not pre-quant

---

## Competition State (as of March 20, 2026)

| Rank | BPB | Author | Key Techniques |
|------|-----|--------|----------------|
| 1 | **1.1483** | raahilshah (PR #162) | Int6 + MLP3x + SmearGate + BigramHash + MuonWD + SWA |
| 2 | 1.1539 | unnir (PR #135) | OrthoInit + Int6 MLP3x + BigramHash + SmearGate |
| 3 | **1.1748** | notapplica (merged #1) | Sliding Window + FP16 Embed + 10L + Muon WD |
| — | 1.2196 | iverbovoy (PR #148) | Depth recurrence (3×4), dim=832, sliding window |
| — | 1.2244 | Baseline | 9L/512dim/1024vocab |

## The Dominant Meta (table stakes for top-10)

1. **Int6 quantization** — 6-bit per-row, frees ~25% artifact space
2. **MLP 3x expansion** — hidden = 3×dim, funded by int6 savings
3. **Sliding window eval** — stride 64, ~0.034 BPB improvement
4. **FP16 tied embedding** — don't quantize the shared embed matrix
5. **Zstd-22 compression** — tighter than zlib-9
6. **SmearGate + Bigram Hash** — token-pair context for ~0.005 BPB
7. **SWA** — checkpoint averaging during warmdown
8. **Muon WD** — weight decay 0.02, critical for scaling Muon
9. **Orthogonal init** — accelerates early convergence

---

## Local Experiment Results

### Layer Sharing (validated)

| Config | Params | Depth | Pre BPB | Post BPB |
|--------|--------|-------|---------|----------|
| Baseline (9 unique, 512d) | 17.1M | 9 | 3.079 | 3.157 |
| **3 shared, 512d** | **6.0M** | 9 | **3.074** | **3.151** |
| 3 shared, 640d, 12 depth | 8.5M | 12 | 3.034 | 3.174 |
| 3 shared, 768d, 12 depth | 12.6M | 12 | 3.054 | 3.208 |

Key: 512d shared **beats** wider shared configs post-quant. Width hurts.

### DEQ Convergence / Lyapunov Diagnostics (validated)

After 100 training steps, extra recurrence cycles show contraction:
```
Cycle  1: δ = 0.0253
Cycle 10: δ = 0.0206
Cycle 20: δ = 0.0167 — still decreasing, not yet converged
```
Contraction rate ~0.8%/cycle → max Lyapunov exponent ≈ -0.008 (barely stable).
At 50 steps, blocks were still expansive (δ increasing). Stability emerges with training.

### What We Learned
1. **Width is not the answer** — post-quant BPB degrades with width at this training budget
2. **Depth recurrence works** — matches/beats baseline at 1/3 params
3. **Blocks become contractive naturally** — no explicit regularization needed after sufficient training
4. **Extra eval depth gives real signal** — loss improved 5.395→5.331 with 20 extra cycles
5. **Per-layer scaling is negligible at 50 steps** — needs longer training or stronger asymmetry (repeat embeddings)

---

## Cross-Disciplinary Ideas (Prioritized)

### Tier 1: Implement Now
- **Bounded recurrence control** — softmax-gated carry/anchor/attn/mlp mixture with tau < 1
- **FTLE-lite row sensitivity** — EMA gradient norms for mixed-precision bit allocation
- **Repeat embeddings** — learned per-cycle phase signal, cheap symmetry breaking
- **Lyapunov delta regularizer** — penalize expansion, train toward edge of chaos

### Tier 2: After Phase 1-2
- **Adaptive eval halting** — stop recurrence when δ < ε per window
- **SWA snapshot variance for sensitivity** — use checkpoint spread as FTLE proxy
- **Nuclear norm regularizer** — encourage low-rank structure for compression

### Tier 3: Deprioritized
- **Kronecker weights** — saves bytes but loses training speed on dense H100 matmuls
- **Full implicit DEQ** — too much solver risk for 600-second budget
- **Symplectic optimizer** — not the current bottleneck
- **Wasserstein loss** — wrong metric (BPB scores exact tokens, not semantic nearness)

---

## Key PRs to Study

| PR | Score | Why |
|----|-------|-----|
| #162 | 1.1483 | Current best — the full meta stack to port sharing into |
| #135 | 1.1539 | Clean implementation of the meta |
| #148 | 1.2196 | Depth recurrence + cross-repeat skips (closest to our approach) |
| #39 | — | Int6 quantization origin |
| #50 | — | Sliding window eval origin |
| #102 | — | SmearGate + BigramHash origin |

---

## Files

| File | Purpose |
|------|---------|
| `train_gpt_mlx_exp.py` | MLX script: layer sharing + per-layer scaling + sliding window + DEQ eval + Lyapunov diagnostics + nuclear norm + Kronecker (experimental) |
| `train_gpt_submission.py` | CUDA script: layer sharing + per-layer scaling + Muon WD + label smoothing + eval knobs |
| `make_mini_shards.py` | Creates ~1MB data subset for local Mac testing |
| `EXPERIMENTS.md` | This file |
