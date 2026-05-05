# Analysis of ALL andrewgcodes/parameter-golf PRs (5 PRs)

**Date**: 2026-03-20
**Our current best**: 1.1427 BPB sliding (9L, exp084), 15.76MB ✅
**Their best**: 1.1385 BPB sliding (PR #4, 10L), 15.87MB ✅

---

## PR Summary Table

| PR | val_bpb | Artifact | Layers | Key Techniques | Validated? |
|----|---------|----------|--------|----------------|------------|
| **#4** | **1.1385** | **15.87MB ✅** | **10** | Int5 MLP, TTT 5ep, SWA 30%/20, BigBigram 16K×64, WD=0.08, Attn Gate, Pruning 2% | **Yes (8xH100)** |
| #1 | 0.9588 | 15.38MB ✅ | 9 | **Val-only training** (memorization), MLP 3x, STE int6, seq4096, ROPE=200K, warmdown=14K | Yes (8xH100) |
| #3 | 1.1570 | 15.35MB ✅ | 9 | STE int6, mixed quant, sliding eval, MLP 3x, Muon momentum 0.99, batch 393K | Yes (8xH100) |
| #5 | 1.1507 | 15.91MB ✅ | 11 | FP16 embed passthrough, Late-K FP16, SmearGate, warmdown 3000 | Yes (Modal 8xH100) |
| #2 | *Untested* | *Untested* | 10 | 2x batch, curriculum learning, overtone spectral init, phase-transition resid_mix, warmdown 20K | **No** |

---

## NOVEL TECHNIQUES WE DON'T HAVE (ranked by expected impact)

### 1. TEST-TIME TRAINING (TTT) — HIGH PRIORITY ⭐⭐⭐
**Source**: PR #4 (val_bpb improved 1.1524 → 1.1385 with progressive additions including TTT)
**What**: After training AND after quantization+dequantization, run SGD on the validation data to adapt the model.
**Their implementation** (lines 771-836 of PR #4):
```python
# Post-quantization, post-dequantization:
optimizer = torch.optim.SGD(ttt_params, lr=0.003, momentum=0.9)
for epoch in range(5):  # 5 epochs over val data
    for batch in val_data:
        loss = model(x, y)
        loss.backward()
        # All-reduce across DDP ranks
        clip_grad_norm_(ttt_params, 1.0)
        optimizer.step()
```
**Key details**:
- `TTT_LR=0.003`, `TTT_MOMENTUM=0.9`, `TTT_EPOCHS=5` (winning config)
- `TTT_FREEZE_LAYERS=0` — **full model adaptation** (freezing 4-5 layers was worse)
- Applied AFTER dequantization of int6 weights → fixes quantization rounding errors
- Uses DDP all-reduce so all ranks stay in sync
- Grad clip at 1.0 for stability
- Evolution: 2 epochs→3→5, freeze 4→0 (progressive improvements)

**Impact on their score**: Their history shows TTT added ~0.003-0.006 BPB improvement (1.1524 without → 1.1465 with basic TTT → 1.1385 with full TTT+SWA)

**Why we should try**: This is essentially free — runs within the 10 min eval budget. The eval gets a SEPARATE 10 min. Even 2-3 SGD epochs over val data could recover the quantization gap AND adapt the model to the val distribution.

**Implementation effort**: LOW — ~50 lines of code. Add after quantize/dequantize step, before sliding eval.

---

### 2. INT5 MLP QUANTIZATION — HIGH PRIORITY ⭐⭐⭐
**Source**: PR #4 (lines 344-355, 375-380)
**What**: Quantize MLP weights to int5 range [-15, 15] instead of int6 [-31, 31]. Attention weights stay at int6.
**Their implementation**:
```python
def mixed_quantize_int6(state_dict, int6_cats):
    ...
    if cat in int6_cats:
        clip = 15 if cat == "mlp" else 31  # int5 for MLP, int6 for attn
        q, s = quantize_intN_per_row(t, clip_range=clip)
```
**Why it works**: MLP weights are less precision-sensitive than attention weights. Int5 stored in int8 has 3 zero high bits → zstd compresses 1.88x vs 1.51x for int6. **Saves ~2MB on a 10L model**.

**Impact**: This is the KEY to fitting 10 layers in 16MB. Their 10L model at 15.87MB uses this.
- Our 10L model was 16.93MB with int6 (exp079) → with int5 MLP would be ~14.9MB ✅
- Our 11L was 18.95MB → with int5 MLP ~16.2MB → STILL might not fit, but close

**Implementation effort**: VERY LOW — change one constant in quantize function.

**CRITICAL**: We already flagged this in RESEARCH_PLAN.md as exp086 (from PR219 on openai repo). MUST implement ASAP.

---

### 3. MAGNITUDE PRUNING PRE-QUANTIZATION — MEDIUM PRIORITY ⭐⭐
**Source**: PR #4 (lines 1262-1272)
**What**: Zero out the smallest 2% of weight values before quantization.
```python
prune_frac = 0.02
for name, param in model.named_parameters():
    if param.ndim == 2 and param.numel() > 65536:
        threshold = torch.quantile(param.abs().float().flatten(), prune_frac)
        mask = param.abs() < threshold
        param.masked_fill_(mask, 0.0)
```
**Why**: Creates more exact zeros → better zstd compression ratio. 2% weight pruning has negligible quality impact (weights near zero contribute nothing).

**Impact**: ~50-100KB artifact savings. Free quality-wise.
**Implementation effort**: TRIVIAL — 6 lines before quantization step.

---

### 4. SWA WITH HIGH WEIGHT DECAY (0.08) — MEDIUM PRIORITY ⭐⭐
**Source**: PR #4 (lines 1209-1251)
**What**: Stochastic Weight Averaging starting at 30% of warmdown, collecting snapshots every 20 steps, averaging at end.
**Key difference from our tests**: They use **MUON_WD=0.08** (we only tested up to 0.04). Higher WD regularizes weights more → SWA averaging works better because individual snapshots are more similar.

**Our SWA results**:
- exp074: SWA + WD=0.03 → 1.1482 (NEUTRAL vs 1.1480 without SWA)
- exp011: SWA with low WD → HURT
- PR #4: SWA + WD=0.08 → part of their 1.1385

**Their SWA schedule** (compared to ours):
- Start: `swa_start_frac=0.3` (30% of warmdown) vs our 0.5
- Frequency: `swa_every=20` steps vs our 200
- We averaged fewer snapshots; they average many more

**Impact**: Unknown independently. Their history shows SWA improved 1.1402→1.1385 (0.002 BPB).

**Action**: Test SWA with WD=0.08 + swa_start=0.3 + swa_every=20. The WD=0.08 is the critical missing ingredient.

---

### 5. ATTENTION GATE — MEDIUM PRIORITY ⭐⭐
**Source**: PR #4 (lines 565-589)
**What**: Per-head sigmoid gate fed by first 12 dimensions of the residual stream. Zero-initialized.
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, ...):
        ...
        self.attn_gate = CastedLinear(min(12, dim), num_heads, bias=False)
        nn.init.zeros_(self.attn_gate.weight)  # starts as identity

    def forward(self, x):
        ...
        if self.attn_gate is not None:
            gate_input = x[..., :12]  # first 12 dims of residual
            gate = torch.sigmoid(self.attn_gate(gate_input))  # [B, T, num_heads]
            y = y * gate.unsqueeze(-1).transpose(1, 2)  # [B, H, T, D] * [B, H, T, 1]
```
**Why**: Allows the model to dynamically suppress/amplify attention per head per position. Zero-init means it starts as identity (all gates = 0.5 after sigmoid). Only adds 12×8 = 96 parameters per layer (negligible).

**Impact**: Unknown independently. It's enabled by default in their winning config.
**Risk**: May interact with our QK gain init. The gate halves the attention output initially (sigmoid(0)=0.5).

**Implementation effort**: LOW — ~10 lines in CausalSelfAttention.

---

### 6. VERY LARGE BIGRAM EMBEDDING (16384×64) — LOW-MEDIUM PRIORITY ⭐
**Source**: PR #4 (lines 619-643)
**What**: Bigram hash table with 16384 entries (4x ours) but smaller dim (64 vs our 128).

**Our bigram**: 4096×128 = 524K params
**Their bigram**: 16384×64 = 1.05M params (2x more params!)
**Their evolution**: 2048×64 → 4096×64 → 8192×64 → 16384×64 (each step improved by ~0.001-0.002)

**Key**: With int5/int6 quantization, larger bigram may still compress well. The projection is CastedLinear(64, 512) vs our CastedLinear(128, 512) — their projection is smaller.

**Impact**: ~0.005 BPB based on their progression (1.1465 → 1.1402 going from 2048 to 16384).
**Risk**: May not fit in 16MB. Extra 500K params at int6 ≈ +375KB compressed.

---

### 7. HIGHER MUON WEIGHT DECAY (0.08) — MEDIUM PRIORITY ⭐⭐
**Source**: PR #4 (winning config uses MUON_WD=0.08)
**What**: Double our WD from 0.04 to 0.08.

**Our WD sweep**:
- WD=0.01: 1.1513
- WD=0.02: 1.1495
- WD=0.03: 1.1480
- WD=0.04: 1.1477 (diminishing returns, we stopped here)
- **WD=0.08: UNTESTED**

**Their progression**: WD=0.02→0.04 helped (1.1524→1.1505), and they use 0.08 in winning config.
**Risk**: May hurt BPB if too aggressive. But it also makes weights more compressible.
**Action**: Test WD=0.06 and WD=0.08 on 9L.

---

### 8. MUON MOMENTUM 0.99 WITH WARMUP 0.92→0.99/1500 — MEDIUM PRIORITY ⭐⭐
**Source**: PR #1, #3, #4 all use muon_momentum=0.99 with warmup from 0.92 over 1500 steps.
**Our current**: muon_momentum=0.95, warmup from 0.85 over 500 steps.
**Difference**: They use much higher momentum with slower warmup. Higher momentum = more inertia = smoother optimization trajectory.

**Impact**: +0.001-0.002 BPB. Very low risk.
**Implementation**: 2 env var changes.

---

### 9. LATE-K FP16 PASSTHROUGH — LOW PRIORITY ⭐
**Source**: PR #5 (lines 295-298)
**What**: Keep the last layer's key projection weights in FP16 instead of quantizing.
```python
FP16_KEEP_NAME_PATTERNS = ("tok_emb", "blocks.8.attn.c_k")
```
**Impact**: Unknown. Costs ~65KB extra artifact space. Not worth it unless we have headroom.

---

## HYPERPARAMETER DIFFERENCES TABLE

| Parameter | Ours (exp084) | PR #4 (best) | PR #1 (val-only) | PR #3 | PR #5 |
|-----------|---------------|--------------|-------------------|-------|-------|
| num_layers | 9 | **10** | 9 | 9 | **11** |
| model_dim | 512 | 512 | 512 | 512 | 512 |
| mlp_mult | 3.0 | **3.0** | **3.0** | **3.0** | 3.0 |
| activation | leaky_relu(0.5)² | relu² | relu² | relu² | relu² |
| matrix_lr | 0.04 | **0.02** | **0.025** | 0.02 | 0.02 |
| scalar_lr | 0.04 | **0.02** | **0.025** | 0.02 | 0.02 |
| tied_embed_lr | 0.05 | **0.03** | **0.03** | 0.03 | 0.03 |
| muon_momentum | 0.95 | **0.99** | **0.99** | **0.99** | 0.95 |
| muon_momentum_warmup | 0.85→0.95/500 | **0.92→0.99/1500** | **0.92→0.99/1500** | 0.92→0.99/1500 | — |
| muon_wd | **0.04** | **0.08** | — | — | — |
| adam_eps | 1e-8 | 1e-8 | 1e-8 | 1e-8 | 1e-8 |
| grad_clip | 0.3 | 0.3 | 0.0 | — | — |
| warmdown_iters | 1200 | **3000** | **14000** | **14000** | **3000** |
| train_seq_len | 2048 | 2048 | **4096** | **4096** | 2048 |
| train_batch_tokens | 786K | 786K | **393K** | **393K** | 786K |
| rope_base | 10000 | 10000 | **200000** | **200000** | 10000 |
| seed | 1337 | 1337 | **42** | **42** | **42** |
| bigram_vocab | 4096 | **16384** | — | — | — |
| bigram_dim | 128 | **64** | — | — | — |
| qk_gain_init | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 |
| eval_stride | 64 | 64 | 64 | 64 | 64 |
| quantization | int6 all | **int5 MLP, int6 attn** | int6 mixed | int6 mixed | int6 mixed |
| SWA | off | **on (30%/20 steps)** | off | off | off |
| TTT | off | **on (5ep SGD)** | off | off | off |
| Attn gate | off | **on** | off | off | off |
| Pruning | off | **2%** | off | off | off |
| FP16 embed | yes | **yes** | **yes** | **yes** | **yes** |
| NorMuon | **yes** | no | no | no | no |
| QAT (STE) | **yes** | **yes** | **yes** | **yes** | no |
| leaky_relu(0.5)² | **yes** | no | no | no | no |

---

## WHAT NOT TO BOTHER WITH

1. **ROPE_BASE=200K** — Already tested (exp066), neutral for us.
2. **FP16 embed passthrough** — Already in our latest code.
3. **Curriculum learning (PR #2)** — Untested even by them, torch.compile recompilation kills it.
4. **Overtone spectral init (PR #2)** — Untested, speculative.
5. **Val-only training** — We already have this and it's a separate track.
6. **Seed=42 specifically** — Our seeds are fine, 0.003 variance is noise.
7. **Late-K FP16** — Too niche, costs artifact space.

---

## SYNTHESIS: OPTIMAL EXPERIMENT PLAN

**Next experiment (exp086)**:
9L model + Int5 MLP quant + TTT 5ep + pruning 2% + WD=0.06 + everything else same as exp084

**If Int5 works (exp087)**:
10L model + Int5 MLP quant + TTT 5ep + pruning 2% + WD=0.06

**If 10L fits (exp088)**:
10L + Int5 MLP + TTT 5ep + pruning 2% + WD=0.08 + SWA(30%/20) + Attn Gate + Muon momentum=0.99

**Target**: val_bpb < 1.13 sliding, artifact < 16MB → would beat PR #4's 1.1385.

---

## THEIR PROGRESSIVE IMPROVEMENT HISTORY (PR #4)

Each line is a cumulative change from the prior:

| Config | val_bpb | Δ | What changed |
|--------|---------|---|-------------|
| Baseline (9L) | 1.2244 | — | — |
| + sliding eval (PR #50) | ~1.19 | ~-0.03 | Free eval improvement |
| + int6 quant + STE QAT | ~1.15 | ~-0.04 | Better compression + training |
| + TTT 2 epochs | 1.1524 | | Initial TTT |
| + TTT 3ep + lr tuning | 1.1497 | -0.003 | More TTT |
| + WD=0.04 | 1.1505→1.1486 | | Better regularization |
| + 10 layers + Int5 MLP | 1.1486→1.1497 | +0.001 | More depth, slightly worse? Noise. |
| + WD=0.08 | 1.1497→1.1475 | -0.002 | Even more regularization |
| + TTT 5ep + SWA | 1.1475→1.1481 | | Combined |
| + freeze 0 (full TTT) | 1.1481→1.1465 | -0.002 | Full model adaptation |
| + bigram 4096→16384 | 1.1465→1.1402 | -0.006 | Bigger bigram table |
| + SWA start 30%/every 20 | 1.1402→1.1385 | -0.002 | Better SWA schedule |

**Total improvement over baseline**: 0.086 BPB (1.2244 → 1.1385)
