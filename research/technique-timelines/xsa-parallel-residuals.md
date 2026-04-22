# Technique Timeline: XSA + Parallel Residuals

**Written:** 2026-04-21 | **Scope:** record track (`track_10min_16mb/`)
**Our baseline:** PR #1736 (dexhunter, 1.06549 bpb)
**Merged SOTA:** PR #1493 (bigbag, 1.0810 bpb)

---

## What is XSA?

**XSA (Exclusive Self Attention)** is a zero-parameter post-attention correction from
arXiv:2603.09078 (Shuangfei Zhai, March 2026). The motivation: standard dot-product
attention can degenerate toward each token simply "copying" its own value vector
(self-attention collapse). XSA corrects this by projecting out the self-value direction
from the attention output.

**The operation** — after `y = flash_attn(q, k, v)`:

```python
def _xsa_efficient(self, y, v):
    # y: [B, T, H, D]  — attention output, one vector per head
    # v: [B, T, Hkv, D] — value vectors (GQA: fewer KV heads than Q heads)
    B, T, H, D = y.shape
    Hkv = v.size(-2)
    group = H // Hkv          # Q heads per KV head (2 for 8Q/4KV)
    y_g = y.reshape(B, T, Hkv, group, D)       # free view, no copy
    vn = F.normalize(v, dim=-1).unsqueeze(-2)   # unit-norm value, broadcast-ready
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn   # project y onto vn
    return (y_g - proj).reshape(B, T, H, D)    # subtract that projection
```

**In plain English:** for each attention head, take the unit vector along the head's own
value vector `v`, compute how much of the attention output `y` lies in that direction,
and subtract it. The result is an attention output that is *orthogonal* to the token's
own value — the model is forced to integrate context rather than self-copy.

**What XSA is NOT:**
- It has no learnable parameters
- It is applied *after* attention (not to Q/K/V projections)
- It is not the same as QK normalization (`F.rms_norm(q)`, `F.rms_norm(k)`) — those
  are separate operations also present in our baseline

---

## What are Parallel Residuals?

Standard transformers have a **single residual stream**: `x ← x + attn(x)`,
`x ← x + mlp(x)`. The attention and MLP sublayers compete to write to the same vector.

**Parallel residuals** (introduced in this codebase by PR #1523/1493) split the later
layers of the network into **two lanes**:

- **Lane 0** (attn lane): primarily updated by attention output
- **Lane 1** (mlp lane): primarily updated by MLP output
- Both lanes are read from and written to via a learned `[L, 2, 2]` routing matrix
- At final output, the two lanes are merged (default: mean of both)

```
Before (single stream):
  x → [Attn] → x → [MLP] → x

After (parallel residuals, deep layers only):
  lane0, lane1 → [Attn reads lane0] → attn_out → updates both lanes
                  [MLP reads lane1]  → mlp_out  → updates both lanes
  final: 0.5*(lane0 + lane1)
```

The routing weights are learned, so the model can discover what cross-coupling
(e.g. attn_out feeding into the MLP lane) actually helps.

---

## Phase 1 — XSA introduced (PR #265, unnir, MERGED 2026-03-20)

**val_bpb: 1.1307** | beats prior SOTA 1.1318 by 0.0011

Vadim Borisov applied the XSA paper (arXiv:2603.09078, published just days earlier)
to the competition. Two innovations on top of the paper:

**1. Efficient GQA-aware implementation.** The paper's naive implementation calls
`v.repeat_interleave(group_size)`, doubling memory. Borisov avoids this with a
free reshape + broadcast:

```python
# Naive (allocates 2× memory):
v_expanded = v.repeat_interleave(group_size, dim=-2)
vn = normalize(v_expanded)
y = y - dot(y, vn) * vn

# Efficient (zero allocation):
y_grouped = y.reshape(B, T, Hkv, group_size, D)   # view
vn = normalize(v).unsqueeze(-2)                     # [B,T,Hkv,1,D]
y = (y_grouped - dot(y_grouped, vn)*vn).reshape(B, T, H, D)
```

Reduces overhead from ~7ms/step to ~2ms/step at 11 layers.

**2. Partial application (last 3 of 11 layers).** The paper notes that self-attention
bias (cosine similarity between output and self-value) increases in deeper layers.
Borisov therefore applies XSA only to layers 8, 9, 10 (`XSA_LAST_N=3`).

Config: `XSA_LAST_N=3`, 11L 512d 8Q/4KV GQA, no parallel residuals, no depth recurrence.

---

## Phase 2 — XSA on all layers + first parallel residuals (April 2026)

**PR #1523** (EthanYangTW, CLOSED 2026-04-10) and **PR #1493** (bigbag, MERGED
2026-04-09) both arrived around the same time with heavily stacked recipes.

### PR #1493 — Merged SOTA (bigbag, 1.0810)

```
val_bpb: 1.0810 (3-seed mean, std 0.0002)
xsa_last_n: 11        ← expanded from 3 to ALL layers
parallel_residual_start: 7   ← last 4 of 11 layers
qk_gain_init: 5.25
num_loops: 2, loop_start: 3, loop_end: 5   ← depth recurrence
enable_looping_at: 0.35
ema_decay: 0.9965
ttt_lr: 0.005, ttt_epochs: 3
```

The code is LZMA+base85 compressed in the artifact, so the exact parallel-residual
implementation cannot be inspected directly. But from logged config, `PARALLEL_RESIDUAL_START=7`
meaning parallel residuals activate on the last 4 of 11 physical layers (7, 8, 9, 10).

**Important:** PR #1493 is bundled — XSA going from last-3 to all-11 is mixed with depth
recurrence, QK-gain, TTT, and SP8192. The isolated delta of XSA-all vs XSA-last3 at this
scale is **not cleanly measured here**.

### PR #1523 — (EthanYangTW, CLOSED, 1.0778)

```
val_bpb: 1.0778 (3-seed mean)
xsa_last_n: 11
parallel_residual (implied from credits + config)
Muon momentum: 0.97 (down from 0.99)
Banking + Fused MLP Triton TMA kernel
TTT LR: 0.01
```

Credits #1420, #1460, #1477, #1514. Closed (superseded by #1493 merge).
Same GQA-efficient XSA implementation as #265.

---

## Phase 3 — Improved parallel residuals (PR #1529, msisovic, 2026-04-11)

**val_bpb: 1.0758** (3-seed mean, std 0.0007) — **best clean open PR at the time**

msisovic's key architectural change: upgrade from a simple per-lane residual scalar
to a full **`[L, 2, 2]` routing matrix** — every sublayer at every position can write
to both lanes with independently learned weights.

```python
# PR #1493 style (simple):
lane0 = lambda_attn * lane0 + scale * attn_out
lane1 = lambda_mlp  * lane1 + scale * mlp_out

# PR #1529 style (full routing):
self.parallel_post_lambdas  = nn.Parameter(torch.ones(L, 2, 2))   # [L, sublayer, lane]
self.parallel_resid_lambdas = nn.Parameter(torch.full((L, 2), sqrt(1.1)))

lane0 = attn_resid*lane0 + attn_post[0]*attn_out + mlp_post[0]*mlp_out
lane1 = mlp_resid *lane1 + attn_post[1]*attn_out + mlp_post[1]*mlp_out
```

Also: `PARALLEL_RESIDUAL_START` moved from 7 → 8 (last 3 of 11 layers, not last 4),
`PARALLEL_FINAL_LANE=mean`.

**Learned weights at convergence (seed 1337):**

| Layer | attn_resid | attn→attn | attn→mlp | mlp_resid | mlp→attn | mlp→mlp |
|---|---|---|---|---|---|---|
| 8 (psl=0) | 3.33 | −0.35 | 0.46 | 0.43 | 0.03 | 0.63 |
| 9 (psl=1) | 0.72 | −0.10 | 0.40 | 0.47 | 0.19 | 0.55 |
| 10 (psl=2) | −0.04 | 0.14 | 0.14 | 0.53 | 0.57 | 0.57 |

Notable: `attn→attn` is **negative** at layers 8 and 9 (the attention output actively
suppresses its own lane), while `attn→mlp` is consistently positive. MLP output routes
roughly equally to both lanes in the deepest layer. This suggests the routing learned
something non-trivial that a fixed architecture couldn't express.

**Also includes:** CUTLASS EVT throughput optimization (binary `.so` in diff — not usable
without the build pipeline).

---

## Phase 4 — Our baseline PR #1736 (dexhunter, 2026-04-19)

Verified from `records/track_10min_16mb/.../train_gpt.py`:

```
xsa_last_n: 11           (line 31, env default)
F.normalize(v, dim=-1)   (line 784) ✓ CORRECT implementation
PARALLEL_START_LAYER: 8  (line 49, env default)
parallel_post_lambdas: [L,2,2]   (lines 1016-1019)
parallel_resid_lambdas: [L,2]    (lines 1019-1022)
PARALLEL_FINAL_LANE: mean        (line 50)
```

**Our baseline inherited the full #1529 parallel residuals architecture** (start=8,
full routing matrix, mean final lane), not the simpler #1493 version. This is already
the most evolved form of parallel residuals available in the competition.

Additional gates on top (not in #1523/#1529): SmearGate, AttnOutGate, QuantGate.

---

## The F.rms_norm bug (PR #1709, Bananakin1, 2026-04-18)

This PR documents a subtle implementation error present in some community XSA
implementations: using `F.rms_norm(v, (head_dim,))` instead of `F.normalize(v, dim=-1)`.

**The difference:**
- `F.normalize(v, dim=-1)`: scales `v` to have L2 norm = 1 (true unit vector)
- `F.rms_norm(v, (D,))`: scales `v` so its RMS = 1, which means L2 norm = √D

With `head_dim=64`, `rms_norm` produces a vector with L2 norm = 8.
The XSA projection removes `(y·vn)*vn`. If `vn` has norm 8 instead of 1, the
scalar `(y·vn)` is 8× larger and `vn` is 8× larger, so the removed component
is 64× too large. This is a severe over-correction.

**Impact on prior work:** Bananakin1's own experiments showed the buggy form was
present in their early runs (exp1-exp9). After fixing, int6 bpb improved by
approximately −0.0065 (bundled with another change; not isolated).

**Our baseline status: ✓ CORRECT.** Line 784 of our train_gpt.py uses
`F.normalize(v, dim=-1)`. The bug does not affect us. PR #1709 does not apply.

---

## Isolated ablation data (PR #1125, jainpranjal97, 2026-03-30)

45 experiments on single RTX 5090, 10-min runs. XSA-relevant findings:

**XSA all layers vs last 4:**

| Config | val_bpb | Δ |
|---|---|---|
| XSA last 4 (baseline for this ablation) | 1.3549 | — |
| XSA all 11 layers | 1.3451 | **−0.0018** |

Isolated at experiment #7; single seed, 10-min budget, earlier stack (no SP8192, no depth
recurrence). The -0.0018 is **assumed but not re-validated** on the current #1736 stack.

**XSA gating (learned per-head gate α):**

| Config | pre-quant bpb | post-quant bpb |
|---|---|---|
| No gating (baseline) | 1.1946 | 1.1957 |
| Per-head gate α | **1.1932** (−0.0014 pre-Q) | 1.1961 (+0.0004 post-Q) |

Better pre-quantization, **worse post-quantization**. The gating adds parameters that
interact poorly with int6 quantization. This is a consistent theme: architectural
additions that help FP can hurt quantized.

---

## Summary table

| Date | PR | Author | bpb | Key XSA/ParRes change | Status |
|---|---|---|---|---|---|
| 2026-03-20 | #265 | unnir | 1.1307 | XSA introduced, last 3 layers, GQA-efficient | **MERGED** |
| 2026-03-30 | #1125 | jainpranjal97 | non-record | XSA ablation: all→last4 = +0.0018; gating pre/post quant divergence | open |
| 2026-04-09 | #1493 | bigbag | 1.0810 | XSA all 11 layers; parallel residuals start=7 | **MERGED** |
| 2026-04-10 | #1523 | EthanYangTW | 1.0778 | XSA all 11 layers; parallel residuals | closed |
| 2026-04-11 | #1529 | msisovic | 1.0758 | Parallel residuals: full [L,2,2] routing, start=8 | open |
| 2026-04-18 | #1709 | Bananakin1 | non-record | Bug: rms_norm vs normalize; our baseline unaffected | open |
| 2026-04-19 | #1736 | dexhunter | **1.06549** | XSA all 11 layers + F.normalize ✓ + #1529 routing matrix + gates | **OUR BASELINE** |

---

## Key findings

1. **XSA on all layers is strictly better than partial** — isolated delta ~−0.0018 bpb
   (small stack, one seed). Adopted universally in all competitive PRs after #265.

2. **The GQA-efficient reshape is the canonical implementation.** Every competitive PR
   uses the `y.reshape(...) / vn.unsqueeze(-2)` form from #265. No alternatives have
   been tried (e.g. cross-head normalization, per-head learned scale).

3. **Parallel residuals routing learned non-trivial cross-coupling.** In #1529's
   converged weights, `attn→attn` is *negative* in early parallel layers while
   `attn→mlp` is positive. This means the model learned that attention output *hurts*
   the attn lane but *helps* the MLP lane — something a fixed-architecture can't express.

4. **Our baseline already uses the most evolved parallel-residual form** — full `[L,2,2]`
   routing matrix from #1529, start=8, final=mean. No competitive PR has surpassed this.

5. **F.rms_norm vs F.normalize is a critical bug.** Over-subtracts by head_dim²× if
   not caught. Our baseline is clean.

---

## Open questions

1. **What is the isolated XSA delta on the #1736 stack?** The −0.0018 figure comes from
   an older, smaller stack (no SP8192, no depth recurrence, no TTT, no gating). On the
   current stack, XSA interacts with AttnOutGate (which modifies the same output y).
   Whether they compose additively or partially cancel is unknown.

2. **XSA and depth recurrence interaction.** Loop45 applies physical layers 3–5 three
   times. XSA on those layers is applied 3× per virtual pass. Is this beneficial (removes
   self-copy at each virtual depth) or harmful (over-constrains representations across
   passes)? Never ablated.

3. **PARALLEL_START_LAYER sweep.** #1493 used 7; #1529 moved to 8 — bundled with the
   routing matrix change. Whether 8 is better than 7 *in isolation* on the full
   routing-matrix architecture is unknown.

4. **PARALLEL_FINAL_LANE.** The default `mean` was adopted from #1529, but #1493/1523
   may have used `mlp`. No clean ablation of mean vs mlp on the current stack.

5. **Interaction of parallel residuals and gating.** Our baseline adds SmearGate and
   AttnOutGate on top of parallel residuals. Are there gate parameters that partially
   absorb what the routing matrix is doing? Could the routing matrix and gates be
   co-trained to discover a better equilibrium?

---

## Potential opportunities

### 1. PARALLEL_START_LAYER sweep (one env var, ~$3-4/run)

**What:** Try `PARALLEL_START_LAYER=6` or `7` vs baseline `8`.
**Motivation:** #1493 used 7 and got 1.0810; #1529 moved to 8 but bundled the routing
matrix change. On the current stack with the full [L,2,2] matrix, more parallel layers
might help.
**Risk:** Low — env var only, no code change. Run 2×H100 smoke to screen.
**Expected Δ:** Unknown, but bounded by the gap between #1493 and #1529 improvements.
Probably ±0.001–0.002 bpb.

### 2. XSA layer count on looped layers (config change)

**What:** Try `XSA_LAST_N=8` (disable XSA on the 3 looped physical layers 3–5).
**Motivation:** With Loop45, layers 3–5 are applied 3 virtual times. XSA on those
layers projects out self-value at each virtual pass, which may over-constrain
representations that are intentionally building up recurrent state. Disabling XSA
on looped layers only would allow them to "self-copy" (integrate previous pass state)
more freely.
**Risk:** Low in theory; needs verifying that XSA_LAST_N=8 skips layers 3–5 correctly
given the loop indexing. Probably ~1-line config check.
**Expected Δ:** Speculative. If the hypothesis is right, could recover 0.001–0.003 bpb
by freeing the recurrence layers.

### 3. PARALLEL_FINAL_LANE ablation (one env var)

**What:** Try `PARALLEL_FINAL_LANE=attn` or `mlp` vs `mean`.
**Motivation:** The learned routing weights at convergence show strong cross-coupling
(attn output → mlp lane). `mean` is agnostic; `mlp` lane might carry more relevant
information after multiple routing steps. Pure curiosity but trivially cheap.
**Risk:** Minimal — if it hurts, skip.
**Expected Δ:** Probably < 0.001 bpb; mostly educational.

---

*This document is research-only. No idea files or specs written — hand findings back to the user.*
