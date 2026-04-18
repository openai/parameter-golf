# SwiGLU activation (replace LeakyReLU²)

**Status:** candidate
**Expected Δ:** +0.002 to +0.005 bpb (untested on our stack; claimed +0.0041 on a single-GPU 5090 base in the source submission)
**Source:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`. Non-record submission; SwiGLU was one of several bundled changes, isolated Δ claim is from the README's ablation.

## Idea
SOTA uses LeakyReLU² as the MLP activation: `proj(leaky_relu(fc(x), 0.5).square())` (`train_gpt_sota.py:386`). SwiGLU is a gated variant widely used in modern LLMs (Llama, PaLM): `proj(SiLU(gate(x)) * up(x))`. The gate mechanism lets the network learn which features to activate conditionally on input, rather than a fixed nonlinearity.

## Why it might help
- Proven in production LLMs (Llama 1/2/3, PaLM, Chinchilla).
- The gating signal gives the MLP extra expressive power per parameter.
- Orthogonal to every other feature in our stack — replaces a single pointwise op in the MLP block.

## Code-change sketch
Replace the MLP class (`train_gpt_sota.py` around L382-386):

```python
# Current: LeakyReLU²
class MLP(nn.Module):
    def __init__(self, dim, hidden):
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), 0.5).square())

# Proposed: SwiGLU
class MLP(nn.Module):
    def __init__(self, dim, hidden):
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.up(x))
```

Add an env toggle (e.g. `ACTIVATION=swiglu` vs `leaky_square`).

## Budget impact — the catch
SwiGLU needs **3 linear layers** (gate, up, proj) vs LeakyReLU²'s **2** (fc, proj). At MLP ratio 4× on 512d, each extra linear is 512×2048 = 1M params per layer × 11 layers = **~11M extra params**.

At INT6 that's ~8.25 MB pre-Brotli. Even assuming 2× Brotli compression, that's ~4 MB post — **totally blows the 16MB budget** (we have ~10KB of headroom).

To make SwiGLU fit, we'd need to cut MLP ratio substantially:
- MLP ratio 4× → 3× would save ~25% of MLP params, roughly offsetting SwiGLU's +50% per-layer cost. Net: slightly larger than baseline. Still over budget.
- MLP ratio 4× → 2.67× would roughly break even on param count. Significant capacity cut.
- Or: use SwiGLU in only some layers (e.g., the 3 looped layers only — saves params since those weights are shared across passes).

Same class of problem as BigramHash: architecturally interesting but budget-constrained.

## Risks / open questions
- **16MB budget fit.** Must pair with an MLP ratio cut or a restricted SwiGLU-on-some-layers scheme.
- **Interaction with depth recurrence.** Our layers 3, 4, 5 each get used 3× per forward pass. A SwiGLU in a looped layer means 3× more gate-matmul per step. Compute overhead is bigger than it looks from param count.
- **Interaction with LeakyReLU² tuning.** SOTA has probably tuned other features (QK gain, init stds) around the current activation's range. SwiGLU has different output statistics — may need retuning.
- **Training dynamics.** SwiGLU is known to need slightly longer training to surpass ReLU family. In our 10-min budget, it may look worse early and only pull ahead near the end.
- **Ablation: "SiLU-only" variant** — use `SiLU(fc(x))` (no gating, same param count as LeakyReLU²). Tests whether the activation shape or the gating is the active ingredient. Cheaper screen.

## Screen plan
Similar shape to spec 003:
- 2×H100 paired runs (or single variant vs Exp 24 baseline).
- If using Exp 24 as control, match its config (`QK_GAIN_INIT=5.0 TTT_ENABLED=0 SEED=1337`) plus `ACTIVATION=swiglu` + corresponding MLP ratio cut (e.g. 3×) to fit 16MB.
- Compare train_loss at matched steps + final pre-quant bpb.
- Accept: variant ≤ 1.0847 (Δ ≤ −0.002 vs 1.08670) — but note the MLP ratio cut may drag the variant down by ~0.001-0.002 on its own.

## If this works
- Orthogonal to BigramHash (different module). Could stack.
- Harder to make fit than BigramHash — MLP ratio cuts are lossier than the small embed-table add.
- Long-term high-confidence technique from the LLM literature; worth trying at least once.

## Priority
**After spec 003 (BigramHash).** Treat as the next-in-queue training-time candidate if BigramHash doesn't promote cleanly, or as a follow-up stacking candidate if it does.
