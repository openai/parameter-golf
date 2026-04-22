# TTT LoRA Warm-Start A

**Source:** PR #1767 (renqianluo, 2026-04-22), with intermediate ablation in PR #1765 (same author)
**Author credibility:** renqianluo is new to the leaderboard but submitted a well-ablated 3-seed result with monotonic improvement across 4 changes. Attribution cites bigbag, EthanYangTW, samacqua, romeerp — standard Trunk A lineage.

## What it is

Four composable changes to `BatchedLinearLoRA` in the phased TTT module:

### (1) Alpha/rank output scaling

```python
self._scale = alpha / rank  # e.g. 144 / 128 = 1.125

def forward(self, x):
    return ((x @ self.A.T) @ self.B.T) * self._scale
```

Standard LoRA scaling trick. Enables stable higher rank (128 vs 96 baseline) — without this, raising rank causes seed-specific divergence.

### (2) Warm-start A across TTT batches

```python
def reset(self):
    with torch.no_grad():
        # A keeps its state (warm)
        self.B.zero_()  # B still zeros → LoRA output = 0 at batch start
```

Phased TTT processes ~780 batches × ~64 documents. Currently both A and B are reset between batches. Warm-start keeps A from the prior batch, letting it accumulate useful feature directions across the full eval context. Since B is zeroed, LoRA output = `(x @ A.T) @ 0.T = 0` at the start of each batch — per-document reset is preserved; legal under Issue #1017 Condition 3.

### (3) Raised WD 0.5 → 1.0

Warm-start A alone causes seed-specific overfit (A over-specializes to early-document patterns). Doubling weight decay counteracts this without sacrificing the warm-start gain on other seeds.

### (4) Alpha lifted 96 → 144 (rank 128 → scale 1.125)

With (1)+(2)+(3) stable, the LoRA is under-utilized at scale=0.75 (96/128). Lifting to 144 gives scale=1.125; WD=1.0 keeps it stable.

## Evidence from #1767

| Seed | baseline | +alpha 96/128 | +warm A +WD=1 | +alpha 144 |
|------|----------|---------------|----------------|------------|
| 1337 | 1.07423 | 1.07379 | 1.07298 | **1.07189** |
| 42   | 1.07341 | 1.07320 | 1.07298 | **1.07248** |
| 314  | 1.07214 | 1.07200 | 1.07203 | **1.07189** |
| Mean | 1.07326 | 1.07300 | 1.07266 | **1.07209** |

All seeds improve monotonically. 3-seed std very low.

Note: their baseline is non-CaseOps (~1.073), not our #1736 CaseOps stack (1.065). So absolute 1.07209 is not below our baseline, but the mechanism is composable.

## Estimated Δ-bucket

−0.001 bpb on the non-CaseOps stack. Unknown on CaseOps stack but likely similar — the phased TTT architecture is identical; the only interaction is that our CaseOps stack already has a better base, so there may be slightly less TTT headroom.

If our spec 021 (recur-alpha-buffer) delivers ~−0.005 pre-quant and the TTT gain is ~−0.005 post-TTT, then warm-start-A is a potential stacking improvement on top.

## Feasibility

Very easy to implement — 3-5 line change in `BatchedLinearLoRA`. No new dependencies. Fully composable with our existing phased TTT.

## Risks

1. **CaseOps interaction:** unknown. Run on mini first.
2. **Already in our stack?** No — #1736 baseline resets A each batch.
3. **Already specced?** No — specs 011-014, 021-023 do not cover this.
4. **Spec 021 interaction:** warm-start-A and buffer-α are orthogonal (one modifies TTT LoRA adaptation, the other modifies training-time recurrence). Should stack cleanly.

## Priority

Low-medium. Spec 021 results must land first. If 021 shows the expected gains (~−0.005), warm-start-A could be spec 024 as a stacking improvement. If 021 disappoints, warm-start-A is a cheap independent try.
