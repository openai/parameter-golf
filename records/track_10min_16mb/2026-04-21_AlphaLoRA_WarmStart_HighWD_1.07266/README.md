# Alpha-Scaled LoRA + Warm-Start A + Higher Weight Decay — 1.07266 BPB

**val_bpb: 1.07266105** (3-seed mean: seeds 1337, 42, 314)

## Results

| Seed | BPB | Steps | Eval time | Artifact |
|------|-----|-------|-----------|----------|
| 1337 | 1.07297641 | — | 449.2s | 15,934,599 B |
| 42   | 1.07297575 | — | 448.3s | 15,933,292 B |
| 314  | 1.07203098 | — | 458.8s | 15,935,775 B |
| **Mean** | **1.07266105** | | | |

All runs: train ≤600s, eval ≤600s, artifact ≤16MB.

## Three novel changes on top of dexhunter's phased-TTT pipeline

Prior phased-TTT submissions (PR #1530 @samacqua, PR #1610 @romeerp, @dexhunter 1.07193)
use `BatchedLinearLoRA` with these defaults:

- `forward(x) = (x @ A.T) @ B.T`   *(no rank scaling)*
- `reset()`: re-randomize A uniform in [-1/√in, +1/√in], zero B
- `TTT_WEIGHT_DECAY = 0.5`
- `TTT_LORA_RANK = 96`

This submission replaces the LoRA module with three composable changes
(all env-controllable, all backward-compatible with the default flags off).

### (1) Alpha/rank output scaling — enables safe higher rank

```python
class BatchedLinearLoRA(nn.Module):
    _ALPHA = float(os.environ.get("TTT_LORA_ALPHA", "96"))

    def __init__(self, bsz, in_features, out_features, rank):
        ...
        self._scale = self._ALPHA / rank    # <-- novel
        ...

    def forward(self, x):
        return ((x @ self.A.T) @ self.B.T) * self._scale   # <-- novel
```

Without this, raw rank 128 **diverges** on seeds 314 and 1337 (TTT BPB collapses to ~1.133)
while working on seed 42. With `alpha=96` rank 96 is numerically identical to the prior
code, and `alpha=96, rank=128` reproduces that effective magnitude with 33% more A capacity.

### (2) Warm-start A across batches

```python
_WARM_START_A = bool(int(os.environ.get("TTT_WARM_START_A", "1")))

def reset(self):
    with torch.no_grad():
        if not self._WARM_START_A:
            self.A.uniform_(-self._bound, self._bound)
        self.B.zero_()
```

The phased-TTT loop processes ~780 batches of ~64 documents each. Previously A was
re-randomized every batch, which throws away whatever feature directions the optimizer
discovered on the earlier docs. Keeping A warm lets those directions accumulate while
still starting each batch cleanly (B is still zeroed, so LoRA output at the start of
a batch is zero).

In isolation warm-start gives a good improvement on seeds 1337 and 42 but **regresses**
on seed 314 (1.07200 → 1.07321) because A drifts into an overfit state for that seed's
document ordering.

### (3) Raised TTT weight decay 0.5 → 1.0 to stabilize (2)

Doubling weight decay explicitly counteracts the across-batch overfit that warm-start
enables. On seed 314 it restores parity with the alpha-only baseline (1.07200 → 1.07203,
essentially noise); on seeds 1337 and 42 the bulk of the warm-start gain is preserved.

### Combined result vs baselines

| Seed | Baseline rank 96 | + alpha rank 128 | + warm A + WD=1.0 |
|------|------------------|-------------------|---------------------|
| 1337 | 1.07423 | 1.07379 (−0.00044) | **1.07298 (−0.00125)** |
| 42   | 1.07341 | 1.07320 (−0.00021) | **1.07298 (−0.00043)** |
| 314  | 1.07214 | 1.07200 (−0.00014) | **1.07203 (−0.00011)** |
| Mean | 1.07326 | 1.07300 | **1.07266 (−0.00060)** |

Every seed improves or stays flat vs the rank-96 baseline.

## Legality (Issue #1017)

- **Condition 1 (Causal)**: single left-to-right pass; LoRA state at position `t`
  depends only on earlier tokens of the same document.
- **Condition 2 (Normalized distribution)**: standard softmax over the 8192 SentencePiece
  tokens, no hash bins / latent structures.
- **Condition 3 (Score-before-update)**: chunk is scored through `forward_ttt_train`
  *before* the optimizer step on that chunk.
- **Condition 4 (Single pass)**: one left-to-right pass, no rescoring.

## Attribution

Everything outside of `BatchedLinearLoRA` + the optimizer's weight-decay default is
unchanged from the existing pipeline:

- `@bigbag` (PR #1493) — triple depth recurrence, parallel residuals, SP8192 baseline
- `@EthanYangTW` (PR #1523) — parameter banking refinements
- `@samacqua` (PR #1530) — VarLen attention, Fused Triton MLP, doc-independent LoRA TTT
- `@romeerp` (PR #1610) — phased TTT (single-phase global SGD)
- `@dexhunter` (1.07193 submission) — multi-phase global SGD, trimmed GPTQ, MATRIX_LR=0.026, per-layer clip sigmas, int7 embeddings
- `@abaybektursun` (PR #549) — legal TTT framework

## Reproduction

```bash
export DATA_DIR=/path/to/parameter-golf/data

# Default seed 1337
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Other seeds
SEED=42  torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are hardcoded as defaults in `train_gpt.py`:
`TTT_LORA_RANK=128`, `TTT_LORA_ALPHA=96`, `TTT_WARM_START_A=1`,
`TTT_WEIGHT_DECAY=1.0`, `PHASED_TTT_ENABLED=1`, `PHASED_TTT_NUM_PHASES=3`,
`MATRIX_LR=0.026`, etc.
