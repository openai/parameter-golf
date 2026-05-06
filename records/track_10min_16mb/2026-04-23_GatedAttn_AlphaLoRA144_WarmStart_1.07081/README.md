# GatedAttn + Alpha-Scaled LoRA + Warm-Start A + WD=1.0 — 1.07081 BPB

**val_bpb: 1.07080923** (3-seed mean: seeds 1337, 42, 314)

## Results

| Seed | BPB | Eval time | Artifact |
|------|-----|-----------|----------|
| 1337 | 1.07146084 | 466.3s | 15,976,807 B |
| 42   | 1.07013846 | 470.2s | 15,975,906 B |
| 314  | 1.07081910 | 474.0s | 15,978,037 B |
| **Mean** | **1.07080923** | | |

All runs: train ≤596s, eval ≤600s, artifact ≤16MB.

## What this submission adds on top of PR #1767

**PR #1767** (this author) contributed four LoRA-TTT improvements that were
adopted by @bigbag in PR #1771:

1. Alpha/rank output scaling on BatchedLinearLoRA
2. Warm-start LoRA A across batches
3. TTT weight decay 0.5 → 1.0
4. Alpha lifted to 144 on rank 128

This submission stacks **GatedAttn** (per-head sigmoid gate on SDPA output,
from @dexhunter PR #1736) on top, plus two novel support changes needed to
make GatedAttn compatible with the LoRA-TTT eval path.

### (A) Novel: mirror the gate inside the LoRA-TTT forward path

`_block_with_lora` and `_parallel_block_with_lora` in the LoRA-TTT eval path
reimplement attention *inline* (they have to inject LoRA into q/k/v/out).
They do not call `block.attn.forward`, so a gate added naively to
`CausalSelfAttention.forward` is **silently dropped** at TTT scoring time.

The consequence: training sees `y * sigmoid(x @ W_g.T)`, TTT scoring sees
`y` (no gate). The distribution that phased TTT conditions on is wrong, and
the gradient signal drives the LoRA into a bad region. In our first attempt
TTT collapsed to **1.40 BPB**.

Fix: apply the gate inside both `_block_with_lora` and `_parallel_block_with_lora`
right after SDPA, before the out_proj LoRA adds its delta:

```python
if getattr(attn, "gated_attn", False):
    n_c = n.contiguous()
    g = torch.sigmoid(F.linear(n_c, attn.attn_gate_w.to(n.dtype)))
    y = y * g[..., None]
```

After the fix, TTT converges normally and GatedAttn gives
**-0.00152 BPB** (3-seed mean) vs the rank-128 alpha-144 baseline.

### (B) Novel: per-row int8 quantization for attn_gate_w

Each gate weight is `(num_heads, dim) = (8, 512)` per block × 11 blocks =
45,056 params. fp16 passthrough pushed the artifact from 15.93 MB to
**16.01 MB**, exceeding the 16 MB cap on all three seeds.

Per-tensor int8 quantization saves ~45 KB but loses **+0.00112 BPB** on
the 3-seed mean — the single shared scale is too coarse given the variance
across heads.

Per-row int8 (one fp16 scale per head, 8 scales per block × 11 blocks =
88 extra bytes total) keeps precision essentially intact: **-0.00128 BPB**
from PR #1767 mean vs -0.00152 with fp16 gates (0.00024 difference,
comfortably inside the 1σ noise band).

```python
if "attn_gate" in name and t.is_floating_point():
    t32 = t.float()
    abs_max_rows = t32.abs().amax(dim=tuple(range(1, t32.ndim)), keepdim=True).clamp_min(1e-8)
    scales = (abs_max_rows / 127.0).squeeze()
    q = torch.round(t32 / abs_max_rows * 127.0).clamp(-127, 127).to(torch.int8)
    result[name + ".q"] = q
    result[name + ".scale"] = scales.to(torch.float16).contiguous()
```

The existing `dequantize_mixed` path already handles `s.ndim > 0` with the
correct broadcast, so no decoder-side change is needed.

### Combined 3-seed result

| Seed | rank-96 baseline | PR #1767 (alpha 144 + warm + WD) | **+ GatedAttn (this)** |
|------|------------------|---------------------------------|-------------------------|
| 1337 | 1.07423 | 1.07189 | **1.07146** (−0.00043) |
| 42   | 1.07341 | 1.07248 | **1.07014** (−0.00234) |
| 314  | 1.07214 | 1.07189 | **1.07082** (−0.00107) |
| Mean | 1.07326 | 1.07209 | **1.07081** (−0.00128) |

Every seed improves monotonically.

## Legality (Issue #1017)

- **Condition 1 (Causal)**: single left-to-right pass; LoRA at position `t`
  depends only on earlier tokens of the same doc.
- **Condition 2 (Full normalized distribution)**: standard softmax over 8192
  SentencePiece tokens.
- **Condition 3 (Score-before-update)**: each chunk is scored through
  `forward_ttt_train` before the optimizer step on that chunk.
- **Condition 4 (Single pass)**: one left-to-right pass, no rescoring.

## Attribution

- @bigbag (PR #1493, #1771) — triple depth recurrence, parallel residuals, SP8192
- @EthanYangTW (PR #1523) — parameter banking refinements
- @samacqua (PR #1530) — VarLen attention, Fused Triton MLP, doc-independent LoRA TTT
- @romeerp (PR #1610) — phased TTT (single-phase global SGD)
- @dexhunter — multi-phase global SGD, trimmed GPTQ, GatedAttn (PR #1736)
- @abaybektursun (PR #549) — legal TTT framework
- **This author** — PR #1767 (LoRA-TTT stack) + this submission's two novel changes (gate mirroring, per-row int8 gate quant)

## Reproduction

```bash
export DATA_DIR=/path/to/parameter-golf/data
torchrun --standalone --nproc_per_node=8 train_gpt.py        # seed 1337
SEED=42  torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters (including `GATED_ATTN_ENABLED=1`, `GATED_ATTN_INIT_STD=0.005`,
`TTT_LORA_RANK=128`, `TTT_LORA_ALPHA=144`, `TTT_WARM_START_A=1`,
`TTT_WEIGHT_DECAY=1.0`) are hardcoded as defaults.
