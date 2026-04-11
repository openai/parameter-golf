# RecurLoRA v2: Per-Pass Low-Rank Adapters on the SP8192 Depth Recurrence Stack

**val_bpb: TBD** (3-seed runs pending) | **~16.0 MB artifact** | 8xH100 SXM

## Summary

RecurLoRA v2 applies per-pass specialization mechanisms on top of the depth recurrence architecture that the current leaderboard top-5 all use. The hypothesis: the SOTA stack (PR #1394/#1493) reuses layers 3-5 identically across all passes, with no signal to distinguish which iteration is running. RecurLoRA v2 adds two complementary mechanisms -- pass index embeddings (Universal Transformer-inspired) and rank-2 LoRA attention corrections -- enabling per-pass specialization at negligible parameter cost (~48KB total, 0.3% of budget).

This submission combines every technique from the current frontier:
- **SP8192 vocabulary** (from PR #1394)
- **3-layer depth recurrence** with delayed activation at 35% (from PR #1493)
- **Parallel residuals** from layer 7+ (from PR #1412/#1477)
- **SDClip quantization** (clip = k * std(row)) with int6 matrices / int8 embeddings (from PR #1394)
- **MuonEq-R** (row-normalized Muon optimizer) (from PR #1394)
- **QK-Gain 5.25** (from PR #1493)
- **Score-first Muon-TTT** (legal variant)
- **RecurLoRA** (our contribution from PR #1181, adapted to the new stack)

## Status

Implementation complete. Syntax validated, all architectural checks passing. Awaiting compute for training runs.

## What RecurLoRA Adds to the Current SOTA

The current #1 (PR #1493, 1.0810 BPB) uses 3-layer recurrence where layers 3-5 are executed identically on every pass. The encoder/decoder traversal order is:

```
Encoder: [0, 1, 2, 3, 4, 5, 3, 4]
Decoder: [5, 3, 4, 5, 6, 7, 8, 9, 10]
```

Every time layer 3 (or 4, or 5) is executed, it uses the exact same weights and receives no signal about which pass it is on. RecurLoRA v2 addresses this with two complementary mechanisms:

**1. Pass Index Embeddings** (inspired by Universal Transformers, Dehghani et al. 2019): A learned vector per pass is added to the hidden state before the repeated block executes, telling the shared weights which iteration they are processing:

```
x_repeat = x + pass_embed[pass_idx]  # [num_extra_passes, model_dim]
```

Cost: 3072 params (6KB). This modifies the *input* to the shared layer.

**2. Low-Rank Attention Corrections (RecurLoRA)**: Rank-2 LoRA on Q, K, V, O attention projections for repeated passes:

```
Q_repeat = Q_shared + alpha * (B_q @ A_q)
```

Cost: ~21K params (42KB). This modifies the *weights* of the shared layer.

Together, pass embeddings and LoRA give shared layers maximum ability to specialize per pass at minimal cost. The MLP weights (4x width = 2048) remain fully shared.

### Why this could improve on raw sharing

1. **Different passes see different input distributions** -- the first pass through layer 3 sees the output of layer 2, while the second pass sees the output of layer 5. Identical weights cannot optimally handle both.
2. **Pass embeddings provide a "which iteration am I?" signal** -- grounded in Universal Transformers, enabling the shared attention/MLP to condition its behavior on recurrence depth.
3. **LoRA corrections are immune to quantization error** -- stored as fp16 passthrough, they don't contribute to the error amplification that kills deeper recurrence.
4. **Combined overhead is ~48KB** -- 0.3% of the 16MB budget.

## Architecture

- 11 physical layers, **17 virtual layers** (layers 3-5 looped with `num_loops=2`)
- SP8192 vocabulary (8192-token SentencePiece BPE)
- 512d, 8 GQA heads / 4 KV heads, MLP 4x (2048)
- LeakyReLU(0.5)^2 activation
- Partial RoPE (16/64 dims), XSA on all layers
- **Parallel residuals** from layer 7+ (attention and MLP read from same input)
- QK-Gain 5.25, LayerNorm scale 1/sqrt(layer+1)
- EMA(0.9965), Parallel MuonEq-R (row-normalized) with WD=0.095
- SDClip: int6 matrices (clip=12.85*std), int8 embeddings (clip=20*std)
- Full Hessian GPTQ + Brotli-11 + byte-shuffle compression
- Depth recurrence activated at 35% of training
- **RecurLoRA**: rank-2 LoRA on attention + pass index embeddings for repeated layer passes

## Run Command

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All defaults are set to match the frontier stack. To disable RecurLoRA (for ablation):
```bash
RECUR_LAYERS="" torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Ablation Plan

| Step | Experiment | Purpose |
|------|-----------|---------|
| 1 | Full stack without RecurLoRA | Confirm baseline matches ~1.081 BPB |
| 2 | Full stack with RecurLoRA | Measure LoRA contribution |
| 3 | Alpha sweep: [0.3, 0.5, 0.6, 0.8] | Optimal correction strength |
| 4 | Report mean +/- std across 3 seeds | Statistical validation |

## Credits

- **SP8192 + SDClip + depth recurrence + MuonEq-R**: PR #1394 by @clarkkev
- **3-layer recurrence + QK-Gain 5.25**: PR #1493 by @bigbag
- **Parallel residuals**: PR #1412 by @Robby955, PR #1477 by @aryanbhosale
- **RecurLoRA concept**: PR #1181 (this author's prior submission)
- **Shallow recurrence insight**: PR #686 by @msisovic
