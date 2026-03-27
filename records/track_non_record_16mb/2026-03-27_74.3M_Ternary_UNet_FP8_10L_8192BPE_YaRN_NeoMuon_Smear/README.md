# Notable: 1.1090 BPB - 74.3M Ternary U-Net Transformer (100k steps, unconstrained)

**Extended training of [#640](https://github.com/openai/parameter-golf/pull/640) / [#641](https://github.com/openai/parameter-golf/pull/641) / [#920](https://github.com/openai/parameter-golf/pull/920) config with SmearGate enabled**

**val_bpb: 1.1090** (sliding, stride=16, T=0.90) | **15.95 MB** artifact | 8xH100 SXM, ~3h

> Same architecture as #641 (10L 768d ternary, EMBED_DIM=312, BF16 scales) trained for 100k steps unconstrained, with SmearGate enabled. Not a valid 10-minute submission - included to show scaling behaviour of the ternary U-Net architecture.

## Results

| Metric | 10min best (#641) | 100k steps (this) | Delta |
|--------|-------------------|-------------------|-------|
| Sliding BPB | 1.1535 | **1.1090** | -0.0445 |
| val_bpb | 1.1802 | 1.1344 | -0.0458 |
| RT bpb | 1.1808 | 1.1366 | -0.0442 |
| RT gap | 0.0006 | 0.0022 | +0.0016 |
| Steps | 6,530 | 100,000 | 15.3x |
| Training time | 600s | ~3h | - |
| Artifact | 15.95 MB | 15.95 MB | identical |
| zero_frac | 0.236 | 0.181 | -0.055 |

Extended training reduces zero_frac (0.236 -> 0.181) as the model utilises more of its ternary weight capacity. RT gap grows slightly (0.0006 -> 0.0022) due to the shrinkage correction amplification at longer training, but remains well-controlled with BF16 scale storage.

### Why BF16 scales matter for extended training

Ternary dequantization applies a shrinkage correction `1/(1-zero_frac)` to compensate for zeros reducing the group mean. FP16 scale storage introduces rounding error that gets multiplied by this factor. As training progresses and zero_frac changes, the amplification grows:

| zero_frac | Correction 1/(1-z) |
|-----------|-------------------|
| 0.25 | 1.33x |
| 0.30 | 1.43x |
| 0.35 | 1.54x |
| 0.40 | 1.67x |
| 0.50 | 2.00x |

The practical impact - FP16 vs BF16 scale storage at different training lengths:

| Config | Steps | Scale storage | RT gap | Notes |
|--------|-------|--------------|--------|-------|
| Ternary 10min | 6,530 | FP16 | 0.0021 | Original #640 submission |
| Ternary 10min | 6,533 | BF16 | 0.0011 | #641, same arch |
| Ternary extended | 150k | FP16 | **0.039** | Catastrophic - unusable |
| **Ternary extended** | **100k** | **BF16** | **0.0022** | **This run - well controlled** |


Without the changes applied, this extended run would have produced a 0.03+ BPB roundtrip gap, making the artifact unusable. The changes cost zero bytes and keep the gap at 0.0022 even at 100k steps.

## Changes from #920

- **SmearGate enabled** (`SMEAR=1`): learnable per-block gating for residual smoothing. Adds minimal params, provides small quality benefit at extended training.
- **100k iterations, no wallclock cap** (`MAX_WALLCLOCK_SECONDS=0`)
- **Checkpointing every 5k steps** for interruptible compute

Architecture, quantisation, compression, and all other hyperparameters identical to #920.

## Setup and Run

```bash
bash setup.sh
conda activate golf
bash run_cuda_ternary.sh
```