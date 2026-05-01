# Record: SP8192 + Parallel Residuals + Score-First TTT — val_bpb 1.0822 (3-seed mean)

**val_bpb = 1.0822** (3-seed mean, std 0.0005) | **~15.99 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | **TTT BPB** | Artifact |
|------|-------------|-------------|----------|
| 42   | 1.0857      | **1.0826**  | 15,991,486 |
| 314  | 1.0854      | **1.0822**  | 15,991,486 |
| 999  | 1.0849      | **1.0817**  | 15,991,486 |
| **Mean** | | **1.0822** | |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.0325 BPB**.

## Novel Contribution: Parallel Residuals + Score-First TTT on SP8192

This submission adds **parallel residuals** (from layer 7) to the SP8192 + score-first TTT stack. Prior work had these separately:
- PR #1413 (@dexhunter): SP8192 + TTT, no parallel residuals → 1.0828
- PR #1412 (@Robby955): SP8192 + parallel residuals, no TTT → 1.0835

Combining both gives **1.0822** — better than either alone.

From layer 7, attention and MLP operate on separate residual lanes. A learned `lane_merge` scalar (init 0.5) blends the lanes after the final layer. This lets attention specialize on context mixing while MLP specializes on token transformations.

## Full Stack

SP8192, MLP 4x, depth recurrence (loop 4-5), parallel residuals (layer 7+), MuonEq-R, QK-Gain 5.0, SDClip, GPTQ embeddings, skip gates, score-first TTT (3 epochs, lr=0.005), brotli compression.

## Compliance (Track B — Score-First TTT)

- Score-first TTT: every token scored BEFORE weight update (PR #461 framework)
- No SLOT, no n-gram cache, no pre-quant TTT
- Model frozen after TTT; standard sliding-window eval
- All four conditions from Issue #1017 satisfied

## Reproduction

```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest
SEED=42 TTT_ENABLED=1 PARALLEL_START_LAYER=7 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1394 @clarkkev (SP8192 base), PR #1413 @dexhunter (score-first TTT on SP8192), PR #1412 @Robby955 (parallel residuals on SP8192), PR #1204 @msisovic (parallel residuals concept), PR #1260 @dexhunter (MuonEq-R + depth recurrence)
