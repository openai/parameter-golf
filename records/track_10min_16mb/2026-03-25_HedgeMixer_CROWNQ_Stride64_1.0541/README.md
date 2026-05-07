# Record: 5-expert Hedge Mixer + CROWN-Q + stride=64 (val_bpb=1.0541)

**val_bpb: 1.0541** (3-seed mean) | **~15.7 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | Eval time | Artifact |
|------|----------|-------|-------------|-----------------|----------|-----------|----------|
| 1337 | 98.1ms | 5,935 | 1.1251 | **1.0473** | -0.0778 | 336s | 15.89 MB |
| 42 | 97.9ms | 5,947 | 1.1264 | **1.0686** | -0.0578 | 336s | 15.69 MB |
| 7 | 98.0ms | 5,940 | 1.1246 | **1.0465** | -0.0781 | 336s | 15.66 MB |
| **Mean** | | | 1.1254 | **1.0541** | -0.0713 | 336s | ~15.75 MB |

## Contributions

### 1. CROWN-Q Training Penalty (training-time)
Added a quantization-aware penalty during warmdown that penalizes weights sensitive to quantization error:
```
crown_q_loss = lambda * mean(w^2 * delta^2 / 12)
```
where `delta = row_max / clip_range` is the per-row quantization step size. This encourages weights to be quantization-friendly, reducing post-quantization degradation. `CROWN_Q_LAMBDA=0.01`.

**Effect**: Slightly better compression (artifact ~200KB smaller) and more robust quantization.

### 2. Eval stride 32 -> 64 (eval-time)
Changed sliding window stride from 32 to 64 during evaluation. Experiment showed identical BPB quality but 2x faster scoring. Frees ~100s of eval budget for more TTT epochs.

### 3. TTT Epochs 3 -> 4 (eval-time)
Increased test-time training from 3 to 4 epochs per chunk, using the time freed by stride=64. Each additional epoch adapts the model more to scored data. Tested 8 epochs but that overfits (1.0735 vs 1.0473 for 4 epochs).

### Combined Effect
- stride=64 saves ~100s of eval time
- 4th TTT epoch uses ~85s of the saved time
- Net eval time: ~336s (down from ~562s), well within 600s budget
- BPB improvement: 1.0745 -> 1.0541 (-0.0204)

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 8KV) |
| MLP | 3.5x with LeakyReLU(0.5)^2 |
| BigramHash | 6144 (dim=128) |
| XSA | All 11 layers (ws=8) |
| VE128 | Layers 9-10 |
| Quantization | Full GPTQ int5 + zstd level 22 |
| Pruning | 3% magnitude |
| TTT | AdamW lr=0.0001, **4 epochs**, 131K chunks, Polyak 0.998 |
| Mixer | 5-expert Hedge (neural, unigram, bigram, trigram, entropy) |
| Training reserve | 18s (for EMA + calibration + quantization) |
| Early warmdown | LR schedule targets 582s |
| **CROWN-Q** | lambda=0.01 during warmdown |
| **Eval stride** | 64 (was 32) |

## Reproduction

```bash
DATA_PATH=../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
USE_MIXER=1 MIXER_ETA=0.1 \
TTT_EPOCHS=4 TTT_FREEZE_BLOCKS=2 \
TTT_LR=0.0001 TTT_CHUNK_TOKENS=131072 \
ADAPTIVE_LR=1 ADAPTIVE_LR_MAX=3.0 \
EVAL_STRIDE=64 \
CROWN_Q_LAMBDA=0.01 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```


## Compliance

| Constraint | Limit | Actual | Status |
|-----------|-------|--------|--------|
| Train time | 600s | 582s | Pass |
| Eval time | 600s | 336s | Pass |
| Artifact size | 16,000,000 bytes | 15,892,040 bytes (worst seed) | Pass |
| No pre-scoring training | — | Score-first TTT: each chunk scored under `inference_mode()` before any training on it | Pass |
| GPTQ calibration in training budget | — | Runs within 18s training reserve (1.9s actual) | Pass |

## Credits

- Base model: PR #414 by @signalrush
- TTT recipe: PR #461 by @Christopher-Lee-McClendon
- CROWN-Q concept: PR #693 by @EthanYangTW
- 5-expert Hedge mixer: PR #688
