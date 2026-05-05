# MuonEq-R + Muon Weight Decay

Builds on the naive baseline with two optimizer improvements that add zero artifact cost.

## Changes

**MuonEq-R** — Row-normalise each gradient matrix before the Newton-Schulz orthogonalisation step inside Muon. Ensures every output channel contributes equally to the update; no single neuron dominates the orthogonalised direction.

```python
X = X / (X.norm(dim=1, keepdim=True) + eps)   # per-row normalisation
X /= X.norm() + eps                             # global normalisation
```

**Muon Weight Decay** — Decoupled weight decay applied to matrix parameters after the Muon update, AdamW-style:

```python
p.add_(g, alpha=-lr)
p.mul_(1.0 - lr * wd)   # wd = 0.085
```

Both changes are controlled via environment variables (`MUON_WEIGHT_DECAY`, `MUON_BACKEND_STEPS`) with no architecture changes and no new parameters.

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Muon weight decay: `MUON_WEIGHT_DECAY=0.085`
- Muon backend steps: `MUON_BACKEND_STEPS=3`

## Command

```bash
RUN_ID=run2_wd SEED=1337 SSM_LAYERS= MUON_WEIGHT_DECAY=0.085 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results (3 seeds)

| Seed | val_bpb (post int8+zlib) | Submission size |
|------|--------------------------|-----------------|
| 1337 | 1.2238 | 15,879,569 B |
| 1338 | 1.2278 | 15,877,120 B |
| 1339 | 1.2256 | 15,875,037 B |
| **Average** | **1.2257** | |

Best seed (1337) key metrics:
- Steps completed: 13,691 / 20,000 (wallclock cap)
- Pre-quant val_bpb: 1.2171
- Post-quant val_bpb: 1.2238
- Step avg: 43.83ms
- Peak memory: 10,119 MiB allocated

## Included Files

- `train_gpt.py` — code snapshot used for all runs
- `train_seed1337.log`, `train_seed1338.log`, `train_seed1339.log` — exact remote training logs
- `submission.json` — leaderboard metadata (best seed)
