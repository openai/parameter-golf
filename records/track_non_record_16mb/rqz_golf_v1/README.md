# RQZ-Golf v1 — Depth Recurrence for Parameter Golf

## Approach

Replace some unique layers with a single shared recurrent layer applied K times.
This saves parameters (shared weights) while increasing effective depth.

### Architecture
- 7 unique layers (3 encoder + 4 decoder with U-Net skip connections)
- 1 recurrent layer applied K=3 times with iteration embeddings
- Effective depth: 10 layers (7 unique + 3 recurrent) vs baseline 9

### Key ideas
1. **Depth recurrence**: last block shares weights across K passes, saving ~3M params
2. **Iteration embeddings**: learned per-pass vector so the layer knows which pass it's on
3. **Stability scaling**: residual scaled by 1/sqrt(K) to prevent amplitude explosion
4. **Test-time compute**: can increase K at inference (K'=6, 8, ...) for better BPB

### Theoretical basis
Inspired by Universal Transformers (Dehghani 2019) and Deep Equilibrium Models (Bai 2019).
Each recurrent pass reconstructs the residual of the previous pass in latent space.

## Config
```
NUM_UNIQUE_LAYERS=7
NUM_RECURRENT_PASSES=3
# All other params same as baseline
```

## Author
Regis Rigaud (@TheCause)
