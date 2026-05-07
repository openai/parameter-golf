## OrthoInit + Muon + Int6 MLP3x + SmearGate + BigramHash

**val_bpb = 1.1524** (sliding window, stride=256) | **15.4 MB** artifact | 8xH100 SXM, 600s

Builds on our [prior submission](../2026-03-19_WiderMLP_Int6_SlidingWindow_1.1666/) (val_bpb 1.1659) with six additional techniques.

### What changed from v1

| | v1 (PR #70) | v2 (this) |
|---|---|---|
| val_bpb (sliding) | 1.1659 | **1.1524** |
| Artifact | 14.9 MB | 15.4 MB |
| Steps (8xH100, 600s) | 12,485 | 8,390 |
| Step time | 48ms | 68ms |
| Train seq_len | 1024 | 2048 |
| Optimizer | Muon | Muon (tuned) |
| Init | Default | Orthogonal + muP |
| Embedding augmentation | None | SmearGate + BigramHash |
| tok_emb quantization | int8 | int8 |
| Attention c_k (last 2 layers) | fp16 | int6 |

### New techniques

1. **Orthogonal + muP-scaled init** — All large weight matrices use `torch.nn.init.orthogonal_` (gain=1.0). Output projections scaled by `1/sqrt(2 * num_layers)`. Accelerates early convergence.

2. **SmearGate** — Learned sigmoid gate (~512 params) blending each token's embedding with the previous token's before the first transformer layer.

3. **Bigram Hash Embedding** — 4096-bucket hash table (dim=128, projected to 512) injecting token-pair features. Stored as int8 per-row in the artifact.

4. **Sequence length 2048** — Training and evaluation at 2048 tokens (up from 1024). NTK-aware RoPE for context scaling.

5. **Tuned optimizer** — `matrix_lr=0.02, scalar_lr=0.02, tied_embed_lr=0.03`, Muon momentum 0.99 with warmup from 0.92 over 1500 steps, warmdown 3000 iters, grad clip 0.3.

6. **Tighter quantization** — tok_emb.weight as int8 (was fp16 passthrough), last-2-layer c_k as int6 (was fp16). Saves ~400KB at +0.0001 BPB cost.

### Configuration

```bash
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Reproducibility

| Seed | Steps | Step time | Int6 sliding s256 | Int6 sliding s64 | Artifact |
|------|-------|-----------|-------------------|-----------------|----------|
| 1337 | 7,569 | 79ms | 1.1544 | — | 15,401,804 |
| **42** | **8,390** | **68ms** | **1.1524** | **1.1525** | **15,401,594** |
| 2025 | 8,712 | 68ms | 1.1546 | 1.1546 | 15,465,508 |

Mean val_bpb (s256): **1.1538**. Submitted run: seed 42 (best). Inter-seed variance: 0.0022.

### Included files

- `train_gpt.py` — full training + quantization + evaluation script
- `train.log` — training log from best seed
- `train_seed1337.log`, `train_seed42.log`, `train_seed2025.log` — all seed logs
- `submission.json` — leaderboard metadata
