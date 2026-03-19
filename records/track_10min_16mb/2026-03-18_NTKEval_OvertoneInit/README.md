## NTK Eval + Overtone Init

Three changes on top of the naive baseline, two to training and one to evaluation:

### 1. Overtone Embedding Initialization

After the standard normal init, we reshape the embedding matrix's singular value spectrum to follow a power-law decay (like guitar harmonics):

```python
U, S, V = torch.linalg.svd(tok_emb.weight.data, full_matrices=False)
target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1)) ** 0.5
tok_emb.weight.data = (U * target_S[None, :]) @ V
```

This gives the embedding a structured spectrum from the start, rather than relying on training to discover it. The intuition is that natural language has a power-law frequency structure, and the embedding should reflect this.

### 2. Phase-Transition Residual Mixing

The `resid_mix` parameters (which blend the current hidden state with the initial embedding `x0`) are initialized with a sigmoid schedule across layers:

```python
phase = sigmoid(3.0 * (layer_index / (num_layers - 1) - 0.5))
resid_mix[0] = phase      # weight on current hidden state
resid_mix[1] = 1 - phase  # weight on initial embedding x0
```

Early layers trust the raw embedding more; late layers trust the evolved representation. This mirrors the information flow in U-Net architectures.

### 3. NTK-Aware RoPE Scaling for Eval@2048

We train at sequence length 1024 but evaluate at 2048 using dynamic NTK-aware RoPE frequency scaling:

```python
if seq_len > train_seq_len:
    scale = seq_len / train_seq_len
    adjusted_base = base * (scale ** (dim / (dim - 2)))
    inv_freq = 1.0 / (adjusted_base ** (arange(0, dim, 2) / dim))
```

This scales the RoPE base frequency at inference time to preserve local attention patterns while extending global context. The competition FAQ explicitly allows evaluation at any sequence length.

### Other Changes

- AdamW with weight decay 0.01 (was 0, reduces quantization gap)
- Warmdown extended to 2500 steps (was 1200)
- Tied embedding LR increased to 0.10 (was 0.05)
- Python 3.12 (faster than 3.13 for torch.compile)

### Results

Three seed runs on 8xH100, 10-minute wallclock:

| Seed | Steps | ms/step | Pre-quant BPB | Post-quant BPB | val_loss (nats) | Artifact |
|------|-------|---------|--------------|----------------|-----------------|----------|
| 1337 | 13024 | 46.07 | 1.2210 | 1.2166 | 2.0542 | 15.78MB |
| 42 | 13239 | 45.32 | 1.2211 | 1.2163 | 2.0537 | 15.79MB |
| 7 | 13230 | 45.35 | 1.2188 | 1.2152 | 2.0519 | 15.80MB |
| **Mean** | | | **1.2203** | **1.2160** | **2.0533** | |

**Improvement over baseline: 0.0194 nats (p = 0.0012, one-sided t-test)**

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings, `TIED_EMBED_LR=0.10`, `WARMDOWN_ITERS=2500`
- `TRAIN_SEQ_LEN=1024`, `EVAL_SEQ_LEN=2048` (NTK-scaled RoPE at eval time)
- `TRAIN_BATCH_TOKENS=524288`

Included files:
- `train_gpt.py` — training script
- `train_seed1337.log`, `train_seed42.log`, `train_seed7.log` — full training logs
- `submission.json` — leaderboard metadata
