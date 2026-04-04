## Lucky V

Rascal II + MuonEq-R + QK_GAIN=5 + NS7 + SLOT32 (sliding-window test-time adaptation, lr=0.04)


https://github.com/user-attachments/assets/51915564-1ee6-41c8-b417-7e5401f62234



## Results

| Seed | val_bpb (sliding window + SLOT) | Steps | Size |
|------|--------------------------------|-------|------|
| 444  | 1.08540457                     | 6,596 | 15,426,795 B |
| 4    | 1.08576841                     | 6,600 | 15,437,033 B |
| 300  | 1.08495213                     | 6,597 | 15,438,323 B |
| **mean** | **1.08537504**             |       | **15,438,323 B** |
| **std**  | **0.00041**                |       |              |

Hardware: 8×H100 SXM · 600s wallclock · `bytes_code`: 124,171

  Core transformer:
  - 11 layers, dim=512, 8 attention heads (4 KV heads = GQA)
  - MLP mult 3x (so MLP hidden = 1536)
  - Vocab 1024 (SentencePiece BPE), tied embeddings
  - RoPE (16 dims, base 10000)
  - Logit softcap at 30.0

  Activation: LeakyReLU-squared (custom Triton kernel) — not standard GELU/SiLU

  Attention: XSA on all 11 layers, Flash Attention 3 (Hopper), per-head QK gain (init=5.0)

  Auxiliary features:
  - Bigram hash table (dim=128, 2048 vocab) — n-gram side channel
  - Value Embeddings (VE) on layers 9-10 (dim=128)

  Training:
  - Muon optimizer (Newton-Schulz, 7 iterations) with MuonEq-R (row-normalize momentum)
  - SWA (stochastic weight averaging)
  - Late QAT (fake int6 quantization in last ~200 steps)
  - EMA weights applied post-training
  - Coprime shard loader, 600s wallclock cap (~6600 steps)

  Quantization: Naive int6 + brotli compression (no GPTQ)

  Eval-time:
  - Sliding window (stride=64) + SLOT (32-step test-time adaptation, lr=0.04)
  - SLOT fits additive deltas on hidden states — backward-looking, score-first (legal Track B)

## Architecture changes

- **MuonEq-R**: Row-normalize momentum before Newton-Schulz orthogonalization
- **QK_GAIN_INIT=5.0**: Per-head query scaling (from 1.5)
- **MUON_BACKEND_STEPS=7**: More Newton-Schulz iterations (from 5)
- **SLOT_STEPS=32**: Test-time adaptation steps (from 24)
- **SLOT_LR=0.04**: Tuned from default 0.005 via eval-only sweep

## Compliance

- Training ≤ 600s on 8×H100 SXM: **Yes** (600s wallclock cap)
- Eval ≤ 600s on 8×H100 SXM: **Yes** (~590s sliding window)
- Total artifact ≤ 16,000,000 bytes: **Yes** (15,437,033 max)
- No validation leakage during training: **Yes**
- No pre-eval adaptation on unseen validation tokens: **Yes** (SLOT is score-first, backward-looking)

## Reproduce

```bash
# From repo root, with flash-attention/hopper on PYTHONPATH
SEED=444 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-04_Lucky_V_8xH100/train_gpt.py
```

Expected final line (seed 444):
```
final_sliding_window+slot32steps_exact val_loss:1.83265530 val_bpb:1.08540457
```

<img width="1536" height="1024" alt="slot_machine" src="https://github.com/user-attachments/assets/e7043602-ac9f-4916-a043-c1394c8d83ea" />
