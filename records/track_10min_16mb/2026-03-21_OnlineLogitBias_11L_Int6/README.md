Non-record submission: 11L int6 with Online Logit Bias eval technique.

## result

**val_bpb: 1.1609** (sliding window, stride=64) | 13.9 MB artifact | 8xH100 SXM, 600s

Note: ran without FlashAttention 3 (SDPA fallback). FA3 would improve step time and final score.

| metric | value |
|--------|-------|
| pre-quant val_bpb | 1.1709 |
| int6 roundtrip val_bpb | 1.1829 |
| int6 sliding val_bpb (s64) | **1.1609** |
| steps | 7,620 / 20,000 (wallclock cap) |
| step time | 78.7ms |
| artifact | 13,977,633 bytes |

## novel technique: online logit bias (OLB)

Learned per-token bias vector added to logits during sliding window eval. Updated after each scored batch using the exact CE gradient: `b -= lr * (softmax(z+b) - onehot(y))`. Only uses already-scored tokens to update - compliant with the TTT rules. Zero model parameters, near-zero compute overhead. Strictly generalizes frequency counting since the gradient naturally captures frequency information plus systematic prediction biases.

`OLB_LR=0.1` enables it. `OLB_LR=0` disables. OLB was not enabled in this run - pending further compute to validate.

## training stack

11 layers, 512 dim, 3x MLP (1536 hidden), relu^2, GQA 8/4 heads, sp1024 tied embeddings, int6 per-row quant + zstd, SmearGate, BigramHash(2048x128), OrthoInit + muP, seq 2048 + NTK RoPE, Muon WD 0.04, EMA (0.997), XSA on last 4 layers, Partial RoPE (16/64 dims), LN Scale, Late QAT.

## command

```bash
OLB_LR=0 SEED=1337 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## files

- `train_gpt.py`
- `submission.json`
- `requirements.txt`
