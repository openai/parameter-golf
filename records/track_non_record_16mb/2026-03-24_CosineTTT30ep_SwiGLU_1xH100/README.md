# Non-Record: 30-Epoch Cosine TTT on PR #462 Architecture (1xH100)

**val_bpb = 1.1175** (sliding window stride=64, seed 1337) | **7.5 MB** artifact | 1xH100 SXM 80GB

## Approach

Single-variable change from PR #462's SwiGLU architecture: increase TTT epochs from 10 to 30. All other hyperparameters and architecture identical.

This is consistent with PR #481's finding that cosine TTT with more epochs improves results, and PR #486's confirmation that 30-epoch cosine TTT improved their stack from 1.1132 to 1.0887 on 8xH100.

## Results (1xH100 SXM, seed 1337)

| Metric | Value |
|--------|-------|
| Training steps | 936 (wallclock capped at 600s) |
| Pre-quant val_bpb | 1.3646 |
| Post-quant roundtrip val_bpb | 1.0684 |
| **Sliding window val_bpb (stride=64)** | **1.1175** |
| Artifact size | 7,505,437 bytes |
| TTT time | 3,376s (30 epochs, cosine LR + per-layer LR) |

## Architecture (from PR #462)

- 11 layers, 512 dim, 8 heads, 8 KV heads
- SwiGLU / Star-ReLU MLP (hidden=1792) with learnable scale+bias
- U-Net skip connections with learned sigmoid gating
- BigramHash (8192 buckets, 128 dim), SmearGate
- EMA (decay=0.9985), Late QAT (threshold=0.15)
- Partial RoPE (16 dims), LN Scale (1/sqrt(layer+1))
- Int6 + zstd-22 compression

## Key Change

```diff
- ttt_epochs = int(os.environ.get("TTT_EPOCHS", "10"))
+ ttt_epochs = int(os.environ.get("TTT_EPOCHS", "30"))
```

## Limitation

Run on 1xH100 — needs 8xH100 verification for record consideration. 30 TTT epochs at seq_len=2048 estimated at ~7 min on 8xH100 (within 10-min eval budget).

## Credits

- **PR #462** (JoeProAI): SwiGLU + U-Net architecture + cosine TTT
- **PR #481** (mrdavtan): Cosine TTT scheduling + per-layer LR discovery
- **PR #442** (sjp611): AdamW TTT
- **PR #398** (felipe-parodi): EMA, TTT, XSA, architectural foundations

## Run Command

```bash
SEED=1337 torchrun --standalone --nproc_per_node=1 train_gpt.py
```

All hyperparameters are defaults in train_gpt.py (TTT_EPOCHS=30).
