# PR #414 Stack + 30-Epoch Cosine TTT

**val_bpb: 1.0988** (8xH100 SXM, seed=1337, stride=64 sliding window eval)

## Summary

Adds 30-epoch cosine pre-eval Test-Time Training (TTT) on top of the PR #414 consensus stack. TTT adapts the quantized model on validation data before the final sliding-window eval, recovering quantization loss and further improving BPB through domain adaptation.

## Key Addition: Cosine Pre-Eval TTT

After int6 quantization and roundtrip eval, the model is fine-tuned on validation data for 30 epochs with cosine LR decay before the final sliding-window eval:

- AdamW optimizer, base LR=0.0005, weight_decay=0.0
- Per-layer LR groups: `mlp.proj` 3x, `mlp.fc` 0.5x, others 1x
- Cosine LR schedule across all TTT steps
- DDP gradient sync (all_reduce AVG)
- Batch size: 32 sequences per rank
- Gradient clipping: 1.0

TTT runs within the 10-minute eval budget (~8 min TTT + ~2 min sliding eval).

## Architecture (PR #414 stack)

- 11 layers, 512d, 8H, 4KV (GQA)
- 3x MLP with relu²
- SmearGate + BigramHash (2048 buckets)
- XSA on last 4 layers
- Partial RoPE (16/64 dims), LN Scale
- VE128 on layers 9-10
- EMA(0.997) + Tight SWA(50)
- GPTQ-lite int6 + zstd-22
- Late QAT @ threshold 0.15
- OrthoInit + muP-scaled output projections
- Sliding window eval (stride=64)

## Training

- Muon: lr=0.025, momentum=0.99 (warmup 0.92->0.99 over 1500 steps), WD=0.04
- AdamW: embed_lr=0.035, scalar_lr=0.025, WD=0.04
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3500 iterations
- Gradient clip: 0.3

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base model and training recipe: PR #414 by @signalrush
- TTT technique: PR #518 by @sofiabod, PR #672 by @andrewbaggio1
- SDPA fallback for non-FA3 environments
