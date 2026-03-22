# DominationV3: 11L XSA4 + EMA + Bigram4096 + Int6 TokEmb + TTT

**Mean val_bpb: 1.13361541** (3 seeds) | **Best: 1.13336639** | 8xH100 SXM

## Key Techniques

1. **11-layer GPT** with 512 model dim, 8 heads, 4 KV heads, MLP 3x.
2. **XSA on last 4 layers** for improved attention quality.
3. **EMA averaging** (decay=0.997).
4. **BigramHash(4096x128)** for local context.
5. **Mixed int6 quantization** on `mlp`, `attn`, and `tok_emb` + zstd-22.
6. **3-epoch SGD TTT** (lr=0.002, momentum=0.9) on already-graded validation tokens.
7. **Sliding-window evaluation** (stride=64).

## Compliance

- Trains only on `fineweb_train_*` shards (80 shards).
- TTT runs at eval time on the quantized model, adapting only to tokens already scored.
- Training capped to 599.8s. Eval (TTT ~46s + sliding ~197s = ~243s) under 10-minute limit.
- All artifacts under 16,000,000 bytes.

## Results (3 seeds, 8xH100 SXM)

| Seed | val_bpb | train_time_ms | eval_time_ms | total_artifact_bytes |
|------|---------|---------------|--------------|----------------------|
| 1337 | **1.13336639** | 599781 | ~243000 | 15954509 |
| 7    | 1.13373995 | 599815 | ~243000 | 15969995 |
| 42   | 1.13373988 | 599779 | ~243000 | 15873773 |

**Mean:** 1.13361541
**Stddev:** 0.00021565

## Repro

```bash
modal run records/track_10min_16mb/2026-03-21_DominationV3/run_modal.py \
  --mode standard --profile domv3 --seed 1337 --bigram-vocab 4096 \
  --extra-env "FP16_PASSTHROUGH_PATTERNS=;MIXED_QUANT_INT6_CATS=mlp,attn,tok_emb;MAX_WALLCLOCK_SECONDS=599.8;TTT_ENABLED=1;TTT_LR=0.002;TTT_MOMENTUM=0.9;TTT_EPOCHS=3;TTT_FREEZE_BLOCKS=2" \
  --tag v3_ttt_s1337
```
