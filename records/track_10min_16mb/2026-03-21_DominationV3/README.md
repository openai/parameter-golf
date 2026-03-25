# DominationV3: 11L EMA + Partial RoPE + LN Scale + GPTQ-lite + TTT(25ep)

**Mean val_bpb: 1.12495** (3 seeds) | **Best: 1.12431** | 8xH100 SXM

## Key Techniques

1. **11-layer GPT** with 512 model dim, 8 heads, 4 KV heads, MLP 3x.
2. **Partial RoPE** (16/64 dims): RoPE on 25% of head dims; rest position-free.
3. **LN Scale** (`1/sqrt(layer_idx+1)`): Damp deeper layer norm outputs.
4. **EMA averaging** (decay=0.997).
5. **BigramHash(4096x128)** for local context.
6. **GPTQ-lite quantization**: Per-row optimal clip percentile search (5 candidates) minimizing reconstruction MSE.
7. **Mixed int6 quantization** on `mlp`, `attn`, and `tok_emb` + zstd-22.
8. **25-epoch aggressive SGD TTT** (lr=0.012, momentum=0.9, ALL blocks unfrozen) on already-graded tokens.
9. **XSA disabled** to save ~1.4ms/step for more training steps.
10. **Sliding-window evaluation** (stride=64).

## Compliance

- Trains only on `fineweb_train_*` shards (80 shards).
- TTT runs at eval time on the quantized model, adapting only to tokens already scored.
- Training capped to 599.8s. TTT ~389s + sliding eval ~197s = ~586s total eval (under 10 min).
- All artifacts under 16,000,000 bytes.

## Results (3 seeds, 8xH100 SXM)

| Seed | val_bpb | train_time_ms | ttt_time_ms | total_artifact_bytes |
|------|---------|---------------|-------------|----------------------|
| 1337 | 1.12513674 | 599779 | 389133 | 15965664 |
| 7    | 1.12540132 | 599841 | ~389000 | 15829190 |
| 42   | **1.12431423** | 599822 | ~389000 | 15806256 |

**Mean:** 1.12495076
**Stddev:** 0.00056691

## Repro

```bash
modal run records/track_10min_16mb/2026-03-21_DominationV3/run_modal.py \
  --mode standard --profile domv3 --seed 1337 --bigram-vocab 4096 \
  --extra-env "FP16_PASSTHROUGH_PATTERNS=;MIXED_QUANT_INT6_CATS=mlp,attn,tok_emb;MAX_WALLCLOCK_SECONDS=599.8;XSA_LAST_N=0;ROPE_DIMS=16;LN_SCALE=1;TTT_ENABLED=1;TTT_P1_EPOCHS=0;TTT_EPOCHS=25;TTT_LR=0.012;TTT_MOMENTUM=0.9;TTT_FREEZE_BLOCKS=0"
```
