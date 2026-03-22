# DominationV3 Compact: 11L XSA4 + EMA + Bigram4096 + Int6 TokEmb

**Mean val_bpb: 1.13488752** (3 seeds) | **Best: 1.13446677** | 8xH100 SXM

This submission is a compact no-TTT path optimized for strict 10-minute compliance on the standard 16MB track.

## Key Techniques

1. **11-layer GPT** with 512 model dim, 8 heads, 4 KV heads, MLP 3x.
2. **XSA on last 4 layers** (`XSA_LAST_N=4`) for improved attention quality without extra parameter-heavy branches.
3. **EMA averaging** (`EMA_DECAY=0.997`) instead of SWA.
4. **BigramHash(4096x128)** for richer local context.
5. **Mixed int6 quantization** on `mlp`, `attn`, and `tok_emb`.
6. **No FP16 passthrough** and **no TTT path** (training/evaluation separation is explicit).
7. **Sliding-window evaluation** (`EVAL_STRIDE=64`) with tokenizer-agnostic BPB accounting.

## Compliance Notes

- Trains only on `fineweb_train_*` shards (80 shards); no validation optimization path.
- Validation is used only for loss reporting/evaluation; no optimizer steps occur in eval functions.
- Training wallclock is capped to **599.8s** (strictly under 10 minutes).
- Artifact size uses **compressed model bytes + train_gpt.py bytes** and stays under **16,000,000** bytes.
- Evaluation is under the additional 10-minute limit.
- `train_gpt.py` does not perform network calls during training/evaluation.

## Results (3 seeds, same recipe)

| Seed | val_bpb | train_time_ms | eval_time_ms | total_artifact_bytes |
|------|---------|---------------|--------------|----------------------|
| 1337 | **1.13446677** | 599804 | 197273 | 15789394 |
| 7    | 1.13490170 | 599991 | 197172 | 15780459 |
| 42   | 1.13529408 | 599941 | 197512 | 15809997 |

**Mean:** 1.13488752  
**Stddev (sample):** 0.00041384

Best-seed artifact breakdown:
- `compressed_model_bytes`: 15744258
- `code_bytes`: 45136
- `total_artifact_bytes`: 15789394

## Repro Commands

```bash
modal run records/track_10min_16mb/2026-03-21_DominationV3/run_modal.py \
  --mode standard --profile domv3 --seed 1337 --tag v3_compact_bg4096_i6tok_s1337_safe

modal run records/track_10min_16mb/2026-03-21_DominationV3/run_modal.py \
  --mode standard --profile domv3 --seed 7 --tag v3_compact_bg4096_i6tok_s7

modal run records/track_10min_16mb/2026-03-21_DominationV3/run_modal.py \
  --mode standard --profile domv3 --seed 42 --tag v3_compact_bg4096_i6tok_s42
```

Logs included in this folder:
- `train.log` (seed 1337 strict-safe run)
- `train_seed7.log`
- `train_seed42.log`
