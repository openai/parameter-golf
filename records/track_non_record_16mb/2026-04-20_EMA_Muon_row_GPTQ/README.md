# Long-context + EMA + Muon warmup + per-row GPTQ-lite quantization

**Author:** Elad Simbalista  
**GitHub:** @elad-simbalista  
**Track:** Non-record (16 MB)  
**Date:** 2026-04-20

---

## Summary

This is a non-record Parameter Golf submission built from the OpenAI baseline and tuned to beat the naive `1.2244` BPB baseline on FineWeb (`sp1024` tokenizer).

The model keeps the baseline architecture size fixed (9 layers, 512 dim) and improves performance through training schedule changes and post-training quantization rather than scaling parameter count.

---

## Result

Single-seed run on 8×H100 under the 10-minute cap:

- Final eval (live model): `val_bpb = 1.2069`
- EMA final eval: `val_bpb = 1.2033`
- **Post-quant roundtrip (submission score): `val_bpb = 1.2098`**
- Artifact size: `15,872,012` bytes
- Training stopped at step `10189` due to wallclock limit

The submission score reported in `submission.json` corresponds to the **post-quant roundtrip evaluation**, which reflects the exported artifact.

---

## Main Changes vs Baseline

1. **Longer context:** `TRAIN_SEQ_LEN=2048`
2. **Muon momentum warmup:** `0.9 → 0.985` over 500 steps
3. **Extended warmdown:** `3000` iterations
4. **EMA weight averaging:** `decay=0.997`
5. **Per-row GPTQ-lite int8 quantization**
6. **Wallclock-aware training schedule**

---

## How to Run

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional overrides (not required for submission):

```bash
ITERATIONS=12000 TRAIN_SEQ_LEN=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Configuration (submission run)

- `train_batch_tokens = 524288`
- `train_seq_len = 2048`
- `iterations = 12000`
- `warmdown_iters = 3000`
- `max_wallclock_seconds = 600`
- `num_layers = 9`
- `model_dim = 512`
- `num_heads = 8`
- `num_kv_heads = 4`
- `mlp_mult = 2`
- `tied_embeddings = 1`
- `matrix_lr = 0.04`
- `scalar_lr = 0.04`
- `tied_embed_lr = 0.05`
- `muon_momentum = 0.985`
- `ema_decay = 0.997`

---

## Files

- `train_gpt.py` — training + export script  
- `submission.json` — metadata  
- `train_log.txt` — full training log  
- `README.md` — this document  

---

## Notes

- This is a **non-record submission**
- Single-seed only (no statistical claim)
- Fully self-contained training + export pipeline
