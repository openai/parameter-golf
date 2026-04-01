# SP4096 Int6+zstd 10L496 Overtone+Phase Sliding64

This record captures a compliant 10-minute / 16MB run with exact final metric:
- `final_int8_zstd_roundtrip_exact val_loss:2.70393625 val_bpb:1.17528238`
- `Total submission size int8+zstd: 14672752 bytes`

## Configuration
- Data/tokenizer: `fineweb10B_sp4096` + `fineweb_4096_bpe.model`
- Layout: `VOCAB_SIZE=4096 NUM_LAYERS=10 MODEL_DIM=496 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Eval: sliding window `EVAL_STRIDE=64 EVAL_BATCH_SEQS=256`
- Quant/export: `QUANT_LEVELS=31` (int6-style values in int8 container), `COMPRESSION_CODEC=zstd`, `COMPRESSION_LEVEL=22`
- Precision keep: `FP16_EMBED=1`, `FP16_PASSTHROUGH_NAMES=tok_emb.weight`
- Training: `MAX_WALLCLOCK_SECONDS=600`, `MATRIX_LR=0.04`, `SCALAR_LR=0.04`, `TIED_EMBED_LR=0.10`, `MUON_MOMENTUM=0.95`, `MUON_WD=0.02`, `WARMDOWN_ITERS=2500`
- Init toggles: `OVERTONE_INIT=1`, `PHASE_RESID_MIX_INIT=1`
- QAT mode: `QAT_START_FRAC=1.1` (off)

## Command Used
```bash
modal run run_modal.py \
  --command train-8gpu \
  --variant sp4096 --vocab-size 4096 \
  --num-layers 10 --model-dim 496 --mlp-mult 2 \
  --compression-codec zstd --compression-level 22 \
  --fp16-embed 1 --fp16-passthrough-names tok_emb.weight \
  --qat-start-frac 1.1 --quant-levels 31 \
  --matrix-lr 0.04 --scalar-lr 0.04 --tied-embed-lr 0.10 \
  --muon-momentum 0.95 --muon-wd 0.02 --warmdown-iters 2500 \
  --eval-stride 64 --eval-batch-seqs 256 \
  --overtone-init 1 --phase-resid-mix-init 1
```

## Key Metrics (from `train.log`)
- Stop step: `9365/20000` at wallclock cap (`600037ms`)
- Pre-quant eval at stop: `val_loss:2.7424`, `val_bpb:1.1920`
- Post-quant roundtrip exact: `val_loss:2.70393625`, `val_bpb:1.17528238`
- Serialized model int8+zstd: `14607000 bytes` (`payload_ratio:3.41x`)
- Code size: `65752 bytes`
- Total submission size int8+zstd: `14672752 bytes`

## Included Files
- `train_gpt.py`: code snapshot used for this run
- `train.log`: exact Modal run log
- `submission.json`: metadata for leaderboard ingestion
