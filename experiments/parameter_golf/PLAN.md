# Parameter Golf Strong Submission Plan

## Objective
Beat the current 10min/16MB SOTA by combining:
- top training recipe (10L, Muon WD, fp16 tied-embedding export)
- stronger evaluation (sliding and LoRA TTT)
- statistically valid multi-seed comparisons.

## Implemented in `train_gpt.py`
- `FINAL_EVAL_MODE=standard|sliding|ttt`
- `EVAL_SEQ_LEN`, `EVAL_STRIDE`, `EVAL_BATCH_SEQS`
- `MUON_WEIGHT_DECAY` (decoupled in Muon optimizer)
- `INT8_ALWAYS_KEEP_FLOAT_NAME_PATTERNS` (default keeps `tok_emb.weight` in fp16)

## Execution Stages
1. Reproduce top-like training quality (single seed smoke, then 3 seeds)
2. Compare final eval modes on same checkpoint family:
   - `standard`
   - `sliding` with `EVAL_STRIDE=64`
   - `ttt` with chunk sweep (`TTT_CHUNK_SIZE=256,128,64`)
3. Promote best eval setup; run 3+ seeds with fixed config
4. If mean improves >= 0.005 nats and p<0.01, package submission

## Recommended Baseline Config
- `NUM_LAYERS=10`
- `MODEL_DIM=512`
- `NUM_HEADS=8 NUM_KV_HEADS=4`
- `MATRIX_LR=0.04`
- `MUON_WEIGHT_DECAY=0.02`
- `WARMDOWN_ITERS=2500`
- `TIED_EMBED_LR=0.10`
- `INT8_ALWAYS_KEEP_FLOAT_NAME_PATTERNS=tok_emb.weight`

## Promotion Criteria
- Primary: `final_*_exact val_loss`
- Secondary: `val_bpb`, eval runtime
- Hard constraints: code+artifact < 16,000,000 bytes and valid significance test vs prior best.
