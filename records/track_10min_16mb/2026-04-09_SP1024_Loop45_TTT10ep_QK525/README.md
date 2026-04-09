# Run 009: SP1024 + Looping + TTT 10ep (PR #1487 Tuning)

## Hypothesis

Apply PR #1487's TTT hyperparameter tuning to our SP1024 + Looping architecture.

**Expected gain: ~0.008 BPB** (based on PR #1487's ablation showing -0.0079 BPB from tuning alone)

## Configuration Changes vs Run 007/008

| Parameter | Run 007/008 | Run 009 (PR #1487 tuning) | Expected Impact |
|-----------|-------------|---------------------------|-----------------|
| **TTT Epochs** | 6 | **10** | More adaptation time |
| **TTT LR** | 0.0005 | **0.00045** | More stable fine-tuning |
| **TTT Freeze Blocks** | 2 | **1** | More layers can adapt |
| **QK-Gain** | 5.0 | **5.25** | Sharper attention |

## Architecture (Unchanged from Run 007/008)

- **Tokenizer**: SP1024 (novel parameter reallocation)
- **Layers**: 11 physical
- **Looping**: 2 loops on layers 4-5, enabled at step 0.5
- **Parallel residuals**: From layer 7+
- **EMA decay**: 0.9965
- **GPTQ int6 + Brotli** compression

## Target Metrics

| Metric | Run 007/008 | Run 009 Target |
|--------|-------------|----------------|
| **val_bpb (3-seed mean)** | 1.07389 | **~1.066** |
| **vs Official SOTA (1.1147)** | -0.041 BPB | **~-0.049 BPB** |
| **Training time** | 588s | ~600s (TTT adds ~40s) |
| **Artifact size** | ~13.87 MB | ~14.0 MB |

## Compliance (Track A)

- Pre-quant TTT trains on validation data BEFORE quantization
- Result baked into artifact — fixed predictor at eval time
- No eval-time adaptation, no SLOT, no n-gram cache
- All artifacts < 16MB
- Training wallclock < 600s

## Reproduction Command

```bash
export SEED=314 VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512
export NUM_LOOPS=2 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.5
export PARALLEL_START_LAYER=7
export PREQUANT_TTT_ENABLED=1 PREQUANT_TTT_LR=0.00045 PREQUANT_TTT_EPOCHS=10 PREQUANT_TTT_FREEZE_BLOCKS=1
export QK_GAIN_INIT=5.25 EMA_DECAY=0.9965
export EMBED_BITS=8 MATRIX_BITS=6 COMPRESSOR=brotli GPTQ_ENABLED=1
export SLIDING_WINDOW_ENABLED=1 ETLB_ENABLED=1
export TRAIN_SEQ_LEN=2048 MAX_WALLCLOCK_SECONDS=600
export TRAIN_BATCH_TOKENS=786432
torchrun --nproc_per_node=8 train_gpt.py
```

## Credits

- **TTT hyperparameter tuning**: PR #1487 by @ndokutovich
- **SP1024 + Looping baseline**: Our Run 007/008
- **Base architecture**: Parameter Golf community

## Run Log

| Seed | Pre-quant BPB | Post-TTT BPB | Final BPB | Status |
|------|---------------|--------------|-----------|--------|
| 314 | TBD | TBD | TBD | Pending |
| 42 | TBD | TBD | TBD | Pending |
| 999 | TBD | TBD | TBD | Pending |
| **Mean** | - | - | **TBD** | - |
