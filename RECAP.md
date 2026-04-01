# Logs

## Recap 2026-03-28

### Notes

#### Curr

- Add a shortened dev validation path for local checks.

```
RUN_ID=mlx_quickcheck \
ITERATIONS=5 \
TRAIN_BATCH_TOKENS=1024 \
GRAD_ACCUM_STEPS=1 \
VAL_LOSS_EVERY=5 \
VAL_BATCH_SIZE=1024 \
VAL_MAX_BATCHES=1 \
SKIP_FINAL_QUANT_EVAL=1 \
python3 train_gpt_mlx.py
```

- Read this model

#### Next

- Edit this model

## Recap 2026-03-20

### Notes

#### Curr

- End-to-end local run works.
- Full validation is too slow on Apple M1.

#### Next

- Add a shortened dev validation path for local checks.

### Environment

```
Chip:   Apple M1
Memory: 16GB
```

### Results

```
- final_int8_zlib_roundtrip val_loss: 3.8937
- final_int8_zlib_roundtrip val_bpb:  2.3061
- eval_time: 2325300 ms (~38.8 min)
```
