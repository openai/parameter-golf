# Notable Non-Record Submission: 0.5755 BPB — Conker-5 Tandem Residual Exact Experts (MLX)

**Tandem-trained `Conker-3` base + sparse exact residual experts (`exact1/2/3`, `delim2`, `special2`, `number2`, `markup2`, `attr2`) with gate-only learned selection, packaged as `int6+zlib`.**

**val_bpb: 0.5755** | **3,811,521 bytes total** | **artifact: 3,720,359 bytes** | local MLX / Apple Silicon

> **This is a non-record submission.** The run is packaged and self-contained under the 16,000,000-byte artifact limit, but it is not submitted as a 10-minute `8xH100` leaderboard record. The result here is from the packaged MLX run in this folder.

## Results

| Metric | Value |
|--------|-------|
| Full held-out fp16 `val_bpb` | `0.5718` |
| Full held-out `int6` `val_bpb` | **`0.5755`** |
| Full held-out `int6` `val_loss` | `0.9717` |
| Validation tokens | `62,021,632` |
| Slice `val_bpb` | `0.5652` |
| Train time | `98.2s` |
| Parameters | `177,364,117` |
| Packed payload estimate | `6,801,706` bytes |
| Packed raw serialized bytes | `8,956,303` |
| Packed artifact bytes (`zlib`) | `3,720,359` |
| Code bytes | `91,162` |
| Total counted bytes | `3,811,521` |

Generated artifacts:
- [submission.json](submission.json)
- [results.json](results.json)
- [train.log](train.log)
- `conker5_tandem_nonrecord.int6.ptz`

## What This Is

This branch starts from the `Conker-3` short-memory oscillatory reservoir model, but instead of freezing it and adding hand-tuned corrections, it trains the base and residual stack together.

The structure is:
- a tandem-trained `Conker-3` smooth predictor
- sparse exact-context residual sources
- learned gate-only selection over those sources

The residual experts are narrow and cheap:
- `exact1`, `exact2`, `exact3`
- delimiter continuation (`delim2`)
- rare / identifier-like continuation (`special2`)
- number-like continuation (`number2`)
- markup continuation (`markup2`)
- attribute/value continuation (`attr2`)

The additive residual interface matters. Direct probability mixing was unstable; sparse residual correction was not.

## Why It Works

The central lesson from this line is that exact-history correction is the real engine.

The tandem base provides:
- smooth temporal prediction
- distributed local context
- a strong default prior

The residual stack adds:
- exact continuation reuse
- discrete symbolic structure
- targeted corrections from orthogonal evidence

This submission uses learned gating only to decide which sparse experts to trust. It does not replace the explicit sparse expert maps with dense learned heads.

## Artifact Budget

The artifact is small because only the trainable state is stored.

Most of the trainable capacity lives in:
- the base linear readout
- the base local readout
- the base embeddings

The explicit residual experts and gates are almost free in byte terms.

The final packaged model uses:
- `int6` quantization for large trainable tensors
- fp16 retention for smaller tensors
- `zlib` compression for the final artifact

This run stayed well under the 16,000,000-byte cap:
- model artifact: `3,720,359` bytes
- counted code: `91,162` bytes
- total: `3,811,521` bytes

## Validity Notes

This submission is intended to be boringly valid:
- score is from the full held-out `fineweb_val_*` split
- no sliding-window eval tricks
- no test-time training on validation tokens
- no network calls during evaluation
- packed score is reported from the packaged `int6` artifact, not inferred from fp16

The broader local research branch also ran hostile checks such as reverse/shuffle validation and fresh-process checkpoint re-evaluation before packaging this submission.

## Run

From this folder:

```bash
python3 train_gpt.py
```

Environment variables can override the default recipe, including:
- `SEED`
- `SEQ_LEN`
- `BATCH_SIZE`
- `ITERATIONS`
- `LEARNING_RATE`
- `QUANT_BITS`

The default packaged recipe used for this submission is:
- `seed=43`
- `seq_len=256`
- `batch_size=16`
- `steps=1000`
- `learning_rate=5e-4`
- `quant_bits=6`

## Status

This is the first packaged non-record `Conker-5` tandem submission, not the final frontier of the line.

Later local validity work found that the tandem branch still improves with more training:
- `256 / 1200` improved further
- `256 / 1500` improved substantially

Those later rows are not claimed here because this record folder is meant to package one concrete, self-contained run cleanly.
