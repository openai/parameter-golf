# Invalidated Submission: PR #998 — Conker-5 Tandem Residual Exact Experts (MLX)

**Historical packaged score:** `0.5755 val_bpb` at `3,811,521` counted bytes.  
**Current status:** invalid as a causal submission.

This folder preserves the original packaged run, but it should not be read as a valid example anymore.

The later artifact audit was decisive:
- the extracted tandem causal mask carries forbidden mass in the upper triangle / diagonal: `upper_plus_diag_frac = 0.04358700722704721`
- the corresponding strict causal mask is clean: `upper_plus_diag_frac = 0.0`
- forcing that strict mask collapses the score from the tandem line to `2.0971244136143423 bpb`

That is the whole story. The score was real. The package was real. The causal cleanliness was not. If the result dies when you enforce the rule it was supposed to satisfy, the side channel was doing the work.

## Independent Convergence

What matters here is not one reconstruction. It is that the same conclusion now survives from multiple directions:
- detector-side audit says the saved artifacts carry forbidden structure
- ledger-side packaging says the public backlog and attached artifacts tell the same story
- the same conclusions survived an independent rebuild of the tooling

The strongest public signals are:
- tandem `Conker-5` causal mask is visibly non-strict: `upper_plus_diag_frac = 0.04358700722704721`
- the extracted strict mask is actually clean: `upper_plus_diag_frac = 0.0`
- strict vs tandem mask drift is large: `max_abs_deviation = 0.5975669622421265`
- saved `Conker-6` mask also carries future structure: `upper_frac = 0.011201489739837839`, `diag_frac = 0.017354798229237627`
- Toeplitz substitution explodes `full_test_bpb` from `0.07209327818598087` to `5.752106388513692`

The broader interpretation is no longer private speculation. The old `Conker` frontier line was contaminated by trainable structural buffers, the absurd `Conker-6` row was structural leakage rather than a compression miracle, and the strict rebuild collapse is real rather than a tooling artifact.

For the public postmortem and attached audit bundle, see:
- `conker-ledger`: <https://github.com/asuramaya/conker-ledger/tree/main/examples/conker-artifact-quick-check-2026-03-28>
- `conker-detect`: <https://github.com/asuramaya/conker-detect>

## Historical Packaged Result

These numbers are left here as a record of what the submission claimed at the time:

| Metric | Value |
|--------|-------|
| Full held-out fp16 `val_bpb` | `0.5718` |
| Full held-out `int6` `val_bpb` | `0.5755` |
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

## What The Model Was

This line used:
- a tandem-trained `Conker-3` smooth base
- sparse exact-context residual experts
- learned gate-only routing over those experts

The expert set included:
- `exact1`, `exact2`, `exact3`
- `delim2`
- `special2`
- `number2`
- `markup2`
- `attr2`

The original claim was that exact-history residual correction plus a smooth learned prior was enough to produce the packaged score cleanly. That claim is no longer credible for this tandem branch because the saved artifact itself failed the later causal audit.

## Artifact Budget

The packaging facts remain true even though the validity claim does not:
- model artifact: `3,720,359` bytes
- counted code: `91,162` bytes
- total: `3,811,521` bytes

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

The default packaged recipe used for this submission was:
- `seed=43`
- `seq_len=256`
- `batch_size=16`
- `steps=1000`
- `learning_rate=5e-4`
- `quant_bits=6`
