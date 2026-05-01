# Legality audit

## Track constraints

- Training is capped at 600 seconds on 8xH100. The source artifacts stopped at `599.546s`, `599.583s`, and `599.657s`.
- Evaluation is capped at 600 seconds. The final PPM evals took `510.410s`, `500.300s`, and `497.643s`.
- The artifact cap is decimal `16,000,000` bytes. The largest quantized artifact is `15,946,930` bytes; with the current checked-in compressed code wrapper and no local minifier, the largest measured total is `15,995,881` bytes.
- The submitted score uses `TTT_ENABLED=0`; no validation-set gradient update is part of the score.

## PPM causality

The PPM mixer scores each byte from prefix counts and updates the count after scoring the current byte. The gate is computed from already-observed context statistics before incorporating the current target byte.

The byte sidecar is used for BPB accounting and byte-stream scoring alignment. It is not a learned table of validation answers and it is not updated from future bytes.

## Packed document leakage

SmearGate's forward-1 mixing is masked at BOS positions:

```python
not_bos = (input_ids[:, 1:] != BOS_ID).to(x.dtype).unsqueeze(-1)
x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1] * not_bos], dim=1)
```

The same mask is present in both the normal forward path and the TTT forward path.

## Compression and dependencies

The artifact uses per-group `lrzip` compression for grouped int6 tensors and Brotli for the remainder/code wrapper. `lrzip` must be installed in the runtime image before training. The script shells out to an already-installed binary; it does not download packages during evaluation.

## Known review surface

This submission inherits the same review surface as the public SP8192 + byte PPM lane:

- custom SP8192 CaseOps tokenizer/data preparation
- per-token byte sidecar used for exact BPB accounting
- causal PPM eval-time adaptation

The v13-specific final change is only the PPM gate retune to `H=0.999`, `L=0.18`, `T=0.80`.
