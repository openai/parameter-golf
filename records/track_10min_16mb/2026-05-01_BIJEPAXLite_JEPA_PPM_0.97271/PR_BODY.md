# BIJEPAX-lite JEPA + SP8192 CaseOps PPM

This record submits a Claude-designed, JEPA-inspired training-only auxiliary regularizer on top of the SP8192 CaseOps + per-group compression + PPM sliding stack.

The final 3-seed mean is:

```text
ppm_sliding val_bpb: 0.97271454
```

## Results

| Seed | Final `ppm_sliding val_bpb` | Quantized diagnostic | Artifact bytes | Train stop | Eval time | Exit |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | `0.97234287` | `1.11544494` | `15,997,180` | `2014` steps / `599.843s` | `502.131s` | `0` |
| 314 | `0.97206308` | `1.11562304` | `15,999,539` | `2012` steps / `599.586s` | `499.038s` | `0` |
| 999 | `0.97373767` | `1.11757370` | `15,997,593` | `2013` steps / `599.821s` | `496.384s` | `0` |

Three-seed sample std: `0.00089703`.

All three runs are under:

- strict decimal `16,000,000` byte artifact cap
- 600s training cap
- 600s evaluation cap

## What is new

BIJEPAX-lite adds a small custom JEPA-style hidden-state prediction objective during training:

- hop-4 forward hidden-state prediction
- hop-4 backward hidden-state prediction
- cosine embedding-space loss
- LayerNorm-stabilized predictor heads
- no cycle head in the submitted lightweight config
- active only from `35%` to `80%` of the wallclock schedule
- separate optimizer and separate module from the base GPT

The predictor heads are **not serialized**. Final scoring is performed by the quantized base model with the existing causal PPM sliding evaluator.

## Compliance notes

- `TTT_ENABLED=0`
- `LQER_TOP_K=1` keeps all seeds below the strict byte cap
- SmearGate BOS masking is present for packed-document cross-boundary safety
- BIJEPAX-lite trains only on training batches from `DocumentPackingLoader`
- BIJEPAX-lite does not access validation tokens or validation byte sidecars during training
- Final score is from `ppm_sliding`

The folder includes:

- `train_gpt.py`
- three seed logs
- full source/log captures for each seed
- `submission.json`
- `LEGALITY_AUDIT.md`
- `STATIC_AUDIT_NOTES.md`
- `REFERENCES.md`
- `JEPA.mp4` as a short visual/demo asset

## Acknowledgements

Thanks to Claude for designing the custom BIJEPAX-lite auxiliary objective and helping turn the JEPA idea into a runnable candidate. Thanks to Codex for implementing the run path, auditing legality, coordinating the 3-seed package, and assembling this PR. Thanks also to the Parameter Golf community for the public ideas and fast iteration that this stack builds on.

## Validation

- `python3 -m py_compile records/track_10min_16mb/2026-05-01_BIJEPAXLite_JEPA_PPM_0.97271/train_gpt.py`
- `python3 -m json.tool records/track_10min_16mb/2026-05-01_BIJEPAXLite_JEPA_PPM_0.97271/submission.json`
- 3 full remote runs on 8xH100 completed with `rc=0`
