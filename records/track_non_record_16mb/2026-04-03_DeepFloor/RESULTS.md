# DeepFloor Results

## Current submission candidate

The current checked-in DeepFloor submission candidate is the best real-`enwik8` small-box matrix run we have so far:

- candidate: `fused_d32_v2`
- device: `1x H100`
- tokenizer: byte-level (`vocab_size=256`)
- cross-token mode: `fused`
- recurrent dim: `32`
- distinct blocks: `1`
- quantization: `ternary`
- artifact: `8,448` bytes estimated model storage
- val_bpb: `7.9221`
- test_bpb: `8.1786`
- bytes_total: `56,477`

This is explicitly a non-record submission candidate today. It demonstrates a reproducible DeepFloor execution surface and a compact recurrent multi-view architecture, not a leaderboard-quality score.

## Small-box matrix

Remote verification on the reused H100 pod produced the following matrix on real `/workspace/data/enwik8`:

| Run | Mode | Dim | Val BPB | Test BPB | Artifact MB |
|-----|------|----:|--------:|---------:|------------:|
| `floor_d32_v2` | floor | 32 | 8.3307 | 8.4970 | 0.0078 |
| `fused_d32_v2` | fused | 32 | **7.9221** | 8.1786 | 0.0081 |
| `floor_d64_v2` | floor | 64 | 8.3321 | 8.1222 | 0.0234 |
| `fused_d64_v2` | fused | 64 | 7.9334 | **7.9010** | 0.0244 |

The best validation score is `fused_d32_v2`, so that is the candidate packaged by `run_submission_candidate.sh`.

## Submission-gate checks

These gates are green:

- local submission preflight against the frozen record-folder snapshot
- remote GPU submission preflight on the reused small pod
- remote small-box suite on real `enwik8`

Checked-in submission accounting for the candidate result:

- `bytes_model_estimated = 8448`
- `bytes_code = 48029`
- `bytes_total = 56477`

Representative preflight accounting for the frozen submission surface:

- `bytes_model_estimated = 8192`
- `bytes_code = 48029`
- `bytes_total = 56221`

## Current limitations

- We do not yet have an 8xH100 end-to-end DeepFloor run proving the 10-minute record track.
- The current candidate uses only a 1-step small-box training/eval configuration for the checked-in result.
- The record folder is PR-ready as a non-record submission candidate, but it still needs stronger final evidence before we should claim a competitive record attempt.
