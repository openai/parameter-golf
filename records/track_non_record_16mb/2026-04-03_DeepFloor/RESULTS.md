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
- remote fullbox suite on `8x H100`, with the resulting artifacts synced back locally under `runs/fullbox/`

Checked-in submission accounting for the candidate result:

- `bytes_model_estimated = 8448`
- `bytes_code = 48029`
- `bytes_total = 56477`

Representative preflight accounting for the frozen submission surface:

- `bytes_model_estimated = 8192`
- `bytes_code = 48029`
- `bytes_total = 56221`

## Current limitations

- We now have an 8xH100 DeepFloor search run, but it is an evolutionary recipe-search campaign, not `3` repeated fixed-candidate 10-minute submission runs.
- The current candidate uses only a 1-step small-box training/eval configuration for the checked-in result.
- The record folder is PR-ready as a non-record submission candidate, but it still needs stronger final evidence before we should claim a competitive record attempt.

## Fullbox search evidence

The fullbox suite completed on `8x H100` and the full artifact set is checked in locally under `runs/fullbox/`.

Best `frontier` runs:

| Run | Mode | State Core | Dim | Views | Eval Steps | Val BPB | Test BPB | Artifact MB |
|-----|------|------------|----:|------:|-----------:|--------:|---------:|------------:|
| `frontier_seed2025` | floor | scalar_decay | 96 | 8 | 64 | **4.1101** | **4.0239** | 0.1377 |
| `frontier_seed4242` | floor | scalar_decay | 128 | 8 | 64 | 4.3356 | 4.5996 | 0.5156 |
| `frontier_seed1337` | floor | scalar_decay | 96 | 8 | 128 | 4.4156 | 4.4372 | 0.1201 |
| `frontier_seed5151` | fused | hippo | 128 | 8 | 512 | 5.2129 | 5.0534 | 0.1602 |

Best `compact` runs:

| Run | Mode | State Core | Dim | Views | Eval Steps | Val BPB | Test BPB | Artifact MB |
|-----|------|------------|----:|------:|-----------:|--------:|---------:|------------:|
| `compact_seed4242` | fused | scalar_decay | 64 | 2 | 16 | **5.1777** | 5.5187 | 0.0410 |
| `compact_seed2025` | fused | hippo_plus_lowrank | 64 | 1 | 16 | 5.2328 | 5.2732 | 0.0334 |
| `compact_seed5151` | fused | scalar_decay | 64 | 2 | 16 | 5.3222 | 5.5699 | 0.0205 |
| `compact_seed1337` | fused | scalar_decay | 64 | 2 | 16 | 5.3619 | 5.4040 | 0.0244 |

The strongest observed signal from the fullbox run is that the larger `frontier` search meaningfully improves over the small-box candidate, and that the best discovered recipes shift toward `floor` mode with larger recurrent dims and wider multi-view configurations.
