Adds the single primary non-record 16MB submission from the deadline search.

We did not get access to an 8xH100 box before the deadline, so this is intentionally submitted to the non-record / unlimited-compute track rather than as a leaderboard-record claim. The goal of this PR is to submit the strongest completed result and preserve the supporting 4xH100 batch evidence.

Search process:
- Started with 12 implementation-backed candidates on 1xH100.
- Promoted the four most promising completed 1xH100 candidates into a 4xH100 batch.
- Could not take the strongest 4xH100 candidate further to a compliant 8xH100 / 3-seed record attempt before the deadline.

The 1xH100 screening set covered a mix of conservative controls and more speculative implementation-backed ideas:

| Candidate family | What it tested |
|---|---|
| baseline AR / non-quant reference | Control runs for the current autoregressive stack and artifact path |
| QK-gain / QK RMSNorm / long-context controls | Whether stronger attention scaling or context handling improved the small SP1024 setup |
| parallel residual | Whether computing attention and MLP residual branches in parallel improved optimization or throughput |
| depth recurrence / looped layers | Whether reusing layers improved quality under the 16MB artifact budget |
| SwiGLU / wide-shallow / narrow-deep variants | Basic architecture and MLP-shape alternatives |
| JEPA / text diffusion / SSM proxy ideas | Early signs-of-life for requested nonstandard approaches, implemented as supported proxy variants |
| MoE proxy | Whether sparse/wider expert-style capacity helped within the artifact budget |

The 4xH100 promotion batch then focused only on the strongest QK5.25-style candidates from that screen:

| Candidate | val_bpb | Defining change |
|---|---:|---|
| `best4x_ttt_disabled_qk525` | 1.26066159 | QK5.25, TTT disabled |
| `best4x_qk525_legal_ttt` | 1.26359539 | QK5.25, legal score-first TTT |
| `best4x_parallel_residual_ttt_qk525` | 1.27689523 | QK5.25, parallel residual, legal TTT |
| `best4x_dense_optimizer_base` | 1.27771744 | QK5.25, parallel residual, TTT disabled, dense optimizer settings |

Primary packaged result:
- candidate_id: `best4x_ttt_disabled_qk525`
- val_bpb: `1.26066159`
- track: `non-record-unlimited-compute-16mb`
- hardware: `4x NVIDIA H100 80GB HBM3`
- artifact: under 16MB

Approach:
- Uses the QK-gain 5.25 autoregressive control stack.
- Primary candidate disables test-time training, testing whether saved runtime is more valuable than legal score-first TTT for this smaller SP1024 stack.
- The promoted 4x batch compares this against legal TTT, parallel-residual + legal TTT, and a dense optimizer/tuning baseline.

Interpretation:
- The 4xH100 run substantially improves over the corresponding 1xH100 source result, but remains far from the current record frontier.
- In this setup, disabling TTT slightly beats legal TTT, suggesting that the TTT runtime cost is not repaid by lower bpb for this particular stack and time budget.
- Parallel residuals did not help in this ablation.

Included:
- `submission.json`
- self-contained `train_gpt.py`
- primary train log and result JSON
- `batch_leaderboard.csv`
- `batch_summary.md`
- `batch_results.json`

Validation:
- `python3 -m json.tool records/track_non_record_16mb/2026-05-01_best4x-ttt-disabled-qk525_1p26066/submission.json >/dev/null`
- `python3 -m py_compile records/track_non_record_16mb/2026-05-01_best4x-ttt-disabled-qk525_1p26066/train_gpt.py`
- `git diff --check`
