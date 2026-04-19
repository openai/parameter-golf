# Submission: SP8192 + QK5 + Freeze10 Loss-Gated Legal TTT - val_bpb 1.08886

**val_bpb: 1.08885521** | **2.81262609 nats** | **15,994,383 bytes** | **train 588.005 s** | **eval 211.815 s**

This submission is a legal 8xH100 / 10 minute / 16 MB result on the `SP8192 + QK5 + LegalTTT` family with a selective test-time adaptation change:

- freeze the first 10 transformer blocks during TTT
- adapt only the final block during TTT
- use a running-mean loss gate to skip low-value update windows

Important scope note:
- this is a **single-seed** submission
- it satisfies the track runtime and size constraints
- it is **not** presented as a new SOTA claim against the current best merged leaderboard result

## Result

| Metric | Value |
|---|---:|
| final `legal_ttt_exact val_bpb` | `1.08885521` |
| final `legal_ttt_exact val_loss` | `2.81262609` |
| final `legal_ttt_exact eval_time` | `211815 ms` |
| train wallclock | `588005 ms` |
| quantized `val_bpb` | `1.10598721` |
| quantized sliding `val_bpb` | `1.08940375` |
| code size | `18203 bytes` |
| quantized+brotli model size | `15976180 bytes` |
| total submission size | `15994383 bytes` |

Relative reading:
- beats the naive baseline `1.2244` by about `0.1355 bpb`
- stays under the 16,000,000-byte cap by `5617 bytes`
- stays under the 600-second train cap
- stays under the 600-second eval cap

## Method

Base family:
- `records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828`

This submission keeps the same core family cues:
- SentencePiece BPE 8192
- 11 layers / 512 dim / 8 heads / 4 KV
- QK gain init `5.0`
- loop structure and SP8192 substrate
- GPTQ int6 matrices + int8 embeddings
- legal score-first TTT semantics

The main change is in the TTT update policy:
- `TTT_FREEZE_BLOCKS=10`
- `TTT_PARAM_MODE=all`
- `TTT_LOSS_GATE_MODE=running_mean`
- `TTT_LOSS_GATE_MARGIN=0.0`

In plain language:
- the model still scores each chunk before adapting on that chunk
- but it only adapts the final block
- and it skips update windows that do not clear the running loss-gate threshold

## Why this submission matters

This run is useful even without a new-SOTA claim because it demonstrates all of the following together on one legal run:
- no timeout in distributed `legal_ttt_exact`
- train and eval both under the official 600-second limits
- artifact under 16 MB
- selective loss-gated TTT working on the PR1413-style stack

It is also a concrete proof that the earlier byte overage was a packaging issue, not a model-size issue:
- previous code-size-inflated run: `50811` code bytes
- this corrected run: `18203` code bytes

## Compliance

- `artifact_under_16mb`: true
- `training_under_600s`: true
- `eval_under_600s`: true
- `score_first_ttt`: true
- `single_seed_only`: true
- `record_claim`: false

## Platform

- hardware: `8xH100 80GB`
- PyTorch: `2.9.1+cu128`
- CUDA: `12.8`
- FlashAttention: `2.8.3`

## Run command

```bash
export QK_GAIN_INIT=5.0
export TTT_ENABLED=1
export TTT_EPOCHS=3
export TTT_LR=0.005
export TTT_MOMENTUM=0.9
export TTT_FREEZE_BLOCKS=10
export TTT_PARAM_MODE=all
export TTT_LOSS_GATE_MODE=running_mean
export TTT_LOSS_GATE_MARGIN=0.0

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

- PR #1394 and its SP8192 / GPTQ / recurrence family surface
- PR #1413 for the `SP8192_QK5_LegalTTT` record folder this builds on
- merged legal score-first TTT precedents in the repo

## Included files

- `README.md`
- `submission.json`
- `requirements.txt`
- `train_gpt.py`
- `train_seed1337.log`
