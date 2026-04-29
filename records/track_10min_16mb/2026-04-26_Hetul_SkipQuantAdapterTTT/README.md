# Hetul SkipQuant Adapter TTT

Final promoted method: SP8192 score-first adapter TTT with int4 quantized skip tensors.

The stored model uses the same trained stack as the experiment branch, but quantizes the small `skip_gates` and `skip_weights` tensors to packed int4 during artifact serialization. The saved bytes fund a rank1024 eval-time adapter while keeping the total code + model artifact under 16,000,000 bytes.

At evaluation, each 4096-token chunk is scored first. Only after scoring does the script run 4 epochs of SGD TTT on already-scored tokens. The adapter is a fixed random projection `A` plus a zero-initialized trainable projection `B`; adapter tensors are created at eval time and are not stored in the artifact.

## 1M-Slice Validation

All comparisons used the same 1,048,576-token SP8192 eval slice, fresh matched controls, and seeds 42, 314, and 999.

| Seed | Control BPB | Candidate BPB | Delta | Eval Tokens | Candidate Runtime | 8xH100 Full-Val Runtime Est. |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.36603251 | 1.31100325 | -0.05502926 | 1,048,576 | 81.6s | 394s |
| 314 | 1.37206818 | 1.31391062 | -0.05815756 | 1,048,576 | 83.1s | 401s |
| 999 | 1.36891427 | 1.31021127 | -0.05870300 | 1,048,576 | 84.5s | 408s |
| Mean | 1.36900499 | 1.31170838 | -0.05729661 | 1,048,576 | 83.0s | 401s |

The runtime estimate scales the measured 1M-token single-H100 runtime by `40,540,160 / 1,048,576 / 8`. It is an estimate, not an official full-validation measurement.

Delta-transfer estimate against current top `1.0810`: `1.0810 - 0.05729661 = 1.02370` BPB. This is not a measured leaderboard score.

## Artifact Budget

Final `train_gpt.py` size: 17,545 bytes.

| Seed | Model Bytes | Code Bytes | Total Bytes | Margin |
|---:|---:|---:|---:|---:|
| 42 | 15,980,605 | 17,545 | 15,998,150 | 1,850 |
| 314 | 15,980,750 | 17,545 | 15,998,295 | 1,705 |
| 999 | 15,981,553 | 17,545 | 15,999,098 | 902 |

Max observed total artifact size is 15,999,098 bytes, under the 16,000,000-byte cap.

## Exact Run Command

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 TTT_ADAPTER_SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For the other validation seeds, replace both `SEED` and `TTT_ADAPTER_SEED` with `314` or `999`.

## Compliance Notes

- Score-first legal TTT: each chunk is scored before any adapter or model update.
- No future-token leakage.
- No validation data is used to choose updates during scoring.
- Total artifact is under 16,000,000 bytes for all validated seeds.
- The final submission code omits the exploratory causal-memory and layerwise-adapter branches.

## Included Files

- `train_gpt.py`
- `submission.json`
- `README.md`
- `eval_seed42.log`
- `eval_seed314.log`
- `eval_seed999.log`
