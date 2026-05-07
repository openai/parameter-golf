# Non-record H200 screening: stabilized phased TTT LR retune

Author: Ayush Ozha (`ayushozha`)

This is a **non-record unlimited-compute screening submission**. It is not a claim for the 10-minute 8xH100 leaderboard because the full validation evidence available at submission time was produced on 1xH200 and the full TTT evaluation exceeded the official 600s wall-clock target on that hardware.

The purpose of this folder is to document a small, reproducible candidate improvement on the current accepted top stack while the official 8xH100 validation jobs are queued.

## Headline result

| run | hardware | seed | setting | val_bpb | val_loss | artifact bytes | eval time |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| H200 artifact build | 1xH200 | 42 | no TTT, quantized | 1.07203421 | 2.34604928 | 15,902,950 | 77.657s diagnostic |
| H200 full TTT evaluation | 1xH200 | 42 | `TTT_LORA_LR=7e-5` | **1.05868018** | 2.31678467 | 15,902,950 | 1892.8s |

The full H200 TTT score comes from `eval_h200_seed42_ttt_lr7e5_full.log`:

```text
quantized_ttt_phased val_loss:2.31678467 val_bpb:1.05868018 eval_time:1892800ms
total_eval_time:1892.8s
```

Velda marks `pg_ttt_lr7e5_full` as `TASK_STATUS_FAILURE` because the log streaming RPC emitted an internal error, but the captured log contains the completed score line above. This folder keeps that status visible rather than presenting it as a clean official run.

## What changed

This submission starts from:

`records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611`

The only intended candidate change is the default phased-TTT LoRA learning rate:

```python
ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.00007))
```

The architecture, tokenizer, CaseOps data path, BOS-fixed SmearGate implementation, LQER stack, per-group lrzip compression, sparse attention gate, rank-80 TTT setting, and phased-prefix schedule are inherited from the accepted top-stack folder.

## Why this is plausible

The current top stack uses aggressive phased test-time training on a quantized artifact. On the H200 artifact, the original `TTT_LORA_LR=1e-4` was strong but slightly noisier in 10 percent validation screens. Lowering the LoRA LR to `7e-5` improved the 10 percent screen and carried through to the full validation pass.

10 percent H200 screens on the same seed-42 artifact:

| TTT setting | 10 percent val_bpb |
| --- | ---: |
| baseline `TTT_LORA_LR=1e-4`, prefix 2500 | 1.05612212 |
| prefix-ratio screen, effective prefix 250 | 1.05615512 |
| `TTT_LORA_LR=7e-5` | **1.05576964** |
| `TTT_LORA_LR=6.5e-5` | 1.05577198 |
| `TTT_LORA_LR=5e-5` | 1.05587670 |
| `TTT_LORA_LR=1.3e-4` | 1.05696493 |
| `TTT_LORA_LR=7e-5`, `TTT_WEIGHT_DECAY=0.3` | 1.05577019 |

The effect is small, so this must be judged by official multi-seed 8xH100 evidence before it can become a record claim.

## Official validation status

As of 2026-04-30, official 8xH100 spot validation has been staged but not completed:

| Velda task | pool | status |
| --- | --- | --- |
| `pg_h100x8_seed42_lr7e5_full` | `anycloud-h100-8-spot` | `TASK_STATUS_QUEUEING` |
| `pg_h100x8_seed0_lr7e5_full` | `anycloud-h100-8-spot` | `TASK_STATUS_PENDING` |
| `pg_h100x8_seed1234_lr7e5_full` | `anycloud-h100-8-spot` | `TASK_STATUS_PENDING` |

If those runs complete under 600s on 8xH100 and reproduce the improvement across seeds, this folder can be promoted or superseded by a proper `track_10min_16mb` record submission with the official logs.

## Reproduction notes

The H200 run used the same environment family as the accepted top-stack folder:

- PyTorch 2.9.1 + CUDA 12.8
- FlashAttention-3 Hopper wheel
- CaseOps SP8192 dataset and reserved tokenizer
- `lrzip` available on `PATH` for `COMPRESSOR=pergroup`

The candidate default in this folder is already set to `TTT_LORA_LR=7e-5`. To reproduce the non-record H200 evaluation from an existing artifact, set `TTT_EVAL_ONLY=1`, point `ARTIFACT_DIR` at the saved artifact directory, and run `python train_gpt.py` with the same dataset/tokenizer paths used by the top-stack submission.

## Compliance notes

- Artifact size: 15,902,950 bytes, under the 16,000,000-byte cap.
- Tokenizer and dataset transform are unchanged from the accepted CaseOps/SP8192 lineage.
- This is not submitted as a new SOTA record because official 8xH100, 3-seed, under-10-minute validation is still pending.
- Included logs are H200 screening logs, not official leaderboard logs.
