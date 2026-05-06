# Non-Record: H100 SXM SP1024 Baseline Probe

**val_bpb: 1.31885410** | **14,061,665 bytes** | 1x H100 SXM | single-seed smoke

This is a deliberately conservative baseline reproduction probe for the Parameter Golf development-grant workflow. It is not a leaderboard attempt and does not improve on the public baseline. The purpose is to provide a complete, auditable first Runpod H100 run with full validation scoring and artifact-size evidence before spending more compute on ablations.

## Result

| Seed | Train shards | Steps | Train wallclock | Final val_bpb | Artifact |
|------|--------------|-------|-----------------|---------------|----------|
| 1337 | 10 | 1,556 | 600.150s | 1.31885410 | 14,061,665 bytes |

Final log lines:

```text
step:1556/20000 val_loss:2.2248 val_bpb:1.3177 train_time:600150ms step_avg:385.70ms
stopping_early: wallclock_cap train_time:600150ms step:1556/20000
Serialized model int8+zlib: 14013979 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 14061665 bytes
final_int8_zlib_roundtrip val_loss:2.2268 val_bpb:1.3189 eval_time:11241ms
final_int8_zlib_roundtrip_exact val_loss:2.22682990 val_bpb:1.31885410
```

## Setup

Runpod pod:

- GPU: 1x H100 SXM (`NVIDIA H100 80GB HBM3`)
- Datacenter: `AP-IN-1`
- Base template: `runpod-torch-v240`
- Runtime upgrade: `torch==2.9.1+cu128`

The stock `runpod-torch-v240` image has PyTorch 2.4, which does not support the `enable_gqa` argument used by the current baseline script. Upgrading to PyTorch 2.9.1 with CUDA 12.8 fixes that compatibility issue.

## Reproduction

From the repository root on a Runpod H100 pod:

```bash
python3 -m pip install -q --upgrade torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -q numpy tqdm huggingface-hub datasets tiktoken sentencepiece typing-extensions==4.15.0
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

RUN_ID=h100_sxm_sp1024_probe_torch291 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Notes

- This run uses only 10 SP1024 training shards as a smoke/probe, not the full default training set.
- Validation still scans the full fixed `fineweb_val_*` split.
- The result is useful as a first verified H100 path and compute-credit evidence, not as a competitive submission.

