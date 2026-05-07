# RunPod Results: 1xH100 Budget Runs

Date: 2026-05-07 UTC

Hardware:

- 1x NVIDIA H100 SXM 80GB
- RunPod on-demand pod
- PyTorch `2.11.0+cu130`
- FlashAttention 3

Data:

- SP4096 tokenizer/data from `kevclark/parameter-golf`
- `86` train shards available on the pod
- Full validation shard

Baseline run:

```bash
DATA_DIR=../../../data \
RUN_ID=sp4096_1xh100_seed42_86shards_retry \
SEED=42 \
MAX_WALLCLOCK_SECONDS=3600 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Training stopped cleanly at the wallclock cap:

```text
stopping_early: wallclock_cap train_time: 3590665ms step: 3975/20000
pre-quantization post-ema val_loss:2.57429177 val_bpb:1.11875755
```

The unmodified quantized artifact was slightly over the 16,000,000 byte cap:

```text
Serialized model int6+brotli: 15995984 bytes
Code size: 68206 bytes
Total submission size int6+brotli: 16064190 bytes
final_int6_sliding_window val_loss:2.55503271 val_bpb:1.11038778
```

Valid artifact salvage:

- Coarsened only `ve_shared.embed.weight.q` by halving integer codes and doubling its scale.
- Wrote `final_model.valid.ptz`.
- This reduces compressed model bytes enough to fit under the cap with negligible BPB movement.

Valid result:

```text
valid_blob_bytes:15917702
code_bytes:70187
total_bytes:15987889
valid_roundtrip val_loss:2.59734720 val_bpb:1.12877717
valid_sliding val_loss:2.55513338 val_bpb:1.11043153
```

Local copied artifacts:

- `sp4096_1xh100_seed42_86shards_retry.txt`
- `final_model.valid.ptz`

## QK 4.5 Improvement

Follow-up run on the migrated RunPod pod:

```bash
DATA_DIR=../../../data \
RUN_ID=sp4096_1xh100_seed42_86shards_qk45 \
SEED=42 \
QK_GAIN_INIT=4.5 \
MAX_WALLCLOCK_SECONDS=3600 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Training stopped cleanly at the wallclock cap:

```text
stopping_early: wallclock_cap train_time: 3590178ms step: 4271/20000
pre-quantization post-ema val_loss:2.56785422 val_bpb:1.11595986
```

The first one-pass artifact fit was still over cap:

```text
Artifact fit: coarsened ve_shared.embed.weight
Serialized model int6+brotli: 16041069 bytes
Total submission size int6+brotli: 16111256 bytes
WARNING: artifact exceeds cap:16111256>16000000
final_int6_roundtrip val_loss:2.59018066 val_bpb:1.12566268
final_int6_sliding_window val_loss:2.54796393 val_bpb:1.10731577
```

After patching the guard to repeatedly coarsen the selected tensor, the salvaged valid artifact is:

```text
valid_blob_bytes:15916816
code_bytes:70379
total_bytes:15987195
valid_roundtrip val_loss:2.59046517 val_bpb:1.12578632
valid_sliding val_loss:2.54823542 val_bpb:1.10743376
```

Local copied artifacts:

- `sp4096_1xh100_seed42_86shards_qk45.txt`
- `sp4096_qk45_valid_eval.txt`
- `final_model.qk45.valid.ptz`
