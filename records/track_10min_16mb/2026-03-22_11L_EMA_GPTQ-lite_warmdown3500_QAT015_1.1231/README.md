## Record: 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 (val_bpb: 1.1231)

**val_bpb = 1.1231** (sliding window, stride=64) | **15.68 MB** artifact | 8xH100 SXM, 600s

This folder captures a stronger validated `seed=1337` run of the patched `#414`-class control stack on real `8x H100 SXM` hardware. It improves on the earlier validated control run while staying under the `16,000,000` byte cap.

### Comparisons

| Run | Sliding s64 val_bpb | Total bytes |
|---|---:|---:|
| Earlier validated control (`pr414_seed1337_remote`) | `1.12946402` | `15,857,552` |
| Best older checked public control snapshot (`pr-287`) | `1.12707468` | `15,534,645` |
| **This validated run** | **`1.12311898`** | **`15,683,276`** |

### What this run validates

1. The patched `#414`-class stack survives a full competition-faithful `8x H100 SXM` run.
2. This validated run trained materially faster than the earlier remote environment: `7142` steps in `600.065s` versus `6314` in `600.040s`.
3. The faster hardware/software stack translated into a better final sliding-window score while staying safely under the `16,000,000` byte cap.

### Configuration

```bash
RUN_ID=pr414_seed1337_record
SEED=1337
NUM_LAYERS=11
BIGRAM_VOCAB_SIZE=2048
BIGRAM_DIM=128
XSA_LAST_N=4
ROPE_DIMS=16
LN_SCALE=1
VE_ENABLED=1
VE_DIM=128
VE_LAYERS=9,10
MUON_WD=0.04
ADAM_WD=0.04
MATRIX_LR=0.025
SCALAR_LR=0.025
TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3500
ITERATIONS=9000
MAX_WALLCLOCK_SECONDS=600
EVAL_STRIDE=64
LATE_QAT_THRESHOLD=0.15
SWA_ENABLED=1
SWA_EVERY=50
python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

### Key metrics

| Metric | Value |
|---|---:|
| Steps at wallclock stop | `7142` |
| Stop-time val_bpb | `1.1394` |
| Post-EMA val_bpb | `1.1383` |
| Int6 roundtrip val_bpb | `1.14681998` |
| **Int6 sliding s64 val_bpb** | **`1.12311898`** |
| Int6+zstd artifact bytes | `15,610,171` |
| Code bytes | `73,105` |
| **Total submission bytes** | **`15,683,276`** |
| Peak memory allocated | `21,630 MiB` |

### Notes

- This folder records one validated seed (`1337`). Multi-seed reproducibility for this exact environment is not yet recorded here.
- The run used the same `train_gpt.py` source as the local control path at [train_gpt.py](/Users/abhi/projects/parametergolf/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py).
- Trust-head preservation work remains separate. This result is from the control path, not the trust-head variant.

### Included files

- `train_gpt.py` - exact training script used for the validated run
- `train.log` - synced training log from the validated run
- `train_seed1337.log` - same log under seed-specific naming
- `submission.json` - metadata for this validated candidate
