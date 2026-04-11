# Record: 8L GQA + Partial RoPE + XSA + Int8 QAT

**single-H100** run | **final sliding-window val_bpb: 1.2498** | **roundtrip val_bpb: 1.2731** | **gain: -0.0233**

This is a non-record **single-H100** experiment derived from the repo-root [`train_gpt.py`](/home/arctic/PycharmProjects/parameter-golf/train_gpt.py). For comparison, the naive baseline below is the repository’s **8xH100** reference run, so the hardware budget is not the same.

## Headline Comparison

| Metric | Naive Baseline (8xH100) | This Run Roundtrip (1xH100) | This Run Sliding (1xH100) |
|---|---:|---:|---:|
| `val_loss` | **2.07269931** | 2.14953084 | 2.11022315 |
| `val_bpb` | **1.22436570** | 1.27307324 | 1.24979460 |
| `delta val_loss vs baseline` | **0.00000000** | +0.07683153 | +0.03752384 |
| `delta val_bpb vs baseline` | **0.00000000** | +0.04870754 | +0.02542890 |

## Loss Graph

Lower is better.

```text
val_loss
Naive baseline      2.0727 | ####################
This run roundtrip   2.1495 | ######################
This run sliding     2.1102 | #####################

val_bpb
Naive baseline      1.2244 | ####################
This run roundtrip   1.2731 | #######################
This run sliding     1.2498 | ######################
```

The naive baseline remains better on both metrics, but the sliding-window pass closes part of the gap relative to the roundtrip score.

## What Changed vs Base `train_gpt.py`

### Model and training setup

- shrinks depth from `9` layers to `8`
- increases sequence length from `1024` to `2048`
- increases MLP expansion from `2x` to `3x`
- keeps GQA with `8` query heads and `4` KV heads
- adds partial RoPE via `rope_dims=32`
- raises `qk_gain_init` from the base script's `1.5`
- increases planned iterations from `20000` to `35000`
- adds `xsa_last_n`, with XSA enabled on the final `2` layers in this run
- switches attention layout to `[batch, seq, heads, dim]` internally and uses the `flash_attn` interface when available
- applies RoPE only to the first `rope_dims` channels and leaves the rest of each head unrotated
- changes block init by orthogonally initializing large linear layers and scaling projection weights by depth
- tightens the attention RMSNorm to `eps=1e-6`

### Optimization and schedule

- lowers tied-embedding LR from `0.05` to `0.04`
- lowers matrix/scalar LR from `0.04` to `0.032`
- changes token/scalar/head optimizers from `Adam` to `AdamW`
- adds decoupled weight decay: `adam_wd=0.04`, `muon_wd=0.02`
- extends `Muon` itself to apply decoupled weight decay
- adds `warmdown_last_frac` so warmdown can be driven by wallclock fraction instead of only `warmdown_iters`
- adds `qat_last_frac` so fake quantization is turned on only near the end of training

### Evaluation

- adds `eval_stride` and `eval_batch_seqs`
- adds `eval_val_sliding(...)` for sliding-window validation
- refactors the model to expose `forward_logits(...)` so sliding evaluation can reuse logits directly

### Quantization and export

- uses a mixed-precision export format: `mixed_int6_int8_per_row_v1`
- adds packed int6 storage for selected tensors using `pack_lowbit_tensor(...)` / `unpack_lowbit_tensor(...)`
- adds `INT6_NAME_PATTERNS` and `INT8_QAT_NAME_PATTERNS` to control export and QAT targeting by parameter name
- adds STE fake quantization during training for selected `CastedLinear` weights
- removes the baseline small-tensor fp16 passthrough heuristic and instead quantizes float tensors into int6 or int8 unless they are non-float passthrough tensors
- adds `compress_quant_payload(...)` / `decompress_quant_payload(...)`, currently using `zlib`
- renames final log lines from `final_int8_zlib_roundtrip...` to `final_mixed_quant_zlib_roundtrip...`

## Logged Run

The saved `train.log` contains the run footer, including the standard roundtrip result and the later sliding-window result.

### Logged config

From the appended log section:

- `model_params:19417152`
- `world_size:1 grad_accum_steps:8`
- `attention_kernel:flash_attn_interface`
- `attention_mode:gqa num_heads:8 num_kv_heads:4`
- `xsa_last_n:2`
- `tie_embeddings:True embed_lr:0.04 head_lr:0.0 matrix_lr:0.032 scalar_lr:0.032 adam_wd:0.04 muon_wd:0.02`
- `train_batch_tokens:524288`
- `train_seq_len:2048`
- `iterations:35000`
- `warmup_steps:20`
- `eval_stride:128 eval_batch_seqs:32`
- `mlp_mult:3 rope_dims:32`
- `warmdown_iters:1200 warmdown_last_frac:0.200`
- `late_qat_last_frac:0.150`
- `seed:1337`

### Logged training progression

- initial validation at `step:0`: `val_loss=6.9393`, `val_bpb=4.1098`
- late QAT enabled at `step:1378/35000`, `train_time:510217ms`, `frac:0.150`
- final pre-quant validation at `step:1621/35000`: `val_loss=2.1482`, `val_bpb=1.2723`
- stop reason: `stopping_early: wallclock_cap train_time:600254ms step:1621/35000`
- peak memory: `10126 MiB` allocated, `10216 MiB` reserved

Selected train-loss checkpoints from the log:

- `step:200` -> `train_loss:2.7105`
- `step:400` -> `train_loss:2.3429`
- `step:600` -> `train_loss:2.4496`
- `step:800` -> `train_loss:2.2998`
- `step:1200` -> `train_loss:2.2549`
- `step:1600` -> `train_loss:2.1083`

### Logged artifact sizes and roundtrip metrics

- raw serialized model: `76650381` bytes
- code size: `59176` bytes
- raw total submission size: `76709557` bytes
- mixed-quant artifact: `15142148` bytes
- quantized payload bytes before torch serialization overhead: `19476680`
- quantized raw torch object bytes: `19529207`
- payload compression ratio vs raw tensor bytes: `3.93x`
- final submission size with mixed quant + zlib: `15201324` bytes
- roundtrip exact: `val_loss=2.14953084`, `val_bpb=1.27307324`
- roundtrip eval time: `12736ms`
- sliding-window exact: `val_loss=2.11022315`, `val_bpb=1.24979460`
- sliding-window eval time: `657418ms`

## Command

```bash
RUN_ID=remote_1gpu_run \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=2048 \
torchrun --standalone --nproc_per_node=1 \
  records/track_non_record_16mb/2026-04-11_8L_GQA_PartialRoPE_Int8_AttnMLP3_QAT015/train_gpt.py
```
