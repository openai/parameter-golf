# Colab Experiment: LaCT Fast-Weight Record Search

This folder is a record-oriented variant of the April 9 SP8192 stack with a
LaCT-style fast-weight TTT path added after quantized export.

Baseline for comparison:

- latest valid record: `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`
- mean TTT BPB: `1.0810`
- mean sliding BPB: `1.0827`
- mean artifact bytes: `15,992,694`

The script logs this target on every run as:

```text
latest_valid_record baseline_date:2026-04-09 ttt_bpb:1.0810 sliding_bpb:1.0827 artifact_bytes_mean:15992694
```

## Paper Idea Used

The paper's useful novelty for Parameter Golf is not just "do TTT". It is a
separate fast-weight learner that is applied causally in large chunks, then
updated on already-scored text. This experiment maps that idea into Golf
constraints by keeping the serialized model small and creating the fast state
only during eval.

What changed in `train_gpt.py`:

- adds `quantized_lact_ttt` as a separate eval metric
- scores each chunk before updating on that chunk
- inserts a fast-weight adapter on the hidden state before the tied output head
- supports nonlinear SwiGLU fast weights with `LACT_FAST_WEIGHT=swiglu`
- supports linear fallback with `LACT_FAST_WEIGHT=linear`
- supports Muon-style fast-weight updates with `LACT_UPDATE=muon`
- can optionally combine the adapter with legacy score-first base-model TTT via
  `LACT_BASE_TTT=1`

This avoids rule-sensitive SLOT/ngram/ETLB tricks and does not consult
validation loss during training.

## Record Attempt

Use this command on an 8xH100 SXM box:

```bash
INSTALL_DEPS=1 NPROC_PER_NODE=8 RECORD_PROFILE=1 TRAIN_SHARDS=128 \
RUN_ID=lact_swiglu_muon_s128_base_ttt \
LACT_TTT_ENABLED=1 TTT_ENABLED=0 LACT_BASE_TTT=1 \
LACT_FAST_WEIGHT=swiglu LACT_STATE_DIM=128 LACT_UPDATE=muon \
LACT_EPOCHS=1 LACT_LR=0.02 LACT_SCALE=0.08 \
bash run.sh
```

The important number is `quantized_lact_ttt`. It should be compared against the
logged `latest_valid_record` target and against `quantized_sliding_window`.

## Smoke Test

For a smaller local or Colab smoke test:

```bash
INSTALL_DEPS=1 bash run.sh
```

Single-GPU defaults keep the nonlinear fast weights enabled, but reduce the
state and eval batch:

- `LACT_STATE_DIM=64`
- `LACT_BATCH_SEQS=4`
- `LACT_BASE_TTT=0`
- `ENABLE_COMPILE=0`

## Sweep Order

Run these in order on the 8xH100 machine:

1. Legacy comparison:

   ```bash
   NPROC_PER_NODE=8 RECORD_PROFILE=1 LACT_TTT_ENABLED=0 TTT_ENABLED=1 bash run.sh
   ```

2. Adapter-only LaCT:

   ```bash
   NPROC_PER_NODE=8 RECORD_PROFILE=1 LACT_BASE_TTT=0 LACT_STATE_DIM=128 bash run.sh
   ```

3. Hybrid record candidate:

   ```bash
   NPROC_PER_NODE=8 RECORD_PROFILE=1 LACT_BASE_TTT=1 LACT_STATE_DIM=128 bash run.sh
   ```

4. If memory and time allow, increase capacity:

   ```bash
   NPROC_PER_NODE=8 RECORD_PROFILE=1 LACT_BASE_TTT=1 LACT_STATE_DIM=192 LACT_BATCH_SEQS=8 bash run.sh
   ```

5. If runtime is too high, fall back while keeping the trigger:

   ```bash
   NPROC_PER_NODE=8 RECORD_PROFILE=1 LACT_FAST_WEIGHT=linear LACT_UPDATE=sgd LACT_STATE_DIM=64 bash run.sh
   ```

## Key Knobs

- `LACT_FAST_WEIGHT=swiglu|linear`
- `LACT_UPDATE=muon|sgd`
- `LACT_STATE_DIM=64|128|192|256`
- `LACT_BASE_TTT=0|1`
- `LACT_CHUNK_TOKENS=32768`
- `LACT_EPOCHS=1`
- `LACT_LR=0.02`
- `LACT_SCALE=0.08`
- `EXPORT_ALLOCATOR=entropy|legacy`

The default record profile uses `EXPORT_ALLOCATOR=entropy`, inherited from the
current best local Colab experiment. Use `EXPORT_ALLOCATOR=legacy` to isolate
the LaCT change from allocator changes.

## Expected Logs

The run should emit:

- `pre-quantization post-ema`
- `quantized`
- `quantized_sliding_window`
- `quantized_lact_ttt`
- `artifact bytes`
- `latest_valid_record ... ttt_bpb:1.0810 ...`

Logs are written to:

```bash
logs/${RUN_ID}.txt
```
