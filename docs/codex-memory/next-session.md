# Next Session

## Phase

**Session 03 anchor is COMPLETE. Session 04 isolated deltas start now.**

## Immediate next action

**Session 04: Isolated Deltas** — FA3 integration, GPTQ-lite, LeakyReLU^2 as independent, measured changes on top of the Session 03 anchor.

## Prerequisites (all satisfied)

- Session 03 anchor verified: sliding s64 val_bpb `1.12904446`, 6564 steps, 91.37ms/step
- Throughput bottleneck identified: SDPA is the limiter, not model fidelity
- NGC container + fscratch path confirmed on Pegasus
- Launcher lesson locked: use `srun --ntasks=8 --gpus-per-task=1 --gpu-bind=none`, NOT torchrun
- int6+zstd roundtrip artifact: `15751324` bytes, headroom `248676` bytes

## Session 04 implementation order

### Delta 1: FA3 integration (throughput unlock)

1. Replace SDPA attention with `flash_attn_3_func` in the Session 03 script
2. Verify compile compatibility with `torch.compile(fullgraph=True)`
3. Measure step_avg improvement (target: significant reduction from `91.37 ms`)
4. Measure val_bpb to confirm no regression
5. This is the highest-leverage single change

### Delta 2: GPTQ-lite compression

1. Add GPTQ-lite quantization to the export path
2. Measure roundtrip val_bpb improvement vs int6+zstd baseline
3. Measure artifact size change
4. Keep as isolated delta: do not combine with other changes in the same run

### Delta 3: LeakyReLU^2 activation

1. Replace relu^2 MLP activation with LeakyReLU^2
2. Measure val_bpb impact
3. Keep as isolated delta

### Measurement discipline

- Each delta is a separate run with one change
- Compare against Session 03 anchor as the fixed reference
- Record: GPU, steps, step_avg, sliding s64 val_bpb, pre-quant EMA val_bpb, int6 roundtrip val_bpb, artifact size
- Only combine deltas after each is measured in isolation

## Target

Session 04 combined best val_bpb: improve on `1.12904446` sliding s64, primarily through more steps via FA3 throughput

## Launcher template for 8xH100 on Pegasus (NGC container)

```bash
salloc -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --time=02:00:00

srun --gpu-bind=none bash -c '
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_IB_DISABLE=1
cd /netscratch/ayach/parameter-golf
RUN_ID=<run_id> \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
AMP_DTYPE=auto \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
python3 -u train_gpt.py
' 2>&1 | tee /netscratch/ayach/<run_id>.log
```
