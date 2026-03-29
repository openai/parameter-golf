# Next Session

## Phase

**Session 04 in progress. Delta 1 (GPTQ-lite) is COMPLETE (FAILED). Delta 2 (LeakyReLU^2) is next.**

## Immediate next action

**Session 04 Delta 2: LeakyReLU^2** — replace relu^2 with LeakyReLU^2 on top of the Session 03 anchor. The H100 node is allocated for ~22 more hours.

## Prerequisites (all satisfied)

- Session 03 anchor verified: sliding s64 val_bpb `1.12904446`, 6564 steps, 91.37ms/step
- Remaining donor gap is small (`+0.00419944` on final sliding), so broad redesign is unnecessary
- NGC container + fscratch path confirmed on Pegasus
- Launcher lesson locked: use `srun --ntasks=8 --gpus-per-task=1 --gpu-bind=none`, NOT torchrun
- int6+zstd roundtrip artifact: `15751324` bytes, headroom `248676` bytes

## Session 04 implementation order

1. ~~Delta 1: GPTQ-lite percentile clip search~~ — **COMPLETE (FAILED)**
   - Sliding s64 val_bpb: `1.12941356` (worse than anchor `1.12904446` by `+0.00036910`)
   - Roundtrip val_bpb: `1.15277272` (worse than anchor `1.15247273` by `+0.00029999`)
   - Artifact: `16219752` bytes — OVER the `16000000` byte cap
   - Conclusion: hurts zstd compressibility more than it helps quantization quality

2. **Delta 2: LeakyReLU^2** — NEXT IMMEDIATE ACTION
   - Replace relu^2 with LeakyReLU^2 on the Session 03 anchor
   - Measure sliding s64, roundtrip, pre-quant EMA val_bpb and artifact size
   - H100 node is allocated for ~22 more hours

3. Delta 3: one small schedule or token-path tweak — pending Delta 2 result

4. Keep backend/perf parity as a separate control if throughput becomes the dominant bottleneck.
   - Do not bundle backend work with export or model deltas in the same run.

## Measurement discipline

- Each delta is a separate run with one change
- Compare against Session 03 anchor as the fixed reference
- Record: GPU, steps, step_avg, sliding s64 val_bpb, pre-quant EMA val_bpb, int6 roundtrip val_bpb, artifact size
- Only combine deltas after each is measured in isolation

## Target

Session 04 goal: beat `1.12904446` on final sliding s64 with an attributable single-delta improvement.

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
python3 -u records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py
' 2>&1 | tee /netscratch/ayach/<run_id>.log
```
