# Next Session

## Phase

**Session 04 in progress. Delta 1 (GPTQ-lite) FAILED. Delta 2 (LeakyReLU^2) NEUTRAL. Delta 3 next.**

## Immediate next action

**Session 04 Delta 3** — next isolated delta on top of the Session 03 anchor. Candidate ranking:
1. EMA freeze during late warmdown (cheapest)
2. ASQU activation (higher upside)
3. MTP auxiliary loss (save for later)

## Prerequisites (all satisfied)

- Session 03 anchor verified: sliding s64 val_bpb `1.12904446`, 6564 steps, 91.37ms/step
- Remaining donor gap is small (`+0.00419944` on final sliding), so broad redesign is unnecessary
- NGC container + fscratch path confirmed on Pegasus
- Launcher lesson locked: use `srun --ntasks=8 --gpus-per-task=1 --gpu-bind=none`, NOT torchrun
- int6+zstd roundtrip artifact: `15751324` bytes, headroom `248676` bytes

## Session 04 implementation order

1. ~~Delta 1: GPTQ-lite percentile clip search~~ — **COMPLETE (FAILED)**
   - Sliding s64 val_bpb: `1.12941356` (worse than anchor by `+0.00036910`)
   - Artifact: `16219752` bytes — OVER the `16000000` byte cap
   - Conclusion: hurts zstd compressibility more than it helps quantization quality

2. ~~Delta 2: LeakyReLU^2~~ — **COMPLETE (NEUTRAL)**
   - Sliding s64 val_bpb: `1.12904123` (effectively identical, `-0.00000323`)
   - Pre-quant EMA val_bpb: `1.14438546` (slightly better, `-0.00033857`)
   - Roundtrip val_bpb: `1.15222198` (slightly better, `-0.00025075`)
   - Artifact: `15582968` bytes (168KB smaller)
   - Step_avg: `92.09 ms` (+0.72 ms slower, -53 steps)
   - Conclusion: not a standalone graduating delta. Keep as possible stack component.

3. **Delta 3** — NEXT IMMEDIATE ACTION
   - Top candidate: EMA freeze during late warmdown
   - Alternative: ASQU activation

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
