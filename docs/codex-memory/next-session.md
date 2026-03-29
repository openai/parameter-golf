# Next Session

## Phase

**Session 04 closed. Session 05 next: throughput audit + pre-TTT base enhancement audit + TTT correctness audit.**

## Immediate next action

**Session 05 planning and audit** on top of the Session 03 anchor. Work in this order:
1. Throughput audit: explain `91.37 ms` anchor vs `83.4 ms` local public record and determine whether FA3 is portable on Pegasus / NGC.
2. Pre-TTT stack-gap audit: rank the easy portable pieces from the local `1.1194` stack (`VE128`, `warmdown3500`, `Bigram 1536`, `tight SWA`, etc.).
3. TTT audit: trace the score-first protocol, legality, eval budget, and portability to the anchor stack.

## Prerequisites (all satisfied)

- Session 03 anchor verified: sliding s64 val_bpb `1.12904446`, 6564 steps, 91.37ms/step
- Remaining donor gap is small (`+0.00419944` on final sliding), so broad redesign is unnecessary
- NGC container + fscratch path confirmed on Pegasus
- Launcher lesson locked: use `srun --ntasks=8 --gpus-per-task=1 --gpu-bind=none`, NOT torchrun
- int6+zstd roundtrip artifact: `15751324` bytes, headroom `248676` bytes

## Session 04 closeout

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

3. Session 04 decision
   - Close the micro-delta sweep at `1 failed + 1 neutral`
   - Do not force a Delta 3 by default
   - Open Session 05 instead

## Measurement discipline

- Each delta is a separate run with one change
- Compare against Session 03 anchor as the fixed reference
- Record: GPU, steps, step_avg, sliding s64 val_bpb, pre-quant EMA val_bpb, int6 roundtrip val_bpb, artifact size
- Only combine deltas after each is measured in isolation

## Session 05 target

- strengthen the pre-TTT base relative to `1.12904446`
- understand and, if justified, integrate TTT on top of a stronger base
- identify the highest-value portable pieces of the local `1.1194` public stack

## Read order for the next fresh session

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`
3. `docs/campaign/artifacts/04_targeted_delta_sweep.md`
4. `docs/campaign/sessions/05_ttt_correctness_audit.md`
5. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

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
