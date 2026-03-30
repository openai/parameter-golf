# Next Session

## Phase

**Session 05c-plus: Training bundle implementation and 8xH100 run.**

GPTQ debugging is parked. The next session is a training-quality implementation session.

## Immediate next action

1. **Commit and push** `records/track_non_record_16mb/2026-03-30_training_bundle_plus/` — code is implemented and validated
2. Run on 8xH100 when a slot opens (launch command in README)
3. After training: evaluate with naive int6 export (built into the script)
4. GPTQ replay is a **separate Phase 2 step** requiring a merge (VE128 + LeakyReLU² into GPTQ script)

## What happened in Session 05b (PARKED)

Seven GPTQ ablations on the Session 03 checkpoint all failed:
- Ablation #6 (PR #1019 verbatim transplant) produced **byte-identical MSE** to the local code
- This proved the GPTQ code is correct — the failure is model-specific
- Ablation #7 (AR self-gen calibration) crashed with non-PD Hessian
- Root cause hypothesis: relu().square() creates sparse Hessians; leaky_relu(0.5).square() does not

## Key finding: SWA is dead code

Both PR #1019 and #634 collect SWA snapshots but only apply EMA at export.
SWA is NOT included in the 05c-plus bundle. Use EMA only.

## 05c-plus bundle details

| Change | Type | Risk |
|--------|------|------|
| XSA 4→11 | one constant | very low |
| VE128 layers 9-10 | new module | low |
| Warmdown 3500 | one constant | none |
| LeakyReLU(0.5)² | one line | low (quality-neutral in isolation) |

Base: Session 03 anchor (`records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py`)

## Decision gate after run

1. Evaluate the run with naive int6 export (built into script)
2. If val_bpb improves: port VE128 + LeakyReLU² into GPTQ script, then test GPTQ replay
3. If GPTQ is sane → continue from there
4. If GPTQ is still bad → park permanently, keep naive-int6 result

## Success criteria

- Sliding s64 val_bpb < 1.126 (anchor is 1.129)
- Pre-quant EMA val_bpb < 1.142 (anchor is 1.145)
- step_avg within +5ms of anchor (91.37 ms)
- Artifact <= 16,000,000 bytes

## Files to read first

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`
3. `docs/superpowers/plans/2026-03-30-session-05c-plus.md`
4. `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py` (if created)
5. PR #1019 reference: `pr-1019-gptq:records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`

## 8xH100 launch command

```bash
cd /netscratch/$USER/parameter-golf && git pull

srun -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 \
  --mem=200G --time=00:20:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    export PYTHONUNBUFFERED=1
    export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond,eth NCCL_P2P_LEVEL=NVL
    pip install --no-cache-dir sentencepiece zstandard 2>/dev/null
    python -u records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
  '
```
