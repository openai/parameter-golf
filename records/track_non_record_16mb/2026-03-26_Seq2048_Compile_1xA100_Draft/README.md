# Non-Record Submission Draft: Seq2048 + `torch.compile` (1xA100)

This is a draft non-record submission built from iterative single-card A100 experiments.  
It is **not** yet a formal `8xH100 / 10 minute` leaderboard submission, but it already fits under the `16,000,000` byte artifact cap and documents a clear improvement path over the local baseline.

## Summary

Current best run:

- Experiment: `seq2048_compile_midlr_v5`
- GPU: `1xA100`
- Post-quant `val_loss`: `2.29268432`
- Post-quant `val_bpb`: `1.35557361`
- Total submission bytes: `14,840,173`
- Sample step average: `239.48 ms`

Relative to the earlier single-A100 baseline:

- `baseline_v1`: `1.43862940`
- current best: `1.35557361`
- improvement: `-0.08306` bpb

## Main Idea

The current best direction is:

1. Increase training context to `TRAIN_SEQ_LEN=2048`
2. Keep the base layout at `9L / 512d / 8 heads / 4 KV heads`
3. Use moderately reduced learning rates
4. Enable `torch.compile`

The largest gain so far came from `torch.compile`. On single A100, this appears to improve throughput enough that the model gets meaningfully more effective optimization within the fixed wallclock budget.

## Experiment History

| Experiment | `val_bpb` | Bytes | Notes |
|------------|----------:|------:|-------|
| `baseline_v1` | `1.43862940` | `14,160,935` | single-A100 baseline reference |
| `ema_v1` | `1.45888600` | `13,692,543` | worse than baseline |
| `lowerlr_v1` | `1.42718185` | `13,427,928` | lower LR helps |
| `seq2048_v1` | `1.43235649` | `13,886,898` | longer context helps slightly |
| `bigbatch_accum_v1` | `1.46243246` | `11,678,074` | clearly worse |
| `seq2048_tunedlr_v2` | `1.41683723` | `13,337,910` | best pre-compile direction |
| `depth10_seq2048_v3` | `1.42052227` | `14,257,245` | deeper model undertrained |
| `depth11_seq2048_v3` | `1.42952177` | `15,214,704` | deeper model worse |
| `seq2048_midlr_v4` | `1.41471497` | `12,981,667` | small gain vs v2 |
| `seq2048_warmdown3500_v4` | `1.42752652` | `11,147,255` | warmdown extension hurts |
| `seq2048_compile_v4` | `1.35715710` | `15,171,942` | first compile breakthrough |
| `seq2048_compile_midlr_v5` | `1.35557361` | `14,840,173` | current best valid run |
| `depth10_compile_midlr_v5` | `1.35656278` | `16,135,555` | close, but over the 16MB cap |

## Configuration of Current Best Run

Track-relevant configuration:

```bash
VOCAB_SIZE=1024
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=2
TRAIN_BATCH_TOKENS=131072
TRAIN_SEQ_LEN=2048
GRAD_ACCUM_STEPS=1
WARMUP_STEPS=20
WARMDOWN_ITERS=1200
TIED_EMBED_LR=0.035
MATRIX_LR=0.028
SCALAR_LR=0.028
TORCH_COMPILE=1
SDP_BACKEND=auto
MAX_WALLCLOCK_SECONDS=600
```

The run was executed through the Colab single-A100 notebook flow and is currently being used to rank candidate ideas before spending `8xH100` time on formal reproduction.

## Why This Is Non-Record

This submission is currently placed in `track_non_record_16mb` because:

- the best reported result so far is from `1xA100`, not `8xH100`
- it has not yet been demonstrated as a formal `10 minute / 8xH100` reproduction
- the purpose of this record is to document a promising transfer candidate for later full-scale validation

## Next Step

The next experiments are focused on the `compile` line:

- `seq2048_compile_lowmidlr_v6`
- `depth10_dim504_compile_midlr_v6`
- `depth10_dim496_compile_midlr_v6`

The goal is to determine whether `compile` plus slightly different LR or slightly deeper models can improve further before moving to `8xH100`.

## Included Files

- `train_gpt.py`: current code snapshot for the draft submission
- `submission.json`: draft metadata for the best known single-A100 run
- `train_v0.txt`: placeholder log file to be replaced with the exact Colab run log before opening a real PR
- `requirements.txt`: dependency reference
