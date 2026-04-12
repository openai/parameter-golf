# Fractal Recurrent Primitive Hybrid: Non-Record Research Submission

**Track:** `track_non_record_16mb`
**Author:** Joseph Abraham ([@abbudjoe](https://github.com/abbudjoe))
**Hardware:** 1xH100 80GB
**Status:** Non-record, research contribution
**Best recurrent-primitive artifact:** 1.357619 BPB, loss 2.292283, 14,440,584 bytes
**Source training run:** 10-minute SP1024 80-shard checkpoint, 988 steps, 600.491s

## Short Version

This submission documents a controlled attempt to bring a custom Fractal recurrent primitive into the Parameter Golf stack. The primitive was ported as a single middle recurrent layer inside an otherwise transformer-derived 11L/512 SP1024 model, using schedule `AAAAAPAAAAA`.

The result is not a leaderboard record. The best recurrent-primitive hybrid trails the pure-attention control under the same checkpoint requantization sweep. The useful finding is narrower: the recurrent primitive can stay close under the 16MB cap when protected with all-large-int8 quantization, but naive attention replacement is not yet the right insertion contract.

## Attribution and Leaderboard Provenance

This experiment was guided by the public leaderboard meta, especially the current #1 record. The recurrent primitive is the new variable here; the surrounding stack borrows several proven ingredients from prior public work.

| Used ingredient | Credit | How it appears here |
|---|---|---|
| Mixed int6/int8 quantization pressure and protected higher-precision export variants | PR #1394 @clarkkev | The source runs use mixed int6 clipsearch + zstd, and the best 10-minute export is an all-large-int8/zstd protection sweep. |
| Learnable per-head QK gain machinery | #1 record stack | The transformer attention path includes learnable per-head query scaling. |
| EMA 0.9965 and warmdown-style schedules | PR #1445 @X-Abhishek-X | Both the 10-minute and 60-minute recurrent runs use EMA decay 0.9965; the 60-minute probe also uses a longer warmdown schedule. |

## What Was Tested

The intended ablation was one variable at a time:

- Keep tokenizer, SP1024 data path, optimizer, training budget, evaluation path, and quantization machinery fixed.
- Replace one middle attention/MLP transformer block with the Fractal recurrent primitive.
- Use the Triton runtime and block-structured recurrent state path from the Fractal prototype.
- Compare the trained recurrent-primitive checkpoint against a pure-attention control through the same quantization sweep.

This PR includes the training script snapshot, recurrent runtime files, checkpoint requantization helper, and run summaries/logs for both the recurrent-primitive hybrid and the pure-attention control.

## Results

| Experiment | Model | Quant/export | Pre BPB | Post BPB | Post loss | Bytes | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| Pure attention control | 11L/512 SP1024 | mixed int6 default | 1.343710 | 1.359737 | 2.295859 | 10,294,744 | Same quant sweep control |
| Pure attention control | 11L/512 SP1024 | all-large-int8 | 1.343710 | 1.344724 | 2.270510 | 14,966,424 | Best control export |
| Fractal recurrent hybrid | 11L/512 SP1024, `AAAAAPAAAAA` | mixed int6 default | 1.356221 | 1.376010 | 2.323335 | 10,044,945 | Single middle recurrent slot |
| Fractal recurrent hybrid | 11L/512 SP1024, `AAAAAPAAAAA` | all-large-int8 | 1.356221 | 1.357619 | 2.292283 | 14,440,584 | Best recurrent export |

The original 10-minute recurrent-primitive source run recorded post-quant BPB 1.376889 at 9,747,772 bytes under its initial export path. The requantization sweep in this folder is the cleaner like-for-like comparison because both the recurrent hybrid and pure attention were re-exported through the same variants.

## Extended 60-Minute Probe

I also ran a longer, non-competition-timed probe to see whether the recurrent primitive was still improving with more training time. This used schedule `AAAPAAAAPAA`, 60 minutes of training time after compile/prewarm, warmdown 4000, EMA 0.9965, mixed int6 clipsearch + zstd, and 1xH100.

| Run | Steps | Train time | Pre BPB | Post BPB | Post loss | Bytes | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Fractal recurrent hybrid 60-minute probe | 5,342 | 3,600.034s | 1.230186 | 1.241819 | 2.701189 | 16,179,345 | Out of the 10-minute rule and slightly over 16MB |

This result is useful for research direction, not leaderboard scoring. It shows the recurrent primitive continued improving substantially with enough wall-clock time, but the current implementation is too slow and too large to translate that longer-run quality into the official 10-minute/16MB track without more systems work or multi-GPU scaling.

## Interpretation

The recurrent-primitive hybrid did not beat the pure-attention control. At high precision under the 16MB cap, it is about 0.0129 BPB behind the corresponding pure-attention export. The replacement also saved bytes in some variants, but the saved bytes were not enough to compensate for lower pre-quant quality and slower recurrent compute.

The positive result is that the quantization failure mode is tractable. For the recurrent primitive, the post-minus-pre gap improved from about +0.0198 BPB under the default mixed export to +0.0014 BPB under all-large-int8 while remaining below 16MB. That suggests future work should focus on better insertion contracts and context/state leverage, not on raw one-for-one attention replacement.

## Why This Is Useful

This is a negative result with a reusable baseline:

- The Fractal recurrent primitive is better treated as a side-channel, looped adapter, or context-state module than as a direct attention replacement.
- The recurrent/state matrices need quantization protection.
- Pure attention remains the stronger 10-minute small-context baseline on this surface.
- The next fair test is not "replace attention"; it is "keep the transformer stack and add recurrent state where it can buy context, TTT, or memory efficiency."

## Included Files

- `train_gpt.py`: training/evaluation script snapshot used for the recurrent experiments.
- Recurrent primitive runtime snapshot.
- Recurrent-primitive TTT adapter support snapshot.
- `supporting_files/requant_checkpoint_sweep.py`: helper used to re-export a trained checkpoint across quantization variants.
- `logs/source_summary_seed42.json`: source 10-minute recurrent checkpoint summary.
- `logs/quant_sweep_seed42.json`: recurrent checkpoint quantization sweep summary.
- `logs/quant_sweep_seed42.log`: recurrent checkpoint quantization sweep console log.
- `logs/baseline_quant_control_seed42.json`: pure-attention quantization control summary.
- `logs/baseline_quant_control_seed42.log`: pure-attention quantization control console log.
- `logs/extended_60min_summary_seed42.json`: longer recurrent-primitive trajectory probe, outside official time/size constraints.

## Reproduction Notes

The headline training run used the following intended contract:

```bash
MODEL_FAMILY=<recurrent_hybrid_internal_label> \
LAYER_SCHEDULE=AAAAAPAAAAA \
RECURRENT_RUNTIME_BACKEND=triton \
RECURRENT_STATE_BLOCKS=auto \
RECURRENT_TTT_MODE=off \
SPM_VOCAB_SIZE=1024 \
TRAIN_TIME_LIMIT_SECONDS=600 \
python train_gpt.py
```

The best reported recurrent-primitive export came from reusing that trained checkpoint and running the included quantization sweep, with `all_large_int8` selected as the best under-16MB export.

## Non-Record Disclaimer

This is intentionally submitted as a non-record PR. It is a controlled research note showing that a Fractal recurrent primitive can approach, but not beat, the pure-attention control under this particular 1xH100 10-minute contract. The contribution is the ablation, the code snapshot, and the evidence that future recurrent work should move toward side-channel/context-state insertion rather than direct block replacement.
