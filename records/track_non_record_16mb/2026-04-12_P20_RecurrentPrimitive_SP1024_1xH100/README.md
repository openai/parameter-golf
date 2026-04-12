# P20 Recurrent Primitive Hybrid: Non-Record Research Submission

**Track:** `track_non_record_16mb`
**Author:** Joseph Abraham ([@abbudjoe](https://github.com/abbudjoe))
**Hardware:** 1xH100 80GB
**Status:** Non-record, research contribution
**Best P20 artifact:** 1.357619 BPB, loss 2.292283, 14,440,584 bytes
**Source training run:** 10-minute SP1024 80-shard checkpoint, 988 steps, 600.491s

## Short Version

This submission documents a controlled attempt to bring a custom recurrent primitive, P20, into the Parameter Golf stack. P20 was ported as a single middle recurrent layer inside an otherwise transformer-derived 11L/512 SP1024 model, using schedule `AAAAAPAAAAA`.

The result is not a leaderboard record. The best P20 hybrid trails the pure-attention control under the same checkpoint requantization sweep. The useful finding is narrower: P20 can stay close under the 16MB cap when protected with all-large-int8 quantization, but naive attention replacement is not yet the right insertion contract.

## What Was Tested

The intended ablation was one variable at a time:

- Keep tokenizer, SP1024 data path, optimizer, training budget, evaluation path, and quantization machinery fixed.
- Replace one middle attention/MLP transformer block with a P20 recurrent primitive.
- Use the P20 Triton runtime and block-structured recurrent state path from the Fractal prototype.
- Compare the trained P20 checkpoint against a pure-attention control through the same quantization sweep.

This PR includes the P20 training script snapshot, the P20 runtime files, the checkpoint requantization helper, and run summaries/logs for both P20 and the pure-attention control.

## Results

| Experiment | Model | Quant/export | Pre BPB | Post BPB | Post loss | Bytes | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| Pure attention control | 11L/512 SP1024 | mixed int6 default | 1.343710 | 1.359737 | 2.295859 | 10,294,744 | Same quant sweep control |
| Pure attention control | 11L/512 SP1024 | all-large-int8 | 1.343710 | 1.344724 | 2.270510 | 14,966,424 | Best control export |
| P20 hybrid | 11L/512 SP1024, `AAAAAPAAAAA` | mixed int6 default | 1.356221 | 1.376010 | 2.323335 | 10,044,945 | Single middle P20 slot |
| P20 hybrid | 11L/512 SP1024, `AAAAAPAAAAA` | all-large-int8 | 1.356221 | 1.357619 | 2.292283 | 14,440,584 | Best P20 export |

The original 10-minute P20 source run recorded post-quant BPB 1.376889 at 9,747,772 bytes under its initial export path. The requantization sweep in this folder is the cleaner like-for-like comparison because both P20 and pure attention were re-exported through the same variants.

## Interpretation

The P20 hybrid did not beat the pure-attention control. At high precision under the 16MB cap, P20 is about 0.0129 BPB behind the corresponding pure-attention export. The replacement also saved bytes in some variants, but the saved bytes were not enough to compensate for lower pre-quant quality and slower recurrent compute.

The positive result is that the quantization failure mode is tractable. For P20, the post-minus-pre gap improved from about +0.0198 BPB under the default mixed export to +0.0014 BPB under all-large-int8 while remaining below 16MB. That suggests future P20 work should focus on better insertion contracts and context/state leverage, not on raw one-for-one attention replacement.

## Why This Is Useful

This is a negative result with a reusable baseline:

- P20 is better treated as a recurrent side-channel, looped adapter, or context-state module than as a direct attention replacement.
- P20 needs quantization protection for its recurrent/state matrices.
- Pure attention remains the stronger 10-minute small-context baseline on this surface.
- The next fair test is not "replace attention with P20"; it is "keep the transformer stack and add P20 where recurrent state can buy context, TTT, or memory efficiency."

## Included Files

- `train_gpt.py`: training/evaluation script snapshot used for P20 experiments.
- `p20_runtime.py`: P20 Triton/runtime implementation.
- `p20_ttt.py`: P20 TTT adapter support code.
- `supporting_files/requant_checkpoint_sweep.py`: helper used to re-export a trained checkpoint across quantization variants.
- `logs/source_summary_seed42.json`: source 10-minute P20 checkpoint summary.
- `logs/quant_sweep_seed42.json`: P20 checkpoint quantization sweep summary.
- `logs/quant_sweep_seed42.log`: P20 quantization sweep console log.
- `logs/baseline_quant_control_seed42.json`: pure-attention quantization control summary.
- `logs/baseline_quant_control_seed42.log`: pure-attention quantization control console log.

## Reproduction Notes

The headline P20 training run used the following intended contract:

```bash
MODEL_FAMILY=p20_hybrid \
P20_LAYER_SCHEDULE=AAAAAPAAAAA \
P20_RUNTIME_BACKEND=triton \
P20_STATE_BLOCKS=auto \
P20_TTT_MODE=off \
SPM_VOCAB_SIZE=1024 \
TRAIN_TIME_LIMIT_SECONDS=600 \
python train_gpt.py
```

The best reported P20 export came from reusing that trained checkpoint and running the included quantization sweep, with `all_large_int8` selected as the best under-16MB export.

## Non-Record Disclaimer

This is intentionally submitted as a non-record PR. It is a controlled research note showing that a P20 recurrent primitive can approach, but not beat, the pure-attention control under this particular 1xH100 10-minute contract. The contribution is the ablation, the code snapshot, and the evidence that future recurrent work should move toward side-channel/context-state insertion rather than direct block replacement.
