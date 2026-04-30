# Non-Record Submission: GQA Macro Meta-Preconditioned

This is a non-record 8xH100 10-minute submission package for an adaptive GQA transformer with macro routing and meta-preconditioned local transforms.

## Summary

- Track: `non-record-10min-16mb`
- Tokenizer/data: SP1024 SentencePiece, `fineweb10B_sp1024`
- Architecture: GQA transformer with macro side-channel cross-attention, adaptive depth/router controls, fp32 logit head, detached macro distillation, meta-preconditioned local transforms, smear gate, and int4 export
- Optimizer: banked parallel Muon backbone with manually all-reduced non-bank parameters
- Quantization: int4+zlib export with final roundtrip evaluation

## Primary Result

From `train.log`:

| Metric | Value |
|---|---:|
| Final exact val_loss | `2.02607191` |
| Final exact val_bpb | `1.19995391` |
| Best logged pre-roundtrip val_bpb | `1.1993` at step `4837` |
| Stop step | `4837 / 20000` |
| Train wallclock | `600.076s` |
| Step average | `124.06ms` |
| Tokens seen | `3,328,475,136` |
| Model params | `33,691,738` |
| Model int4+zlib bytes | `15,456,516` |
| Run-reported code bytes | `102,713` |
| Total artifact bytes | `15,559,229` |
| Artifact under 16MB | `true` |

## Included Evidence

- `train.log`: primary 8xH100 run, size-compliant final int4+zlib roundtrip.

## Reproduction Command

This chassis uses no precomputed priors — only the SentencePiece tokenizer + tokenized shards.

```bash
cd /workspace/parameter-golf/records/track_non_record_16mb/2026-04-30_GQA_Macro_Meta_Preconditioned
RUN_ID=gqa_macro_meta_preconditioned_8g \
torchrun --standalone --nproc_per_node=8 \
train_gpt.py
```

## Files

- `train_gpt.py`: code snapshot used for packaging.
- `train.log`: raw run log.
- `submission.json`: structured metadata.
- `results.json`: parsed run metrics.
