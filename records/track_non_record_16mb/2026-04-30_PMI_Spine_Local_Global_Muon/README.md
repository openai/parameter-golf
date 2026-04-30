# Non-Record Submission: PMI-Spine Local-Global Muon

This is a non-record 8xH100 10-minute submission package for a distinct local-global recurrent language model with a PMI-anchored spine path.

## Summary

- Track: `non-record-10min-16mb`
- Tokenizer/data: SP1024 SentencePiece, `fineweb10B_sp1024`
- Architecture: bifurcated shared-stem LM with full-resolution local windows, pooled global recurrent summaries, PMI/SPINE basis projection, pointer-copy logit merge, lag-interferometer residuals, and int4 QAT export
- Optimizer: explicit hidden-matrix Muon taxonomy, row-sharded local Muon, stabilized Gram-NS backend, fifth-order polar coefficients, same-shape batching
- Quantization: int4+zlib export with roundtrip evaluation

## Primary Result

From `train.log`:

| Metric | Value |
|---|---:|
| Final exact val_loss | `2.11775515` |
| Final exact val_bpb | `1.25425389` |
| Best pre-roundtrip checkpoint | `1.25422432` at step `4697` |
| Stop step | `4697 / 20000` |
| Train wallclock | `600.069s` |
| Step average | `127.76ms` |
| Tokens seen | `3,232,137,216` |
| Model params | `36,620,134` |
| Model int4+zlib bytes | `15,679,191` |
| Package source bytes | `216,223` |
| Total package bytes | `15,895,414` |
| Artifact under 16MB | `true` |

## Included Evidence

- `train.log`: primary 8xH100 run, size-compliant final roundtrip.
- `priors.py`: included prior builders for independent reproduction.

## Reproduction Command

This chassis loads `bigram_logprior.pt`, `bigram_lowrank.pt`, and `trigram_cp.pt` at startup. Build them once with the included `priors.py`, then launch training:

```bash
cd /workspace/parameter-golf
# Step 1: build priors (~5–10 min on CPU/1 GPU; idempotent — skip if .pt files already exist)
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
VOCAB_SIZE=1024 \
python records/track_non_record_16mb/2026-04-30_PMI_Spine_Local_Global_Muon/priors.py

# Step 2: train
RUN_ID=pmi_spine_local_global_8g \
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-04-30_PMI_Spine_Local_Global_Muon/train_gpt.py
```

## Files

- `train_gpt.py`: code snapshot used for packaging.
- `priors.py`: prior-builder helpers.
- `train.log`: raw primary run log.
- `submission.json`: structured metadata.
- `results.json`: parsed run metrics.
- `STATUS.md`: copied local status notes.
