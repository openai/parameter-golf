# Non-Record Package: Future-State Encoder Planner

This package stages a 1xH100 smoke/evidence run for a recurrent GQA transformer with an explicit future-state encoder target.

## Summary

- Track status: `non-record-local-evidence`
- Tokenizer/data: SP1024 SentencePiece, `fineweb10B_sp1024`
- Architecture: recurrent GQA schedule with prefix-hybrid macro routing, explicit future-state encoder target, latent planner conditioning, future-embed MTP head, meta-preconditioner, bigram-prior signals, smear gate, and int4 export
- Optimizer: banked parallel Muon backbone with `turbo4_aol` geometry
- Quantization: int4+zlib export with final roundtrip evaluation

## Primary Result

From `logs/future_state_encoder_planner_1g.txt`:

| Metric | Value |
|---|---:|
| Final exact val_loss | `2.36019942` |
| Final exact val_bpb | `1.39784304` |
| Best logged pre-roundtrip val_bpb | `1.3978` at step `539` |
| Stop step | `539 / 20000` |
| Train wallclock | `600.970s` |
| Step average | `1114.97ms` |
| Tokens seen | `370,900,992` |
| Model params | `29,739,499` |
| Model int4+zlib bytes | `14,802,210` |
| Package source bytes | `201,482` |
| Total package bytes | `15,003,692` |
| Artifact under 16MB | `true` |

## Included Evidence

- `logs/future_state_encoder_planner_1g.txt`: 1xH100 600s run, size-compliant final int4+zlib roundtrip.
- `priors.py`: included prior builders for independent reproduction.

## Reproduction Command

This chassis loads `bigram_logprior.pt`, `bigram_lowrank.pt`, and `trigram_cp.pt` at startup. Build them once with the included `priors.py`, then launch training:

```bash
cd /workspace/parameter-golf
# Step 1: build priors (~5–10 min on CPU/1 GPU; idempotent — skip if .pt files already exist)
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
VOCAB_SIZE=1024 \
python non-record-local/2026-04-30_future_state_encoder_planner/priors.py

# Step 2: train
RUN_ID=future_state_encoder_planner_1g \
torchrun --standalone --nproc_per_node=1 \
non-record-local/2026-04-30_future_state_encoder_planner/train_gpt.py
```

## Files

- `train_gpt.py`: code snapshot used for packaging.
- `priors.py`: prior-builder helpers.
- `logs/`: raw run log.
- `submission.json`: structured metadata.
- `results.json`: parsed run metrics.
