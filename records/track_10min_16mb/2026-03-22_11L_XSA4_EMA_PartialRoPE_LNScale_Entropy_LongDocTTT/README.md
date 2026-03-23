# 11L XSA4 + EMA + Partial RoPE + Rank-8 TTT Hooks

## Summary

This submission uses an 11-layer PR315-derived stack with:

- XSA on the last 4 layers
- EMA (`0.997`)
- Partial RoPE (`16/64` head dims)
- layerwise LN scaling
- SmearGate + BigramHash embeddings
- mixed int6/int8 export with `zstd`
- an integrated long-document LoRA TTT path

The claimed score for this submission is the exact roundtrip metric from the completed seed-42 `8xH100` run:

- `final_quant_roundtrip_exact val_bpb: 1.15910316`
- `final_quant_roundtrip_exact val_loss: 1.95709714`
- total artifact size: `15,528,215` bytes

This is a non-SOTA leaderboard submission, not a new-record claim.

## What Is Being Claimed

The final run finished the `600s` training budget and emitted the exact roundtrip metric successfully. The Runpod container disappeared after that line, before later post-train eval variants could be recovered. Because of that, this submission only claims the exact roundtrip result that is present in the attached log.

I am **not** claiming a sliding-window exact score or a final adaptive-TTT exact score for this submission.

## Model And Training Setup

- 11 transformer layers
- model dim `512`
- `8` attention heads, `4` KV heads
- `2048` train/eval sequence length
- tied embeddings
- `26829913` parameters
- `786432` train tokens/step
- `5248` steps completed in the 10-minute wallclock budget

The run command was:

```bash
COMPARE_EXPORT_CODECS=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
EXPORT_CODEC=zstd \
MAX_WALLCLOCK_SECONDS=600 \
RUN_ID=combo_001_adaptive_ttt_on__ttt_rank8 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
TTT_BATCH_SIZE=16 \
TTT_CHUNK_SIZE=256 \
TTT_DOC_PERCENTILE=95 \
TTT_ENABLED=1 \
TTT_LORA_RANK=8 \
TTT_MAX_DOCS=4 \
TTT_MAX_EVAL_SECONDS=45 \
TTT_MIN_PRED_TOKENS=2000 \
TTT_TARGET_LAST_N=2 \
VOCAB_SIZE=1024 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Recovered Official Run

- hardware: `8xH100-80GB-SXM`
- seed: `42`
- train wallclock: `600.000s`
- train stop: `step 5248/9000`
- peak memory: `21846 MiB` allocated, `21896 MiB` reserved
- serialized model size: `15,427,293` bytes after `mixed+zstd`
- total submission size: `15,528,215` bytes
- exact roundtrip score: `1.15910316 bpb`

## Artifact Notes

The folder includes:

- `train_gpt.py`
- `custom_entropy_codec.py`
- `submission.json`
- `train_seed42.log`

`custom_entropy_codec.py` is included because `train_gpt.py` imports it directly and counts it toward code size.

## Important Implementation Note

The adaptive eval path keeps the variable-length short-document no-TTT scoring path eager instead of compiled. This avoids Torch Dynamo recompile-limit failures in the final evaluation path.
