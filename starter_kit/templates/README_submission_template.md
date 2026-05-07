# {{RUN_NAME}}

- Date: {{DATE}}
- Track: {{TRACK}}
- Author: {{AUTHOR_NAME}} ({{GITHUB_ID}})
- Reported val_bpb: {{VAL_BPB}}

## Summary

Short summary of the idea and why it may help.

## What Changed

- List architecture changes.
- List optimization and schedule changes.
- List quantization or eval changes.

## Repro Command

```bash
RUN_ID={{RUN_NAME}} \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Results

- val_bpb:
- val_loss:
- compressed_bytes:
- wallclock_seconds:

## Notes

Any caveats, negative findings, or follow-up experiments.
