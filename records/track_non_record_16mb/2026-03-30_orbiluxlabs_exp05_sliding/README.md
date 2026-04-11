# Non-record submission: exp05_sliding

This is a non-record 16MB submission for the OpenAI Parameter Golf challenge.

## Summary
Main idea:
- sliding-window evaluation (EVAL_STRIDE=512)
- 1024 vocab tokenizer
- int8 + zlib compression

## Final result
- val_bpb: 1.34398504
- stopping step: 1765
- training time: 900 seconds
- compressed size: 14727167 bytes

## Setup
- model params: 17059912
- seq_len: 1024
- batch_tokens: 524288
- grad_accum_steps: 8

## Logs
See exp05_sliding.txt
