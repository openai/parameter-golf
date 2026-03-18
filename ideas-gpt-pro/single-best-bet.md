# Single Best Bet

The most likely winning branch from `gpt-pro.md` is:

1. untie the output head
2. cut to `1` KV head
3. add EMA
4. train at `512`
5. evaluate with streamed `4k-8k` context plus RoPE scaling
6. layer on a strictly causal cache mixture

## Why This Branch

This combines the ideas the source document treats as both high-upside and structurally clean in the existing codebase:

- evaluation freedom is currently underused
- the quantized artifact matters more than the bf16 checkpoint
- the current parameter allocation is probably not where the bytes are buying the most quality

## What Comes After

If there is still wallclock and artifact slack after that branch, the next move is to spend it on a slightly larger model rather than on exotic optimizer tuning.
