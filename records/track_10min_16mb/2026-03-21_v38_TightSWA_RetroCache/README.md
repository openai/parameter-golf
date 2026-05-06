# Frontier Rebase + RetroCache

This record folder rebases onto the March 21, 2026 `#374` frontier stack and adds **RetroCache**, an eval-only retrieval layer over already-scored validation tokens.

This folder is an implementation package, not a claimed scored submission yet. The imported `#374` training recipe is intact; the new work is isolated to the evaluation path plus metadata/tests.

## Base Stack
- 11 transformer layers, 512 dim, 8 heads / 4 KV heads
- Tight SWA on the last ~600 training steps
- Shared value embedding on layers 9-10
- Partial RoPE (16/64 dims)
- LN scale, XSA on the last 4 layers
- SmearGate + BigramHash + orthogonal init
- Int6/int8 mixed quantization with zstd-22

## RetroCache
- `forward_eval_features(input_ids)` returns both logits and normalized `final_norm` features
- A pure-PyTorch GPU memory stores:
  - dense recent keys/targets for the last `CACHE_RECENT_TOKENS`
  - stride-subsampled older keys/targets up to `CACHE_MAX_TOKENS`
- Retrieval is eval-only and backward-looking:
  - top-k nearest neighbors over stored keys
  - token distribution from retrieved next-token ids
  - entropy-gated interpolation with the LM probabilities
- New tokens are appended to memory only after their chunk has been scored

## Public Flags
```bash
CACHE_ENABLED=1
CACHE_MAX_TOKENS=32768
CACHE_RECENT_TOKENS=4096
CACHE_OLD_STRIDE=4
CACHE_TOPK=32
CACHE_BETA=24
CACHE_LAMBDA_MAX=0.35
CACHE_WARMUP_TOKENS=2048
CACHE_KEY_SOURCE=final_norm
CACHE_RESET_ON_BOS=0
```

## Expected Workflow
1. Reproduce the imported `#374` baseline with `CACHE_ENABLED=0`.
2. Re-run the same model with `CACHE_ENABLED=1`.
3. Ablate `topk`, memory size, and BOS reset behavior if the first cached run is promising.

## Notes
- `CACHE_ENABLED=0` keeps the original sliding-window evaluator path.
- `CACHE_RESET_ON_BOS=0` is the default because the competition currently treats validation as a continuous scored stream.
- No new dependencies were added for retrieval; the cache uses only PyTorch tensors.
