# Random Map Adapter Projections V1

This record-track submission keeps standard autoregressive training and `val_bpb` evaluation, but replaces two dense projection families with seeded frozen random maps plus tiny learned adapter cores:

- `blocks.*.attn.proj`
- `blocks.*.mlp.proj`

The frozen down/up maps are regenerated from integer seeds and stored as non-persistent buffers, so they do not consume serialized weight bytes. Only the learned diagonal scales and tiny low-rank bottleneck-space adapters are trained and exported.

## Why This Direction

The challenge request list explicitly asks for `learning adapters on random linear maps`. This implementation takes the middle road:

- not a cosmetic side adapter on top of a normal model
- not a full random-feature network
- a direct replacement of selected dense projections inside a strong AR baseline

That makes the idea structurally different while still preserving the current best legal training stack and evaluation path.

## Base Stack

This branch starts from the `2026-03-24_LeakyReLU2_VRL_LZMA` record line and keeps:

- 11 layers, 512 model dim, 8 heads / 4 KV heads
- LeakyReLU(0.5)^2 MLP activation
- BigramHash, Partial RoPE, XSA4, VRL, VE128, SmearGate
- EMA + tight SWA
- Late QAT + CROWN-Q
- GPTQ-lite int6 + lzma compression
- FlashAttention 3 when available
- portable fallback runtime with `COMPILE_ENABLED=0` and `SDP_BACKEND=auto -> math`

## Random Map Module

Each replaced projection uses:

1. a seeded frozen Rademacher down map
2. a learned diagonal scale in the random bottleneck
3. a tiny learned low-rank bottleneck-space residual
4. a seeded frozen Rademacher up map

The v1 objective is signs-of-life, not immediate leaderboard parity. If the first ablations learn competitively at equal steps or recover byte headroom without collapsing, the next step is to reinvest saved bytes into width.

## Status

The first real smoke run completed successfully on this branch.

- Train setup: 1 local shard of `fineweb10B_sp1024`, `ITERATIONS=4`, `TRAIN_SEQ_LEN=512`, `TRAIN_BATCH_TOKENS=65536`
- Eval setup: full validation split, int6+lzma roundtrip, sliding-window eval also completed
- Exact roundtrip metric: `val_loss=6.93100635`, `val_bpb=4.10493482`
- Submission size: `4,667,521` bytes total, `68,333` bytes of code

The portability patch is operationally important here: this machine does not have a working traced flash-SDPA path for the fallback attention branch, so the trainer now exposes `COMPILE_ENABLED` and `SDP_BACKEND` explicitly. On H100 runs, those can be re-tuned once the architecture ablations are worth optimizing.

For faster local iteration, the trainer now also exposes `EMA_DIAGNOSTIC_ENABLED` and `FINAL_SLIDING_EVAL_ENABLED`. With both disabled, the same 4-step smoke path completed in about 989 seconds locally while preserving the final int6 roundtrip metric (`val_bpb=4.10493925`) and leaving the default full-eval path unchanged.
