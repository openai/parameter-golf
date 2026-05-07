## Candidate: PR315 Base With Budgeted Experimental Branches

This folder is the new mainline candidate forked from the exact [PR #315](https://github.com/openai/parameter-golf/pull/315) frontier base, not from the older local 10-layer runs.

The starting point is the validated `11L + XSA4 + EMA + Partial RoPE + LN Scale` stack:
- 11 transformer layers
- XSA on the last 4 layers
- EMA with decay `0.997`
- Partial RoPE on `16/64` head dims
- LN Scale `1/sqrt(layer+1)`
- FA3, seq2048, bigram hash `2048x128`, SmearGate
- int6 mixed quantization on MLP/attention with int8 embeddings

## What Changed In This Candidate

1. **Exact frontier base recovered**
   - The original `PR315` record is checked in separately at [2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248](/Users/divy/Downloads/paragolf/parameter-golf/records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248).
   - This candidate starts from that codepath rather than the older 10-layer int5/int6 family.

2. **Late QAT audited**
   - `PR315`’s own post-submission note says Late QAT was inert under `torch.compile`.
   - This candidate keeps the flag available but does not rely on it as an active gain source.

3. **Custom packed codec experiment**
   - The export path supports a structured codec that removes monolithic `torch.save + zstd` as the primary artifact path.
   - Int6 tensors are packed densely at 6 bits/value before optional per-tensor deflate.
   - Non-int6 tensors use per-tensor raw-or-deflate selection.
   - The script can still log the legacy `zstd` artifact size for direct comparison.
   - Current directional evidence rejects this codec for submission: on the reduced 4090 gate run it produced `6,183,803` total bytes versus `5,055,622` for `zstd`.

4. **Gated adaptive TTT experiment**
   - A document-aware LoRA TTT path is integrated but gated behind `TTT_ENABLED=1`.
   - Only the longest documents are routed into TTT, controlled by `TTT_DOC_PERCENTILE` and defaulting to the longest `5%`.
   - Adaptation is limited to the LM head and Q/V projections of the last few blocks.
   - Hard caps `TTT_MAX_DOCS` and `TTT_MAX_EVAL_SECONDS` were added so overflowed TTT work falls back to normal scoring instead of stalling eval.

## Default Behavior

By default this candidate runs the frontier base with the standard `zstd` export and TTT disabled:

```bash
EXPORT_CODEC=zstd \
COMPARE_EXPORT_CODECS=0 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script bakes in the PR315-style structural and optimizer defaults:
- `NUM_LAYERS=11`
- `BIGRAM_VOCAB_SIZE=2048`
- `XSA_LAST_N=4`
- `EMA_ENABLED=1`, `EMA_DECAY=0.997`, `SWA_ENABLED=0`
- `ROPE_DIMS=16`, `LN_SCALE=1`
- `MATRIX_LR=0.025`, `SCALAR_LR=0.025`, `TIED_EMBED_LR=0.035`
- `MUON_MOMENTUM=0.99`, `MUON_MOMENTUM_WARMUP_START=0.92`, `MUON_MOMENTUM_WARMUP_STEPS=1500`
- `MUON_WD=0.04`, `ADAM_WD=0.04`
- `ITERATIONS=9000`, `WARMDOWN_ITERS=3000`

To benchmark the rejected codec experiment explicitly:

```bash
EXPORT_CODEC=custom \
COMPARE_EXPORT_CODECS=1 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To enable the long-document TTT branch with explicit caps:

```bash
EXPORT_CODEC=zstd \
COMPARE_EXPORT_CODECS=0 \
TTT_ENABLED=1 \
TTT_DOC_PERCENTILE=95 \
TTT_TARGET_LAST_N=2 \
TTT_LORA_RANK=4 \
TTT_CHUNK_SIZE=256 \
TTT_BATCH_SIZE=16 \
TTT_MAX_DOCS=4 \
TTT_MAX_EVAL_SECONDS=45 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Logging Expectations

The script now emits:
- legacy `mixed+zstd` size when `COMPARE_EXPORT_CODECS=1` or `EXPORT_CODEC=zstd`
- custom `mixed+custom-packed` size when the custom codec is actually built
- `final_quant_roundtrip_exact`
- `final_quant_sliding_window_exact`
- `final_longdoc_ttt_exact` when `TTT_ENABLED=1`

## Current Validation Status

Inherited reference numbers from `PR315`:
- sliding `val_bpb`: `1.1248`
- total bytes: `15,612,308`

Official `8xH100` validation run on this candidate:
- seed: `42`
- stopped at `step 4625` on the `600.037s` wallclock cap
- peak memory: `26152 MiB` allocated, `26526 MiB` reserved
- serialized model: `105,783,402` bytes
- total submission size: `15,733,011` bytes
- final quantized roundtrip exact `val_bpb`: `1.16892776`
- final quantized sliding-window exact `val_bpb`: `1.14586586`

The run is valid under the `16,000,000` byte cap, but it does not beat the inherited `PR315` reference. This folder is therefore a truthful official record candidate and implementation branch, not a promoted new SOTA.

Preliminary reduced gate results from an earlier 4090 smoke path, collected before these PR315-style defaults were baked directly into the script:
- `zstd` total submission size: `5,055,622` bytes.
- custom packed codec total submission size: `6,183,803` bytes.

That early smoke signal still suggests the custom codec is low-EV, but it should be treated as directional only until rerun on the corrected default base. The TTT branch also remains experimental; it is implemented with budget caps, but it has not yet cleared the directional gate strongly enough to be promoted into the default submission path.
