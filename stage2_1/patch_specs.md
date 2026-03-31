# Stage 2_1 Patch Specs

This is the revised `stage2_1` patch slate after the 2026-03-24 upstream PR update.

Implemented today in [patches.py]( nanoevolve/pgolf/parameter-golf/stage2_1/patches.py):

- `P201` sliding eval port
- `P204` Muon weight decay
- `P211` zstd export
- `P214` LeakyReLU(0.5)^2
- `P215` EMA(0.997)
- `P216` XSA4
- `P218` GPTQ-lite clip search
- `P220` Partial RoPE 16/64
- `P221` LN Scale
- `P223` curriculum shard ordering
- `P224` XSA-all promotion via generalized `XSA_LAYERS`

Deferred target patches:

- `P202` doc-isolated sliding eval
- `P217` full GPTQ int6
- `P219` VRL
- `P222` VE128
- `P225` K-LoRA plus Min-NLL TTT

The old `P206/P207/P208/P209/P212` cluster is now secondary rather than primary.

Primary file:

- [train_gpt.py]( nanoevolve/pgolf/parameter-golf/train_gpt.py)

Implementation rule:

- runtime patch application is the feature gate
- env flags only tune patches where explicitly consumed below

## P214: LeakyReLU(0.5)^2

- Family: training dynamics
- Target region:
  - MLP forward around the current `relu(...).square()` call
- Change:
  - replace `torch.relu(self.fc(x)).square()` with `F.leaky_relu(self.fc(x), 0.5).square()`
- New env flags:
  - `LEAKY_RELU_SLOPE`
- Expected lane:
  - pre-quant first, then deployed score
- Matched control:
  - same frontier-aligned base without the activation patch
- Acceptance:
  - repeatable pre-quant lift with no obvious throughput regression

## P215: EMA(0.997)

- Family: training-to-deployment bridge
- Target regions:
  - training loop state around the optimizer step
  - export path so EMA weights can be chosen for deployment
- New env flags:
  - `EMA_ENABLE`
  - `EMA_DECAY`
  - `EMA_EXPORT_ONLY`
- Expected lane:
  - deployed score
- Matched control:
  - same run, raw final weights
- Acceptance:
  - EMA export beats raw final export on the same run

## P216: XSA4

- Family: architecture/context
- Target regions:
  - attention forward path
  - optional new module/state for cross-sequence KV support
- New env flags:
  - `XSA_ENABLE`
  - `XSA_LAYERS`
- Expected lane:
  - pre-quant first, then deployed score
- Matched control:
  - same frontier base without XSA
- Acceptance:
  - meaningful quality lift that survives wallclock alignment

## P217: Full GPTQ Int6

- Family: deployment quantization
- Target regions:
  - export / post-training quantization path
  - calibration-data collection utilities
- New env flags:
  - `GPTQ_ENABLE`
  - `GPTQ_BITS`
  - `GPTQ_BLOCK_SIZE`
  - `GPTQ_ACTORDER`
  - `GPTQ_CALIBRATION_BATCHES`
- Expected lane:
  - deployed score
- Matched control:
  - same checkpoint with clip-and-round int6 export
- Acceptance:
  - major deployed-score lift on the same checkpoint

## P218: GPTQ-Lite Clip Search

- Family: deployment quantization fallback
- Target regions:
  - quantizer row clipping / scale selection logic
- New env flags:
  - `GPTQ_LITE_ENABLE`
  - `GPTQ_LITE_PCTS`
- Expected lane:
  - deployed score
- Matched control:
  - same checkpoint with fixed clip percentile
- Acceptance:
  - consistent improvement when full GPTQ is unavailable

## P219: VRL

- Family: architecture
- Target regions:
  - attention/value path
  - layer state needed to carry the early value residual forward
- New env flags:
  - `VRL_ENABLE`
  - `VRL_LAYERS`
  - `VRL_INIT`
- Expected lane:
  - pre-quant first, then deployed score
- Matched control:
  - same frontier base without VRL
- Acceptance:
  - meaningful structural gain at acceptable runtime cost

## P220: Partial RoPE 16/64

- Family: architecture refinement
- Target regions:
  - rotary embedding construction and application
- New env flags:
  - `ROPE_PARTIAL_DIM`
- Expected lane:
  - pre-quant and deployed score
- Matched control:
  - same frontier base with full RoPE
- Acceptance:
  - small but repeatable gain on 11L branches

## P221: LN Scale

- Family: architecture refinement
- Target regions:
  - residual / norm path
- New env flags:
  - `LN_SCALE_ENABLE`
- Expected lane:
  - pre-quant and deployed score
- Matched control:
  - same frontier base without LN scale
- Acceptance:
  - small but repeatable gain on deeper branches

## P222: VE128

- Family: funded architecture child
- Target regions:
  - embedding/front-end path
  - export inclusion rules
- New env flags:
  - `VALUE_EMBED_DIM`
  - `VALUE_EMBED_ENABLE`
- Expected lane:
  - architecture under deployment headroom
- Matched control:
  - best promoted core stack without VE128
- Acceptance:
  - only after GPTQ/zstd/float passthrough prove there is enough budget

## P223: Curriculum Shard Ordering

- Family: data/training
- Target regions:
  - dataset / shard enumeration order
  - any `TokenStream` or shard-loader logic that currently consumes files in fixed order
- New env flags:
  - `SHARD_ORDER`
  - `SHARD_ORDER_SOURCE`
- Expected lane:
  - equal-wallclock training quality
- Matched control:
  - same base and wallclock with default shard order
- Acceptance:
  - real gain at equal wallclock, not just different step count

## P224: XSA-All Promotion

- Family: architecture/context child
- Target regions:
  - XSA layer selection logic added for `P216`
- New env flags:
  - `XSA_ENABLE`
  - `XSA_LAYERS`
- Expected lane:
  - pre-quant and deployed score on top of a stable `XSA4` winner
- Matched control:
  - best promoted `XSA4` branch
- Acceptance:
  - incremental gain that survives runtime and artifact cost

## P225: K-LoRA Plus Min-NLL TTT

- Family: evaluation-time adaptation
- Target regions:
  - fresh TTT LoRA module construction to be ported from records or open PRs
  - per-epoch scoring and document-level selection logic
- New env flags:
  - `TTT_K_LORA`
  - `TTT_MIN_NLL_SELECT`
  - `TTT_EPOCHS`
- Expected lane:
  - final TTT score only
- Matched control:
  - same checkpoint with the current root TTT path
- Acceptance:
  - large TTT-only gain under the eval wallclock cap

## Current Runnable Order

1. `P214` LeakyReLU(0.5)^2
2. `P215` EMA(0.997)
3. `P216` XSA4
4. `P204` Muon weight decay at `0.04`
5. `P218` GPTQ-lite clip search
6. `P220` Partial RoPE 16/64
7. `P221` LN Scale
8. `P201` sliding eval
9. `P211` zstd export
10. `P223` curriculum shard ordering
11. `P224` XSA-all promotion

## Deferred Frontier Targets

1. `P217` full GPTQ int6
2. `P202` doc-isolated sliding eval
3. `P219` VRL
4. `P222` VE128

## Deferred For Now

- NorMuon
- OrthoInit + muP
- solo SmearGate
- solo BigramHash
- FA3-first throughput patching
- label smoothing
- MTP
- `P225` K-LoRA plus Min-NLL TTT for the no-TTT portfolio

They are not banned. They are just no longer on the shortest path to a winning no-TTT stack.
