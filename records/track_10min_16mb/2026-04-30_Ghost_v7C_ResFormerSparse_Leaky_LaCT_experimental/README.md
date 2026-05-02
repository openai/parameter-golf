# Ghost v7C — ResFormer Sparse + LeakyReLU² + LaCT TTT (Experimental / Non-record)

**Author:** lock757  
**Date:** 2026-04-30  
**Track:** `track_10min_16mb`  
**Status:** experimental / non-record candidate unless official 8×H100 logs are added.

This submission is a late emergency packaging of the Ghost v7C branch. It is not claiming a leaderboard record without full official validation logs.

## Main ideas

- **Sparse ResFormer-style Value Residual Learning**: cache layer-0 V and blend it into later layers.
- **LeakyReLU(0.5)^2** MLP activation.
- **LaCT-style chunked test-time training path**: intended to score chunks before adapting on those same chunks.
- **TTT no-QV mask**: freeze Q/V during TTT; adapt K/MLP/norms.
- **Corrected quant/dequant roundtrip** from earlier Ghost drafts.
- **Clean export before TTT** so validation-adapted weights are not serialized as the base model.
- **Corrected sliding-window scoring** using per-token loss for the stride-scored region.

## Important scoring note

If TTT is enabled, the only intended valid score is the score produced by the online score-first TTT loop. Do **not** report any BPB from a second pass that evaluates an already validation-adapted model.

Preferred score/log label:

```text
final_legal_ttt_roundtrip_exact
```

## Validation status

A tiny CPU micro-simulation is included to check core mechanics only. It does **not** validate real FineWeb BPB, official artifact size, or 8×H100 wall-clock performance.

The micro-sim checks:

- forward pass
- per-token loss path
- quant/dequant roundtrip
- score-first TTT loop shape
- no-QV freeze behavior
- compression/decompression path

See `MICRO_SIM_REPORT.md`.

## Reproduction / smoke test

```bash
python3 micro_sim.py
```

For a real run, use the challenge environment and run `train_gpt.py` with the intended dataset/tokenizer paths. This folder is packaged to be reviewed as an experimental architecture branch if full GPU logs are not present before the challenge deadline.

## Environment toggles of interest

```bash
RESFORMER_ENABLED=1
RESFORMER_MODE=sparse
RESFORMER_LEARNED=1
RESFORMER_DETACH_V0=1
LEAKY_RELU_SLOPE=0.5
LACT_CHUNK_SIZE=32
TTT_NO_QV=1
```

Suggested ablation if time permits:

```text
Run A: RESFORMER_ENABLED=0
Run B: RESFORMER_ENABLED=1
```

Same seed, same settings. Treat ResFormer as helpful only if B wins or stays flat without breaking artifact size/time.
