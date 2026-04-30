# Non-record: SP8192 + QK 5.40 + Legal Score-First TTT(4) with strict 570s cap

**score-first val_bpb (quantized_ttt):** `1.08149059` (seed `1337`)

**reference roundtrip val_bpb (quantized):** `1.09962158`
**reference sliding val_bpb (no TTT):** `1.08292146`

Hardware: `8xH100 80GB SXM`

## Run Configuration

- Base: `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`
- `QK_GAIN_INIT=5.40`
- `MUON_WD=0.100`
- `TTT_ENABLED=1`
- `TTT_EPOCHS=4`
- `TTT_LR=0.0055`
- `MAX_WALLCLOCK_SECONDS=570`
- `GPTQ_RESERVE_SECONDS=20`

## Measured Outputs (seed 1337)

- train time: `550.146s`
- sliding eval time: `91.746s`
- ttt eval time: `411.361s`
- artifact size: `15,994,638 bytes`

## Compliance Notes

- No tokenizer changes
- No dataset changes
- Score-first ordering in TTT path (score chunk first, update after scoring)
- No n-gram overlays / no SLOT / no multi-pass rescoring
- Artifact under `16,000,000` bytes
- Train and eval each under `600s`

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed1337.log`
