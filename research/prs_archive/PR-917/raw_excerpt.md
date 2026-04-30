# PR 917 — ConsensusWindow Bypass (FAT-Golf)

**Author:** TheDryhtscipe
**Claimed BPB:** small-scale only (256d/6L/500steps): pre-EMA 2.3208 (vs baseline 2.3477); post-EMA+int6 2.3438 (vs baseline 2.4185). submission.json lists val_bpb 0.0 (WIP).
**Artifact size:** not stated; ~47K new params (~0.2% of ~22M model)
**Seeds:** 3-seed means (seeds not enumerated)

## Files retrieved
- `records__track_non_record_16mb__2026-03-27_ConsensusWindow_Bypass_FAT_Golf__README.md`
- `records__track_non_record_16mb__2026-03-27_ConsensusWindow_Bypass_FAT_Golf__submission.json`
- `records__track_non_record_16mb__2026-03-27_ConsensusWindow_Bypass_FAT_Golf__train_gpt.py`

## Environment variables (from README)
CONSENSUS_WINDOW_SIZE=32 CONSENSUS_BYPASS_LAST_N=4 CONSENSUS_EMA_EXCLUDE=0

## Claimed changes (from README, verbatim)
> Adds a depthwise causal convolution bypass path to the SOTA baseline, derived from the ORC FAT-AR architecture (Factorized Attention Transformer for Autoregressive generation).

> Two additions (~90 lines of new code, ~47K params, ~0.2% of ~22M model):
> 1. ConsensusWindowEmbed: replaces SmearGate (1-token lookback, 512 params) with a depthwise causal conv1d (16-token receptive field, ~9K params). Learns per-channel weighted sum over local context at the embedding level.
> 2. ConsensusBlockBypass on deepest 4 layers: gated parallel path alongside attention. Each block gets a depthwise causal conv that processes the same normed input as attention, with a per-dimension sigmoid gate (initialized 80% attention / 20% bypass) blending the outputs.

> Everything else is identical: Muon, parameter banking, int6 QAT, EMA/SWA, BigramHash, XSA, Partial RoPE, LN Scale, VE, LeakyReLU(0.5)^2, TTT.

> Small-scale results only — awaiting H100 compute for full-scale validation. Tested at 256d, 6 layers, 500 steps on a single 4060 Ti 8GB.

> Key finding: the combined architecture produces weights far more robust to EMA averaging and int6 quantization (EMA+quant penalty +0.023 vs baseline's +0.071). Neither component alone beats baseline post-quantization — they must be combined for the synergistic effect.
