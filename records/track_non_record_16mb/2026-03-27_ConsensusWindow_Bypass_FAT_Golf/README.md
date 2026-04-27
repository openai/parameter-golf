# ConsensusWindow Bypass (FAT-Golf)

Adds a depthwise causal convolution bypass path to the SOTA baseline, derived from the ORC FAT-AR architecture (Factorized Attention Transformer for Autoregressive generation).

## Changes from baseline (abaybektursun, 1.1194 BPB)

Two additions (~90 lines of new code, ~47K params, ~0.2% of ~22M model):

1. **ConsensusWindowEmbed**: replaces SmearGate (1-token lookback, 512 params) with a depthwise causal conv1d (16-token receptive field, ~9K params). Learns per-channel weighted sum over local context at the embedding level.

2. **ConsensusBlockBypass** on deepest 4 layers: gated parallel path alongside attention. Each block gets a depthwise causal conv that processes the same normed input as attention, with a per-dimension sigmoid gate (initialized 80% attention / 20% bypass) blending the outputs.

Everything else is identical: Muon, parameter banking, int6 QAT, EMA/SWA, BigramHash, XSA, Partial RoPE, LN Scale, VE, LeakyReLU(0.5)^2, TTT.

## Status

**Small-scale results only** — awaiting H100 compute for full-scale validation.

Tested at 256d, 6 layers, 500 steps on a single 4060 Ti 8GB.

### Results (3-seed means)

Pre-EMA val_bpb: baseline 2.3477, ours 2.3208 (delta -0.027)
Post-EMA+int6 val_bpb: baseline 2.4185, ours 2.3438 (delta -0.075)

Key finding: the combined architecture produces weights far more robust to EMA averaging and int6 quantization (EMA+quant penalty +0.023 vs baseline's +0.071). Neither component alone beats baseline post-quantization — they must be combined for the synergistic effect.

## Environment variables

```
CONSENSUS_WINDOW_SIZE=32    # Conv1d receptive field (0 = use SmearGate)
CONSENSUS_BYPASS_LAST_N=4   # Number of deepest layers with bypass
CONSENSUS_EMA_EXCLUDE=0     # Exclude consensus from EMA (not recommended)
```

## Source

Full repository with tests and ablation scripts: https://github.com/TheDryhtscipe/golf-model-1
