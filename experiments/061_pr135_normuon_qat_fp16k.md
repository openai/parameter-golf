# Experiment 061: PR135 + NorMuon + QAT + FP16 last-2 K projections

## Status: LAUNCHING

## Config
- Same as 060 (1.1474 BPB) but keep last 2 layers' K projections in fp16
- PR114/99 found these are quantization-sensitive
- blocks.7.attn.c_k.weight and blocks.8.attn.c_k.weight stay fp16

## Hypothesis
Selective fp16 for late K projections helps ~0.001 BPB from sharper late-layer attention.

## Results
*Pending*
