# Experiment 037: Everything Stacked — PR#42 + LAWA + Sliding Window + NTK RoPE

## Status: RUNNING on instance 2

## Config:
- Baseline relu² (no SwiGLU — fastest per step)
- MATRIX_LR=0.06, SCALAR_LR=0.06, TIED_EMBED_LR=0.03 (PR#42 LR)
- WARMDOWN_ITERS=400 (fixed for 2K — warmdown starts at step 1600)
- FP16 embedding passthrough (eliminates 93% quant gap)
- LAWA every 50 steps during warmdown
- Sliding window eval stride=256
- NTK RoPE eval at 2048 context
- MUON_BACKEND_STEPS=3
- cuDNN SDP enabled
- COMPILE_MODE=default (CUDA graphs crash with Rotary cache)
- BYTE_GROUPING=0 (disabled, marginal savings not worth bug risk)

## Hypothesis:
Stack ALL validated improvements on top of PR#42's baseline config.
LAWA helped 0.015 BPB in exp033. Sliding window expected 0.01-0.04 BPB.
NTK RoPE expected 0.005-0.02 BPB. Combined with PR#42's training config
(which matched their 1.2197 at full scale), total could reach ~1.17-1.19 BPB.

## What Makes This Different From PR#42:
- LAWA checkpoint averaging during warmdown (PR#42 doesn't do this)
- Sliding window eval (nobody in the competition does this)
- NTK RoPE longer context eval (nobody does this either)
- Muon NS steps reduced 5→3 (minor speed gain)
