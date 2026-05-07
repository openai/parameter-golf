# Record: SkipQuant Adapter TTT + Causal PPM-D Byte Mixture

## Summary
SP8192 + SkipQuant Adapter TTT + causal byte-level PPM-D mixture.

1M-token slice, 3 seeds:
- Mean PPM BPB: 1.18765
- Mean delta vs fast no-PPM control: -0.11750

## Stack
- SP8192 tokenizer
- SkipQuant Adapter Transformer
- mixed quantized artifact
- score-first TTT
- order-5 byte PPM-D
- confidence-gated convex mixture

## Difference vs other PPM submissions
This is not only PPM-D on a dense/sliding baseline. The backbone is the submitted SkipQuant Adapter TTT stack with low-epoch fast eval:
- TTT_EPOCHS=2
- TTT_CHUNK_TOKENS=8192
- PPM_ORDER=5
- PPM_CONF_THRESH=0.78
- PPM_LAMBDA_HI=0.90
- PPM_LAMBDA_LO=0.05

## Method
TTT is score-first: each chunk is scored before adapting on that chunk.

PPM-D is byte-causal and score-first: for each emitted byte, the model reads PPM counts, records log-probability, then updates counts.

Mixture:
p_mix = lambda * p_nn + (1 - lambda) * p_ppm

High PPM confidence uses lambda=0.05. Low PPM confidence uses lambda=0.90.

Neural token probability is distributed consistently over emitted UTF-8 bytes for byte-level mixing and BPB accounting.

## Compliance
- C1: causal evaluation
- C2: convex normalized mixture
- C3: score-before-update for TTT and PPM
- C4: single left-to-right pass
- No validation prefill
- No external data
- Total artifact < 16,000,000 bytes
- Estimated full-val 8xH100 eval time: ~490s
