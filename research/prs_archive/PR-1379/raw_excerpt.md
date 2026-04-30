# PR 1379 — Mixed Precision Quantization + Causal Backoff N-gram + Complementary Training

**Author:** Lucas Ercolano (LucasErcolano)
**Claimed BPB:** 0.416182 (3-seed avg; seed 42=0.415890, 1337=0.415507, 7=0.417149)
**Artifact size:** 15,623,718 bytes (max 15.62 MB)
**Seeds:** [42, 1337, 7]
**Track:** 10min_16mb, 8xH100 SXM, 600s
**Date:** 2026-04-04

## Files retrieved
- `records__track_10min_16mb__2026-04-04_LucasErcolano_MixedQuantNgram__README.md`
- `records__track_10min_16mb__2026-04-04_LucasErcolano_MixedQuantNgram__submission.json`
- `records__track_10min_16mb__2026-04-04_LucasErcolano_MixedQuantNgram__train_gpt.py`
- `records__track_10min_16mb__2026-04-04_LucasErcolano_MixedQuantNgram__eval__eval.sh`

## Environment variables (from eval script)
`COMPILE_MODEL=1`, `COMPILE_MUON=1`, `ADAM_FUSED=1`, `USE_LIBUV=0`, `EVAL_ONLY=1`, `CHECKPOINT_PATH=final_model.pt`, `EVAL_TIMEOUT_SECONDS=580`, `EVAL_STRIDE=256`, `EVAL_BATCH_SEQS=32`, `TTT_ENABLED=0`, `NPROC_PER_NODE=8`.

## Claimed changes (from README, verbatim)
"Combines a highly optimized neural baseline with a strict, DDP-safe causal n-gram mixer and complementary training, fitted into the 16MB artifact limit via asymmetric mixed-precision quantization. (1) Mixed Precision Quantization (Int5/Int6): MLP Layers quantized to int5; Attention/Embeddings quantized to int6. Dynamic QAT: CastedLinear modules simulate QAT with dynamic clipping based on target layer. (2) Complementary Training: neural model trained to specialize in tokens hard for n-grams; loss re-weighted as w_i = 1 - alpha * p_bigram(token_i). (3) Strictly Legal Causal Backoff N-gram Mixer: entropy-adaptive alpha blending; Score-First Legality — mixer updates only after sliding-window evaluation; DDP Synchronization with dist.barrier() to prevent causal leaks. Base Neural Stack (from PR #549): 11L GQA Transformer, 512d, 8 heads, 4 KV heads, MLP 3.0x with LeakyReLU(0.5)^2, Parallel Muon optimizer, SmearGate + BigramHash(2048) + OrthoInit, Value-Residual Embeddings (VE128). Artifact: lzma compression of mixed-precision state dict, ~15.6 MB."
