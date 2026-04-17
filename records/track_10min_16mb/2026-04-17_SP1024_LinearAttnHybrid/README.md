# SP1024 Linear-Attention Hybrid

Small 12-layer language model for the 10-minute / 16MB track of the
Parameter Golf challenge.

## Architecture

| Component              | Value                                     |
|------------------------|-------------------------------------------|
| Vocabulary             | SentencePiece 1024                        |
| Layers                 | 12 transformer-style blocks               |
| Width                  | 512                                       |
| Attention              | GQA 8q/2kv, head dim 64                   |
| Positional encoding    | Half-RoPE (first 32 dims)                 |
| Sliding window         | 512 on the first 8 layers, full on top 4  |
| Linear-time mixer      | layers {1, 3, 5, 7, 9} — chunked gated state update |
| MLP                    | SwiGLU, expansion ratio 5/2               |
| Output                 | Tied embedding head with `tanh` softcap=30|

The linear-time mixer runs full softmax attention within 64-token chunks and
carries a per-head low-rank state `S ∈ R^{d_k × d_v}` across chunks.  The
chunk-level decay is the product of a learned per-token sigmoid gate, so the
model can dynamically switch between fast-forgetting and long-range behaviour
per head.

## Training recipe (8×H100, 10-minute budget)

- Muon (Newton–Schulz 5-step) for 2-D weights; AdamW for embeddings and
  scalars; separate LR groups for embedding vs. head-scale vs. other 1-D.
- Trapezoidal LR: 60-step warmup → plateau → linear cool-down from
  step 1900 to 2400.
- Gradient clipping at 1.0; muon weight decay 0.05.
- EMA decay 0.9965 updated every 32 steps.
- SWA from step 2100 every 75 steps (≈4 checkpoints during cool-down).
- Late QAT (fake-quant STE, int6) enabled from step 1950 so the model
  adapts to post-training quantisation noise.

## Artifact pipeline

1. Merge SWA (fallback to EMA if SWA has no checkpoints yet).
2. Collect a diagonal-Hessian input-activation estimate on validation tokens
   for every `Linear` weight.
3. Hessian-aware per-row symmetric GPTQ-style quantisation:
   - Matrices → int6 with role-specific σ-clip (3.0 MLP, 3.1 ATTN, 3.2 MIXER).
   - Embedding → int7 with σ-clip 3.4.
   - Small 1-D tensors (scales, gains) kept in fp16.
4. Bit-pack integer streams and Brotli-11 compress (bz2 fallback).
5. Round-trip decode, re-evaluate under sliding-window eval, and report
   `final_int8_zlib_roundtrip_exact val_loss val_bpb`.

Sliding-window eval scores every token exactly once at stride 64 under a
1024-token context.

## Files

- `train_gpt.py` — single-file trainer (≤1500 lines).
- `run_leaderboard_8xh100.sh` — production launcher.
- `run_smoke_1gpu.sh` — single-GPU sanity run.
- `submission.json` — leaderboard metadata.
- `train.log` — attached after the official run is completed.

## Reproducing

```bash
# (one-time) download the SP1024 FineWeb export + tokenizer
python3 data/cached_challenge_fineweb.py --variant sp1024

# full 8×H100 leaderboard run
bash records/track_10min_16mb/2026-04-17_SP1024_LinearAttnHybrid/run_leaderboard_8xh100.sh
```
