# PR 162 — Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA

**Author:** Raahil Shah (raahilshah)
**Branch created:** 2026-03-20
**Claimed BPB:** 1.14582 mean (3-seed: 1.14597 / 1.14656 / 1.14492, std 0.00082)
**Artifact size:** 15,862,650 bytes (15.86 MB, int6+zstd-22)
**Seeds:** 1337, 42, 7

## Files retrieved
- `records__track_10min_16mb__2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA__README.md`
- `records__track_10min_16mb__2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA__submission.json`
- `records__track_10min_16mb__2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA__train_gpt.py`

## Claimed changes (from README, verbatim)

> Seven techniques stacked on the baseline 9-layer, 512-dim GPT:
> 1. Per-Row Int6 Quantization + zstd-22 Compression — MLP and attention weight matrices quantized to int6 ([-32, 31]) with per-row scaling. Tied embeddings remain in fp16. The last transformer layer's key projection is kept in fp16 to reduce the quantization penalty on late-layer attention. zstd at level 22 provides ~5% better compression than zlib-9 on int6 data.
> 2. 3x MLP Expansion — MLP hidden dimension increased from 1024 (2x) to 1536 (3x). Single largest contributor.
> 3. SmearGate — learned gate blending each token's embedding with the previous token's embedding. ~512 parameters.
> 4. BigramHash Embedding — 4096-bucket hash table (dim=128, projected to 512) mapping adjacent token pairs via (prev * 31 + curr) % 4096. ~524K parameters.
> 5. Orthogonal Weight Initialization — all large weight matrices initialized with orthogonal_(gain=1.0). Output projections scaled by 1/sqrt(2 * num_layers) (muP).
> 6. Muon Optimizer with Weight Decay — WD=0.04 (swept 0.01–0.05). Momentum warmup 0.92 to 0.99 over 1500 steps. AdamW WD=0.01 for embedding/scalar.
> 7. Stochastic Weight Averaging — SWA every 50 steps over last 50% of training (~30 checkpoints). Swept swa_every 200-25, optimal 50.

Config: num_layers=9, model_dim=512, mlp_mult=3.0, train_seq_len=2048, train_batch_tokens=786432, warmdown_iters=3000, grad_clip_norm=0.3, eval_stride=64, bigram_vocab_size=4096, bigram_dim=128. Pre-quant 1.1616, quantization penalty 0.016 bpb, 7379 steps in 600s (81.3 ms/step). Improvement over prior SOTA (1.1748): -0.0290 bpb.
