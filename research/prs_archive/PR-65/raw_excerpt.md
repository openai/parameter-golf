# PR 65 — SmearGate + OrthoInit + Muon WD + Int6 STE QAT + MLP 3x + Sliding Window

**Author:** aquariouseworkman
**Branch created:** 2026-03-19
**Claimed BPB:** 1.1556 (post-quant sliding window val_bpb, stride=64)
**Artifact size:** 15,878,809 bytes (15.1 MB, int6+zstd-22)
**Seeds:** 1337 (single seed)

## Files retrieved
- `records__track_10min_16mb__2026-03-19_smeargate_orthoinit_muonwd__README.md`
- `records__track_10min_16mb__2026-03-19_smeargate_orthoinit_muonwd__submission.json`
- `records__track_10min_16mb__2026-03-19_smeargate_orthoinit_muonwd__train_gpt_v5.py`

## Claimed changes (from README, verbatim)

> A 22.4M parameter transformer language model trained in under 10 minutes on 8xH100 GPUs, compressed to a 15.1MB artifact via int6 quantization-aware training and zstd-22. The architecture combines a SmearGate bigram embedding layer, orthogonal weight initialization, 3x MLP expansion, U-Net skip connections, and decoupled Muon weight decay, evaluated with sliding window context at stride 64.

Techniques listed in submission.json: SmearGate, Orthogonal Init, Muon Weight Decay, Int6 STE QAT, MLP 3x, Sliding Window Eval, Bigram Hash Embedding, U-Net Skip Connections, Muon Momentum Warmup, zstd-22 Compression.

Config: NUM_LAYERS=9, MODEL_DIM=512, NUM_HEADS=8, NUM_KV_HEADS=4, MLP_MULT=3, TRAIN_SEQ_LEN=1024, MUON_WEIGHT_DECAY=0.01, MUON_MOMENTUM=0.99 (warmup 0.92 over 1500 steps), WARMDOWN_ITERS=3000, EVAL_STRIDE=64, BIGRAM_HASH_BUCKETS=4096, BIGRAM_HASH_DIM=128, TIED_EMBED_LR=0.030, MATRIX_LR=0.020, SCALAR_LR=0.020, quantization_gap_bpb=0.0001.
