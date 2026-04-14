# PR 461 — Depth Recurrence + Legal Score-First TTT with SGD Momentum (11L VE128 PartialRoPE LegalTTT, non-record)

**Author:** Chris McClendon (Christopher-Lee-McClendon)
**Claimed BPB:** val_bpb 1.14458409 (val_loss 1.93257718). Pre-TTT val_bpb 1.1611 / val_loss 1.9605. TTT gain −0.0165 BPB.
**Artifact size:** 14,789,419 bytes total (14,717,713 model int6+zstd, 71,706 code)
**Seeds:** Not explicitly specified (single run on 4×A100-40GB; eval 1046s on 1×A100)

## Files retrieved
- `records__track_non_record_16mb__2026-03-22_11L_VE128_PartialRoPE_LegalTTT__README.md`
- `records__track_non_record_16mb__2026-03-22_11L_VE128_PartialRoPE_LegalTTT__submission.json`
- `records__track_non_record_16mb__2026-03-22_11L_VE128_PartialRoPE_LegalTTT__train_gpt.py`

## Environment variables (from run command in README)

```
RUN_ID=i15_11L_ve128
NUM_LAYERS=11 UNIQUE_LAYERS=10
DATA_PATH=./data/datasets/fineweb10B_sp1024/
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=0 ITERATIONS=5200 VAL_LOSS_EVERY=500
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10
ROPE_DIMS=16 LN_SCALE=1
BIGRAM_VOCAB_SIZE=2048
XSA_LAST_N=4 EVAL_STRIDE=64
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3
TTT_FREEZE_BLOCKS=2 TTT_BATCH_SEQS=32 TTT_MOMENTUM=0.9
torchrun --standalone --nproc_per_node=4 ...train_gpt.py
```

## Claimed changes (from README, verbatim)

> Non-record unlimited-compute submission. Headline: val_bpb = 1.14458, Pre-TTT 1.1611, TTT gain −0.0165, Artifact 14.79 MB.
>
> Novel & Creative Contributions:
>
> 1. High-Yield Legal TTT via Selective Freezing + SGD Momentum. SGD + momentum (0.9) instead of AdamW — simpler optimizer with implicit regularization; lower memory footprint (no second-moment buffers) enables larger effective batch processing. 3 epochs per chunk instead of 1 — repeated passes over each 32K-token chunk. Freeze the first 2 blocks during TTT — keeps 19.9M of 24.6M parameters trainable on later layers. TTT gain −0.0165 BPB (1.1611 → 1.1446), vs −0.0068 with prior AdamW-1-epoch approach.
>
> 2. Depth Recurrence. 11 logical layers but only 10 unique BlockCores — one core is reused at two different depths. Each Block wraps a shared core with its own per-layer LayerNorm buffers and scaling factors.
>
> 3. Partial Rotary Position Embeddings (16 of 64 dims).
>
> 4. Value Embeddings on Deep Layers Only. Layers 9 and 10 receive 128-dim learned value embeddings. Per-layer scale factors initialized to 0.1.
>
> 5. Layer-Norm Depth Scaling. 1/√(layer_idx + 1).
>
> TTT Protocol (Legal Score-First): for each 32K-token chunk: (1) model.eval() + torch.inference_mode() → Forward pass on chunk, accumulate NLL (SCORE); (2) model.train() → SGD(lr=0.002, momentum=0.9), 3 epochs (TRAIN); (3) Advance. TTT optimizer SGD momentum=0.9, lr 0.002, 3 epochs per chunk, chunk size 32768, stride 64, frozen first 2 of 11 blocks, trainable params 19,911,748/24,634,452, eval time 1046s on 1×A100.
>
> Architecture: 11 logical (10 unique shared BlockCores), emb dim 512, 8 heads (64/head), 4 KV heads, MLP 3× (1536) ReLU² SmearGate, Vocab 1024 SP BPE, BigramHash 2048, RoPE partial 16/64 NTK-aware, Value Embeddings 128d on layers 9-10 per-layer scale init 0.1, LN scale 1/√(layer+1), XSA last 4, U-Net skips, 24,634,452 params total.
>
> Training: 4×A100-40GB, 5200 steps, 2472s wallclock, Muon (hidden/attn) + Adam (embeddings/scalars), SWA 12 checkpoints from step 4650, Late QAT at step 4901 (scale<0.1), Int6+zstd-22.
