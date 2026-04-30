# PR 1293 — Universal Transformer with Adaptive Computation Time (ACT)

**Author:** Farcas Antonio Matei (5en5e1)
**Claimed BPB:** 1.24094165 (1 seed reported — seed 1337)
**Artifact size:** 15,792,804 bytes total (int8+zlib); model 15,743,230 bytes; code 49,574 bytes
**Track:** non_record_16mb
**Val_loss:** 2.09527800 nats

## Files retrieved
- `records__track_non_record_16mb__2026-04-03_UniversalTransformer_ACT__README.md`
- `records__track_non_record_16mb__2026-04-03_UniversalTransformer_ACT__submission.json`
- `records__track_non_record_16mb__2026-04-03_UniversalTransformer_ACT__train_gpt.py`

## Environment variables (from README run command)
RUN_ID=ut_act_d512_L9_P2, DATA_PATH=./data/datasets/fineweb10B_sp1024, TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model, VOCAB_SIZE=1024, MAX_WALLCLOCK_SECONDS=600, VAL_LOSS_EVERY=200, TRAIN_LOG_EVERY=50, NUM_SHARED_LAYERS=9, MAX_PASSES=2, MODEL_DIM=512, NUM_HEADS=8, NUM_KV_HEADS=4, MLP_MULT=2, TIE_EMBEDDINGS=1, SEED=1337

## Claimed changes (from README, verbatim)

> Instead of stacking N unique transformer blocks, this model uses a smaller set of shared blocks that are applied recurrently for multiple passes. A learned halting mechanism (ACT) allows the model to decide per-token how much computation to use.

> Key components: Shared blocks (transformer layers reused across passes); Pass embeddings (learned vectors injected at each pass); ACT halting head (single linear layer predicting per-token halt probability at each pass); Ponder cost (weighted penalty encouraging early halting, normalized by max_passes).

> This submission: 9 shared layers x 2 passes = 18 effective depth, 7,392 steps at 81.18 ms/step, 15.79 MB compressed, val_bpb 1.2409. Baseline (9 unique x 1 pass): 13,780 steps at 43.54 ms/step, 15.86 MB, val_bpb 1.2244.

> Baseline components retained: GQA (8 heads, 4 KV heads), RoPE, RMSNorm, relu^2 MLP, Muon optimizer, tied embeddings, logit softcap, int8 quantization + zlib compression.

> Bottleneck is step speed: each pass re-runs all shared blocks, doubling forward/backward time (81ms vs 44ms), ~46% fewer training steps.
