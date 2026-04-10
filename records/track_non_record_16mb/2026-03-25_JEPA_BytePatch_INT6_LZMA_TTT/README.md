Non-record submission using JEPA (Joint Embedding Predictive Architecture) encoder-decoder as an alternative to GPT.

Architecture:

Unlike the standard causal GPT used by all leaderboard entries, this submission uses a two-stage JEPA architecture:

| Component | Config |
|-----------|--------|
| Tokenizer | Pure byte-level (vocab 260, no BPE) |
| Encoder | 5 layers × 2 depth-recurrent repeats, dim 480, 6 heads (3 KV), GQA |
| Encoder output | Patch-based (patch_size=8), latent projection (dim 192) with SIGReg regularization |
| Decoder | 7 causal layers, dim 480, 4 heads, conditioned on encoder latents |
| Total blocks per forward | 17 (10 encoder + 7 decoder) + projector/predictor MLPs |

The encoder processes input patches into latent representations via a JEPA objective (predicting latent targets with a predictor network, regularized by SIGReg). The decoder autoregressively predicts bytes conditioned on these latents. Training uses a combined loss: JEPA prediction loss (weight 0.5) + byte cross-entropy (weight 3.0).

Quantization & Compression:

- INT6 optimal-clip quantization: All weight categories (MLP, attention, embeddings, other) quantized to [-31, 31] range stored as int8, with per-row scales in fp16. Clip percentile grid search over [0.9990, 0.9995, 0.9999, 0.99999, 1.0] minimizing reconstruction MSE.
- STE QAT: Straight-through estimator quantization-aware training activated during warmdown when LR scale drops below 0.15, simulating INT6 rounding in the forward pass.
- LZMA compression (preset 9): Exploits the reduced entropy from INT6's 63-value range for better compression than zlib/zstd.
- Small/control tensors passed through as fp16.

Test-Time Training:

- Sliding window TTT with chunk-sequential adaptation (chunk_tokens=32768)
- SGD optimizer (lr=0.002, momentum=0.9, cosine LR across chunks)
- 2 epochs per chunk, stride 256, batch_seqs=32
- All parameters adapt

Results:

| Metric | Value |
|--------|-------|
| Pre-quantization val_bpb | 1.2957 |
| Final TTT val_bpb | 1.2622 |
| Training steps | 10,635 / 20,000 |
| Step avg | 56.39 ms |
| Model params | 24,593,530 |
| Compressed model (INT6+LZMA) | 15,625,312 bytes |
| Code size | 66,315 bytes |
| Total submission size | 15,691,627 bytes |
| TTT eval time | 542s |
| Peak memory | 9,994 MiB allocated |

Setup & Run:

This submission uses a pure byte-level tokenizer (vocab 260) instead of the upstream default SentencePiece BPE (vocab 1024). The byte260 variant is not in the pre-built HuggingFace cache, so generate it locally with the export pipeline using the included `tokenizer_specs.json`:

```bash
python3 data/download_hf_docs_and_tokenize.py --output-root data \
  --tokenizer-config records/track_non_record_16mb/2026-03-25_JEPA_BytePatch_INT6_LZMA_TTT/tokenizer_specs.json
```

This downloads `docs_selected.jsonl` from HuggingFace, byte-tokenizes it, and populates `./data/datasets/fineweb10B_byte260/` and `./data/tokenizers/fineweb_pure_byte_260.json`.

Then run:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script defaults to the byte260 paths (`DATA_PATH=./data/datasets/fineweb10B_byte260`, `TOKENIZER_PATH=./data/tokenizers/fineweb_pure_byte_260.json`).

Hyperparams (UNTUNED!):

```
NUM_LAYERS=5 ENCODER_REPEATS=2 DECODER_LAYERS=7
MODEL_DIM=480 NUM_HEADS=6 NUM_KV_HEADS=3 DECODER_HEADS=4
TRAIN_SEQ_LEN=2047 TRAIN_BATCH_TOKENS=524032
WARMDOWN_ITERS=3500 MUON_MOMENTUM=0.99
MATRIX_LR=0.025 SCALAR_LR=0.025
EMA_DECAY=0.997 LATE_QAT_THRESHOLD=0.15
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=2 TTT_CHUNK_TOKENS=32768
TTT_BATCH_SEQS=32 VAL_SLIDING_STRIDE=256
```
