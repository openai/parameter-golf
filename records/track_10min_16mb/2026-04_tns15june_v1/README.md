## SP8192 Frontier: Final Submission

### Approach

SP8192-tokenized FineWeb with the frontier component set that this branch can
support without external kernels:

1. **SP8192 tokenizer + data**: pre-tokenized mirror from
   `kevclark/parameter-golf` via `data/cached_challenge_fineweb.py`.
2. **11 layers x 512 dim x MLP 4x**: tied embeddings, QK gain 5.25,
   LeakyReLU(0.5)^2 MLP for the final frontier run.
3. **Partial RoPE (25%)**: first 16 of 64 head dims rotate.
4. **Layerwise RMSNorm scale**: learnable norm scales in each block.
5. **Targeted middle-layer recurrence**: layers 3..5 looped 3x in an
   11-layer physical backbone (17 effective visits).
6. **Parallel residuals**: attention and MLP share the pre-residual state, plus
   a learned softmax mix from encoder outputs into decoder layers.
7. **Optimizer**: MuonEq-R with row normalization, WD 0.095, matrix LR 0.022,
   scalar LR 0.02, tied embedding LR 0.03, grad clipping 0.3, EMA 0.9965, and
   wallclock-fraction warmdown 0.72.
8. **Export**: int6 GPTQ on non-embedding matrices with SDClip k=12.85, int8
   amax embeddings, byte-shuffle, and Brotli quality 11 compression.
9. **Eval**: score-first sliding TTT with lr 0.005, 3 epochs, 32K-token chunks,
   and stride 256. Tokens are scored before any update on the same chunk.

### Configuration

Run from the repository root, but execute this record folder's `train_gpt.py`.
Key defaults:

```bash
DATA_PATH=data/datasets/fineweb10B_sp8192
TOKENIZER_PATH=data/tokenizers/fineweb_8192_bpe.model
VOCAB_SIZE=8192
NUM_UNIQUE_LAYERS=11 NUM_RECURRENCES=3 TARGETED_RECURRENCE=1
RECURRENCE_START_LAYER=3 RECURRENCE_END_LAYER=5
MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
MLP_NEGATIVE_SLOPE=0.5 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1
QK_GAIN_INIT=5.25 ROPE_FRACTION=0.25
PARALLEL_RESIDUALS=1 PARALLEL_LATER_RESIDUALS=1 LAYERWISE_NORM_SCALE=1
MATRIX_LR=0.022 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_WEIGHT_DECAY=0.095 MUON_ROW_NORM=1
GRAD_CLIP_NORM=0.3 WARMDOWN_FRAC=0.72 EMA_DECAY=0.9965
EXPORT_BITS=6 EMBED_EXPORT_BITS=8 QUANT_METHOD=gptq GPTQ_EMBED=0
USE_SDCLIP=1 SDCLIP_K=12.85
EVAL_STRIDE=256 TTT_ENABLED=1 TTT_LR=5e-3
TTT_CHUNK_TOKENS=32768 TTT_EPOCHS=3
COMPRESS_METHOD=brotli BYTE_SHUFFLE_STRIDE=2
TRAIN_BATCH_TOKENS=524288 MAX_WALLCLOCK_SECONDS=600
```

### Data Prep

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192
```

### Run

```bash
bash records/track_10min_16mb/2026-04_tns15june_v1/run.sh
```

After the run, copy `logs/submission_frontier.txt` into this folder as
`train.log` and fill `submission.json` from the final
`final_int8_zlib_roundtrip_exact`, `Code size`, and `Total submission size`
lines.

### Key Metrics

Filled post-run from `logs/submission_frontier.txt`.

- Pre-quant eval: `val_loss:___ val_bpb:___`
- Post-quant roundtrip: `val_loss:___ val_bpb:___`
- Export gap: `___`
- Train time: `___ ms`
- Eval time: `___ ms`
- Serialized model: `___ bytes`
- Code size: `___ bytes`
- Total submission size: `___ bytes`

### Included Files

- `train_gpt.py`: self-contained compressed wrapper for the final training code
- `requirements.txt`: extra Python dependencies for self-setup
- `train.log`: full 8-GPU run log, filled after the final run
- `submission.json`: final metrics, filled after the final run

### Known Gaps

- No flash-attn-3 dependency; this branch uses PyTorch SDPA for portability.
- Progressive recurrence activation from the upstream record is not implemented.
- TTT is score-first and causal, but adaptation is rank-partitioned under DDP.
