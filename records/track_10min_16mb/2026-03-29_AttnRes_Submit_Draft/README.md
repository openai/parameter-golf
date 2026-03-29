Main contributions:
- Block-level AttnRes depth mixing via `ATTN_RES_*`, motivated by Attention Residuals (`arXiv:2603.15031`, https://arxiv.org/abs/2603.15031). In this run: `ATTN_RES_ENABLED=1`, `ATTN_RES_NUM_BLOCKS=1`, `ATTN_RES_MIX=0.15`, `ATTN_RES_TEMPERATURE=1.0`.
- Stronger baseline capacity at fixed depth/width using `MLP_MULT=3` (vs common `2`) with tied embeddings.
- Sliding-window validation (`EVAL_STRIDE=64`) so scored tokens see near-max context during evaluation.
- Lean single-path post-training export: per-row GPTQ-lite-style int8 quantization + compressed artifact + explicit roundtrip evaluation.
- Submission safety checks in the training script: `REQUIRE_ZSTD=1` (fail fast if zstd is unavailable) and `MAX_SUBMISSION_BYTES=16000000` (hard cap check on code + compressed model).

AttnRes:
- The paper’s core idea is depth-wise softmax aggregation over prior representations; we implement that as `DepthAttentionResidual` with per-layer query vectors and softmax over depth sources.
- The paper’s Block AttnRes idea is to partition layers and attend over block-level states; this script builds an append-only `source_bank` of detached block-boundary states and attends over that bank.
- Temperature-controlled depth attention is exposed through `ATTN_RES_TEMPERATURE` and used directly in the depth-attention score scaling.
- This is an implementation inspired by the Block AttnRes mechanism, not a full reproduction of all systems details from the paper (for example, pipeline communication strategy).

Summary of the run setup:
- Architecture: 9-layer GPT (`MODEL_DIM=512`, `NUM_HEADS=8`, `NUM_KV_HEADS=4`, `MLP_MULT=3`) with block AttnRes enabled.
- Eval metric: tokenizer-agnostic `val_bpb` on full `fineweb_val_*` split.
- Export path: int8 quantization + compressor from runtime (`zstd` required in this submission path).
- Submission guards in script:
  - `REQUIRE_ZSTD=1` (fails fast if zstd is unavailable).
  - `MAX_SUBMISSION_BYTES=16000000` (fails if `code + compressed model` exceeds cap).

Script used:
- `train_gpt.py` (copied from repository root `train_gpt_attn_res_submit.py`)

Command (track-relevant params):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RUN_ID=train_seed1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=20 \
USE_COMPILE=1 \
ATTN_RES_ENABLED=1 \
ATTN_RES_NUM_BLOCKS=1 \
ATTN_RES_MIX=0.15 \
ATTN_RES_TEMPERATURE=1.0 \
SMEAR_ENABLED=0 \
BIGRAM_VOCAB_SIZE=0 \
MUON_WEIGHT_DECAY=0.06 \
GRAD_CLIP_NORM=0.2 \
REQUIRE_ZSTD=1 \
MAX_SUBMISSION_BYTES=16000000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (fill from run log):
- Final float eval: `final_selected_float_exact val_loss:<FILL> val_bpb:<FILL>`
- Final roundtrip eval: `final_int8_<compressor>_roundtrip_exact val_loss:<FILL> val_bpb:<FILL>`
- Serialized model int8+<compressor>: `<FILL>` bytes
- Code size: `<FILL>` bytes
- Total submission size int8+<compressor>: `<FILL>` bytes
- Peak memory: `<FILL>`
- Train time: `<FILL>`

Included files in this record folder:
- `README.md`
- `submission.json`
- `train_seed1337.log` (copy of the produced log file from `logs/<RUN_ID>.txt`)
- `train_gpt.py` (script snapshot used in the run)
