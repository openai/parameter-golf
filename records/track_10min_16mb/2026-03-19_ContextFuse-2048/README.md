This records the submissions for `ContextFuse-2048`.


Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 MLP_HIDDEN=992`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Training context: `TRAIN_SEQ_LEN=2048`
- Evaluation mode: sliding window with `EVAL_STRIDE=64 EVAL_BATCH_SEQS=256`
- Export strategy: `FP16_EMBED_PASSTHROUGH=1 FP16_LATE_K_LAYERS=0`
- Learning rates: `TIED_EMBED_LR=0.03 MATRIX_LR=0.02 SCALAR_LR=0.02`
- Muon tuning: `MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_BACKEND_STEPS=5`
- Schedule: `WARMDOWN_ITERS=3000 MAX_WALLCLOCK_SECONDS=599`

Methods implemented in `ContextFuse-2048`:
1. Long-context training at `TRAIN_SEQ_LEN=2048`.
   This increases the amount of temporal context seen in each update relative to the naive `1024`-token baseline.
2. Sliding-window final evaluation with `EVAL_STRIDE=64`.
   This scores most validation tokens with substantially richer left context than a simple non-overlapping evaluation pass.
3. FP16 tied-embedding export.
   The tied embedding matrix is used for both token lookup and output projection, so preserving it in fp16 reduces post-quantization damage where it matters most.
4. Byte-safe width adjustment with `MLP_HIDDEN=992`.
   This offsets the fp16 embedding cost while keeping the model close to the baseline family.
5. Lower-learning-rate, Muon-smoothed optimization.
   `TIED_EMBED_LR=0.03`, `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, stronger Muon momentum, and longer warmdown improve convergence in the `2048`-context regime.
6. Byte-safe export revision with `FP16_LATE_K_LAYERS=0`.
   This keeps the stronger fp16 tied-embedding win while removing the narrower Late-K fp16 passthrough so the artifact fits cleanly under the size cap.

Why the name `ContextFuse-2048`:
- The method is built around combining multiple context-improving ideas into one baseline-derived implementation rather than introducing a new backbone.
- `Context` refers to the two strongest context levers in the run:
  - longer-context training at `2048`
  - sliding-window evaluation that gives tokens richer left context at scoring time
- `Fuse` refers to the way the submission combines training-context, evaluation-context, and export-fidelity improvements into one package.
- `2048` is included because the move from `1024` to `2048` training context is one of the defining choices in the method.

Canonical successful run:
- Run ID: `attempt002_retry_h100x8_baseline2048_slide64_fp16embed_s1337`
- Hardware: `8x NVIDIA H100 80GB HBM3 (SXM)`
- Final exact quantized score: `final_int8_zlib_roundtrip_exact val_bpb:1.17792945`
- Final exact quantized loss: `final_int8_zlib_roundtrip_exact val_loss:1.98887927`
- Train stop: `step:10403/20000 val_bpb:1.1985 train_time:599075ms`
- Eval time: `132220ms`
- Serialized model int8+zlib: `15875305 bytes`
- Total submission size printed during the canonical run: `15961043 bytes`

Artifact accounting note:
- The included `train.log` is the original automatically produced training log from the successful canonical run.
- That canonical run was launched through the Modal wrapper snapshot, so the printed `Code size` in the log is `85738 bytes`.
- The `train_gpt.py` included in this record folder is the standalone submission form of the same training/eval/export path, with the Modal orchestration removed so it runs from inside the record folder.
- The standalone `train_gpt.py` in this folder is `53800 bytes`, so the estimated artifact size for this folder is:
  - model: `15875305 bytes`
  - code: `53800 bytes`
  - total: `15929105 bytes`
- This leaves `70895` bytes under the `16000000` byte cap.

Operational note:
- The canonical successful run used Modal as the GPU provider and was launched through the local budget-tracked wrapper before syncing the resulting `train.log` and artifacts back to the workspace.
- Modal is not required for the intended standalone evaluation artifact in this folder.

Why this entry exists:
- `ATTEMPT-001` proved the method family was strong but missed the size cap.
- The successful revision dropped `FP16_LATE_K_LAYERS` while keeping the stronger public win, `FP16_EMBED_PASSTHROUGH=1`.
- That change fixed the byte issue cleanly and slightly improved score over the previous attempt.

Key method choices:
1. `TRAIN_SEQ_LEN=2048` improves training quality relative to the plain baseline by giving each update materially more context.
2. Sliding-window eval with `stride=64` improves the final scored context coverage without violating the separate evaluation-time budget.
3. FP16 tied-embedding export preserves the highest-value tensor in the model under quantization.
4. `MLP_HIDDEN=992` offsets the fp16 embedding overhead while keeping the architecture close to the baseline family.
5. Lower LR plus stronger Muon smoothing materially improve the 2048-context regime.

Comparison points:
- Beats the public Naive Baseline (`1.22436570`) by `0.04643625` BPB.
- Does not beat the current pulled public best (`1.15744040`), so this is not a SOTA claim.
- This is intended as a valid, reproducible leaderboard-track submission rather than a record claim.

Reproduction command:
```bash
cd records/track_10min_16mb/2026-03-19_ContextFuse-2048
RUN_ID=baseline2048_slide64_fp16embed_bytesafe \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=599 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
TRAIN_BATCH_TOKENS=524288 \
VAL_BATCH_SIZE=524288 \
TRAIN_SEQ_LEN=2048 \
NUM_LAYERS=9 \
MLP_HIDDEN=992 \
FP16_EMBED_PASSTHROUGH=1 \
FP16_LATE_K_LAYERS=0 \
TIED_EMBED_LR=0.03 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_BACKEND_STEPS=5 \
WARMDOWN_ITERS=3000 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=256 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Included files:
- `train_gpt.py` — standalone record-folder script for the winning code path
- `train.log` — canonical successful training log from `ATTEMPT-002` retry
- `submission.json` — metadata for the entry
- `README.md` — this file
