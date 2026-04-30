This record captures a non-record 16MB submission centered on an SP8192 BPE run, plus a JEPA-style distillation implementation integrated into `train_gpt.py`.

This run itself is a stable 600s 8xH100 baseline under budget. The key contribution in this package is the JEPA path design and integration: latent prediction over repeated mid-depth passes, paired with EMA self-distillation scheduling and export-safe inference.

Configuration:
- Track: `non-record` (under `16,000,000` bytes)
- Layout: `VOCAB_SIZE=8192 NUM_LAYERS=9 MODEL_DIM=448 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tokenizer: SentencePiece BPE 8192 (`fineweb_8192_bpe.model`)
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Eval: sliding-window validation with `EVAL_STRIDE_FRAC=0.5`
- Quant/export: GPTQ int8 + zstd

## Why this direction

This submission targets the “JEPA” idea explicitly listed in the repo’s “Requests for PRs” section. The goal here is signs-of-life and reproducible implementation quality under challenge constraints, not a record claim. This package focuses on a clean JEPA-style latent prediction path (with EMA-teacher distillation coupling) and a reproducible non-record submission artifact.

JEPA/distillation engineering (core modeling ideas):
- **Latent predictive objective over intra-loop depth**: the model is trained to predict future hidden-state targets in looped middle layers, not just next-token logits. This adds a representation-level learning signal that complements CE.
- **EMA teacher + delayed activation**: teacher targets come from an EMA trajectory and are activated later in training (step/wallclock-gated) so the model first learns a stable token model before adding latent prediction pressure.
- **Gated residual integration of predicted states**: predicted latent states are not hard-swapped; they are mixed back through a learned gate, allowing the model to control when latent prediction should influence the forward path.
- **Coupled CE + distill + JEPA objective**: token CE remains the anchor objective while distill/JEPA terms are weighted additions, preserving strong language modeling behavior and avoiding pure-representation drift.
- **Roundtrip-aware implementation**: all of this is implemented in a way that still supports post-training GPTQ export and on-budget submission artifacts.

Operational implementation details (stability/perf):
- Cadenced JEPA application (`JPCR_APPLY_EVERY`) to control overhead.
- DDP-safe conditional branches with explicit graph retention for conditional params.
- Compile-phase control (`TORCH_COMPILE_DYNAMIC`) for phase-switch workloads.

Dataset/tokenizer requirement:
- This package expects an **SP8192 exported dataset** at:
  - `./sp8192_data/datasets/fineweb10B_sp8192`
- And uses tokenizer assets in this folder by default:
  - `./fineweb_8192_bpe.model`
  - `./fineweb_8192_bpe.vocab`
- Build the dataset with:
  - `bash ./setup_sp8192_data.sh`

Command (track-relevant params):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=sp8192_bpe_submission_8gpu_20260428_133231 \
DATA_PATH=./sp8192_data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
NUM_LAYERS=9 \
MODEL_DIM=448 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
USE_SWIGLU=1 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE_FRAC=0.5 \
DISTILL_ENABLED=0 \
JPCR_ENABLED=0 \
QUANT_SCHEME=int8 \
COMPRESSOR=zstd \
GPTQ=1 GPTQ_NSAMPLES=128 GPTQ_BLOCKSIZE=128 GPTQ_PERCDAMP=0.01 \
torchrun --standalone --nproc_per_node=8 ./train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `11771/20000` steps due to wallclock cap.
- Pre-quant eval at stop: `val_loss:3.0472`, `val_bpb:1.1797`
- Post-quant roundtrip eval: `val_loss:3.06730555`, `val_bpb:1.18746481`
- Exact printed metric: `final_int8_zstd_roundtrip_exact val_bpb:1.18746481`
- Train time: `600085ms` (`step_avg:50.98ms`)
- Code size: `218848 bytes`
- Serialized model int8+zstd: `15062425 bytes`
- Total submission size int8+zstd: `15281273 bytes` (headroom `718727`)

Included files:
- `train_gpt.py` (code snapshot used for the run package)
- `train.log` (exact run log)
- `submission.json` (metadata)
- `reqs.txt` (dependencies)
- `fineweb_8192_bpe.model` and `fineweb_8192_bpe.vocab` (tokenizer assets)
