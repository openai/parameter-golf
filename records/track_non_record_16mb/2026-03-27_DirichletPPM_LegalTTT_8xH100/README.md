# Non-Record Submission: Dirichlet PPM + Legal TTT on 8xH100

This record captures the canonical successful `sota_plus_ppm_dirichlet` full run on the stabilized frontier trainer path.

It is a **non-record** submission because the run finished legally under the 16,000,000-byte artifact cap, but the full train-plus-eval process exceeded the 10-minute wall-clock limit once exact sliding-window evaluation and legal score-first TTT were included.

## Canonical Run

- Run: `h100_ppm_dirichlet_full_run_8xh100`
- Run dir: `/workspace/parameter-golf/research/results/runs/20260327_065341_h100_ppm_dirichlet_full_run_8xh100`
- Commit: `c3a23b4c2c33e78e51229e0c9ff6bc6f2c6ab945`
- Preset: `sota_plus_ppm_dirichlet`
- Scale: `full_run`
- GPU profile: `8xh100`
- World size: `8`
- Seed: `1337`
- Completion: `completed`
- Legality: `legal`

## Official Submission Metric

- `legal_ttt_exact val_loss = 0.62355877`
- `legal_ttt_exact val_bpb = 0.36930761`

Other exact evals from the same successful run:

- `final_int6_roundtrip_exact val_loss = 2.30528416`
- `final_int6_roundtrip_exact val_bpb = 1.36531913`
- `final_int6_sliding_window_exact val_loss = 0.62378916`
- `final_int6_sliding_window_exact val_bpb = 0.36944405`

Training-best validation from the same run:

- `val_loss = 2.0886360661952943`
- `val_bpb = 1.2370079255856536`

The training-best validation checkpoint metric is not the official submission metric.

## Why This Is Non-Record

- `train_time_seconds = 600.261`
- `wall_clock_seconds = 1831.174`
- `submission_readiness.wall_clock_constraint_appears_satisfied = false`

Training respected the configured 600s cap, but total end-to-end wall clock still exceeded 10 minutes because post-train exact sliding-window evaluation and legal TTT remained expensive. This is therefore a completed legal artifact, but not a 10-minute end-to-end leaderboard submission.

## Technical Change Validated

This run validates the distributed exact-eval path for cache-enabled post-train evaluation on a real 8xH100 full run.

Observed distributed exact-eval breadcrumbs:

- `sliding_eval: distributed_cache_shards=1 world_size=8 chunk_tokens=32768 stride=64`
- `ttt_sliding:distributed_cache_shards world_size=8 chunk_tokens=32768`

Resolved cache config:

```text
causal_cache: mode=ppm order=7 alpha=0.30 gating=dirichlet_posterior
alpha_min=0.10 alpha_max=0.50 entropy_center=3.50 entropy_slope=2.00
order_entropy_centers=7:3.0,6:3.2,5:3.5,4:3.8,3:4.2,2:4.5
posterior_strength=4.00 mixing=dirichlet count_smoothing=4.00
```

Legality summary:

- Posterior predictive backoff uses only previously committed counts plus the current score-step model probability.
- No future-token signal or target-aware chooser is introduced by the recursive update.
- Cache updates and TTT remain strictly score-first.

## Artifact / Budget

- Exported model bytes: `10044244`
- Code bytes measured in the successful run: `132164`
- Artifact bytes measured: `10176408`
- Remaining headroom to 16 MB: `5823592`
- Byte budget satisfied: `true`

The measured artifact bytes above are the values from the successful run artifacts. The record folder here is a reproducibility snapshot and is not itself the byte-accounted submission artifact.

The earlier post-train serialization printout in `train.log` is an intermediate size line. Use the final measured artifact bytes above as the canonical budget result for this run.

## Reproduction Command

From this folder on a RunPod-style 8xH100 node with the dataset and tokenizer mounted under `/workspace/parameter-golf/data/`:

```bash
cd records/track_non_record_16mb/2026-03-27_DirichletPPM_LegalTTT_8xH100
export PYTHONPATH="$PWD"

OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=h100_ppm_dirichlet_full_run_8xh100 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
ADAM_WD=0.04 \
BIGRAM_DIM=128 \
BIGRAM_VOCAB_SIZE=1536 \
CAUSAL_CACHE_ALPHA=0.30 \
CAUSAL_CACHE_BUCKETS=4194304 \
CAUSAL_CACHE_COUNT_SMOOTHING=4.0 \
CAUSAL_CACHE_MAX_ORDER=7 \
CAUSAL_CACHE_MIN_COUNT=2 \
CAUSAL_CACHE_MIXING=dirichlet \
CAUSAL_CACHE_MODE=ppm \
CHECKPOINT_EVERY=500 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
GATED_ATTENTION=0 \
ITERATIONS=20000 \
LN_SCALE=1 \
MATRIX_LR=0.025 \
MAX_WALLCLOCK_SECONDS=600 \
MLP_MULT=3.0 \
MODEL_DIM=512 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
NUM_LAYERS=11 \
ROPE_DIMS=16 \
ROTARY_FIX=0 \
SCALAR_LR=0.025 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
TIED_EMBED_LR=0.035 \
TIE_EMBEDDINGS=1 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_LOG_EVERY=100 \
TRAIN_SEQ_LEN=2048 \
TTT_BATCH_SEQS=32 \
TTT_CHUNK_TOKENS=32768 \
TTT_ENABLED=1 \
TTT_EPOCHS=3 \
TTT_FREEZE_BLOCKS=0 \
TTT_GRAD_CLIP=1.0 \
TTT_LR=0.002 \
TTT_MOMENTUM=0.9 \
VALUE_RESIDUAL=0 \
VAL_LOSS_EVERY=4000 \
VAL_TOKEN_LIMIT=0 \
VE_DIM=128 \
VE_ENABLED=1 \
VE_LAYERS=9,10 \
VOCAB_SIZE=1024 \
WARMDOWN_ITERS=3500 \
WARMUP_STEPS=20 \
XSA_LAST_N=4 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py` — code snapshot of the control frontier trainer path used for the successful run
- `frontier_cache.py`
- `frontier_checkpoint.py`
- `frontier_eval.py`
- `flash_attn_interface.py`
- `research/submission_metrics.py`
- `requirements.txt`
- `submission.json`
- `train.log`

`train.log` is the successful canonical trainer log extracted from the raw pod session transcript. The broader shell transcript also included setup retries and the earlier failed `h100_ppm_dirichlet_full_run_8xh100_real` attempt; this record keeps the canonical successful run log plus the historical note below.

## Historical Failed Attempt

- `h100_ppm_dirichlet_full_run_8xh100_real` failed with an NCCL timeout after `final_int6_roundtrip_exact`
- it should remain historical only and is not the canonical submission result
