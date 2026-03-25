# Wavelet-Lite PR549 Parallel Muon

**val_bpb: 1.1483** | **15.86 MB** | **8x H100 80GB** | **90.24 ms/step**

This is a **non-record 10min/16MB submission**. It does not beat the current SOTA, but it is under the official cap and materially different from the merged wavelet- and routing-adjacent entries already in the repo.

## Idea

This submission takes the strong PR `#549` Parallel Muon frontier stack and adds one science-flavored architectural change: a tiny causal wavelet-lite mixer inside each residual block.

- The first `16` post-attention activation channels are split into low/high Haar-style bands using the current token and a one-token lagged copy.
- A learned low-band drift scale perturbs only the coarse band before the transform is folded back.
- To stay comfortably under the 16 MB cap, the run uses `BIGRAM_VOCAB_SIZE=1024` and disables TTT in the final budgeted training recipe.

This is a derivative frontier stack, but the added mechanism is architectural rather than a retune.

## Why this is not duplicate work

Nearest prior PRs and exact differences:

1. [PR #549](https://github.com/openai/parameter-golf/pull/549) `LeakyReLU² + Legal Score-First TTT + Parallel Muon`
   This is the closest parent stack. The present submission adds a new causal wavelet mixer inside the model, removes TTT from the final run, and trims the bigram table to fit the byte budget. It is not a rename or pure hyperparameter sweep of `#549`.
2. [PR #211](https://github.com/openai/parameter-golf/pull/211) `WaveletWeightedWidenet`
   That work is a wavelet-branded widen/compress/VQ design. This submission does not widen the network or introduce VQ compression; it changes activation-space token mixing inside the residual stream.
3. [PR #632](https://github.com/openai/parameter-golf/pull/632) `Attention-Residuals`
   That work changes depthwise residual routing over layer history. This submission leaves depth routing alone and injects local multiresolution token mixing inside each block.
4. [PR #507](https://github.com/openai/parameter-golf/pull/507) `11L U-Net + Catalytic + SwiGLU + SW64`
   That work is built around U-Net-style skip structure. This submission adds no U-Net transport or skip gating.
5. [PR #530](https://github.com/openai/parameter-golf/pull/530) `Basis Block Interpolation`
   That work reuses/interpolates depth blocks. This submission does not interpolate parameters across layers; it adds a fixed-form token transform inside each block.

## Final result

Final 8xH100 run:

- Commit synced to pod: `9c0eba6`
- `step:6648/9000 val_bpb=1.1409`
- `DIAGNOSTIC post_ema val_bpb=1.1400`
- `step_avg=90.24 ms/step`
- `peak memory allocated=22015 MiB`

Recovered final artifact:

- `final_model.int6.ptz`: `15,768,240` bytes
- Code size: `91,471` bytes
- Total submission size: `15,859,711` bytes
- Exact saved-artifact roundtrip: `val_bpb=1.14825550`

Why this counts as a solid entry:

- It is under the artifact cap by `140,289` bytes.
- It beats the best existing `1.15015359` / `1.1556` / `1.15744040` local merged results on the fetched `upstream/main` tree.
- Against `upstream/main` fetched on **March 25, 2026**, it would rank **7th** among merged non-null `track_10min_16mb` submission scores.

## Throughput and artifact notes

- Full training used the canonical cached `sp1024` setup on `8x H100` with `MAX_WALLCLOCK_SECONDS=600`.
- The decisive systems trick was not model-side: logs and artifacts stayed on the MO-1 volume, but data and tokenizer were copied onto local `/workspace` NVMe before training.
- The training pod exported the full-precision checkpoint; the int6 artifact and exact roundtrip eval were then recovered from the saved checkpoint on a short-lived 1xH100 helper pod.

## What failed along the way

- The first PR549 wavelet attempt (`8deb532`) hit `post_ema val_bpb=1.1439` but missed the size cap at `16,052,223` bytes and also crashed in the quantized eval path.
- An ephemeral under-cap rerun proved the speed path (`~90.40 ms/step`) but vanished before producing a durable artifact.
- A volume-only rerun was durable but too slow (`126.64 ms/step`) because it trained directly from the network volume.

## Files included here

- `train_gpt.py`: exact training script
- `final_model.int6.ptz`: final saved artifact
- `logs/train.log`: 8xH100 training log
- `logs/roundtrip_eval.log`: exact 1xH100 int6 roundtrip eval log
- `results.tsv`: local experiment ledger snapshot

## Run command

```bash
RUN_ID=pr549_wavelet_bigram1024_nottt_8xh100_nvme_seed1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=1024 \
XSA_LAST_N=4 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
TTT_ENABLED=0 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
QAT_ENABLED=1 \
GATED_ATTENTION=1 \
VALUE_RESIDUAL=1 \
WAVELET_ENABLED=1 \
WAVELET_DIM=16 \
WAVELET_INIT=0.25 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
