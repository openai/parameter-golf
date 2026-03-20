This submission packages the strongest qscale PR135 variant I found: a 9x512 SP-1024 model with qscale-compressed mixed int6 export, `MLP_HIDDEN=1376`, full bigram hash embedding, no fp16 passthrough, and stride-16 sliding-window evaluation.

The best packaged run is a RunPod `8x H100 SXM` rerun at **`1.16134475` val_bpb** with an actual packaged size of **`15,884,596` bytes**. The model payload itself is `15,816,799` bytes; the packaged code is `67,797` bytes across `train_gpt.py` and `pr135_base_train_gpt.py`.

## Key Idea

This is a direct descendant of the PR135 family, but the export path is tightened further:

- Keep the PR135 training recipe and architecture family.
- Replace the mixed-int6 export's large per-row scale vectors with quantized log-space uint8 `qscale` metadata.
- Keep the larger `1376`-hidden MLP and full `BIGRAM_DIM=128` / `BIGRAM_VOCAB_SIZE=4096`.
- Remove fp16 passthrough tensors entirely.
- Use stride-16 sliding-window eval so each scored token sees near-max context.

The training recipe itself is unchanged across the reruns below; only machine/provider changes.

## Best Run

Best packaged run from `train.log`:

- Post-quant exact sliding-window metric: `final_int6_sliding_window_exact val_loss:1.96087605 val_bpb:1.16134475`
- Pre-quant stop metric: `val_loss:1.9838 val_bpb:1.1749`
- Timed training stop: `7526/20000` at `600060ms`
- Sliding eval time: `532114ms`
- Model artifact: `15816799` bytes
- Packaged code: `67797` bytes
- Packaged total: `15884596` bytes

## Significance

The current public README leaderboard shows a best score of `1.1748`, so the required improvement threshold is `1.1698`.

I reran the exact same config 5 times on `8x H100` hardware:

| Provider | Run ID | Seed | Exact val_bpb |
|---|---|---:|---:|
| Modal H100! | `20260320-074904-candidate-pr135-qscale-mlp1376-bigram128-nofp16-stride16-full` | 1337 | 1.16628107 |
| RunPod H100 SXM | `runpod-verify-20260320b` | 1337 | 1.16138246 |
| RunPod H100 SXM | `runpod-verify-20260320b-seed1338` | 1338 | 1.16238551 |
| RunPod H100 SXM | `runpod-verify-20260320b-seed1339` | 1339 | 1.16775348 |
| RunPod H100 SXM | `runpod-verify-20260320b-seed1337r1` | 1337 | 1.16134475 |

Using a one-sided Student t-test against the required threshold `1.1698`:

- Mean exact `val_bpb`: `1.16382945`
- Sample stddev: `0.00298559`
- One-sided `p`: `0.005530214662640183`

Important caveat:

- This pooled significance estimate mixes one Modal `H100!` rerun with four RunPod `H100 SXM` reruns of the exact same config.
- The RunPod-only subset is still slightly short of the bar at about `p = 0.0116`.
- I am including all logs so maintainers can judge whether the mixed-provider pooling is acceptable. If maintainers require same-provider-only significance, this package should be treated as a strong draft rather than a finalized record claim.

The raw rerun table is included in `significance_runs.tsv`.

## Command

Best run command:

```bash
RUN_ID=runpod-verify-20260320b-seed1337r1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
ITERATIONS=20000 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
TRAIN_BATCH_TOKENS=786432 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_SEQ_LEN=2048 \
EVAL_STRIDE=16 \
MLP_HIDDEN=1376 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
GRAD_CLIP_NORM=0.3 \
WEIGHT_DECAY=0.01 \
BIGRAM_VOCAB_SIZE=4096 \
BIGRAM_DIM=128 \
FP16_KEEP_NAME_PATTERNS= \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py`: qscale wrapper used in the submission folder
- `pr135_base_train_gpt.py`: self-contained local dependency for the wrapper
- `train.log`: best packaged RunPod rerun
- `rerun_modal_h100.log`: same-config Modal `8x H100!` rerun
- `rerun_runpod_seed1337_a.log`: first clean RunPod `8x H100 SXM` rerun
- `rerun_runpod_seed1338.log`: second clean RunPod rerun
- `rerun_runpod_seed1339.log`: third clean RunPod rerun
- `significance_runs.tsv`: compact table of the five exact reruns used above
