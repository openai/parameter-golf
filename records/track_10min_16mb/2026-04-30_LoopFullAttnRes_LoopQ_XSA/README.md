# Leaderboard Submission: LoopFullAttnRes + LoopQ + XSA

**val_bpb: 1.18871** (3-seed mean, std 0.00011, post-int8) | **3.07055 nats** | **~14.24 MB** | 8xH100 SXM, 600s | No TTT

This is a leaderboard-format 10-minute / 16MB submission exploring recurrent weight tying with learned routing over loop/depth residual history. It is not a SOTA claim. The model uses a prelude-core-coda layout, a recurrent middle section, full attention residual mixing, loop-specific learned depth queries, and exclusive self-attention in the recurrent core.

The design question was: how can I benefit the most from weight tying in a recurrent kind of way, but without having the model keep applying the same exact representation? If I process through my layer once, maybe if I put it through again, it would be processed differently. But I do not know that. So the architecture lets repeated representations be weighted by learned parameters, which fits naturally with full attention residuals.

I included more personal design notes and what I learned from this project in `REFLECTIONS.md`.

## Results (8xH100 80GB SXM, 600s, no TTT)

| Seed | Steps | ms/step | Pre-quant BPB | **Post-int8 BPB** | Post-int8 val_loss | Int8 artifact | Counted bytes |
|------|------:|--------:|--------------:|------------------:|-------------------:|--------------:|--------------:|
| 42   | 5,507 | 108.96 | 1.1832 | **1.18875284** | 3.07067205 | 14,179,856 | 14,236,084 |
| 1337 | 5,510 | 108.90 | 1.1830 | **1.18878580** | 3.07075719 | 14,182,414 | 14,238,642 |
| 2025 | 5,511 | 108.88 | 1.1829 | **1.18857907** | 3.07022317 | 14,176,192 | 14,232,420 |
| **Mean** | **5,509** | **108.91** | **1.18303** | **1.18870590** | **3.07055080** | **14,179,487** | **14,235,715** |
| **Std** | 2.1 | 0.04 | 0.00015 | **0.00011** | 0.00029 | 3,127 | 3,127 |

The pre-quant final validation mean was `1.18303` BPB / `3.05590` nats. The table's headline score uses the final exact int8 zlib roundtrip validation, matching the score-bearing artifact path printed by the script.

## Architecture

The model is `loop_fullattnres_loopq_xsa`.

The core layout is:

- **Prelude**: two non-recurrent setup blocks.
- **Core**: two shared recurrent blocks, run for three loop passes.
- **Coda**: two readout blocks after recurrence.

The effective depth is therefore ten block applications, but the middle computation is weight-tied through recurrence.

The "loopq" part is the learned query over depth/loop history. Each attention-residual mixer keeps prior sources and learns how to weight them before the attention and MLP sublayers. In full-attnres mode, those sources include the initial embedding state and the accumulated attention/MLP residual outputs from earlier block and loop positions.

Exclusive self-attention is used in the recurrent core. The motivation was that a recurrent architecture can waste capacity repeatedly reinforcing the token's own representation. XSA removes the self-aligned component from the attention output, encouraging recurrent passes to use more cross-token signal instead of repeatedly self-copying.

## Rule compliance

- **Artifact <= 16,000,000 bytes decimal**: max counted bytes here are 14,238,642 (`train_gpt.py` bytes + compressed int8 model bytes).
- **10-minute track shape**: logs stop on the 600-second wallclock cap at `600.020s` to `600.034s` logged train time, with final validation printed immediately afterward.
- **8xH100 run**: all logs were produced with `torchrun --standalone --nproc_per_node=8`.
- **No validation-data training / no TTT**: the run performs ordinary final validation and exact int8 roundtrip validation only.
- **Tokenizer/data**: canonical SP8192 FineWeb data and tokenizer; no tokenizer or dataset changes. A copy of the canonical tokenizer is included in this folder for reproducibility.
- **Self-contained code**: the model, optimizer, validation, and int8 zlib serialization path are in `train_gpt.py`. The submitted script is stripped to the run-used looped/full-AttnRes/XSA architecture and removes alternate architecture branches that were not active in this run.

## Reproducibility

Run from this folder in the standard Parameter Golf environment after staging the canonical SP8192 data in the repo-level `data/` directory.

The run used the canonical Hugging Face data export:

```bash
huggingface-cli download LightSpeedUp/parameter-golf-data \
  --include "fineweb_sp8192/*" \
  --local-dir /workspace/data
```

That export provides the tokenizer copied here as `tokenizers/fineweb_8192_bpe.model`. The data shards should be staged at `data/datasets/fineweb10B_sp8192/` at the repository root.

```bash
for SEED in 42 1337 2025; do
  DATA_PATH=../../../data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe.model \
  VOCAB_SIZE=8192 \
  TRAIN_SEQ_LEN=2048 \
  TRAIN_BATCH_TOKENS=524288 \
  VAL_BATCH_SIZE=1048576 \
  GRAD_ACCUM_STEPS=8 \
  ITERATIONS=1000000 \
  MAX_WALLCLOCK_SECONDS=600 \
  WARMUP_STEPS=20 \
  VAL_LOSS_EVERY=0 \
  TORCH_COMPILE=1 \
  TORCH_COMPILE_FULLGRAPH=0 \
  SEED=$SEED \
  RUN_ID=loopres-xsa-seed${SEED} \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > train_seed${SEED}.log 2>&1
done
```

The original run used commit `1f0a0c343bd4fc91f0aa2420287271e1fda839f7` as the reference checkout and ran with PyTorch CUDA on 8x NVIDIA H100 80GB SXM.

## Reflections

The longer design notes are included in `REFLECTIONS.md`. In short, the architecture came from asking how to get more benefit from weight tying and recurrence without just applying the same representation repeatedly. Full attention residuals gave the recurrent core a way to learn how much to use earlier loop/depth states, while XSA was added because the recurrent middle seemed especially likely to waste capacity on self-copying.

I also found that prelude and coda processing helped around the recurrent core. The PARCAE terminology was a useful way to describe that structure after the fact: a prelude in front, looped computation in the middle, and a coda at the end.

## Included files

- `train_gpt.py` - training script and architecture implementation.
- `tokenizers/fineweb_8192_bpe.model` - canonical SP8192 SentencePiece tokenizer from `LightSpeedUp/parameter-golf-data`.
- `submission.json` - structured metadata.
- `README.md` - this file.
- `REFLECTIONS.md` - design reflections.
- `train_seed42.log` - seed 42 run log.
- `train_seed1337.log` - seed 1337 run log.
- `train_seed2025.log` - seed 2025 run log.
