## Non-record: HelixRecur v2

HelixRecur v2 is the active non-record recurrence champion from this branch. It keeps the HelixRecur v1 shared-depth recurrence intact and adds only a tiny virtual-depth conditioning table so repeated passes can recover a small amount of depth-specific behavior without giving up the recurrence byte win.

This is an exploratory non-record submission, not a record claim. The longer pass remained well behind the accepted SOTA, but the variant is still worth preserving because it materially improved the recurrence line without giving up its compact artifact story.

### Exact architecture summary

- Base lineage: donor -> HelixRecur v1 -> HelixRecur v2
- Tokenizer and dataset: unchanged `sp1024` setup
- Core transformer shape:
  - `11` virtual layers
  - `6` shared recurrent blocks
  - recurrence schedule `0,1,2,3,4,5,4,3,2,1,0`
  - model dim `512`
  - `8` attention heads
  - `4` KV heads
  - MLP multiplier `3.0`
- Preserved donor features:
  - tied embeddings
  - logit softcap `30.0`
  - BigramHash with `2048` buckets and `128`-dim embedding
  - SmearGate local-feature mixer
  - partial RoPE with `16` rotary dims
  - shared value embeddings with `VE_DIM=128` on layers `9,10`
  - XSA on the last `4` layers
  - LN scale path
  - donor optimizer, quantization, compression, and eval path
- v2-specific addition:
  - exactly `44` trainable parameters
  - an `11 x 4` virtual-depth conditioning table
  - per-virtual-pass modulation of existing scalar pathways only:
    - LN scale multiplier
    - attention output scale multiplier
    - MLP output scale multiplier
    - attention `q_gain` multiplier
  - each multiplier is bounded as `1 + 0.05 * tanh(param)`

### Why this is DNA-inspired in engineering terms

- The model reuses a small shared block set across an ordered pass schedule, analogous to reusing a compact genetic program across repeated developmental stages rather than storing a fully separate block for every depth.
- The tiny virtual-depth table acts like a minimal gene-expression control sheet: it does not add new heavy pathways, it only modulates existing scalar controls at each virtual depth.
- BigramHash plus SmearGate provides a cheap motif-sensitive local branch, which fits the intended "sequence motifs + compact regulatory controls" engineering direction.

### Recorded metrics

Quick comparison used for judgment, `1xH100`, `SEED=1337`, `MAX_WALLCLOCK_SECONDS=180`, `EVAL_SEQ_LEN=64`:

| Model | val_loss | val_bpb | compressed bytes | total bytes | step_avg |
|---|---:|---:|---:|---:|---:|
| Donor quick | `7.55493163` | `4.47445606` | `5,019,273` | `5,086,876` | `668.25ms` |
| HelixRecur v1 quick | `7.85509273` | `4.65222837` | `3,081,539` | `3,150,613` | `655.32ms` |
| HelixRecur v2 quick | `7.54165596` | `4.46659346` | `3,042,658` | `3,113,435` | `675.97ms` |

Longer non-record pass, `1xH100`, `SEED=1337`, `MAX_WALLCLOCK_SECONDS=600`, `EVAL_SEQ_LEN=64`:

| Model | val_loss | val_bpb | compressed bytes | total bytes | train stop | step_avg |
|---|---:|---:|---:|---:|---:|---:|
| HelixRecur v2 long | `4.63764717` | `2.74667588` | `4,224,324` | `4,295,101` | `600.504s` | `676.24ms` |

### Byte profile

- v2 code bytes: `70,777`
- v2 parameter count: `15,187,040`
- v2 added parameters vs v1: `+44`
- v2 quick compressed bytes: `3,042,658`
- v2 quick total bytes: `3,113,435`
- v2 long compressed bytes: `4,224,324`
- v2 long total bytes: `4,295,101`
- v1 quick total bytes: `3,150,613`
- donor reproduced artifact for reference: compressed `16,073,037`, total `16,140,640`

### Exact commands already used

Compile sanity:

```bash
python -m py_compile records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py
```

Train smoke:

```bash
env RUN_ID=helixrecur2-train-smoke SEED=1337 MAX_WALLCLOCK_SECONDS=45 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py > helixrecur2_train_smoke.out 2>&1
```

Eval smoke:

```bash
env RUN_ID=helixrecur2-eval-smoke SEED=1337 MAX_WALLCLOCK_SECONDS=1 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py > helixrecur2_eval_smoke.out 2>&1
```

Initial quick comparison attempt:

```bash
env RUN_ID=helixrecur2-quickcmp SEED=1337 MAX_WALLCLOCK_SECONDS=180 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py > helixrecur2_quickcmp.out 2>&1
```

Fair solo quick comparison used for judgment:

```bash
env RUN_ID=helixrecur2-quickcmp-solo SEED=1337 MAX_WALLCLOCK_SECONDS=180 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py > helixrecur2_quickcmp_solo.out 2>&1
```

Longer non-record pass:

```bash
env RUN_ID=helixrecur2-long SEED=1337 MAX_WALLCLOCK_SECONDS=600 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py > helixrecur2_long.out 2>&1
```

### Known limitations

- This is not record-competitive. The longer pass result `2.74667588 val_bpb` is far from the accepted SOTA gate.
- The evidence is single-seed and non-record only.
- The gains are local to this recurrence line; they do not show that recurrence alone is a winning final submission direction.
- The quick proxy improved, but longer-pass behavior is still fragile enough that later micro-variants did not replace v2.

### Next direction

Keep the shared-depth recurrence and byte discipline, but move the next experiment toward gene-coded low-rank specialization: tiny learned depth-conditioned low-rank adapters that preserve the compact shared-block base while giving each virtual pass a more explicit specialization channel.
