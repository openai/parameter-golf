# OpenAI Parameter Golf: Challenge Reference

Last updated: 2026-03-18
Repo snapshot reviewed: `openai/parameter-golf` local clone at `/Users/kevin/Code/ParameterGolf_OAI`

## Purpose of this file

This file is our durable memory for the challenge itself.
It is not the experiment journal.
It should answer, in one place, what the challenge is, why it exists, what counts as winning, what is being measured, what the hard constraints are, how submissions work, and what the current repo/code implies beyond the README text.

## Primary sources reviewed

- Official challenge page: <https://openai.com/index/parameter-golf/>
- Official repo: <https://github.com/openai/parameter-golf>
- Local README: [`/Users/kevin/Code/ParameterGolf_OAI/README.md`](/Users/kevin/Code/ParameterGolf_OAI/README.md)
- Data workflow docs: [`/Users/kevin/Code/ParameterGolf_OAI/data/README.md`](/Users/kevin/Code/ParameterGolf_OAI/data/README.md)
- Dataset/tokenizer manifest: [`/Users/kevin/Code/ParameterGolf_OAI/data/manifest.json`](/Users/kevin/Code/ParameterGolf_OAI/data/manifest.json)
- Baseline trainer: [`/Users/kevin/Code/ParameterGolf_OAI/train_gpt.py`](/Users/kevin/Code/ParameterGolf_OAI/train_gpt.py)
- Local iteration trainer: [`/Users/kevin/Code/ParameterGolf_OAI/train_gpt_mlx.py`](/Users/kevin/Code/ParameterGolf_OAI/train_gpt_mlx.py)
- Existing leaderboard records:
  - [`/Users/kevin/Code/ParameterGolf_OAI/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`](/Users/kevin/Code/ParameterGolf_OAI/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md)
  - [`/Users/kevin/Code/ParameterGolf_OAI/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`](/Users/kevin/Code/ParameterGolf_OAI/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md)

## One-paragraph summary

OpenAI Parameter Golf is an open model-craft competition where the goal is to train the best possible language model under an unusually tight artifact-size constraint: the total submission artifact must fit within `16,000,000` bytes, while record-track training must reproducibly finish within `10` minutes on `8xH100` GPUs. The score is not plain tokenizer loss; it is tokenizer-agnostic compression quality on a fixed FineWeb validation set, measured as `val_bpb` (bits per byte), where lower is better. In practice, this means the challenge rewards the entire stack at once: architecture, tokenizer, training recipe, systems efficiency, serialization/compression friendliness, and post-quantization robustness.

## What the challenge is really optimizing

The official framing is: given a fixed parameter/artifact budget, get the lowest held-out loss possible.

Important nuance:

- The public score is `val_bpb`, not raw `val_loss`.
- `val_bpb` is intended to be tokenizer-agnostic.
- The artifact limit counts model bytes and code bytes together.
- Leaderboard eligibility adds a compute budget: reproducible training under `10` minutes on `8xH100`.
- Non-record submissions can still matter if they are interesting, novel, reproducible, and stay inside the size limit.

This is best thought of as constrained model-design optimization under multiple bottlenecks:

- Capacity bottleneck: extremely small final artifact.
- Compute bottleneck: record track limited to 10 minutes on 8 H100s.
- Metric bottleneck: tokenizer changes only help if they improve byte-level compression on the held-out set.
- Engineering bottleneck: the final submitted script has to run cleanly and be reproducible.

## Why OpenAI is running it

There are two explicit motives in the official materials.

1. Research motive
OpenAI wants to see creative model-design ideas under extreme constraints, similar in spirit to elite optimization competitions.

2. Talent motive
OpenAI says the challenge is designed to surface exceptional researchers and engineers they may want to hire. The official page and repo both connect participation with recruiting visibility, especially for early-career researchers, undergrads, recent grads, Olympiad-level competitors, and strong technical builders.

This matters strategically:

- A good submission is not only a better score.
- A strong submission is also a signal of taste, rigor, reproducibility, and ability to reason under constraints.
- Clear writeups and clean experimental discipline are likely part of the meta-game.

## Challenge dates and support

Snapshot from the official materials reviewed on 2026-03-18:

- Challenge window: `2026-03-18` through `2026-04-30`
- Compute support: OpenAI says it is sponsoring `$1,000,000` in compute credits
- Compute-credit path: Runpod partnership and request form on the official page
- Community/support path: OpenAI Discord, especially Parameter Golf channels
- Participant form: optional, used for attribution and recruiting outreach

## What counts as winning

Practical answer:

- Be the top verified leaderboard entry by the end of the challenge.

Operational answer:

- Achieve the lowest accepted `val_bpb` on the record track.
- Stay under the artifact limit.
- Reproduce in under 10 minutes on `8xH100`.
- Beat the previous SOTA by enough margin to clear the submission threshold.

Formal record-track threshold from the README:

1. A new SOTA must beat the existing SOTA by at least `0.005` nats.
2. Because of run-to-run variance, the submission must include enough logs to show `p < 0.01` for that `0.005`-nat improvement.
3. If the tokenizer or dataset changes, the submitter must prove the `val_bpb` computation is correct.
4. The run must reproducibly complete in under `10` minutes on `8xH100`.

Important distinction:

- The README expresses the improvement threshold in `nats` (`val_loss` space), while the public leaderboard score is shown as `val_bpb`.
- In other words, leaderboard bragging is by `val_bpb`, but record acceptance still explicitly talks about statistical improvement in the loss metric as well.

## Hard constraints that should always be in our head

### 1. Artifact size cap

The cap is:

- `16,000,000` bytes total
- Decimal MB, not MiB

The README FAQ explicitly says:

- Counted artifact = code bytes + compressed model bytes
- No external downloads, training-dataset access, or network calls during evaluation
- The artifact must be self-contained and reproducible
- All counted code should live in `train_gpt.py`

Implication:

- Compression is part of the model design problem, not a post-processing afterthought.
- If a parameter improves training loss but bloats the compressed artifact or breaks roundtrip quality, it may still be a bad move.

### 2. Record-track compute cap

For the main leaderboard:

- Training must reproducibly finish in under `600` seconds on `8xH100`

Important repo/code nuance:

- The baseline script measures a timed training window, not “everything that happens from process start to process exit”.
- Warmup/compile priming happens before the timed loop.
- Periodic validation is outside the timed training accumulator.
- Final post-quant roundtrip validation also happens after timed training stops.

That is current code behavior in `train_gpt.py`, not a general law of physics.
If we rely on similar behavior later, we should document it clearly in every serious run.

### 3. Evaluation restrictions

Official FAQ highlights:

- Evaluation cannot take more than 10 minutes on `8xH100`
- Evaluation sequence length is otherwise flexible
- Training data may not be accessed during evaluation unless those bits are included in the artifact budget

### 4. No “spirit-of-the-challenge” abuse

The README explicitly keeps discretion around external compute abuse.
Reasonable hyperparameter tuning is fine.
Extreme brute-force or clearly unfair external search can be rejected even if it is not banned by a single crisp rule.

## Official submission process

Record and non-record submissions are made by pull request.

The repo says a valid submission should add a new folder to the appropriate `records` subfolder and include:

1. `README.md`
2. `submission.json`
3. a train log
4. `train_gpt.py` and any other dependencies required to run the submission

The README also warns:

- The script must successfully compile and run from inside the records folder.
- Broken scripts will not be accepted.

The official challenge page adds another practical detail:

- Once a PR is approved and merged, the leaderboard updates automatically.

## Tracks that currently matter

### Main record track

Intent:

- Beat the public SOTA under both constraints: artifact size and 10-minute training budget.

Local path in repo:

- `records/track_10min_16mb/`

### Non-record 16MB track

Intent:

- Submit interesting approaches even if they do not beat SOTA or are still exploratory.

### Unlimited-compute non-record track

Intent:

- Explore the frontier of parameter-limited performance even when not respecting the 10-minute training cap.

Local path in repo:

- `records/track_non_record_16mb/`

Strategic implication:

- Non-record runs are useful for discovering promising ideas before compressing them back into the 10-minute record-track regime.

## Current official baseline state in this repo snapshot

### Leaderboard baseline

Current record snapshot in the local repo:

- Run: `Naive Baseline`
- Track: main `10min_16mb`
- Score: `1.22436570 val_bpb` after final int8+zlib roundtrip
- Date shown in repo: `2026-03-18`

### Notable unlimited-compute reference

- Run: `4-Hour Baseline`
- Track: non-record unlimited compute
- Score: `1.20737944 val_bpb` after final int8+zlib roundtrip
- Pre-quant stop metric: `1.1749 val_bpb`
- Date shown in repo: `2026-03-18`

The gap between pre-quant and post-quant is already a critical lesson:

- Better raw trained weights are not enough.
- Compression and quantization robustness are part of the competitive objective.

## Dataset and tokenizer facts from the local manifest

These are concrete values from [`data/manifest.json`](/Users/kevin/Code/ParameterGolf_OAI/data/manifest.json):

- Dataset version: `10B`
- Total documents: `15,368,808`
- Validation documents: `50,000`
- Shuffle seed: `1337`
- Shard size: `100,000,000` tokens
- Default exported tokenizer in this snapshot: `sp1024`
- Dataset directory name: `fineweb10B_sp1024`
- Training shards available: `195`
- Validation shards available: `1`
- Total tokens: `19,535,223,186`
- Validation tokens: `62,021,846`
- Training tokens: `19,473,201,340`

Tokenizer facts from the same manifest:

- Tokenizer name: `sp_bpe_1024`
- Kind: `sentencepiece_bpe`
- Vocab size: `1024`
- Model file: `data/tokenizers/fineweb_1024_bpe.model`
- Vocab file: `data/tokenizers/fineweb_1024_bpe.vocab`
- Source spec says the tokenizer was trained on `5,000,000` docs

Practical implication:

- The published dataset/tokenizer pair is frozen enough to compare against the baseline.
- Any tokenizer change can move both the model capacity tradeoff and the scoring surface.
- Tokenizer work is allowed, but proof burden becomes higher.

## Important local workspace reality

The manifest describes the full published export, but the current local working tree does not contain the full training set yet.

Verified locally on 2026-03-18:

- present training shards in `data/datasets/fineweb10B_sp1024/`: `1`
- present validation shards in `data/datasets/fineweb10B_sp1024/`: `1`
- local train shard file: `fineweb_train_000000.bin`
- local validation shard file: `fineweb_val_000000.bin`

Meaning:

- This repository is currently in a smoke-test-friendly state, not a full-baseline-ready downloaded-data state.
- We can understand the challenge and test plumbing locally right now.
- We cannot assume we already have the full train prefix used by the published baseline unless we download more shards first.
- Any future experiment log must say whether it used:
  - the tiny checked-in local subset
  - a larger downloaded prefix such as 10 or 80 train shards
  - a custom rebuilt export

## How data is provided and why reproducibility matters

The repo ships a manifest-driven downloader:

- Script: [`data/cached_challenge_fineweb.py`](/Users/kevin/Code/ParameterGolf_OAI/data/cached_challenge_fineweb.py)
- Default remote dataset repo: `willdepueoai/parameter-golf`
- Default remote root prefix: `datasets`

Default behavior:

- Download the full validation split
- Download `80` train shards by default
- That corresponds to the first `8B` training tokens of the published frozen export

Why this matters:

- Training on the first `N` shards means training on a prefix of the same shuffled export, which keeps comparisons aligned.
- The downloader can optionally fetch `docs_selected.jsonl` and the source sidecar for tokenizer rebuilding or shard re-export.
- The docs sidecar is the reproducibility anchor for exact-document reconstruction.

## What metric is actually optimized and reported

There are two metrics in the training scripts:

1. `val_loss`
- Standard token cross-entropy in natural log units

2. `val_bpb`
- Bits per byte
- This is the public challenge score
- Lower is better

In `train_gpt.py`, `val_bpb` is computed as:

- `bits_per_token = val_loss / ln(2)`
- `tokens_per_byte = total_tokens / total_bytes`
- `val_bpb = bits_per_token * tokens_per_byte`

Important detail:

- Byte count is not a naive token count conversion.
- The script builds SentencePiece lookup tables and accounts for UTF-8 byte length, boundary tokens, and leading-space behavior.
- This is why tokenizer changes are sensitive and scrutinized.

## Validation-set behavior

The code makes the following challenge-relevant assumptions:

- Validation always uses the full `fineweb_val_*` split.
- That split is described in code/comments as the fixed first-50k-document validation set.
- Validation is sequence-packed to `TRAIN_SEQ_LEN`.
- If the validation split is too short for the chosen sequence length, the script errors.

Practical implication:

- Changing `TRAIN_SEQ_LEN` changes how validation is chunked.
- We should treat seq-length changes as first-class experimental variables, not just training knobs.

## Baseline training behavior hidden in the code

These points do not all jump out from the README, but they matter.

### Timed budget accounting

The current CUDA trainer starts timing after setup and optional warmup.
It accumulates training time during the main loop and pauses the timer around validation.

Meaning in practice:

- Setup time is not counted
- Warmup/compile priming is not counted
- Periodic validation is not counted
- Final quantized roundtrip eval is not counted

This is one of the most important implicit facts in the repo.

### Warmup is “free” in the baseline code path

`WARMUP_STEPS` defaults to `20`.
The script performs warmup steps, then restores the initial model weights and optimizer state before timed training begins.

Meaning:

- Warmup is used to prime compiled paths and kernels, not to advance training.
- It is effectively a systems optimization device built into the default script.
- The MLX path mirrors this idea and resets the token loader after warmup as well, so local smoke comparisons also benefit from this behavior.

### Gradient accumulation is normalized to an 8-GPU reference

The script enforces:

- `WORLD_SIZE` must divide `8`
- `grad_accum_steps = 8 / WORLD_SIZE`

Meaning:

- The code preserves an effective 8-way global-batch convention even when testing on fewer GPUs.
- Single-GPU experiments can emulate the 8-GPU token budget via gradient accumulation.

### Data loading is deterministic and sequential

Training data is read by a token stream that:

- walks shards in sorted order
- consumes contiguous chunks
- wraps around forever
- does not random-sample documents on the fly

Meaning:

- Data order is simple and reproducible.
- If we compare runs, we should assume training order is part of the baseline contract unless we intentionally change it.
- Long runs silently wrap around and start reusing data again once available shards are exhausted.

### The baseline is already optimized for fast kernels

The script enables:

- CUDA
- bf16 autocast
- `torch.compile`
- Flash SDP attention
- TF32 math on CUDA backends

Meaning:

- Some “easy” systems wins may already be taken.
- Genuine improvements likely need architecture, optimizer, tokenizer, serialization, or training-dynamics ideas.

## Default baseline configuration in `train_gpt.py`

Current defaults:

- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- `TIE_EMBEDDINGS=1`
- `TRAIN_SEQ_LEN=1024`
- `TRAIN_BATCH_TOKENS=524288`
- `ITERATIONS=20000`
- `WARMUP_STEPS=20`
- `WARMDOWN_ITERS=1200`
- `MAX_WALLCLOCK_SECONDS=600`

Relevant optimizer defaults:

- Embedding/head/scalar groups use Adam
- Matrix-shaped transformer parameters use Muon
- `TIED_EMBED_LR=0.05`
- `EMBED_LR=0.6`
- `HEAD_LR=0.008`
- `MATRIX_LR=0.04`
- `SCALAR_LR=0.04`

This matters because any experiment should be described as a delta from this baseline, not as an isolated setting dump.

## Compression and artifact accounting are part of the objective

The baseline trainer does not judge the raw checkpoint alone.

It performs:

1. raw save for debugging (`final_model.pt`)
2. int8 quantization
3. zlib compression to `final_model.int8.ptz`
4. reload and dequantize
5. final validation on the round-tripped weights

The printed line that matters most is:

- `final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...`

This is critical:

- A run can look better before compression and worse after compression.
- “Winning” means excelling after the exact artifact path that counts.

### Current quantization behavior in the baseline code

The default quantizer in `train_gpt.py` does all of the following:

- 2D float tensors: per-row int8 quantization
- vectors/scalars: per-tensor int8 quantization
- some small tensors: kept in float to avoid metadata overhead dominating
- named control tensors: preserved in higher precision
- compression after serialization: `zlib` level `9`

Important constants in the script:

- `INT8_KEEP_FLOAT_MAX_NUMEL = 65536`
- `INT8_CLIP_PERCENTILE = 99.99984`

This means compression-aware modeling is not optional.
If a design produces weights that are hard to quantize or hard to compress, the final score can degrade substantially even if the raw model is better.

## What the MLX script is for

`train_gpt_mlx.py` is the local iteration path for Apple Silicon.

Its purpose is not to win the leaderboard directly.
Its purpose is to let us iterate quickly on ideas, smoke tests, data-path correctness, tokenizer compatibility, and rough directional changes before using remote CUDA hardware.

Important practical note:

- The MLX path preserves the same broad scoring logic and final roundtrip validation concept.
- That makes it useful for local sanity checks, but not a substitute for actual leaderboard-grade CUDA validation.
- In the current checked-in workspace, it is especially useful because the repo already contains enough local data for a smoke run, but not the full published training prefix.

## Practical failure modes to remember

These are not necessarily formal challenge rules, but they are real gotchas in the current code.

1. `TRAIN_BATCH_TOKENS` must remain compatible with the implicit reshape assumptions.
The loader computes local token counts from `TRAIN_BATCH_TOKENS`, `WORLD_SIZE`, `grad_accum_steps`, and `TRAIN_SEQ_LEN`; incompatible values can fail at reshape time.

2. Validation batch sizing has a hard floor.
`VAL_BATCH_SIZE` must provide at least one sequence per rank after dividing by `WORLD_SIZE * grad_accum_steps`.

3. The current tokenizer-agnostic path is only implemented for SentencePiece `.model`.
So “tokenizer-agnostic metric” does not mean “all tokenizer implementations already supported in the code”.

4. The trainer logs the artifact size but does not hard-fail when size exceeds `16,000,000`.
So a run can finish successfully and still be leaderboard-invalid.

5. The top-of-file comment says `train_gpt.py` and `train_gpt_mlx.py` must never exceed `1500` lines, but the current script does not enforce that rule programmatically.

## Existing baseline records and what they already teach us

### Naive Baseline

From the current record folder:

- Main-track roundtrip score: `1.22436570 val_bpb`
- Train time: about `600038ms`
- Step average: about `43.54ms`
- Int8+zlib model bytes: `15,815,847`
- Code bytes: `47,642`
- Total bytes: `15,863,489`

Implications:

- The baseline already uses most of the 16,000,000-byte budget.
- There is little spare room for code growth.
- Architecture/serialization changes may require code-size discipline, not just model-size discipline.

### 4-Hour Baseline

From the non-record reference folder:

- Pre-quant stop score: `1.1749 val_bpb`
- Final roundtrip score: `1.20737944 val_bpb`
- Total bytes: `15,810,161`

Implications:

- Longer training materially helps the raw model.
- But quantization/compression still eats a large fraction of the gain.
- Therefore, two optimization fronts matter:
  - improve trained-model quality
  - reduce roundtrip degradation

## Strategic conclusions for our future work

These are not official rules, but they are strongly implied by the repo and baseline behavior.

1. We should think in terms of final roundtrip score, not raw in-training score.
2. We should track artifact bytes as aggressively as we track loss.
3. We should separate “good idea in unlimited compute” from “record-track-feasible in 10 minutes”.
4. We should treat tokenizer changes as high-upside but high-proof-burden.
5. We should be careful with code growth because code bytes count too.
6. We should document whether a run is apples-to-apples against the main baseline:
   - same dataset export?
   - same tokenizer?
   - same 10-minute rule?
   - same post-quant metric?
7. We should expect the strongest ideas to combine ML and systems effects instead of only tweaking one hyperparameter.

## Questions we should keep asking during experiments

Before we trust any result:

- Did the score improve after the final int8+zlib roundtrip, not only before?
- Did total artifact bytes stay under `16,000,000`?
- Was the run measured under the correct track assumptions?
- Was the dataset/tokenizer pair exactly the one we think it was?
- Is the comparison fair to the previous run?
- Is the gain large enough to survive variance?
- Is the idea likely to survive the 10-minute `8xH100` record-track requirement?

## Things that are easy to forget but matter

- Lower `val_bpb` is better.
- The artifact cap is decimal bytes, not MiB.
- The baseline code size already uses about `47 KB`.
- Validation is full fixed-val, not a tiny sample.
- The leaderboard score shown publicly is the post-quant roundtrip score.
- Record improvement threshold is phrased in `nats`, not directly in `val_bpb`.
- Tokenizer changes are allowed but scrutinized.
- Non-record submissions are acceptable and strategically useful.
- The repo explicitly wants weird, out-of-the-box ideas, not only polished near-SOTA entries.

## Update rule for this file

Update this file whenever one of these changes:

- official rules or dates
- leaderboard baseline
- submission requirements
- artifact accounting interpretation
- metric calculation understanding
- dataset/tokenizer reproducibility understanding
- any major code-path fact in the baseline trainer that affects fair comparison

Do not turn this file into an experiment diary.
All run-by-run observations belong in the experiment journal.
