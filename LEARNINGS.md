# LEARNINGS 1

Date: 2026-02-19
Repo: `/Users/Williamd/code/openai-parameter-challenge`
Scope: `train_gpt_simple` size-search tooling + shortlist interpretation + remote sweep bring-up on `speedruna2`.

## 1) New local sizing/search tooling for `train_gpt_simple.py`

Added:
- `calc_train_gpt_simple_params.py`
- `search_calc_train_gpt_simple_params.py`

What was learned:
- `train_gpt_simple` needs its own sizing math; `train_gpt.py` formulas are not applicable.
- Correct param components for `train_gpt_simple` include:
  - embedding, lm_head
  - per-block attention/MLP matrices
  - `skip_weights`, `v_mix`, `resid_mix`
- `RMSNorm` in `train_gpt_simple.py` is parameter-free, so norm params must not be counted.
- Estimator parity was validated against live model construction:
  - `sum(component_params(cfg).values()) == sum(p.numel() for p in GPT(...).parameters())`
  - matched on multiple configs.

CLI/UX learned:
- Defaults in `search_calc_train_gpt_simple_params.py` are under-30MB oriented:
  - `--target-mb 30`
  - `--max-mb 30`
- The `0` column users may notice is usually `delta_params` (when `--target-params` is unset), not `model_dim`.

## 2) Interpreting search results correctly

For fixed `(MODEL_DIM, NUM_LAYERS, MLP_MULT)` in this architecture:
- parameter count and byte count are identical across different `NUM_HEADS`
- only `head_dim = MODEL_DIM / NUM_HEADS` changes

Implication:
- search tables often have repeated rows across head counts
- dedupe/rank by shape `(MODEL_DIM, NUM_LAYERS, MLP_MULT)` first
- choose `NUM_HEADS` second (practical bias: keep `head_dim` in a sane range, e.g. ~64-96)

## 3) Shortlist selected from user-provided candidates

Primary shortlist we chose to run:
- `MODEL_DIM=384 NUM_LAYERS=8 NUM_HEADS=6 MLP_MULT=4`
- `MODEL_DIM=320 NUM_LAYERS=11 NUM_HEADS=4 MLP_MULT=4`
- `MODEL_DIM=480 NUM_LAYERS=5 NUM_HEADS=6 MLP_MULT=4`
- `MODEL_DIM=512 NUM_LAYERS=4 NUM_HEADS=8 MLP_MULT=4`

## 4) Brix operational learnings on `speedruna2`

### Targeting the box
- `brix run --pods speedruna2 ...` failed with "No Pods matched criteria"
- `speedruna2` is a pool name, not pod name
- correct invocation is `brix run --pools speedruna2 -- ...`

### Preflight process checks
- `tmux` existed and had only shell activity.
- no active `train_gpt`/`train_gpt_simple` process before launch.

### Sync learned behavior
- `brix git push speedruna2 --yes` synced local changes and waited for remote checkout.

## 5) Data/tokenizer compatibility learning (critical)

Initial blocker:
- `speedruna2` default training shards at `data/fineweb10B/*.bin` had token ids up to `50256`.
- user-selected runs were `VOCAB_SIZE=1024`.
- this mismatch would fail token LUT/indexing paths.

Verified:
- on `speedruna2`, default shard sample showed `min=0 max=50256`.

Resolution:
- Bootstrapped matched 1024 assets to `speedruna2` under `/tmp`:
  - `/tmp/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model`
  - `/tmp/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/fineweb_train_000001.bin`
  - `/tmp/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/fineweb_val_000000.bin`
- Confirmed bootstrapped shard token range: `min=1 max=1023`.

Source used for bootstrap:
- `speedruna1` had the required `/tmp/matched_10B_docs2m_seed1337/...` assets.
- copied via local hop with `brix scp` (speedruna1 -> local `/tmp` -> speedruna2).

## 6) Sweep orchestration learning

### Launch reliability
- First tmux launch attempt silently failed due quoting/redirection style in `tmux new-session`.
- Reliable pattern:
  - pass `bash` script path directly to tmux command
  - perform logging from inside script (`exec > >(tee -a "$log") 2>&1`)

### Successful sweep session details
- tmux session: `sweep_simple_20260219_233808`
- master log: `/tmp/sweep_simple_20260219_233808.log`
- per-run logs:
  - `/root/code/openai-parameter-challenge/logs/sweep_s2_20260219_233808_d384_l8_h6.txt`
  - `/root/code/openai-parameter-challenge/logs/sweep_s2_20260219_233808_d320_l11_h4.txt`
  - `/root/code/openai-parameter-challenge/logs/sweep_s2_20260219_233808_d480_l5_h6.txt`
  - `/root/code/openai-parameter-challenge/logs/sweep_s2_20260219_233808_d512_l4_h8.txt`

## 7) Sweep results and failure signature

All 4 configs failed (`rc=1`) with the same root signature:
- `RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling cublasCreate(handle)`

Run boundaries from master log:
- `d384_l8_h6`: start `23:38:08Z`, end `23:38:50Z`, `rc=1`
- `d320_l11_h4`: start `23:38:50Z`, end `23:39:32Z`, `rc=1`
- `d480_l5_h6`: start `23:39:32Z`, end `23:40:11Z`, `rc=1`
- `d512_l4_h8`: start `23:40:11Z`, end `23:40:51Z`, `rc=1`

Reported model params at startup:
- `d384_l8_h6`: `14,942,236`
- `d320_l11_h4`: `14,172,199`
- `d480_l5_h6`: `14,807,058`
- `d512_l4_h8`: `13,631,502`

## 8) Practical next-step hypothesis

Given identical failure across all shapes, likely this is environment/runtime-path related, not config-specific.

Most direct next debugging path:
- run 1-GPU smoke (`--nproc_per_node=1`) on one config with same dataset/tokenizer
- if stable on 1 GPU, test 8-GPU with compile disabled/fallback path
- if still failing, isolate cuBLAS init issue from compiled graph path and distributed startup timing.

# LEARNINGS 2 (2026-02-19)

## Scope
This file captures concrete learnings from the tokenizer + dataset + training pipeline work in this repo.

## 1) SentencePiece and 260-vocab constraints
- A 260-vocab SentencePiece model with `byte_fallback` is not feasible with default settings on this corpus.
- The observed error was:
  - `Vocabulary size is smaller than required_chars. 260 vs 345`
- Why this happens:
  - Byte fallback introduces 256 byte pieces (`<0x00>`..`<0xFF>`).
  - SentencePiece still requires additional meta/special/required char pieces.
  - Net required minimum exceeded 260.
- Conclusion:
  - If you want exactly 260 with true byte behavior, use a custom pure-byte tokenizer (not standard SentencePiece BPE fallback).

## 2) Pure-byte tokenizer approach
- Implemented/used a fixed pure-byte tokenizer with mapping:
  - `0:<pad>`, `1:<bos>`, `2:<eos>`, `3:<unk>`, `4..259: raw bytes 0..255`
- Relevant files:
  - `data/pure_byte_tokenizer.py`
  - `data/create_pure_byte_tokenizer.py`
  - `data/fineweb_pure_byte.py`

## 3) Matched dataset export across tokenizers
- Goal: identical underlying docs/split across tokenizer variants to avoid confounded training.
- Export script:
  - `data/export_matched_fineweb_tokenizer_datasets.py`
- Working command:
```bash
python3 data/export_matched_fineweb_tokenizer_datasets.py \
  --version 10B \
  --num_docs 2000000 \
  --num_val_docs 50000 \
  --shuffle_seed 1337 \
  --sp_vocab_sizes 512,1024,2048,4096 \
  --output_root data/matched_10B_docs2m_seed1337
```
- Output contains five datasets from the same docs cache:
  - `fineweb10B_byte260`
  - `fineweb10B_sp512`
  - `fineweb10B_sp1024`
  - `fineweb10B_sp2048`
  - `fineweb10B_sp4096`
- Manifest path:
  - `data/matched_10B_docs2m_seed1337/manifest.json`

## 4) Comparability rules
- Comparable runs require matching:
  - `docs_jsonl` content hash (`docs_sha256`),
  - `num_docs`, `num_val_docs`, `shuffle_seed`,
  - `append_eos`, `shard_size`,
  - tokenizer artifact used.
- Re-export behavior:
  - If existing docs cache is reused, downstream exports are aligned to that cache.
  - If cache is larger than requested `num_docs`, script deterministically truncates to first `N` docs.
  - Rebuilding docs cache (`--rebuild_docs_cache`) can change data sequence and confound comparisons.

## 5) Verification tests
- Added/used:
  - `tests/test_matched_exports.py`
- What it checks:
  - docs cache hash matches manifest,
  - doc counts/splits match across datasets,
  - tokenizer prefix alignment between expected tokenization and exported shards.

## 6) Blobstore upload learnings
- Internal destination pattern from README:
  - `az://oaidatasets2/speedrunkits/...`
- Uploaded matched export to:
  - `az://oaidatasets2/speedrunkits/matched_10B_docs2m_seed1337`
- Final parity check:
  - local files: `176`
  - remote files: `176`
  - missing/extra: `0/0`
- Operational notes:
  - `bbb cptree` overwrites files and can re-copy large trees after interruption.
  - High concurrency (`64`) caused many transient socket timeouts.
  - Lower concurrency plus targeted `cptree` by missing directories was more reliable.

## 7) `train_gpt_simple.py` dataset wiring
- Current script behavior:
  - reads only `DATA_PATH/data/fineweb10B/fineweb_train_*.bin`
  - reads only `DATA_PATH/data/fineweb10B/fineweb_val_*.bin`
- It does **not** consume `TRAIN_FILES` / `VAL_FILES` env vars.
- Practical pattern:
  - symlink selected dataset to `DATA_PATH/data/fineweb10B`.

## 8) 1k vocab training launch used on `speedruna1`
- Pod:
  - `speedruna1-0`
- Repo path on pod:
  - `/root/code/openai-parameter-challenge`
- Effective run command:
```bash
TORCHINDUCTOR_CUDAGRAPHS=0 \
DATA_PATH=/tmp/gpt_simple_fineweb10B_sp1024 \
VOCAB_SIZE=1024 \
RUN_ID=debug_sp1024 \
torchrun --standalone --nproc_per_node=8 train_gpt_simple.py \
2>&1 | tee logs/debug_sp1024.console.log
```
- Observed progress during run:
  - reached at least step `169/3000`
  - observed val line: `step:125/3000 val_loss:3.6437`

## 9) Compression metric (`val_bpb`) in `train_gpt_simple.py`
- Simplified by request:
  - fixed tokenizer path constant:
    - `FIXED_TOKENIZER_PATH=/tmp/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model`
  - `bytes_per_token(input_ids, target_ids)` is now implemented for SentencePiece.
  - Validation now logs `val_bpb`.
- Implementation details:
  - Builds and caches per-device LUTs from the SentencePiece model.
  - Handles leading-space sentencepiece behavior (`▁`) in byte counting.
  - Raises on invalid shape / invalid token range / zero-byte totals.
- Current limitation:
  - This implementation is SentencePiece-specific.
  - Byte-level tokenizer support should be implemented in a separate script/variant.

## 10) Safety/ops notes when launching via Brix
- Pool name vs pod name matters:
  - use pod name for `brix run --pods` (e.g., `speedruna1-0`).
- `tmux` env propagation can fail if env vars are not inlined in the tmux command.
- For visibility, always keep a direct log file and a simple tail command.

## 11) Useful commands
```bash
# Verify upload count
bbb lstree az://oaidatasets2/speedrunkits/matched_10B_docs2m_seed1337 | wc -l

# Tail active run log on speedruna1
brix run --pods speedruna1-0 -- bash -lc 'cd /root/code/openai-parameter-challenge && tail -f logs/debug_sp1024.console.log'

# Check training processes on speedruna1
brix run --pods speedruna1-0 -- bash -lc 'ps -eo pid,etime,cmd | rg "torchrun|train_gpt_simple.py" | rg -v rg'
```
# LEARNINGS_3

Date: 2026-02-19 (UTC)  
Repo: `/Users/Williamd/code/openai-parameter-challenge`  
Target box: `speedruna1` (`speedruna1-0` pod)

## TL;DR

- A working sweep runner for `train_gpt_simple.py` now exists: `sweep_train_gpt_simple.py`.
- Requirements were installed on `speedruna1-0` with `pip install -r requirements.txt`.
- A clean 6-trial sweep completed successfully on 8x H100.
- Best trial from this batch is `lr_mid_1` with `final_val_loss=4.0473` at step `600/600`.

## What Was Implemented

### 1) New sweep script

File: `sweep_train_gpt_simple.py`

Capabilities:
- Runs sequential trials, each with its own `RUN_ID`.
- Waits for active `train_gpt*` jobs to finish before starting (`--wait-for-idle`).
- Launches DDP via `python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt_simple.py`.
- Parses final `val_loss` from `logs/<RUN_ID>.txt`.
- Writes:
  - `results.jsonl` (machine-readable rows)
  - `summary.tsv` (ranked leaderboard)

Defaults used in this run:
- `NUM_SCHEDULED_ITERATIONS=600`
- `WARMUP_STEPS=30`
- `VAL_LOSS_EVERY=100`
- `VAL_TOKENS=1048576`
- `SAVE_CHECKPOINT=0`

### 2) Robustness patch to `train_gpt_simple.py`

File: `train_gpt_simple.py`

Changes:
- Added `ENABLE_VAL_BPB` env flag (default `True`).
- Added runtime fallback so `val_bpb` computation disables itself if tokenizer/bytes accounting fails.
- Guarded against zero-byte-count before computing `tokens_per_byte`.

Why this mattered:
- Some runs failed due tokenizer bytes accounting path dependencies.
- This patch keeps training and `val_loss` evaluation alive even when BPB dependencies are unavailable/misconfigured.

## Operational Learnings (Brix + speedruna1)

- `speedruna1` is a pool; actual pod name is `speedruna1-0`.
- `brix run --pods speedruna1` did not match pods; `--pods speedruna1-0` worked.
- Always preflight active GPU jobs before launching:
  - `nvidia-smi`
  - `pgrep -af "train_gpt|torchrun"`
- Non-disruptive queueing is important; do not interrupt active jobs unless explicitly intended.
- `brix scp` is useful for syncing only touched files when local git tree is dirty.

## Failure Modes Observed and Fixes

### Failure: sweep trials exiting early with return code 1

Observed error signature:
- `FileNotFoundError` / `NotImplementedError` from `bytes_per_token` path expectations.
- Hardcoded tokenizer path: `/tmp/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model`.

Fixes:
- Added runtime fallback in `train_gpt_simple.py` so BPB issues do not crash training.
- Reran sweep with `--set ENABLE_VAL_BPB=0` to isolate hyperparameter tuning from BPB/tokenizer path issues.

## Final Sweep Results (Completed)

Sweep output directory:
- `/Users/Williamd/code/openai-parameter-challenge/logs/sweeps/train_gpt_simple/20260219_202527`

Ranking (`final_val_loss`, lower is better):
1. `lr_mid_1` -> `4.0473`
2. `baseline` -> `4.0754`
3. `lr_low_1` -> `4.0808`
4. `lr_mid_2` -> `4.1257`
5. `lr_low_2` -> `4.1392`
6. `lr_high_1` -> `4.1791`

Best config from this sweep:
- `EMBED_LR=0.50`
- `HEAD_LR=0.0065`
- `MATRIX_LR=0.0018`
- `SCALAR_LR=0.0070`
- `BETA2=0.95`

Runtime profile:
- Each 600-step trial took about 142–145 seconds.
- Final step average was roughly 138 ms/step.

## Repro Command (same pattern used for successful run)

```bash
cd /root/code/openai-parameter-challenge
python3 sweep_train_gpt_simple.py \
  --prefix speedruna1_sweep_r3 \
  --max-trials 6 \
  --nproc-per-node 8 \
  --set ENABLE_VAL_BPB=0 \
  --wait-for-idle \
  --idle-timeout-seconds 10800
```

## Artifact Index

- Ranked summary:
  - `/Users/Williamd/code/openai-parameter-challenge/logs/sweeps/train_gpt_simple/20260219_202527/summary.tsv`
- JSON rows:
  - `/Users/Williamd/code/openai-parameter-challenge/logs/sweeps/train_gpt_simple/20260219_202527/results.jsonl`
- Per-trial console logs:
  - `/Users/Williamd/code/openai-parameter-challenge/logs/sweeps/train_gpt_simple/20260219_202527/*.console.log`
- Per-trial trainer logs:
  - `/Users/Williamd/code/openai-parameter-challenge/logs/speedruna1_sweep_r3_*.txt`

## Practical Next Steps

- Expand around the best region (`lr_mid_1`) with tighter local search.
- Re-enable `ENABLE_VAL_BPB=1` once tokenizer artifact handling is standardized and deterministic.
- Keep sweep runs isolated from unrelated active jobs via `--wait-for-idle`.
# LEARNINGS_4

Date: 2026-02-19
Repo: `/Users/Williamd/code/openai-parameter-challenge`
Primary file: `/Users/Williamd/code/openai-parameter-challenge/train_gpt_simple.py`
Remote test target: `speedruna3-0`

## What was reviewed

Focused review of `train_gpt_simple.py` for:
- correctness bugs
- mistuned training behavior
- silent no-op configuration paths
- startup/runtime pitfalls on current remote environment

## Key issues found

1. `NUM_EXTENSION_ITERATIONS` was effectively a no-op.
- Cause: LR multiplier returned `0.0` for all steps `>= scheduled_iterations`.
- Effect: extension steps ran but did not update parameters.

2. fp32 cast bug after bf16 model construction.
- Cause: all `nn.Linear` modules were immediately cast to fp32 after `.bfloat16()`.
- Effect: unnecessary memory/perf regression versus intended bf16 training.

3. Warmup path had avoidable high overhead.
- Cause: deep-copy of model + optimizer state, then restore after warmup.
- Effect: large memory/time cost for kernel warmup.

4. `weight_decay` hyperparameter was dead config.
- Cause: argument existed but was never passed to optimizers.
- Effect: misleading knob; potential confusion during tuning.

5. Compile stability is environment-dependent on remote.
- Observed on `speedruna3`: `torch.compile` failed at runtime with Triton import mismatch (`triton_key`).
- Note: this was intentionally left unchanged per user direction.

## Changes applied

All changes were made in `train_gpt_simple.py`:

1. Removed dead `weight_decay` hyperparameter from `Hyperparameters`.
2. Removed forced fp32 cast of all linear layers (model now remains bf16 as constructed).
3. Simplified warmup by removing deep-copy snapshot/restore logic.
4. Fixed LR schedule so extension iterations use stage LR instead of forced zero.
5. Kept `torch.compile(...)` behavior unchanged (explicit user request to ignore compile issue for now).

## Verification done

Local checks:
- `python3 -m py_compile train_gpt_simple.py` passed.
- Schedule probe confirmed extension LR is non-zero after scheduled phase.

Remote sync:
- `brix git push speedruna3 -y` succeeded.
- Remote repo commit observed: `bab7d40`.

Remote run attempts on `speedruna3-0`:

1. Compile-enabled smoke run (full shape) failed before training:
- Error: `ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler'`.

2. Compile-disabled smoke run (full shape) reached warmup and OOMed:
- Error: CUDA OOM during warmup backward, attempted allocation ~12.28 GiB.
- This is consistent with very large default token budget in 1-GPU eager mode.

3. Compile-disabled reduced-shape smoke run failed fast on val config:
- Assertion: `VAL_TOKENS must be divisible by VAL_BATCH_SIZE`.

4. Compile-disabled reduced-shape smoke run succeeded end-to-end:
- Env: `NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 MLP_MULT=2 WARMUP_STEPS=2 NUM_SCHEDULED_ITERATIONS=1 NUM_EXTENSION_ITERATIONS=1 VAL_LOSS_EVERY=1 VAL_TOKENS=524288 VAL_BATCH_SIZE=524288`.
- Log file: `/root/code/openai-parameter-challenge/logs/codex_smoke_small2_20260219_192744.txt`.
- Key output:
- `warmup_step:1/2`
- `warmup_step:2/2`
- `step:0/2 val_loss:10.8260`
- `step:1/2 train_loss:10.8260`
- `step:1/2 val_loss:10.8258`
- `step:2/2 train_loss:10.8258`
- `step:2/2 val_loss:10.5471`
- Completed and wrote `final_model.pt`.

## Practical takeaways

1. Extension steps now actually train after the LR fix.
2. bf16 path is no longer accidentally overridden by fp32 linear casts.
3. Warmup logic is lighter and avoids unnecessary state snapshots.
4. On this pod/image, `torch.compile` can still fail due Triton mismatch; compile-disabled smoke is the reliable fallback.
5. For 1-GPU eager smoke, reduce model/token budget aggressively to avoid false OOM blockers.

## Useful commands used

Sync:
```bash
brix git push speedruna3 -y
```

Compile-enabled smoke (expected to fail on current Triton mismatch):
```bash
brix run --pods speedruna3-0 --dir /root/code/openai-parameter-challenge -- /bin/bash -lc '
  CUDA_VISIBLE_DEVICES=0 \
  WARMUP_STEPS=2 \
  NUM_SCHEDULED_ITERATIONS=1 \
  NUM_EXTENSION_ITERATIONS=1 \
  VAL_LOSS_EVERY=1 \
  VAL_TOKENS=65536 \
  VAL_BATCH_SIZE=524288 \
  python -u train_gpt_simple.py
'
```

Compile-disabled successful reduced-shape smoke:
```bash
brix run --pods speedruna3-0 --dir /root/code/openai-parameter-challenge -- /bin/bash -lc '
  TORCHDYNAMO_DISABLE=1 \
  CUDA_VISIBLE_DEVICES=0 \
  NUM_LAYERS=2 \
  MODEL_DIM=128 \
  NUM_HEADS=4 \
  MLP_MULT=2 \
  WARMUP_STEPS=2 \
  NUM_SCHEDULED_ITERATIONS=1 \
  NUM_EXTENSION_ITERATIONS=1 \
  VAL_LOSS_EVERY=1 \
  VAL_TOKENS=524288 \
  VAL_BATCH_SIZE=524288 \
  RUN_ID=codex_smoke_small2_20260219_192744 \
  python -u train_gpt_simple.py
'
```
# Learnings 5

## Scope

This note captures what we validated while figuring out how to run and speed-tune `train_gpt_simple.py` on `1xH100`.

## Key Findings

1. `train_gpt_simple.py` runs as single-process by default.
   - It only enables distributed mode if `RANK` and `WORLD_SIZE` are present in env.
   - For `python3 train_gpt_simple.py`, it falls back to `world_size=1`.
2. You do not need `torchrun` for 1 GPU with `train_gpt_simple.py`.
   - A plain `python3` launch is sufficient.
3. The default shard paths for `train_gpt_simple.py` are:
   - `data/fineweb10B/fineweb_train_*.bin`
   - `data/fineweb10B/fineweb_val_*.bin`
4. The repo docs mention `USE_FLASH_ATTN=0` for 1x runs, but that applies to `train_gpt.py` commands in `README.md` / `SETUP.md`.  
   `train_gpt_simple.py` does not read `USE_FLASH_ATTN`.
5. For remote execution hygiene, use local edits + sync, then run on a single pod with:
   - `brix git push`
   - `brix run --pods <pod> -- bash -lc '...'`

## 1xH100 Launch Pattern (train_gpt_simple)

```bash
CUDA_VISIBLE_DEVICES=0 RUN_ID=simple_1xh100_full python3 train_gpt_simple.py
```

Recommended remote form:

```bash
brix run --pods "$POD" -- bash -lc '
  cd /root/openai-parameter-challenge &&
  CUDA_VISIBLE_DEVICES=0 RUN_ID=simple_1xh100_full \
  python3 train_gpt_simple.py 2>&1 | tee /tmp/train_gpt_simple_1xh100_full.log
'
```

Tail:

```bash
brix run --pods "$POD" -- tail -f /tmp/train_gpt_simple_1xh100_full.log
```

## Practical Speed Knobs (1xH100)

Most impactful without changing core code:

1. `WARMUP_STEPS=0`
   - Skips warmup passes that otherwise consume startup time.
2. Increase `VAL_LOSS_EVERY`
   - Fewer eval interruptions.
3. Decrease `VAL_TOKENS`
   - Cheaper each eval.
4. Keep `USE_AMP=1`
   - Preserves bf16 autocast throughput.
5. Keep `LOGIT_CHUNK_TOKENS=0`
   - Comment in code notes this is materially faster on H100 stacks.

Fast preset example:

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=simple_1xh100_fast \
USE_AMP=1 \
LOGIT_CHUNK_TOKENS=0 \
WARMUP_STEPS=0 \
VAL_LOSS_EVERY=1000 \
VAL_TOKENS=2097152 \
VAL_BATCH_SIZE=524288 \
python3 train_gpt_simple.py
```

Optional extra speed via smaller model:

```bash
NUM_LAYERS=10 MODEL_DIM=640 NUM_HEADS=8 MLP_MULT=3
```

## Failure Modes To Check First

1. Wrong remote repo path (`cd` into wrong directory).
2. Missing shards under `data/fineweb10B/...`.
3. Assuming `train_gpt.py` flags/envs automatically apply to `train_gpt_simple.py` (they do not always).

## Source Anchors Used

- `train_gpt_simple.py`
  - hyperparameter env vars and defaults
  - distributed init behavior
  - warmup/eval/train flow
- `README.md`
  - 1xH100 launch note for `train_gpt.py`
- `SETUP.md`
  - same 1xH100 launch note and setup context
# LEARNINGS_6

## Scope
- Target was making `train_gpt_simple.py` fast and clean with `torch.compile`, aligned with `train_gpt.py` behavior.
- Priority order was:
- compile reliability
- speed on `speedruna2`
- code cleanliness

## Final Design (What We Kept)
- `train_gpt_simple.py` now compiles unconditionally with:
- `torch.compile(raw_model, dynamic=False, fullgraph=True)`
- No compile gating, no fallback-to-eager path, no Triton shim path, no compile `try/except` wrappers.
- Logits are unchunked by default:
- `LOGIT_CHUNK_TOKENS=0`
- This gave materially better throughput on H100 for compiled runs.
- Warmup is now a separate pre-training phase:
- `WARMUP_STEPS` controls warmup count.
- `TIMING_WARMUP_STEPS` is accepted as a fallback alias for compatibility.
- Warmup logs are explicit:
- `warmup_steps:N`
- `warmup_step:i/N`
- Warmup does not appear as `step:X/...` training steps.
- After warmup, model and optimizer are restored to pre-warmup state:
- warmup is compile/kernel priming only
- measured training starts from restored model/optimizer
- training data loader is reset before measured training

## Performance Findings
- Compile warmup overhead is large in early steps, then stabilizes strongly.
- In compiled 8-GPU runs, steady-state step deltas were around ~140 ms/step after initial compile ramp.
- Default compile-on run validated:
- log: `logs/bench_simple_default_1771494196.txt`
- showed `compile_active:True` and stable post-ramp throughput.
- Eager with `LOGIT_CHUNK_TOKENS=0` is not viable at current batch shape:
- log: `logs/bench_simple_eager_1771494271.txt`
- OOM during backward on multiple ranks.
- `COMPILE_MODE=reduce-overhead` was unstable for this model:
- log: `logs/bench_simple_mode_ro_1771494067.txt`
- failed with CUDAGraph tensor overwrite error in Rotary cache path.
- `COMPILE_FULLGRAPH=1` did not provide a clear steady-state win over the current default compile path in observed tests.

## Correctness/Infra Findings
- Data shards on remote were valid binary shards, not git-lfs pointer files.
- `speedruna2-0` can have competing runs; co-tenant load can distort timing.
- For clean benchmarking, verify no other `torchrun train_gpt_simple.py` is active first.

## Logging/UX Changes
- Removed `nanms` output during warmup handling by removing pseudo-step warmup accounting entirely.
- Warmup is now explicitly separate from training steps, which makes step logs easier to read and avoids confusion in speedrun reporting.

## Commands That Worked
- Standard run on `speedruna2`:
- `brix run --pods speedruna2-0 -- bash -lc 'cd /root/code/openai-parameter-challenge && RUN_ID=simple_$(date +%s) torchrun --standalone --nproc_per_node=8 train_gpt_simple.py'`
- With explicit warmup:
- `... WARMUP_STEPS=200 torchrun --standalone --nproc_per_node=8 train_gpt_simple.py`
- Tail latest log:
- `brix run --pods speedruna2-0 -- bash -lc 'cd /root/code/openai-parameter-challenge && tail -f \"$(ls -1t logs/*.txt | head -n 1)\"'`

## Verified Example Logs
- `logs/bench_simple_pre_1771493756.txt`
- `logs/bench_simple_default_1771494196.txt`
- `logs/bench_simple_mode_ro_1771494067.txt`
- `logs/bench_simple_eager_1771494271.txt`
- `logs/simple_clean_smoke_1771524403.txt`
- `logs/simple_timing_warmup_smoke_1771525491.txt`
- `logs/simple_no_nan_1771527777.txt`
- `logs/simple_warmup_format_1771528075.txt`
- `logs/simple_reset_smoke_1771528206.txt`

## Practical Guidance
- Keep compile unconditional for this script unless a real blocker appears.
- Keep `dynamic=False` and `fullgraph=True` for parity with `train_gpt.py` style and best observed behavior.
- Keep warmup separate and reset state afterward so speed metrics are clean and train trajectory is not altered by warmup activity.
