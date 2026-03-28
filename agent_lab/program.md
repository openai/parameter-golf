# Agent lab

Do not change tokenizer/byte accounting, validation split, or `eval_val` semantics unless the human explicitly asked; if you do, prove `val_bpb` is still valid.

## Setup

1. Propose run tag (e.g. date `mar28`). Branch `agent_lab/<tag>` must not exist.
2. `git checkout -b agent_lab/<tag>` from `main` (or branch the human names).
3. Read: repo [`README.md`](../README.md), this folder [`README.md`](README.md), **`agent_lab/train_gpt.py`** (only file you edit).
4. Ensure `DATA_PATH` shards and `TOKENIZER_PATH` exist; else tell human to run `python3 data/cached_challenge_fineweb.py` per main README.
5. Create **`agent_lab/results.tsv`** with header only (see Logging).
6. Read **`agent_lab/experiments.tsv`** (what’s already been tried; stable IDs **`AL-YYYYMMDD-NNN`**).
7. After human confirms, start the loop.

## Experimentation

Run from **repo root**:

```bash
RUN_ID=... DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  torchrun --standalone --nproc_per_node=1 agent_lab/train_gpt.py
```

`MAX_WALLCLOCK_SECONDS`: default **600** (`Hyperparameters`). Human may set shorter (e.g. `300`) or `0` for no cap.

**CAN:** edit **`agent_lab/train_gpt.py`** only.

**CANNOT:** edit root **`train_gpt.py`**; add deps without human approval; leak val / break challenge rules; break **`val_bpb`** definition (leave `eval_val`, byte LUTs, val loader alone unless instructed).

**Primary metric** (pick one for the whole run; lower `val_bpb` wins):

- Default: **`final_int8_ttt_lora`**
  ```bash
  grep '^final_int8_ttt_lora' agent_lab/run.log
  ```
- Alt: **`final_int8_zlib_roundtrip`**
  ```bash
  grep '^final_int8_zlib_roundtrip ' agent_lab/run.log
  ```

Do not mix metrics across experiments.

**Artifact:** if `Total submission size int8+zlib:` + code exceeds **16_000_000** bytes, discard or fix.

**VRAM:** soft cap — don’t explode memory for tiny gains.

**Simplicity:** prefer smaller diff / less code at equal or better `val_bpb`.

**First run:** baseline = unmodified `agent_lab/train_gpt.py` (env vars for hardware OK).

## Scoring lines

Use your chosen `final_*` line. Memory: `peak memory allocated: <N> MiB` → GB ≈ `N / 1024`.

## Logging

Append to **`agent_lab/results.tsv`** (tabs, not commas):

```
exp_id	commit	val_bpb	memory_gb	status	description
```

- `exp_id`: optional; use **`AL-YYYYMMDD-NNN`** to match [`experiments.tsv`](experiments.tsv)  
- `commit`: 7-char hash  
- `val_bpb`: from primary metric; `0.000000` if crash  
- `memory_gb`: one decimal; `0.0` if crash  
- `status`: `keep` | `discard` | `crash`  
- `description`: short  

Do not commit `results.tsv`.

**Also append [`agent_lab/experiments.tsv`](experiments.tsv)** (tracked): one row per experiment with `exp_id`, parent commit, hypothesis, **verdict** (`correct` / `wrong` / `partial` / `n_a`), metric, `val_bpb`, notes.

**Commits:** use **Conventional Commits** with scope `agent-lab` and a **rich body** (`Exp:`, `Parent:`, `Hypothesis:`, `Result:`). See [`COMMIT_CONVENTIONS.md`](COMMIT_CONVENTIONS.md). Official train **and** eval time limits for leaderboard: [`CHALLENGE_TIMELIMITS.md`](CHALLENGE_TIMELIMITS.md).

**Human journal / pedagogy:** update **`docs/build-logs/<date>-agent-lab.md`** with thoughts, what changed in code, and lessons — not only numbers.

## Loop

Branch: `agent_lab/<tag>` or `agent_lab/<tag>-gpu0`.

1. Note branch/commit.
2. Change **`agent_lab/train_gpt.py`** (one idea).
3. `git add agent_lab/train_gpt.py && git commit` — subject **`feat(agent-lab): …`** (see [`COMMIT_CONVENTIONS.md`](COMMIT_CONVENTIONS.md))
4. `torchrun ... agent_lab/train_gpt.py > agent_lab/run.log 2>&1` (no `tee`; full redirect).
5. `grep '^final_int8_ttt_lora\|^peak memory allocated' agent_lab/run.log` (adjust grep if you chose the zlib metric).
6. Empty primary line → likely crash → `tail -n 80 agent_lab/run.log`; fix trivial errors or log `crash` and revert.
7. Append TSV row.
8. Better `val_bpb` → keep commit. Else → `git reset --hard` to last good commit.

**Timeout:** if wall time ≫ **2×** `MAX_WALLCLOCK_SECONDS` + eval slack (e.g. >25 min at 600s cap), kill → discard, revert.

**Crashes:** fix typos or log `crash` and move on.

After setup, **never** ask whether to continue. Run until the human stops you. If stuck: re-read `agent_lab/train_gpt.py`, vary ideas.
