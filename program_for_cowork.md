# ParameterGolf — Local + Remote GPU Cowork Mode

This is an autonomous experiment to have the LLM optimize a GPT model for the OpenAI Parameter Golf Challenge. The goal is to achieve the lowest val_bpb (bits per byte) within a 16MB artifact size limit and 10-minute training time on 8xH100 GPUs.

The model architecture, optimizer, and training hyperparameters are all fair game for experimentation. The training script (`train_gpt.py`) uses PyTorch and runs on CUDA GPUs.

## Architecture Overview

```
┌──────────────────────────────────┐      shared filesystem
│  LOCAL (agent / CPU machine)     │◄────────────────────────►  REMOTE (GPU machine)
│  - edits train_gpt.py            │                            - runs gpu_watcher.py
│  - git commit                    │   .gpu_state.json          - executes experiments
│  - writes remote_gpu_run.sh      │◄──────────────────────►   - writes run.log
│  - sets state → pending          │                            - updates state
│  - blocks on wait_for_result.py  │
│  - reads results, keep/discard   │
└──────────────────────────────────┘
```

### State Machine

```
idle
 │  local: commit code, write remote_gpu_run.sh, write state=pending
 ▼
pending
 │  remote watcher: picks up, git pull, verifies script
 ▼
running  ←── heartbeat (updated_at refreshed every 30 s)
 │  remote watcher: bash remote_gpu_run.sh > run.log
 ├─── exit 0 ──────────────────────────────► completed
 └─── exit non-0 / timeout / exception ───► failed
                                              │
completed / failed ◄──────────────────────── │
 │  local: reads run.log, records results.tsv
 │  local: writes state=idle
 ▼
idle  (next iteration)
```

### Protocol Files

| File | Written by | Purpose |
|------|------------|---------|
| `.gpu_state.json` | both sides | state machine, atomically updated |
| `remote_gpu_run.sh` | local | exact shell command(s) for this experiment |
| `run.log` | remote | full stdout+stderr of the training run |

`.gpu_state.json` schema:
```json
{
  "state":       "idle | pending | running | completed | failed",
  "commit":      "<7-char git hash or null>",
  "run_script":  "remote_gpu_run.sh",
  "pid":         12345,
  "started_at":  "2025-03-27T10:00:00+00:00",
  "finished_at": "2025-03-27T10:12:00+00:00",
  "exit_code":   0,
  "error":       null,
  "updated_at":  "2025-03-27T10:12:00+00:00"
}
```

---

## Setup

To set up a new experiment session, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is focused. Read these files for full context:
   - `README.md` — repository context, challenge rules, and leaderboard.
   - `train_gpt.py` — the file you modify. Model architecture, optimizer, training loop, hyperparameters.
   - `data/README.md` — data preparation and tokenizer information.
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains training shards and `./data/tokenizers/` contains the tokenizer. If not, tell the human to run `python3 data/cached_challenge_fineweb.py --variant sp1024`.
5. **Verify remote watcher is running**: Ask the human to confirm that on the remote GPU machine, `gpu_watcher.py` is already running:
   ```bash
   # On remote GPU machine (run once, keep in background/tmux):
   python3 gpu_watcher.py --repo-dir /path/to/shared/repo --poll-interval 5
   ```
6. **Initialize results.tsv**: Create `results.tsv` with header row.
7. **Initialize state**: Ensure `.gpu_state.json` is in `idle` state. If it doesn't exist, create it:
   ```bash
   python3 -c "
   import json, pathlib, datetime
   pathlib.Path('.gpu_state.json').write_text(json.dumps({
     'state': 'idle', 'commit': None, 'run_script': None,
     'pid': None, 'started_at': None, 'finished_at': None,
     'exit_code': None, 'error': None,
     'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
   }, indent=2) + '\n')
   print('Created .gpu_state.json (idle)')
   "
   ```
8. **Run baseline**: Run the training script once with default hyperparameters to establish YOUR baseline (follow the experiment submission procedure below).
9. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation loop.

---

## Submitting an Experiment (the critical procedure)

Each time you have modified `train_gpt.py` and are ready to run, follow these steps **exactly**:

### Step 1 — Commit the code change

```bash
git add train_gpt.py && git commit -m "experiment: <description>"
```

Get the commit hash:
```bash
COMMIT=$(git rev-parse --short=7 HEAD)
echo "Commit: $COMMIT"
```

### Step 2 — Write `remote_gpu_run.sh`

Write the exact command the remote machine should run. **Always redirect output to run.log** (the watcher opens run.log itself, but the script can add extra logging):

```bash
cat > remote_gpu_run.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

# This file is generated by the local agent for each experiment.
# It is executed by gpu_watcher.py on the remote GPU machine.

RUN_ID=exp_run \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
EOF
chmod +x remote_gpu_run.sh
```

> **Note**: Do NOT redirect inside the script — `gpu_watcher.py` already captures all stdout+stderr to `run.log`.

### Step 3 — Set state to `pending`

This is the signal that triggers the remote watcher:

```bash
python3 -c "
import json, pathlib, datetime

commit = '$(git rev-parse --short=7 HEAD)'
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')

state = {
  'state': 'pending',
  'commit': commit,
  'run_script': 'remote_gpu_run.sh',
  'pid': None,
  'started_at': None,
  'finished_at': None,
  'exit_code': None,
  'error': None,
  'updated_at': now,
}
tmp = pathlib.Path('.gpu_state.json.tmp')
tmp.write_text(json.dumps(state, indent=2) + '\n')
tmp.replace('.gpu_state.json')
print(f'State → pending  (commit={commit})')
"
```

### Step 4 — Block until experiment finishes

**Do not proceed until this command exits.** It will block and print live status:

```bash
python3 wait_for_result.py --repo-dir . --poll-interval 5 --timeout 1200
```

- Exit code `0` → `completed` (success)
- Exit code `1` → `failed` (crash, timeout, or OOM)
- Exit code `2` → local safety timeout hit (something went wrong)

### Step 5 — Read results

```bash
grep "^final_int8_zlib_roundtrip_exact val_bpb:\|^Total submission size int8+zlib:\|^peak memory" run.log
```

If the output is empty or incomplete, the run crashed. Read the tail of the log:
```bash
tail -n 50 run.log
```

### Step 6 — Reset state to `idle`

Always do this after reading results, so the next experiment can be submitted:

```bash
python3 -c "
import json, pathlib, datetime
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
state = {
  'state': 'idle', 'commit': None, 'run_script': None,
  'pid': None, 'started_at': None, 'finished_at': None,
  'exit_code': None, 'error': None, 'updated_at': now,
}
tmp = pathlib.Path('.gpu_state.json.tmp')
tmp.write_text(json.dumps(state, indent=2) + '\n')
tmp.replace('.gpu_state.json')
print('State → idle')
"
```

---

## Output Format

Once the script finishes it prints a summary like this:

```
---
step:20000/20000 val_loss:1.2244 val_bpb:1.2244 train_time:600000ms step_avg:30.00ms
...
final_int8_zlib_roundtrip val_loss:1.2244 val_bpb:1.2244 eval_time:5000ms
final_int8_zlib_roundtrip_exact val_loss:1.22440000 val_bpb:1.22440000
Total submission size int8+zlib: 15800000 bytes (payload:12000000 raw_torch:45000000 payload_ratio:3.75x)
```

The key metrics are:
- `val_bpb` — the main metric to minimize
- `Total submission size int8+zlib` — must stay under 16,000,000 bytes

---

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide peak memory by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.2244	45.0	keep	baseline
b2c3d4e	1.2180	45.2	keep	increase matrix LR to 0.05
c3d4e5f	1.2350	45.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

---

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar25`).

LOOP FOREVER:

1. **Check git state**: current branch/commit we're on, current `.gpu_state.json` (must be `idle`)
2. **Tune `train_gpt.py`** with an experimental idea. Consider:
   - Hyperparameter changes (learning rates, momentum, weight decay, warmup/warmdown)
   - Architecture changes (number of layers, model dimension, heads, MLP size, sequence length)
   - Optimizer changes (Muon momentum, Adam betas, learning rate schedules)
   - Novel techniques (quantization-aware training, parameter sharing, test-time training)
3. **Submit experiment** (Steps 1–3 of the submission procedure above):
   - `git add train_gpt.py && git commit -m "experiment: <description>"`
   - Write `remote_gpu_run.sh`
   - Set `.gpu_state.json` state to `pending`
4. **Block until done** (Step 4):
   ```bash
   python3 wait_for_result.py --repo-dir . --poll-interval 5 --timeout 1200
   ```
   This command does not return until the remote experiment finishes (or times out locally).
5. **Read results** (Step 5):
   ```bash
   grep "^final_int8_zlib_roundtrip_exact val_bpb:\|^Total submission size int8+zlib:\|^peak memory" run.log
   ```
6. **Reset state to idle** (Step 6) — **always do this before proceeding**
7. **Handle crashes**: if `wait_for_result.py` exits with code 1 (failed), read `tail -n 50 run.log` for the stack trace. Attempt a fix if easy. If the idea is fundamentally broken, skip it.
8. **Record results** in `results.tsv` (keep `results.tsv` untracked — do not commit it)
9. **Keep or discard**:
   - If val_bpb improved (lower) AND artifact size ≤ 16MB: `git add train_gpt.py results.tsv && git commit --amend --no-edit` to advance the branch
   - If val_bpb is equal or worse OR artifact size > 16MB: `git reset --hard <previous kept commit>` to discard

---

## Constraints

**Timeout**: Each experiment should take ~10 minutes total (training + eval overhead). The hard timeout in `gpu_watcher.py` is 15 minutes. If `wait_for_result.py` exits with code 2 (local timeout, 20 min), something is seriously wrong — check the remote watcher.

**Artifact size limit**: HARD constraint. If `Total submission size int8+zlib` exceeds 16,000,000 bytes, reject the change regardless of val_bpb improvement. Log it as "discard".

**Crashes**: If a run crashes (OOM, bug, etc.), use judgment: fix trivial issues (typo, missing import) and re-run. Skip fundamentally broken ideas (log "crash").

**What you CAN do:**
- Modify `train_gpt.py` — the only file you edit.

**What you CANNOT do:**
- Modify data loading or tokenizer files in `data/`.
- Install new packages beyond `requirements.txt`.
- Modify the evaluation harness (`eval_val` function).
- Exceed the 16MB artifact size limit.

---

## NEVER STOP

Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

---

## Recommended Exploration Directions

Based on current leaderboard entries, consider exploring these directions:

1. **Quantization schemes**: Mixed int6/int8, int5 for MLP layers, quantization-aware training (QAT)
2. **MLP architecture**: Increase expansion factor (2x → 3x or higher), try alternative activation functions
3. **Model depth**: Experiment with number of layers (current baseline is 9 layers)
4. **Sequence length**: Try longer context (2048, 4096 tokens) — note trade-off with throughput
5. **Learning rate tuning**: Optimize separate LRs for embeddings, matrices, scalars
6. **Weight decay**: Find optimal WD for Muon optimizer (current leaderboard uses 0.04)
7. **Novel architectures**: SmearGate, BigramHash, test-time training, parameter sharing
8. **Evaluation tricks**: Sliding window evaluation at different strides

Always balance improvements against the 16MB artifact limit and 10-minute time budget.

---

## Troubleshooting

### State stuck in `pending` (remote watcher not picking up)
- Check that `gpu_watcher.py` is running on the remote machine
- Verify the shared filesystem is accessible from both sides
- Check remote watcher logs for errors

### State stuck in `running` (watcher heartbeat stopped)
- `gpu_watcher.py` refreshes `updated_at` every 30 s during a run
- If `updated_at` goes stale for >5 min, the watcher auto-recovers to `idle` on next startup
- To manually recover: set `state` back to `idle` in `.gpu_state.json`

### `wait_for_result.py` timed out (exit code 2)
- Default local timeout is 1200 s (20 min)
- Check the remote machine: is `gpu_watcher.py` still running?
- Read `run.log` for partial output

### Git pull failed on remote
- The state will be set to `failed` with `error: git_pull_failed`
- Investigate network/filesystem access on the remote machine
