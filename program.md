# ParameterGolf

This is an autonomous experiment to have the LLM optimize a GPT model for the OpenAI Parameter Golf Challenge. The goal is to achieve the lowest val_bpb (bits per byte) within a 16MB artifact size limit and 10-minute training time on 8xH100 GPUs.

The model architecture, optimizer, and training hyperparameters are all fair game for experimentation. The training script (`train_gpt.py`) uses PyTorch and runs on CUDA GPUs.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is focused. Read these files for full context:
   - `README.md` — repository context, challenge rules, and leaderboard.
   - `train_gpt.py` — the file you modify. Model architecture, optimizer, training loop, hyperparameters.
   - `data/README.md` — data preparation and tokenizer information.
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains training shards and `./data/tokenizers/` contains the tokenizer. If not, tell the human to run `python3 data/cached_challenge_fineweb.py --variant sp1024`.
5. **Initialize results.tsv**: Create `results.tsv` with header row. Run the training script once to establish YOUR baseline.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on CUDA via PyTorch. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). You launch it with:

```bash
RUN_ID=test_run \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train_gpt.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization schemes, activation functions, etc.

**What you CANNOT do:**
- Modify data loading or tokenizer files in `data/`. These are fixed for the challenge.
- Install new packages or add dependencies beyond what's in `requirements.txt`.
- Modify the evaluation harness. The `eval_val` function that computes val_bpb is the ground truth metric.
- Exceed the 16MB artifact size limit (code bytes + compressed model bytes must stay under 16,000,000 bytes).

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed (10 minutes on 8xH100), you don't need to worry about training time — it's always capped. However, you should be mindful that architectural changes must still finish within the time budget. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the sequence length. The only constraints are:
1. Code runs without crashing
2. Finishes within the 10-minute time budget
3. Final artifact (code + compressed model) stays under 16MB

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically and prevent the run from completing.

**Artifact size (16MB limit)** is a hard constraint. The final submission must be `train_gpt.py` code bytes + compressed model bytes ≤ 16,000,000 bytes. The script automatically quantizes to int8 and compresses with zlib. Monitor the output line: `Total submission size int8+zlib: {bytes} bytes` to ensure you're under the limit.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script with default Hyperparameters.

## Output format

Once the script finishes it prints a summary like this:

```
---
step:20000/20000 val_loss:1.2244 val_bpb:1.2244 train_time:600000ms step_avg:30.00ms
...
final_int8_zlib_roundtrip val_loss:1.2244 val_bpb:1.2244 eval_time:5000ms
final_int8_zlib_roundtrip_exact val_loss:1.22440000 val_bpb:1.22440000
Total submission size int8+zlib: 15800000 bytes (payload:12000000 raw_torch:45000000 payload_ratio:3.75x)
```

Note that the script stops after the 10-minute wallclock cap. The exact numbers (steps, training_time, eval_time) will vary based on hardware and configuration. The key metrics are:
- `val_bpb` — the main metric to minimize
- `Total submission size int8+zlib` — must stay under 16,000,000 bytes

You can extract the key metrics from the log file:

```bash
grep "^final_int8_zlib_roundtrip_exact val_bpb:\|^Total submission size int8+zlib:" run.log
```

## Logging results

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

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar25` or `autoresearch/mar25-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train_gpt.py` with an experimental idea by directly hacking the code. Consider:
   - Hyperparameter changes (learning rates, momentum, weight decay, warmup/warmdown)
   - Architecture changes (number of layers, model dimension, heads, MLP size, sequence length)
   - Optimizer changes (Muon momentum, Adam betas, learning rate schedules)
   - Novel techniques (quantization-aware training, parameter sharing, test-time training)
3. `git add train_gpt.py && git commit -m "experiment: <description>"`
4. Run the experiment:
   ```bash
   RUN_ID=exp_run \
   MAX_WALLCLOCK_SECONDS=600 \
   torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
   ```
   (Redirect everything — do NOT use tee or let output flood your context)
5. Read out the results:
   ```bash
   grep "^final_int8_zlib_roundtrip_exact val_bpb:\|^Total submission size int8+zlib:\|^peak memory" run.log
   ```
6. If the grep output is empty or incomplete, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (keep results.tsv untracked — do not commit it)
8. If val_bpb improved (lower) AND artifact size ≤ 16MB, `git add train_gpt.py results.tsv && git commit --amend --no-edit` to include the log, advancing the branch
9. If val_bpb is equal or worse OR artifact size > 16MB, record the discard commit hash, then `git reset --hard <previous kept commit>` to discard it cleanly

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (training + eval overhead). If a run exceeds 12 minutes, kill it and treat it as a failure (discard and revert).

**Artifact size limit**: This is a HARD constraint. If `Total submission size int8+zlib` exceeds 16,000,000 bytes, you must reject the change regardless of val_bpb improvement. Log it as "discard" with a note about size limit.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes (quantization, parameter tying, depth recurrence, low-rank adapters, etc.). The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 6/hour, for a total of about 50 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

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
