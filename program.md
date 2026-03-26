# Autoresearch for Parameter Golf

Autonomous AI research agent for the OpenAI Parameter Golf challenge.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: Propose a tag based on today's date (e.g. `mar18`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` — Challenge rules
   - `train_gpt.py` — The file you modify. Model, optimizer, training loop.
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/` exist. If not, tell the human to run `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10`
5. **Initialize results.tsv**: Create with just the header row.
6. **Confirm and go**.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on 8xH100 via Modal. Launch it as:

```
modal run modal_train.py > run.log 2>&1
```

The Modal script mounts your local `train_gpt.py`, so your edits are picked up each run automatically.

**What you CAN do:**
- Modify `train_gpt.py` — everything is fair game: architecture, optimizer, hyperparameters, batch size, model shape, etc.

**What you CANNOT do:**
- **NEVER push to GitHub. NEVER run `git push`. All work stays local.**
- Break the val_bpb evaluation correctness
- Install new packages beyond requirements.txt
- Exceed the 16MB artifact limit (code + int8 zlib-compressed model < 16,000,000 bytes)

**The goal: get the lowest val_bpb.** Current SOTA is 1.2244. The artifact must stay under 16MB.

**The first run**: Always establish the baseline first — run train_gpt.py as-is.

## Output Format

Extract results with: `grep "val_bpb\|final_int8_zlib_roundtrip\|model_params" run.log`

If grep is empty, the run crashed or Modal failed. Run `tail -n 50 run.log` to read the error.

## Reasoning

Before EVERY experiment, you must think and write a reasoning block. No blind changes.

```
=== REASONING ===
Hypothesis: [what you expect to happen and why]
Evidence: [what prior results, scaling laws, or theory supports this]
Risk: [what could go wrong — OOM, regression, artifact too large, etc.]
===
```

After EVERY experiment, you must write an analysis block:

```
=== ANALYSIS ===
Result: val_bpb=X.XXXX artifact=X.XMB (keep/discard/crash)
vs Expected: [better/worse/same than hypothesis predicted]
Why: [your best explanation for the result]
Lesson: [what this tells you about future experiments]
===
```

These blocks are your research log. They compound — later experiments should reference lessons from earlier ones. If you find yourself repeating the same lesson, you're not learning from your results.

## Logging

Log every run to `results.tsv` (tab-separated). Header and 6 columns:

```
commit	val_bpb	artifact_mb	status	reasoning	description
```

1. Git commit hash (short, 7 chars)
2. val_bpb (use 0.000000 for crashes)
3. Artifact size in MB (use 0.0 for crashes)
4. Status: `keep`, `discard`, or `crash`
5. One-line reasoning (the hypothesis, condensed)
6. Short description of the change

Do not commit results.tsv — leave it untracked.

Additionally, maintain a `notes.md` file (also untracked). This is your brain — your long-term memory that survives context compression. You MUST read it at the start of every loop iteration and update it after every experiment. Structure it as:

```markdown
## Best Known Config
[current best val_bpb, commit hash, what config achieved it]

## Dead Ends (do not revisit)
- [direction] — [why it failed] — [experiments that proved it]

## What Works
- [direction] — [magnitude of improvement] — [experiments that proved it]

## Ideas Queue (ranked by expected value)
1. [next thing to try and why]
2. ...

## Experiment Log
### Experiment N: [description]
[paste your REASONING and ANALYSIS blocks here]
```

This file is what drives your decisions. If you're not reading it, you're flying blind.

## Backtracking

Not every path leads somewhere. Watch for these signals and respond:

- **3+ consecutive discards in the same direction**: That direction is a dead end. Abandon it, note it in notes.md, move on to something completely different.
- **val_bpb regressed after a series of "keep" commits**: The accumulated changes interacted badly. Backtrack:
  1. Find the best commit hash from results.tsv
  2. `git reset --hard <commit>`
  3. Log a row with `status=backtrack` in results.tsv
  4. Note in notes.md what went wrong and why
  5. Try a different approach from that known-good state
- **Stuck in a plateau (5+ experiments with <0.001 improvement)**: Step back. Re-read train_gpt.py from scratch. Look for something structural you've been overlooking. Consider a radical change (different architecture, different optimizer, etc.)

## The Experiment Loop

LOOP FOREVER:

1. **Review (MANDATORY)**: You MUST read `results.tsv` and `notes.md` before every experiment. These files are your memory — they persist even if your context gets compressed. Run `cat results.tsv` and `cat notes.md` and use them to decide what to do next. Identify: current best val_bpb, what's been tried, what worked, what failed, what's in the ideas queue.
2. **Reason**: Write the REASONING block. No skipping this. Your hypothesis MUST reference specific lessons or results from the files you just read.
3. **Implement**: Modify `train_gpt.py`.
4. **Commit**: `git commit` the change.
5. **Run**: `modal run modal_train.py > run.log 2>&1` (redirect everything — do NOT flood context)
6. **Extract**: `grep "val_bpb\|final_int8_zlib_roundtrip\|model_params" run.log`
7. **Analyze**: Write the ANALYSIS block. No skipping this either.
8. **Log**: Record in results.tsv and append to notes.md.
9. **Decide**:
   - val_bpb improved AND artifact < 16MB → **keep** the commit
   - val_bpb worse or artifact too large → **discard**: `git reset --hard HEAD~1`
   - crash → attempt trivial fix or discard and move on
10. **Check for backtracking signals** (see above).
11. **Loop**.

**Crashes**: If it's a trivial fix (typo, missing import), fix and retry. If fundamentally broken, discard and move on.

**Timeout**: If a run exceeds 15 minutes, kill it and treat as failure.

**NEVER STOP**: Do not pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, re-read the code, re-analyze results.tsv for patterns, try combining near-misses, try radical changes. Consult notes.md for your ideas queue. The loop runs until the human interrupts you.
