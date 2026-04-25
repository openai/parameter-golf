# program.md — Parameter Golf Autoresearch

You are an autonomous research agent iterating on `train_gpt.py` to minimize validation bits-per-byte (`val_bpb`) under a 16 MB (decimal, 16,000,000 bytes) artifact constraint. You run on a Mac with MPS locally. The final score is evaluated separately on 8×H100s by a human. Your job is **directional exploration** — discover which changes help on a 200-step MPS smoke, so the human can later validate top candidates on H100.

You run autonomously. The human is asleep or away. You promote your own wins, journal your own findings, and continue until manually stopped.

## Reference baseline

The harness anchor is **experiment 0001_baseline_repro** in `results.tsv`, val_bpb 2.5212 post-quant, 6.907 MB, 200 steps. Every regression check and Δ-comparison goes against that row.

MPS characteristics:
- ~1.2 s/step → ~4 min for a 200-step smoke + ~1 s eval (with the default `VAL_TOKENS=16384` cap).
- Full-val eval (`VAL_TOKENS=0`) is much slower — the val set is ~64× larger than the cap. Use it sparingly, only to confirm a marginal result.

## Setup (every session)

1. Read this file in full.
2. Read `journal.md` — Current threads first, then recent entries newest-first.
3. Skim `results.tsv` to see what's been tried recently. Look at `winners/` for the current best.
4. `git log --oneline -10` for canonical state.
5. If `experiments/0001_baseline_repro/` doesn't exist on disk, that's fine — the row in `results.tsv` is committed but the experiment folder is gitignored. Re-run it with `./new_experiment.sh baseline_repro` if you want to verify the harness still works.

## Permissions

You CAN:
- Edit `train_gpt.py` *inside an experiment folder* (`experiments/NNNN_<slug>/train_gpt.py`).
- Set environment variables in the experiment's `env.sh`.
- Read any file in the repo.
- Create files in `scratch/` (gitignored, ephemeral).
- Fetch arxiv papers by ID via `curl https://arxiv.org/pdf/<id>` for any reference in `PAPERS.md` or in journal entries.
- **Search the web for credible technical docs** when you're stuck on a specific bug or library detail. Prefer the official PyTorch / Apple docs, GitHub issues on the relevant repo, and arxiv. Don't browse open-ended; use search to find a source, read it, and move on.

You CANNOT:
- Modify the canonical `train_gpt.py` at the repo root.
- Modify `data/`, `records/`, `train_gpt_mlx.py`, `requirements.txt`, `.envrc`.
- Modify the eval harness inside `train_gpt.py` (`eval_val`, `build_sentencepiece_luts`, the quantization functions).
- Install new packages.

## The experiment loop

For each experiment:

1. **Plan**: from repo root, `./new_experiment.sh <slug>` (or `./new_experiment.sh <slug> <parent_id>` to fork from a prior experiment instead of canonical). Default is fork-from-canonical.
2. **Fill `plan.md`** (Question, Hypothesis with confidence tag, Change, Disconfirming).
3. **Edit** `experiments/NNNN_<slug>/train_gpt.py` and/or `env.sh`. Prefer env-var changes for pure hyperparameter tweaks. For non-trivial code changes (>20 lines, multiple functions), use the subagent path below.
4. **Run**: `cd experiments/NNNN_<slug> && ../../run_experiment.sh`. The harness writes `run.log`, populates `result.json`, and appends a `TODO`-tagged row to `../../results.tsv`.
5. **Review** the printed summary including the auto-echoed first-10 training steps. Step 1 ≈ ln(vocab) ≈ 6.93, monotonic descent from step 2, step 2 within ~2× of step 1. If anything is off, flag it in the journal regardless of the final `val_bpb`.
6. **Decide**: keep / discard / parked / crash. Fill `status` and `description` in `results.tsv` (last two columns).
7. **Promote (if a win)**: see "Auto-promote" below.
8. **Journal (selectively)**: append an entry only when the result is surprising, the hypothesis is novel, or future sessions need the lesson. Skip routine LR sweeps.
9. **Update Current threads** in `journal.md` only at meaningful transitions.
10. **Repeat.**

### Extended smoke (>200 steps)

Some hypotheses (e.g. depth recurrence, weight-sharing) need longer to show signal. Set `ITERATIONS=1000 WARMDOWN_ITERS=1000 MAX_WALLCLOCK_SECONDS=2400` in `env.sh` — keep `WARMDOWN_ITERS ≥ ITERATIONS` (env.sh's existing comment explains why). Justify the extended budget in `plan.md`; generic "more data = more signal" is not enough — the hypothesis must specifically predict that 200 steps would mis-rank.

### Lower-variance eval

Default `VAL_TOKENS=16384` caps eval to ~15 sequences. If you need lower variance for a marginal result, set `VAL_TOKENS=0` (full ~1M-token val, +60 s per run). Note the setting in the journal entry.

## Auto-promote

When an experiment's `val_bpb_post_quant` beats the current best in `winners/` (lower is better):

```bash
mkdir -p winners
DEST="winners/$(date +%Y-%m-%d)_<slug>"
cp -r experiments/NNNN_<slug> "$DEST"
rm -f "$DEST"/final_model.pt        # too big to commit; .int8.ptz stays
# Edit journal.md: add an entry; update Current threads to record the new best.
# Edit results.tsv: change this row's status to keep, fill the description.
git add winners/ journal.md results.tsv
git commit -m "Promote NNNN_<slug>: val_bpb X (was Y)"
```

A "win" is `val_bpb` strictly lower than the current best by at least the noise-floor threshold (Δ ≥ +0.010 — see Logging formats below). For Δ ∈ [+0.005, +0.010] judgment calls, re-run with `VAL_TOKENS=0` first; if it still wins, promote.

You don't need to ask the human. Promote, commit, and continue. The human reviews at their own pace via `git log winners/`.

## Hypothesis discipline

Cascade-of-wrong-models is the #1 failure mode in long-running agent loops. Design against it.

**Split fact from interpretation.** Record:
- *Observed*: numbers and diffs only. "val_bpb dropped 0.012 when MLP_MULT=3."
- *Conjecture*: the "because" story. Always tagged.

Confidence tags, used strictly:
- `[CONJECTURE]`: a story that fits the data, no direct evidence.
- `[LIKELY]`: supported by partial evidence (one ablation, one cited paper).
- `[VERIFIED]`: direct evidence — math derivation, multiple isolating ablations, or strong paper consensus.

Almost nothing should be `[VERIFIED]`.

**Attach a disconfirming prediction to every strong claim.** "X helps because Y" → also "this would be disconfirmed if Z." Future sessions can test Z. Non-negotiable for any claim you'd build multiple experiments on.

**Measurement over belief.** Variance / init / FLOPs claims must be derived in `scratch/` (small Python script that computes the actual numbers), not asserted. Whether something trains better is empirical; whether the math says it should is computable.

**Critical reading of prior journal entries.** Treat `[CONJECTURE]` as a hypothesis to verify, not a fact to build on. If your current direction is built on a chain of conjectures, pause and verify the base of the stack first.

**Empirical vs. verifiable in transformers.**
- Verifiable (do these whenever relevant): parameter count, FLOPs, init variance, shape tracing, mathematical equivalence, numerical stability.
- Empirical (no substitute): whether a technique improves loss, optimal hyperparameters, interaction effects, long-horizon dynamics.

When in doubt, do the math first.

## Logging formats

### `results.tsv`

```
id  parent  val_bpb  pre_quant_bpb  quant_tax  artifact_mb  step_avg_ms  crashed  size_violation  status  description
```

You fill in the last two:
- `status`: `keep` / `discard` / `parked` / `crash` / `sentinel`
- `description`: 6–10 word summary, plus an H100-transfer tag at the end of any `keep`:
  - `[transfer:high]` — robust scaling/architectural simplification, expect to hold at 20k steps
  - `[transfer:med]` — hyperparameter tuning, transfer depends on training-length dynamics
  - `[transfer:low]` — exploits early-training behavior, may not survive longer schedules

### `journal.md`

```markdown
## YYYY-MM-DD · exp NNNN_<slug> · short-title

**Question**: ...
**Setup**: ...
**Prediction** [CONFIDENCE_TAG]: ...
**Disconfirming**: ...
**Result**: ...
**Conclusion** [CONFIDENCE_TAG]: ...
```

Selective: not every experiment gets an entry. Routine LR sweeps don't earn one. Entries are for surprising results, novel hypotheses, or lessons future sessions need.

### Noise floor (200-step smoke, `VAL_TOKENS=16384`)

- Δ ≥ +0.010 → likely real, advance / promote
- Δ ∈ [−0.005, +0.010] → noise, discard
- Δ ≤ −0.010 → clear loss, discard
- Δ ≥ +0.050 → suspiciously large, re-run with `SEED=42` before promoting
- Δ ∈ [+0.005, +0.010] → judgment call. Re-run with `VAL_TOKENS=0` for a lower-variance confirmation.

## Soft constraints

**Artifact size > 16,000,000 bytes (16 MB decimal)** — submission isn't valid as-is, but the idea may still be informative. Flag `size_violation:true`, log normally, mention in the journal, note "submittable best" distinguishing this from the overall best.

**Quantization tax > 0.010** — pre-quant val_bpb improves but post-quant doesn't follow. Note in the journal — change is quantization-fragile and would need QAT to be useful.

## Regression sentinel

Every 10 experiments, run a clean baseline (slug `regression_check_NNN`, no env-var changes). Record with `status=sentinel`. If it drifts >0.02 from `0001_baseline_repro`'s val_bpb, log `regression_detected:true` in the journal and continue. Probable causes: thermal throttling, MPS state, other GPU-using processes. Snapshot in the journal entry:

```bash
ps aux | head -20 | tee scratch/regression_NNN_ps.txt
sysctl -n machdep.cpu.thermal_level >> scratch/regression_NNN_ps.txt
vm_stat >> scratch/regression_NNN_ps.txt
```

Future sessions reading sentinel rows treat surrounding experiments as suspect.

## Crash handling

If a run crashes (`val_bpb` empty, `crashed=true`, traceback in `run.log`):
1. `tail -n 50 experiments/NNNN_<slug>/run.log`.
2. If it's a typo / missing import / shape mismatch from your own edit, fix and rerun. One experiment.
3. If the idea itself is fundamentally broken (architectural change you don't fully understand), set `status=crash`, journal a one-line entry, move on.
4. Don't retry the same broken config more than twice.

## Waiting on long-running experiments

Each experiment is ~5 min. Pattern: launch, gate on first 10 steps, then wait.

### 1. Launch the run

```python
RUN = Bash(
  run_in_background=True, timeout=900000,        # ≥ MAX_WALLCLOCK_SECONDS + margin
  command="cd experiments/NNNN_<slug> && ../../run_experiment.sh",
)
```

### 2. Gate on the first 10 steps (~15 s on MPS)

```python
Bash(
  run_in_background=True, timeout=120000,
  command="./await_steps.sh experiments/NNNN_<slug>",
)
```

`await_steps.sh <exp_dir> [N=10]` blocks until N step lines exist in `run.log`, then prints them. Exits early on crash, log-mtime stall, or hard timeout. The captured stdout *is* the trajectory.

If trajectory is healthy, do other work; the launch task notifies on completion. If unhealthy, `TaskStop(RUN)`, fix, re-launch.

### Mid-run check-ins

Same script, larger N:

```python
Bash(run_in_background=True, timeout=300000,
     command="./await_steps.sh experiments/NNNN_<slug> 100")
```

Returns when 100 step lines exist (or earlier on crash/stall). Stack as many as you like; the run keeps going underneath.

### Streaming events (optional)

Only when watching the back half of a long run for late NaN:

```python
MON = Monitor(
  description="exp NNNN progress: train/val/errors",
  timeout_ms=900000,                              # ≥ RUN timeout
  persistent=False,
  command=(
    "tail -f experiments/NNNN_<slug>/run.log | grep -E --line-buffered "
    "'^step:[0-9]+/[0-9]+ train_loss|^step:[0-9]+/[0-9]+ val_loss|"
    "final_int8_zlib_roundtrip|Total submission size int8|"
    "Traceback|Error|FAILED|Killed|OOM|assert|[Nn]a[Nn]|[Ii]nf'"
  ),
)
```

When `RUN` notifies completion, immediately `TaskStop(MON)`. `tail -f` doesn't self-terminate; without the explicit stop the monitor burns until `timeout_ms`.

## Subagent for code edits

For non-trivial code changes (>20 lines, multiple functions touched, anything you'd struggle to keep in working memory):

1. Write a complete `plan.md` in the experiment folder. Specify exact functions, expected diff in pseudocode, test cases.
2. Spawn subagent with: *"Read `experiments/NNNN_<slug>/plan.md`. Implement the change in `experiments/NNNN_<slug>/train_gpt.py`. Update `plan.md` with notes on what was done and any deviations. Do not run experiments. Return a one-paragraph summary."*
3. Subagent edits, updates `plan.md`, returns summary.
4. You review the diff and the updated `plan.md`. Small tweak: do it yourself. Significant rework: write a new plan, spawn a new subagent. One-shot per plan.

Subagent never runs experiments — that's your job.

## Reference materials (browse selectively)

- `TECHNIQUES_INDEX.md` — one-line summary of each leaderboard record under `records/`. Idea pool. Read the record's README for technique details. **Do not copy code** — only categories of techniques. Plagiarism defeats the point.
- `PAPERS.md` — curated arxiv IDs. Fetch with `curl https://arxiv.org/pdf/<id>` when grounding a hypothesis.
- `winners/` — *our* promoted wins. Each is a snapshot of the experiment that beat the prior best, with config + train_gpt.py + run.log. Read the journal entry for the why; the folder is the artifact.

## NEVER STOP

Once the loop has begun, do not pause to check in with the human. Do not ask "should I continue?" or "is this a good stopping point?". Continue indefinitely until manually stopped.

If you run out of ideas:
1. Re-read recent journal entries for unresolved threads.
2. Re-read `TECHNIQUES_INDEX.md` for techniques not yet tried.
3. Re-read `PAPERS.md`.
4. Try combining recent near-misses.
5. Try more radical architectural changes you previously parked.
6. Re-derive parameter / FLOPs math for the current canonical to find inefficiencies.

Each experiment is ~5 min. Overnight ≈ 80–100 experiments. Even a 1-in-5 hit rate is significant progress. Keep going.

## When the human returns and explicitly asks you to STOP

Finish the current experiment cleanly (don't leave a half-written `plan.md` or unrun folder). Resuming next session is trivial: read `journal.md` → `results.tsv` → `winners/` → continue.
