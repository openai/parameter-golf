# program.md — Parameter Golf Autoresearch

You are an autonomous research agent iterating on `train_gpt.py` to minimize validation bits-per-byte (`val_bpb`) under a **16 MB** (decimal, 16,000,000 bytes) artifact constraint. You run on a Mac with MPS locally. The final score is evaluated separately on 8×H100s by a human. Your job is **directional exploration** — discover which changes help on a ~200-step MPS smoke test, so the human can later validate top candidates on H100.

## Reference baseline

The harness anchor is **experiment 0001_baseline_repro** (see `results.tsv`). Every regression check and noise-floor comparison is against that row, not against any historical number from the canonical repo (because `VAL_TOKENS` capping and MPS-vs-H100 hardware both shift the absolute number).

Rough MPS characteristics:
- Step time: ~1.2 s/step on an Apple Silicon MPS laptop → ~4 min training + ~1 s eval for a 200-step smoke (with default `VAL_TOKENS=16384`).
- Full-val eval adds ~60 s per experiment; use it only when you need lower variance than the smoke gives.

## Setup (do this every session)

1. Read this file in full.
2. Read `journal.md` — Current threads first, then recent entries newest-first.
3. Skim `results.tsv` to see what's been tried recently. The first row is exp `0001_baseline_repro`; that's your anchor.
4. Check `git status` and `git log --oneline -10` to understand canonical state. (Experiments are not committed; the repo tracks only canonical wins the human has promoted.)
5. If `experiments/0001_baseline_repro/` doesn't exist, stop and tell the human — the harness was set up but not verified.
6. Confirm you understand the experiment loop before starting.

## Hard rules

**You CAN:**
- Edit `train_gpt.py` *inside an experiment folder* (`experiments/NNNN_slug/train_gpt.py`).
- Set environment variables in the experiment folder's `env.sh`.
- Read any file in the repo.
- Create files in `scratch/` (gitignored, ephemeral).
- Fetch papers by arxiv ID via `curl` when they're named in `PAPERS.md` or referenced as `[REF: arxiv:XXXX]` in journal/plan.

**You CANNOT:**
- Modify the canonical `train_gpt.py` at the repo root (only the human promotes wins there).
- Modify `data/`, `records/`, `train_gpt_mlx.py`, `requirements.txt`, `pyproject.toml`, `.envrc`.
- Modify the eval harness inside `train_gpt.py` (`eval_val`, `build_sentencepiece_luts`, quantization functions).
- Install new packages.
- Trigger RunPod or H100 runs.
- Browse the web freely.
- Commit to `main` or the `autoresearch` branch; experiments are only committed when the human promotes a win.

## The experiment loop

For each experiment:

1. **Plan**: from repo root, run `./new_experiment.sh <slug>` (or `./new_experiment.sh <slug> <parent_id>` to fork from a prior experiment's `train_gpt.py` instead of canonical). This creates `experiments/NNNN_<slug>/` with `plan.md`, `env.sh`, `train_gpt.py`, `result.json` templates.
   - **Default is fork-from-canonical.** Forking from a prior experiment is a deliberate choice — only do it when you're explicitly building on that experiment's change. Otherwise you carry uncommitted drift forward without meaning to.
2. **Fill `plan.md`** with Question, Hypothesis (with confidence tag), Change, Disconfirming. Be specific.
3. **Edit** `experiments/NNNN_<slug>/train_gpt.py` and/or `env.sh`. Prefer env-var changes when the modification is purely hyperparameter. If a code change is non-trivial (>20 lines or multiple functions), use the subagent path (see below).
4. **Run**: `cd experiments/NNNN_<slug> && ../../run_experiment.sh`. Output goes to `run.log`; metrics auto-populate `result.json` and one row appends to `../../results.tsv` with `TODO` in the last two columns.
5. **Review** the printed summary: `val_bpb_post_quant`, `quant_tax`, `size_violation`, `crashed`, **and the first-10 training steps** (auto-echoed at the end of the summary). The trajectory of those ten steps is the cheapest signal you have that training is healthy. If step 2 spikes to >2× step 1, or loss isn't monotonically decreasing by step 10, something is off — flag it in the journal regardless of the final `val_bpb`. This is also how you catch half-working experiments that converge to a worse basin than the baseline.
6. **Decide**: keep / discard / parked / crash. Fill `status` and `description` in `results.tsv` (last two columns).
7. **Journal (selective)**: if the experiment had a hypothesis worth remembering, append an entry to `journal.md`. Skip routine tuning.
8. **Update Current threads** in `journal.md` only at meaningful transitions, not after every experiment.
9. **Repeat.**

### Extended smoke (>200 steps)

Default smoke is 200 steps. Some hypotheses (e.g., depth recurrence, weight-sharing) need longer to show signal. Set `ITERATIONS=1000 WARMDOWN_ITERS=200 MAX_WALLCLOCK_SECONDS=1500` in `env.sh`. **Justify the extended budget in `plan.md`** — generic "more data = more signal" is not enough. The hypothesis must specifically predict that 200 steps would mis-rank.

### LR warmup (MPS stability)

The `env.sh` template sets `LR_WARMUP_STEPS=10`. Leave it on. Without it, the first Adam step on the tied token embedding (init std 0.005, `TIED_EMBED_LR=0.05`) produces a per-element update 10× the init scale and sends step-2 loss from ~6.93 to ~19. Today's MPS state deterministically lands in that bad basin; an Apr-18 MPS run got lucky once, not reproducible. Warmup makes the first step effectively `lr/10` and sidesteps the issue for any reasonable config. You can override for experiments testing warmup-free configs (e.g., recreating an H100 submission exactly), but document why in `plan.md`.

### Lower-variance eval (full val set)

Default `VAL_TOKENS=16384` caps eval to ~15 sequences for speed but introduces 200-step noise that can swamp Δ ≈ 0.01 signals. If you want a lower-variance measurement (e.g., for a marginal result on the fence), set `VAL_TOKENS=0` in `env.sh` (uses the full val set, ~1M tokens, adds ~60 s to the run). **Justify in `plan.md` and note the setting in the journal entry** so future sessions don't misread the results.tsv row.

## Hypothesis discipline (the most important section — read carefully)

Cascade of wrong models is the #1 failure mode of long-running agent loops. Design against it.

**Split fact from interpretation.** When recording any result:
- *Observed*: numbers and diff only. "val_bpb dropped 0.012 when MLP_MULT=3."
- *Conjecture*: the "because" story. Always tagged `[CONJECTURE]`, `[LIKELY]`, or `[VERIFIED]`.

**Use confidence tags strictly:**
- `[CONJECTURE]`: a story that fits the data, no direct evidence
- `[LIKELY]`: supported by partial evidence (one ablation, one cited paper)
- `[VERIFIED]`: direct evidence — math derivation, multiple isolating ablations, or strong paper consensus

Almost nothing should be VERIFIED. That's the point.

**Attach a disconfirming prediction to every strong claim.** If you write "X helps because Y," also write "this would be disconfirmed if Z." Future sessions can test Z. Non-negotiable for any claim you'd build multiple experiments on.

**Measurement over belief.** Variance/init/FLOPs claims must be derived in `scratch/`, not asserted. Use Python to compute the actual numbers — operations on small tensors, parameter counts, gradient norms. Whether a thing trains better is empirical; whether the math says it should is computable. Don't conflate them.

**Critical reading of prior journal entries.** Treat `[CONJECTURE]` as a hypothesis to verify or work around, never a fact to build on. If your current direction is built on a chain of conjectures, pause and verify the base of the stack before extending.

**Hedge mechanism, not action.** The norm in transformer research is "we observe X; one possible mechanism is Y." Match that — try things, log results, but don't assert mechanisms confidently. Don't refuse to act because nothing is fully proven; do act, then label the why with appropriate confidence.

**Empirical vs verifiable in transformers**:
- Verifiable (do these whenever relevant): parameter count, FLOPs, init variance, shape tracing, mathematical equivalence between two formulations, numerical stability of specific ops
- Empirical (no substitute): whether a technique improves loss, optimal hyperparameters, interaction effects, long-horizon dynamics

When in doubt, do the math first.

## Logging formats

### `results.tsv` (auto-populated except status, description)

```
id  parent  val_bpb  pre_quant_bpb  quant_tax  artifact_mb  step_avg_ms  crashed  size_violation  status  description
```

You fill in the last two:
- `status`: `keep` / `discard` / `parked` / `crash` / `sentinel`
- `description`: 6–10 word summary

### `journal.md` format

Top-of-file structure:

```markdown
# Journal

## Current threads
- <ongoing direction or open question, 1 line each>
- <kept short, 5–10 bullets max, updated at transitions only>

---

## Entries (newest first)

## YYYY-MM-DD · exp NNNN_slug · short-title

**Question**: ...

**Setup**: ...

**Prediction** [CONFIDENCE_TAG]: ...

**Disconfirming**: ...

**Result**: ...

**Conclusion** [CONFIDENCE_TAG]: ...
```

Selective: not every experiment gets an entry. Routine LR sweeps and obvious-loss-confirmations don't earn one. Entries are for surprising results, novel hypotheses, or lessons future sessions need.

### Noise floor (200-step smoke, `VAL_TOKENS=16384`)

- Δ ≥ +0.010 improvement → likely real, advance
- Δ ∈ [−0.005, +0.010] → noise, discard
- Δ ≤ −0.010 → clear loss, discard
- Δ ≥ +0.050 → suspiciously large, re-run with `SEED=42` before advancing
- For Δ ∈ [+0.005, +0.010]: judgment call. If mechanistically composes with current direction, advance. If orthogonal, safer to discard, or re-run with `VAL_TOKENS=0` for a lower-variance confirmation.

## Soft constraints (warn, don't block)

**Artifact size > 16,000,000 bytes (16 MB decimal).** Submission won't be valid as-is, but the idea may still be informative. Flag `size_violation:true`, log normally, mention in the journal, and note a "submittable best" distinguishing this from the overall best.

**Quantization tax > 0.010.** Pre-quant `val_bpb` improves but post-quant doesn't follow. Note it in the journal — some changes are quantization-fragile and would need QAT to be useful. Worth flagging for future structural exploration.

## Regression sentinel

Every 10 experiments, run a clean baseline experiment (slug `regression_check_NNN`). Use canonical settings, no env-var changes. Record `val_bpb` in `results.tsv` with `status=sentinel`.

If the sentinel drifts >0.02 from `0001_baseline_repro`'s `val_bpb`, log a `regression_detected:true` note in the journal and continue. Probable causes: thermal throttling, MPS state, other processes. Snapshot for context (in journal entry):

```bash
ps aux | head -20 | tee scratch/regression_NNN_ps.txt
sysctl -n machdep.cpu.thermal_level >> scratch/regression_NNN_ps.txt
vm_stat >> scratch/regression_NNN_ps.txt
```

Future sessions reading sentinel rows treat surrounding experiments as suspect.

## Crash handling

If a run crashes (`val_bpb` empty in the summary, traceback in `run.log`):
1. Read `tail -n 50 run.log`.
2. If it's a typo / missing import / obviously fixable bug from your edit, fix it and rerun. Treat as one experiment.
3. If the idea itself is fundamentally broken (e.g., shape mismatch from an architectural change you don't fully understand), set `status=crash` in `results.tsv` with a brief description, and move on.
4. Do not retry the same broken config more than twice.

## H100-transfer flag

When marking a win as `keep`, append a 1-line transfer-confidence note in the description column:
- `[transfer:high]` — change is robust scaling/architectural simplification, expect to hold at 20k steps
- `[transfer:med]` — change tunes hyperparameters, transfer depends on training-length dynamics
- `[transfer:low]` — change exploits early-training behavior, may not survive longer schedules

This helps the human pick winners for RunPod validation.

## Waiting on long-running experiments (Bash run_in_background, Monitor)

Each experiment is ~5 min. Pattern: launch, gate on first 10 steps, then wait.

### 1. Launch the run

```python
# timeout must be ≥ MAX_WALLCLOCK_SECONDS + margin
# 200-step smoke: 900_000 ms.  1000-step extended: 2_400_000 ms.
RUN = Bash(
  run_in_background=True, timeout=900000,
  command="cd experiments/NNNN_slug && ../../run_experiment.sh",
)
```

### 2. Gate on the first 10 steps (cheap, ~15 s on MPS)

```python
Bash(
  run_in_background=True, timeout=120000,
  command="./await_steps.sh experiments/NNNN_slug",
)
```

`await_steps.sh <exp_dir> [N=10]` blocks until N `step:N/N train_loss:...` lines exist in `run.log`, then prints them. It exits early if the python process dies, the log goes stale (hung), or it hits a hard `MAX_WAIT_SECONDS` ceiling. When the notification arrives, the captured stdout *is* the trajectory.

Judge health: step 1 ≈ ln(vocab) ≈ 6.93, monotonic descent from step 2, step 2 within ~2× of step 1. If it looks off, `TaskStop(RUN)`, fix, re-launch — saves ~4 min of bad compute.

### Optional: mid-run check-ins

Same script, larger N. Useful when you suspect something will go wrong later (e.g. you've seen NaN around step 175 historically and want to confirm a fix held to step 100):

```python
Bash(run_in_background=True, timeout=300000,
     command="./await_steps.sh experiments/NNNN_slug 100")
```

The script returns when 100 step lines exist (or earlier if the run dies/stalls), and the captured stdout is the trajectory through step 100. Stack as many of these as you like — the run keeps going underneath.

### 3. Wait for completion

If the first 10 steps are healthy, do other work. `RUN`'s completion notification will arrive when the run exits. Then review `run.log` and the new row in `results.tsv`.

### Optional: stream late-run events

Use only when you need to watch the back half of a healthy long run (e.g. late NaN at step 800 of an extended smoke). Copy:

```python
MON = Monitor(
  description="exp NNNN progress: train/val/errors",
  timeout_ms=900000,                        # ≥ RUN timeout
  persistent=False,
  command=(
    "tail -f experiments/NNNN_slug/run.log | grep -E --line-buffered "
    "'^step:[0-9]+/[0-9]+ train_loss|^step:[0-9]+/[0-9]+ val_loss|"
    "final_int8_zlib_roundtrip|Total submission size int8|"
    "Traceback|Error|FAILED|Killed|OOM|assert|[Nn]a[Nn]|[Ii]nf'"
  ),
)
```

When `RUN` notifies completion, immediately `TaskStop(MON)`. `tail -f` does not self-terminate when Python exits — without the explicit stop, the monitor burns until `timeout_ms` and floods you with a stale event at the end.

## Subagent for code edits

For non-trivial code changes (>20 lines, multiple functions touched, anything you'd struggle to keep in working memory):

1. Write a complete `plan.md` in the experiment folder. The plan must specify exact functions to modify, expected diff in pseudocode, and any test cases.
2. Spawn subagent with prompt: *"Read `experiments/NNNN_<slug>/plan.md`. Implement the change in `experiments/NNNN_<slug>/train_gpt.py`. Update `plan.md` with notes about what was done and any deviations. Do not run experiments. Return a one-paragraph summary."*
3. Subagent does the edit, updates `plan.md` in place, returns summary.
4. You review the diff and the updated `plan.md`. If a small tweak is needed, do it yourself. If significant rework, write a new plan and spawn a new subagent. One-shot per plan.

Subagent never runs experiments — that's your job. Subagent never modifies anything outside the named experiment folder.

## Reference materials (optional, browse selectively)

- `TECHNIQUES_INDEX.md` — one-line summary of each leaderboard record under `records/`. Use as an idea pool. Read the full record's README for technique details when interested. **Do not copy code** — only categories of techniques. Plagiarism defeats the point and won't be merged.
- `PAPERS.md` — curated list of relevant papers with arxiv IDs. Read when grounding a hypothesis. Fetch with `curl https://arxiv.org/pdf/<id>` when needed.

## NEVER STOP

Once the loop has begun, do not pause to check in with the human. Do not ask "should I continue?" or "is this a good stopping point?". The human may be asleep or away, and expects you to continue indefinitely until manually stopped.

If you run out of ideas:
1. Re-read recent journal entries for unresolved threads.
2. Re-read `TECHNIQUES_INDEX.md` for techniques not yet tried.
3. Re-read `PAPERS.md`.
4. Try combining recent near-misses.
5. Try more radical architectural changes you previously parked.
6. Re-derive parameter/FLOPs math for the current canonical to find inefficiencies.

Each experiment is ~5 min. Overnight = 80–100 experiments. Even a 1-in-5 hit rate is significant progress. Keep going.

## When the human returns

Finish the current experiment cleanly (don't leave a half-written `plan.md` or unrun folder). Resuming next session is trivial: read `journal.md`, `results.tsv`, current threads, continue.
