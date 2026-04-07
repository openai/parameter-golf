---
name: optimize-anything
description: >
  Instantly scaffold Karpathy-style autoresearch loops for ANY optimization problem.
  Use this skill whenever the user wants to set up autonomous experimentation on a
  measurable problem: trading strategy optimization, ML training, API latency tuning,
  build time reduction, test suite speed, prompt engineering, compression ratios, game
  AI scores, competitive programming solutions, or literally anything with a numeric
  metric. Also trigger when the user mentions "autoresearch", "autonomous optimization",
  "experiment loop", "overnight optimization", or wants an agent to iterate on code
  while they sleep. This skill generates the complete scaffold: program.md (agent
  instructions), prepare.py (environment validation), eval.sh (metric extraction),
  and supporting files. Even if the user just says "optimize X" or "make this better
  automatically", use this skill.
---

# Optimize Anything

Generate a complete autoresearch scaffold for any optimization problem following
Karpathy's autoresearch pattern.

**The contract**: one agent reads `program.md`, modifies one file, optimizes one
metric, and loops — keeping improvements, discarding failures, running
indefinitely. The human steers by editing `program.md`. That's the whole system.

The only extension: for slow experiments (>2 min), the agent spawns a single
background **Researcher** subagent that generates hypotheses in parallel. The
agent consults the Researcher's output but also thinks for itself — it is never
a dumb executor.

---

## Prerequisites — Is Autoresearch Right for This Problem?

Before scaffolding anything, verify these six conditions. ALL must be true
for autoresearch to be beneficial. If any fail, tell the user and suggest
an alternative approach (manual iteration, A/B testing, etc.).

| # | Condition | Why it matters |
|---|-----------|---------------|
| 1 | **One thing to measure.** There is a single function, script, or file whose output you're optimizing. | The agent needs a clear, bounded target. If the optimization is spread across 10 files with unclear interactions, the agent can't isolate what works. |
| 2 | **Experiments run in < 5 minutes.** A single eval cycle completes quickly. | Autoresearch's power comes from volume. At 5 min/experiment you get ~12/hour, ~100 overnight. At 30 min/experiment you get 2/hour — not enough to find non-obvious improvements. If experiments take > 5 min, the quick screen (1-min prescreen) is mandatory to filter bad ideas before committing full eval time. |
| 3 | **You can run a lot of experiments.** There's no cost, rate limit, or quota that prevents running 50-100+ experiments. | If you can only run 5 experiments total (API credits, GPU budget, etc.), autoresearch is the wrong tool. Manual, carefully planned experiments are better. |
| 4 | **Cost of a bad iteration is low.** A failed experiment wastes only time, not money, data, or reputation. | The agent WILL produce broken code, regressions, and crashes. If a bad experiment corrupts your database, costs $100 in API calls, or deploys to production, autoresearch is dangerous. The git keep/discard ratchet protects the codebase but not external systems. |
| 5 | **Frozen evaluator.** The evaluation function does not change during the experiment loop. | If the eval changes, experiments aren't comparable. The agent can't tell if a metric change came from its code modification or from the eval shifting. `prepare.py` and `eval.sh` must be immutable. |
| 6 | **Objective numeric metric.** There is a single number that captures "better" vs "worse," and the direction (higher/lower) is unambiguous. | The agent makes binary keep/discard decisions. It needs an unambiguous signal. Subjective quality ("does this look better?") or multi-metric tradeoffs without a composite formula don't work. |

If conditions 1-6 are met but experiments take > 5 minutes, autoresearch
still works — it just requires the quick screen and background Researcher
to maintain throughput.

If condition 2 barely fails (5-15 min experiments), autoresearch can work
with the quick screen filtering most bad ideas in 1 minute. If experiments
take > 30 minutes, seriously consider whether the problem is better suited
to manual experimentation with careful planning.

---

## Phase 0: Codebase Investigation

Before asking the user anything, silently analyze the project.

### 0a. Read everything relevant

```bash
ls -la
find . -name "*.py" -o -name "*.js" -o -name "*.rs" -o -name "*.toml" \
  -o -name "*.json" -o -name "*.yaml" -o -name "*.md" | head -60
cat README.md 2>/dev/null
```

Build a mental model: what the project does, language/framework, existing
eval scripts, available data, hard constraints, competitive landscape.

### 0b. Determine what "success" means

- What is the FINAL evaluation this code will face?
- What does the user ACTUALLY care about?
- What failure modes would make a "high-scoring" result useless?

Form a hypothesis about the right metric. Propose it with reasoning.

### 0c. Measure eval speed

```bash
time <eval_command>
```

| Eval Speed | Quick Screen? | Background Researcher? |
|------------|--------------|----------------------|
| **< 2 min** | No | No — agent researches between experiments |
| **2 – 10 min** | Yes, 1-min screen | Yes — researches during experiments |
| **> 10 min** | Yes, mandatory | Yes — critical |

Also compute the **overhead budget**: eval time determines how much time
the agent should spend thinking between experiments. If eval takes 10
minutes, the agent should spend < 3 minutes on planning + git + setup.
If eval takes 30 seconds, the agent should spend < 30 seconds planning.
The goal: maximize experiments per hour, not deliberation per experiment.

### 0d. Inspect actual eval output

Run eval once, read raw output character by character. Build metric
extraction from REAL output. Not guesses.

---

## Phase 1: Strategy Design

### 1a. Quick Screen (experiments > 2 min only)

A 1-minute shortened eval compared to the **last kept run at the same
shortened duration**.

**Flow:**
1. `prepare.py` runs initial code at quick-screen length → saves to `quick_baseline.txt`
2. After every **keep**, re-run kept code at quick-screen length → overwrite baseline
3. Each new experiment: quick screen first, compare to baseline
4. If > 2% worse relative to baseline → skip. Otherwise promote to full eval.

The threshold is lenient — false-promoting is cheap, false-rejecting is expensive.

**Implementation**: `QUICK=1 bash eval.sh` runs shortened eval.

### 1b. Domain Investigation & Proposal

Research what's standard in the detected domain. Propose enhancements:
- Robustness testing (perturbation, data subsetting, seed rotation)
- Data splitting (walk-forward, temporal cross-validation)
- Composite metrics (weighted objectives, hard constraint gates)
- Domain-specific techniques

Present as: "In [domain], it's standard to do [X] because [Y]. For your
setup I recommend [Z]. Want me to include this?"

---

## Phase 2: Scaffold Generation

After user confirmation, generate the files.

### The Three-File Contract (from Karpathy)

| File | Who modifies | Purpose |
|------|-------------|---------|
| `prepare.py` | **Nobody** — immutable | Environment validation, data prep, evaluation utilities. The trusted ground truth. Never modified by agent or human during experiments. |
| `[target_file]` | **Agent** | The single file the agent edits. All changes happen here. |
| `program.md` | **Human** | Agent instructions. The human's control surface. Edit this to steer research direction, adjust strategy, change constraints. |

Additional generated files (`eval.sh`, `compute_score.py`, etc.) are also
**immutable during experiments**.

### File 1: `program.md` — The Agent's Instructions

This is the single most important file. It must be complete enough that the
agent can run indefinitely without human input, but concise enough to fit
in context alongside the target file.

Structure:

```markdown
# [Project] Autoresearch

**Goal**: [1-2 sentences. What does "winning" look like?]

## Setup

1. Read `[target_file]` completely. Understand every function, every constant.
2. Read `prepare.py` to understand evaluation (but NEVER modify it).
3. [Read any reference files — record submissions, docs, etc.]
4. Run `python prepare.py` to validate environment.
5. [If slow experiments] Spawn background Researcher:
   `claude -p "$(cat researcher_program.md)" > researcher.log 2>&1 &`
6. Begin the experiment loop.

## Speed Rules

Your #1 job is to MAXIMIZE EXPERIMENTS PER HOUR.

A fast discard teaches more than a slow deliberation. Bias toward action.

**Overhead budget**: Your planning + git + setup time between experiments
must be UNDER [overhead_budget]. If you're spending longer than that
deciding what to try, you're overthinking. Commit and run.

| Eval Time | Max Overhead |
|-----------|-------------|
| < 30 sec  | 30 seconds  |
| 30s – 2m  | 1 minute    |
| 2 – 10m   | 3 minutes   |
| > 10m     | 5 minutes   |

This means: no multi-paragraph deliberation in your thinking. Decide,
implement, commit, run. Learn from the result, not from speculation.

## The Experiment Loop

LOOP FOREVER:

### 1. Decide what to try

Use ALL inputs: your own analysis, results.tsv, research_queue.md (if
Researcher is running), your domain knowledge.

You are the strategist AND the executor. Don't blindly pull from the
research queue — evaluate, combine, modify, reject.

**Escalation**: The further you are from your last keep, the further your
next experiment should be from the current code. Nearby changes after a
keep. Progressively more radical changes as discards accumulate. This is
a continuous gradient, not a fixed schedule.

**Deduplication**: Check results.tsv. Don't retry an experiment with
status `discard` or `crash` unless EITHER:
- The implementation is meaningfully different, OR
- A major keep has changed the codebase since the original attempt
  (see "Re-evaluation after major keeps" below)

### 2. Implement

Modify ONLY `[target_file]`.

Do NOT modify `prepare.py`, `eval.sh`, `compute_score.py`, or any data
files. If you want to change the evaluation, STOP — you're gaming the
metric.

### 3. Commit

`git commit -am "exp: <short description>"`

### 4. Evaluate

[If quick screen enabled]:
`timeout 90 env QUICK=1 bash eval.sh > quick.log 2>&1`
Compare quick_score to quick_baseline.txt.
If > 2% worse → log as `prescreen_fail`, revert, go to step 1.

Full eval:
`timeout [3x_expected] bash eval.sh > run.log 2>&1`

[If eval is slow and Researcher is running]:
While eval runs in background (`&`), DO NOT IDLE. Use the time:
- Read a paper, competition submission, or record entry
- Review results.tsv for patterns
- Plan next 2-3 experiments
- Update your mental model of what's working
This time is NOT optional. Every minute of idle training time is a
minute of wasted research opportunity.

### 5. Extract results

`grep "[metric_name]" run.log`

If empty → crashed. `tail -n 50 run.log`.
- Fixable (typo, import): fix, re-commit, re-run. Max 2 attempts.
- Fundamental (OOM, impossible): log crash, revert, move on.

Check all hard constraints.

### 6. Record

Append to results.tsv (tab-separated):
```
commit	[metric]	[constraint_cols]	status	description
```
Do NOT commit results.tsv to git.

### 7. Keep or discard

**Keep** (metric improved AND constraints satisfied):
- `git push origin HEAD`
- [If quick screen]: Update baseline:
  `timeout 90 env QUICK=1 bash eval.sh > qk.log 2>&1`
  Save to quick_baseline.txt.

**Discard** (metric worse OR constraint violated):
- `git reset --hard HEAD~1`

### 8. Re-evaluation after major keeps

After any keep that improves the metric by MORE THAN 5% relative to the
previous best, the codebase has changed significantly. Ideas that failed
before might now succeed because the context is different.

When this happens:
1. Scan the last 5 discarded experiments in results.tsv.
2. For each: would this idea interact differently with the new code?
3. If plausible, re-queue it. Mark in results.tsv description that it's
   a retry: "exp: retry BigramHash (post-warmdown context)"

This prevents the agent from permanently blacklisting ideas that failed
in an earlier codebase state.

### 9. Reflect (every 10 experiments)

Run `python summarize_results.py` and review:
- What directions are fruitful? Double down.
- What directions are exhausted? Abandon.
- Are you being too incremental? Go bolder.

Go to step 1.

## Structural Change Bias

Parameter tuning (adjusting values ±10-20%) has a LOW improvement ceiling.
It will never find the large hidden improvements in the codebase. Those
come from STRUCTURAL changes: different algorithms, different architectures,
different training schedules, new components, removing dead weight.

History shows that single non-obvious structural changes routinely produce
10x more improvement than all careful tuning combined. Your job is to find
those structural changes, not to polish the current approach.

**Rule: structural experiments should outnumber tuning experiments 2:1.**
For every parameter sweep, you should run at least 2 experiments that
change WHAT the code does, not just the values it uses.

Examples of structural vs tuning:
- Tuning: LR 0.04 → 0.06, warmdown 800 → 500, EMA decay 0.997 → 0.99
- Structural: add XSA attention, switch optimizer, add depth recurrence,
  change activation function, add a new training phase, remove a component

## Diversity Requirement

Never spend more than 2 CONSECUTIVE experiments in the same category.
Categories are defined by what PART of the code you're changing, not
just the specific values.

Examples of categories:
- Learning rate / schedule tuning
- Optimizer changes
- Architecture changes (layers, attention, MLP)
- Quantization / compression
- Regularization (EMA, dropout, weight decay)
- Training loop (batch size, gradient accumulation, phases)

After 2 experiments in one category, you MUST try a different category
before returning. This forces exploration breadth.

## Research Directions

[Generated from codebase analysis. Every item references actual code.]

### Starting Points
- [Specific structural change 1 with line reference]
- [Specific structural change 2 with line reference]
- [Specific structural change 3 with line reference]

### Deeper Ideas
- [Idea 1 with reasoning and mechanism]
- [Idea 2 with reasoning and mechanism]

### Speculative
- [Radical idea 1]
- [Radical idea 2]

These are STARTING POINTS. Generate your own ideas from execution
experience. Consult research_queue.md. The best experiments come from
combining execution insight with broader research.

## Constraints

[Hard constraints — artifact size, time limits, position limits, etc.]

## NEVER STOP

Run indefinitely. If out of ideas: re-read reference material, retry
discarded ideas in the new context, try combinations of keeps, consult
research_queue.md, try radical departures. The loop runs until the human
interrupts you.
```

### File 2: `researcher_program.md` — Background Researcher

Only generated when experiments take > 2 minutes.

```markdown
# Researcher

You run in the background while the main agent executes experiments.
Your job: generate hypotheses and maintain research_queue.md.

## Observe → Hypothesize → Escalate

Read experiment_feedback.md and results.tsv to understand what's happening.
Generate hypotheses for changes that might improve the metric.

Your hypotheses are DOMAIN-AGNOSTIC. You think: "what change to this code
might improve this number?" Ideas can come from anywhere — the domain,
analogous domains, papers, patterns in the data, intuition.

## Escalation

Hypotheses become MORE RADICAL the further the agent is from a keep.

**Principle**: distance from last keep → distance from current code.

- Recent keep → nearby: refine, tune, combine.
- Several discards → moderate: swap components, add features, try alternatives.
- Many discards → significant: different algorithms, new modules, other fields.
- Deep in discards → fundamental: challenge core assumptions, try the unlikely.

When a radical hypothesis gets a keep, zoom back in to refine.

## Pattern Observation

After each experiment, read experiment_feedback.md. Look for:
- **Directional signals**: Metric budged even slightly? Promising direction.
- **Correlated failures**: Different changes failing the same way → structural bottleneck.
- **Sub-metric patterns**: Which components improve vs which are stuck?
- **Context changes after keeps**: A major keep means previously failed ideas
  deserve re-evaluation. Flag these in the queue.

## Research Sources

- Fetch arXiv papers relevant to the domain
- Read competition submissions / records if available
- Study the target file — what ISN'T being used that could be?
- Analogies from other fields

## Your Loop

LOOP FOREVER:
1. Read experiment_feedback.md and results.tsv.
2. Count discards since last keep. Calibrate radicality.
3. Check: did a recent major keep change the context? If so, flag
   previously discarded ideas for retry.
4. Generate 3-5 hypotheses at appropriate radicality.
5. Research each: papers, precedent, mechanism of improvement.
6. Write to research_queue.md:
   ```
   # Research Queue
   # Updated: <timestamp>
   # Discards since last keep: <N>

   ## Next Up
   ### <Hypothesis Name>
   - Reasoning: <why this might improve the metric>
   - Changes: <specific code changes>
   - Radicality: nearby | moderate | significant | fundamental
   - Mechanism: <change X → affects Y → improves metric because Z>
   - Category: <which part of code this touches — for diversity check>

   ## Retries (previously failed, context has changed)
   ### <Hypothesis Name> (retry)
   - Original result: <what happened before>
   - Why retry: <what changed in the codebase since>
   ...

   ## Queue
   ...

   ## Rejected
   ...
   ```
7. Sleep 2 minutes, repeat.
```

### File 3: `prepare.py` — Immutable Environment Validation

**Never modified during experiments.** The trusted anchor.

Must validate:
1. Runtime/language version, packages
2. Required files exist
3. Data present and valid
4. End-to-end smoke test: run eval, verify metric parseable
5. Quick screen baseline (if applicable): run `QUICK=1 bash eval.sh`,
   save to `quick_baseline.txt`
6. File integrity checksums of all immutable files → `file_checksums.txt`
7. Git configured
8. results.tsv initialized
9. .gitignore updated
10. Shared files created (research_queue.md seeded with initial hypotheses,
    experiment_feedback.md empty)

### File 4: `eval.sh` — Evaluation Wrapper (Immutable)

- Supports `QUICK=1` for 1-minute shortened eval (if applicable)
- Timeout on all subprocess calls
- Seed rotation: derive from git commit hash
- Stages labeled in output
- Final line: `composite_score: <number>` or `quick_score: <number>`

### File 5: `compute_score.py` (when composite metric needed, immutable)

Built from ACTUAL observed eval output. Defensive parsing. Prints all sub-metrics.

### File 6: `summarize_results.py`

Reads results.tsv, computes:
- Experiments per hour (throughput)
- Keep rate overall and per category
- Structural vs tuning experiment ratio
- Consecutive discards in current direction
- Directions exhausted vs promising
- Suggestions for next area to explore

### File 7: `.gitignore`

```
results.tsv
*.log
eval_output/
quick_baseline.txt
file_checksums.txt
experiment_feedback.md
research_queue.md
researcher.log
__pycache__/
```

### File 8: `pyproject.toml` (if project uses uv/pip)

### File 9: Domain-specific eval scripts (if proposed and confirmed, immutable)

---

## Phase 3: End-to-End Validation

1. `python prepare.py` — passes? Baseline saved? Checksums stored?
2. `bash eval.sh` — produces score?
3. `QUICK=1 bash eval.sh` — works? (if applicable)
4. Metric grep works?
5. Git clean?

Fix failures before presenting.

---

## Phase 4: Launch

```
Setup complete.

  program.md              — Your control surface. Edit to steer research.
  [target_file]           — Agent modifies this.
  prepare.py              — Immutable. Evaluation anchor.
  eval.sh                 — Immutable. Evaluation pipeline.
  [If composite]          compute_score.py
  [If slow experiments]   researcher_program.md
  summarize_results.py    — Experiment summarizer.

Metric: [description]
Eval: [full time], Quick screen: [1 min vs last keep]
Overhead budget: [max planning time between experiments]

To start:
  tmux
  claude
  > "Read program.md and start."

To steer: edit program.md's Research Directions.
```

---

## Core Principles

1. **One agent, one file, one metric, one loop.** The Karpathy contract.
2. **program.md is the human's control surface.** One file to steer everything.
3. **prepare.py is immutable.** The trusted evaluation anchor.
4. **The agent thinks AND executes.** Never a dumb queue executor.
5. **Maximize experiments per hour.** Fast discards > slow deliberation.
   The overhead budget caps planning time proportional to eval time.
6. **Structural changes > parameter tuning.** Single structural discoveries
   produce 10x more improvement than all tuning combined. Maintain at
   least a 2:1 ratio of structural to tuning experiments.
7. **Diversity is mandatory.** Max 2 consecutive experiments in the same
   category. Forces exploration breadth.
8. **Re-evaluate after major keeps.** When the codebase changes significantly,
   previously failed ideas deserve a retry in the new context.
9. **Dead time is research time.** During slow experiments, the agent (or
   background Researcher) reads papers, studies submissions, plans ahead.
   Never idle during training.
10. **Hypotheses escalate with failure.** Distance from keep → distance
    from current code. Continuous gradient, not fixed schedule.
11. **Hypotheses are domain-agnostic.** "What change improves this number?"
12. **Investigate, don't interrogate.** Read codebase, propose plan.
13. **Metric design is 80% of the work.**
14. **Propose domain-standard enhancements.** Be investigative.
15. **Exploit cheap computation** for robustness testing.
16. **Quick screen compares to last keep's** quick score.
17. **Protect immutable files.** Agent never modifies eval or data.
18. **Prevent wasted experiments.** Deduplication, summarization.
19. **Timeout everything.**
20. **Seeds rotate** (derive from git commit hash).
21. **Be specific.** Reference actual code: lines, variables, values.
22. **Simplicity wins.** Minimum scaffold, maximum autonomy.
