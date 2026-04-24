OK, decision phase is over. Drafting now.

## Decisions I'm committing to (with reasons)

**Folder structure with setup scripts.** You're right that scripts are cheap. I was wrong to swing toward Karpathy-pure. The folder structure gives us inspectability across sessions and makes "rerun any old experiment" trivial. Setup scripts make the boilerplate disappear.

**Journal: chronological with internal entry structure.** I'm picking this over the categorized version. Reasons: (a) categories require active migration which is unmaintained burden, (b) reverse-chronological matches how we actually read research notes, (c) the structural prompts (Question / Prediction / Disconfirming / Result) inside each entry give us the "force thinking" benefit without the file-level scaffolding. The "Current threads" section at top compensates for losing categorical scan.

**Selective journaling, not per-experiment.** Every experiment hits results.tsv. Only experiments where there's a *hypothesis worth remembering* get a journal entry. Not every LR sweep deserves prose. Roughly 1 in 4 experiments by my guess.

**Plan.md per experiment, not in journal.** Forward-looking plan goes in the experiment folder. Journal entries (when written) are post-hoc, distilled. Different artifacts, different jobs.

---

## v2 program.md draft

```markdown
# program.md — Parameter Golf Autoresearch

You are an autonomous research agent iterating on `train_gpt.py` to minimize validation bits-per-byte (`val_bpb`) under a 16 MB artifact constraint. You run on a Mac with MPS locally. The final score is evaluated separately on 8×H100s by a human. Your job is **directional exploration** — discover which changes help on a 200-step MPS smoke test, so the human can later validate top candidates on H100.

## Reference baseline

- val_bpb (post-quantization int8+zlib): **2.5540**
- Artifact size: 6.9 MB / 16 MB (~9 MB headroom available)
- Step time: ~1.25 s/step on MPS → ~5 min per experiment
- Source: experiments/0001_baseline/run.log (after first run) or logs/887989ba-3dec-49be-84f6-d6cb5643837d.txt

## Setup (do this every session)

1. Read this file in full.
2. Read journal.md — Current threads first, then recent entries newest-first.
3. Skim results.tsv to see what's been tried recently.
4. Check git status and `git log --oneline -10` to see canonical state.
5. If experiments/0001_baseline/ doesn't exist, run baseline first as exp 0001 with no changes. Should match val_bpb 2.5540 ± 0.01.
6. Confirm you understand the experiment loop before starting.

## Hard rules

**You CAN:**
- Edit train_gpt.py *inside an experiment folder*
- Set environment variables in the experiment folder's env.sh
- Read any file in the repo
- Create files in scratch/ (gitignored, ephemeral)
- Fetch papers by arxiv ID via curl, only when explicitly named in PAPERS.md or referenced as `[REF: arxiv:XXXX]` in journal/plan

**You CANNOT:**
- Modify the canonical train_gpt.py at the repo root (only the human promotes wins)
- Modify data/, records/, train_gpt_mlx.py, requirements.txt, pyproject.toml, .envrc
- Modify the eval harness inside train_gpt.py (eval_val, build_sentencepiece_luts, quantization functions)
- Install new packages
- Trigger RunPod or H100 runs
- Browse the web freely
- Commit to master/main; experiments are committed only when promoted by the human

## The experiment loop

For each experiment:

1. **Plan**: from repo root, run `./new_experiment.sh <slug>`. This creates experiments/NNNN_slug/ forked from the most recent experiment, with plan.md and env.sh templates.
2. **Fill plan.md** with hypothesis, change, prediction, disconfirming criterion. Be specific.
3. **Edit** experiments/NNNN_slug/train_gpt.py and/or env.sh. Prefer env vars when the change is purely hyperparameter. If a code change is non-trivial (>20 lines or multiple functions), use the subagent path (see below).
4. **Run**: `cd experiments/NNNN_slug && ../../run_experiment.sh`. Output → run.log. Metrics auto-populate result.json. One row appends to ../../results.tsv with TODO description.
5. **Review** the printed summary. Check val_bpb (post), quant_tax, size_violation, crashed.
6. **Decide** keep / discard / parked / crash. Fill `status` and `description` in results.tsv (last two columns). If keep, the next experiment forks from this one. If discard, just move on.
7. **Journal (selective)**: if the experiment had a hypothesis worth remembering, append an entry to journal.md. Skip for routine tuning.
8. **Update Current threads** in journal.md only at meaningful transitions, not after every experiment.
9. **Repeat**.

Do not git-commit experiment folders. The human reviews and promotes wins to canonical periodically.

### Extended smoke (1000 steps)

Default smoke is 200 steps (5 min). Some hypotheses (e.g., depth recurrence, weight-sharing schemes) need longer to show signal. Set `ITERATIONS=1000 WARMDOWN_ITERS=200` in env.sh. **Justify the extended budget in plan.md** — generic "more data = more signal" is not enough. The hypothesis must specifically predict that 200 steps would mis-rank.

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

**Measurement over belief.** Variance/init/FLOPs claims must be derived in scratch/, not asserted. Use Python to compute the actual numbers — operations on small tensors, parameter counts, gradient norms. Whether a thing trains better is empirical; whether the math says it should is computable. Don't conflate them.

**Critical reading of prior journal entries.** Treat `[CONJECTURE]` as a hypothesis to verify or work around, never a fact to build on. If your current direction is built on a chain of conjectures, pause and verify the base of the stack before extending.

**Hedge mechanism, not action.** The norm in transformer research is "we observe X; one possible mechanism is Y." Match that — try things, log results, but don't assert mechanisms confidently. Don't refuse to act because nothing is fully proven; do act, then label the why with appropriate confidence.

**Empirical vs verifiable in transformers**:
- Verifiable (do these whenever relevant): parameter count, FLOPs, init variance, shape tracing, mathematical equivalence between two formulations, numerical stability of specific ops
- Empirical (no substitute): whether a technique improves loss, optimal hyperparameters, interaction effects, long-horizon dynamics

When in doubt, do the math first.

## Logging formats

### results.tsv (auto-populated except status, description)

```
id  parent  val_bpb  pre_quant_bpb  quant_tax  artifact_mb  step_avg_ms  crashed  size_violation  status  description
```

You fill in the last two:
- `status`: keep / discard / parked / crash
- `description`: 6-10 word summary

### journal.md format

Top-of-file structure:

```markdown
# Journal

## Current threads
- <ongoing direction or open question, 1 line each>
- <kept short, 5-10 bullets max, updated at transitions only>

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

### Noise floor (200-step smoke)

- Δ ≥ +0.010 improvement → likely real, advance
- Δ ∈ [-0.005, +0.010] → noise, discard
- Δ ≤ -0.010 → clear loss, discard
- Δ ≥ +0.050 → suspiciously large, re-run with SEED=42 before advancing
- For Δ ∈ [+0.005, +0.010]: judgment call. If mechanistically composes with current direction, advance. If orthogonal, safer to discard or re-run.

## Soft constraints (warn, don't block)

**Artifact size > 16 MB.** Submission won't be valid as-is, but the idea may still be informative for what to build next. Flag `size_violation:true`, log normally, mention in journal, and make a "submittable best" note distinguishing this from your overall best.

**Quantization tax > 0.010.** Pre-quant val_bpb improves but post-quant doesn't follow. Note it in journal — some changes are quantization-fragile and would need QAT (Quantization-Aware Training) to be useful. Worth flagging for future structural exploration.

## Regression sentinel

Every 10 experiments, run a clean baseline experiment (slug `regression_check_NNN`). Use canonical settings, no env-var changes. Record val_bpb in results.tsv with status=`sentinel`.

If the sentinel drifts >0.02 from the original baseline 2.5540, log a `regression_detected:true` note in journal and continue. Probable causes: thermal throttling, MPS state, other processes. Snapshot for context (in journal entry):

```bash
ps aux | head -20 | tee scratch/regression_NNN_ps.txt
sysctl -n machdep.cpu.thermal_level >> scratch/regression_NNN_ps.txt
vm_stat >> scratch/regression_NNN_ps.txt
```

Future sessions reading sentinel rows treat surrounding experiments as suspect.

## Crash handling

If a run crashes (val_bpb empty, traceback in run.log):
1. Read `tail -n 50 run.log`.
2. If it's a typo / missing import / obviously fixable bug from your edit, fix it and rerun. Treat as one experiment.
3. If the idea itself is fundamentally broken (e.g., shape mismatch from architectural change you don't fully understand), set status=`crash` in results.tsv with a brief description, and move on.
4. Do not retry the same broken config more than twice.

## H100-transfer flag

When marking a win as `keep`, append a 1-line transfer-confidence note in the description column:
- `[transfer:high]` — change is robust scaling/architectural simplification, expect to hold at 20k steps
- `[transfer:med]` — change tunes hyperparameters, transfer depends on training-length dynamics
- `[transfer:low]` — change exploits early-training behavior, may not survive longer schedules

This helps the human pick winners for RunPod validation.

## Subagent for code edits

For non-trivial code changes (>20 lines, multiple functions touched, anything you'd struggle to keep in working memory):

1. Write a complete plan.md in the experiment folder. The plan must specify exact functions to modify, expected diff in pseudocode, and any test cases.
2. Spawn subagent with prompt: "Read experiments/NNNN_slug/plan.md. Implement the change in experiments/NNNN_slug/train_gpt.py. Update plan.md with notes about what was done and any deviations. Do not run experiments. Return a one-paragraph summary."
3. Subagent does the edit, updates plan.md in place, returns summary.
4. You review the diff and updated plan.md. If small tweak needed, do it yourself. If significant rework, write a new plan and spawn a new subagent. One-shot per plan.

Subagent never runs experiments — that's your job. Subagent never modifies anything outside the named experiment folder.

## Reference materials (optional, browse selectively)

- TECHNIQUES_INDEX.md: one-line summary of each leaderboard record under records/. Use as an idea pool. Read the full record's README for technique details when interested. **Do not copy code** — only categories of techniques. Plagiarism defeats the point and won't be merged.
- PAPERS.md: curated list of relevant papers with arxiv IDs. Read when grounding a hypothesis. Fetch with `curl https://arxiv.org/pdf/<id>` when needed.

## NEVER STOP

Once the loop has begun, do not pause to check in with the human. Do not ask "should I continue?" or "is this a good stopping point?". The human may be asleep or away, and expects you to continue indefinitely until manually stopped.

If you run out of ideas:
1. Re-read recent journal entries for unresolved threads
2. Re-read TECHNIQUES_INDEX.md for techniques not yet tried
3. Re-read PAPERS.md
4. Try combining recent near-misses
5. Try more radical architectural changes you previously parked
6. Re-derive parameter/FLOPs math for the current canonical to find inefficiencies

Each experiment is ~5 min. Overnight = 80-100 experiments. Even a 1-in-5 hit rate is significant progress. Keep going.

## When the human returns

Finish the current experiment cleanly (don't leave half-written plan.md or unrun folder). Resuming next session is trivial: read journal, results.tsv, current threads, continue.
```

---

## Setup script: `new_experiment.sh`

Drop this at repo root, `chmod +x`.

```bash
#!/usr/bin/env bash
# Usage: ./new_experiment.sh <slug>
# Creates experiments/NNNN_<slug>/, forked from the most recent experiment.

set -euo pipefail

SLUG="${1:-}"
if [[ -z "$SLUG" ]]; then
  echo "Usage: $0 <slug>" >&2
  exit 1
fi
if ! [[ "$SLUG" =~ ^[a-z0-9_-]+$ ]]; then
  echo "Slug must be lowercase alphanumeric with - or _" >&2
  exit 1
fi

mkdir -p experiments

# Find most recent experiment (last numerical id)
LAST=""
if compgen -G "experiments/[0-9][0-9][0-9][0-9]_*" >/dev/null; then
  LAST=$(ls -d experiments/[0-9][0-9][0-9][0-9]_* 2>/dev/null | sort | tail -1)
  LAST=$(basename "$LAST")
fi

NEXT_ID=1
if [[ -n "$LAST" ]]; then
  NEXT_ID=$((10#${LAST:0:4} + 1))
fi
ID=$(printf "%04d" "$NEXT_ID")
NAME="${ID}_${SLUG}"
DIR="experiments/${NAME}"

# Source: parent experiment's train_gpt.py if exists, else canonical
if [[ -n "$LAST" && -f "experiments/${LAST}/train_gpt.py" ]]; then
  SOURCE_FILE="experiments/${LAST}/train_gpt.py"
  PARENT="$LAST"
else
  SOURCE_FILE="train_gpt.py"
  PARENT="canonical"
fi

mkdir "$DIR"
cp "$SOURCE_FILE" "$DIR/train_gpt.py"

cat > "$DIR/plan.md" <<EOF
# Experiment ${NAME}

Parent: ${PARENT}

## Question
<!-- What are you actually asking? Be specific. -->

## Hypothesis [CONJECTURE]
<!-- Predicted direction and magnitude of val_bpb change, with confidence tag. -->

## Change
<!-- Exact env vars / code edits. -->

## Disconfirming
<!-- What outcome would falsify the hypothesis? -->

## Notes from execution
<!-- Filled by you during edit, or subagent if invoked. Note any deviations. -->
EOF

cat > "$DIR/env.sh" <<EOF
# Source this from inside the experiment folder before running.
export RUN_ID="${NAME}"
export DATA_PATH="../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export ITERATIONS=200
export WARMUP_STEPS=0
export WARMDOWN_ITERS=40
export MAX_WALLCLOCK_SECONDS=600
export TRAIN_BATCH_TOKENS=8192
export TRAIN_SEQ_LEN=1024
export VAL_BATCH_SIZE=8192
export VAL_LOSS_EVERY=0
export VAL_TOKENS=16384
# Experiment-specific overrides go below:
EOF

cat > "$DIR/result.json" <<EOF
{
  "id": "${NAME}",
  "parent": "${PARENT}",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "metrics": null,
  "flags": {},
  "status": null,
  "description": null
}
EOF

echo "Created $DIR (parent: $PARENT)"
echo ""
echo "Next steps:"
echo "  1. Edit $DIR/plan.md  (fill hypothesis, change, disconfirming)"
echo "  2. Edit $DIR/train_gpt.py and/or $DIR/env.sh"
echo "  3. cd $DIR && ../../run_experiment.sh"
```

---

## Run script: `run_experiment.sh`

Also at repo root, `chmod +x`. Runs from inside an experiment folder.

```bash
#!/usr/bin/env bash
# Usage: cd experiments/NNNN_slug && ../../run_experiment.sh
# Runs the experiment, parses metrics, writes result.json and appends results.tsv.

set -euo pipefail

if [[ ! -f result.json || ! -f train_gpt.py || ! -f env.sh ]]; then
  echo "Error: must be run from inside experiments/NNNN_slug/" >&2
  exit 1
fi

EXPERIMENT_ID=$(basename "$(pwd)")
REPO_ROOT="$(cd ../.. && pwd)"

# Source env vars
source env.sh

echo "Running $EXPERIMENT_ID..."
set +e
python train_gpt.py > run.log 2>&1
RUN_RC=$?
set -e

# Parse metrics
extract() { grep -oE "$1" run.log | tail -1 | grep -oE "$2" | cut -d: -f2 || echo "null"; }

VAL_BPB_PRE=$(grep -oE 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+' run.log | tail -1 | grep -oE 'val_bpb:[0-9.]+' | cut -d: -f2 || echo "")
VAL_BPB_POST=$(grep -oE 'final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:[0-9.]+' run.log | grep -oE 'val_bpb:[0-9.]+' | cut -d: -f2 || echo "")
STEP_AVG_MS=$(grep -oE 'step:[0-9]+/[0-9]+ train_loss:[0-9.]+ train_time:[0-9]+ms step_avg:[0-9.]+ms' run.log | tail -1 | grep -oE 'step_avg:[0-9.]+' | cut -d: -f2 || echo "")
ARTIFACT_BYTES=$(grep -oE 'Total submission size int8\+zlib: [0-9]+ bytes' run.log | grep -oE '[0-9]+' | head -1 || echo "")
NUM_STEPS=$(grep -oE 'step:[0-9]+/[0-9]+ train_loss:' run.log | tail -1 | grep -oE 'step:[0-9]+' | cut -d: -f2 || echo "")

CRASHED="false"
if [[ -z "$VAL_BPB_POST" || $RUN_RC -ne 0 ]]; then
  CRASHED="true"
fi

QUANT_TAX=""
if [[ -n "$VAL_BPB_PRE" && -n "$VAL_BPB_POST" ]]; then
  QUANT_TAX=$(python3 -c "print(round(${VAL_BPB_POST} - ${VAL_BPB_PRE}, 6))")
fi

SIZE_VIOLATION="false"
ARTIFACT_MB=""
if [[ -n "$ARTIFACT_BYTES" ]]; then
  ARTIFACT_MB=$(python3 -c "print(round(${ARTIFACT_BYTES} / 1048576, 3))")
  if (( ARTIFACT_BYTES > 16000000 )); then
    SIZE_VIOLATION="true"
  fi
fi

# Update result.json
python3 - <<PYEOF
import json
with open("result.json") as f:
    r = json.load(f)
def n(s): return float(s) if s else None
r["metrics"] = {
    "val_bpb_pre_quant": n("${VAL_BPB_PRE}"),
    "val_bpb_post_quant": n("${VAL_BPB_POST}"),
    "quant_tax": n("${QUANT_TAX}"),
    "step_avg_ms": n("${STEP_AVG_MS}"),
    "num_steps": n("${NUM_STEPS}"),
    "artifact_bytes": n("${ARTIFACT_BYTES}"),
    "artifact_mb": n("${ARTIFACT_MB}"),
}
r["flags"] = {
    "crashed": ${CRASHED^},
    "size_violation": ${SIZE_VIOLATION^},
}
with open("result.json", "w") as f:
    json.dump(r, f, indent=2)
PYEOF

# Append to global results.tsv
TSV="${REPO_ROOT}/results.tsv"
if [[ ! -f "$TSV" ]]; then
  printf "id\tparent\tval_bpb\tpre_quant_bpb\tquant_tax\tartifact_mb\tstep_avg_ms\tcrashed\tsize_violation\tstatus\tdescription\n" > "$TSV"
fi

PARENT=$(python3 -c "import json; print(json.load(open('result.json'))['parent'])")
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tTODO\tTODO\n" \
  "$EXPERIMENT_ID" "$PARENT" "${VAL_BPB_POST:-null}" "${VAL_BPB_PRE:-null}" "${QUANT_TAX:-null}" "${ARTIFACT_MB:-null}" "${STEP_AVG_MS:-null}" "$CRASHED" "$SIZE_VIOLATION" \
  >> "$TSV"

# Print summary
cat <<EOF

=== Experiment $EXPERIMENT_ID ===
  val_bpb_post_quant: ${VAL_BPB_POST:-null}
  val_bpb_pre_quant:  ${VAL_BPB_PRE:-null}
  quant_tax:          ${QUANT_TAX:-null}
  step_avg_ms:        ${STEP_AVG_MS:-null}
  num_steps:          ${NUM_STEPS:-null}
  artifact_mb:        ${ARTIFACT_MB:-null}
  crashed:            $CRASHED
  size_violation:     $SIZE_VIOLATION

Now: review run.log if needed, fill status+description in ${REPO_ROOT}/results.tsv, optionally write journal entry.
EOF
```

---

## What's needed before this goes live

1. **Update `.gitignore`** to add: `experiments/*/run.log`, `experiments/*/final_model.pt`, `experiments/*/final_model.int8.ptz`, `experiments/*/logs/`, `scratch/`. Tony or Claude Code applies.
2. **Stub TECHNIQUES_INDEX.md and PAPERS.md.** Don't need to be complete on day 1 — even one-line stubs with "agent: skim records/ for ideas" / "agent: ask human if a paper is needed" will work. We can flesh out later.
3. **First run is exp 0001 baseline.** Nothing in env.sh changed from defaults, no train_gpt.py edit. Confirms baseline reproduces at 2.554 in the new harness. This is the agent's first action.
4. **Test the scripts before kicking off overnight.** The bash parsing is brittle — first run will probably need a small fix. Worth doing one full cycle in front of you (or with Claude Code) before walking away.

## Open questions for you

1. **Should journal.md and results.tsv start with stubs, or empty?** I'd lean: start journal.md with the structure (Current threads + Entries header) but no content. Start results.tsv empty (run script creates header on first append).

2. **TECHNIQUES_INDEX.md and PAPERS.md — do you want me to draft those next, or are stub versions enough to launch?**

3. **The scripts — do you want me to dry-run the parsing logic against the existing log file (logs/887989ba-3dec-49be-84f6-d6cb5643837d.txt) to confirm the regexes work before you hand off to Claude Code?** That would catch the obvious bugs.

What to fix in the program.md before you hand it to Claude Code?

[user message skipped]

Got it. Pulling back on the bash, drafting the indices carefully, dry-running the parsing.

## Dry-run the parsing first

Let me grab the actual log lines I need to parse and verify the extraction logic works.Parsing works clean against the real log. Two small findings:

**The artifact unit matters.** The challenge defines 16 MB as **16,000,000 bytes (decimal)**, not 16 MiB (16,777,216 bytes). My calculation showed 6.586 MiB but it's actually 6.906 MB (decimal). The size_violation check must use 16,000,000 — getting this wrong silently lets a sub-MiB-violating-but-MB-violating submission through. Spec the script to use decimal MB.

**No need to modify train_gpt.py output.** It parses cleanly already (I have it down to ~10 regex lines). Don't add complexity there.

## Script specs (Claude Code implements)

I shouldn't have written full bash. Here's what each script needs to do, in plain spec form. Claude Code can implement.

### `new_experiment.sh <slug>`

Inputs: one positional arg, lowercase alphanumeric slug with `-` or `_`.

Behavior:
1. Find max `NNNN` in `experiments/[0-9][0-9][0-9][0-9]_*` directories. Increment.
2. Determine parent: most recent experiment folder. If none exists, parent is `canonical`.
3. Create `experiments/NNNN_<slug>/`.
4. Copy parent's `train_gpt.py` (or root canonical if parent is `canonical`) into the new folder.
5. Drop a `plan.md` template with sections: Question / Hypothesis [CONJECTURE] / Change / Disconfirming / Notes from execution.
6. Drop an `env.sh` template with the standard smoke-test env vars (`ITERATIONS=200`, etc.) and a comment block "Experiment-specific overrides go below."
7. Drop a stub `result.json` with id, parent, created_at (ISO UTC), and null metrics/flags/status/description.
8. Print: created path, parent, "next steps: edit plan.md, edit code/env, then `cd experiments/NNNN_<slug> && ../../run_experiment.sh`".

Failure modes: invalid slug, slug already used at any NNNN, no parent and no canonical at root.

### `run_experiment.sh`

Run from inside an experiment folder (validates this).

Behavior:
1. `source env.sh`. Validate that required env vars are set.
2. `python train_gpt.py > run.log 2>&1`. Capture exit code; do not let crash kill the script.
3. Parse `run.log`. Extract (use regex on log lines):
   - `val_bpb_pre_quant`: from last `step:N/N val_loss:X val_bpb:Y` line
   - `val_loss_pre_quant`: same line
   - `val_bpb_post_quant`: from `final_int8_zlib_roundtrip_exact ... val_bpb:Y` (8-digit), fall back to `final_int8_zlib_roundtrip val_bpb:Y` (4-digit)
   - `val_loss_post_quant`: same line
   - `step_avg_ms`: from last `step:N/N train_loss:... step_avg:Yms`
   - `num_steps`: same line, the N
   - `artifact_bytes`: from `Total submission size int8+zlib: N bytes`
   - `code_bytes`: from `Code size: N bytes`
   - `compression_ratio`: from `payload_ratio:X.XXx`
4. Derive: `quant_tax = post - pre` (round 6); `artifact_mb = bytes / 1_000_000` (decimal!); `size_violation = bytes > 16_000_000`; `crashed = (post is None or exit_code != 0)`.
5. Update `result.json` with metrics + flags. Leave `status` and `description` null.
6. Append one row to repo-root `results.tsv`. Create file with header if missing. Columns:
   `id  parent  val_bpb  pre_quant_bpb  quant_tax  artifact_mb  step_avg_ms  crashed  size_violation  status  description`
   Last two columns: literal `TODO` (so the agent sees them in `grep TODO results.tsv`).
7. Print compact summary block to stdout (one line per metric, like the dry-run output above).

Failure modes: not in an experiment folder, env.sh missing, parsing produces no val_bpb (still write row with crashed=true and what was parseable).

That's the spec. Claude Code writes it; I'd estimate 80-100 lines of bash + a small Python block for JSON updates.

## Baseline pre-population

Yes — agree. results.tsv shouldn't be empty when the agent first reads it. It needs the reference number from the start. Approach:

Initialize `results.tsv` (manually, or with a tiny `init.sh` Claude Code writes) with two rows:

```
id                  parent       val_bpb   pre_quant_bpb  quant_tax  artifact_mb  step_avg_ms  crashed  size_violation  status      description
0000_reference      -            2.5540    2.5485         0.0055     6.906        1244.22      false    false           reference   Pre-existing run logs/887989ba
```

Agent's first action is still to create exp 0001 and run baseline through the new harness. Exp 0001 should land at val_bpb ≈ 2.554 ± 0.01. If it drifts more than that, the new folder-based harness has a bug and the agent must investigate before continuing.

This makes the harness validation explicit instead of assumed.

## TECHNIQUES_INDEX.md draft

Pure navigable index. No summaries by me — pulled directly from the leaderboard text in records. Agent reads the linked README for actual technique detail.

```markdown
# Techniques Index

Each leaderboard submission is in `records/track_10min_16mb/<date>_<slug>/`. Read the README for the technique. **Do not copy code** — only learn categories of techniques. This file is your idea menu, not a substitute for reading the source.

## How to use this file

- Filter by score range to see what's working at different levels
- Filter by tag to find prior work in a technique family
- Open the linked README. Read the writeup. Then close it before designing your experiment.

## Records (sorted by date, newest first)

| Score  | Date       | Slug                                                                                  | Tags                                                       |
| ------ | ---------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1.0810 | 2026-04-09 | 2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT                                 | sp8192, 3-layer-recur, parallel-resid, qk-gain, ttt        |
| 1.0822 | 2026-04-08 | 2026-04-08_SP8192_ParallelResid_ScoreFirstTTT                                         | sp8192, parallel-resid, ttt                                |
| 1.0828 | 2026-04-06 | 2026-04-06_SP8192_QK5_LegalTTT_1.0828                                                 | sp8192, qk-gain, ttt                                       |
| 1.0835 | 2026-04-06 | 2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence                                 | sp8192, hessian-clip, progressive-recur                    |
| 1.0856 | 2026-04-05 | 2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2                                     | sp8192, gptq, depth-recur, sdclip                          |
| 1.0897 | 2026-04-04 | 2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR                               | sp4096, depth-recur, parallel-resid, muon-eqr, qk-gain     |
| 1.0912 | 2026-04-03 | 2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6                                      | muon-eqr, depth-recur, wd, int6                            |
| 1.0979 | 2026-04-01 | 2026-04-01_Vocab4096_MLPMult4_WD085                                                   | sp4096, mlp4x, wd                                          |
| 1.1063 | 2026-03-31 | 2026-03-31_ParallelResiduals_MiniDepthRecurrence                                      | parallel-resid, depth-recur, gptq                          |
| 1.1147 | 2026-03-25 | 2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072                                           | gptq, xsa, bigram-hash                                     |
| 1.1194 | 2026-03-23 | 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon                                            | leakyrelu², ttt, parallel-muon                             |
| 1.1228 | 2026-03-22 | 2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233                               | 11L, ema, gptq, qat                                        |
| 1.1248 | 2026-03-21 | 2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248                                    | 11L, xsa, ema, partial-rope, qat                           |
| 1.1271 | 2026-03-20 | 2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271                                        | 11L, xsa, ema, int6, mlp3x, wd                             |
| 1.1307 | 2026-03-20 | 2026-03-20_11L_EfficientPartialXSA_FA3_SWA120                                         | 11L, partial-xsa, swa                                      |
| 1.1428 | 2026-03-20 | 2026-03-20_10L_Int5MLP_MuonWD04_SWA50                                                 | 10L, int5, muon-wd, swa, bigram-hash                       |
| 1.1458 | 2026-03-20 | 2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA                                 | int6, mlp3x, smear-gate, bigram-hash, muon-wd              |
| 1.1502 | 2026-03-20 | 2026-03-19_MLP3x_QAT_Int6_SlidingWindow                                               | 11L, mlp3x, qat, int6, sliding-window                      |
| 1.1556 | 2026-03-19 | 2026-03-19_smeargate_orthoinit_muonwd                                                 | smear-gate, ortho-init, muon-wd, bigram-hash, mlp3x, qat   |
| 1.1570 | 2026-03-24 | 2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon                              | ternary-quant, 10L, unet, fp8, sp8192, yarn, neomuon       |
| 1.1586 | 2026-03-19 | 2026-03-19_Seq2048_FP16Emb_TunedLR                                                    | 10L, int6, qat, mlp2.6x, sliding-window                    |
| 1.1630 | 2026-03-19 | 2026-03-19_MixedQuant_Int6Int8_SlidingWindow                                          | int6, int8, mlp3x, sliding-window                          |
| 1.1748 | 2026-03-19 | 2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit                              | sliding-window, fp16-emb, 10L, muon-wd, ortho-init         |
| 1.1925 | 2026-03-19 | 2026-03-19_SlidingWindowEval                                                          | sliding-window-eval                                        |
| 1.1928 | 2026-03-19 | 2026-03-17_LoRA_TTT                                                                   | lora, ttt                                                  |
| 1.2014 | 2026-03-19 | 2026-03-19_TrainingOptSeq4096                                                         | seq4096                                                    |
| 1.2060 | 2026-03-18 | 2026-03-18_LongContextSeq2048                                                         | seq2048                                                    |
| 1.2147 | 2026-03-18 | 2026-03-19_10L_MixedPrecision                                                         | 10L, int6, int8                                            |
| 1.2197 | 2026-03-18 | 2026-03-18_FP16Embed_WD3600                                                           | fp16-emb, wd                                               |
| 1.2244 | 2026-03-18 | 2026-03-17_NaiveBaseline                                                              | baseline, 9L, sp1024                                       |

## Non-record / unlimited-compute track

| Score  | Date       | Slug                                                                                                  | Tags                                              |
| ------ | ---------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 1.1239 | 2026-03-24 | 2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear (4hr training)              | 1bit-quant, asym, unet, fp8, 15L, sp8192, yarn    |
| 1.2074 | 2026-03-18 | 2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3 (4hr training)                                   | sp1024, 9L, 4h-baseline                           |

## Tag glossary (one line each)

These are anchors for searching. Not technique definitions — read the source.

- `sp1024` / `sp4096` / `sp8192` — SentencePiece tokenizer vocab size variant
- `9L` / `10L` / `11L` — number of unique transformer layers
- `mlp2x` / `mlp3x` / `mlp4x` — MLP hidden dim multiplier
- `depth-recur` / `3-layer-recur` / `progressive-recur` — depth recurrence variants (loop layers, sometimes increasing R during training)
- `parallel-resid` — parallel attention/MLP residual lanes (vs sequential)
- `qk-gain` / `qk-norm` — multiplicative gain on Q (and K) before SDPA
- `gptq` / `gptq-lite` / `gptq-embeddings` — post-training int quantization with calibration data
- `qat` / `qat015` / `late-qat` — quantization-aware training, sometimes only late in training
- `int5` / `int6` / `int8` — quantization bit-width
- `ternary-quant` / `1bit-quant` — extreme low-bit quantization (often paired with mixed-precision)
- `ema` — exponential moving average of weights for the eval checkpoint
- `xsa` / `xsa4` / `partial-xsa` / `efficient-partial-xsa` — exponential / sparse / partial attention variants
- `swa` / `sliding-window` / `sliding-window-eval` — sliding window attention (in train and/or eval)
- `ttt` / `legal-ttt` / `score-first-ttt` / `lora-ttt` — test-time training, "legal" = only on already-evaluated tokens
- `muon-wd` / `muon-eqr` / `cautious-decay` — Muon optimizer variants (weight decay, equivariant, etc.)
- `parallel-muon` / `neomuon` — Muon kernel/timing variants
- `sdclip` / `hessian-clip` — gradient clipping schemes
- `partial-rope` / `yarn` — RoPE variants (rotate fewer dims, extended context)
- `unet` — U-Net style encoder-decoder skip structure (the baseline already has this; explicit label means stronger variant)
- `bigram-hash` — hashed bigram embedding to augment unigram tokenizer
- `smear-gate` / `relu²` / `leakyrelu²` — activation function variants
- `fp16-emb` / `fp8` — embedding/weight precision tweaks
- `wd` / `wd040` / `wd085` / `wd090` — weight decay tuning
- `ortho-init` / `overtone-init` — initialization schemes
- `warmdown3500` — extended LR warmdown
- `seq2048` / `seq4096` — longer training sequence length
```

## PAPERS.md draft

Tight. Curated. No summaries — categories and arxiv IDs only.

```markdown
# Papers

Reference list. Read selectively when grounding a hypothesis. Fetch with `curl https://arxiv.org/pdf/<id>` only when you specifically need it. **Do not browse the web freely.**

## Optimization

- Loshchilov & Hutter, AdamW — `arxiv:1711.05101`
- Keller Jordan et al., Muon optimizer — `https://kellerjordan.github.io/posts/muon/`
- Liu et al., Sophia — `arxiv:2305.14342`
- Zhao et al., Cautious Optimizers — `arxiv:2411.16085`

## Parameter efficiency / weight tying / sharing

- Press & Wolf, Tied embeddings — `arxiv:1608.05859`
- Lan et al., ALBERT (cross-layer sharing) — `arxiv:1909.11942`
- Dehghani et al., Universal Transformers — `arxiv:1807.03819`
- Bae et al., Relaxed Recursive Transformers — `arxiv:2410.20672`

## Architecture

- Su et al., RoPE — `arxiv:2104.09864`
- Ainslie et al., GQA — `arxiv:2305.13245`
- DeepSeek-AI, MLA in DeepSeek-V2 — `arxiv:2405.04434`
- Shazeer, GLU variants — `arxiv:2002.05202`
- Loshchilov et al., nGPT (normalized transformer) — `arxiv:2410.01131`
- Zhu et al., ResFormer / value residuals — `arxiv:2410.17897`

## Quantization

- Frantar et al., GPTQ — `arxiv:2210.17323`
- Esser et al., LSQ (learned step size QAT) — `arxiv:1902.08153`
- Bhalgat et al., LSQ+ — `arxiv:2004.09576`
- Ma et al., BitNet b1.58 (ternary) — `arxiv:2402.17764`
- Dettmers et al., LLM.int8() — `arxiv:2208.07339`

## Scaling laws / training dynamics

- Kaplan et al., Neural scaling laws — `arxiv:2001.08361`
- Hoffmann et al., Chinchilla — `arxiv:2203.15556`
- Bahri et al., Explaining scaling laws — `arxiv:2102.06701`

## Test-time compute / TTT

- Sun et al., Test-time training (general) — `arxiv:1909.13231`
- Akyürek et al., Surprising effectiveness of test-time training — `arxiv:2411.07279`

## Tokenization

- Kudo & Richardson, SentencePiece — `arxiv:1808.06226`
- Schmidt et al., H-Net (no tokenizer) — `arxiv:2509.18781` (verify ID before fetching; this one's recent and I'm less sure of it)

## Reference codebases (not papers, but read selectively)

- modded-nanogpt — github.com/KellerJordan/modded-nanogpt
- nanoGPT — github.com/karpathy/nanoGPT
```

A note on PAPERS.md: I'm 95%+ sure on the arxiv IDs in the first six sections. The H-Net one I flagged — I'm less sure about the ID, agent should verify before fetching. Honest hedge.

## What goes to Claude Code

To execute:
1. Create `program.md` (the v2 draft from previous turn)
2. Create the two scripts to spec (`new_experiment.sh`, `run_experiment.sh`)
3. Create `TECHNIQUES_INDEX.md`, `PAPERS.md` (drafts above — Tony reviews first)
4. Initialize `results.tsv` with header + the `0000_reference` row
5. Create empty `journal.md` with structure (Current threads section + Entries header)
6. Update `.gitignore` to exclude: `experiments/*/run.log`, `experiments/*/final_model*`, `experiments/*/logs/`, `scratch/`, `results.tsv` (Karpathy gitignores it; we follow)
7. Verify by running exp 0001 baseline through the new harness; confirm val_bpb ≈ 2.554

## Decisions left for you

- **PAPERS.md curation**: anything missing or anything you'd cut? I included optimization, parameter efficiency, architecture, quantization, scaling, TTT, tokenization. I deliberately did not include OpenAI/Anthropic foundation model papers — too general.
- **TECHNIQUES_INDEX.md tag glossary**: did I overdo it? Could be cut to half the tags if the table is enough. I lean keep it — first-time readers benefit.
- **Should `results.tsv` be gitignored or committed?** Karpathy gitignores. Ours arguably has more value to commit since it's the primary cross-session memory. Lean: gitignore for now (matches Karpathy), promote to committed if cross-session loss becomes a real problem.

[My response]
Everything looks good. Results can be in git.

## Handoff for Claude Code

```markdown
# Parameter Golf Autoresearch Setup — Implementation Handoff

You are setting up an autoresearch loop for the OpenAI Parameter Golf challenge. The design is finalized; this is an execution task. The repo is at `/Users/tonyliu/Desktop/projects/parameter-golf/`.

The full design lives in two prior messages copied into this conversation: the **v2 program.md draft** and the **script specs / TECHNIQUES_INDEX.md / PAPERS.md drafts**. Read both fully before starting. The design has been carefully discussed — do not redesign. Implement to spec.

## Files to create

1. **`program.md`** — copy the v2 draft verbatim into the repo root.

2. **`TECHNIQUES_INDEX.md`** — copy the draft verbatim into the repo root.

3. **`PAPERS.md`** — copy the draft verbatim into the repo root.

4. **`new_experiment.sh`** — implement to the script spec. `chmod +x`. Validate slug regex. Forks from most recent experiment if one exists, else from canonical root `train_gpt.py`.

5. **`run_experiment.sh`** — implement to the script spec. `chmod +x`. Must run from inside an experiment folder. Validates that. Captures `python train_gpt.py` output to `run.log` without letting crashes kill the script. Parses metrics via the regexes in the spec. Updates `result.json`. Appends one row to `results.tsv` with `TODO` literals in `status` and `description`.

   **Critical**: use **decimal MB (1,000,000)** for `artifact_mb`, not MiB. The 16 MB constraint is decimal. `size_violation = artifact_bytes > 16_000_000`.

   Use `final_int8_zlib_roundtrip_exact` (8-digit precision) as primary source for post-quant val_bpb; fall back to `final_int8_zlib_roundtrip` (4-digit) if the exact line is missing.

6. **`journal.md`** — initialize with this structure, no content:
   ```markdown
   # Journal

   ## Current threads
   <!-- Maintained as a loose TODO. Update only at meaningful milestones. -->

   ---

   ## Entries (newest first)
   ```

7. **`results.tsv`** — initialize committed with header row + the reference baseline row:
   ```
   id	parent	val_bpb	pre_quant_bpb	quant_tax	artifact_mb	step_avg_ms	crashed	size_violation	status	description
   0000_reference	-	2.5540	2.5485	0.0055	6.906	1244.22	false	false	reference	Pre-existing run from logs/887989ba; harness check
   ```
   (Tabs between columns, not spaces. Note `artifact_mb` is decimal: 6905876 / 1_000_000 = 6.906.)

8. **`.gitignore`** — add these lines (don't overwrite existing entries):
   ```
   experiments/*/run.log
   experiments/*/final_model.pt
   experiments/*/final_model.int8.ptz
   experiments/*/logs/
   scratch/
   ```
   Note: `results.tsv` is **committed**, not gitignored — diverges from Karpathy intentionally.

## Verification (do this before declaring done)

1. Run `./new_experiment.sh baseline_repro`. Confirm `experiments/0001_baseline_repro/` is created with `train_gpt.py`, `plan.md`, `env.sh`, `result.json`, with `parent: "canonical"` in result.json.

2. Don't edit anything. `cd experiments/0001_baseline_repro && ../../run_experiment.sh`. This is the harness self-test. ~5 min runtime. Should complete without crashing.

3. After it finishes, confirm:
   - `result.json` has populated `metrics` and `flags`
   - `val_bpb_post_quant` is approximately 2.554 (within ±0.01 of the reference)
   - One new row in `../../results.tsv` for `0001_baseline_repro` with `TODO` in status/description
   - `run.log` is in the experiment folder, not the repo root

4. If exp 0001 val_bpb drifts more than ±0.01 from the reference 2.5540, **stop and surface it to the human** — the harness has a bug or there's environmental drift. Do not proceed with further setup.

## Out of scope

- Do not start running experiments beyond the verification baseline. The agent loop kicks off in a separate session.
- Do not modify the canonical `train_gpt.py` at the root.
- Do not create stub experiments beyond 0001.
- Do not fill in TECHNIQUES_INDEX.md or PAPERS.md with extra content beyond what the drafts specify.

## When done

Report back:
- All files created (list)
- Both scripts pass `bash -n` (syntax check)
- exp 0001 result (val_bpb, drift from reference, timing)
- Any deviations from spec and why
```

That should be everything Claude Code needs. After it returns with verification, you and I can sanity-check the result and decide whether we're ready to launch the actual loop in a fresh session.
