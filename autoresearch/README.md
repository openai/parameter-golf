# Autoresearch

This directory contains the single Parameter Golf autoresearch controller program:

```bash
uv run python run_pgolf_experiment.py --hours 8
uv run python run_pgolf_experiment.py --forever
```

The controller is designed to keep the GPU busy while still maintaining traceability. It drafts reviewed patches in the background, queues only pre-approved candidates, runs them remotely, and records both the reasoning and the outcome.

**Requirements**

- Run the program from inside the `parameter-golf` git repository.
- The local git worktree must be clean before the controller starts.
- `codex` must be available in `PATH`.
- `REMOTE_HOST` must be set to your GPU box, for example `user@host`.
- The remote machine must already have the repo cloned and the dataset/tokenizer paths available.
- The remote repo should start clean before the controller begins applying and pushing experiment commits.
- `uv` is the expected runner for this subproject.

Common remote env vars:

- `REMOTE_HOST`: required SSH destination.
- `REMOTE_PORT`: defaults to `22`.
- `REMOTE_REPO_DIR`: defaults to `/workspace/parameter-golf`.
- `REMOTE_BRANCH`: defaults to `runpod-autoresearch`.
- `REMOTE_TORCHRUN`: defaults to `torchrun`.
- `REMOTE_IDENTITY`: optional SSH key path.
- `REMOTE_SSH_FORCE_TTY`: optional boolean override. The controller enables `ssh -tt`
  automatically for `*.runpod.io` hosts because that gateway expects a PTY.

Common experiment env vars:

- `DATA_PATH`
- `TOKENIZER_PATH`
- `VOCAB_SIZE`
- `ITERATIONS`
- `VAL_LOSS_EVERY`
- `MAX_WALLCLOCK_SECONDS`
- `NPROC_PER_NODE`

Controller-specific env vars:

- `PROPOSER_MODEL`
- `PRE_REVIEW_MODEL`
- `POST_REVIEW_MODEL`
- `MAX_PRE_REVIEW_ROUNDS`
- `PREP_QUEUE_DEPTH`
- `TRACE_ROOT`
- `RESULTS_FILE`
- `REVIEWS_FILE`
- `HARNESS_LOG`

**Program Structure**

The controller has three Codex roles:

- `proposer`: proposes exactly one bounded change to `train_gpt.py`, writes a rationale, and commits exactly one candidate patch in a scratch clone.
- `pre-reviewer`: reviews the patch and rationale before it reaches the GPU, and either `approve`s or requests `revise` with concrete feedback.
- `post-reviewer`: reviews the finished run, focuses mainly on metric quality and trustworthiness, and decides `keep` or `revert`.

The runtime loop is:

1. Draft a candidate from the last reviewed base commit.
2. Pre-review it for correctness and trustworthiness.
3. Repeat proposer plus pre-review for up to `MAX_PRE_REVIEW_ROUNDS`.
4. Queue only approved patches.
5. Apply the next queued patch with `git am --3way`.
6. Push it to the remote branch and run training.
7. Post-review the result.
8. Keep or revert the experiment commit.
9. Append the final result to `results.tsv` and `reviews.tsv`.
10. Commit the ledger update so the repo returns to a clean reviewed state.

Two details matter:

- The proposer is intentionally based on the last reviewed commit, not the currently running speculative experiment.
- The queue unit is a reviewed patch, not a raw idea.

**Artifacts And Traceability**

The controller writes durable artifacts under `TRACE_ROOT`, which defaults to:

```text
controller_state/autoresearch/
```

Important paths:

- `history/ledger.jsonl`: append-only structured event log.
- `history/summary.md`: compact summary of recent runs and decisions for future proposer rounds.
- `candidates/candidate_XXXX/`: one directory per proposed candidate.
- `runs/<run-id>/`: one directory per executed run.

Each candidate directory contains round-by-round trace material such as:

- proposer prompt
- proposer log
- candidate patch
- candidate rationale/env file
- pre-review prompt
- pre-review log
- pre-review decision
- candidate manifest

Each run directory contains:

- copied candidate patch
- candidate reference manifest
- remote training log
- parsed metrics
- post-review prompt
- post-review log
- post-review decision
- run manifest

The proposer is expected to read both the structured history and the raw artifacts. That is why the controller maintains both `ledger.jsonl` and per-candidate / per-run directories.

**Rationale File**

The proposer must write `controller_state/current_candidate.env` in its scratch clone with these fields:

- `IDEA`
- `HYPOTHESIS`
- `EXPECTED_SIGNALS`
- `NOTES`
- `EXTRA_ENV`

This is what makes the experiment traceable. The controller persists that rationale alongside the patch and the eventual result.

**Commands**

Run for a fixed amount of time:

```bash
cd autoresearch
uv run python run_pgolf_experiment.py --hours 8
```

Run until you stop it after the current in-flight work finishes:

```bash
cd autoresearch
uv run python run_pgolf_experiment.py --forever
```

Override models if needed:

```bash
cd autoresearch
PROPOSER_MODEL=gpt-5.4 \
PRE_REVIEW_MODEL=gpt-5.4 \
POST_REVIEW_MODEL=gpt-5.4 \
uv run python run_pgolf_experiment.py --hours 8
```

**Good To Know**

- The proposer is restricted to changing `train_gpt.py`. Run config changes belong in `EXTRA_ENV`.
- The controller expects exactly one candidate commit per proposer round.
- `controller_state/current_candidate.env` is trace metadata, not part of the experiment patch. Leave it untracked.
- A candidate can be rejected before it ever reaches the GPU.
- A queued patch can still fail to apply later if the reviewed base has moved too far; those failures are recorded in trace artifacts.
- `results.tsv` and `reviews.tsv` are still the high-level human-readable ledgers.
- Detailed artifacts live outside git-tracked history under `TRACE_ROOT`.
- The prompt files in this directory are part of the controller contract. If you change them, do it deliberately.
- Under `systemd --user`, prefer `journalctl --user -u parameter-golf-autoresearch`
  for logs instead of systemd-managed file append targets.

**Checks**

The subproject is set up for `uv`, `ruff`, and `ty`:

```bash
uv run --group dev ruff check .
uv run --group dev ty check
```

If the tools are not already available in the local `uv` environment, run:

```bash
uv sync --group dev
```
