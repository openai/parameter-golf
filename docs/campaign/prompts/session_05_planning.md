# Session 05 Planning Prompt

Paste everything below the line into a fresh Claude Code session.

---

Session 05 planning: throughput audit + pre-TTT stack-gap audit + TTT correctness audit.

## Read order

Read these files in this exact order before doing anything else:

1. `docs/campaign/AGENT_SYNC.md` — source of truth for objective, scope, results
2. `CLAUDE.md` — standing coordination rules
3. `docs/campaign/artifacts/04_targeted_delta_sweep.md` — Session 04 closeout
4. `docs/campaign/sessions/05_ttt_correctness_audit.md` — Session 05 audit plan
5. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` — #1 entry docs
6. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` — #1 entry code
7. `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py` — our anchor code

## Current fixed facts

- Session 03 anchor sliding s64: `1.12904446`
- Session 03 anchor pre-quant EMA: `1.14472403`
- Session 03 anchor roundtrip: `1.15247273`
- Session 03 anchor step_avg: `91.37 ms`, steps: `6564`
- Session 03 anchor artifact: `15751324` bytes (headroom `248676` bytes)
- Session 04 Delta 1 GPTQ-lite: **FAILED** (artifact over cap)
- Session 04 Delta 2 LeakyReLU²: **NEUTRAL** (effectively identical sliding s64)
- Local public #1 record: pre-TTT `1.1218`, post-TTT `1.1194`, step_avg `83.4 ms`
- Leaderboard entry threshold: `≤ 1.1178`
- Gap from anchor to threshold: `0.0112 BPB`

## Goal

Produce a ranked Session 05 implementation plan that improves leaderboard odds by:
1. Narrowing the throughput gap (`91.37 ms` → `83.4 ms`)
2. Improving the pre-TTT base (`1.1290` → closer to `1.1218`)
3. Planning TTT integration correctly (expected `-0.0025` on top of base)

## Constraints

- **Planning first, not broad implementation.** This session produces an audit artifact and ranked plan.
- Be strict about attribution and challenge legality.
- Separate portable first-wave changes from harder second-wave rewrites.
- Do not assume TTT alone closes the gap — #1's pre-TTT base is already `1.1218`.
- Do not assume FA3 is available on Pegasus NGC container — verify portability requirements first.
- Do not combine throughput, pre-TTT, and TTT changes in one unattributable run.

## Required outputs

### 1. Throughput audit
- Explain likely contributors to `91.37 ms` vs `83.4 ms`
- Rank by portability and expected impact
- Specifically answer: is FA3 the first thing to try?
- Verify whether `flash_attn_interface` / `flash_attn` is already present in the Pegasus NGC 26.03 container path; if not, estimate portability cost and risk — do not turn this into a package-install session
- Identify the tensor layout change needed (SDPA uses `B,H,T,D`, FA3 uses `B,T,H,D`)

### 2. Pre-TTT stack-gap audit
- Compare our anchor vs the local `1.1194` public stack feature by feature
- Rank easy portable features: VE128, warmdown3500, Bigram 1536, tight SWA
- Separate from hard items: Parameter Banking, Parallel Muon
- For each: expected BPB gain, engineering cost, artifact size impact

### 3. TTT audit
- Trace the score-first legality path (inference_mode for scoring, backward only on already-scored tokens)
- Quantify eval-time cost (how much of the eval budget does TTT consume?)
- Identify what's portable to our anchor's `eval_val_sliding()` function
- Document key hyperparameters: `lr=0.002`, `epochs=3`, `chunk=32768`, `grad_clip=1.0`

### 4. Final recommendation
- What to implement first (first-wave)
- What to defer (second-wave)
- What to keep out of scope
- Expected cumulative BPB improvement if first-wave succeeds

## Deliverable

One concise ranked plan saved to `docs/campaign/artifacts/05_ttt_correctness_audit.md` with:
- First-wave changes (portable, high-value, low-risk)
- Second-wave changes (harder, deferred)
- Per-item: expected upside, engineering cost, risk, artifact size impact

## Tools and skills

Use these if available in your session (prefer but do not require):

- **`/research-engineer`** skill — for structured audit with scientific rigor
- **`deepwiki` MCP** — for codebase questions against `amrayach/parameter-golf`
- **`context7` MCP** — for PyTorch / flash_attn library docs
- **MCP search tools** — for checking past decisions and observations if available
- **Code navigation MCPs** — for symbol-level exploration if available
- **Parallel Agent subagents** (subagent_type=Explore) — for comparing code sections across files simultaneously

If any MCP or skill is not available, fall back to Grep, Glob, and Read tools.

## Git conventions

Follow the established commit message prefixes:
- `research(protocol):` — before running an experiment (implementation commit)
- `research(results):` — after a run, with measured results
- `docs:` or `docs(campaign):` — documentation-only changes
- `fix:` — bug fixes
- `perf:` — performance changes
- `ops:` — infrastructure/launcher changes

Rules:
- Never use `--no-verify` or `--no-gpg-sign`
- Stage specific files, never `git add -A` or `git add .`
- The worktree has an unrelated modified file: `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/README.md` — do NOT stage it
- There are untracked files (`.serena/`, `docs/*.pdf`, `docs/*.txt`) — do NOT stage them

## Pegasus server conventions

- **Partition:** `H100` (default), `A100` fallback
- **Allocation:** `salloc -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --mem=200G --time=1-00:00:00`
- **Launch:** NEVER use `torchrun`. Always use `srun --gpu-bind=none` with manual rank env vars:
  ```
  LOCAL_RANK=$SLURM_LOCALID
  RANK=$SLURM_PROCID
  WORLD_SIZE=$SLURM_NTASKS
  ```
- **Launcher script:** `scripts/pegasus_optimized_launcher.sh <run_id> [script_path]`
- **Container:** NGC 26.03 (`nvcr.io_nvidia_pytorch_26.03-py3.sqsh`), auto-detected by launcher
- **Data path:** `/fscratch` preferred (low-latency), `/netscratch` fallback
- **Dependencies inside container:** `pip install sentencepiece zstandard` (launcher handles this)
- **Logs:** `/netscratch/ayach/<run_id>.log`
- **Sync:** `git pull` on Pegasus before running, `rsync` only for uncommitted quick-push

## Documentation conventions

After completing meaningful work, update these files in order:

### Shared docs (canonical truth)
1. `docs/campaign/AGENT_SYNC.md` — update if objective, scope, results, or next steps change
2. `docs/campaign/artifacts/<artifact>.md` — create deliverable artifacts (e.g., `05_ttt_correctness_audit.md`)
3. `docs/campaign/sessions/<session>.md` — update session status if it changes

### Codex memory (project-persistent, repo-committed)
4. `docs/codex-memory/project-state.md` — update after session completion
5. `docs/codex-memory/next-session.md` — update with next immediate action
6. `docs/codex-memory/decisions.md` — record any campaign-level decisions
7. `docs/codex-memory/session-handoff.md` — update handoff state for next session

### Claude memory (cross-conversation persistent)
8. `~/.claude/projects/-home-amay-Work-parameter-golf/memory/project_parameter_golf.md` — update project state
9. `~/.claude/projects/-home-amay-Work-parameter-golf/memory/MEMORY.md` — update index if new memories created

### Record folders
- Each experiment gets its own folder: `records/track_non_record_16mb/YYYY-MM-DD_<tag>/`
- Never mutate the anchor folder
- Each folder contains: `train_gpt.py`, `README.md`, `submission.json`, `requirements.txt`
- `submission.json` uses the flat schema: `author`, `github_id`, `name`, `blurb`, `track`, `val_bpb`, `steps`, `bytes_total`, etc.
