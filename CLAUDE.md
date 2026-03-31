# Coordination Rules

This repo uses one shared handoff protocol for Claude Code, Codex, and Antigravity.

## Entry Point

Every new session must read in this order:
1. `AGENTS.md` — shared entry point, current working mode
2. `docs/campaign/AGENT_SYNC.md` — mutable source of truth for objectives, results, next steps
3. This file (`CLAUDE.md`) — standing rules and operational constraints

This file (`CLAUDE.md`) contains **stable standing rules only**. Do not duplicate mutable campaign state (current objective, latest metrics, next commands) that is already tracked in `AGENT_SYNC.md`.

## Session Rules

1. Treat `docs/campaign/AGENT_SYNC.md` as the current source of truth for:
   - active objective
   - current scope
   - next commands to run
   - latest measured results
2. Before starting new work, check `docs/campaign/artifacts/` and `records/` to avoid duplicating completed work.
3. If you change the objective, next step, or interpretation of results, update `docs/campaign/AGENT_SYNC.md`.
4. If you make a campaign-level decision or disagree with an earlier recommendation, record it in `docs/codex-memory/decisions.md`.
5. If a run produces a measured result, append one JSON record to `docs/campaign/results_log.jsonl`. Do not rewrite prior lines.
6. If you finish a meaningful session, update `docs/codex-memory/project-state.md` and `docs/codex-memory/next-session.md`.
7. If the task touches campaign strategy or prior experiments, also read:
   - `docs/codex-memory/project-state.md`
   - `docs/codex-memory/next-session.md`
   - `docs/codex-memory/decisions.md`
8. For competition re-implementations, use source priority:
   - `openai/parameter-golf` PR code first
   - local repo code second
   - papers and generic web sources only to resolve ambiguous math or API details
9. If a post-training export path looks wrong, debug it on the same checkpoint before spending more H100 time on retraining.

## Working Agreement

- Pegasus `8xH100` is the active development target.
- Pegasus `A100-80GB` is fallback or grant-supporting evidence, not the mainline path.
- RunPod is reserved for final validation only.
- `git clone` and `git pull` are the default sync path for remote workspaces.
- Use `rsync` only to push local uncommitted changes quickly.

## Challenge Submission Rules

These are stable public rules from the challenge README and should not be rediscovered every session.

- Official leaderboard entry is **record-gated**, not top-5-open-entry.
- A record submission must beat the current official SOTA by at least `0.005` nats and provide enough logs for `p < 0.01`.
- Train and eval must each run under `10 minutes` on `8xH100`.
- Total artifact size is `16,000,000` bytes decimal for code plus compressed model.
- If a submission does not beat the current record bar, it is a non-record submission, not official leaderboard entry.

## Pegasus Operational Rules

These are stable constraints learned from operational experience. They apply to all Pegasus jobs.

### Launcher
- **Never use `torchrun --standalone`** on Pegasus multi-GPU. It hangs at rendezvous.
- Use Slurm-native `srun` with manual rank env vars: `LOCAL_RANK=$SLURM_LOCALID`, `RANK=$SLURM_PROCID`, `WORLD_SIZE=$SLURM_NTASKS`.

### Job output
- **Never use `| tail -1`** on Pegasus training or install commands. It hides errors and progress.
- Always set `PYTHONUNBUFFERED=1` or use `python -u` to prevent output buffering.

### Allocation shape
- **Always include `--nodes=1`** for challenge-shaped `8xH100` runs. Without it, Slurm may split across nodes, breaking NVSwitch locality.
- Use `--ntasks=8 --gpus-per-task=1 --gpu-bind=none` (not `--gpus=8`).
- If a job lands on multiple nodes, cancel and relaunch with `--nodes=1`.

### FA3 container path
- Saved FA3 container: `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`
- **Do not use `--no-deps`** for FA3 wheel on stock NGC 25.02. The container's torch 2.7.0 is ABI-incompatible with FA3 (`undefined symbol: aoti_torch_abi_version`).
- Do not do ad hoc per-job `pip install` of FA3 once the saved container exists.
- See `docs/campaign/PEGASUS_H100_RUNBOOK.md` for full container build and benchmark commands.
