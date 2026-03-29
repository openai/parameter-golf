# Coordination Rules

This repo uses one shared handoff protocol for Claude Code and Codex.

## Read Order

1. Read `docs/campaign/AGENT_SYNC.md` first.
2. If the task touches campaign strategy or prior experiments, also read:
   - `docs/codex-memory/project-state.md`
   - `docs/codex-memory/next-session.md`
   - `docs/codex-memory/decisions.md`

## Session Rules

1. Treat `docs/campaign/AGENT_SYNC.md` as the current source of truth for:
   - active objective
   - current scope
   - next commands to run
   - latest measured results
2. Before starting new work, check `docs/campaign/artifacts/` and `records/` to avoid duplicating completed work.
3. If you change the objective, next step, or interpretation of results, update `docs/campaign/AGENT_SYNC.md`.
4. If you make a campaign-level decision or disagree with an earlier recommendation, record it in `docs/codex-memory/decisions.md`.
5. If you finish a meaningful session, update `docs/codex-memory/project-state.md` and `docs/codex-memory/next-session.md`.

## Working Agreement

- Current goal is stronger Pegasus `8xH100` competition evidence, not just a grant-only evidence package.
- Pegasus `8xH100` is the active development target.
- Pegasus `A100-80GB` is now fallback or grant-supporting evidence, not the mainline path.
- RunPod is reserved for final validation only.
- `git clone` and `git pull` are the default sync path for remote workspaces.
- Use `rsync` only to push local uncommitted changes quickly.
