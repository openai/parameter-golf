# Execution Protocol

Read this alongside `CLAUDE.md` at the start of any execution session.

## You are in execution mode if…
- A pod is live (or about to be launched).
- User handed you a spec number (e.g., "run spec 003").
- Your job is to execute one spec end-to-end and produce artifacts.

**You do not write logic code.** If the spec's commit has a bug, stop, hand back to research, do not patch on the fly.

## Hardware ladder

| Rung | Hardware | Purpose | Typical wall time | Typical cost |
|---|---|---|---|---|
| Smoke | **1×H100** | Syntax, imports, data load, 10–50 training steps, sanity-check loss curve | ~3 min | <$0.50 |
| Mini | **2×H100 (NA-1)** | Full short training run, single seed. Cheap signal for ideas. | ~30–40 min | ~$3 |
| Official | **8×H100 (NA-1)** | Competition-spec run. Seed 42 for first, 3 seeds for final submission. | ~12 min wall per seed | ~$3.50 per seed |

**Rule:** never let an 8×H100 pod discover a bug a 1×H100 could catch. If the spec involves multi-GPU-specific logic (collectives, sharding), a 2×H100 dry-run may be warranted before 8×.

The spec dictates which rungs apply; execution doesn't skip rungs without the user's say-so.

## Persistent storage (NA-1)

- Network volume mounted at `/workspace/` on any pod in NA-1.
- Holds:
  - Training data: `/workspace/data/datasets/fineweb10B_sp1024/` (and SP8192 equivalent)
  - Tokenizers: `/workspace/data/tokenizers/`
  - Run artifacts: `/workspace/runs/NNN-slug/`
  - Checkpoints: `/workspace/runs/NNN-slug/checkpoints/step_NNNN.pt`
- Survives pod cycles. Never download data that should already be on the volume.

## Spec interview protocol

On receiving spec number `NNN`:

1. Open **only** `research/specs/NNN-slug.md`, `CLAUDE.md`, `EXECUTION.md`. Don't browse other specs/runs unless this spec explicitly references them (e.g., hotstart source).
2. **Walk the user through the spec.** Ask about:
   - Hypothesis — does it still hold given anything we've learned since the spec was written?
   - Config diff — any values look wrong?
   - Branch + commit — does the commit exist on remote, is the diff the intended change?
   - Inputs — do the referenced paths exist on the NA-1 volume? Hotstart checkpoint still there?
   - Accept criteria + stop-early criteria — clear enough to act on?
   - Checkpoints to emit — which steps, what state, where?
   - Open questions flagged by the spec — resolve each with the user.
3. Surface ambiguities as questions, not assumptions. This is cheap — launching under ambiguity wastes $3+.
4. Only after the interview passes: preflight, then launch.

## Preflight checklist (before any launch)

- [ ] PyTorch version matches spec (default: what `train_gpt_sota.py` expects — check the file).
- [ ] `sentencepiece` installed in pod (user memory: this needs explicit pip install).
- [ ] Data path from spec exists on `/workspace/`.
- [ ] Tokenizer file present.
- [ ] Checkpoint dir writeable (`CKPT_DIR` env var set, directory exists).
- [ ] Hotstart checkpoint exists and is readable (if spec specifies one).
- [ ] `git rev-parse HEAD` matches the commit hash in the spec.
- [ ] Enough free disk on `/workspace/` for expected checkpoints.

If any check fails and it's an **environment** issue (missing dep, path typo), fix it and re-check. If it's a **logic** issue (wrong commit, bad config), stop and hand back to research.

## Launch & monitor

1. Smoke first (1×H100) unless spec explicitly waives it. Check: no NaN in first 50 steps, loss decreasing, step time reasonable.
2. Promote to the spec's target rung.
3. During the run, watch for:
   - NaN loss → kill, mark failed.
   - Hung collectives (step time blows up) → kill, mark failed.
   - Unexpected memory pressure → kill, mark failed, reduce batch or seq_len only if spec permits.
   - Step-time drift beyond spec tolerance → note in `notes.md`, keep running unless stop-early criterion hits.
4. Apply **spec's stop-early criteria** literally.

## Artifact shape in `runs/NNN-slug/`

For single-seed runs:
```
runs/NNN-slug/
  config.json        # resolved hyperparams (final, post-defaults)
  train.log          # full stdout/stderr; gzip if >1MB, keep tail as train.log.tail
  loss_curve.csv     # columns: step, train_loss, val_bpb (whenever evaluated)
  final.json         # see below
  notes.md           # execution session's observations, anomalies
  checkpoints.md     # pointer file: step N → /workspace/runs/NNN-slug/checkpoints/step_N.pt
```

For multi-seed runs (per-seed subdirs):
```
runs/NNN-slug/
  seed_42/{config.json, train.log, loss_curve.csv, final.json, notes.md, checkpoints.md}
  seed_43/…
  seed_44/…
```

### `final.json` schema (required)
```json
{
  "spec": "003-bigram-hash",
  "seed": 42,
  "status": "completed",          // or "failed"
  "reason": null,                  // if failed, short string
  "val_bpb": 1.0812,
  "hardware": "8xH100",
  "region": "NA-1",
  "wall_time_sec": 712,
  "cost_usd": 3.45,
  "pod_id": "abc123",
  "git_commit": "def456...",
  "started_at": "2026-04-20T14:32:11Z",
  "ended_at": "2026-04-20T14:44:03Z",
  "checkpoints_emitted": ["/workspace/runs/003-bigram-hash/checkpoints/step_4550.pt"]
}
```

Checkpoints themselves live on the NA-1 volume, **not in git**. `checkpoints.md` is the in-repo pointer.

## Stop protocol

- **Immediately after eval writes**: `runpodctl stop pod $RUNPOD_POD_ID`.
- Sync small artifacts (everything except checkpoints) to the in-repo `runs/NNN-slug/` directory.
- Do not leave pods idle. Every minute costs money.

## Handback to user

When done, report to the user (one paragraph):
- Spec number and status (completed / failed).
- Final bpb (mean + per-seed if multi-seed).
- Wall time and cost.
- Any anomalies worth flagging.
- Path to `runs/NNN-slug/` for research session to pick up.

Evaluation is **not** execution's job. Don't interpret the number, don't write `experiments.md`, don't decide promote/kill. That's research.
