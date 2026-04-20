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
| Mini | **2×H100 (NA-1)** | Smoke + signal in one. Full short training run, single seed. Catches bugs and gives a cheap bpb signal. | ~30–40 min | ~$3 |
| Official | **8×H100 (NA-1)** | Competition-spec run. Seed 42 for first, 3 seeds for final submission. | ~12 min wall per seed | ~$3.50 per seed |

**Rule:** never let an 8×H100 pod discover a bug the 2×H100 mini could have caught.

The spec dictates which rungs apply; execution doesn't skip rungs without the user's say-so.

## Branches, worktrees, and what you check out

Execution sessions **do not use worktrees.** Worktrees are a research-session convenience for parallel code editing on the laptop. On a pod:

1. Fresh `git clone` of the `parameter-golf` repo (if not already present on the volume).
2. `git fetch` all branches.
3. `git checkout <commit>` — the exact commit hash the spec pins. This commit lives on some `exp/<slug>` branch (or directly on `research` for hyperparam-only ideas).
4. Run.

You are on a detached HEAD or the branch tip; either way, the working tree is whatever that commit snapshots. No worktree sibling dirs.

If the spec's commit hash doesn't exist on the remote yet, **stop** — the research session forgot to push. Hand back.

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
4. **Confirm the hardware ladder — especially smoke.** Research sets the default in the spec. You cannot silently skip a rung.
   - If the spec marks smoke **required**, run it. No exceptions without user's explicit waiver.
   - If the spec marks smoke **skipped** citing a recent clean run: verify that prior run is actually recent (hours to a day, not weeks) and that the code hasn't moved. If either is off, ask the user before proceeding.
   - If you're tempted to skip a rung "because the change is small," ask first. A $0.50 smoke test is insurance against a $3.50 wasted full run.
5. Only after the interview passes: preflight, then launch.

## Preflight checklist (before any launch)

- [ ] PyTorch version matches spec (default: what `train_gpt_sota.py` expects — check the file).
- [ ] `sentencepiece` installed in pod (user memory: this needs explicit pip install).
- [ ] `brotli` installed (`pip install brotli --break-system-packages -q` — container disk, reinstalls on every pod start).
- [ ] Data path from spec exists. The real path is `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/`, not `/workspace/data/...`. Verify via `ls`.
- [ ] Tokenizer file present at `/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model`.
- [ ] Checkpoint dir writeable (`CKPT_DIR` env var set, directory created with `mkdir -p` before launch so `setsid` redirects don't fail).
- [ ] Hotstart checkpoint exists and is readable (if spec specifies one).
- [ ] `git rev-parse HEAD` on the pod matches the commit hash in the spec. Use `git stash push -u -m "pod-local"` to clear any uncommitted pod-local edits that might block `checkout`.
- [ ] Enough free disk on `/workspace/` for expected checkpoints (9 × ~300 MB = ~2.7 GB for the standard phase-boundary set).
- [ ] `TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache` set on the launch env, and `mkdir -p /workspace/.torch_inductor_cache` on the volume beforehand. Persists the torch.compile cache across pod cycles; reruns of the same commit skip ~80% of the ~5min compile (saves 3-4min wallclock). Graph-hash-keyed so it's safe — different commits just don't reuse each other's entries.

If any check fails and it's an **environment** issue (missing dep, path typo), fix it and re-check. If it's a **logic** issue (wrong commit, bad config), stop and hand back to research.

## Launch & monitor

1. Mini first (2×H100) unless spec explicitly waives it. Check: no NaN in first ~50 steps, loss decreasing, step time reasonable — then let it run to completion for the bpb signal.
2. Promote to 8×H100 official if mini looks clean and the spec calls for it.
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

Order matters:
1. **Rsync artifacts first** (small files only — logs, final.json, notes; skip `checkpoints/` which live on the volume) to the in-repo `runs/NNN-slug/` directory.
2. **Then stop the pod:** `runpodctl pod stop <id>`. Same-day → stop (preserves container disk for fast resume). End of day → `runpodctl pod delete <id>` (fully terminates, frees all billing).
3. Do not leave pods idle. Every minute on 8×H100 is ~$0.40.

Reversing the order costs ~$0.40 of pod-churn: stopping first means you must `pod start <id>` again, wait for SSH, re-install any non-persistent deps (e.g., `brotli`), rsync, then stop again.

## Handback to user

When done, report to the user (one paragraph):
- Spec number and status (completed / failed).
- Final bpb (mean + per-seed if multi-seed).
- Wall time and cost.
- Any anomalies worth flagging.
- Path to `runs/NNN-slug/` for research session to pick up.

Evaluation is **not** execution's job. Don't interpret the number, don't write `experiments.md`, don't decide promote/kill. That's research.

## Pod operations playbook

Lessons from spec 000. Read these before your first pod session — they save real money.

### Runpodctl CLI — use the new subcommand form

The old `runpodctl create pod …` / `runpodctl stop pod <id>` form still works but is deprecated and has **different flag names** than the new form. Use the new resource-based form and its exact flags:

```bash
# Create (new form)
runpodctl pod create \
  --name parameter-golf-NNN-slug \
  --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 8 \
  --data-center-ids US-NE-1 \
  --network-volume-id hvpdph5i3g \
  --template-id y5cejece4j \
  --ssh \
  --env "$(jq -cn --arg k "$(cat ~/.runpod/ssh/RunPod-Key-Go.pub)" '{PUBLIC_KEY:$k}')"

# Stop (preserves container disk for fast same-day resume)
runpodctl pod stop <pod-id>

# Terminate (frees all billing — end of day)
runpodctl pod delete <pod-id>

# Query individual (works on EXITED pods too)
runpodctl pod get <pod-id>

# List all (may only show non-EXITED — don't trust empty list to mean "no pods")
runpodctl pod list
```

Flag-name gotchas vs. the old `create pod` form:
- `--gpu-id` not `--gpuType`
- `--gpu-count` not `--gpuCount`
- `--data-center-ids` not `--dataCenterId` (plural, comma-separated)
- `--network-volume-id` not `--networkVolumeId`
- `--template-id` not `--templateId`
- `--ssh` not `--startSSH`
- `--env` takes a **JSON object string**, not repeated `KEY=VAL`

### Getting SSH access to a new pod

`runpodctl pod get <id>` has a `runtime` field that's often `null` for the first 30–90s after create even when the pod is "RUNNING". Don't block on polling it. Instead:

```bash
# Returns ip + port as soon as SSH daemon binds
runpodctl ssh info <pod-id>
# Then TCP-probe to confirm the SSH port is actually listening
timeout 3 bash -c "</dev/tcp/$HOST/$PORT"
```

SSH command (from `runpodctl ssh info` output): `ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port>`. The key at `~/.runpod/ssh/RunPod-Key-Go` is auto-synced to the runpod account by `runpodctl doctor`; every pod that gets `PUBLIC_KEY=<pubkey>` in its env (or is created via `--ssh` with an account-synced key) will accept it.

### Launching long-running commands via SSH

**Don't** do this (connection close can SIGHUP the process):
```bash
ssh … 'nohup bash /tmp/launch.sh > out.log 2>&1 &'
```

**Do** this:
```bash
# 1) Pre-create the log's directory (setsid can't mkdir before redirecting)
ssh … 'mkdir -p /workspace/runs/NNN-slug/checkpoints'
# 2) scp the launch script (don't inline via heredoc — SSH+heredoc+nested quoting is flaky)
scp … /tmp/launch_pod.sh root@$HOST:/tmp/launch_pod.sh
# 3) Detach with setsid + stdin closed + full stdio redirect + disown
ssh … 'setsid bash /tmp/launch_pod.sh </dev/null >/workspace/runs/NNN-slug/launch.out 2>&1 & disown'
# 4) Verify pgrep -af torchrun shows the process
```

### Environment persistence across pod stop/start

| Path | Persists across pod stop? | Notes |
|---|---|---|
| `/workspace/` | YES (network volume) | Training data, tokenizers, `runs/`, `parameter-golf/` repo, checkpoints |
| Container disk (`/`, `/root/`, `/tmp/`, pip installs) | NO | Gets reset each time pod is stopped/started |
| `~/.ssh/authorized_keys` on pod | NO (reloaded from `PUBLIC_KEY` env) | |

Consequence: **reinstall `brotli` after every pod start** (takes ~5-10s via `pip install brotli --break-system-packages -q`). The run scripts already do this on launch; just don't rely on it from an earlier session.

### Wallclock budgets — what "10 min" actually means

The competition has **two separate 10-min budgets**, not one:
- **Training ≤600s** (enforced by `max_wallclock_seconds=600` in code)
- **Eval ≤600s** (enforced by the leaderboard's runtime on their hardware)

Realistic pod wall time per seed:
- Compile + warmup + setup: ~60s
- Training: 588s (just under 600s)
- EMA + GPTQ + brotli: ~20s
- Quantized val: ~25s
- Sliding-window val: ~2 min
- **TTT val: ~6 min** ← the big one
- Pod startup + SSH handshake overhead: ~60-90s

**Total pod wall per seed: ~22-25 min at ~$23.92/hr → ~$9-10/seed.** Pre-spec-000 estimates of "12 min / $3.50" were training-only thinking. Budget for the real number in spec + cost estimates.

### Throughput variance in the H100 pool is real

Same nominal hardware (8×H100 SXM, 80GB HBM3) in the same datacenter can run at very different tok/s depending on which physical node Runpod assigns. Spec 000 ran at **~85% of the SOTA submission's step rate** in the same 588s training window — 3849 steps vs. 4550 steps — directly costing ~0.005 bpb at every eval stage. The code was identical.

Mitigations worth trying in future specs (by increasing cost/complexity):
1. **Add a tok/s preflight.** After compile, measure tok/s over ~30 steps. If below ~6.5M tok/s, kill and re-provision. Costs ~$0.40 per bad pod but guarantees a fast one. Easiest.
2. **Prefer secure cloud** if flags allow (`--cloud-type SECURE`, which is actually the CLI default). Community cloud has more variance.
3. **Multi-pod shop.** Provision 2-3 pods in parallel, benchmark each with a 60s torch all-reduce test, keep the fastest. Overkill for single-seed runs, maybe worth it for 3-seed submission runs.

Never let an official 3-seed submission land on slow hardware — that ~0.005 bpb deficit is the entire width of the accept window.

### Data path — verify, don't trust the spec

`train_gpt_sota.py` uses `DATA_DIR=./data/` (relative), resolved from `cd /workspace/parameter-golf/` → so the real data paths are:
- Train data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_train_*.bin`
- Val data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_val_*.bin`
- Tokenizer: `/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model`

Spec files may claim `/workspace/data/…` (absolute) — that path does **not** exist. The `./data/` inside the repo does. SSH-in and `ls` to confirm before launching.

### Kill fast when something's off

Pod cost on 8×H100 is ~$0.40/min. If an SSH heredoc hangs, or runtime doesn't populate, or the launch script fails to spawn torchrun — the default should be to **stop the pod immediately** (saves ~$0.40/min burn while you debug), not to babysit a broken pod. You can always `pod start <id>` to resume from the same container.

Spec 000's early SSH+heredoc mishap + recovery cost ~$3.60 of pure churn before the real run started. Cheaper to stop-and-restart than to wait-and-see.

### Monitoring pattern that works

A 60-second SSH poll loop parsing `tail -n 400` of the pod's `train.log`, posting one Discord row per tick, with event-triggered pings for each ckpt file landing and each new val, closes out cleanly when the final `quantized_ttt val_bpb:` line lands. Helpers live at `.claude/scripts/discord_post.sh` and `discord_post_table.sh`. The state file at `/tmp/spec000_monitor.state` records what's already been pinged so nothing double-posts.

### Polling cadence during active runs — 30 seconds

**During any live training run (smoke, screen, or submission): poll the pod's `train.log` every 30 seconds.** This is the user's expected cadence and takes precedence over the default self-pacing. ScheduleWakeup's 60s floor means using a tight Bash `until` loop that `sleep 30` and re-SSH-tail. Do not drift to 1-min, 2-min, 5-min poll intervals unless the user explicitly says to slow down — the short cadence is load-bearing for the user's ability to intercept bugs early.
