# Hetzner Controller Deployment

This directory contains the first-pass automation for deploying the autoresearch
controller onto a long-lived controller host such as Hetzner.

The deployment model is:

- Hetzner runs the controller, Codex CLI, trace storage, and queue.
- Runpod runs the disposable GPU worker that the controller SSHes into.

## Prerequisites

On the controller host:

- `python3`
- `git`
- `systemd --user`
- `codex` installed and already authenticated

The deploy script will install `uv` automatically if it is missing.

## Runtime Contract

Prepare an env file from
[autoresearch.env.example](/var/home/carlulsoechristensen/Documents/parameter-golf/infra/hetzner/autoresearch.env.example).

That file defines:

- the remote Runpod worker SSH target
- the dataset/tokenizer paths visible on the GPU worker
- the default experiment budget
- the controller trace and ledger locations

## Deploy

Run from the repository root, after committing your deployment target state:

```bash
python3 infra/hetzner/deploy_controller.py \
  --host your-user@your-hetzner-host \
  --env-file /absolute/path/to/autoresearch.env \
  --start
```

Useful flags:

- `--identity ~/.ssh/hetzner_ed25519`
- `--port 2222`
- `--controller-args "--hours 8"`
- `--enable-linger`
- `--dry-run`

## Notes

- The deploy script requires a clean local git worktree.
- It syncs the current repository contents to the controller host with `rsync`.
- The installed service is a user service named `parameter-golf-autoresearch` by default.
- Prefer `journalctl --user -u parameter-golf-autoresearch -f --output=cat` for service logs.
- This bootstraps the controller host only. The GPU worker bootstrap belongs in the Runpod path.

## Operational Notes

These are safe to keep in the public repo because they are process notes, not credentials:

- Do not commit controller hostnames, Tailscale names, SSH keys, OpenAI credentials, or Runpod SSH destinations.
- The controller host repo must be trusted by git. The deploy script now adds the remote repo to `safe.directory`.
- Do not exclude tracked artifacts from deploy sync. Missing tracked files make the controller repo dirty and the controller refuses to start.
- If you use Runpod's `ssh.runpod.io` gateway, the controller should allocate a PTY for remote commands. `REMOTE_SSH_FORCE_TTY` exists for that, and the controller auto-enables it for `*.runpod.io`.
- On typical Runpod pods, a working default is `REMOTE_TORCHRUN=/usr/local/bin/torchrun`.
- The controller host may need to push experiment branches to a writable fork remote while the GPU worker fetches that branch from its own `origin`. Use `PUSH_REMOTE` and `REMOTE_FETCH_REMOTE` instead of assuming both sides use the same remote name.
- If the worker already contains the dataset and tokenizer inside the repo under `/workspace/parameter-golf/data/...`, prefer those concrete paths over speculative network-volume placeholders.
