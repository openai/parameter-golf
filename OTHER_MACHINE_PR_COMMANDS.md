# Other Machine PR Commands

Run these on the machine where `gh auth status` is authenticated as `tenet-diver`.

```bash
cd /workspace/parameter-golf
```

```bash
git fetch origin submission/nonrecord-ttt-disabled-runtime
```

```bash
git switch submission/nonrecord-ttt-disabled-runtime || git switch -c submission/nonrecord-ttt-disabled-runtime --track origin/submission/nonrecord-ttt-disabled-runtime
```

```bash
git pull --ff-only
```

```bash
git status --short
```

```bash
python3 -m json.tool records/track_non_record_16mb/2026-04-30_gpuq-ttt-disabled-runtime-control_1p44421/submission.json >/dev/null
```

```bash
python3 -m py_compile records/track_non_record_16mb/2026-04-30_gpuq-ttt-disabled-runtime-control_1p44421/train_gpt.py
```

```bash
git diff --check
```

```bash
gh auth status
```

```bash
cat > /tmp/pg_pr_body.md <<'EOF'
Adds a single non-record 16MB submission documenting the best completed candidate from the 1xH100 deadline batch.

Final candidate:
- candidate_id: `gpuq_ttt_disabled_runtime_control`
- val_bpb: `1.44421409`
- val_loss: `2.438495`
- hardware: `1x NVIDIA H100 80GB HBM3`
- seed: `1337`
- train time: `668.949s`
- model bytes: `10,095,619`
- code bytes: `104,891`
- total package bytes: `10,200,510`

This is submitted as non-record because it was run on 1xH100 rather than 8xH100 SXM, includes one seed, and the packaged run exceeded 600 seconds.

The result is useful as a documented deadline-batch control: disabling the runtime TTT path was the strongest completed candidate in this batch, ahead of the completed QK-gain, dense-control, optimizer, and kernel-policy variants.

Included files:
- `README.md`
- `submission.json`
- self-contained `train_gpt.py`
- `train.log` / `train_seed1337.log`
- `candidate_env.json`
- `run_result_seed1337.json`
- `packaged_batch_rows.json`
- `stderr_seed1337.txt`

Validation:
- `python3 -m json.tool records/track_non_record_16mb/2026-04-30_gpuq-ttt-disabled-runtime-control_1p44421/submission.json >/dev/null`
- `python3 -m py_compile records/track_non_record_16mb/2026-04-30_gpuq-ttt-disabled-runtime-control_1p44421/train_gpt.py`
- `git diff --check`
EOF
```

```bash
gh pr create --repo openai/parameter-golf --base main --head tenet-diver:submission/nonrecord-ttt-disabled-runtime --title "Non-record: TTT-disabled runtime control (1xH100, val_bpb 1.4442)" --body-file /tmp/pg_pr_body.md
```

If the final command says a PR already exists, run:

```bash
gh pr list --repo openai/parameter-golf --head tenet-diver:submission/nonrecord-ttt-disabled-runtime --web
```
