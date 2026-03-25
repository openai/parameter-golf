# Parameter Golf Colab Workflow

This workflow keeps active code and active dataset on Colab runtime disk (`/content`) for speed and reproducibility.

## Storage Layout

- GitHub repository (source of truth): scripts, docs, code changes, lightweight metadata you want versioned.
- Colab runtime disk (`/content/...`): active repo checkout, Python env, downloaded dataset, active training runs.
- Google Drive (optional persistence): notebooks, run logs, parsed summaries, exported artifacts.

Do not keep the active repo checkout or active dataset under Drive.

## One-Time Setup (New Colab Runtime)

1. Select a GPU runtime in Colab.
2. Optional: mount Drive only for persistent logs/summaries.
3. Run:

```bash
cd /content
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
git checkout -b bootstrap/colab-harness
bash scripts/bootstrap_env.sh
source .cache_env.sh
```

If you are using your fork:

```bash
cd /content
git clone https://github.com/<your-user>/parameter-golf.git
cd parameter-golf
git remote add upstream https://github.com/openai/parameter-golf.git
git checkout bootstrap/colab-harness || git checkout -b bootstrap/colab-harness
bash scripts/bootstrap_env.sh
source .cache_env.sh
```

## Smoke Test (1 GPU, 1 Train Shard)

```bash
cd /content/parameter-golf
bash scripts/download_data.sh --variant sp1024 --train-shards 1
bash scripts/run_smoke_1gpu.sh
python3 scripts/parse_logs.py
```

`run_smoke_1gpu.sh` stages a 1-train-shard view when needed, keeps a short wallclock cap, and writes logs to `logs/runs/<timestamp>_<run_id>/`.

## Baseline Run (1 GPU, Official Path)

```bash
cd /content/parameter-golf
bash scripts/download_data.sh --variant sp1024 --train-shards 80
bash scripts/run_baseline_1gpu.sh
python3 scripts/parse_logs.py
```

This wrapper keeps the official baseline command shape and records run metadata for reproducibility.

## Sync Logs And Summaries To Drive

If Drive is mounted:

```bash
cd /content/parameter-golf
bash scripts/sync_artifacts_to_drive.sh --dest /content/drive/MyDrive/parameter-golf-artifacts
```

This copies logs and results metadata only. It does not copy repo code or datasets.

## Pull Latest Branch Updates

For fork workflow:

```bash
cd /content/parameter-golf
git fetch upstream
git checkout bootstrap/colab-harness
git pull --ff-only upstream main
git fetch origin
git pull --ff-only origin bootstrap/colab-harness
```

For upstream-only clone:

```bash
cd /content/parameter-golf
git fetch origin
git checkout bootstrap/colab-harness
git pull --ff-only origin bootstrap/colab-harness
```

If you have local edits, commit or stash before pulling.
