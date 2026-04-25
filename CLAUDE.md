# CLAUDE.md — Parameter Golf

Agent-facing quick reference for this repo. For the canonical human-facing doc, see [README.md](README.md).

## TL;DR for Agents

- This is **OpenAI's Model Craft Challenge: Parameter Golf** — train the best LM that fits in **16,000,000 bytes** (decimal) and finishes in **< 10 min on 8xH100s**, scored by `val_bpb` on FineWeb.
- An "edit" is almost never to the root `train_gpt.py`. It is usually a **new folder** under `records/track_10min_16mb/YYYY-MM-DD_Description/` containing a self-contained submission.
- The current SOTA is whichever folder's date is newest in [records/track_10min_16mb/](records/track_10min_16mb/) and sits at the top of the README leaderboard table.
- Leaderboard runs happen on **Runpod 8xH100 SXM pods** (template id `y5cejece4j`). This host has a single **RTX 4090 (24 GB)** — enough for smoke tests and iteration at reduced batch size; not enough to reproduce leaderboard numbers. See "Running on This Machine" below.
- All knobs are **environment variables**. The `Hyperparameters` class at [train_gpt.py:39-87](train_gpt.py#L39-L87) is the authoritative list. A curated subset is in [.env.example](.env.example).

## Hard Rules (Do Not Violate)

1. **16,000,000-byte artifact cap** (decimal, not MiB). Counted as `code_bytes + compressed_model_bytes`.
2. **Never access validation data during training.** No "paid prefix" tricks. Test-time training is only allowed on val tokens already scored.
3. **`train_gpt.py` and `train_gpt_mlx.py` must stay under 1500 lines** — see docstrings at [train_gpt.py:1-5](train_gpt.py#L1-L5) and [train_gpt_mlx.py:1-5](train_gpt_mlx.py#L1-L5).
4. **Do not add test frameworks, linters, type checkers, Docker, Makefiles, `pyproject.toml`, or CI.** None exist by convention; adding them creates PR friction.
5. **Do not edit existing `records/track_*/<DATE>_*/` folders.** Each submission is a frozen artifact. New work goes in a new dated folder.
6. **New SOTA requires a new dated folder, 3 fresh seed logs, ≥ 0.005 nats over current SOTA at p < 0.01.**
7. **CUDA is available locally (RTX 4090, 24 GB)**, but **the active Python env has the CPU-only torch wheel** (`torch==2.11.0+cpu`). Install a CUDA build before running `train_gpt.py` locally — e.g. `pip install --index-url https://download.pytorch.org/whl/cu124 torch` in a fresh venv. Leaderboard reproduction still needs 8xH100s.
8. **Do not download datasets unprompted.** `data/cached_challenge_fineweb.py` writes 8 GB+ by default.
9. **Do not commit anything in `.gitignore`**: `data/datasets`, `data/tokenizers`, `logs/`, `.venv`, `__pycache__/`, `modded-nanogpt/`, `.mypy_cache/`, `.DS_Store`.
10. **New dependencies belong in a submission's local `requirements.txt`**, not the root `requirements.txt`. The root file is intentionally minimal (10 deps).

## Project Structure

```
train_gpt.py                   CUDA trainer (baseline); < 1500 lines; Hyperparameters class at L39-87
train_gpt_mlx.py               MLX trainer for Apple Silicon; < 1500 lines
data/
  cached_challenge_fineweb.py  one-shot dataset + tokenizer download
  download_hf_docs_and_tokenize.py  rebuild tokenizers from docs cache
  tokenizer_specs.json         tokenizer family configs (sp1024, sp4096, sp8192, ...)
  README.md                    dataset/tokenizer workflow doc
records/
  track_10min_16mb/            leaderboard submissions (10-min / 16MB rule)
    YYYY-MM-DD_<Name>/         one folder per submission, frozen artifact
      README.md, submission.json, train_gpt.py, train_seed*.log
  track_non_record_16mb/       experimental / unlimited-compute submissions
README.md                      human-facing source of truth (setup, leaderboard, FAQ, rules)
THIRD_PARTY_NOTICES.md         modded-nanogpt attribution
requirements.txt               minimal root deps (10 lines)
.env.example                   curated env-var starter (full list: train_gpt.py:39-87)
.gitignore                     excludes data/{datasets,tokenizers}, logs/, .venv, __pycache__, modded-nanogpt
CLAUDE.md, AGENTS.md           agent-facing docs (this file + pointer)
```

## Setup

Canonical setup lives in [README.md](README.md) "Getting Started". Short version:

```bash
# venv — Git Bash on Windows uses Scripts/; WSL/Linux/macOS use bin/
python -m venv .venv
source .venv/Scripts/activate       # Git Bash / MSYS on Windows
# source .venv/bin/activate         # WSL, Linux, macOS

python -m pip install --upgrade pip
pip install -r requirements.txt     # root deps (minimal)
```

Notes:
- `torch` with CUDA wheels and `mlx` are NOT in `requirements.txt`. The **Runpod template pre-installs `torch`**. `mlx` is platform-specific (Apple Silicon only).
- `requirements.txt` is intentionally minimal (`numpy`, `tqdm`, `torch`, `huggingface-hub`, `kernels`, `setuptools`, `typing-extensions==4.15.0`, `datasets`, `tiktoken`, `sentencepiece`). Do not expand it.
- No `poetry`, `uv`, or `pipenv` — plain `pip` + `venv`.

## How Training is Configured (Env-Driven)

Every knob is an environment variable. The `Hyperparameters` class at [train_gpt.py:39-87](train_gpt.py#L39-L87) is the full list (~40 vars). A curated 17-var subset is in [.env.example](.env.example) for quick starts.

Canonical invocations:

```bash
# 1xH100 iteration (single process on Runpod)
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 leaderboard run (10-min cap enforced by MAX_WALLCLOCK_SECONDS=600 default)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# MLX local smoke on Apple Silicon (skips validation)
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

Caveat: `RANK`, `WORLD_SIZE`, `LOCAL_RANK` are set by `torchrun` — do not export them manually.

## Running on This Machine (Windows 11, RTX 4090)

Verified hardware and tooling on this host:

- **CPU/RAM:** 64 GB RAM, multi-core x86_64.
- **GPU:** 1× **NVIDIA GeForce RTX 4090, 24 GB VRAM**, driver 595.79, CUDA runtime 13.2.
- **Shells:** bash (Git Bash / MSYS) and PowerShell. Docs here use bash.
- **WSL2:** Ubuntu 24.04 installed (default distro, currently stopped). NVIDIA GPU passthrough works in WSL2 with an appropriate driver — this is a good place to run the CUDA training stack without Windows-path surprises.
- **Docker Desktop:** installed (currently stopped).
- **Python env caveat:** the currently-active Python is `c:\users\jjor3\appdata\local\programs\python\python310` with **`torch==2.11.0+cpu`** — CPU-only wheel. `torch.cuda.is_available()` returns `False` in this env. Before running `train_gpt.py`, either install a CUDA build into this env or (preferred) create a fresh `.venv` and install from `requirements.txt` plus a CUDA torch wheel.

**What runs locally with the CPU torch (no setup needed):**
- Python parse / byte-compile: `python -m py_compile train_gpt.py`
- Line-count check: `wc -l train_gpt.py train_gpt_mlx.py` (must be < 1500)
- Dependency resolution without installing heavy wheels: `pip install --dry-run -r requirements.txt`
- Reading and editing any file, git operations (read-only unless asked)
- `python -c "import json; json.load(open('records/.../submission.json'))"`

**What runs locally after installing a CUDA torch wheel:**
- Single-GPU smoke runs on the 4090: `torchrun --standalone --nproc_per_node=1 train_gpt.py` with *reduced* `TRAIN_BATCH_TOKENS` (baseline 524,288 is tuned for H100 80 GB — on 24 GB you'll typically need to drop this to e.g. 65,536 or enable gradient accumulation).
- Short iteration runs and ablations (`ITERATIONS=200`, `VAL_LOSS_EVERY=0`) — good for verifying a code change compiles and converges, not for competing.
- Dataset download (`python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1` for a minimal smoke subset; default 80 shards is 8 GB+).
- Running under WSL2 Ubuntu 24.04 if you prefer Linux toolchain (same 4090 via GPU passthrough).

**What still requires remote hardware:**
- **Leaderboard reproduction** — needs 8xH100 80 GB SXM (~$20/hr on Runpod template `y5cejece4j`). A 4090 cannot match H100 throughput, memory, or FP8 support.
- `train_gpt_mlx.py` — Apple Silicon / MLX only. Not runnable on this box.
- Full-scale 8B-token training — even as a single-GPU run, 10 min on 1×4090 ≠ 10 min on 8×H100.

**FP8 note:** some SOTA record submissions use FP8 paths that rely on H100 Transformer Engine. The 4090 (Ada) supports FP8 tensors but not the full H100 FP8 attention fused paths — some records may silently fall back or error. Treat local runs on the 4090 as smoke/dev only.

**Docker:** no `Dockerfile` in the repo. The challenge's "container" is the Runpod template (id `y5cejece4j`). Docker Desktop on this host is unused for this project.

## Tests / Lint / Type-check / CI

- **No test framework.** No `pytest`, no test directory. Do not add one.
- **No linter / formatter / type checker.** No `ruff`, `black`, `mypy`, `pyright`. Do not add one.
- **No CI workflow.** No `.github/workflows/`. Do not add one.
- **"Verification" is the 3 seed logs** accompanying a record submission. That is the convention.

## Submission Workflow (PR Anatomy)

A new PR adds exactly one folder under the right track:

- `records/track_10min_16mb/YYYY-MM-DD_DescriptiveName/` — leaderboard track (10-min / 16MB rule)
- `records/track_non_record_16mb/YYYY-MM-DD_DescriptiveName/` — experimental / unlimited compute

Required files:

| File | Purpose |
|---|---|
| `README.md` | Submission writeup: what you changed, why, results, compliance notes |
| `submission.json` | Machine-readable metadata. See shapes below. |
| `train_gpt.py` | The actual script that produced the result. Self-contained. Must compile and run from this folder. |
| `train_seed*.log` × 3 | Training logs. Seed `42` is nearly universal; the other two vary (e.g., `42 / 314 / 999` or `42 / 1337 / 2025`). |

Submission.json shapes:
- Minimal baseline: [records/track_10min_16mb/2026-03-17_NaiveBaseline/submission.json](records/track_10min_16mb/2026-03-17_NaiveBaseline/submission.json)
- Full (with compliance + attribution): [records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/submission.json](records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/submission.json)

Leaderboard bar: **≥ 0.005 nats over current SOTA at p < 0.01** across the 3 seeds (waived only for pure systems-optimization PRs).

Non-record submissions: same format, can be below SOTA, must be "unique or interesting".

## Common Mistakes (Read Before Editing)

- Exceeding the 16,000,000-byte cap because of a loose `requirements.txt` import you added to `train_gpt.py` in a record folder (every imported module's code size counts? No — library code is free per the FAQ, but the submission script itself must compress under the cap).
- Assuming the active Python env has CUDA torch. It does not (`torch==2.11.0+cpu`). `torch.cuda.is_available()` is `False` until you install a CUDA wheel, even though the 4090 is physically present.
- Trying to run the baseline 524,288 `TRAIN_BATCH_TOKENS` on the 4090 — it is sized for 80 GB H100s and will OOM on 24 GB. Reduce batch / enable accumulation for local smoke runs.
- Treating a 4090 smoke result as a leaderboard-relevant number. Numbers only count on the Runpod 8xH100 reference hardware.
- Editing a prior record's folder instead of creating a new dated folder. Prior submissions are frozen.
- Bumping `train_gpt.py` past 1500 lines. Hard stop in the file's docstring.
- Adding `pytest`, `ruff`, a `pyproject.toml`, a `Dockerfile`, or a GitHub Actions workflow "to help". None belong here.
- Committing `data/datasets/`, `data/tokenizers/`, or `logs/` — all in `.gitignore`; don't force-add them.
- Changing the tokenizer without proving `val_bpb` is computed correctly. Per the submission rules, tokenizer-changing PRs get extra scrutiny.
- Accessing validation data during training (e.g., compressing the val set into the artifact as a "paid prefix"). Instant disqualification.

## Verification Checklist (Local, CPU-torch env)

Run these after edits to confirm the repo is still well-formed. Copy-paste into bash:

```bash
# Syntax + byte-compile
python -m py_compile train_gpt.py
python -m py_compile train_gpt_mlx.py
python -m py_compile data/cached_challenge_fineweb.py
python -m py_compile data/download_hf_docs_and_tokenize.py

# 1500-line hard cap (both numbers must be < 1500)
wc -l train_gpt.py train_gpt_mlx.py

# Dependency resolution without downloading heavy wheels
pip install --dry-run -r requirements.txt

# Sample submission.json parses as valid JSON
python -c "import json; json.load(open('records/track_10min_16mb/2026-03-17_NaiveBaseline/submission.json'))"
python -c "import json; json.load(open('records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/submission.json'))"

# Git sanity (read-only)
git status
git log --oneline -5
```

If any step fails, document the failure and likely cause; do not "fix" Runpod-only assumptions (e.g., `torch` with CUDA wheels) without asking.

## When You're About to Modify `train_gpt.py` (Root)

- The 1500-line limit in the file's docstring is a **hard stop**, not a guideline.
- Add new knobs to the `Hyperparameters` class at [train_gpt.py:39-87](train_gpt.py#L39-L87) using the existing `os.environ.get(...)` + int/float/bool-cast pattern.
- Do not introduce imports that aren't in root `requirements.txt` unless you also update `requirements.txt` — and think twice before expanding root deps (per README FAQ, record-local `requirements.txt` is the right place).
- The root scripts are explicitly marked as "baseline launch pads, not SOTA configs" in the docstring. Prefer making your changes inside a new record folder instead of editing the root.
- After edits, run the Verification Checklist.

## Pointers

- [README.md](README.md) — full human-facing doc, getting started, leaderboard, FAQ, submission process.
- [data/README.md](data/README.md) — dataset / tokenizer workflows.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — modded-nanogpt attribution.
- [requirements.txt](requirements.txt) — minimal root deps.
- [.env.example](.env.example) — curated env-var starter.
- [AGENTS.md](AGENTS.md) — slim pointer for non-Claude agents.
