#!/usr/bin/env bash
set -euo pipefail

# End-to-end submit helper after run_remote_fast.sh finishes.
# It creates a records folder from the best run, commits, pushes, and optionally opens a PR.
#
# Required env for push/PR:
#   FORK_REMOTE   (default: origin)
#   SUBMIT_BRANCH (default: auto/pg-record-<timestamp>)
#   AUTHOR_NAME   (default: your git user.name)
#   GITHUB_ID     (required for submission.json)
#
# Optional:
#   BASE_REPO     (default: openai/parameter-golf)
#   BASE_BRANCH   (default: main)
#   OPEN_PR       (default: 1)
#   RUN_DIR       (default: latest runs/remote_fast_*)

FORK_REMOTE="${FORK_REMOTE:-fork}"
BASE_REPO="${BASE_REPO:-openai/parameter-golf}"
BASE_BRANCH="${BASE_BRANCH:-main}"
OPEN_PR="${OPEN_PR:-1}"
RUN_DIR="${RUN_DIR:-}"
SUBMIT_BRANCH="${SUBMIT_BRANCH:-auto/pg-record-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(ls -dt runs/remote_fast_* | head -n1)"
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}"
  exit 1
fi

SUMMARY_JSON="${RUN_DIR}/summary.json"
MANIFEST_JSON="${RUN_DIR}/candidate_manifest.json"
if [[ ! -f "${SUMMARY_JSON}" ]]; then
  echo "Missing summary file: ${SUMMARY_JSON}"
  exit 1
fi
if [[ ! -f "${MANIFEST_JSON}" ]]; then
  echo "Missing candidate manifest: ${MANIFEST_JSON}"
  exit 1
fi

AUTHOR_NAME="${AUTHOR_NAME:-$(git config user.name || true)}"
if [[ -z "${AUTHOR_NAME}" ]]; then
  AUTHOR_NAME="YOUR_NAME"
fi
GITHUB_ID="${GITHUB_ID:-}"
if [[ -z "${GITHUB_ID}" ]]; then
  echo "GITHUB_ID is required (export GITHUB_ID=your_handle)."
  exit 1
fi

python3 - << 'PY'
import json
import os
import pathlib
import statistics
from datetime import datetime

run_dir = pathlib.Path(os.environ["RUN_DIR"])
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
manifest = json.loads((run_dir / "candidate_manifest.json").read_text(encoding="utf-8"))

runs = summary.get("runs", [])
if not runs:
    raise SystemExit("No runs in summary.json")

best = min(runs, key=lambda x: x["val_bpb"])
best_candidate = best["candidate"]
if best_candidate not in manifest:
    raise SystemExit(f"Best candidate {best_candidate} missing in manifest")

cand_runs = [r for r in runs if r["candidate"] == best_candidate]
vals = [r["val_bpb"] for r in cand_runs]
mean_bpb = statistics.mean(vals)
std_bpb = statistics.stdev(vals) if len(vals) > 1 else 0.0

script_rel = manifest[best_candidate]["script"]
script_path = pathlib.Path(script_rel)
if not script_path.exists():
    raise SystemExit(f"Candidate script not found: {script_path}")

stamp = datetime.now().strftime("%Y-%m-%d")
folder = f"{stamp}_AutoSweep_{best_candidate}"
out_dir = pathlib.Path("records") / "track_10min_16mb" / folder
out_dir.mkdir(parents=True, exist_ok=True)

(out_dir / "train_gpt.py").write_bytes(script_path.read_bytes())

rows = []
for r in cand_runs:
    seed = pathlib.Path(r["log"]).stem.replace("seed", "")
    src = pathlib.Path(r["log"])
    dst = out_dir / f"train_seed{seed}.log"
    if src.exists():
        dst.write_bytes(src.read_bytes())
    rows.append((seed, r["val_bpb"], r.get("total_bytes"), str(dst)))

readme = [
    f"# AutoSweep Submission: {best_candidate}",
    "",
    "Generated from run_remote_fast.sh + submit_remote_fast.sh.",
    "",
    "## Metrics",
    "",
    f"- Best val_bpb: {best['val_bpb']}",
    f"- Mean val_bpb: {mean_bpb}",
    f"- Std val_bpb: {std_bpb}",
    "",
    "## Seed Runs",
    "",
    "| Seed | val_bpb | total_bytes | log |",
    "|------|---------|-------------|-----|",
]
for seed, bpb, total, path in rows:
    readme.append(f"| {seed} | {bpb} | {total} | {path} |")

(out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

submission = {
    "name": f"AutoSweep {best_candidate}",
    "val_loss": round(mean_bpb, 6),
    "bytes_total": int(best.get("total_bytes") or 15900000),
    "blurb": f"Auto-generated from parallel sweep. Best candidate: {best_candidate}. mean={mean_bpb:.6f}, std={std_bpb:.6f}.",
    "author": os.environ.get("AUTHOR_NAME", "YOUR_NAME"),
    "github_id": os.environ["GITHUB_ID"],
    "date": datetime.now().strftime("%Y-%m-%d"),
}
(out_dir / "submission.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")

print(str(out_dir))
PY

SUBMISSION_DIR="$(RUN_DIR="${RUN_DIR}" AUTHOR_NAME="${AUTHOR_NAME}" GITHUB_ID="${GITHUB_ID}" python3 - << 'PY'
import json
import os
import pathlib
from datetime import datetime
run_dir = pathlib.Path(os.environ["RUN_DIR"])
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
best = min(summary["runs"], key=lambda x: x["val_bpb"])
print(str(pathlib.Path("records") / "track_10min_16mb" / f"{datetime.now().strftime('%Y-%m-%d')}_AutoSweep_{best['candidate']}"))
PY
)"

echo "Submission folder: ${SUBMISSION_DIR}"

git checkout -b "${SUBMIT_BRANCH}"
git add "${SUBMISSION_DIR}"
git commit -m "Add auto-sweep submission: ${SUBMISSION_DIR##*/}"
git push -u "${FORK_REMOTE}" "${SUBMIT_BRANCH}"

if [[ "${OPEN_PR}" == "1" ]]; then
  if command -v gh >/dev/null 2>&1; then
    gh pr create \
      --repo "${BASE_REPO}" \
      --base "${BASE_BRANCH}" \
      --head "${SUBMIT_BRANCH}" \
      --title "AutoSweep submission: ${SUBMISSION_DIR##*/}" \
      --body "Automated submission generated from multi-candidate sweep in ${RUN_DIR}. Includes train_gpt.py, train logs, README, and submission.json."
  else
    echo "gh CLI not found; skipping PR creation."
  fi
fi

echo "Done."
