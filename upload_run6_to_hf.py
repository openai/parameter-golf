"""Upload Run 6 (top_pr1855_hparams_s42_pergroup) artifacts to HuggingFace.

Pushes to shikhar007/parameter-golf-gram-ns under:
  models/top_pr1855_hparams_s42_pergroup.pt          (FP checkpoint)
  models/top_pr1855_hparams_s42_pergroup.int6.ptz    (quantized + pergroup blob)
  logs/top_pr1855_hparams_s42_pergroup.txt           (per-rank training log)
  logs/top_pr1855_hparams_s42_pergroup.stdout        (torchrun stdout/stderr)

Run after the training run finishes and the artifact files exist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "shikhar007/parameter-golf-gram-ns"
RUN_ID = "top_pr1855_hparams_s42_pergroup"
ARTIFACT_DIR = Path("artifacts") / RUN_ID
LOG_DIR = Path("logs")

UPLOADS: list[tuple[Path, str]] = [
    (ARTIFACT_DIR / "final_model.pt", f"models/{RUN_ID}.pt"),
    (ARTIFACT_DIR / "final_model.int6.ptz", f"models/{RUN_ID}.int6.ptz"),
    (ARTIFACT_DIR / f"{RUN_ID}.txt", f"logs/{RUN_ID}.txt"),
    (LOG_DIR / f"{RUN_ID}.stdout", f"logs/{RUN_ID}.stdout"),
]


def main() -> None:
    api = HfApi()
    me = api.whoami()
    print(f"Authenticated as {me['name']!r} (token role: {me['auth']['accessToken']['role']})")

    missing = [str(local) for local, _ in UPLOADS if not local.exists()]
    if missing:
        print("MISSING:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    for local, remote in UPLOADS:
        size = local.stat().st_size
        print(f"uploading {local} ({size:,} B) -> {REPO_ID}:{remote}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Add Run 6 ({RUN_ID}) artifact: {remote}",
        )
        print(f"  OK")


if __name__ == "__main__":
    main()
