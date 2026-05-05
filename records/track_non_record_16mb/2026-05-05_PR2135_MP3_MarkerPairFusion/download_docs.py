"""download_docs.py — fetch the canonical FineWeb-10B doc stream from Hugging Face.

Downloads ``docs_selected.jsonl`` (~45 GiB) and its sidecar manifest from the
``willdepueoai/parameter-golf`` HF dataset repo. The downloaded jsonl is the
input to ``prepare_caseops_data.py`` (step 1b in the pipeline; see README).

Usage::

    python3 download_docs.py
    # writes to ./data/datasets/{docs_selected.jsonl, docs_selected.source_manifest.json}
    # override target with: BASE_DIR=/abs/path python3 download_docs.py

Why a small wrapper instead of ``data/download_hf_docs_and_tokenize.py`` from
the upstream parameter-golf repo: we only need the raw jsonl. The upstream
script also tokenizes with multiple vocab specs (sp1024 / sp4096 / sp8192 /
byte260) which adds ~10-20 minutes that prepare_caseops_data.py replaces.
"""

import os
import time

from huggingface_hub import hf_hub_download


REPO_ID = os.environ.get("HF_REPO_ID", "willdepueoai/parameter-golf")
# BASE_DIR is the parent of the ``datasets/`` directory created on disk.
BASE_DIR = os.environ.get("BASE_DIR", "./data")
FILES = [
    "datasets/docs_selected.jsonl",
    "datasets/docs_selected.source_manifest.json",
]


def main() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    t0 = time.time()
    for fn in FILES:
        print("[" + time.strftime("%H:%M:%S") + "] downloading " + fn, flush=True)
        p = hf_hub_download(
            repo_id=REPO_ID,
            filename=fn,
            repo_type="dataset",
            local_dir=BASE_DIR,
        )
        sz = os.path.getsize(p)
        gib = round(sz / (1024 ** 3), 2)
        print("  -> " + p, flush=True)
        print("  size: " + str(sz) + " bytes (" + str(gib) + " GiB)", flush=True)
    print("[" + time.strftime("%H:%M:%S") + "] done in " + str(round(time.time() - t0, 1)) + "s", flush=True)


if __name__ == "__main__":
    main()
