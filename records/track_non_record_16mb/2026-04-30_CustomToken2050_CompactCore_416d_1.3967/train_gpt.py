"""Public non-record stub for the CustomToken2050 compact-core run.

The corresponding experiment used private pre-tokenized token IDs and a
proprietary tokenizer implementation. This file is intentionally limited to a
public-safe disclosure stub so the record folder has an executable entry point
without publishing private tokenizer or training-loss code.

This is not a leaderboard-reproducible record submission. It is an in-progress
non-record artifact documenting the observed run result and research direction.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    submission = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    print("CustomToken2050 compact-core non-record artifact")
    print(f"val_bpb: {submission['val_bpb']}")
    print(f"val_loss: {submission['val_loss']}")
    print(f"steps: {submission['steps']}")
    print(f"batch_tokens: {submission['batch_tokens']}")
    print()
    print("Reproduction boundary:")
    print("  This public folder excludes the proprietary tokenizer and private")
    print("  training-loss implementation used to produce the pre-tokenized IDs.")
    print("  See README.md and train.log for the submitted non-record evidence.")


if __name__ == "__main__":
    main()
