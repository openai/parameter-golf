#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import eval_doc_cache


def main() -> None:
    train_script = str(Path(__file__).with_name("train_gpt.py"))
    if "--train-script" not in sys.argv:
        sys.argv[1:1] = ["--train-script", train_script]
    eval_doc_cache.main()


if __name__ == "__main__":
    main()
