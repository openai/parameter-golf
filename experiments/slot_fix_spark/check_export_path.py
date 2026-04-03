from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "train_gpt.py"
SAFEPOINT = Path("/home/frosty40/parameter-golf-lab/neural/VAULT: NUERAL SAFE/1.110_15.5mb_train_gpt.py")


def _extract_export_block(text: str) -> str:
    start = text.find("full_state_dict = base_model.state_dict()")
    if start < 0:
        raise RuntimeError("missing safepoint export entrypoint start")
    stop = text.find("sd_cpu =", start)
    if stop < 0:
        raise RuntimeError("missing safepoint export entrypoint stop")
    raw = text[start:stop]
    raw = re.sub(r"\s+", "", raw)
    return raw.strip()


def main() -> int:
    target = _extract_export_block(TARGET.read_text(encoding="utf-8"))
    safe = _extract_export_block(SAFEPOINT.read_text(encoding="utf-8"))
    if target != safe:
        raise SystemExit(
            "export-structure mismatch: SLOT edits touched pre-export lines in this file."
        )
    print("export block structurally matches safepoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
