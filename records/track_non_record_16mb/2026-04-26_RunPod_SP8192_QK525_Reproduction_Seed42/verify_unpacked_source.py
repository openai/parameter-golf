"""Verify that train_gpt_unpacked.py matches the compressed wrapper payload."""

from __future__ import annotations

import base64
import hashlib
import lzma
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WRAPPER = ROOT / "train_gpt.py"
UNPACKED = ROOT / "train_gpt_unpacked.py"
HASH_FILE = ROOT / "train_gpt_unpacked.sha256"


def main() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    match = re.search(r'B\.b85decode\("(.*)"\)', wrapper, re.S)
    if match is None:
        raise SystemExit("could not locate base85 payload in train_gpt.py")

    decoded = lzma.decompress(
        base64.b85decode(match.group(1)),
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2}],
    )
    unpacked = UNPACKED.read_bytes()
    digest = hashlib.sha256(unpacked).hexdigest()
    expected = HASH_FILE.read_text(encoding="ascii").split()[0]

    if decoded != unpacked:
        raise SystemExit("train_gpt_unpacked.py does not match train_gpt.py payload")
    if digest != expected:
        raise SystemExit(f"hash mismatch: got {digest}, expected {expected}")

    print(f"verified {UNPACKED.name} sha256={digest}")


if __name__ == "__main__":
    main()
