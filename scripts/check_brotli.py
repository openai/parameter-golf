"""Build-input and installed-artifact checks for the Brotli security update."""

import argparse
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def requirement():
    pins = set()
    for path in (ROOT / "records").rglob("requirements.txt"):
        for line in path.read_text().splitlines():
            if line.strip().lower().startswith("brotli"):
                pins.add(line.strip())
    assert len(pins) == 1, f"Record Brotli pins disagree: {pins}"
    pin = pins.pop()
    assert pin.startswith("brotli @ https://github.com/google/brotli/archive/")
    assert "#sha256=" in pin
    assert pin in (ROOT / "scripts/run_1868_rerun.py").read_text()
    return pin


def verify_decoder():
    import _brotli
    import brotli

    decoder = ctypes.CDLL(_brotli.__file__)
    decoder.BrotliDecoderErrorString.argtypes = [ctypes.c_int]
    decoder.BrotliDecoderErrorString.restype = ctypes.c_char_p
    # Upstream retained version 1.2.0, so check the native repair itself.
    assert decoder.BrotliDecoderErrorString(-17) == b"_ERROR_FORMAT_BLOCK_SWITCH"
    for data in (b"", bytes(range(256)) * 1024, b"parameter golf" * 8192):
        assert brotli.decompress(brotli.compress(data)) == data
    print("Patched native Brotli decoder and round trips verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-requirement", type=Path)
    args = parser.parse_args()
    pin = requirement()
    if args.write_requirement:
        args.write_requirement.write_text(pin + "\n")
    else:
        verify_decoder()
