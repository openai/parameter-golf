#!/usr/bin/env python3
import re
import sys
from pathlib import Path

VAL_RE = re.compile(r"val_bpb[^0-9]*([0-9]+\.[0-9]+)")
ARTIFACT_RE = re.compile(r"(?:artifact|compressed|roundtrip)[^0-9]*([0-9]{6,})", re.IGNORECASE)


def parse_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    vals = VAL_RE.findall(text)
    arts = ARTIFACT_RE.findall(text)
    val = vals[-1] if vals else "NA"
    art = arts[-1] if arts else "NA"
    return val, art


def main():
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")
    files = sorted(log_dir.glob("*.log"))
    if not files:
        print(f"No log files found in {log_dir}")
        return

    print("run_id,val_bpb,artifact_bytes")
    for f in files:
        val, art = parse_file(f)
        print(f"{f.stem},{val},{art}")


if __name__ == "__main__":
    main()
