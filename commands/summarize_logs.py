#!/usr/bin/env python3
"""
Summarize Parameter Golf train logs. Prefer leaderboard-style sliding BPB
(final_int6_sliding_window_exact) over plain val_bpb or roundtrip lines.
"""
import re
import sys
from pathlib import Path

# Leaderboard-style metric (see record README "Sliding BPB")
SLIDING_EXACT = re.compile(
    r"final_int6_sliding_window_exact val_loss:[0-9.e+-]+ val_bpb:([0-9.]+)"
)
SLIDING_S64_EXACT = re.compile(
    r"final_int6_sliding_window_s64_exact val_loss:[0-9.e+-]+ val_bpb:([0-9.]+)"
)
ROUNDTRIP_EXACT = re.compile(
    r"final_int6_roundtrip_exact val_loss:[0-9.e+-]+ val_bpb:([0-9.]+)"
)
VAL_ANY = re.compile(r"val_bpb:([0-9.]+)")
ARTIFACT_RE = re.compile(
    r"Total submission size int6\+lzma:\s*([0-9]+)\s*bytes?", re.IGNORECASE
)


def parse_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    val = "NA"
    for pattern in (SLIDING_EXACT, SLIDING_S64_EXACT, ROUNDTRIP_EXACT):
        m = pattern.search(text)
        if m:
            val = m.group(1)
            break
    if val == "NA":
        vals = VAL_ANY.findall(text)
        val = vals[-1] if vals else "NA"

    art_m = ARTIFACT_RE.search(text)
    art = art_m.group(1) if art_m else "NA"

    return val, art


def main():
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")
    files = sorted(log_dir.glob("*.log"))
    if not files:
        print(f"No log files found in {log_dir}")
        return

    print("run_id,sliding_val_bpb_or_best_available,artifact_bytes_int6_lzma")
    for f in files:
        val, art = parse_file(f)
        print(f"{f.stem},{val},{art}")


if __name__ == "__main__":
    main()
