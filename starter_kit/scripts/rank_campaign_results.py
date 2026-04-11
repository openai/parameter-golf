#!/usr/bin/env python3
import argparse
import glob
import re
from pathlib import Path

ROUNDTRIP_RE = re.compile(r"final_int8_zlib_roundtrip(?:_exact)?\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)")
SIZE_RE = re.compile(r"Total submission size int8\+zlib:\s*([0-9]+)\s*bytes")


def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    roundtrip_matches = ROUNDTRIP_RE.findall(text)
    if not roundtrip_matches:
        return None
    val_loss_s, val_bpb_s = roundtrip_matches[-1]
    size_matches = SIZE_RE.findall(text)
    total_size = int(size_matches[-1]) if size_matches else None
    return {
        "run_id": path.stem,
        "val_loss": float(val_loss_s),
        "val_bpb": float(val_bpb_s),
        "bytes_total": total_size,
        "log_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank campaign runs by final roundtrip val_bpb.")
    parser.add_argument("--logs-glob", default="logs/R*.log")
    args = parser.parse_args()

    rows = []
    for p in sorted(glob.glob(args.logs_glob)):
        parsed = parse_log(Path(p))
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        print("No completed run logs with final_int8_zlib_roundtrip found.")
        return

    rows.sort(key=lambda r: r["val_bpb"])

    print("Ranked campaign results (lower val_bpb is better):")
    for i, r in enumerate(rows, start=1):
        size_txt = str(r["bytes_total"]) if r["bytes_total"] is not None else "n/a"
        print(
            f"{i:02d}. {r['run_id']}  val_bpb={r['val_bpb']:.6f}  "
            f"val_loss={r['val_loss']:.6f}  bytes_total={size_txt}"
        )


if __name__ == "__main__":
    main()
