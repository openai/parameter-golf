#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PATTERNS = {
    "roundtrip": re.compile(r"final_int8_zlib_roundtrip_exact val_loss:(\S+) val_bpb:(\S+)"),
    "ttt": re.compile(r"final_int8_ttt_lora val_loss:(\S+) val_bpb:(\S+)"),
    "artifact": re.compile(r"Total submission size int8\+zlib: (\d+) bytes"),
    "step_avg": re.compile(r"step:(\d+)/\d+ val_loss:(\S+) val_bpb:(\S+) train_time:(\d+)ms step_avg:(\S+)ms"),
    "peak_mem": re.compile(r"peak memory allocated: (\d+) MiB reserved: (\d+) MiB"),
}


def maybe_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def parse_log(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"log": str(path), "missing": True}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, object] = {"log": str(path)}
    if m := PATTERNS["roundtrip"].search(text):
        val_loss = maybe_float(m.group(1))
        val_bpb = maybe_float(m.group(2))
        if val_loss is not None and val_bpb is not None:
            out["roundtrip_val_loss"] = val_loss
            out["roundtrip_val_bpb"] = val_bpb
    if m := PATTERNS["ttt"].search(text):
        val_loss = maybe_float(m.group(1))
        val_bpb = maybe_float(m.group(2))
        if val_loss is not None and val_bpb is not None:
            out["ttt_val_loss"] = val_loss
            out["ttt_val_bpb"] = val_bpb
    if m := PATTERNS["artifact"].search(text):
        out["artifact_bytes"] = int(m.group(1))
    if m := PATTERNS["peak_mem"].search(text):
        out["peak_alloc_mib"] = int(m.group(1))
        out["peak_reserved_mib"] = int(m.group(2))
    step_matches = PATTERNS["step_avg"].findall(text)
    if step_matches:
        step, val_loss, val_bpb, train_time_ms, step_avg = step_matches[-1]
        parsed_val_loss = maybe_float(val_loss)
        parsed_val_bpb = maybe_float(val_bpb)
        parsed_step_avg = maybe_float(step_avg)
        if parsed_val_loss is not None and parsed_val_bpb is not None and parsed_step_avg is not None:
            out["last_val_step"] = int(step)
            out["last_val_loss"] = parsed_val_loss
            out["last_val_bpb"] = parsed_val_bpb
            out["train_time_ms"] = int(train_time_ms)
            out["step_avg_ms"] = parsed_step_avg
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python3 scripts/parse_run.py <log1> [log2 ...]", file=sys.stderr)
        return 1
    rows = [parse_log(Path(arg)) for arg in sys.argv[1:]]
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
