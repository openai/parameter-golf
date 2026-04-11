import re
import sys
from pathlib import Path

patterns = {
    "last_val": re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)"),
    "roundtrip": re.compile(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)"),
    "ttt": re.compile(r"final_int8_ttt_lora val_loss:([0-9.]+) val_bpb:([0-9.]+)"),
    "size": re.compile(r"Total submission size int8\+zlib: (\d+) bytes"),
    "serialized": re.compile(r"Serialized model int8\+zlib: (\d+) bytes"),
    "stop": re.compile(r"stopping_early: wallclock_cap train_time:([0-9.]+)ms step:(\d+)/(\d+)"),
}

def extract(text: str):
    out = {}

    last_vals = patterns["last_val"].findall(text)
    if last_vals:
        step, total, loss, bpb = last_vals[-1]
        out["last_step"] = int(step)
        out["last_val_loss"] = float(loss)
        out["last_val_bpb"] = float(bpb)

    m = patterns["roundtrip"].search(text)
    if m:
        out["roundtrip_val_loss"] = float(m.group(1))
        out["roundtrip_val_bpb"] = float(m.group(2))

    m = patterns["ttt"].search(text)
    if m:
        out["ttt_val_loss"] = float(m.group(1))
        out["ttt_val_bpb"] = float(m.group(2))

    m = patterns["size"].search(text)
    if m:
        out["total_submission_bytes"] = int(m.group(1))

    m = patterns["serialized"].search(text)
    if m:
        out["model_blob_bytes"] = int(m.group(1))

    m = patterns["stop"].search(text)
    if m:
        out["train_time_ms"] = float(m.group(1))
        out["stop_step"] = int(m.group(2))

    return out

def main():
    if len(sys.argv) < 2:
        print("usage: python tools/extract_metrics.py <log1> [<log2> ...]")
        raise SystemExit(1)

    for path_str in sys.argv[1:]:
        path = Path(path_str)
        text = path.read_text(errors="ignore")
        metrics = extract(text)
        print(f"\n=== {path.name} ===")
        for k in sorted(metrics):
            print(f"{k}: {metrics[k]}")

if __name__ == "__main__":
    main()