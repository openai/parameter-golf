#!/usr/bin/env python3
"""Parse a training log and extract signals for the sweep harness.

Usage: python3 parse_log.py <logfile>
Prints one CSV-safe line:
  train_loss,pre_quant_bpb,quant_bpb,sliding_bpb,ttt_bpb,delta_bpb,tok_s,peak_mem_gb,failure_class

All fields empty-string when not found. failure_class non-empty implies a crash.
"""
import re
import sys
from pathlib import Path


def _last_float(pattern: str, txt: str, flags=re.IGNORECASE) -> str:
    m = re.findall(pattern, txt, flags)
    return m[-1] if m else ""


def parse(path: Path) -> dict:
    txt = path.read_text(errors="replace") if path.exists() else ""
    size_cap_bytes = 16_000_000
    out = {
        "train_loss": "",
        "pre_quant_bpb": "",
        "quant_bpb": "",
        "sliding_bpb": "",
        "ttt_bpb": "",
        "delta_bpb": "",
        "tok_s": "",
        "peak_mem_gb": "",
        "failure_class": "",
    }

    # Final train_loss line, e.g. "150/150 train_loss: 5.1548 train_time: 5.2m tok/s: 15798"
    m = re.findall(
        r"\d+/\d+\s+train_loss:\s+([0-9]+\.[0-9]+)\s+train_time:\s*\S+\s+tok/s:\s*([0-9]+)",
        txt,
    )
    if m:
        out["train_loss"], out["tok_s"] = m[-1]

    # timed_eval lines, e.g. "pre-quantization post-ema val_loss:X val_bpb:0.12345678 eval_time:..."
    def _bpb_for_label(label_re: str) -> str:
        pat = rf"{label_re}\s+val_loss:[0-9.]+\s+val_bpb:([0-9]+\.[0-9]+)"
        return _last_float(pat, txt)

    out["pre_quant_bpb"] = _bpb_for_label(r"pre-quantization post-ema")
    out["quant_bpb"] = _bpb_for_label(r"quantized(?!_sliding|_ttt|_etlb)")
    out["sliding_bpb"] = _bpb_for_label(r"quantized_sliding_window")
    out["ttt_bpb"] = _bpb_for_label(r"quantized_ttt")

    # Fallback: any val_bpb (for step-level logs like "100/150 val_loss: X val_bpb: Y")
    if not out["pre_quant_bpb"] and not out["quant_bpb"]:
        fb = _last_float(r"val[_ ]bpb[=:\s]+([0-9]+\.[0-9]+)", txt)
        if fb:
            # Can't tell which stage — put in pre_quant as best guess
            out["pre_quant_bpb"] = fb

    # Delta BPB = quant - pre_quant (degradation from quantization)
    if out["quant_bpb"] and out["pre_quant_bpb"]:
        try:
            out["delta_bpb"] = f"{float(out['quant_bpb']) - float(out['pre_quant_bpb']):+.5f}"
        except ValueError:
            pass

    # peak memory: "peak memory allocated: 10245 MiB reserved: ..."
    m = re.search(r"peak memory allocated:\s*([0-9]+)\s*MiB", txt, re.IGNORECASE)
    if m:
        out["peak_mem_gb"] = f"{int(m.group(1))/1024:.2f}"

    # Artifact-size compliance gate
    m = re.findall(r"Total submission size quantized\+\w+:\s*([0-9]+)\s*bytes", txt, re.IGNORECASE)
    if m:
        try:
            total_bytes = int(m[-1])
            if total_bytes > size_cap_bytes:
                out["failure_class"] = "oversize"
        except ValueError:
            pass

    # failure classification
    low = txt.lower()
    if "cuda out of memory" in low or "outofmemoryerror" in low:
        out["failure_class"] = "oom"
    elif re.search(r"(loss|grad)[^\n]{0,40}nan|nan[^\n]{0,40}(loss|grad)", low):
        out["failure_class"] = "nan"
    elif "modulenotfounderror" in low or "importerror" in low:
        out["failure_class"] = "import_error"
    elif "traceback" in low and not (out["quant_bpb"] or out["pre_quant_bpb"]):
        out["failure_class"] = "crash"

    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: parse_log.py <logfile>", file=sys.stderr)
        return 2
    r = parse(Path(sys.argv[1]))
    # CSV order matches header in run_experiment.sh append_row()
    print(
        f"{r['train_loss']},{r['pre_quant_bpb']},{r['quant_bpb']},"
        f"{r['sliding_bpb']},{r['ttt_bpb']},{r['delta_bpb']},"
        f"{r['tok_s']},{r['peak_mem_gb']},{r['failure_class']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
