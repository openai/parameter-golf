#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


CONTROL_RUN = "twice_eval2048_ttt1024_clean2"
PROFILES = ["drope_eval", "yarn_eval", "mtp_low", "muon_balance", "hybrid_delta"]
ARTIFACT_CAP = 16_000_000


def parse_logs(paths: list[str]) -> list[dict[str, object]]:
    out = subprocess.check_output([sys.executable, "scripts/parse_run.py", *paths], text=True)
    return json.loads(out)


def score_key(row: dict[str, object]) -> tuple[float, float, int, float]:
    return (
        float(row.get("ttt_val_bpb", math.inf)),
        float(row.get("roundtrip_val_bpb", math.inf)),
        int(row.get("artifact_bytes", 10**12)),
        float(row.get("step_avg_ms", math.inf)),
    )


def run_id_from_log(path: str) -> str:
    return Path(path).stem


def promote(row: dict[str, object], control: dict[str, object] | None) -> tuple[bool, str]:
    artifact = int(row.get("artifact_bytes", ARTIFACT_CAP + 1))
    if artifact > ARTIFACT_CAP:
        return False, "artifact cap exceeded"
    if control is None:
        return True, "no control available"
    row_ttt = float(row.get("ttt_val_bpb", math.inf))
    row_roundtrip = float(row.get("roundtrip_val_bpb", math.inf))
    ctl_ttt = float(control.get("ttt_val_bpb", math.inf))
    ctl_roundtrip = float(control.get("roundtrip_val_bpb", math.inf))
    if run_id_from_log(str(row["log"])) == "hybrid_delta" and (row_ttt < ctl_ttt or row_roundtrip < ctl_roundtrip):
        return True, "hybrid delta beat control on at least one final metric"
    if row_ttt < ctl_ttt or row_roundtrip < ctl_roundtrip:
        return True, "beat control on at least one final metric"
    return False, "did not beat control"


def main() -> int:
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = [f"logs/{name}.txt" for name in [CONTROL_RUN, *PROFILES] if Path(f"logs/{name}.txt").exists()]
    rows = parse_logs(paths)
    for row in rows:
        row["run_id"] = run_id_from_log(str(row["log"]))
    control = next((row for row in rows if row["run_id"] == CONTROL_RUN), None)
    ranked = sorted((row for row in rows if row["run_id"] != CONTROL_RUN), key=score_key)
    result = []
    for row in ranked:
        ok, reason = promote(row, control)
        row = dict(row)
        row["promote"] = ok
        row["promotion_reason"] = reason
        result.append(row)
    print(json.dumps({"control": control, "ranked": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
