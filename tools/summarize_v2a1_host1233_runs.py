#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def resolve_result_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        candidate = path / "result.json"
        if candidate.exists():
            return candidate
    return path


def load_row(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    context = data.get("eval_context", {}) or {}
    online = data.get("eval_online_residual", {}) or {}
    artifact = data.get("artifact", {}) or {}
    residual = data.get("residual", {}) or {}
    return {
        "run": path.parent.name,
        "seed": data.get("seed", ""),
        "context_bpb": context.get("val_bpb", ""),
        "online_bpb": online.get("val_bpb", ""),
        "delta_online": online.get("delta_bpb", ""),
        "prob_dev": online.get("prob_sum_max_deviation", ""),
        "table": residual.get("table_size", ""),
        "rank": residual.get("rank", ""),
        "artifact_mb": float(artifact.get("total_bytes", 0.0)) / (1024.0 * 1024.0),
    }


def render_table(rows: list[dict[str, object]]) -> str:
    headers = ["run", "seed", "context_bpb", "online_bpb", "delta_online", "prob_dev", "table", "rank", "artifact_mb"]
    widths = {header: len(header) for header in headers}
    rendered_rows: list[dict[str, str]] = []
    for row in rows:
        rendered: dict[str, str] = {}
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                if header.startswith("delta"):
                    text = f"{value:+.6f}"
                elif "bpb" in header or header == "artifact_mb":
                    text = f"{value:.6f}"
                elif "prob" in header:
                    text = f"{value:.3e}"
                else:
                    text = str(value)
            else:
                text = str(value)
            rendered[header] = text
            widths[header] = max(widths[header], len(text))
        rendered_rows.append(rendered)
    lines = []
    lines.append("  ".join(header.ljust(widths[header]) for header in headers))
    lines.append("  ".join("-" * widths[header] for header in headers))
    for row in rendered_rows:
        lines.append("  ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: summarize_v2a1_host1233_runs.py <run-dir-or-result.json> [...]", file=sys.stderr)
        return 2
    rows = []
    for arg in argv[1:]:
        path = resolve_result_path(arg)
        if not path.exists():
            print(f"missing result path: {path}", file=sys.stderr)
            return 1
        rows.append(load_row(path))
    print(render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
