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
    data = json.loads(path.read_text())
    cfg = data.get("config", {})
    eval_context = data.get("eval_context", {})
    eval_static = data.get("eval_static_residual", {})
    eval_online = data.get("eval_online_residual", {})
    return {
        "run": str(path.parent.name),
        "spine": cfg.get("spine_variant", ""),
        "xsa_last_n": cfg.get("xsa_last_n", ""),
        "seq_len": cfg.get("seq_len", ""),
        "model_dim": cfg.get("model_dim", ""),
        "layers": cfg.get("num_layers", ""),
        "rank": cfg.get("residual_rank", ""),
        "table": cfg.get("residual_table_size", ""),
        "context_bpb": eval_context.get("val_bpb", ""),
        "static_bpb": eval_static.get("val_bpb", ""),
        "online_bpb": eval_online.get("val_bpb", ""),
        "delta_static": data.get("eval_delta_static_bpb", ""),
        "delta_online": data.get("eval_delta_online_bpb", ""),
        "artifact_mb": (float(data.get("total_artifact_bytes", 0.0)) / (1024.0 * 1024.0)),
    }


def render_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "run",
        "spine",
        "xsa_last_n",
        "seq_len",
        "model_dim",
        "layers",
        "rank",
        "table",
        "context_bpb",
        "static_bpb",
        "online_bpb",
        "delta_static",
        "delta_online",
        "artifact_mb",
    ]
    widths = {header: len(header) for header in headers}
    rendered_rows: list[dict[str, str]] = []
    for row in rows:
        rendered: dict[str, str] = {}
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                if "delta" in header:
                    text = f"{value:+.6f}"
                elif "bpb" in header or header == "artifact_mb":
                    text = f"{value:.6f}"
                else:
                    text = f"{value}"
            else:
                text = str(value)
            rendered[header] = text
            widths[header] = max(widths[header], len(text))
        rendered_rows.append(rendered)
    lines = []
    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    sep_line = "  ".join("-" * widths[header] for header in headers)
    lines.append(header_line)
    lines.append(sep_line)
    for row in rendered_rows:
        lines.append("  ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: summarize_v2a_runs.py <run-dir-or-result.json> [...]", file=sys.stderr)
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
