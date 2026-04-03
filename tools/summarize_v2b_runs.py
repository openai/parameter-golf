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
    cfg = data.get("config", {}) or {}
    context = data.get("eval_context", {}) or {}
    online = data.get("eval_online_persistent_hidden", {}) or {}
    memory = online.get("persistent_memory", {}) or {}
    lookup_flops = float(online.get("memory_lookup_flops_estimate", 0.0))
    update_flops = float(online.get("memory_update_flops_estimate", 0.0))
    maintenance_flops = float(online.get("memory_maintenance_flops_estimate", 0.0))
    total_flops = float(online.get("memory_total_flops_estimate", lookup_flops + update_flops + maintenance_flops))
    return {
        "run": path.parent.name,
        "seq_len": cfg.get("seq_len", ""),
        "model_dim": cfg.get("model_dim", ""),
        "layers": cfg.get("num_layers", ""),
        "table": cfg.get("memory_table_size", ""),
        "read_gate": cfg.get("memory_min_read_count", ""),
        "maint_passes": cfg.get("maintenance_passes", ""),
        "maint_mode": cfg.get("maintenance_mode", ""),
        "maint_step": cfg.get("maintenance_step_size", ""),
        "maint_slots": cfg.get("maintenance_max_slots", ""),
        "maint_metric": cfg.get("maintenance_metric", ""),
        "grad_mix": cfg.get("maintenance_grad_mix", ""),
        "replay_depth": cfg.get("maintenance_replay_depth", ""),
        "replay_cand": cfg.get("maintenance_replay_candidates", ""),
        "use_grad": cfg.get("maintenance_use_grad", ""),
        "context_bpb": context.get("val_bpb", ""),
        "online_bpb": online.get("val_bpb", ""),
        "delta_online": data.get("eval_delta_online_bpb", ""),
        "lookup_gflop": lookup_flops / 1e9,
        "update_gflop": update_flops / 1e9,
        "maint_gflop": maintenance_flops / 1e9,
        "total_gflop": total_flops / 1e9,
        "active_slots": online.get("active_slots_mean", ""),
        "readable_slots": online.get("readable_slots_mean", ""),
        "delta_norm": online.get("delta_norm_mean", ""),
        "readable_fraction": memory.get("readable_fraction", ""),
        "replay_fraction": memory.get("replay_fraction", ""),
        "loss_ema": memory.get("mean_loss_ema", ""),
        "resident_mb": memory.get("resident_mb", ""),
        "artifact_mb": float(data.get("model_artifact_bytes", 0.0)) / (1024.0 * 1024.0),
    }


def format_value(header: str, value: object) -> str:
    if isinstance(value, float):
        if header.startswith("delta"):
            return f"{value:+.6f}"
        if "bpb" in header:
            return f"{value:.6f}"
        if header.endswith("gflop"):
            return f"{value:.3f}"
        if header.endswith("_mb"):
            return f"{value:.3f}"
        if header in {"maint_step", "grad_mix"}:
            return f"{value:.3f}"
        if "slots" in header or "fraction" in header or "norm" in header or header == "read_gate":
            return f"{value:.3f}"
        return str(value)
    return str(value)


def render_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "run",
        "seq_len",
        "model_dim",
        "layers",
        "table",
        "read_gate",
        "maint_passes",
        "maint_mode",
        "maint_step",
        "maint_slots",
        "maint_metric",
        "grad_mix",
        "replay_depth",
        "replay_cand",
        "use_grad",
        "context_bpb",
        "online_bpb",
        "delta_online",
        "lookup_gflop",
        "update_gflop",
        "maint_gflop",
        "total_gflop",
        "active_slots",
        "readable_slots",
        "delta_norm",
        "readable_fraction",
        "replay_fraction",
        "loss_ema",
        "resident_mb",
        "artifact_mb",
    ]
    widths = {header: len(header) for header in headers}
    rendered_rows: list[dict[str, str]] = []
    for row in rows:
        rendered = {header: format_value(header, row.get(header, "")) for header in headers}
        rendered_rows.append(rendered)
        for header, text in rendered.items():
            widths[header] = max(widths[header], len(text))
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]
    for row in rendered_rows:
        lines.append("  ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: summarize_v2b_runs.py <run-dir-or-result.json> [...]", file=sys.stderr)
        return 2
    rows = []
    for arg in argv[1:]:
        path = resolve_result_path(arg)
        if not path.exists():
            print(f"missing result path: {path}", file=sys.stderr)
            return 1
        rows.append(load_row(path))
    rows.sort(key=lambda row: float(row.get("delta_online", 0.0)))
    print(render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
