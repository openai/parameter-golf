#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import lzma
import math
import zlib
from pathlib import Path

try:
    import zstandard
except ImportError:
    zstandard = None

import torch


ARTIFACT_LIMIT_BYTES = 16_000_000


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _classify_tensor(name: str) -> str:
    lowered = name.lower()
    if "tok_emb" in lowered or "lm_head" in lowered:
        return "embeddings"
    if "bigram" in lowered or "hash" in lowered:
        return "bigram_tables"
    if "qo_bank" in lowered or "kv_bank" in lowered or ".attn." in lowered or "attn_" in lowered:
        return "attention"
    if "mlp_up_bank" in lowered or "mlp_down_bank" in lowered or ".mlp." in lowered:
        return "mlp"
    return "norms_scales_other"


def _strip_quant_suffix(name: str) -> str:
    for suffix in (".q", ".scale"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _summarize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for name, tensor in state_dict.items():
        bucket = _classify_tensor(_strip_quant_suffix(name))
        entry = summary.setdefault(bucket, {"tensor_count": 0, "param_count": 0, "measured_bytes": 0})
        entry["tensor_count"] += 1
        entry["param_count"] += int(tensor.numel())
        entry["measured_bytes"] += _tensor_nbytes(tensor)
    return summary


def _load_raw_state_dict(run_dir: Path) -> dict[str, torch.Tensor] | None:
    raw_model = run_dir / "final_model.pt"
    if not raw_model.is_file():
        return None
    payload = torch.load(raw_model, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        return {name: tensor for name, tensor in payload.items() if isinstance(tensor, torch.Tensor)}
    return None


def _decompress_quant_blob(blob: bytes) -> bytes:
    errors: list[str] = []
    try:
        return lzma.decompress(blob)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"lzma:{exc}")
    if zstandard is not None:
        try:
            return zstandard.ZstdDecompressor().decompress(blob)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"zstd:{exc}")
    try:
        return zlib.decompress(blob)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"zlib:{exc}")
    raise RuntimeError("could not decompress quantized blob: " + "; ".join(errors))


def _load_quantized_state_dict(run_dir: Path) -> dict[str, torch.Tensor] | None:
    quant_files = sorted(run_dir.glob("*.ptz"))
    if not quant_files:
        return None
    blob = quant_files[0].read_bytes()
    payload = torch.load(io.BytesIO(_decompress_quant_blob(blob)), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    weights = payload.get("w")
    if isinstance(weights, dict):
        return {name: tensor for name, tensor in weights.items() if isinstance(tensor, torch.Tensor)}
    return None


def render_budget_table(report: dict[str, object]) -> str:
    class_rows = report.get("tensor_class_breakdown") or {}
    lines = [
        "Metric                         Value",
        "-----------------------------  ----------------",
        f"param_count                     {report.get('param_count', '-')}",
        f"raw_bytes_fp16_est              {report.get('raw_bytes_fp16_est', '-')}",
        f"raw_bytes_fp32_est              {report.get('raw_bytes_fp32_est', '-')}",
        f"post_quant_bytes_est            {report.get('post_quant_bytes_est', '-')}",
        f"exported_bytes_measured         {report.get('exported_bytes_measured', '-')}",
        f"code_bytes_measured             {report.get('code_bytes_measured', '-')}",
        f"artifact_bytes_measured         {report.get('artifact_bytes_measured', '-')}",
        f"remaining_headroom_to_16MB      {report.get('remaining_headroom_to_16MB', '-')}",
    ]
    if class_rows:
        lines.extend(
            [
                "",
                "Tensor Class Breakdown",
                "Class                         Params        Bytes",
                "---------------------------  ------------  ------------",
            ]
        )
        for name in ("embeddings", "attention", "mlp", "bigram_tables", "norms_scales_other"):
            row = class_rows.get(name)
            if not row:
                continue
            lines.append(f"{name:27}  {row['param_count']:12}  {row['measured_bytes']:12}")
    return "\n".join(lines) + "\n"


def analyze_run_budget(run_dir: Path, counted_code_paths: tuple[str, ...]) -> dict[str, object]:
    raw_state = _load_raw_state_dict(run_dir)
    quant_state = _load_quantized_state_dict(run_dir)
    exported_bytes = None
    quant_files = sorted(run_dir.glob("*.ptz"))
    if quant_files:
        exported_bytes = quant_files[0].stat().st_size

    code_bytes = 0
    for rel_path in counted_code_paths:
        code_bytes += len((Path(__file__).resolve().parents[1] / rel_path).read_bytes())

    tensor_class_breakdown = _summarize_state_dict(raw_state) if raw_state else {}
    param_count = sum(item["param_count"] for item in tensor_class_breakdown.values()) if tensor_class_breakdown else None
    post_quant_bytes_est = (
        sum(_tensor_nbytes(tensor) for tensor in quant_state.values())
        if quant_state
        else None
    )
    artifact_bytes = code_bytes + exported_bytes if exported_bytes is not None else None
    report: dict[str, object] = {
        "run_dir": str(run_dir),
        "counted_code_paths": list(counted_code_paths),
        "param_count": param_count,
        "raw_bytes_fp16_est": (param_count * 2) if param_count is not None else None,
        "raw_bytes_fp32_est": (param_count * 4) if param_count is not None else None,
        "post_quant_bytes_est": post_quant_bytes_est,
        "exported_bytes_measured": exported_bytes,
        "code_bytes_measured": code_bytes,
        "artifact_bytes_measured": artifact_bytes,
        "remaining_headroom_to_16MB": (ARTIFACT_LIMIT_BYTES - artifact_bytes) if artifact_bytes is not None else None,
        "tensor_class_breakdown": tensor_class_breakdown,
    }
    return report


def write_budget_reports(run_dir: Path, counted_code_paths: tuple[str, ...]) -> dict[str, object]:
    report = analyze_run_budget(run_dir, counted_code_paths)
    (run_dir / "byte_budget.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "byte_budget.txt").write_text(render_budget_table(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Parameter Golf artifact and code-byte budget for a run directory.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--code-path", action="append", default=[], dest="code_paths")
    args = parser.parse_args()

    report = write_budget_reports(args.run_dir.resolve(), tuple(args.code_paths))
    print(render_budget_table(report), end="")


if __name__ == "__main__":
    main()
