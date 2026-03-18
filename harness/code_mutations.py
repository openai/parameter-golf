from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .common import canonical_code_mutation


def _source_hash(script_path: str | Path) -> str:
    source_text = Path(script_path).read_text(encoding="utf-8")
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def _mutation(
    name: str,
    *,
    family: str,
    params: dict[str, Any],
    objective: str,
    hypothesis: str,
    rationale: str,
    changes_by_script: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "params": params,
        "objective": objective,
        "hypothesis": hypothesis,
        "rationale": rationale,
        "changes_by_script": changes_by_script,
    }


_MUTATIONS: list[dict[str, Any]] = [
    _mutation(
        "quant_clip_tighter",
        family="quantization",
        params={"int8_clip_percentile": 99.99992},
        objective="reduce roundtrip loss",
        hypothesis="A slightly tighter int8 clip percentile may reduce outlier damage and improve post-quant BPB.",
        rationale="Both trainers expose a shared clean int8 quantization path, so percentile clipping is a compact high-leverage code mutation.",
        changes_by_script={
            "train_gpt.py": [
                {
                    "match": "INT8_CLIP_PERCENTILE = 99.99984",
                    "replace": "INT8_CLIP_PERCENTILE = 99.99992",
                    "description": "Tighten the torch int8 clipping percentile.",
                }
            ],
            "train_gpt_mlx.py": [
                {
                    "match": "INT8_CLIP_PERCENTILE = 99.99984",
                    "replace": "INT8_CLIP_PERCENTILE = 99.99992",
                    "description": "Tighten the MLX int8 clipping percentile.",
                }
            ],
        },
    ),
    _mutation(
        "quant_scale_fp32",
        family="quantization",
        params={"per_row_scale_dtype": "float32"},
        objective="reduce dequantization error",
        hypothesis="Keeping row scales in fp32 should preserve more quantization fidelity at modest byte cost.",
        rationale="Per-row scales are part of the roundtrip-critical metadata path and are shared between the torch and MLX exporters.",
        changes_by_script={
            "train_gpt.py": [
                {
                    "match": "INT8_PER_ROW_SCALE_DTYPE = torch.float16",
                    "replace": "INT8_PER_ROW_SCALE_DTYPE = torch.float32",
                    "description": "Store torch per-row int8 scales in fp32 instead of fp16.",
                }
            ],
            "train_gpt_mlx.py": [
                {
                    "match": "INT8_PER_ROW_SCALE_DTYPE = np.float16",
                    "replace": "INT8_PER_ROW_SCALE_DTYPE = np.float32",
                    "description": "Store MLX per-row int8 scales in fp32 instead of fp16.",
                }
            ],
        },
    ),
    _mutation(
        "quant_keep_float_bigger",
        family="quantization",
        params={"int8_keep_float_max_numel": 98_304},
        objective="protect small tensors from over-quantization",
        hypothesis="Raising the small-float passthrough threshold should preserve more sensitive tensors before the int8 export step.",
        rationale="The keep-float cutoff is a concise code-level policy knob that can trade size for roundtrip accuracy.",
        changes_by_script={
            "train_gpt.py": [
                {
                    "match": "INT8_KEEP_FLOAT_MAX_NUMEL = 65_536",
                    "replace": "INT8_KEEP_FLOAT_MAX_NUMEL = 98_304",
                    "description": "Increase the torch small-float passthrough threshold.",
                }
            ],
            "train_gpt_mlx.py": [
                {
                    "match": "INT8_KEEP_FLOAT_MAX_NUMEL = 65_536",
                    "replace": "INT8_KEEP_FLOAT_MAX_NUMEL = 98_304",
                    "description": "Increase the MLX small-float passthrough threshold.",
                }
            ],
        },
    ),
    _mutation(
        "quant_keep_float_smaller",
        family="quantization",
        params={"int8_keep_float_max_numel": 32_768},
        objective="reduce artifact bytes",
        hypothesis="Lowering the small-float passthrough threshold should save bytes, at some risk to roundtrip accuracy.",
        rationale="The keep-float cutoff is also a direct size-control lever when the submission is too close to the 16 MB ceiling.",
        changes_by_script={
            "train_gpt.py": [
                {
                    "match": "INT8_KEEP_FLOAT_MAX_NUMEL = 65_536",
                    "replace": "INT8_KEEP_FLOAT_MAX_NUMEL = 32_768",
                    "description": "Decrease the torch small-float passthrough threshold.",
                }
            ],
            "train_gpt_mlx.py": [
                {
                    "match": "INT8_KEEP_FLOAT_MAX_NUMEL = 65_536",
                    "replace": "INT8_KEEP_FLOAT_MAX_NUMEL = 32_768",
                    "description": "Decrease the MLX small-float passthrough threshold.",
                }
            ],
        },
    ),
]


def _resolve_definition(script_path: str | Path, name: str) -> dict[str, Any]:
    basename = Path(script_path).name
    for mutation in _MUTATIONS:
        if mutation["name"] != name:
            continue
        if basename in mutation["changes_by_script"]:
            return mutation
    known = ", ".join(sorted(m["name"] for m in _MUTATIONS))
    raise KeyError(f"Unknown code mutation {name!r} for {basename}. Known mutations: {known}")


def available_code_mutations(script_path: str | Path) -> list[dict[str, Any]]:
    basename = Path(script_path).name
    mutations: list[dict[str, Any]] = []
    for mutation in _MUTATIONS:
        if basename not in mutation["changes_by_script"]:
            continue
        planned = planned_code_mutation(script_path, mutation["name"])
        planned["objective"] = mutation["objective"]
        planned["hypothesis"] = mutation["hypothesis"]
        planned["rationale"] = mutation["rationale"]
        mutations.append(planned)
    return mutations


def planned_code_mutation(script_path: str | Path, name: str) -> dict[str, Any]:
    mutation = _resolve_definition(script_path, name)
    planned = {
        "name": mutation["name"],
        "family": mutation["family"],
        "params": dict(mutation["params"]),
        "script_basename": Path(script_path).name,
        "source_hash": _source_hash(script_path),
    }
    planned["signature"] = canonical_code_mutation(planned)
    return planned


def _apply_change_once(source_text: str, change: dict[str, str]) -> tuple[str, dict[str, str]]:
    match = change["match"]
    count = source_text.count(match)
    if count != 1:
        raise ValueError(f"Expected exactly one anchor match for {match!r}, found {count}")
    return (
        source_text.replace(match, change["replace"], 1),
        {
            "description": change["description"],
            "match": change["match"],
            "replace": change["replace"],
        },
    )


def preview_code_mutation(script_path: str | Path, code_mutation: dict[str, Any] | None) -> dict[str, Any]:
    source_path = Path(script_path)
    source_text = source_path.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    source_bytes = len(source_text.encode("utf-8"))
    if not code_mutation:
        return {
            "source_hash": source_hash,
            "source_code_bytes": source_bytes,
            "materialized_code_bytes": source_bytes,
            "code_bytes_delta": 0,
            "applied": [],
            "code_mutation": None,
            "mutated_text": source_text,
        }

    planned = planned_code_mutation(script_path, code_mutation["name"])
    mutation = _resolve_definition(script_path, code_mutation["name"])
    mutated_text = source_text
    applied: list[dict[str, str]] = []
    for change in mutation["changes_by_script"][source_path.name]:
        mutated_text, applied_change = _apply_change_once(mutated_text, change)
        applied.append(applied_change)

    runtime_mutation = dict(planned)
    runtime_mutation["source_hash"] = source_hash
    runtime_mutation["applied"] = applied
    materialized_bytes = len(mutated_text.encode("utf-8"))
    return {
        "source_hash": source_hash,
        "source_code_bytes": source_bytes,
        "materialized_code_bytes": materialized_bytes,
        "code_bytes_delta": materialized_bytes - source_bytes,
        "applied": applied,
        "code_mutation": runtime_mutation,
        "mutated_text": mutated_text,
    }


def materialize_script_for_run(
    script_path: str | Path,
    run_dir: Path,
    code_mutation: dict[str, Any] | None,
) -> dict[str, Any]:
    preview = preview_code_mutation(script_path, code_mutation)
    source_path = Path(script_path)
    materialized_path = run_dir / source_path.name
    materialized_path.write_text(preview["mutated_text"], encoding="utf-8")
    preview["materialized_script_path"] = str(materialized_path)
    return preview
