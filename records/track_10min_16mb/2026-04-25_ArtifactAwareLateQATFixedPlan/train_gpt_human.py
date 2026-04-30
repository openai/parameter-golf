#!/usr/bin/env python3
"""Review helper for the compact Artifact-Aware LateQAT submission.

The actual submitted entrypoint is train_gpt.py. It is a compact lzma/base85
wrapper because the artifact budget counts code bytes. This helper stays short:
it documents the runtime layout and can extract the exact embedded files from
train_gpt.py without executing the training pipeline.
"""

from __future__ import annotations

import argparse
import ast
import base64
import lzma
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMPACT_ENTRYPOINT = ROOT / "train_gpt.py"

RESULTS = {
    "val_bpb": 1.1919522011324903,
    "prequant_val_bpb": 1.1802617550781567,
    "quant_gap_bpb": 0.01169044605433367,
    "artifact_bytes": 15415044,
    "compact_train_gpt_bytes": 79879,
    "pipeline_seconds": 588.6408305168152,
}

RUNTIME_FILES = {
    "_artifact_pipeline.py": "orchestrates train, fixed-plan quantization, and final eval",
    "final_bit_plan.json": "fixed tensor bit plan used by both late QAT and compiler",
    "train_gpt_cuda_blessed_depth9_bigram10240_v1.py": "8xH100 trainer with artifact-aware late QAT",
    "train_gpt_base.py": "shared model definitions used by train and eval",
    "ttt_lora.py": "legacy-compatible helper module imported by the shared stack",
    "tools/quantize_blessed_checkpoint.py": "budgeted_v2 artifact compiler and evaluator",
    "tools/quant_budget.py": "mixed-precision budget allocation helpers",
    "tools/quant_calib.py": "calibration token and activation collection helpers",
    "tools/gptq_refine.py": "GPTQ refinement helper",
    "tools/__init__.py": "package marker for local tools",
}


def _literal_assigned_value(module: ast.Module, name: str) -> object:
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise ValueError(f"could not find assignment to {name!r}")


def _decode_payload(entrypoint: Path = COMPACT_ENTRYPOINT) -> str:
    module = ast.parse(entrypoint.read_text(encoding="utf-8"))
    chunks = _literal_assigned_value(module, "_DATA")
    compressed = base64.b85decode("".join(chunks))
    return lzma.decompress(compressed).decode("utf-8")


def load_sources(entrypoint: Path = COMPACT_ENTRYPOINT) -> dict[str, str]:
    """Return the exact runtime sources embedded in train_gpt.py.

    This parses the wrapper statically. It does not exec the payload and does
    not start training.
    """

    payload = _decode_payload(entrypoint)
    module = ast.parse(payload)
    files = _literal_assigned_value(module, "FILES")
    if not isinstance(files, dict):
        raise TypeError("embedded FILES payload is not a dict")
    return {str(name): str(source) for name, source in files.items()}


def print_overview() -> None:
    print("Artifact-Aware LateQAT Fixed Bit Plan")
    print()
    print("Submitted entrypoint:")
    print(f"  {COMPACT_ENTRYPOINT.name}")
    print()
    print("Key result:")
    for key, value in RESULTS.items():
        print(f"  {key}: {value}")
    print()
    print("Embedded runtime files:")
    for name, description in RUNTIME_FILES.items():
        print(f"  {name}: {description}")
    print()
    print("Use --list to inspect embedded sizes or --extract DIR to review sources.")


def list_sources() -> None:
    sources = load_sources()
    for name in sorted(sources):
        source = sources[name]
        print(f"{len(source):8d} bytes  {source.count(chr(10)) + 1:5d} lines  {name}")


def extract_sources(dest: Path) -> None:
    sources = load_sources()
    dest.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    print(f"extracted {len(sources)} files to {dest}")


def print_source(name: str) -> None:
    sources = load_sources()
    try:
        print(sources[name], end="")
    except KeyError as exc:
        available = ", ".join(sorted(sources))
        raise SystemExit(f"unknown source {name!r}; available: {available}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list embedded runtime files")
    parser.add_argument("--extract", type=Path, help="extract embedded runtime files")
    parser.add_argument("--print-source", metavar="PATH", help="print one embedded source")
    args = parser.parse_args()

    if args.list:
        list_sources()
    elif args.extract is not None:
        extract_sources(args.extract)
    elif args.print_source:
        print_source(args.print_source)
    else:
        print_overview()


if __name__ == "__main__":
    main()
