from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.artifact_core import build_packed_quantized_state_dict, serialize_quant_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit quantized artifact bytes by format and section.")
    parser.add_argument(
        "--state-dict-path",
        type=Path,
        default=Path("final_model.pt"),
        help="Path to a dense PyTorch state_dict checkpoint.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        help="zlib compression level used for the final artifact.",
    )
    parser.add_argument(
        "--keep-large-patterns",
        type=str,
        default="",
        help="Comma-separated substrings for large tensors to keep in float passthrough form.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.keep_large_patterns:
        os.environ["INT8_KEEP_LARGE_FLOAT_NAME_PATTERNS"] = args.keep_large_patterns
    else:
        os.environ.pop("INT8_KEEP_LARGE_FLOAT_NAME_PATTERNS", None)

    import importlib
    import core.quant_core as quant_core

    quant_core = importlib.reload(quant_core)

    state_dict = torch.load(args.state_dict_path, map_location="cpu")
    quant_obj, quant_stats = quant_core.quantize_state_dict_int8(state_dict)

    reports: list[dict[str, object]] = []

    torch_blob, torch_raw_len = serialize_quant_artifact(
        quant_obj,
        "torchsave_zlib",
        compression_level=args.compression_level,
    )
    reports.append(
        {
            "format": "torchsave_zlib",
            "scale_codec": "raw",
            "raw_serialized_bytes": torch_raw_len,
            "compressed_bytes": len(torch_blob),
            "payload_bytes": quant_stats["int8_payload_bytes"],
        }
    )

    for scale_codec in ("raw", "log_u8"):
        raw_blob, packed_stats = build_packed_quantized_state_dict(quant_obj, scale_codec=scale_codec)
        packed_blob, packed_raw_len = serialize_quant_artifact(
            quant_obj,
            "packed_zlib",
            compression_level=args.compression_level,
            scale_codec=scale_codec,
        )
        reports.append(
            {
                "format": "packed_zlib",
                "scale_codec": scale_codec,
                "raw_serialized_bytes": packed_raw_len,
                "compressed_bytes": len(packed_blob),
                "payload_bytes": quant_stats["int8_payload_bytes"],
                "meta_bytes": packed_stats["meta_bytes"],
                "packed_payload_bytes": packed_stats["payload_bytes"],
                "section_stats": packed_stats["section_stats"],
                "raw_blob_matches": len(raw_blob) == packed_raw_len,
            }
        )

    summary = {
        "checkpoint": str(args.state_dict_path),
        "keep_large_patterns": args.keep_large_patterns,
        "baseline_tensor_bytes": quant_stats["baseline_tensor_bytes"],
        "int8_payload_bytes": quant_stats["int8_payload_bytes"],
        "large_float_passthrough_bytes": quant_stats["large_float_passthrough_bytes"],
        "num_large_float_passthrough_tensors": quant_stats["num_large_float_passthrough_tensors"],
        "top_quantized_tensors": [
            {
                "name": name,
                "nbytes": quant_core.tensor_nbytes(tensor),
                "shape": list(tensor.shape),
                "scale_nbytes": quant_core.tensor_nbytes(quant_obj["scales"][name]),
            }
            for name, tensor in sorted(
                quant_obj["quantized"].items(),
                key=lambda item: quant_core.tensor_nbytes(item[1]),
                reverse=True,
            )[:20]
        ],
        "top_passthrough_tensors": [
            {
                "name": name,
                "nbytes": quant_core.tensor_nbytes(tensor),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in sorted(
                quant_obj["passthrough"].items(),
                key=lambda item: quant_core.tensor_nbytes(item[1]),
                reverse=True,
            )[:20]
        ],
        "reports": reports,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
