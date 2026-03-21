from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from core.artifact_core import build_packed_quantized_state_dict, serialize_quant_artifact
from core.quant_core import quantize_state_dict_int8


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dict = torch.load(args.state_dict_path, map_location="cpu")
    quant_obj, quant_stats = quantize_state_dict_int8(state_dict)

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
        "baseline_tensor_bytes": quant_stats["baseline_tensor_bytes"],
        "int8_payload_bytes": quant_stats["int8_payload_bytes"],
        "reports": reports,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
