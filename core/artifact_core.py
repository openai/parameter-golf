from __future__ import annotations

import json
import struct
import zlib
from typing import Literal, TypedDict

import torch
from torch import Tensor

from core.quant_core import QuantMetaEntry, QuantizedStateDict

PACKED_ARTIFACT_MAGIC = b"PGQ1"
PACKED_ARTIFACT_VERSION = 1
DEFAULT_QUANT_ARTIFACT_FORMAT = "torchsave_zlib"
SUPPORTED_QUANT_ARTIFACT_FORMATS = ("torchsave_zlib", "packed_zlib")


class PackedTensorEntry(TypedDict, total=False):
    name: str
    section: Literal["quantized", "scales", "passthrough"]
    dtype: str
    shape: list[int]
    offset: int
    nbytes: int
    logical_dtype: str


class PackedArtifactMeta(TypedDict, total=False):
    artifact_format: str
    version: int
    quant_format: str
    entries: list[PackedTensorEntry]
    qmeta: dict[str, QuantMetaEntry]
    passthrough_orig_dtypes: dict[str, str]


DTYPE_BY_NAME = {
    "bool": torch.bool,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in DTYPE_BY_NAME:
        raise ValueError(f"Unsupported dtype for packed artifact: {dtype}")
    return name


def tensor_to_bytes(tensor: Tensor) -> bytes:
    cpu = tensor.detach().to("cpu").contiguous()
    return cpu.view(torch.uint8).numpy().tobytes()


def tensor_from_buffer(buffer: memoryview, dtype_name_str: str, shape: list[int]) -> Tensor:
    try:
        dtype = DTYPE_BY_NAME[dtype_name_str]
    except KeyError as exc:
        raise ValueError(f"Unsupported packed dtype: {dtype_name_str}") from exc
    tensor = torch.frombuffer(bytearray(buffer), dtype=dtype)
    if shape:
        tensor = tensor.reshape(shape)
    return tensor.clone().contiguous()


def pack_quantized_state_dict(quant_obj: QuantizedStateDict) -> bytes:
    entries: list[PackedTensorEntry] = []
    payload_chunks: list[bytes] = []
    offset = 0

    def add_entry(section: Literal["quantized", "scales", "passthrough"], name: str, tensor: Tensor) -> None:
        nonlocal offset
        chunk = tensor_to_bytes(tensor)
        entry: PackedTensorEntry = {
            "name": name,
            "section": section,
            "dtype": dtype_name(tensor.dtype),
            "shape": list(tensor.shape),
            "offset": offset,
            "nbytes": len(chunk),
        }
        if section == "quantized":
            entry["logical_dtype"] = quant_obj["dtypes"][name]
        entries.append(entry)
        payload_chunks.append(chunk)
        offset += len(chunk)

    quantized_tensors = quant_obj["quantized"]
    scales_tensors = quant_obj["scales"]
    passthrough_tensors = quant_obj["passthrough"]
    for name in sorted(quantized_tensors.keys()):
        add_entry("quantized", name, quantized_tensors[name])
    for name in sorted(scales_tensors.keys()):
        add_entry("scales", name, scales_tensors[name])
    for name in sorted(passthrough_tensors.keys()):
        add_entry("passthrough", name, passthrough_tensors[name])

    meta: PackedArtifactMeta = {
        "artifact_format": "packed_quantized_state_dict",
        "version": PACKED_ARTIFACT_VERSION,
        "quant_format": quant_obj["__quant_format__"],
        "entries": entries,
    }
    qmeta = quant_obj.get("qmeta")
    if qmeta:
        meta["qmeta"] = qmeta
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes")
    if passthrough_orig_dtypes:
        meta["passthrough_orig_dtypes"] = passthrough_orig_dtypes

    meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header = PACKED_ARTIFACT_MAGIC + struct.pack("<II", PACKED_ARTIFACT_VERSION, len(meta_bytes))
    return header + meta_bytes + b"".join(payload_chunks)


def unpack_quantized_state_dict(blob: bytes) -> QuantizedStateDict:
    if len(blob) < len(PACKED_ARTIFACT_MAGIC) + 8:
        raise ValueError("Packed artifact blob is too short")
    if blob[: len(PACKED_ARTIFACT_MAGIC)] != PACKED_ARTIFACT_MAGIC:
        raise ValueError("Packed artifact magic mismatch")

    version, meta_len = struct.unpack("<II", blob[len(PACKED_ARTIFACT_MAGIC) : len(PACKED_ARTIFACT_MAGIC) + 8])
    if version != PACKED_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported packed artifact version: {version}")

    meta_start = len(PACKED_ARTIFACT_MAGIC) + 8
    meta_end = meta_start + meta_len
    meta = json.loads(blob[meta_start:meta_end].decode("utf-8"))
    payload = memoryview(blob)[meta_end:]

    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}

    for entry in meta["entries"]:
        offset = int(entry["offset"])
        nbytes = int(entry["nbytes"])
        view = payload[offset : offset + nbytes]
        tensor = tensor_from_buffer(view, entry["dtype"], entry["shape"])
        section = entry["section"]
        name = entry["name"]
        if section == "quantized":
            quantized[name] = tensor
            dtypes[name] = entry["logical_dtype"]
        elif section == "scales":
            scales[name] = tensor
        elif section == "passthrough":
            passthrough[name] = tensor
        else:
            raise ValueError(f"Unknown packed artifact section: {section}")

    obj: QuantizedStateDict = {
        "__quant_format__": meta["quant_format"],
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if "qmeta" in meta:
        obj["qmeta"] = meta["qmeta"]
    if "passthrough_orig_dtypes" in meta:
        obj["passthrough_orig_dtypes"] = meta["passthrough_orig_dtypes"]
    return obj


def serialize_quant_artifact(quant_obj: QuantizedStateDict, artifact_format: str, compression_level: int = 9) -> tuple[bytes, int]:
    if artifact_format == "torchsave_zlib":
        import io

        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        raw = buf.getvalue()
        return zlib.compress(raw, level=compression_level), len(raw)
    if artifact_format == "packed_zlib":
        raw = pack_quantized_state_dict(quant_obj)
        return zlib.compress(raw, level=compression_level), len(raw)
    raise ValueError(
        f"Unsupported QUANT_ARTIFACT_FORMAT={artifact_format!r}; "
        f"expected one of {SUPPORTED_QUANT_ARTIFACT_FORMATS}"
    )


def deserialize_quant_artifact(blob: bytes, artifact_format: str) -> QuantizedStateDict:
    raw = zlib.decompress(blob)
    if artifact_format == "torchsave_zlib":
        import io

        return torch.load(io.BytesIO(raw), map_location="cpu")
    if artifact_format == "packed_zlib":
        return unpack_quantized_state_dict(raw)
    raise ValueError(
        f"Unsupported QUANT_ARTIFACT_FORMAT={artifact_format!r}; "
        f"expected one of {SUPPORTED_QUANT_ARTIFACT_FORMATS}"
    )
