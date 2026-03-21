from __future__ import annotations

import json
import struct
import zlib
from typing import Literal, TypedDict

import torch
from torch import Tensor

from core.quant_core import QuantizedStateDict, QuantMetaEntry

PACKED_ARTIFACT_MAGIC = b"PGQ1"
PACKED_ARTIFACT_VERSION = 1
DEFAULT_QUANT_ARTIFACT_FORMAT = "packed_zlib"
SUPPORTED_QUANT_ARTIFACT_FORMATS = ("torchsave_zlib", "packed_zlib", "torchsave_zstd", "packed_zstd")
DEFAULT_PACKED_SCALE_CODEC = "raw"
SUPPORTED_PACKED_SCALE_CODECS = ("raw", "log_u8")


class PackedTensorEntry(TypedDict, total=False):
    name: str
    section: Literal["quantized", "scales", "passthrough"]
    dtype: str
    shape: list[int]
    offset: int
    nbytes: int
    logical_dtype: str
    codec: str
    orig_dtype: str
    log_min: float
    log_max: float


class PackedArtifactMeta(TypedDict, total=False):
    artifact_format: str
    version: int
    quant_format: str
    scale_codec: str
    entries: list[PackedTensorEntry]
    qmeta: dict[str, QuantMetaEntry]
    passthrough_orig_dtypes: dict[str, str]


class ArtifactSectionStats(TypedDict):
    num_tensors: int
    payload_bytes: int
    standalone_zlib_bytes: int


class PackedArtifactBuildStats(TypedDict):
    raw_bytes: int
    meta_bytes: int
    payload_bytes: int
    scale_codec: str
    section_stats: dict[str, ArtifactSectionStats]


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


def encode_scale_tensor(tensor: Tensor, scale_codec: str) -> tuple[Tensor, PackedTensorEntry]:
    if scale_codec == "raw":
        return tensor.detach().to("cpu").contiguous(), {"codec": "raw"}
    if scale_codec == "log_u8":
        t32 = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
        tiny = torch.finfo(torch.float32).tiny
        log_scales = torch.log2(t32.clamp_min(tiny))
        if log_scales.numel() == 0:
            return torch.empty_like(t32, dtype=torch.uint8), {
                "codec": "log_u8",
                "orig_dtype": dtype_name(tensor.dtype),
                "log_min": 0.0,
                "log_max": 0.0,
            }
        log_min = float(log_scales.min().item())
        log_max = float(log_scales.max().item())
        if log_max <= log_min:
            q = torch.zeros_like(t32, dtype=torch.uint8)
        else:
            q = torch.round((log_scales - log_min) * (255.0 / (log_max - log_min))).clamp(0, 255).to(torch.uint8)
        return q.contiguous(), {
            "codec": "log_u8",
            "orig_dtype": dtype_name(tensor.dtype),
            "log_min": log_min,
            "log_max": log_max,
        }
    raise ValueError(
        f"Unsupported PACKED_SCALE_CODEC={scale_codec!r}; "
        f"expected one of {SUPPORTED_PACKED_SCALE_CODECS}"
    )


def decode_scale_tensor(entry: PackedTensorEntry, buffer: memoryview) -> Tensor:
    codec = entry.get("codec", "raw")
    stored = tensor_from_buffer(buffer, entry["dtype"], entry["shape"])
    if codec == "raw":
        return stored
    if codec == "log_u8":
        log_min = float(entry["log_min"])
        log_max = float(entry["log_max"])
        q = stored.to(dtype=torch.float32)
        if log_max <= log_min:
            log_scales = torch.full_like(q, log_min, dtype=torch.float32)
        else:
            log_scales = log_min + q * ((log_max - log_min) / 255.0)
        return torch.exp2(log_scales).to(dtype=DTYPE_BY_NAME[entry["orig_dtype"]]).contiguous()
    raise ValueError(f"Unsupported packed scale codec: {codec}")


def build_packed_quantized_state_dict(
    quant_obj: QuantizedStateDict,
    scale_codec: str = DEFAULT_PACKED_SCALE_CODEC,
) -> tuple[bytes, PackedArtifactBuildStats]:
    if scale_codec not in SUPPORTED_PACKED_SCALE_CODECS:
        raise ValueError(
            f"Unsupported PACKED_SCALE_CODEC={scale_codec!r}; "
            f"expected one of {SUPPORTED_PACKED_SCALE_CODECS}"
        )
    entries: list[PackedTensorEntry] = []
    payload_chunks: list[bytes] = []
    section_chunks: dict[str, list[bytes]] = {"quantized": [], "scales": [], "passthrough": []}
    section_counts: dict[str, int] = {"quantized": 0, "scales": 0, "passthrough": 0}
    offset = 0

    def add_entry(section: Literal["quantized", "scales", "passthrough"], name: str, tensor: Tensor) -> None:
        nonlocal offset
        stored = tensor.detach().to("cpu").contiguous()
        extra_entry: PackedTensorEntry = {}
        if section == "scales":
            stored, extra_entry = encode_scale_tensor(stored, scale_codec)
        chunk = tensor_to_bytes(stored)
        entry: PackedTensorEntry = {
            "name": name,
            "section": section,
            "dtype": dtype_name(stored.dtype),
            "shape": list(stored.shape),
            "offset": offset,
            "nbytes": len(chunk),
        }
        if section == "quantized":
            entry["logical_dtype"] = quant_obj["dtypes"][name]
        entry.update(extra_entry)
        entries.append(entry)
        payload_chunks.append(chunk)
        section_chunks[section].append(chunk)
        section_counts[section] += 1
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
        "scale_codec": scale_codec,
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
    raw_blob = header + meta_bytes + b"".join(payload_chunks)
    section_stats: dict[str, ArtifactSectionStats] = {}
    for section, chunks in section_chunks.items():
        payload = b"".join(chunks)
        section_stats[section] = {
            "num_tensors": section_counts[section],
            "payload_bytes": len(payload),
            "standalone_zlib_bytes": len(zlib.compress(payload, level=9)),
        }
    stats: PackedArtifactBuildStats = {
        "raw_bytes": len(raw_blob),
        "meta_bytes": len(header) + len(meta_bytes),
        "payload_bytes": offset,
        "scale_codec": scale_codec,
        "section_stats": section_stats,
    }
    return raw_blob, stats


def pack_quantized_state_dict(
    quant_obj: QuantizedStateDict,
    scale_codec: str = DEFAULT_PACKED_SCALE_CODEC,
) -> bytes:
    raw_blob, _ = build_packed_quantized_state_dict(quant_obj, scale_codec=scale_codec)
    return raw_blob


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
        section = entry["section"]
        name = entry["name"]
        if section == "quantized":
            tensor = tensor_from_buffer(view, entry["dtype"], entry["shape"])
            quantized[name] = tensor
            dtypes[name] = entry["logical_dtype"]
        elif section == "scales":
            scales[name] = decode_scale_tensor(entry, view)
        elif section == "passthrough":
            tensor = tensor_from_buffer(view, entry["dtype"], entry["shape"])
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


def compress_blob(raw: bytes, artifact_format: str, compression_level: int) -> bytes:
    if artifact_format.endswith("_zlib"):
        return zlib.compress(raw, level=compression_level)
    if artifact_format.endswith("_zstd"):
        import zstandard as zstd

        return zstd.ZstdCompressor(level=compression_level).compress(raw)
    raise ValueError(
        f"Unsupported QUANT_ARTIFACT_FORMAT={artifact_format!r}; "
        f"expected one of {SUPPORTED_QUANT_ARTIFACT_FORMATS}"
    )


def decompress_blob(blob: bytes, artifact_format: str) -> bytes:
    if artifact_format.endswith("_zlib"):
        return zlib.decompress(blob)
    if artifact_format.endswith("_zstd"):
        import zstandard as zstd

        return zstd.ZstdDecompressor().decompress(blob)
    raise ValueError(
        f"Unsupported QUANT_ARTIFACT_FORMAT={artifact_format!r}; "
        f"expected one of {SUPPORTED_QUANT_ARTIFACT_FORMATS}"
    )


def serialize_quant_artifact(
    quant_obj: QuantizedStateDict,
    artifact_format: str,
    compression_level: int = 9,
    scale_codec: str = DEFAULT_PACKED_SCALE_CODEC,
) -> tuple[bytes, int]:
    if artifact_format in {"torchsave_zlib", "torchsave_zstd"}:
        import io

        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        raw = buf.getvalue()
        return compress_blob(raw, artifact_format, compression_level), len(raw)
    if artifact_format in {"packed_zlib", "packed_zstd"}:
        raw = pack_quantized_state_dict(quant_obj, scale_codec=scale_codec)
        return compress_blob(raw, artifact_format, compression_level), len(raw)
    raise ValueError(
        f"Unsupported QUANT_ARTIFACT_FORMAT={artifact_format!r}; "
        f"expected one of {SUPPORTED_QUANT_ARTIFACT_FORMATS}"
    )


def deserialize_quant_artifact(blob: bytes, artifact_format: str) -> QuantizedStateDict:
    raw = decompress_blob(blob, artifact_format)
    if artifact_format in {"torchsave_zlib", "torchsave_zstd"}:
        import io

        return torch.load(io.BytesIO(raw), map_location="cpu")
    if artifact_format in {"packed_zlib", "packed_zstd"}:
        return unpack_quantized_state_dict(raw)
    raise ValueError(
        f"Unsupported QUANT_ARTIFACT_FORMAT={artifact_format!r}; "
        f"expected one of {SUPPORTED_QUANT_ARTIFACT_FORMATS}"
    )
