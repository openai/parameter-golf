from __future__ import annotations

import json
import struct
import zlib
from typing import Any

import numpy as np


MAGIC = b"PGEC1\0"


def _maybe_zlib(payload: bytes) -> tuple[str, bytes]:
    compressed = zlib.compress(payload, 9)
    if len(compressed) < len(payload):
        return "zlib", compressed
    return "raw", payload


def _pack_int6(values: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(values.reshape(-1), dtype=np.uint8)
    out = bytearray((flat.size * 6 + 7) // 8)
    bit_pos = 0
    for value in flat.tolist():
        byte_index = bit_pos >> 3
        shift = bit_pos & 7
        out[byte_index] |= ((value & 0x3F) << shift) & 0xFF
        if shift > 2:
            out[byte_index + 1] |= (value & 0x3F) >> (8 - shift)
        bit_pos += 6
    return bytes(out)


def _unpack_int6(payload: bytes, count: int) -> np.ndarray:
    src = memoryview(payload)
    out = np.empty((count,), dtype=np.uint8)
    bit_pos = 0
    for i in range(count):
        byte_index = bit_pos >> 3
        shift = bit_pos & 7
        value = (src[byte_index] >> shift) & 0x3F
        if shift > 2:
            value |= (src[byte_index + 1] << (8 - shift)) & 0x3F
        out[i] = value
        bit_pos += 6
    return out


def encode_quantized_state(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> bytes:
    entries: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    cursor = 0

    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        entry: dict[str, Any] = {
            "name": name,
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
        }
        if arr.dtype == np.int8 and name.endswith(".q"):
            arr16 = arr.astype(np.int16, copy=False)
            if arr16.size and int(arr16.min()) >= -32 and int(arr16.max()) <= 31:
                payload = _pack_int6((arr16 + 32).astype(np.uint8, copy=False))
                codec, packed = _maybe_zlib(payload)
                entry.update({"kind": "q", "bits": 6, "count": int(arr.size), "codec": codec})
                payload = packed
            else:
                raw = arr.view(np.uint8).tobytes(order="C")
                codec, packed = _maybe_zlib(raw)
                entry.update({"kind": "q", "bits": 8, "count": int(arr.size), "codec": codec})
                payload = packed
        else:
            raw = arr.tobytes(order="C")
            codec, packed = _maybe_zlib(raw)
            entry.update({"kind": "raw", "codec": codec, "nbytes": len(raw)})
            payload = packed
        entry["offset"] = cursor
        entry["length"] = len(payload)
        payload_parts.append(payload)
        entries.append(entry)
        cursor += len(payload)

    header = {"format": "pg_custom_entropy_v1", "meta": meta, "entries": entries}
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(payload_parts)


def decode_quantized_state(blob: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if blob[: len(MAGIC)] != MAGIC:
        raise ValueError("Unexpected codec magic")
    header_len = struct.unpack("<I", blob[len(MAGIC) : len(MAGIC) + 4])[0]
    header_start = len(MAGIC) + 4
    header_end = header_start + header_len
    header = json.loads(blob[header_start:header_end].decode("utf-8"))
    payload = memoryview(blob)[header_end:]

    arrays: dict[str, np.ndarray] = {}
    for entry in header["entries"]:
        start = int(entry["offset"])
        end = start + int(entry["length"])
        chunk = bytes(payload[start:end])
        if entry["codec"] == "zlib":
            chunk = zlib.decompress(chunk)
        dtype = np.dtype(entry["dtype"])
        if entry["kind"] == "q" and int(entry["bits"]) == 6:
            unpacked = _unpack_int6(chunk, int(entry["count"])).astype(np.int16, copy=False) - 32
            arr = unpacked.astype(np.int8, copy=False).reshape(entry["shape"])
        else:
            arr = np.frombuffer(chunk, dtype=dtype).copy().reshape(entry["shape"])
        arrays[entry["name"]] = arr
    return arrays, header["meta"]


def _self_test() -> None:
    rng = np.random.default_rng(0)
    arrays = {
        "blocks.0.attn.c_q.weight.q": rng.integers(-32, 32, size=(17, 19), dtype=np.int16).astype(np.int8),
        "blocks.0.attn.c_q.weight.scale": rng.normal(size=(17,)).astype(np.float16),
        "tok_emb.weight.q": rng.integers(-127, 128, size=(13, 11), dtype=np.int16).astype(np.int8),
        "skip_weights": rng.normal(size=(7, 5)).astype(np.float32),
    }
    meta = {"blocks.0.attn.c_q.weight": {"type": "int6"}, "tok_emb.weight": {"type": "int8"}}
    blob = encode_quantized_state(arrays, meta)
    restored, restored_meta = decode_quantized_state(blob)
    assert restored_meta == meta
    for name, arr in arrays.items():
        np.testing.assert_array_equal(restored[name], arr)
    print(f"codec_self_test_ok bytes:{len(blob)}")


if __name__ == "__main__":
    _self_test()
