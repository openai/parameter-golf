from __future__ import annotations

import torch

from core.artifact_core import deserialize_quant_artifact, serialize_quant_artifact
from core.quant_core import dequantize_state_dict_int8, quantize_state_dict_int8


def build_state_dict() -> dict[str, torch.Tensor]:
    return {
        "tok_emb.weight": torch.linspace(-0.25, 0.25, 70000, dtype=torch.float32).reshape(280, 250),
        "block.proj.weight": torch.linspace(-0.5, 0.5, 70000, dtype=torch.float32).reshape(280, 250),
        "block.attn_scale": torch.linspace(0.1, 1.0, 32, dtype=torch.float32),
        "counter": torch.arange(8, dtype=torch.int64),
    }


def test_packed_artifact_roundtrip_preserves_quantized_object_and_dequantized_state():
    state_dict = build_state_dict()
    quant_obj, _ = quantize_state_dict_int8(state_dict)
    baseline_restored = dequantize_state_dict_int8(quant_obj)

    artifact_blob, raw_len = serialize_quant_artifact(quant_obj, "packed_zlib")
    restored_quant_obj = deserialize_quant_artifact(artifact_blob, "packed_zlib")
    restored_state_dict = dequantize_state_dict_int8(restored_quant_obj)

    assert raw_len > 0
    assert restored_quant_obj["__quant_format__"] == quant_obj["__quant_format__"]
    assert restored_quant_obj.get("qmeta", {}) == quant_obj.get("qmeta", {})
    assert restored_quant_obj.get("passthrough_orig_dtypes", {}) == quant_obj.get("passthrough_orig_dtypes", {})
    assert restored_quant_obj["dtypes"] == quant_obj["dtypes"]
    assert restored_quant_obj["quantized"].keys() == quant_obj["quantized"].keys()
    assert restored_quant_obj["scales"].keys() == quant_obj["scales"].keys()
    assert restored_quant_obj["passthrough"].keys() == quant_obj["passthrough"].keys()

    for name, tensor in quant_obj["quantized"].items():
        restored = restored_quant_obj["quantized"][name]
        assert restored.dtype == tensor.dtype
        assert restored.shape == tensor.shape
        assert torch.equal(restored, tensor)
    for name, tensor in quant_obj["scales"].items():
        restored = restored_quant_obj["scales"][name]
        assert restored.dtype == tensor.dtype
        assert restored.shape == tensor.shape
        assert torch.equal(restored, tensor)
    for name, tensor in quant_obj["passthrough"].items():
        restored = restored_quant_obj["passthrough"][name]
        assert restored.dtype == tensor.dtype
        assert restored.shape == tensor.shape
        assert torch.equal(restored, tensor)

    for name, original in state_dict.items():
        restored = restored_state_dict[name]
        assert restored.dtype == original.dtype
        assert restored.shape == original.shape
        if original.is_floating_point():
            assert torch.equal(restored, baseline_restored[name])
        else:
            assert torch.equal(restored, original)


def test_packed_artifact_bytes_are_deterministic():
    quant_obj, _ = quantize_state_dict_int8(build_state_dict())
    blob_a, raw_a = serialize_quant_artifact(quant_obj, "packed_zlib")
    blob_b, raw_b = serialize_quant_artifact(quant_obj, "packed_zlib")

    assert raw_a == raw_b
    assert blob_a == blob_b


def test_packed_artifact_log_u8_scales_preserve_structure_and_positive_scales():
    state_dict = build_state_dict()
    quant_obj, _ = quantize_state_dict_int8(state_dict)

    blob, _ = serialize_quant_artifact(quant_obj, "packed_zlib", scale_codec="log_u8")
    restored_quant_obj = deserialize_quant_artifact(blob, "packed_zlib")
    restored_state_dict = dequantize_state_dict_int8(restored_quant_obj)

    assert restored_quant_obj["quantized"].keys() == quant_obj["quantized"].keys()
    assert restored_quant_obj["passthrough"].keys() == quant_obj["passthrough"].keys()
    assert restored_quant_obj["scales"].keys() == quant_obj["scales"].keys()

    for name, tensor in quant_obj["quantized"].items():
        assert torch.equal(restored_quant_obj["quantized"][name], tensor)
    for name, tensor in quant_obj["passthrough"].items():
        assert torch.equal(restored_quant_obj["passthrough"][name], tensor)
    for name, original in quant_obj["scales"].items():
        restored = restored_quant_obj["scales"][name]
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        assert torch.all(restored > 0)
        rel_error = ((restored.float() - original.float()).abs() / original.float().clamp_min(1e-12)).max().item()
        assert rel_error < 0.10

    for name, original in state_dict.items():
        restored = restored_state_dict[name]
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        assert torch.isfinite(restored).all()
