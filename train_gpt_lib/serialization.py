from __future__ import annotations

import io
import math
import os
import zlib
from typing import Callable

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .config import (
    INT8_CLIP_Q,
    INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
    INT8_KEEP_FLOAT_MAX_NUMEL,
    INT8_KEEP_FLOAT_STORE_DTYPE,
    INT8_PER_ROW_SCALE_DTYPE,
)
from .eval import eval_val
from .config import Hyperparameters


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def estimate_int8_payload_bytes_gpt(
    vocab_size: int,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: int,
    tie_embeddings: bool,
    *,
    use_mhc: bool = False,
) -> int | None:
    """Exact ``int8_payload_bytes`` for :class:`~train_gpt_lib.model.GPT` + ``quantize_state_dict_int8``.

    Mirrors tensor layout (``nn.Linear`` is ``(out_features, in_features)``) and the same rules as
    :func:`quantize_state_dict_int8` (small tensors / control patterns / per-row int8 + fp16 scales).

    Returns ``None`` if ``use_mhc`` is True (extra MHC parameters; build the model to measure).
    """
    if use_mhc:
        return None

    scale_row_bytes = int(torch.empty((), dtype=INT8_PER_ROW_SCALE_DTYPE).element_size())

    def contrib(name: str, shape: tuple[int, ...]) -> int:
        numel = 1
        for x in shape:
            numel *= x
        if numel <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
                return numel * 4
            return numel * 2
        if len(shape) == 2:
            rows, _cols = shape
            return numel + rows * scale_row_bytes
        return numel + 4

    d = model_dim
    head_dim = d // num_heads
    kv_dim = num_kv_heads * head_dim
    ne = num_layers // 2
    nd = num_layers - ne
    skip_n = min(ne, nd)
    total = 0
    total += contrib("tok_emb.weight", (vocab_size, d))
    total += contrib("skip_weights", (skip_n, d))
    for i in range(num_layers):
        p = f"blocks.{i}"
        total += contrib(f"{p}.attn.c_q.weight", (d, d))
        total += contrib(f"{p}.attn.c_k.weight", (kv_dim, d))
        total += contrib(f"{p}.attn.c_v.weight", (kv_dim, d))
        total += contrib(f"{p}.attn.proj.weight", (d, d))
        total += contrib(f"{p}.attn.q_gain", (num_heads,))
        total += contrib(f"{p}.mlp.fc.weight", (mlp_mult * d, d))
        total += contrib(f"{p}.mlp.proj.weight", (d, mlp_mult * d))
        total += contrib(f"{p}.attn_scale", (d,))
        total += contrib(f"{p}.mlp_scale", (d,))
        total += contrib(f"{p}.resid_mix", (2, d))
    if not tie_embeddings:
        total += contrib("lm_head.weight", (vocab_size, d))
    return int(total)


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def save_and_validate_roundtrip(
    *,
    args: Hyperparameters,
    base_model: nn.Module,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    code: str,
    log0: Callable[[str, bool], None],
    master_process: bool,
    on_model_size: Callable[..., None] | None = None,
) -> None:
    model_bytes = 0
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        model_mb_ceil = int(math.ceil(model_bytes / (1024.0 * 1024.0)))
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model (fp32 state_dict .pt): {model_bytes} bytes")
        log0(f"Serialized model size ceil (.pt): {model_mb_ceil} MiB")
        log0(f"MODEL_FP32_PT_BYTES:{model_bytes}")
        log0(f"MODEL_FP32_PT_MB_CEIL:{model_mb_ceil}")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size (fp32 .pt + code): {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        zlib_mb_ceil = int(math.ceil(quant_file_bytes / (1024.0 * 1024.0)))
        log0(f"MODEL_INT8_ZLIB_BYTES:{quant_file_bytes}")
        log0(f"MODEL_INT8_ZLIB_MB_CEIL:{zlib_mb_ceil}")
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")
        if on_model_size is not None:
            on_model_size(
                fp32_pt_bytes=model_bytes,
                int8_zlib_bytes=quant_file_bytes,
                int8_payload_bytes=int(quant_stats["int8_payload_bytes"]),
            )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = torch.cuda.Event(enable_timing=True)
    t_qeval_end = torch.cuda.Event(enable_timing=True)
    t_qeval.record()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    t_qeval_end.record()
    torch.cuda.synchronize()
    eval_ms = t_qeval.elapsed_time(t_qeval_end)
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{eval_ms:.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
