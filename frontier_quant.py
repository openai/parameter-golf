from __future__ import annotations

import io
import json
import lzma
import os
import zlib
from dataclasses import dataclass
from typing import Callable, Mapping

import torch
from torch import Tensor

try:  # pragma: no cover - depends on optional dependency
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional dependency
    zstd = None


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale",
    "attn_scales",
    "mlp_scale",
    "mlp_scales",
    "resid_mix",
    "resid_mixes",
    "q_gain",
    "skip_weight",
    "skip_weights",
    "skip_gates",
)
SUPPORTED_CALIBRATION_STRATEGIES = ("uniform", "recent", "mixed", "loop_heavy")
SUPPORTED_GPTQ_MODES = ("full", "lite", "off")
SUPPORTED_COMPRESSORS = ("zlib", "zstd", "lzma")
DEFAULT_PASSTHROUGH_MAX_NUMEL = 65_536


@dataclass(frozen=True)
class QuantPolicy:
    name: str
    matrix_bits: int
    embed_bits: int
    other_bits: int
    matrix_clip_sigmas: float
    embed_clip_sigmas: float
    other_clip_sigmas: float


@dataclass(frozen=True)
class QuantConfig:
    policy_name: str
    calibration_strategy: str
    calibration_batches: int
    gptq_mode: str
    gptq_block_size: int
    gptq_damp_factor: float
    compressor: str
    bits_by_class: dict[str, int]
    clip_sigmas_by_class: dict[str, float]
    passthrough_max_numel: int = DEFAULT_PASSTHROUGH_MAX_NUMEL


QUANT_POLICIES: dict[str, QuantPolicy] = {
    "sdclip": QuantPolicy(
        name="sdclip",
        matrix_bits=6,
        embed_bits=8,
        other_bits=8,
        matrix_clip_sigmas=12.85,
        embed_clip_sigmas=20.0,
        other_clip_sigmas=16.0,
    ),
    "grouped_sdclip": QuantPolicy(
        name="grouped_sdclip",
        matrix_bits=6,
        embed_bits=8,
        other_bits=8,
        matrix_clip_sigmas=12.9,
        embed_clip_sigmas=20.0,
        other_clip_sigmas=16.0,
    ),
    "embed_split": QuantPolicy(
        name="embed_split",
        matrix_bits=6,
        embed_bits=8,
        other_bits=8,
        matrix_clip_sigmas=12.85,
        embed_clip_sigmas=22.0,
        other_clip_sigmas=16.0,
    ),
}


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def classify_tensor(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
        return "control"
    if ".attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    return "other"


def _default_bits_by_class(policy: QuantPolicy) -> dict[str, int]:
    return {
        "attn": policy.matrix_bits,
        "mlp": policy.matrix_bits,
        "embed": policy.embed_bits,
        "other": policy.other_bits,
        "control": policy.other_bits,
    }


def _default_clip_sigmas_by_class(policy: QuantPolicy) -> dict[str, float]:
    return {
        "attn": policy.matrix_clip_sigmas,
        "mlp": policy.matrix_clip_sigmas,
        "embed": policy.embed_clip_sigmas,
        "other": policy.other_clip_sigmas,
        "control": policy.other_clip_sigmas,
    }


def quant_config_from_env(environ: Mapping[str, str] | None = None) -> QuantConfig:
    env = dict(os.environ if environ is None else environ)
    policy_name = env.get("QUANT_POLICY", "sdclip")
    if policy_name not in QUANT_POLICIES:
        raise KeyError(f"Unknown QUANT_POLICY {policy_name!r}; expected one of {sorted(QUANT_POLICIES)}")
    policy = QUANT_POLICIES[policy_name]
    bits_by_class = _default_bits_by_class(policy)
    clip_sigmas_by_class = _default_clip_sigmas_by_class(policy)

    matrix_bits = int(env.get("MATRIX_BITS", policy.matrix_bits))
    other_bits = int(env.get("OTHER_BITS", policy.other_bits))
    bits_by_class["attn"] = int(env.get("ATTN_BITS", matrix_bits))
    bits_by_class["mlp"] = int(env.get("MLP_BITS", matrix_bits))
    bits_by_class["embed"] = int(env.get("EMBED_BITS", policy.embed_bits))
    bits_by_class["other"] = int(env.get("OTHER_BITS", other_bits))
    bits_by_class["control"] = int(env.get("CONTROL_BITS", bits_by_class["other"]))

    matrix_clip = float(env.get("MATRIX_CLIP_SIGMAS", policy.matrix_clip_sigmas))
    other_clip = float(env.get("OTHER_CLIP_SIGMAS", policy.other_clip_sigmas))
    clip_sigmas_by_class["attn"] = float(env.get("ATTN_CLIP_SIGMAS", matrix_clip))
    clip_sigmas_by_class["mlp"] = float(env.get("MLP_CLIP_SIGMAS", matrix_clip))
    clip_sigmas_by_class["embed"] = float(env.get("EMBED_CLIP_SIGMAS", policy.embed_clip_sigmas))
    clip_sigmas_by_class["other"] = float(env.get("OTHER_CLIP_SIGMAS", other_clip))
    clip_sigmas_by_class["control"] = float(env.get("CONTROL_CLIP_SIGMAS", clip_sigmas_by_class["other"]))

    calibration_strategy = env.get("GPTQ_CALIBRATION_STRATEGY", "uniform")
    if calibration_strategy not in SUPPORTED_CALIBRATION_STRATEGIES:
        raise ValueError(
            f"Unsupported GPTQ_CALIBRATION_STRATEGY={calibration_strategy!r}; "
            f"expected one of {SUPPORTED_CALIBRATION_STRATEGIES}"
        )
    gptq_mode = env.get("GPTQ_MODE", "full")
    if gptq_mode not in SUPPORTED_GPTQ_MODES:
        raise ValueError(f"Unsupported GPTQ_MODE={gptq_mode!r}; expected one of {SUPPORTED_GPTQ_MODES}")
    compressor = env.get("COMPRESSOR", "zlib")
    if compressor not in SUPPORTED_COMPRESSORS:
        raise ValueError(f"Unsupported COMPRESSOR={compressor!r}; expected one of {SUPPORTED_COMPRESSORS}")

    return QuantConfig(
        policy_name=policy_name,
        calibration_strategy=calibration_strategy,
        calibration_batches=int(env.get("GPTQ_CALIBRATION_BATCHES", env.get("GPTQ_CALIB_BATCHES", "64"))),
        gptq_mode=gptq_mode,
        gptq_block_size=int(env.get("GPTQ_BLOCK_SIZE", "128")),
        gptq_damp_factor=float(env.get("GPTQ_DAMP_FACTOR", "0.01")),
        compressor=compressor,
        bits_by_class=bits_by_class,
        clip_sigmas_by_class=clip_sigmas_by_class,
        passthrough_max_numel=int(env.get("QUANT_PASSTHROUGH_MAX_NUMEL", str(DEFAULT_PASSTHROUGH_MAX_NUMEL))),
    )


def quant_surface_record(config: QuantConfig) -> dict[str, object]:
    return {
        "policy_name": config.policy_name,
        "calibration_strategy": config.calibration_strategy,
        "calibration_batches": config.calibration_batches,
        "gptq_mode": config.gptq_mode,
        "gptq_block_size": config.gptq_block_size,
        "gptq_damp_factor": config.gptq_damp_factor,
        "compressor": config.compressor,
        "bits_by_class": dict(config.bits_by_class),
        "clip_sigmas_by_class": dict(config.clip_sigmas_by_class),
        "passthrough_max_numel": config.passthrough_max_numel,
    }


def bits_for_tensor(name: str, config: QuantConfig) -> int:
    return int(config.bits_by_class[classify_tensor(name)])


def clip_sigmas_for_tensor(name: str, config: QuantConfig) -> float:
    return float(config.clip_sigmas_by_class[classify_tensor(name)])


def quantize_tensor_per_row(t: Tensor, *, bits: int, clip_sigmas: float) -> tuple[Tensor, Tensor]:
    levels = 2 ** (bits - 1) - 1
    t32 = t.float()
    if t32.ndim == 2:
        row_std = t32.std(dim=1).clamp_min(1e-8)
        clip_abs = clip_sigmas * row_std
        scale = (clip_abs / float(levels)).clamp_min(1.0 / float(levels))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        q = torch.clamp(torch.round(clipped / scale[:, None]), -levels, levels).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(t32.std().clamp_min(1e-8).item()) * clip_sigmas
    scale = torch.tensor(clip_abs / float(levels) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -levels, levels).to(torch.int8)
    return q.contiguous(), scale


def quantize_tensor_gptq(
    weight: Tensor,
    hessian: Tensor | None,
    *,
    bits: int,
    clip_sigmas: float,
    block_size: int,
    damp_factor: float,
) -> tuple[Tensor, Tensor]:
    if hessian is None or weight.ndim != 2:
        return quantize_tensor_per_row(weight, bits=bits, clip_sigmas=clip_sigmas)

    clip_range = 2 ** (bits - 1) - 1
    t32 = weight.float()
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = damp_factor * torch.mean(torch.diag(H)).clamp_min(1e-6)
    H.diagonal().add_(damp)
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    try:
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except RuntimeError:
        H.diagonal().add_(damp * 10)
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

    row_std = t32.std(dim=1).clamp_min(1e-8)
    scale = (clip_sigmas * row_std / float(clip_range)).clamp_min(1.0 / float(clip_range)).to(torch.float16)
    scale_f = scale.float()
    q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W.clone()
    for start in range(0, cols, block_size):
        end = min(start + block_size, cols)
        W_block = W_work[:, start:end].clone()
        Hinv_block = Hinv[start:end, start:end]
        err_block = torch.zeros(rows, end - start)
        for j in range(end - start):
            w_col = W_block[:, j]
            denom = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / scale_f), -clip_range, clip_range).to(torch.int8)
            q[:, start + j] = q_col
            err = (w_col - q_col.float() * scale_f) / denom
            err_block[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if end < cols:
            W_work[:, end:] -= err_block @ Hinv[start:end, end:]
    return q[:, inv_perm].contiguous(), scale.contiguous()


def _should_keep_float(
    name: str,
    tensor: Tensor,
    *,
    config: QuantConfig,
    passthrough_predicate: Callable[[str, Tensor], bool] | None,
) -> bool:
    if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
        return True
    if passthrough_predicate is not None and passthrough_predicate(name, tensor):
        return True
    return tensor.numel() <= config.passthrough_max_numel


def quantize_state_dict(
    state_dict: dict[str, Tensor],
    *,
    config: QuantConfig,
    hessians: Mapping[str, Tensor] | None = None,
    passthrough_predicate: Callable[[str, Tensor], bool] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    payload: dict[str, object] = {
        "__quant_format__": f"{config.policy_name}_gptq_surface_v1",
        "config": quant_surface_record(config),
        "quantized": {},
        "scales": {},
        "dtypes": {},
        "passthrough": {},
        "passthrough_orig_dtypes": {},
        "qmeta": {},
    }
    summary = {
        **quant_surface_record(config),
        "baseline_tensor_bytes": 0,
        "quantized_tensor_bytes": 0,
        "hessian_tensor_count": 0 if hessians is None else len(hessians),
        "tensor_classes": {},
    }
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        tensor_class = classify_tensor(name)
        class_summary = summary["tensor_classes"].setdefault(
            tensor_class,
            {
                "count": 0,
                "baseline_bytes": 0,
                "quantized_bytes": 0,
                "bits": None,
                "clip_sigmas": None,
                "methods": {},
            },
        )
        class_summary["count"] += 1
        class_summary["baseline_bytes"] += tensor_nbytes(t)
        summary["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            payload["passthrough"][name] = t
            class_summary["quantized_bytes"] += tensor_nbytes(t)
            class_summary["methods"]["passthrough_nonfloat"] = class_summary["methods"].get("passthrough_nonfloat", 0) + 1
            summary["quantized_tensor_bytes"] += tensor_nbytes(t)
            continue

        if _should_keep_float(name, t, config=config, passthrough_predicate=passthrough_predicate):
            kept = t.to(dtype=torch.float16) if t.dtype in {torch.float32, torch.bfloat16} else t
            payload["passthrough"][name] = kept
            if kept.dtype != t.dtype:
                payload["passthrough_orig_dtypes"][name] = str(t.dtype).removeprefix("torch.")
            class_summary["quantized_bytes"] += tensor_nbytes(kept)
            class_summary["methods"]["passthrough_float"] = class_summary["methods"].get("passthrough_float", 0) + 1
            summary["quantized_tensor_bytes"] += tensor_nbytes(kept)
            continue

        bits = bits_for_tensor(name, config)
        clip_sigmas = clip_sigmas_for_tensor(name, config)
        hessian = None if hessians is None else hessians.get(name)
        use_gptq = config.gptq_mode == "full" and hessian is not None and t.ndim == 2
        if use_gptq:
            q, s = quantize_tensor_gptq(
                t,
                hessian,
                bits=bits,
                clip_sigmas=clip_sigmas,
                block_size=config.gptq_block_size,
                damp_factor=config.gptq_damp_factor,
            )
            method = "gptq"
        else:
            q, s = quantize_tensor_per_row(t, bits=bits, clip_sigmas=clip_sigmas)
            method = "rowwise" if config.gptq_mode == "full" else f"rowwise_{config.gptq_mode}"
        payload["quantized"][name] = q
        payload["scales"][name] = s
        payload["dtypes"][name] = str(t.dtype).removeprefix("torch.")
        payload["qmeta"][name] = {
            "scheme": "per_row" if t.ndim == 2 else "per_tensor",
            "bits": bits,
            "clip_sigmas": clip_sigmas,
            "tensor_class": tensor_class,
            "method": method,
        }
        quantized_bytes = tensor_nbytes(q) + tensor_nbytes(s)
        class_summary["quantized_bytes"] += quantized_bytes
        class_summary["bits"] = bits
        class_summary["clip_sigmas"] = clip_sigmas
        class_summary["methods"][method] = class_summary["methods"].get(method, 0) + 1
        summary["quantized_tensor_bytes"] += quantized_bytes

    summary["compression_ratio_estimate"] = round(
        summary["baseline_tensor_bytes"] / max(summary["quantized_tensor_bytes"], 1),
        4,
    )
    return payload, summary


def dequantize_tensor(q: Tensor, scale: Tensor, *, dtype: torch.dtype) -> Tensor:
    if scale.ndim > 0:
        return (q.float() * scale.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype)
    return (q.float() * float(scale.item())).to(dtype=dtype)


def dequantize_state_dict(payload: dict[str, object]) -> dict[str, Tensor]:
    state_dict: dict[str, Tensor] = {}
    passthrough_orig_dtypes = payload.get("passthrough_orig_dtypes", {})
    for name, tensor in payload.get("passthrough", {}).items():
        restored = tensor.detach().to("cpu").contiguous()
        dtype_name = passthrough_orig_dtypes.get(name)
        if isinstance(dtype_name, str):
            restored = restored.to(dtype=getattr(torch, dtype_name)).contiguous()
        state_dict[name] = restored
    for name, q in payload.get("quantized", {}).items():
        dtype = getattr(torch, payload["dtypes"][name])
        state_dict[name] = dequantize_tensor(q, payload["scales"][name], dtype=dtype).contiguous()
    return state_dict


def serialize_quantized_payload(payload: dict[str, object], *, compressor: str = "zlib") -> tuple[bytes, bytes]:
    raw = io.BytesIO()
    torch.save(payload, raw)
    raw_bytes = raw.getvalue()
    if compressor == "zlib":
        return raw_bytes, zlib.compress(raw_bytes, level=9)
    if compressor == "zstd":
        if zstd is None:
            raise ImportError("zstandard is required for COMPRESSOR=zstd")
        return raw_bytes, zstd.ZstdCompressor(level=22).compress(raw_bytes)
    if compressor == "lzma":
        return raw_bytes, lzma.compress(raw_bytes, preset=6)
    raise ValueError(f"Unsupported compressor {compressor!r}")


def quant_gap_report(training_best_val_bpb: float | None, post_quant_bpb: float | None) -> dict[str, float | None]:
    if training_best_val_bpb is None or post_quant_bpb is None:
        return {
            "training_best_val_bpb": training_best_val_bpb,
            "post_quant_bpb": post_quant_bpb,
            "prequant_to_postquant_delta_bpb": None,
        }
    return {
        "training_best_val_bpb": training_best_val_bpb,
        "post_quant_bpb": post_quant_bpb,
        "prequant_to_postquant_delta_bpb": round(post_quant_bpb - training_best_val_bpb, 6),
    }


def quant_summary_json(summary: dict[str, object]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"
