"""
Quantise an HNet checkpoint to int8 + zlib for the parameter-golf 16MB artifact.

Mirrors train_gpt.py's INT8 pipeline (per-row int8 for 2D matrices, per-tensor
int8 for 1D, tiny tensors kept as fp16) and zlib-compresses the pickled dict.

Usage:
    python quantize_hnet.py logs/hnet_real_final.pt
Produces:
    logs/hnet_real_final.int8.ptz
"""
from __future__ import annotations

import io
import pickle
import sys
import zlib
from pathlib import Path

import torch
from torch import Tensor


INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict(sd: dict[str, Tensor]) -> dict:
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_dtypes: dict[str, str] = {}

    for name, tensor in sd.items():
        t = tensor.detach().to("cpu").contiguous()
        if not t.is_floating_point():
            passthrough[name] = t
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_dtypes[name] = str(t.dtype).removeprefix("torch.")
                passthrough[name] = t.to(INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
            else:
                passthrough[name] = t
            continue
        q, s = quantize_float_tensor(t)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")

    return {
        "__quant_format__": "int8_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "passthrough_orig_dtypes": passthrough_dtypes,
    }


def dequantize(blob: dict) -> dict[str, Tensor]:
    """Inverse of quantize_state_dict — returns a state_dict ready for load_state_dict."""
    out: dict[str, Tensor] = {}
    quantized = blob["quantized"]
    scales = blob["scales"]
    dtypes = blob["dtypes"]
    for name, q in quantized.items():
        s = scales[name].to(torch.float32)
        target_dtype = getattr(torch, dtypes[name])
        if q.ndim == 2:
            out[name] = (q.to(torch.float32) * s[:, None]).to(target_dtype)
        else:
            out[name] = (q.to(torch.float32) * s).to(target_dtype)
    for name, t in blob["passthrough"].items():
        orig = blob.get("passthrough_orig_dtypes", {}).get(name)
        out[name] = t.to(getattr(torch, orig)) if orig else t
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python quantize_hnet.py <path_to_final.pt>")
        sys.exit(1)
    in_path = Path(sys.argv[1])

    print(f"loading {in_path}")
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    n_params = sum(t.numel() for t in sd.values() if t.is_floating_point())
    raw_bytes = sum(t.numel() * t.element_size() for t in sd.values())
    print(f"params:       {n_params:,}")
    print(f"raw size:     {raw_bytes:>12,} bytes  ({raw_bytes/1e6:6.2f} MB)")

    blob = quantize_state_dict(sd)
    buf = io.BytesIO()
    pickle.dump(blob, buf, protocol=pickle.HIGHEST_PROTOCOL)
    pickled = buf.getvalue()
    compressed = zlib.compress(pickled, level=9)

    out_path = in_path.with_suffix(".int8.ptz")
    out_path.write_bytes(compressed)
    print(f"int8 pickled: {len(pickled):>12,} bytes  ({len(pickled)/1e6:6.2f} MB)")
    print(f"int8+zlib:    {len(compressed):>12,} bytes  ({len(compressed)/1e6:6.2f} MB)")
    print(f"compression:  {raw_bytes / max(1, len(compressed)):.2f}x")
    print(f"saved:        {out_path}")

    # Verify round-trip: pickled-decompress-dequantize and compare a few tensors.
    rt = pickle.loads(zlib.decompress(out_path.read_bytes()))
    rt_sd = dequantize(rt)
    n_check = 0
    max_err = 0.0
    for name, orig in sd.items():
        if orig.is_floating_point() and orig.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            err = (orig.float() - rt_sd[name].float()).abs().max().item()
            max_err = max(max_err, err)
            n_check += 1
    print(f"round-trip:   {n_check} tensors checked, max abs error = {max_err:.4f}")


if __name__ == "__main__":
    main()
