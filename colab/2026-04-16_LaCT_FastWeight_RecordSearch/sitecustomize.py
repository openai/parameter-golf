from __future__ import annotations

import os


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like string, got {value!r}")


try:
    import torch
except Exception:
    torch = None


if torch is not None and _env_flag("PG_COLAB_DISABLE_COMPILE", False):
    def _compile_passthrough(model, *args, **kwargs):
        return model

    torch.compile = _compile_passthrough


if torch is not None and _env_flag("PG_COLAB_DISABLE_FUSED_ADAM", False):
    _orig_adamw = torch.optim.AdamW
    _orig_adam = torch.optim.Adam

    class _AdamWNoFused(_orig_adamw):
        def __init__(self, *args, **kwargs):
            kwargs.pop("fused", None)
            super().__init__(*args, **kwargs)

    class _AdamNoFused(_orig_adam):
        def __init__(self, *args, **kwargs):
            kwargs.pop("fused", None)
            super().__init__(*args, **kwargs)

    torch.optim.AdamW = _AdamWNoFused
    torch.optim.Adam = _AdamNoFused


if torch is not None and _env_flag("PG_COLAB_FORCE_FP16", False):
    _orig_autocast = torch.autocast
    _orig_tensor_bfloat16 = torch.Tensor.bfloat16
    _orig_module_bfloat16 = torch.nn.Module.bfloat16

    def _autocast_fp16(device_type: str, dtype=None, *args, **kwargs):
        if device_type == "cuda" and dtype == torch.bfloat16:
            dtype = torch.float16
        return _orig_autocast(device_type=device_type, dtype=dtype, *args, **kwargs)

    def _tensor_bfloat16_as_half(self, memory_format=torch.preserve_format):
        return self.half(memory_format=memory_format)

    def _module_bfloat16_as_half(self):
        return self.half()

    torch.autocast = _autocast_fp16
    torch.Tensor.bfloat16 = _tensor_bfloat16_as_half
    torch.nn.Module.bfloat16 = _module_bfloat16_as_half
