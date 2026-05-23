"""Re-export SmolLM2 from HF with cumulative-shape `present.*` outputs.

optimum 2.1.0 + transformers 4.57.x has a bug: `patched_dynamic_layer_update`
in optimum/exporters/onnx/model_patcher.py:490 takes `if self.keys is None`
on the first ONNX trace call, stores `key_states` directly, and never traces
the `torch.cat([self.keys, key_states], dim=-2)` branch into the graph. The
exported ONNX therefore returns delta-only `present.*` (just the new tokens'
KV) — making the with-past export incompatible with the standard
"replace cache from present.*" decode loop.

Workaround: monkey-patch the function before the patcher's __enter__ caches
its reference. Replace it with a version that initializes `self.keys` to an
empty-along-T tensor on first call so the always-concat path traces cleanly.
"""

from __future__ import annotations

import sys

import optimum.exporters.onnx.model_patcher as mp
import torch


def _always_concat_dynamic_layer_update(
    self, key_states, value_states, cache_kwargs=None
):
    """Replacement for patched_dynamic_layer_update that always exercises the
    Concat path during ONNX tracing.

    The original function had `if self.keys is None: self.keys = key_states`
    on the first call — which is the only call seen during a single-pass
    trace. By initializing the cache to an empty-along-seq-len tensor we
    force the `torch.cat` to be reached on every layer's first call, which
    is what gets recorded into the graph as
    `Concat([past_key_values_input, key_states], axis=-2)`.
    """
    if self.keys is None:
        zero_shape = list(key_states.shape)
        zero_shape[-2] = 0  # empty along seq_len axis (dim -2 in [B, H, T, D])
        self.keys = key_states.new_empty(zero_shape)
        self.values = value_states.new_empty(zero_shape)
        self.device = key_states.device
        self.dtype = key_states.dtype
        self.is_initialized = True
    self.keys = torch.cat([self.keys, key_states], dim=-2)
    self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values


# Replace the module-level reference *before* anything calls
# ModelPatcher.__enter__ (which captures the function via
# `DynamicLayer.update = patched_dynamic_layer_update`).
mp.patched_dynamic_layer_update = _always_concat_dynamic_layer_update
print("[patch] optimum.exporters.onnx.model_patcher.patched_dynamic_layer_update "
      "→ always-concat replacement", flush=True)

# Now invoke optimum-cli's main as if we'd run it directly.
from optimum.commands.optimum_cli import main as optimum_main

if __name__ == "__main__":
    sys.argv = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "--task",
        "text-generation-with-past",
        "--opset",
        "14",
        ".",
    ]
    optimum_main()
