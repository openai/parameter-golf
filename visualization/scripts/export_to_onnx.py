#!/usr/bin/env python3
"""Export AR (GPT) and MDLM (DiffusionTransformer) to ONNX for fast inference
in the browser via onnxruntime-web. The hand-rolled JS forward stays — it's
our parity checker — but onnxruntime-web runs the same graph orders of
magnitude faster on WebGPU/WASM-SIMD.

Usage:
  python visualization/scripts/export_to_onnx.py \\
      --kind mdlm --artifact artifacts/mdlm-submission-seed42-19M.ptz \\
      --out visualization/public/models/mdlm/model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "text_diffusion"))
sys.path.insert(0, str(REPO_ROOT / "visualization" / "scripts"))

from generate_goldens import build_ar_model, build_diffusion_model


class MdlmWrapper(torch.nn.Module):
    """Wrap DiffusionTransformer so its forward() takes (input_ids, mask)."""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, mask):
        # mask comes in as int64 0/1; the model expects bool
        return self.m.forward_logits(input_ids, mask.bool())


class ArWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids):
        return self.m.forward_logits(input_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["ar", "mdlm"], required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seq-len", type=int, default=512,
                   help="Reference seq_len for the dummy export input. The "
                        "exported graph supports dynamic seq_len anyway.")
    args = p.parse_args()

    if args.kind == "mdlm":
        model, label = build_diffusion_model(args.artifact)
        wrap = MdlmWrapper(model).eval()
        dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.int64)
        dummy_mask = torch.zeros(1, args.seq_len, dtype=torch.int64)
        inputs = (dummy_ids, dummy_mask)
        input_names = ["input_ids", "mask"]
        dynamic_axes = {
            "input_ids": {1: "seq_len"},
            "mask": {1: "seq_len"},
            "logits": {1: "seq_len"},
        }
    else:
        model, label = build_ar_model(args.artifact)
        wrap = ArWrapper(model).eval()
        dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.int64)
        inputs = (dummy_ids,)
        input_names = ["input_ids"]
        dynamic_axes = {
            "input_ids": {1: "seq_len"},
            "logits": {1: "seq_len"},
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"exporting {label} → {out}")
    print(f"  dummy input seq_len: {args.seq_len}  (graph is dynamic in seq_len)")
    torch.onnx.export(
        wrap,
        inputs,
        str(out),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  wrote {out}  ({size_mb:.1f} MB)")

    # Sanity check: load the ONNX, run it on dummy inputs, diff vs the torch model.
    print("\n  sanity check via onnxruntime CPU...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    n = 32
    test_ids = torch.arange(1, n + 1, dtype=torch.int64).unsqueeze(0)
    if args.kind == "mdlm":
        mask = torch.zeros(1, n, dtype=torch.int64)
        mask[0, ::3] = 1
        with torch.inference_mode():
            ref = wrap(test_ids, mask).float()
        got = sess.run(["logits"], {"input_ids": test_ids.numpy(), "mask": mask.numpy()})[0]
    else:
        with torch.inference_mode():
            ref = wrap(test_ids).float()
        got = sess.run(["logits"], {"input_ids": test_ids.numpy()})[0]
    import numpy as np
    diff = np.abs(ref.numpy() - got).max()
    rel = float(diff) / max(float(np.abs(ref.numpy()).max()), 1e-9)
    print(f"  forward_logits diff: max abs = {diff:.3e}  rel = {rel:.3e}")
    # Also list any external-data files dropped next to the .onnx
    for f in out.parent.glob(f"{out.stem}*"):
        if f != out:
            print(f"  side file: {f.name}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
