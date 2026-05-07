#!/usr/bin/env python3
"""Patch PR #1953 train_gpt.py with Path-A-v3-style small tensor int8 routing.

#1953 already accounts code via a compressed wrapper, so source packing does not
create real headroom there. The useful Path A v3 lever for this lineage is model
tensor routing: control tensors and tiny 2-D control matrices should not remain
fp16 passthrough when artifact margin is only a few KB.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "small_control_int8_tensor" in text:
        print(f"{path}: Path A v3 small-tensor patch already present")
        return

    text = replace_once(
        text,
        '    gated_attn_quant_gate = bool(int(os.environ.get("GATED_ATTN_QUANT_GATE", "0")))\n',
        '    gated_attn_quant_gate = bool(int(os.environ.get("GATED_ATTN_QUANT_GATE", "0")))\n'
        '    path_a_v3_small = bool(int(os.environ.get("PATH_A_V3_SMALL", "0")))\n',
        "Hyperparameters gated_attn_quant_gate",
    )

    text = replace_once(
        text,
        "        if not t.is_floating_point() or t.numel() <= 65536:\n"
        "            result[name] = t.to(torch.float16) if t.is_floating_point() else t\n"
        "            meta[name] = \"passthrough (float16)\"\n"
        "            continue\n",
        "        if (\n"
        "            bool(getattr(h, \"path_a_v3_small\", False))\n"
        "            and t.is_floating_point()\n"
        "            and t.numel() <= 65536\n"
        "            and any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)\n"
        "        ):\n"
        "            if t.ndim == 2:\n"
        "                gq, gs = _quantize_gate_int8_row(t)\n"
        "                result[name + \".gq\"] = gq\n"
        "                result[name + \".gs\"] = gs\n"
        "                meta[name] = \"small_control_int8_row\"\n"
        "            else:\n"
        "                qmax = 127\n"
        "                s = (t.float().abs().amax().clamp_min(1e-10) / qmax).to(torch.float16)\n"
        "                q = torch.clamp(torch.round(t.float() / float(s)), -qmax, qmax).to(torch.int8)\n"
        "                result[name + \".iq\"] = q\n"
        "                result[name + \".is\"] = s\n"
        "                meta[name] = \"small_control_int8_tensor\"\n"
        "            continue\n"
        "        if not t.is_floating_point() or t.numel() <= 65536:\n"
        "            result[name] = t.to(torch.float16) if t.is_floating_point() else t\n"
        "            meta[name] = \"passthrough (float16)\"\n"
        "            continue\n",
        "quant small control tensors",
    )

    if '        if info == "gate_int8_row":\n' in text:
        text = text.replace(
            '        if info == "gate_int8_row":\n',
            '        if info in ("gate_int8_row", "small_control_int8_row"):\n',
            1,
        )
    elif '        if info in ("gate_int8_row", "bigram_proj_int8_row"):\n' in text:
        text = text.replace(
            '        if info in ("gate_int8_row", "bigram_proj_int8_row"):\n',
            '        if info in ("gate_int8_row", "bigram_proj_int8_row", "small_control_int8_row"):\n',
            1,
        )
    else:
        raise SystemExit("missing patch anchor: dequant small row")

    text = replace_once(
        text,
        "            continue\n"
        "        q, s = result[name + \".q\"], result[name + \".scale\"]\n",
        "            continue\n"
        "        if info == \"small_control_int8_tensor\":\n"
        "            out[name] = (result[name + \".iq\"].float() * float(result[name + \".is\"].item())).to(orig_dtype)\n"
        "            continue\n"
        "        q, s = result[name + \".q\"], result[name + \".scale\"]\n",
        "dequant small tensor",
    )

    path.write_text(text, encoding="utf-8")
    print(f"{path}: applied Path A v3 small-tensor patch")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_path_a_v3_small_1953.py /path/to/train_gpt.py")
    patch(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
