#!/usr/bin/env python3
"""Phase 1-A: Re-serialize an existing FP32 .pt checkpoint with embedding PTQ.

Reads a model.pt (FP32 state_dict from training) and writes a new
.rans.ptz (+ optional .xz) using the Phase 1 train_gpt.py serialize_hybrid_rans
with EMBED_QUANT_BITS env var controlling embedding PTQ.

No retraining needed.

Usage (run from parameter-golf root):
    EMBED_QUANT_BITS=4 python records/track_10min_16mb/2026-04-09_v62_phase1_quantize/reserialize_with_ptq.py \
        runs/v61_fa3_seq2048_s1337/model.pt \
        runs/v62_phase1a_int4_s1337/model.rans.ptz
"""
import os
import sys
import lzma
from pathlib import Path

import torch

# Make local train_gpt.py importable
sys.path.insert(0, str(Path(__file__).parent))
from train_gpt import (
    make_model,
    serialize_hybrid_rans,
)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    in_pt = sys.argv[1]
    out_ptz = sys.argv[2]
    out_dir = Path(out_ptz).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[reserialize] in:  {in_pt}")
    print(f"[reserialize] out: {out_ptz}")
    spec = os.environ.get("EMBED_QUANT_BITS", "0")
    print(f"[reserialize] EMBED_QUANT_BITS={spec}")

    # Load FP32 checkpoint
    print(f"[reserialize] loading {in_pt} ...")
    ckpt = torch.load(in_pt, map_location="cpu", weights_only=False)
    if "model" in ckpt and "step" in ckpt:
        if "ema_shadow" in ckpt:
            ema_state = ckpt["ema_shadow"]
            if "fast" in ema_state:
                state_dict = ema_state["smoother"]
            else:
                state_dict = ema_state
        else:
            state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Build empty model with same config and load weights
    print("[reserialize] building model and loading weights ...")
    model = make_model()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[reserialize] WARNING missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"[reserialize] WARNING unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")
    model.eval()

    print("[reserialize] running serialize_hybrid_rans ...")
    obj = serialize_hybrid_rans(model)
    torch.save(obj, out_ptz)
    rans_size = os.path.getsize(out_ptz)
    print(f"[reserialize] wrote {out_ptz} ({rans_size:,} bytes = {rans_size/2**20:.2f} MB)")

    # lzma9 extreme post-compression for size comparison
    if int(os.environ.get("LZMA9_AFTER_RANS", "1")):
        with open(out_ptz, "rb") as f:
            rans_bytes = f.read()
        xz_path = out_ptz + ".xz"
        with open(xz_path, "wb") as f:
            f.write(lzma.compress(rans_bytes, preset=9 | lzma.PRESET_EXTREME))
        xz_size = os.path.getsize(xz_path)
        print(f"[reserialize] +lzma9 wrote {xz_path} ({xz_size:,} bytes = {xz_size/2**20:.2f} MB, "
              f"{(rans_size-xz_size)/rans_size*100:.1f}% saved)")
        print(f"[reserialize] under 16MB: {'YES' if xz_size < 16_000_000 else 'NO'}")

    print("[reserialize] done.")


if __name__ == "__main__":
    main()
