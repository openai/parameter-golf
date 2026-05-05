#!/usr/bin/env python3
"""Phase 1+3: Re-serialize an existing FP32 .pt with embedding PTQ AND
optionally write the HQGRANS1 binary container instead of torch.save .ptz.

Usage:
    EMBED_QUANT_BITS=pent EMBED_QUANT_TOK_EMB=1 \
    HQG_BINARY=1 \
    python records/track_10min_16mb/2026-04-09_v62_phase3_binary_container/reserialize_with_ptq_binary.py \
        runs/v61_fa3_seq2048_s1337/model.pt \
        runs/v62_phase3_pent_tok_bin_s1337/model.rans.bin
"""
import os
import sys
import lzma
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_gpt import (
    make_model,
    serialize_hybrid_rans,
    serialize_hybrid_binary,
)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    in_pt = sys.argv[1]
    out_path = sys.argv[2]
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[reserialize] in:  {in_pt}")
    print(f"[reserialize] out: {out_path}")
    spec = os.environ.get("EMBED_QUANT_BITS", "0")
    use_binary = int(os.environ.get("HQG_BINARY", "1"))
    print(f"[reserialize] EMBED_QUANT_BITS={spec}  HQG_BINARY={use_binary}")

    print(f"[reserialize] loading {in_pt} ...")
    ckpt = torch.load(in_pt, map_location="cpu", weights_only=False)
    if "model" in ckpt and "step" in ckpt:
        if "ema_shadow" in ckpt:
            ema_state = ckpt["ema_shadow"]
            state_dict = ema_state["smoother"] if "fast" in ema_state else ema_state
        else:
            state_dict = ckpt["model"]
    else:
        state_dict = ckpt

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

    if use_binary:
        print("[reserialize] running serialize_hybrid_binary (HQGRANS1 V1) ...")
        blob = serialize_hybrid_binary(model)
        with open(out_path, "wb") as f:
            f.write(blob)
        rans_size = os.path.getsize(out_path)
        print(f"[reserialize] wrote {out_path} ({rans_size:,} bytes = {rans_size/2**20:.2f} MB)")
    else:
        print("[reserialize] running serialize_hybrid_rans (torch.save .ptz) ...")
        obj = serialize_hybrid_rans(model)
        torch.save(obj, out_path)
        rans_size = os.path.getsize(out_path)
        print(f"[reserialize] wrote {out_path} ({rans_size:,} bytes = {rans_size/2**20:.2f} MB)")

    if int(os.environ.get("LZMA9_AFTER_RANS", "1")):
        with open(out_path, "rb") as f:
            rans_bytes = f.read()
        xz_path = out_path + ".xz"
        with open(xz_path, "wb") as f:
            f.write(lzma.compress(rans_bytes, preset=9 | lzma.PRESET_EXTREME))
        xz_size = os.path.getsize(xz_path)
        print(f"[reserialize] +lzma9 wrote {xz_path} ({xz_size:,} bytes = {xz_size/2**20:.2f} MB, "
              f"{(rans_size-xz_size)/rans_size*100:.1f}% saved)")
        print(f"[reserialize] under 16MB: {'YES' if xz_size < 16_000_000 else 'NO'}")


if __name__ == "__main__":
    main()
