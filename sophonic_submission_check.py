from __future__ import annotations

import argparse
import io
import os
import time
import zlib
from pathlib import Path

import sentencepiece as spm
import torch

from sophonic_eval import (
    EvalConfig,
    GPT,
    build_sentencepiece_luts,
    eval_bpb,
    int8_roundtrip,
    load_validation_tokens,
)
from train_gpt import quantize_state_dict_int8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-val and artifact-size check for a merged Sophonics checkpoint."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--val-batch-size", type=int, default=int(os.environ.get("VAL_BATCH_SIZE", "131072")))
    parser.add_argument(
        "--code-files",
        default="train_gpt.py,sophonic_lora_repair.py",
        help="Comma-separated list of code files to count toward artifact size.",
    )
    parser.add_argument(
        "--save-int8-path",
        default="",
        help="Optional path to save the zlib-compressed int8 artifact.",
    )
    return parser


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_state_dict(raw_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in raw_sd):
        return {k.replace("module.", ""): v for k, v in raw_sd.items()}
    return raw_sd


def eval_state_dict(
    label: str,
    model: GPT,
    state_dict: dict[str, torch.Tensor],
    val_tokens: torch.Tensor,
    cfg: EvalConfig,
    base_bytes_lut: torch.Tensor,
    has_space_lut: torch.Tensor,
    is_boundary_lut: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()}, strict=False)
    t0 = time.time()
    val_loss, bpb = eval_bpb(
        model,
        val_tokens,
        cfg.seq_len,
        cfg.val_batch_size,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
    )
    dt = time.time() - t0
    print(f"{label:<36} val_bpb={bpb:.4f}  val_loss={val_loss:.4f}  ({dt:.0f}s)")
    return val_loss, bpb


def code_bytes(paths: list[str]) -> int:
    total = 0
    for raw in paths:
        path = Path(raw.strip())
        if not raw.strip():
            continue
        if not path.exists():
            raise FileNotFoundError(f"Code file not found: {path}")
        total += path.stat().st_size
    return total


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    print(f"Device: {device}")

    cfg = EvalConfig()
    cfg.model_path = args.model
    cfg.val_batch_size = args.val_batch_size

    print(f"Loading model: {cfg.model_path}")
    raw_sd = normalize_state_dict(torch.load(cfg.model_path, map_location="cpu", weights_only=True))

    print(f"Loading validation tokens: {cfg.val_files}")
    val_tokens = load_validation_tokens(cfg.val_files, cfg.seq_len)
    print(f"Validation tokens: {val_tokens.numel() - 1:,}")

    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    base_bytes_lut, has_space_lut, is_boundary_lut = build_sentencepiece_luts(sp, cfg.vocab_size, device)

    model = GPT(cfg).to(device).float()

    print("\n" + "=" * 72)
    print("FULL-VAL CHECK")
    print("=" * 72)
    _, fp32_bpb = eval_state_dict(
        "Merged fp32 checkpoint",
        model,
        raw_sd,
        val_tokens,
        cfg,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
    )

    int8_sd = int8_roundtrip(raw_sd)
    _, int8_bpb = eval_state_dict(
        "Merged int8 roundtrip",
        model,
        int8_sd,
        val_tokens,
        cfg,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
    )

    print("\n" + "=" * 72)
    print("ARTIFACT SIZE")
    print("=" * 72)
    quant_obj, _ = quantize_state_dict_int8(raw_sd)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw_bytes = buf.getvalue()
    compressed = zlib.compress(raw_bytes, level=9)
    if args.save_int8_path:
        out_path = Path(args.save_int8_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(compressed)
        print(f"Saved int8+zlib artifact to {out_path}")

    code_list = [part for part in args.code_files.split(",") if part.strip()]
    bytes_code = code_bytes(code_list)
    bytes_model = len(compressed)
    bytes_total = bytes_model + bytes_code

    print(f"bytes_model_int8_zlib: {bytes_model}")
    print(f"bytes_code: {bytes_code}")
    print(f"bytes_total: {bytes_total}")
    print(f"fp32_val_bpb: {fp32_bpb:.6f}")
    print(f"int8_val_bpb: {int8_bpb:.6f}")


if __name__ == "__main__":
    main()
