from __future__ import annotations

import argparse
from pathlib import Path

from .codecs import ByteCodec
from .model import OpenPredictiveCoder


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opc", description="Open Predictive Coder CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="fit on a text file and optionally sample")
    fit_parser.add_argument("--input", required=True, help="Path to a UTF-8 text file")
    fit_parser.add_argument("--prompt", default="", help="Prompt used for sampling after fit")
    fit_parser.add_argument("--generate", type=int, default=0, help="Number of bytes to generate after the prompt")
    fit_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    fit_parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "fit":
        text = Path(args.input).read_text(encoding="utf-8")
        model = OpenPredictiveCoder()
        report = model.fit(text)
        print(f"train bits/byte: {report.train_bits_per_byte:.4f}")
        print(f"patches: {report.patches}")
        print(f"mean patch size: {report.mean_patch_size:.2f}")
        if args.generate > 0:
            prompt = ByteCodec.encode_text(args.prompt or text[:16])
            sample = model.generate(
                prompt,
                steps=args.generate,
                temperature=args.temperature,
                greedy=args.greedy,
            )
            print(ByteCodec.decode_text(sample))


if __name__ == "__main__":
    main()

