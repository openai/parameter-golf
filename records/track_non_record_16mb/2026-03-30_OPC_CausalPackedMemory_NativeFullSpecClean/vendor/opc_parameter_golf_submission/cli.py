from __future__ import annotations

import argparse

from .packet import build_packet_from_patterns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a standalone opc-native packed-memory parameter-golf-style legal packet.")
    parser.add_argument("--train-pattern", action="append", required=True, dest="train_patterns")
    parser.add_argument("--eval-pattern", action="append", required=True, dest="eval_patterns")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-train-tokens", type=int)
    parser.add_argument("--max-eval-tokens", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--bytes-per-token", type=float)
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--name", default="OPC causal packed-memory legal packet stress test")
    parser.add_argument("--track", default="track_non_record_16mb")
    parser.add_argument("--candidate-id", default="opc-causal-packed-memory-stress-test")
    parser.add_argument("--submission-pr", default="https://github.com/openai/parameter-golf/pull/998")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = build_packet_from_patterns(
        train_patterns=args.train_patterns,
        eval_patterns=args.eval_patterns,
        out_dir=args.out_dir,
        max_train_tokens=args.max_train_tokens,
        max_eval_tokens=args.max_eval_tokens,
        vocab_size=args.vocab_size,
        bytes_per_token=args.bytes_per_token,
        tokenizer_model=args.tokenizer_model,
        submission_name=args.name,
        track=args.track,
        candidate_id=args.candidate_id,
        submission_pr=args.submission_pr,
    )
    print(f"built packet: {result.output_root}")
    print(f"run_id: {result.run_id}")
    print(f"pre_quant_val_bpb: {result.pre_quant_val_bpb}")


__all__ = ["build_parser", "main"]
