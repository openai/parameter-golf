"""Train a SentencePiece BPE tokenizer at vocab=10240 on CaseOps-transformed
FineWeb text. Mirrors the SP8192 CaseOps tokenizer used by PR #1797.

The lossless_caps_v2 transform reserves 4 user-defined symbols: TITLE, ALLCAPS,
CAPNEXT, ESC. We reserve them as `user_defined_symbols=...` so SP keeps them
as singleton ids regardless of frequency.

Outputs:
  --out_prefix.model
  --out_prefix.vocab
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "records" / "track_10min_16mb" / "2026-04-25_PR1797Base_NGramMix"))
from lossless_caps import encode_lossless_caps_v2  # noqa: E402


# Match prepare_caseops_data.py reserved-token set.
RESERVED_OPS = [chr(0xE001), chr(0xE002), chr(0xE003), chr(0xE004), chr(0xE005)]


def _iter_docs(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["text"] if isinstance(obj, dict) else obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, type=Path,
                    help="Path to docs_selected.jsonl")
    ap.add_argument("--out_prefix", required=True, type=str,
                    help="Output prefix (writes <prefix>.model and <prefix>.vocab)")
    ap.add_argument("--vocab_size", type=int, default=10240)
    ap.add_argument("--input_sentences", type=int, default=2_000_000,
                    help="Number of CaseOps-transformed lines to feed SP trainer")
    ap.add_argument("--num_threads", type=int, default=os.cpu_count() or 32)
    args = ap.parse_args()

    print(f"vocab_size={args.vocab_size}  input_sentences={args.input_sentences}  threads={args.num_threads}")

    # Stream docs through CaseOps transform into a temp file (sentencepiece
    # accepts file path or python iterator; file is faster for large input).
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".caseops.txt", delete=False)
    n = 0
    for text in _iter_docs(args.docs):
        if n >= args.input_sentences:
            break
        try:
            t = encode_lossless_caps_v2(text)
        except Exception as e:
            continue
        # SP wants newline-separated lines. Replace embedded newlines.
        t = t.replace("\n", " ")
        tmp.write(t + "\n")
        n += 1
        if n % 100_000 == 0:
            print(f"  written {n} caseops-transformed lines", flush=True)
    tmp.close()
    print(f"wrote {n} sentences to {tmp.name}")

    # Train SentencePiece BPE.
    spm.SentencePieceTrainer.train(
        input=tmp.name,
        model_prefix=args.out_prefix,
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=0.9999,
        user_defined_symbols=",".join(RESERVED_OPS),
        bos_id=1, eos_id=2, unk_id=3, pad_id=0,
        byte_fallback=True,
        normalization_rule_name="identity",  # CaseOps already normalized
        num_threads=args.num_threads,
        input_sentence_size=args.input_sentences,
        shuffle_input_sentence=False,
        train_extremely_large_corpus=True,
    )
    print(f"trained {args.out_prefix}.model")
    os.unlink(tmp.name)


if __name__ == "__main__":
    main()
