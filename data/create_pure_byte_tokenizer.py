"""Create a fixed pure byte-level tokenizer artifact.

This tokenizer does not require corpus training.
"""

import argparse
import os

from pure_byte_tokenizer import default_pure_byte_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Create pure byte tokenizer JSON artifact")
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "tokenizers", "fineweb_pure_byte_260.json"),
    )
    args = parser.parse_args()

    tok = default_pure_byte_tokenizer()
    tok.save_json(args.output_json)
    print(f"Wrote {args.output_json}")
    print(
        f"vocab_size={tok.vocab_size} "
        f"pad_id={tok.pad_id} bos_id={tok.bos_id} eos_id={tok.eos_id} unk_id={tok.unk_id}"
    )


if __name__ == "__main__":
    main()

