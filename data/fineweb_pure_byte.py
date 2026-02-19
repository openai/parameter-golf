"""FineWeb preprocessing for train_gpt.py using a pure byte tokenizer."""

import argparse
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from pure_byte_tokenizer import PureByteTokenizer, default_pure_byte_tokenizer


def write_datafile(filename: str, toks: np.ndarray) -> None:
    """Saves token data as a .bin file: 256 int32 header + uint16 payload."""
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(toks)

    if toks.dtype != np.uint16:
        assert (0 <= toks).all() and (toks < 2**16).all(), "token dictionary too large for uint16"
        toks = toks.astype(np.uint16)

    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="FineWeb pure byte preprocessing")
    parser.add_argument("-v", "--version", type=str, default="10B", choices=["10B", "100B"])
    parser.add_argument("-s", "--shard_size", type=int, default=10**8)
    parser.add_argument(
        "--num_docs",
        type=int,
        default=0,
        help="Maximum number of documents to tokenize (0 means all docs)",
    )
    parser.add_argument(
        "--tokenizer_json",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "tokenizers", "fineweb_pure_byte_260.json"),
        help="Path to pure-byte tokenizer JSON created by create_pure_byte_tokenizer.py. "
        "If missing, defaults are used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output shard directory; default is fineweb10B_byte260 or fineweb100B_byte260.",
    )
    parser.add_argument(
        "--append_eos",
        action="store_true",
        help="Append eos_id after each document (default: disabled).",
    )
    args = parser.parse_args()

    if args.version == "10B":
        remote_name = "sample-10BT"
        default_output = "fineweb10B_byte260"
    else:
        remote_name = "sample-100BT"
        default_output = "fineweb100B_byte260"

    data_cache_dir = os.path.join(os.path.dirname(__file__), args.output_dir or default_output)
    os.makedirs(data_cache_dir, exist_ok=True)

    if os.path.isfile(args.tokenizer_json):
        tokenizer = PureByteTokenizer.from_json(args.tokenizer_json)
        print(f"Loaded tokenizer: {args.tokenizer_json}")
    else:
        tokenizer = default_pure_byte_tokenizer()
        print(f"Tokenizer JSON not found at {args.tokenizer_json}; using built-in defaults.")

    assert tokenizer.vocab_size <= 2**16, "token dictionary too large for uint16"
    print(
        f"Tokenizer config: vocab_size={tokenizer.vocab_size} "
        f"bos_id={tokenizer.bos_id} eos_id={tokenizer.eos_id}"
    )

    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for doc_idx, doc in enumerate(fw, start=1):
        if args.num_docs > 0 and doc_idx > args.num_docs:
            break

        token_ids = [tokenizer.bos_id]
        token_ids.extend(tokenizer.encode(doc["text"]))
        if args.append_eos:
            token_ids.append(tokenizer.eos_id)
        tokens = np.array(token_ids, dtype=np.int32)
        assert (0 <= tokens).all() and (tokens < 2**16).all(), "token dictionary too large for uint16"
        tokens = tokens.astype(np.uint16)

        if token_count + len(tokens) < args.shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
            continue

        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(data_cache_dir, f"fineweb_{split}_{shard_index:06d}.bin")
        remainder = args.shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None

        all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
        token_count = len(tokens) - remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(data_cache_dir, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    main()

