"""
FineWeb preprocessing for train_gpt.py.

This script writes shards tokenized with a SentencePiece model.
"""

import argparse
import os

import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm


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
    parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
    parser.add_argument("-v", "--version", type=str, default="10B", choices=["10B", "100B"])
    parser.add_argument("-s", "--shard_size", type=int, default=10**8)
    parser.add_argument(
        "--num_docs",
        type=int,
        default=0,
        help="Maximum number of documents to tokenize (0 means all docs)",
    )
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "tokenizers", "fineweb_4k_bpe.model"),
        help="Path to a SentencePiece model (expected vocab size 4096)",
    )
    args = parser.parse_args()

    assert os.path.isfile(args.tokenizer_model), f"Missing tokenizer model: {args.tokenizer_model}"

    if args.version == "10B":
        local_dir = "fineweb10B_sp4k"
        remote_name = "sample-10BT"
    else:
        local_dir = "fineweb100B_sp4k"
        remote_name = "sample-100BT"

    data_cache_dir = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(data_cache_dir, exist_ok=True)

    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_model)
    bos_id = tokenizer.bos_id()
    assert bos_id >= 0, "SentencePiece model must define bos_id"
    assert tokenizer.vocab_size() <= 2**16, "token dictionary too large for uint16"

    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for doc_idx, doc in enumerate(fw, start=1):
        if args.num_docs > 0 and doc_idx > args.num_docs:
            break

        token_ids = [bos_id]
        token_ids.extend(tokenizer.encode(doc["text"], out_type=int))
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
