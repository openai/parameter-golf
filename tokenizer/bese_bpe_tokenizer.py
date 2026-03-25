"""
BESE+BPE Tokenizer: Base-Efficient Subword Encoding with BPE

Two-layer tokenization:
  Layer 1 (BESE): Maps characters to a 38-token structured alphabet using
    frequency-weighted single-token codes and context-aware grouped codes.
  Layer 2 (BPE): Learns common patterns in the BESE token stream and merges
    them into single tokens, reducing sequence length.

This gives BPE a linguistically structured foundation instead of raw bytes,
while keeping the vocabulary small enough to save significant embedding parameters.

Designed for OpenAI Parameter Golf challenge.

Usage:
    # Train merges on text data
    merges = train_bpe_merges(texts, num_merges=250)

    # Create tokenizer
    tok = BESEBPETokenizer(merges=merges)

    # Encode/decode
    tokens = tok.encode("The cat sat on the mat.")
    text = tok.decode(tokens.tolist())

    # Save/load
    tok.save("tokenizer.json")
    tok = BESEBPETokenizer.load("tokenizer.json")

    # Get BPB lookup table for training
    base_bytes_lut, has_leading_space_lut, is_boundary_lut = tok.build_luts_for_training(device)
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from collections import Counter

try:
    from .bese_constants import (
        BASE_VOCAB_SIZE,
        BOS_ID,
        BYTES_PER_TOKEN,
        DECODE_TABLE,
        EOS_ID,
        ENCODE_TABLE,
        GROUP_START,
        GROUPS,
        OTHER_PUNCT_ID,
        PAD_ID,
        SINGLE_LETTERS,
        UNK_ID,
    )
except ImportError:  # script: python tokenizer/bese_bpe_tokenizer.py
    from bese_constants import (
        BASE_VOCAB_SIZE,
        BOS_ID,
        BYTES_PER_TOKEN,
        DECODE_TABLE,
        EOS_ID,
        ENCODE_TABLE,
        GROUP_START,
        GROUPS,
        OTHER_PUNCT_ID,
        PAD_ID,
        SINGLE_LETTERS,
        UNK_ID,
    )


def _text_to_base_tokens(text: str) -> list[int]:
    tokens = []
    for ch in text:
        lower = ch.lower()
        if lower in ENCODE_TABLE:
            utf8_len = len(ch.encode("utf-8"))
            mapped = ENCODE_TABLE[lower]
            mapped_bytes = sum(BYTES_PER_TOKEN[t] for t in mapped)
            if utf8_len == mapped_bytes:
                tokens.extend(mapped)
            else:
                tokens.extend([OTHER_PUNCT_ID] * utf8_len)
        else:
            utf8_len = len(ch.encode("utf-8"))
            tokens.extend([OTHER_PUNCT_ID] * utf8_len)
    return tokens


def _get_pair_counts(sequences):
    counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] += 1
    return counts


def _apply_merge(sequences, pair, new_id):
    result = []
    for seq in sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        result.append(new_seq)
    return result


def train_bpe_merges(texts, num_merges=250, verbose=True):
    """Learn BPE merges from text data on top of BESE alphabet."""
    if verbose:
        print(f"Encoding {len(texts)} texts with base BESE tokenizer...")
    sequences = [_text_to_base_tokens(text) for text in texts]
    total_base = sum(len(s) for s in sequences)
    if verbose:
        print(f"Base tokens: {total_base:,}")
        print(f"Learning {num_merges} BPE merges...")

    merges = []
    next_id = BASE_VOCAB_SIZE

    for merge_num in range(num_merges):
        pairs = _get_pair_counts(sequences)
        if not pairs:
            break
        best_pair, count = pairs.most_common(1)[0]
        if count < 2:
            break
        merges.append((best_pair, next_id))
        sequences = _apply_merge(sequences, best_pair, next_id)
        if verbose and (merge_num < 20 or merge_num % 50 == 0 or merge_num == num_merges - 1):
            total_now = sum(len(s) for s in sequences)
            print(
                f"  Merge {merge_num+1:4d}: ({best_pair[0]:3d},{best_pair[1]:3d}) -> {next_id:4d}"
                f"  count={count:6d}  tokens={total_now:,} ({total_now/total_base:.2%})"
            )
        next_id += 1

    if verbose:
        final = sum(len(s) for s in sequences)
        print(f"\nDone. {total_base:,} -> {final:,} tokens ({final/total_base:.2%})")
        print(
            f"Vocabulary: {BASE_VOCAB_SIZE} base + {len(merges)} merges = "
            f"{BASE_VOCAB_SIZE + len(merges)} total"
        )
    return merges


class BESEBPETokenizer:
    """BESE + BPE tokenizer for parameter-constrained language models."""

    def __init__(self, merges=None):
        self.merges = merges or []
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        self._merge_map = {pair: new_id for pair, new_id in self.merges}
        self._bpt = self._build_bpt()
        self._decode_chains = {new_id: pair for pair, new_id in self.merges}

    @property
    def vocab_size(self):
        return BASE_VOCAB_SIZE + len(self.merges)

    def _build_bpt(self):
        bpt = np.zeros(self.vocab_size, dtype=np.int16)
        bpt[:BASE_VOCAB_SIZE] = BYTES_PER_TOKEN
        merge_bpt = {i: int(BYTES_PER_TOKEN[i]) for i in range(BASE_VOCAB_SIZE)}
        for pair, new_id in self.merges:
            merge_bpt[new_id] = merge_bpt[pair[0]] + merge_bpt[pair[1]]
            bpt[new_id] = merge_bpt[new_id]
        return bpt

    def _apply_merges(self, tokens):
        for pair, new_id in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text):
        base_tokens = _text_to_base_tokens(text)
        return np.array(
            self._apply_merges(base_tokens) if self.merges else base_tokens,
            dtype=np.uint16,
        )

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode_token_to_base(self, token_id):
        if token_id < BASE_VOCAB_SIZE:
            return [token_id]
        if token_id in self._decode_chains:
            left, right = self._decode_chains[token_id]
            return self.decode_token_to_base(left) + self.decode_token_to_base(right)
        return [UNK_ID]

    def decode(self, token_ids):
        base_tokens = []
        for tid in token_ids:
            base_tokens.extend(self.decode_token_to_base(tid))
        result = []
        i = 0
        while i < len(base_tokens):
            tid = base_tokens[i]
            if GROUP_START <= tid < GROUP_START + len(GROUPS):
                if i + 1 < len(base_tokens):
                    key = (tid, base_tokens[i + 1])
                    result.append(DECODE_TABLE.get(key, "?"))
                    i += 2
                    continue
            key = (tid,)
            if key in DECODE_TABLE:
                result.append(DECODE_TABLE[key])
            elif tid not in (PAD_ID, BOS_ID, EOS_ID, UNK_ID):
                result.append("?")
            i += 1
        return "".join(result)

    def get_bytes_per_token_lut(self):
        return self._bpt.copy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_type": "bese_bpe",
            "version": 1,
            "base_vocab_size": BASE_VOCAB_SIZE,
            "num_merges": len(self.merges),
            "vocab_size": self.vocab_size,
            "single_letters": SINGLE_LETTERS,
            "groups": GROUPS,
            "merges": [[list(pair), new_id] for pair, new_id in self.merges],
        }
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        merges = [(tuple(pair), new_id) for pair, new_id in payload["merges"]]
        return cls(merges=merges)

    def build_luts_for_training(self, device=None):
        """Build lookup tables compatible with train_gpt.py eval_val function."""
        import torch

        vs = self.vocab_size
        has_leading_space = np.zeros(vs, dtype=np.bool_)
        is_boundary = np.zeros(vs, dtype=np.bool_)
        is_boundary[PAD_ID] = True
        is_boundary[BOS_ID] = True
        is_boundary[EOS_ID] = True
        is_boundary[UNK_ID] = True
        kwargs = {"device": device} if device is not None else {}
        return (
            torch.tensor(self._bpt.copy(), dtype=torch.int16, **kwargs),
            torch.tensor(has_leading_space, dtype=torch.bool, **kwargs),
            torch.tensor(is_boundary, dtype=torch.bool, **kwargs),
        )


def build_bese_bpe_tokenizer(*, spec, docs_jsonl, tokenizers_dir):
    """Build BESE+BPE tokenizer for the parameter-golf data pipeline."""
    import itertools

    num_merges = int(spec.get("num_merges", 250))
    max_train_docs = spec.get("tokenizer_train_docs", 100000)

    def iter_docs(path):
        import json as jm

        with open(path, encoding="utf-8") as f:
            for line in f:
                yield jm.loads(line)["text"]

    train_texts = list(itertools.islice(iter_docs(docs_jsonl), max_train_docs))
    merges = train_bpe_merges(train_texts, num_merges=num_merges, verbose=True)
    tok = BESEBPETokenizer(merges=merges)
    save_path = tokenizers_dir / spec.get("filename", "bese_bpe.json")
    tok.save(save_path)

    return {
        "name": spec.get("name", f"bese_bpe_{num_merges}"),
        "kind": "bese_bpe",
        "dataset_suffix": spec.get("dataset_suffix", f"bese{num_merges}"),
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "encode": tok.encode,
        "encode_batch": tok.encode_batch,
        "manifest": {"path": str(save_path), "num_merges": len(merges)},
    }


if __name__ == "__main__":
    sample = [
        "The cat sat on the mat.",
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
    ] * 200

    merges = train_bpe_merges(sample, num_merges=100, verbose=True)
    tok = BESEBPETokenizer(merges=merges)

    for text in ["The cat sat on the mat.", "Hello world!", "Testing 123."]:
        enc = tok.encode(text)
        dec = tok.decode(enc.tolist())
        bpt = tok.get_bytes_per_token_lut()
        tb = sum(bpt[t] for t in enc)
        ub = len(text.encode("utf-8"))
        print(
            f'"{text}" -> {len(enc)} tokens, bytes {tb}/{ub} '
            f'{"OK" if tb == ub else "FAIL"}, decoded: "{dec}"'
        )
