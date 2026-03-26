from __future__ import annotations

import argparse
import gzip
import json
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4


@dataclass
class DictionaryStats:
    tokens_seen: int
    bigram_types: int
    trigram_types: int
    top_bigram_coverage: int
    top_trigram_coverage: int


def load_data_shard(path: Path) -> list[int]:
    with path.open("rb") as f:
        header = struct.unpack("<256i", f.read(HEADER_BYTES))
        if header[0] != SHARD_MAGIC or header[1] != SHARD_VERSION:
            raise ValueError(f"Unexpected shard header for {path}")
        num_tokens = int(header[2])
        payload = f.read()
    expected = num_tokens * 2
    if len(payload) != expected:
        raise ValueError(f"Shard size mismatch for {path}: expected {expected} token bytes, got {len(payload)}")
    return list(struct.unpack(f"<{num_tokens}H", payload))


def iter_tokens(files: Iterable[Path], limit_tokens: int | None = None) -> Iterator[int]:
    seen = 0
    for path in files:
        for token in load_data_shard(path):
            yield token
            seen += 1
            if limit_tokens is not None and seen >= limit_tokens:
                return


def count_bigrams_trigrams(files: Iterable[Path], limit_tokens: int | None = None) -> tuple[Counter[tuple[int, int]], Counter[tuple[int, int, int]], int]:
    bigrams: Counter[tuple[int, int]] = Counter()
    trigrams: Counter[tuple[int, int, int]] = Counter()
    prev1: int | None = None
    prev2: int | None = None
    total = 0
    for token in iter_tokens(files, limit_tokens=limit_tokens):
        total += 1
        if prev1 is not None:
            bigrams[(prev1, token)] += 1
        if prev2 is not None and prev1 is not None:
            trigrams[(prev2, prev1, token)] += 1
        prev2, prev1 = prev1, token
    return bigrams, trigrams, total


def top_k(counter: Counter, k: int) -> list[tuple[tuple[int, ...], int]]:
    return counter.most_common(k)


def build_dictionary(files: Iterable[Path], top_bigrams: int = 10_000, top_trigrams: int = 50_000, limit_tokens: int | None = None) -> dict[str, object]:
    bigrams, trigrams, total_tokens = count_bigrams_trigrams(files, limit_tokens=limit_tokens)
    top_bigram_items = top_k(bigrams, top_bigrams)
    top_trigram_items = top_k(trigrams, top_trigrams)
    stats = DictionaryStats(
        tokens_seen=total_tokens,
        bigram_types=len(bigrams),
        trigram_types=len(trigrams),
        top_bigram_coverage=sum(count for _, count in top_bigram_items),
        top_trigram_coverage=sum(count for _, count in top_trigram_items),
    )
    return {
        "stats": asdict(stats),
        "top_bigrams": [{"tokens": list(key), "count": count} for key, count in top_bigram_items],
        "top_trigrams": [{"tokens": list(key), "count": count} for key, count in top_trigram_items],
    }


def save_dictionary(obj: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def verify_dictionary(obj: dict[str, object]) -> None:
    stats = obj["stats"]
    if stats["tokens_seen"] <= 0:
        raise AssertionError("No tokens were processed")
    if not obj["top_bigrams"]:
        raise AssertionError("No bigrams collected")
    if not obj["top_trigrams"]:
        raise AssertionError("No trigrams collected")
    top_bigram = obj["top_bigrams"][0]
    if len(top_bigram["tokens"]) != 2 or top_bigram["count"] <= 0:
        raise AssertionError("Top bigram entry malformed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build top-k bigram/trigram dictionary for Model 1 Codec.")
    p.add_argument("--input-glob", default="data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
    p.add_argument("--output", default="builds/artifacts/model1_dictionary.json.gz")
    p.add_argument("--top-bigrams", type=int, default=10_000)
    p.add_argument("--top-trigrams", type=int, default=50_000)
    p.add_argument("--limit-tokens", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(Path().glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No shards matched {args.input_glob}")
    obj = build_dictionary(
        files,
        top_bigrams=args.top_bigrams,
        top_trigrams=args.top_trigrams,
        limit_tokens=args.limit_tokens,
    )
    verify_dictionary(obj)
    save_dictionary(obj, Path(args.output))
    print(
        json.dumps(
            {
                "output": args.output,
                "tokens_seen": obj["stats"]["tokens_seen"],
                "top_bigram_count": len(obj["top_bigrams"]),
                "top_trigram_count": len(obj["top_trigrams"]),
            }
        )
    )


if __name__ == "__main__":
    main()
