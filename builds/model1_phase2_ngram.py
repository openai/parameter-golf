from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_BYTES = 256 * 4

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def load_data_shard(path: Path) -> list[int]:
    with path.open("rb") as f:
        header = struct.unpack("<256i", f.read(HEADER_BYTES))
        if header[0] != SHARD_MAGIC or header[1] != SHARD_VERSION:
            raise ValueError(f"Unexpected shard header for {path}")
        num_tokens = int(header[2])
        payload = f.read()
    if len(payload) != num_tokens * 2:
        raise ValueError(f"Shard size mismatch for {path}")
    return list(struct.unpack(f"<{num_tokens}H", payload))


def iter_tokens(files: Iterable[Path], limit_tokens: int | None = None) -> Iterator[int]:
    seen = 0
    for path in files:
        for token in load_data_shard(path):
            yield token
            seen += 1
            if limit_tokens is not None and seen >= limit_tokens:
                return


@dataclass
class DistributionResult:
    probs: list[float]
    order_used: int


class KneserNeyNGram:
    def __init__(self, vocab_size: int, max_order: int = 7, discount: float = 0.75):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.discount = discount
        self.ngram_counts: list[Counter[tuple[int, ...]]] = [Counter() for _ in range(max_order)]
        self.context_totals: list[Counter[tuple[int, ...]]] = [Counter() for _ in range(max_order - 1)]
        self.successor_types: list[defaultdict[tuple[int, ...], set[int]]] = [defaultdict(set) for _ in range(max_order - 1)]
        self.continuation_types: Counter[int] = Counter()
        self.total_bigram_types = 0

    def fit(self, files: Iterable[Path], limit_tokens: int | None = None) -> None:
        self.fit_token_sequence(iter_tokens(files, limit_tokens=limit_tokens))

    def fit_token_sequence(self, tokens: Iterable[int]) -> None:
        history: list[int] = []
        prev_token: int | None = None
        seen_bigrams: set[tuple[int, int]] = set()
        for token in tokens:
            history.append(token)
            if len(history) > self.max_order:
                history.pop(0)
            for order in range(1, min(len(history), self.max_order) + 1):
                ngram = tuple(history[-order:])
                self.ngram_counts[order - 1][ngram] += 1
                if order > 1:
                    context = ngram[:-1]
                    self.context_totals[order - 2][context] += 1
                    self.successor_types[order - 2][context].add(ngram[-1])
            if prev_token is not None:
                pair = (prev_token, token)
                if pair not in seen_bigrams:
                    seen_bigrams.add(pair)
                    self.continuation_types[token] += 1
            prev_token = token
        self.total_bigram_types = max(len(seen_bigrams), 1)

    def _unigram_prob(self, token: int) -> float:
        continuation = self.continuation_types.get(token, 0)
        if continuation > 0:
            return continuation / self.total_bigram_types
        total = sum(self.ngram_counts[0].values())
        return self.ngram_counts[0].get((token,), 0) / max(total, 1)

    def _prob(self, context: tuple[int, ...], token: int, order: int) -> float:
        if order <= 1 or not context:
            return self._unigram_prob(token)
        trimmed_context = context[-(order - 1):]
        ngram = trimmed_context + (token,)
        count_hw = self.ngram_counts[order - 1].get(ngram, 0)
        count_h = self.context_totals[order - 2].get(trimmed_context, 0)
        if count_h == 0:
            return self._prob(trimmed_context[1:], token, order - 1)
        unique_successors = len(self.successor_types[order - 2].get(trimmed_context, ()))
        backoff = self.discount * unique_successors / count_h
        discounted = max(count_hw - self.discount, 0.0) / count_h
        return discounted + backoff * self._prob(trimmed_context[1:], token, order - 1)

    def next_distribution(self, context: Iterable[int]) -> DistributionResult:
        ctx = tuple(context)[-(self.max_order - 1) :]
        order_used = min(len(ctx) + 1, self.max_order)
        probs = [self._prob(ctx, token, order_used) for token in range(self.vocab_size)]
        total = sum(probs)
        if total <= 0:
            probs = [1.0 / self.vocab_size] * self.vocab_size
        else:
            probs = [p / total for p in probs]
        return DistributionResult(probs=probs, order_used=order_used)

    def next_distribution_tensor(self, context: Iterable[int]):
        result = self.next_distribution(context)
        if torch is None:
            raise RuntimeError("torch is not available")
        return torch.tensor(result.probs, dtype=torch.float32)

    def sequence_nll(self, tokens: list[int]) -> float:
        nll = 0.0
        for idx in range(1, len(tokens)):
            ctx_start = max(0, idx - (self.max_order - 1))
            dist = self.next_distribution(tokens[ctx_start:idx]).probs
            nll -= math.log(max(dist[tokens[idx]], 1e-12))
        return nll / max(len(tokens) - 1, 1)


def verify_model() -> dict[str, object]:
    toy = [1, 2, 3, 1, 2, 3, 1, 2, 4]
    model = KneserNeyNGram(vocab_size=8, max_order=4)
    model.fit_token_sequence(toy)
    result = model.next_distribution([1, 2])
    best_token = max(range(len(result.probs)), key=result.probs.__getitem__)
    if best_token != 3:
        raise AssertionError(f"Expected token 3 after context [1,2], got {best_token}")
    if abs(sum(result.probs) - 1.0) > 1e-5:
        raise AssertionError("Distribution does not sum to 1")
    return {"best_token": best_token, "order_used": result.order_used, "toy_nll": model.sequence_nll(toy)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Model 1 Codec n-gram model.")
    p.add_argument("--input-glob", default="data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--max-order", type=int, default=7)
    p.add_argument("--discount", type=float, default=0.75)
    p.add_argument("--limit-tokens", type=int, default=None)
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        print(json.dumps(verify_model()))
        return
    files = sorted(Path().glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No shards matched {args.input_glob}")
    model = KneserNeyNGram(vocab_size=args.vocab_size, max_order=args.max_order, discount=args.discount)
    model.fit(files, limit_tokens=args.limit_tokens)
    probe = model.next_distribution([0] * (args.max_order - 1))
    print(
        json.dumps(
            {
                "files": len(files),
                "vocab_size": args.vocab_size,
                "max_order": args.max_order,
                "order_used": probe.order_used,
                "top5": sorted(enumerate(probe.probs), key=lambda item: item[1], reverse=True)[:5],
            }
        )
    )


if __name__ == "__main__":
    main()
