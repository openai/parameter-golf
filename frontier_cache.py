from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_HASH_PRIMES = np.array(
    [np.uint64(36313), np.uint64(27191), np.uint64(51647), np.uint64(81929), np.uint64(131071)],
    dtype=np.uint64,
)
_DEFAULT_ORDER_ENTROPY_CENTERS = {7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5}
_DEFAULT_ORDER_ENTROPY_CENTERS_TEXT = "7:3.0,6:3.2,5:3.5,4:3.8,3:4.2,2:4.5"


def _parse_order_entropy_centers(raw: str) -> dict[int, float]:
    centers: dict[int, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "CAUSAL_CACHE_ORDER_ENTROPY_CENTERS must use order:center pairs, "
                f"got {raw!r}"
            )
        order_text, center_text = item.split(":", 1)
        order = int(order_text.strip())
        if order in centers:
            raise ValueError(f"Duplicate order {order} in CAUSAL_CACHE_ORDER_ENTROPY_CENTERS")
        centers[order] = float(center_text.strip())
    return centers


def format_order_entropy_centers(centers: dict[int, float]) -> str:
    return ",".join(f"{order}:{centers[order]:.1f}" for order in sorted(centers, reverse=True))


@dataclass(frozen=True)
class CausalCacheConfig:
    mode: str
    max_order: int
    alpha: float
    min_count: int
    buckets: int
    mixing: str = "fixed"
    count_smoothing: float = 4.0
    alpha_min: float = 0.10
    alpha_max: float = 0.50
    entropy_center: float = 3.5
    entropy_slope: float = 2.0
    order_entropy_centers: dict[int, float] | None = None

    def validate(self) -> None:
        if self.mode not in {"off", "ngram7", "ppm"}:
            raise ValueError(f"CAUSAL_CACHE_MODE must be one of off/ngram7/ppm, got {self.mode!r}")
        if self.max_order < 2:
            raise ValueError(f"CAUSAL_CACHE_MAX_ORDER must be >= 2, got {self.max_order}")
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError(f"CAUSAL_CACHE_ALPHA must be in [0, 1], got {self.alpha}")
        if self.alpha_min < 0.0 or self.alpha_min > 1.0:
            raise ValueError(f"CAUSAL_CACHE_ALPHA_MIN must be in [0, 1], got {self.alpha_min}")
        if self.alpha_max < 0.0 or self.alpha_max > 1.0:
            raise ValueError(f"CAUSAL_CACHE_ALPHA_MAX must be in [0, 1], got {self.alpha_max}")
        if self.alpha_min > self.alpha_max:
            raise ValueError(
                f"CAUSAL_CACHE_ALPHA_MIN must be <= CAUSAL_CACHE_ALPHA_MAX, got {self.alpha_min} > {self.alpha_max}"
            )
        if self.min_count < 1:
            raise ValueError(f"CAUSAL_CACHE_MIN_COUNT must be >= 1, got {self.min_count}")
        if self.buckets <= 0 or self.buckets & (self.buckets - 1):
            raise ValueError(f"CAUSAL_CACHE_BUCKETS must be a positive power of two, got {self.buckets}")
        if self.mixing not in {"fixed", "count", "entropy", "order_entropy"}:
            raise ValueError(
                f"CAUSAL_CACHE_MIXING must be fixed, count, entropy, or order_entropy, got {self.mixing!r}"
            )
        if self.count_smoothing <= 0.0:
            raise ValueError(f"CAUSAL_CACHE_COUNT_SMOOTHING must be > 0, got {self.count_smoothing}")
        if self.entropy_slope <= 0.0:
            raise ValueError(f"CAUSAL_CACHE_ENTROPY_SLOPE must be > 0, got {self.entropy_slope}")
        if self.uses_entropy:
            centers = self.order_entropy_centers or {}
            for order, center in centers.items():
                if order < 2:
                    raise ValueError(f"CAUSAL_CACHE_ORDER_ENTROPY_CENTERS order must be >= 2, got {order}")

    @property
    def orders(self) -> list[int]:
        if self.mode == "off":
            return []
        if self.mode == "ngram7":
            return [self.max_order]
        return list(range(2, self.max_order + 1))

    @property
    def uses_entropy(self) -> bool:
        return self.mixing in {"entropy", "order_entropy"}


class ScoreFirstCausalCache:
    """Deterministic backward-looking cache with an explicit score-then-commit API.

    Call `reset()` whenever the evaluator wants to enforce a document boundary.
    """

    def __init__(self, config: CausalCacheConfig):
        config.validate()
        self.config = config
        self.mask = np.uint64(config.buckets - 1)
        self.ctx_tables = {order: np.zeros((config.buckets,), dtype=np.uint32) for order in config.orders}
        self.full_tables = {order: np.zeros((config.buckets,), dtype=np.uint32) for order in config.orders}
        self._pending_positions: np.ndarray | None = None

    def reset(self) -> None:
        for table in self.ctx_tables.values():
            table.fill(0)
        for table in self.full_tables.values():
            table.fill(0)
        self._pending_positions = None

    def score_segment(
        self,
        token_stream: np.ndarray,
        global_target_positions: np.ndarray,
        model_target_probs: np.ndarray,
        model_entropies: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.config.mode == "off":
            return model_target_probs
        if self._pending_positions is not None:
            raise RuntimeError("cache score/commit ordering violated: commit the pending segment before scoring again")
        if self.config.uses_entropy:
            if model_entropies is None:
                raise RuntimeError("entropy-gated cache mixing requires model entropies during score_segment")
            if len(model_entropies) != len(model_target_probs):
                raise RuntimeError("entropy-gated cache mixing received mismatched entropy and probability shapes")

        mixed = np.array(model_target_probs, copy=True, dtype=np.float64)
        best_ng = np.zeros_like(mixed)
        best_ctx_count = np.zeros_like(mixed)
        best_order = np.zeros_like(global_target_positions, dtype=np.int16)
        matched = np.zeros_like(mixed, dtype=bool)

        for order in reversed(self.config.orders):
            ctx_width = order - 1
            valid = (global_target_positions >= ctx_width) & ~matched
            if not valid.any():
                continue
            idx = np.nonzero(valid)[0]
            positions = global_target_positions[idx]
            ctx_hash = np.zeros(len(positions), dtype=np.uint64)
            for offset in range(ctx_width):
                tok = token_stream[positions - (ctx_width - offset)].astype(np.uint64)
                ctx_hash ^= tok * _HASH_PRIMES[offset % len(_HASH_PRIMES)]
            ctx_key = (ctx_hash & self.mask).astype(np.int64)
            tgt = token_stream[positions].astype(np.uint64)
            full_key = ((ctx_hash ^ (tgt * _HASH_PRIMES[ctx_width % len(_HASH_PRIMES)])) & self.mask).astype(np.int64)

            ctx_counts = self.ctx_tables[order][ctx_key].astype(np.float64)
            full_counts = self.full_tables[order][full_key].astype(np.float64)
            can_mix = ctx_counts >= float(self.config.min_count)
            if not can_mix.any():
                continue
            chosen = idx[can_mix]
            p_ng = np.minimum(full_counts[can_mix], ctx_counts[can_mix]) / np.maximum(ctx_counts[can_mix], 1.0)
            best_ng[chosen] = np.clip(p_ng, 0.0, 1.0)
            best_ctx_count[chosen] = ctx_counts[can_mix]
            best_order[chosen] = order
            matched[chosen] = True

        mix_idx = np.nonzero(matched)[0]
        if mix_idx.size:
            if self.config.mixing == "count":
                alpha_vec = self.config.alpha * (
                    best_ctx_count[mix_idx] / (best_ctx_count[mix_idx] + self.config.count_smoothing)
                )
            elif self.config.mixing == "entropy":
                centers = np.full(mix_idx.size, self.config.entropy_center, dtype=np.float64)
                alpha_vec = self._entropy_alpha(np.asarray(model_entropies[mix_idx], dtype=np.float64), centers)
            elif self.config.mixing == "order_entropy":
                centers = np.array(
                    [
                        (self.config.order_entropy_centers or {}).get(int(order), self.config.entropy_center)
                        for order in best_order[mix_idx]
                    ],
                    dtype=np.float64,
                )
                alpha_vec = self._entropy_alpha(np.asarray(model_entropies[mix_idx], dtype=np.float64), centers)
            else:
                alpha_vec = np.full(mix_idx.size, self.config.alpha, dtype=np.float64)
            mixed[mix_idx] = (1.0 - alpha_vec) * mixed[mix_idx] + alpha_vec * best_ng[mix_idx]

        self._pending_positions = np.array(global_target_positions, copy=True, dtype=np.int64)
        return mixed

    def commit_segment(self, token_stream: np.ndarray, global_target_positions: np.ndarray) -> None:
        if self.config.mode == "off":
            return
        if self._pending_positions is None:
            raise RuntimeError("cache score/commit ordering violated: cannot commit before score_segment")
        if not np.array_equal(self._pending_positions, np.asarray(global_target_positions, dtype=np.int64)):
            raise RuntimeError("cache score/commit ordering violated: commit_segment received a different segment")

        for order in self.config.orders:
            ctx_width = order - 1
            valid = global_target_positions >= ctx_width
            if not valid.any():
                continue
            positions = global_target_positions[valid]
            ctx_hash = np.zeros(len(positions), dtype=np.uint64)
            for offset in range(ctx_width):
                tok = token_stream[positions - (ctx_width - offset)].astype(np.uint64)
                ctx_hash ^= tok * _HASH_PRIMES[offset % len(_HASH_PRIMES)]
            ctx_key = (ctx_hash & self.mask).astype(np.int64)
            tgt = token_stream[positions].astype(np.uint64)
            full_key = ((ctx_hash ^ (tgt * _HASH_PRIMES[ctx_width % len(_HASH_PRIMES)])) & self.mask).astype(np.int64)
            np.add.at(self.ctx_tables[order], ctx_key, 1)
            np.add.at(self.full_tables[order], full_key, 1)

        self._pending_positions = None

    def _entropy_alpha(self, entropies: np.ndarray, centers: np.ndarray) -> np.ndarray:
        scaled = self.config.entropy_slope * (entropies - centers)
        gate = 1.0 / (1.0 + np.exp(-scaled))
        return self.config.alpha_min + (self.config.alpha_max - self.config.alpha_min) * gate


def causal_cache_config_from_env(env: dict[str, str] | None = None) -> CausalCacheConfig:
    source = env if env is not None else {}
    return CausalCacheConfig(
        mode=source.get("CAUSAL_CACHE_MODE", "off"),
        max_order=int(source.get("CAUSAL_CACHE_MAX_ORDER", "7")),
        alpha=float(source.get("CAUSAL_CACHE_ALPHA", "0.40")),
        min_count=int(source.get("CAUSAL_CACHE_MIN_COUNT", "2")),
        buckets=int(source.get("CAUSAL_CACHE_BUCKETS", "4194304")),
        mixing=source.get("CAUSAL_CACHE_MIXING", "fixed"),
        count_smoothing=float(source.get("CAUSAL_CACHE_COUNT_SMOOTHING", "4.0")),
        alpha_min=float(source.get("CAUSAL_CACHE_ALPHA_MIN", "0.10")),
        alpha_max=float(source.get("CAUSAL_CACHE_ALPHA_MAX", "0.50")),
        entropy_center=float(source.get("CAUSAL_CACHE_ENTROPY_CENTER", "3.5")),
        entropy_slope=float(source.get("CAUSAL_CACHE_ENTROPY_SLOPE", "2.0")),
        order_entropy_centers=_parse_order_entropy_centers(
            source.get("CAUSAL_CACHE_ORDER_ENTROPY_CENTERS", _DEFAULT_ORDER_ENTROPY_CENTERS_TEXT)
        ),
    )


def causal_cache_from_env(env: dict[str, str] | None = None) -> ScoreFirstCausalCache | None:
    config = causal_cache_config_from_env(env)
    if config.mode == "off":
        return None
    return ScoreFirstCausalCache(config)
