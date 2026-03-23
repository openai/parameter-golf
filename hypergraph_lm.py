"""
hypergraph_lm.py — Hypergraph Pattern Store for Parameter Golf

Multi-level pattern extractor using Cantor-recursive emergence theory.
Replaces/extends BigramHash with a principled, binding-energy-weighted
pattern hierarchy:

  Ω₁: Bigram patterns    (token pairs → conditional distributions)
  Ω₂: Trigram patterns    (token triples → conditional distributions)
  Ω₃: 5-gram patterns     (5-token contexts → conditional distributions)

Each pattern's binding energy B(C) determines:
  1. Whether it's stored (B > threshold → keep, else drop)
  2. How many bits it gets in the 16MB budget
  3. Its interpolation weight at prediction time

At inference:
  P(next|context) = λ₃·P_Ω₃ + λ₂·P_Ω₂ + λ₁·P_Ω₁ + (1-λ₁-λ₂-λ₃)·P_neural

where λᵢ ∝ B(matched_pattern_at_level_i).
"""

import math
import struct
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import io
import zlib


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class PatternEntry:
    """A single pattern in the hypergraph store."""
    pattern: tuple           # token id tuple (context)
    next_dist: Dict[int, float]  # token_id → probability
    count: int               # total occurrences
    binding: float           # B(C) for this pattern's context cluster
    level: int               # Cantor level (1=bigram, 2=trigram, 3=5gram)


@dataclass
class LevelStore:
    """All patterns at one Cantor level."""
    level: int
    context_len: int         # number of context tokens (1 for bigram, 2 for trigram, etc.)
    patterns: Dict[tuple, PatternEntry] = field(default_factory=dict)
    total_binding: float = 0.0
    budget_bytes: int = 0

    def size_estimate(self) -> int:
        """Estimate serialized size in bytes."""
        total = 0
        for entry in self.patterns.values():
            # pattern keys + top-k distribution + metadata
            total += self.context_len * 2  # uint16 per context token
            total += len(entry.next_dist) * 4  # uint16 token + uint16 scaled prob
            total += 8  # binding float + count
        return total


class HypergraphPatternStore:
    """
    Multi-level pattern store built from token streams.

    The binding energy for a pattern context C is:

        B(C) = (1/|pairs|) Σ_{i<j} W(tᵢ, tⱼ)

    where W(tᵢ, tⱼ) = σ(tᵢ)·σ(tⱼ) is specificity-weighted co-occurrence,
    and σ(t) = 1/freq(t) is inverse frequency (rare tokens bind tighter).

    This is the same binding formula from the epistemic hypergraph, applied
    to token-level patterns rather than propositions.
    """

    def __init__(self, vocab_size: int = 1024, max_budget_bytes: int = 6_000_000):
        self.vocab_size = vocab_size
        self.max_budget_bytes = max_budget_bytes

        # Token frequency for specificity computation
        self.token_freq: np.ndarray = np.zeros(vocab_size, dtype=np.float64)
        self.total_tokens: int = 0

        # Pattern counters (built during scan phase)
        self._bigram_counts: Dict[int, Counter] = defaultdict(Counter)   # prev → {next: count}
        self._trigram_counts: Dict[tuple, Counter] = defaultdict(Counter) # (t-2,t-1) → {next: count}
        self._fivegram_counts: Dict[tuple, Counter] = defaultdict(Counter) # (t-4..t-1) → {next: count}

        # Total context counts for normalization
        self._bigram_totals: Counter = Counter()
        self._trigram_totals: Counter = Counter()
        self._fivegram_totals: Counter = Counter()

        # Finalized stores (after build phase)
        self.levels: Dict[int, LevelStore] = {}
        self._built = False

    # -------------------------------------------------------------------
    # Phase 1: Scan token stream
    # -------------------------------------------------------------------

    def scan_tokens(self, tokens: np.ndarray):
        """
        Scan a token array to accumulate pattern counts and frequencies.
        Call this on each training shard.

        Args:
            tokens: 1D uint16 array of token ids
        """
        n = len(tokens)
        if n < 2:
            return

        # Token frequencies
        for t in range(self.vocab_size):
            self.token_freq[t] += np.sum(tokens == t)
        self.total_tokens += n

        # Bigrams: tokens[i] → tokens[i+1]
        for i in range(n - 1):
            prev = int(tokens[i])
            nxt = int(tokens[i + 1])
            self._bigram_counts[prev][nxt] += 1
            self._bigram_totals[prev] += 1

        # Trigrams: (tokens[i], tokens[i+1]) → tokens[i+2]
        for i in range(n - 2):
            ctx = (int(tokens[i]), int(tokens[i + 1]))
            nxt = int(tokens[i + 2])
            self._trigram_counts[ctx][nxt] += 1
            self._trigram_totals[ctx] += 1

        # 5-grams: (tokens[i..i+3]) → tokens[i+4]
        for i in range(n - 4):
            ctx = (int(tokens[i]), int(tokens[i + 1]),
                   int(tokens[i + 2]), int(tokens[i + 3]))
            nxt = int(tokens[i + 4])
            self._fivegram_counts[ctx][nxt] += 1
            self._fivegram_totals[ctx] += 1

    def scan_tokens_fast(self, tokens: np.ndarray):
        """
        Optimized scan using numpy operations for bigrams.
        Falls back to loop for trigrams/5-grams but only on frequent patterns.

        Args:
            tokens: 1D uint16 array of token ids
        """
        n = len(tokens)
        if n < 2:
            return

        # Token frequencies — vectorized
        counts = np.bincount(tokens.astype(np.int32), minlength=self.vocab_size)
        self.token_freq[:len(counts)] += counts[:self.vocab_size]
        self.total_tokens += n

        # Bigrams — vectorized pair counting
        prev_tokens = tokens[:-1].astype(np.int32)
        next_tokens = tokens[1:].astype(np.int32)
        # Pack pairs into single int for fast counting
        pair_keys = prev_tokens * self.vocab_size + next_tokens
        pair_counts = Counter(pair_keys.tolist())
        for key, count in pair_counts.items():
            prev = key // self.vocab_size
            nxt = key % self.vocab_size
            self._bigram_counts[prev][nxt] += count
            self._bigram_totals[prev] += count

        # Trigrams — vectorized triple counting
        if n >= 3:
            t0 = tokens[:-2].astype(np.int64)
            t1 = tokens[1:-1].astype(np.int64)
            t2 = tokens[2:].astype(np.int64)
            tri_keys = t0 * (self.vocab_size ** 2) + t1 * self.vocab_size + t2
            tri_counts = Counter(tri_keys.tolist())
            for key, count in tri_counts.items():
                t2_val = key % self.vocab_size
                remainder = key // self.vocab_size
                t1_val = remainder % self.vocab_size
                t0_val = remainder // self.vocab_size
                ctx = (int(t0_val), int(t1_val))
                self._trigram_counts[ctx][int(t2_val)] += count
                self._trigram_totals[ctx] += count

        # 5-grams — only count if enough tokens, sample if too large
        if n >= 5:
            # For 5-grams, use loop but it's fine — we'll prune aggressively
            step = max(1, n // 2_000_000)  # subsample for very large shards
            for i in range(0, n - 4, step):
                ctx = (int(tokens[i]), int(tokens[i + 1]),
                       int(tokens[i + 2]), int(tokens[i + 3]))
                nxt = int(tokens[i + 4])
                self._fivegram_counts[ctx][nxt] += step  # scale by step
                self._fivegram_totals[ctx] += step

    # -------------------------------------------------------------------
    # Binding energy computation
    # -------------------------------------------------------------------

    def specificity(self, token_id: int) -> float:
        """σ(t) = 1/freq(t) — rare tokens have high specificity."""
        freq = self.token_freq[token_id]
        if freq <= 0:
            return 0.0
        return 1.0 / freq

    def binding_energy_bigram(self, prev_token: int) -> float:
        """
        B for a bigram context: just σ(prev) weighted by distribution entropy.
        Low entropy (predictable next token) = high binding.
        """
        sigma = self.specificity(prev_token)
        total = self._bigram_totals[prev_token]
        if total == 0:
            return 0.0

        # Entropy of next-token distribution
        dist = self._bigram_counts[prev_token]
        entropy = 0.0
        for count in dist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Max entropy = log2(vocab_size) ≈ 10 for vocab 1024
        max_entropy = math.log2(self.vocab_size)

        # Binding = specificity × (1 - normalized_entropy)
        # High binding = rare token + predictable next token
        binding = sigma * total * (1.0 - entropy / max_entropy)
        return binding

    def binding_energy_ngram(self, context: tuple) -> float:
        """
        B(C) for an n-gram context.
        Uses the full binding formula: average pairwise specificity-weighted
        co-occurrence across context tokens, modulated by prediction certainty.
        """
        n = len(context)
        if n < 1:
            return 0.0

        # Pairwise specificity binding (entity overlap analog)
        pairwise_sum = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                si = self.specificity(context[i])
                sj = self.specificity(context[j])
                pairwise_sum += si * sj
                n_pairs += 1

        avg_pairwise = pairwise_sum / max(1, n_pairs)

        # Prediction certainty (low entropy = high binding)
        if n == 2:
            counts = self._trigram_counts.get(context, {})
            total = self._trigram_totals.get(context, 0)
        elif n == 4:
            counts = self._fivegram_counts.get(context, {})
            total = self._fivegram_totals.get(context, 0)
        else:
            return avg_pairwise

        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(self.vocab_size)
        certainty = 1.0 - entropy / max_entropy

        # Final binding = structural coherence × prediction power × evidence mass
        return avg_pairwise * certainty * math.log1p(total)

    # -------------------------------------------------------------------
    # Phase 2: Build finalized stores
    # -------------------------------------------------------------------

    def build(self,
              bigram_budget: int = 2_000_000,
              trigram_budget: int = 2_500_000,
              fivegram_budget: int = 1_500_000,
              min_count: int = 5,
              top_k_next: int = 32):
        """
        Finalize the pattern stores by:
        1. Computing binding energy for each pattern
        2. Selecting top patterns by binding (within budget)
        3. Storing sparse conditional distributions (top-k)

        Args:
            bigram_budget: bytes for level 1
            trigram_budget: bytes for level 2
            fivegram_budget: bytes for level 3
            min_count: minimum occurrence count to consider
            top_k_next: max next-tokens to store per pattern
        """
        # --- Level 1: Bigrams ---
        level1 = LevelStore(level=1, context_len=1, budget_bytes=bigram_budget)
        bigram_entries = []
        for prev, dist in self._bigram_counts.items():
            total = self._bigram_totals[prev]
            if total < min_count:
                continue
            binding = self.binding_energy_bigram(prev)
            if binding <= 0:
                continue
            # Top-k next tokens
            top_next = dist.most_common(top_k_next)
            next_dist = {tok: count / total for tok, count in top_next}
            entry = PatternEntry(
                pattern=(prev,),
                next_dist=next_dist,
                count=total,
                binding=binding,
                level=1,
            )
            bigram_entries.append(entry)

        # Sort by binding, fill budget
        bigram_entries.sort(key=lambda e: -e.binding)
        self._fill_level(level1, bigram_entries, bigram_budget)
        self.levels[1] = level1

        # --- Level 2: Trigrams ---
        level2 = LevelStore(level=2, context_len=2, budget_bytes=trigram_budget)
        trigram_entries = []
        for ctx, dist in self._trigram_counts.items():
            total = self._trigram_totals[ctx]
            if total < min_count:
                continue
            binding = self.binding_energy_ngram(ctx)
            if binding <= 0:
                continue
            top_next = dist.most_common(top_k_next)
            next_dist = {tok: count / total for tok, count in top_next}
            entry = PatternEntry(
                pattern=ctx,
                next_dist=next_dist,
                count=total,
                binding=binding,
                level=2,
            )
            trigram_entries.append(entry)

        trigram_entries.sort(key=lambda e: -e.binding)
        self._fill_level(level2, trigram_entries, trigram_budget)
        self.levels[2] = level2

        # --- Level 3: 5-grams ---
        level3 = LevelStore(level=3, context_len=4, budget_bytes=fivegram_budget)
        fivegram_entries = []
        for ctx, dist in self._fivegram_counts.items():
            total = self._fivegram_totals[ctx]
            if total < min_count:
                continue
            binding = self.binding_energy_ngram(ctx)
            if binding <= 0:
                continue
            top_next = dist.most_common(top_k_next)
            next_dist = {tok: count / total for tok, count in top_next}
            entry = PatternEntry(
                pattern=ctx,
                next_dist=next_dist,
                count=total,
                binding=binding,
                level=3,
            )
            fivegram_entries.append(entry)

        fivegram_entries.sort(key=lambda e: -e.binding)
        self._fill_level(level3, fivegram_entries, fivegram_budget)
        self.levels[3] = level3

        # Free raw counters
        self._bigram_counts.clear()
        self._trigram_counts.clear()
        self._fivegram_counts.clear()
        self._bigram_totals.clear()
        self._trigram_totals.clear()
        self._fivegram_totals.clear()

        self._built = True

    def _fill_level(self, store: LevelStore, entries: list, budget: int):
        """Add entries to store until budget is exhausted."""
        used = 0
        for entry in entries:
            # Estimate entry size: context tokens + distribution + metadata
            entry_size = store.context_len * 2 + len(entry.next_dist) * 4 + 8
            if used + entry_size > budget:
                break
            store.patterns[entry.pattern] = entry
            store.total_binding += entry.binding
            used += entry_size
        return used

    # -------------------------------------------------------------------
    # Phase 3: Prediction
    # -------------------------------------------------------------------

    def predict(self, context: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Given context tokens, produce a probability distribution over next token
        using multi-level pattern matching with binding-weighted interpolation.

        Returns:
            (distribution, confidence):
                distribution: np.ndarray of shape (vocab_size,) or None if no match
                confidence: total binding confidence (higher = more trustworthy)
        """
        if not self._built:
            return None, 0.0

        result = np.zeros(self.vocab_size, dtype=np.float64)
        total_weight = 0.0

        # Level 3: 5-gram (highest priority)
        if len(context) >= 4:
            ctx = tuple(int(x) for x in context[-4:])
            entry = self.levels[3].patterns.get(ctx)
            if entry is not None:
                weight = entry.binding
                for tok, prob in entry.next_dist.items():
                    result[tok] += weight * prob
                total_weight += weight

        # Level 2: Trigram
        if len(context) >= 2:
            ctx = tuple(int(x) for x in context[-2:])
            entry = self.levels[2].patterns.get(ctx)
            if entry is not None:
                weight = entry.binding
                for tok, prob in entry.next_dist.items():
                    result[tok] += weight * prob
                total_weight += weight

        # Level 1: Bigram
        if len(context) >= 1:
            ctx = (int(context[-1]),)
            entry = self.levels[1].patterns.get(ctx)
            if entry is not None:
                weight = entry.binding
                for tok, prob in entry.next_dist.items():
                    result[tok] += weight * prob
                total_weight += weight

        if total_weight > 0:
            result /= total_weight
            # Ensure valid distribution
            result = np.clip(result, 1e-10, None)
            result /= result.sum()
            return result, total_weight
        else:
            return None, 0.0

    def predict_batch(self, contexts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for efficiency during training/eval.

        Args:
            contexts: (batch_size, seq_len) uint16 array

        Returns:
            distributions: (batch_size, vocab_size) float array
            confidences: (batch_size,) float array
        """
        batch_size = contexts.shape[0]
        dists = np.zeros((batch_size, self.vocab_size), dtype=np.float64)
        confs = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            d, c = self.predict(contexts[i])
            if d is not None:
                dists[i] = d
                confs[i] = c
            else:
                # Uniform fallback
                dists[i] = 1.0 / self.vocab_size

        return dists, confs

    # -------------------------------------------------------------------
    # Serialization (for 16MB artifact)
    # -------------------------------------------------------------------

    def serialize(self) -> bytes:
        """
        Serialize the pattern store to a compact binary format.

        Format per level:
            [num_patterns: uint32]
            For each pattern:
                [context_tokens: context_len × uint16]
                [binding: float32]
                [num_next: uint16]
                For each next token:
                    [token_id: uint16]
                    [prob_scaled: uint16]  (prob × 65535)
        """
        buf = io.BytesIO()

        # Header
        buf.write(struct.pack('<I', 3))  # num_levels

        for level_id in [1, 2, 3]:
            store = self.levels.get(level_id)
            if store is None:
                buf.write(struct.pack('<IB', 0, 0))  # empty level
                continue

            patterns = list(store.patterns.values())
            buf.write(struct.pack('<I', len(patterns)))
            buf.write(struct.pack('<B', store.context_len))

            for entry in patterns:
                # Context tokens
                for t in entry.pattern:
                    buf.write(struct.pack('<H', t))
                # Binding energy
                buf.write(struct.pack('<f', entry.binding))
                # Distribution
                buf.write(struct.pack('<H', len(entry.next_dist)))
                for tok, prob in entry.next_dist.items():
                    buf.write(struct.pack('<H', tok))
                    buf.write(struct.pack('<H', min(65535, int(prob * 65535))))

        raw = buf.getvalue()
        # Compress with zlib
        compressed = zlib.compress(raw, level=9)
        # Prepend uncompressed size for decompression
        return struct.pack('<I', len(raw)) + compressed

    @classmethod
    def deserialize(cls, data: bytes, vocab_size: int = 1024) -> 'HypergraphPatternStore':
        """Deserialize from compact binary format."""
        store = cls(vocab_size=vocab_size)

        # Uncompressed size
        raw_size = struct.unpack('<I', data[:4])[0]
        raw = zlib.decompress(data[4:])

        buf = io.BytesIO(raw)

        num_levels = struct.unpack('<I', buf.read(4))[0]

        for _ in range(num_levels):
            num_patterns = struct.unpack('<I', buf.read(4))[0]
            context_len = struct.unpack('<B', buf.read(1))[0]

            if num_patterns == 0:
                continue

            # Determine level from context_len
            level_id = {1: 1, 2: 2, 4: 3}.get(context_len, 1)
            level_store = LevelStore(level=level_id, context_len=context_len)

            for _ in range(num_patterns):
                # Context tokens
                pattern = tuple(
                    struct.unpack('<H', buf.read(2))[0]
                    for _ in range(context_len)
                )
                # Binding
                binding = struct.unpack('<f', buf.read(4))[0]
                # Distribution
                num_next = struct.unpack('<H', buf.read(2))[0]
                next_dist = {}
                for _ in range(num_next):
                    tok = struct.unpack('<H', buf.read(2))[0]
                    prob_scaled = struct.unpack('<H', buf.read(2))[0]
                    next_dist[tok] = prob_scaled / 65535.0

                entry = PatternEntry(
                    pattern=pattern,
                    next_dist=next_dist,
                    count=0,
                    binding=binding,
                    level=level_id,
                )
                level_store.patterns[pattern] = entry
                level_store.total_binding += binding

            store.levels[level_id] = level_store

        store._built = True
        return store

    # -------------------------------------------------------------------
    # Stats / debugging
    # -------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics for the pattern store."""
        result = {
            'total_tokens_scanned': self.total_tokens,
            'vocab_size': self.vocab_size,
            'built': self._built,
            'levels': {},
        }
        for level_id, store in self.levels.items():
            result['levels'][level_id] = {
                'context_len': store.context_len,
                'num_patterns': len(store.patterns),
                'total_binding': store.total_binding,
                'mean_binding': (store.total_binding / max(1, len(store.patterns))),
                'budget_bytes': store.budget_bytes,
                'estimated_size': store.size_estimate(),
            }

        # Serialized size
        if self._built:
            serialized = self.serialize()
            result['serialized_bytes'] = len(serialized)

        return result


# ---------------------------------------------------------------------------
# Torch integration for hybrid prediction
# ---------------------------------------------------------------------------

def hypergraph_to_torch_logits(hyper_dist: np.ndarray,
                                confidence: float,
                                neural_logits,  # torch.Tensor
                                temperature: float = 1.0,
                                min_confidence: float = 0.1):
    """
    Combine hypergraph prediction with neural logits using
    binding-energy-weighted interpolation.

    P(next) = λ · P_hyper + (1-λ) · softmax(neural_logits)

    where λ = sigmoid(log(confidence) - log(min_confidence))

    Args:
        hyper_dist: (vocab_size,) numpy probability distribution
        confidence: binding confidence from hypergraph
        neural_logits: (vocab_size,) torch tensor of raw logits
        temperature: softmax temperature for neural logits
        min_confidence: confidence threshold below which neural dominates

    Returns:
        combined_logits: torch tensor of log-probabilities
    """
    import torch

    # Compute interpolation weight
    if confidence > min_confidence:
        lam = 1.0 / (1.0 + math.exp(-(math.log(confidence) - math.log(min_confidence))))
    else:
        lam = 0.0

    # Neural softmax
    neural_probs = torch.softmax(neural_logits / temperature, dim=-1)

    # Hypergraph probs as tensor
    hyper_probs = torch.tensor(hyper_dist, dtype=neural_probs.dtype,
                                device=neural_probs.device)

    # Interpolate
    combined = lam * hyper_probs + (1.0 - lam) * neural_probs

    # Back to log space
    return torch.log(combined.clamp(min=1e-10))


def batch_hypergraph_logits(store: HypergraphPatternStore,
                            context_tokens: np.ndarray,
                            neural_logits,  # torch.Tensor (batch, vocab)
                            temperature: float = 1.0):
    """
    Batch version of hypergraph + neural interpolation.

    Args:
        store: built HypergraphPatternStore
        context_tokens: (batch_size, seq_len) uint16 numpy array
        neural_logits: (batch_size, vocab_size) torch tensor
        temperature: softmax temperature

    Returns:
        combined_log_probs: (batch_size, vocab_size) torch tensor
    """
    import torch

    batch_size = context_tokens.shape[0]
    hyper_dists, confidences = store.predict_batch(context_tokens)

    # Convert to torch
    hyper_probs = torch.tensor(hyper_dists, dtype=neural_logits.dtype,
                                device=neural_logits.device)
    conf_tensor = torch.tensor(confidences, dtype=neural_logits.dtype,
                                device=neural_logits.device)

    # Compute lambda per sample
    min_conf = 0.1
    lam = torch.sigmoid(torch.log(conf_tensor.clamp(min=1e-10)) - math.log(min_conf))
    lam = lam.unsqueeze(-1)  # (batch, 1)

    # Neural softmax
    neural_probs = torch.softmax(neural_logits / temperature, dim=-1)

    # Interpolate
    combined = lam * hyper_probs + (1.0 - lam) * neural_probs

    return torch.log(combined.clamp(min=1e-10))


# ---------------------------------------------------------------------------
# FineWeb binary data loading
# ---------------------------------------------------------------------------

def load_fineweb_tokens(path: str) -> np.ndarray:
    """
    Load tokens from a FineWeb .bin file.
    Format: 256 x int32 header, then uint16 tokens.
    """
    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, f"Bad magic: {header[0]}"
        n_tokens = header[2]
        tokens = np.frombuffer(f.read(n_tokens * 2), dtype=np.uint16)
    return tokens


def build_store_from_shards(shard_paths: List[str],
                             vocab_size: int = 1024,
                             budget_bytes: int = 6_000_000,
                             min_count: int = 5,
                             top_k_next: int = 32,
                             max_shards: int = 10) -> HypergraphPatternStore:
    """
    Build a HypergraphPatternStore from FineWeb training shards.

    Args:
        shard_paths: list of .bin file paths
        vocab_size: token vocabulary size
        budget_bytes: total byte budget for pattern store
        min_count: minimum pattern count
        top_k_next: max next-tokens per pattern
        max_shards: max shards to scan (for time budget)

    Returns:
        Built HypergraphPatternStore
    """
    store = HypergraphPatternStore(vocab_size=vocab_size,
                                    max_budget_bytes=budget_bytes)

    # Budget split: 33% bigram, 42% trigram, 25% 5-gram
    bigram_budget = int(budget_bytes * 0.33)
    trigram_budget = int(budget_bytes * 0.42)
    fivegram_budget = int(budget_bytes * 0.25)

    for i, path in enumerate(shard_paths[:max_shards]):
        tokens = load_fineweb_tokens(path)
        store.scan_tokens_fast(tokens)
        print(f"  Scanned shard {i+1}/{min(len(shard_paths), max_shards)}: "
              f"{len(tokens):,} tokens")

    store.build(
        bigram_budget=bigram_budget,
        trigram_budget=trigram_budget,
        fivegram_budget=fivegram_budget,
        min_count=min_count,
        top_k_next=top_k_next,
    )

    return store
