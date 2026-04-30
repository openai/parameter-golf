"""
Fast BPE training and encoding for BESE tokenizer.

Replaces the O(num_merges * total_tokens) pure-Python implementation with
an O(total_tokens * log(total_tokens)) priority-queue based approach.

The key insight: instead of scanning the entire corpus once per merge,
we maintain a priority queue of pair counts and a doubly-linked list of
tokens. Each merge only touches positions where the merged pair exists.

On 10K FineWeb docs (~40M base tokens), this reduces BPE training from
~22 minutes to ~30 seconds, and encoding from ~25 minutes to ~60 seconds.
"""

from __future__ import annotations

import json
import heapq
import numpy as np
from collections import defaultdict
from pathlib import Path

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
except ImportError:
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
    """Convert text to BESE base token sequence."""
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


# ---------------------------------------------------------------------------
# Fast BPE training using indexed pair counting
# ---------------------------------------------------------------------------

class _Node:
    """Doubly-linked list node for fast BPE merge operations."""
    __slots__ = ("token", "prev", "next", "doc_id")

    def __init__(self, token: int, doc_id: int):
        self.token = token
        self.prev = None
        self.next = None
        self.doc_id = doc_id


def _encode_texts_worker(texts: list[str]) -> list[list[int]]:
    """Worker function for parallel base-token encoding."""
    return [_text_to_base_tokens(text) for text in texts]


def train_bpe_merges_fast(texts: list[str], num_merges: int = 250, verbose: bool = True) -> list:
    """
    Learn BPE merges using an efficient indexed approach.

    Instead of scanning all sequences for every merge, we:
    1. Build a doubly-linked list of all tokens (encoding parallelized across CPUs)
    2. Maintain a max-heap of pair counts for O(log n) best-pair lookup
    3. For each merge, update only the affected positions

    This is O(total_tokens + num_merges * avg_pair_count * log(num_pairs)) instead of
    O(num_merges * total_tokens).
    """
    import multiprocessing as mp

    if verbose:
        print(f"Encoding {len(texts)} texts with base BESE tokenizer...")

    # Step 1: Parallel encode all texts to base tokens
    n_workers = min(mp.cpu_count(), 128)
    chunk_size = max(1, len(texts) // n_workers)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    import time as _time
    t_enc = _time.time()
    with mp.Pool(n_workers) as pool:
        encoded_chunks = pool.map(_encode_texts_worker, chunks)
    all_encoded = [tokens for chunk in encoded_chunks for tokens in chunk]
    if verbose:
        print(f"  Parallel encoding: {n_workers} workers, {_time.time() - t_enc:.1f}s")

    # Step 1b: Build linked lists and pair index from encoded tokens
    doc_heads = []
    pair_positions = defaultdict(set)
    all_nodes = []

    total_base = 0
    for doc_id, base_tokens in enumerate(all_encoded):
        total_base += len(base_tokens)
        if not base_tokens:
            doc_heads.append(None)
            continue

        nodes = []
        for t in base_tokens:
            node = _Node(t, doc_id)
            node_id = len(all_nodes)
            if nodes:
                prev_node = nodes[-1]
                prev_node.next = node_id
                node.prev = len(all_nodes) - 1
            nodes.append(node)
            all_nodes.append(node)

        doc_heads.append(len(all_nodes) - len(nodes))

        for i in range(len(nodes) - 1):
            nid = len(all_nodes) - len(nodes) + i
            pair = (nodes[i].token, nodes[i + 1].token)
            pair_positions[pair].add(nid)

    del all_encoded

    if verbose:
        print(f"Base tokens: {total_base:,}")
        print(f"Unique pairs: {len(pair_positions):,}")
        print(f"Learning {num_merges} BPE merges (heap mode)...")

    # Step 2: Greedily merge most frequent pairs using a max-heap
    merges = []
    next_id = BASE_VOCAB_SIZE

    pair_counts = {pair: len(positions) for pair, positions in pair_positions.items()}

    # Build max-heap (negate counts for max-heap via min-heap)
    heap = [(-count, pair) for pair, count in pair_counts.items() if count >= 2]
    heapq.heapify(heap)

    merge_num = 0
    while merge_num < num_merges and heap:
        # Pop best pair from heap (skip stale entries)
        while heap:
            neg_count, best_pair = heapq.heappop(heap)
            current_count = pair_counts.get(best_pair, 0)
            if current_count >= 2 and current_count == -neg_count:
                best_count = current_count
                break
        else:
            break

        # Get all positions where this pair occurs and filter stale entries
        raw_positions = pair_positions.get(best_pair, set())
        positions = []
        for nid in raw_positions:
            node_a = all_nodes[nid]
            if node_a.next is None:
                continue
            node_b = all_nodes[node_a.next]
            if node_a.token != best_pair[0] or node_b.token != best_pair[1]:
                continue
            positions.append(nid)

        # Remove the pair from tracking
        if best_pair in pair_positions:
            del pair_positions[best_pair]
        if best_pair in pair_counts:
            del pair_counts[best_pair]

        # Re-check count after filtering stale positions
        if len(positions) < 2:
            continue  # don't increment merge_num — this wasn't a real merge

        new_id = next_id
        merges.append((best_pair, new_id))

        # Apply the merge at each valid position
        for nid in positions:
            node_a = all_nodes[nid]
            node_b = all_nodes[node_a.next]

            # Remove old pairs involving node_a and node_b from index
            # Left neighbor pair: (prev, a)
            if node_a.prev is not None:
                prev_node = all_nodes[node_a.prev]
                old_left_pair = (prev_node.token, node_a.token)
                nid_prev = node_a.prev
                pair_positions.get(old_left_pair, set()).discard(nid_prev)
                if old_left_pair in pair_positions and not pair_positions[old_left_pair]:
                    del pair_positions[old_left_pair]
                if old_left_pair in pair_counts:
                    pair_counts[old_left_pair] = max(pair_counts[old_left_pair] - 1, 0)
                    if pair_counts[old_left_pair] == 0:
                        del pair_counts[old_left_pair]

            # Right neighbor pair: (b, next)
            if node_b.next is not None:
                next_node = all_nodes[node_b.next]
                old_right_pair = (node_b.token, next_node.token)
                b_nid = node_a.next
                pair_positions.get(old_right_pair, set()).discard(b_nid)
                if old_right_pair in pair_positions and not pair_positions[old_right_pair]:
                    del pair_positions[old_right_pair]
                if old_right_pair in pair_counts:
                    pair_counts[old_right_pair] = max(pair_counts[old_right_pair] - 1, 0)
                    if pair_counts[old_right_pair] == 0:
                        del pair_counts[old_right_pair]

            # Merge: node_a becomes new_id, node_b is unlinked
            node_a.token = new_id
            node_a.next = node_b.next
            if node_b.next is not None:
                all_nodes[node_b.next].prev = nid

            # Add new pairs and push onto heap
            if node_a.prev is not None:
                prev_node = all_nodes[node_a.prev]
                new_left_pair = (prev_node.token, new_id)
                pair_positions.setdefault(new_left_pair, set()).add(node_a.prev)
                new_count = pair_counts.get(new_left_pair, 0) + 1
                pair_counts[new_left_pair] = new_count
                heapq.heappush(heap, (-new_count, new_left_pair))

            if node_a.next is not None:
                next_node = all_nodes[node_a.next]
                new_right_pair = (new_id, next_node.token)
                pair_positions.setdefault(new_right_pair, set()).add(nid)
                new_count = pair_counts.get(new_right_pair, 0) + 1
                pair_counts[new_right_pair] = new_count
                heapq.heappush(heap, (-new_count, new_right_pair))

        next_id += 1
        merge_num += 1

        if verbose and (merge_num <= 20 or merge_num % 50 == 0 or merge_num == num_merges):
            # Count remaining tokens (approximate)
            print(
                f"  Merge {merge_num:4d}: ({best_pair[0]:3d},{best_pair[1]:3d}) -> {new_id:4d}"
                f"  count={best_count:6d}"
            )

    if verbose:
        print(f"\nDone. Learned {len(merges)} merges.")
        print(
            f"Vocabulary: {BASE_VOCAB_SIZE} base + {len(merges)} merges = "
            f"{BASE_VOCAB_SIZE + len(merges)} total"
        )
    return merges


# ---------------------------------------------------------------------------
# HuggingFace tokenizers backend (Rust, ~100x faster than pure Python)
# ---------------------------------------------------------------------------

def train_bpe_merges_hf(texts: list[str], num_merges: int = 1024, verbose: bool = True) -> list:
    """
    Train BPE merges using the HuggingFace `tokenizers` library (Rust backend).

    ~100x faster than train_bpe_merges_fast: reduces 2-3 hours to ~2-5 minutes
    on 32 cores for 100K FineWeb docs.  Returns merges in the same format as
    train_bpe_merges_fast: [((left_id, right_id), new_id), ...]

    Requires: pip install tokenizers
    """
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    except ImportError:
        raise ImportError(
            "HuggingFace tokenizers not installed. Run: pip install tokenizers\n"
            "Falling back to pure-Python BPE is possible via train_bpe_merges_fast()."
        )

    import time as _time
    import multiprocessing as mp
    import tempfile, os

    # Map base token IDs 0-(BASE_VOCAB_SIZE-1) to unique Unicode private-use chars.
    # U+E000..U+E027 are valid single-codepoint, valid UTF-8, never in real text.
    BASE_CHARS = [chr(0xE000 + i) for i in range(BASE_VOCAB_SIZE)]

    # ------------------------------------------------------------------
    # Step 1: Parallel encode texts → base token IDs
    # ------------------------------------------------------------------
    if verbose:
        print(f"Encoding {len(texts)} texts with base BESE tokenizer...")

    n_workers = min(mp.cpu_count(), 128)
    chunk_size = max(1, len(texts) // n_workers)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    t_enc = _time.time()
    with mp.Pool(n_workers) as pool:
        encoded_chunks = pool.map(_encode_texts_worker, chunks)
    all_encoded = [doc for chunk in encoded_chunks for doc in chunk]
    if verbose:
        print(f"  Parallel encoding: {n_workers} workers, {_time.time() - t_enc:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Write corpus to a temp file — one doc per line, no spaces.
    # Each base token becomes one Unicode private-use char.
    # Whitespace pre-tokenizer treats each line (= no-space sequence) as
    # one "word", so BPE merges happen freely within each document.
    # ------------------------------------------------------------------
    if verbose:
        print("  Writing corpus temp file...")
    t_write = _time.time()
    fd, tmp_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for doc_tokens in all_encoded:
                f.write("".join(BASE_CHARS[t] for t in doc_tokens) + "\n")
        if verbose:
            size_mb = os.path.getsize(tmp_path) / 1e6
            print(f"  Corpus file: {size_mb:.0f} MB  ({_time.time() - t_write:.1f}s)")

        # ------------------------------------------------------------------
        # Step 3: Train BPE with HF tokenizers (Rust, multi-threaded)
        # ------------------------------------------------------------------
        if verbose:
            print(f"  Training BPE ({num_merges} merges) with HuggingFace tokenizers...")
        t_bpe = _time.time()

        tokenizer = Tokenizer(models.BPE())
        # Whitespace splits only on Unicode whitespace — our private-use chars
        # are never whitespace, so each doc line becomes exactly one "word".
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=BASE_VOCAB_SIZE + num_merges,
            min_frequency=2,
            initial_alphabet=BASE_CHARS,
            special_tokens=[],
            show_progress=verbose,
        )
        tokenizer.train([tmp_path], trainer)

        if verbose:
            print(f"  HF BPE done in {_time.time() - t_bpe:.1f}s")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Step 4: Extract merges and convert char-strings → token ID pairs.
    # Save the model to a temp dir to read merges.txt (one merge per line:
    # "str_a str_b").  Replay the sequence to reconstruct integer IDs.
    # ------------------------------------------------------------------
    merge_dir = tempfile.mkdtemp()
    try:
        tokenizer.model.save(merge_dir)
        merges_path = os.path.join(merge_dir, "merges.txt")
        with open(merges_path, encoding="utf-8") as mf:
            raw_lines = mf.read().splitlines()
    finally:
        import shutil
        shutil.rmtree(merge_dir, ignore_errors=True)

    # Filter header lines (start with #) and parse "str_a str_b" pairs
    hf_merges = []
    for line in raw_lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2:
            hf_merges.append((parts[0], parts[1]))

    str_to_id: dict[str, int] = {c: i for i, c in enumerate(BASE_CHARS)}
    result: list = []

    for str_a, str_b in hf_merges:
        id_a = str_to_id.get(str_a)
        id_b = str_to_id.get(str_b)
        if id_a is None or id_b is None:
            continue   # shouldn't happen with a clean corpus
        new_id = BASE_VOCAB_SIZE + len(result)
        str_to_id[str_a + str_b] = new_id
        result.append(((id_a, id_b), new_id))
        if len(result) >= num_merges:
            break

    if verbose:
        print(f"\nDone. Learned {len(result)} merges.")
        print(
            f"Vocabulary: {BASE_VOCAB_SIZE} base + {len(result)} merges = "
            f"{BASE_VOCAB_SIZE + len(result)} total"
        )
    return result


# ---------------------------------------------------------------------------
# Fast encoding using merge priority (hash-map based)
# ---------------------------------------------------------------------------

def _build_merge_priority(merges):
    """Build a priority lookup: (a, b) -> (priority, new_id).
    Lower priority = should be applied first."""
    return {pair: (i, new_id) for i, (pair, new_id) in enumerate(merges)}


def encode_fast(text: str, merges: list, _merge_priority=None) -> np.ndarray:
    """
    Encode text to BESE+BPE tokens using hash-map based merging.

    Instead of scanning for each merge in order (O(merges * tokens)),
    we use a linked list + priority queue approach:
    1. Start with base tokens in a linked list
    2. Find all adjacent pairs and their merge priorities
    3. Apply merges in priority order (lowest first)
    4. After each merge, check if new pairs can be merged

    This is O(tokens * log(tokens)) instead of O(merges * tokens).
    """
    base_tokens = _text_to_base_tokens(text)
    if not base_tokens or not merges:
        return np.array(base_tokens, dtype=np.uint16)

    if _merge_priority is None:
        _merge_priority = _build_merge_priority(merges)

    # Build doubly linked list
    n = len(base_tokens)
    tokens = list(base_tokens)
    prev_arr = list(range(-1, n - 1))  # prev[i] = i-1
    next_arr = list(range(1, n + 1))   # next[i] = i+1, n = sentinel
    next_arr[-1] = n  # sentinel

    # Priority queue: (priority, position_id, pair_at_creation)
    # pair_at_creation is used to validate the entry is still current
    heap = []

    # Initialize heap with all mergeable pairs
    for i in range(n - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair in _merge_priority:
            pri, _ = _merge_priority[pair]
            heapq.heappush(heap, (pri, i, pair))

    while heap:
        pri, pos, pair_check = heapq.heappop(heap)

        # Validate: position must still have this exact pair
        if tokens[pos] != pair_check[0]:
            continue
        nxt = next_arr[pos]
        if nxt >= n or tokens[nxt] != pair_check[1]:
            continue

        _, new_id = _merge_priority[pair_check]

        # Merge: pos takes new_id, nxt is removed
        tokens[pos] = new_id
        next_arr[pos] = next_arr[nxt]
        if next_arr[nxt] < n:
            prev_arr[next_arr[nxt]] = pos

        # Check new left pair
        if prev_arr[pos] >= 0:
            left = prev_arr[pos]
            new_pair = (tokens[left], new_id)
            if new_pair in _merge_priority:
                lp, _ = _merge_priority[new_pair]
                heapq.heappush(heap, (lp, left, new_pair))

        # Check new right pair
        if next_arr[pos] < n:
            right = next_arr[pos]
            new_pair = (new_id, tokens[right])
            if new_pair in _merge_priority:
                rp, _ = _merge_priority[new_pair]
                heapq.heappush(heap, (rp, pos, new_pair))

    # Collect result
    result = []
    pos = 0
    while pos < n:
        result.append(tokens[pos])
        nxt = next_arr[pos]
        if nxt <= pos:
            break  # safety
        pos = nxt

    # Handle case where linked list walk doesn't start from a valid head
    if not result:
        # Find first valid node
        for i in range(n):
            if prev_arr[i] < 0 or i == 0:
                pos = i
                while pos < n:
                    result.append(tokens[pos])
                    pos = next_arr[pos]
                break

    return np.array(result, dtype=np.uint16)


class FastBESEBPETokenizer:
    """Fast BESE+BPE tokenizer using indexed merge operations."""

    def __init__(self, merges=None):
        self.merges = merges or []
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        self._merge_map = {pair: new_id for pair, new_id in self.merges}
        self._merge_priority = _build_merge_priority(self.merges)
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

    def encode(self, text: str) -> np.ndarray:
        return encode_fast(text, self.merges, self._merge_priority)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(text) for text in texts]

    def decode_token_to_base(self, token_id: int) -> list[int]:
        if token_id < BASE_VOCAB_SIZE:
            return [token_id]
        if token_id in self._decode_chains:
            left, right = self._decode_chains[token_id]
            return self.decode_token_to_base(left) + self.decode_token_to_base(right)
        return [UNK_ID]

    def decode(self, token_ids: list[int]) -> str:
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

    def get_bytes_per_token_lut(self) -> np.ndarray:
        return self._bpt.copy()

    def save(self, path):
        """Save in same format as BESEBPETokenizer for compatibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_type": "bese_bpe",
            "version": 2,
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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    sample = [
        "The cat sat on the mat.",
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Parameter Golf is a challenge to train the best language model.",
        "BESE uses a 40-token structured alphabet with BPE merges on top.",
    ] * 200

    print("=== Fast BPE Training ===")
    t0 = time.time()
    merges = train_bpe_merges_fast(sample, num_merges=100, verbose=True)
    t1 = time.time()
    print(f"Training took {t1-t0:.2f}s")

    print("\n=== Fast Encoding ===")
    tok = FastBESEBPETokenizer(merges=merges)

    # Correctness check
    test_texts = [
        "The cat sat on the mat.",
        "Hello world!",
        "Testing 123 with special chars: é, ñ, ü",
        "Multiple\nlines\nof\ntext.",
    ]
    for text in test_texts:
        enc = tok.encode(text)
        dec = tok.decode(enc.tolist())
        bpt = tok.get_bytes_per_token_lut()
        tb = int(sum(bpt[t] for t in enc))
        ub = len(text.encode("utf-8"))
        status = "OK" if tb == ub else "FAIL"
        print(f'  [{status}] "{text[:40]}..." -> {len(enc)} tokens, bytes {tb}/{ub}')

    # Speed benchmark
    print("\n=== Speed Benchmark ===")
    big_text = " ".join(sample)
    t0 = time.time()
    for _ in range(10):
        tok.encode(big_text)
    t1 = time.time()
    print(f"Encoded {len(big_text)} chars x 10 in {t1-t0:.2f}s ({len(big_text)*10/(t1-t0):.0f} chars/sec)")
