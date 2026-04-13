"""Hybrid tokenizer engine for Parameter Golf.

Combines TokenMonster's top-down candidate generation, Length-MAX's scoring,
and DP-optimal encoding to build a vocabulary that minimizes tokens/byte.

Components:
    ByteTrie        — Byte-level trie for fast prefix matching
    dp_encode       — DP-optimal tokenization (minimum-token segmentation)
    generate_candidates       — In-memory hierarchical substring counting (small corpora)
    generate_candidates_large — Chunked/streaming version for large corpora (5GB+)
    build_vocabulary    — Greedy selection with periodic DP re-scoring
    create_sp_model     — SentencePiece .model creation via protobuf injection
"""
import concurrent.futures
import functools
import heapq
import json
import os
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from numba import njit, prange

print = functools.partial(print, flush=True)

# Vocabulary layout (matches baseline fineweb_1024_bpe.model)
ID_PAD = 0      # <pad>  type=CONTROL
ID_BOS = 1      # <s>    type=CONTROL
ID_EOS = 2      # </s>   type=CONTROL
ID_UNK = 3      # <unk>  type=UNKNOWN
BYTE_OFFSET = 4  # IDs 4-259 = 256 byte fallback tokens
LEARNED_OFFSET = 260  # IDs 260+ = learned subword tokens
NUM_FIXED = LEARNED_OFFSET  # 4 special + 256 byte = 260

# SP protobuf piece types
TYPE_NORMAL = 1
TYPE_UNKNOWN = 2
TYPE_CONTROL = 3
TYPE_BYTE = 6

# Document separator byte (never appears in normalized UTF-8 text)
SEP = 0x00


# ---------------------------------------------------------------------------
# ByteTrie
# ---------------------------------------------------------------------------

class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children: dict[int, "_TrieNode"] = {}
        self.token_id: int = -1  # -1 = not a token endpoint


class ByteTrie:
    """Byte-level trie for fast prefix matching in DP encoder."""

    def __init__(self):
        self.root = _TrieNode()
        self._token_bytes: dict[int, bytes] = {}  # token_id -> bytes

    def insert(self, token_bytes: bytes, token_id: int):
        node = self.root
        for b in token_bytes:
            if b not in node.children:
                node.children[b] = _TrieNode()
            node = node.children[b]
        node.token_id = token_id
        self._token_bytes[token_id] = token_bytes

    def token_len(self, token_id: int) -> int:
        return len(self._token_bytes[token_id])

    def token_bytes(self, token_id: int) -> bytes:
        return self._token_bytes[token_id]


def make_base_trie() -> ByteTrie:
    """Create a trie with the 256 byte fallback tokens (IDs 4-259)."""
    trie = ByteTrie()
    for b in range(256):
        trie.insert(bytes([b]), BYTE_OFFSET + b)
    return trie


# ---------------------------------------------------------------------------
# Flat Trie (for JIT DP encoder)
# ---------------------------------------------------------------------------

class FlatTrie:
    """Flattened trie for Numba JIT: contiguous int32 arrays, O(1) child lookup.

    Converts a ByteTrie into two arrays:
      children[node * 256 + byte] → child node index (-1 = no child)
      token_ids[node]             → token_id (-1 = not an endpoint)

    This enables the DP encoder to run entirely in JIT without Python objects.
    Memory: n_nodes × 256 × 4 bytes.  For ~5K nodes → ~5 MB.
    """

    def __init__(self, byte_trie: ByteTrie):
        # BFS to assign node indices and build flat arrays
        from collections import deque
        node_map: dict[int, int] = {id(byte_trie.root): 0}
        queue = deque([byte_trie.root])
        nodes = [byte_trie.root]
        while queue:
            node = queue.popleft()
            for b, child in node.children.items():
                child_id = id(child)
                if child_id not in node_map:
                    node_map[child_id] = len(nodes)
                    nodes.append(child)
                    queue.append(child)

        n_nodes = len(nodes)
        self.children = np.full(n_nodes * 256, -1, dtype=np.int32)
        self.token_ids = np.full(n_nodes, -1, dtype=np.int32)

        for node in nodes:
            idx = node_map[id(node)]
            self.token_ids[idx] = node.token_id
            for b, child in node.children.items():
                self.children[idx * 256 + b] = node_map[id(child)]

        self.n_nodes = n_nodes


# ---------------------------------------------------------------------------
# DP-Optimal Encoder
# ---------------------------------------------------------------------------

@njit(cache=True)
def _dp_encode_jit(data, n, dp, bt_id, bt_len, trie_children, trie_token_ids):
    """JIT-compiled DP encoder core.  ~50-100× faster than Python version."""
    INF = n + 1
    for i in range(n + 1):
        dp[i] = INF
        bt_id[i] = 0
        bt_len[i] = 0
    dp[0] = 0

    for i in range(n):
        if dp[i] >= INF:
            continue
        cost_i = dp[i]
        node = np.int32(0)  # root
        for j in range(i, n):
            b = data[j]
            next_node = trie_children[node * 256 + b]
            if next_node < 0:
                break
            node = next_node
            tid = trie_token_ids[node]
            if tid >= 0:
                mlen = j - i + 1
                new_cost = cost_i + 1
                dest = j + 1
                if new_cost < dp[dest] or (new_cost == dp[dest] and mlen > bt_len[dest]):
                    dp[dest] = new_cost
                    bt_id[dest] = tid
                    bt_len[dest] = mlen


def dp_encode(data: bytes, trie: ByteTrie, flat: FlatTrie | None = None) -> list[int]:
    """Find the minimum-token segmentation of data using dynamic programming.

    Tiebreaker: when two segmentations use the same number of tokens, prefer
    the one that uses longer tokens (max-munch). This produces more consistent
    segmentation and better aligns with SentencePiece conventions.

    If ``flat`` is provided, uses JIT-compiled encoder (~50-100× faster).
    Time: O(N × K) where K = max token length in trie.
    """
    n = len(data)
    if n == 0:
        return []

    if flat is not None:
        arr = np.frombuffer(data, dtype=np.uint8)
        dp_arr = np.empty(n + 1, dtype=np.int32)
        bt_id_arr = np.empty(n + 1, dtype=np.int32)
        bt_len_arr = np.empty(n + 1, dtype=np.int32)
        _dp_encode_jit(arr, n, dp_arr, bt_id_arr, bt_len_arr,
                        flat.children, flat.token_ids)
        # Backtrack
        tokens = []
        pos = n
        while pos > 0:
            tokens.append(int(bt_id_arr[pos]))
            pos -= int(bt_len_arr[pos])
        tokens.reverse()
        return tokens

    INF = n + 1
    # dp[i] = minimum tokens to encode data[0:i]
    dp = [INF] * (n + 1)
    # backtrack[i] = (token_id, match_len) for the best transition arriving at i
    bt_id = [0] * (n + 1)
    bt_len = [0] * (n + 1)
    dp[0] = 0

    root = trie.root
    for i in range(n):
        if dp[i] >= INF:
            continue
        cost_i = dp[i]
        node = root
        for j in range(i, n):
            b = data[j]
            child = node.children.get(b)
            if child is None:
                break
            node = child
            if node.token_id >= 0:
                mlen = j - i + 1
                new_cost = cost_i + 1
                dest = j + 1
                # Prefer fewer tokens; tiebreak on longer match
                if new_cost < dp[dest] or (new_cost == dp[dest] and mlen > bt_len[dest]):
                    dp[dest] = new_cost
                    bt_id[dest] = node.token_id
                    bt_len[dest] = mlen

    # Backtrack
    tokens = []
    pos = n
    while pos > 0:
        tokens.append(bt_id[pos])
        pos -= bt_len[pos]
    tokens.reverse()
    return tokens


# ---------------------------------------------------------------------------
# Text Normalization (matches SentencePiece nmt_nfkc + escape_whitespaces)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Apply SentencePiece-equivalent normalization.

    Matches the baseline model's normalizer_spec:
      - name: nmt_nfkc (NFKC unicode normalization)
      - remove_extra_whitespaces: true (collapse runs of whitespace)
      - escape_whitespaces: true (spaces -> U+2581 ▁)
      - add_dummy_prefix: false (no leading ▁)
    """
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())  # collapse all whitespace to single space
    text = text.replace(" ", "\u2581")  # escape spaces
    return text


# ---------------------------------------------------------------------------
# JSONL Streaming
# ---------------------------------------------------------------------------

def iter_jsonl_texts(path: Path, limit: int | None = None):
    """Yield document texts from a JSONL file, one per line."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                if text:
                    yield text
                    count += 1
            except json.JSONDecodeError:
                continue


def write_normalized_corpus(
    jsonl_path: Path,
    output_path: Path,
    max_bytes: int,
) -> int:
    """Stream JSONL → normalize → write binary file with SEP separators.

    Returns total bytes written. The output file can be memory-mapped for
    chunked candidate generation.
    """
    total = 0
    count = 0
    t0 = time.time()
    with open(output_path, "wb") as out:
        for text in iter_jsonl_texts(jsonl_path):
            normalized = normalize_text(text)
            encoded = normalized.encode("utf-8")
            if total + len(encoded) > max_bytes:
                break
            if count > 0:
                out.write(bytes([SEP]))
                total += 1
            out.write(encoded)
            total += len(encoded)
            count += 1
            if count % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"    {count:,} docs, {total / 1e6:.1f} MB, {elapsed:.0f}s...")
    print(f"  Corpus: {count:,} docs, {total:,} bytes ({total / 1e6:.1f} MB)")
    return total


# ---------------------------------------------------------------------------
# Hierarchical Candidate Generation (in-memory, for small corpora ≤ ~2GB)
# ---------------------------------------------------------------------------

def generate_candidates(
    corpus_bytes: bytes,
    max_len: int = 32,
    min_freq: int = 100,
) -> dict[bytes, int]:
    """Extract frequent byte substrings using anti-monotonicity pruning.

    For each length L from 2 to max_len:
      - Only count L-grams whose prefix AND suffix (L-1)-grams survived
      - Keep L-grams with frequency >= min_freq
      - UTF-8 validity is checked at the END, not during counting (otherwise
        anti-monotonicity pruning kills valid multi-byte characters whose
        constituent sub-byte-sequences aren't valid UTF-8 on their own)

    corpus_bytes: concatenated normalized texts separated by 0x00.
    Returns: dict mapping byte sequence -> frequency (only valid UTF-8 keys).
    """
    arr = np.frombuffer(corpus_bytes, dtype=np.uint8)
    N = len(arr)
    all_candidates: dict[bytes, int] = {}

    # Track positions where surviving (L-1)-grams start.
    # L=2 baseline: every non-separator position is valid.
    survived = np.ones(N, dtype=np.bool_)
    survived[arr == SEP] = False

    for L in range(2, max_len + 1):
        n_end = N - L + 1
        if n_end <= 0:
            break

        # Anti-monotonicity: both prefix & suffix (L-1)-grams must have survived
        valid = survived[:n_end] & survived[1 : n_end + 1]
        # Exclude n-grams containing the separator byte
        for j in range(L):
            valid &= arr[j : j + n_end] != SEP

        n_valid = int(valid.sum())
        if n_valid == 0:
            print(f"  L={L:2d}: 0 valid positions — stopping")
            break

        # Count n-grams at valid positions (NO utf-8 filter — that's applied at the end)
        if L <= 4:
            level = _count_hash32(arr, L, valid, n_end, min_freq)
        elif L <= 8:
            level = _count_hash64(arr, L, valid, n_end, min_freq)
        else:
            level = _count_python(corpus_bytes, L, valid, n_end, min_freq)

        n_surv = len(level)
        print(f"  L={L:2d}: {n_valid:>12,} valid positions -> {n_surv:>8,} survivors")

        if n_surv == 0:
            break

        all_candidates.update(level)

        # Update survived-position bitmap for next level
        survived = _update_survived(arr, L, valid, n_end, level, corpus_bytes, N)

    # Final UTF-8 validity filter — only emit candidates that decode cleanly
    valid_candidates: dict[bytes, int] = {}
    rejected = 0
    for key, freq in all_candidates.items():
        try:
            key.decode("utf-8")
            valid_candidates[key] = freq
        except UnicodeDecodeError:
            rejected += 1
    if rejected:
        print(f"  UTF-8 filter: kept {len(valid_candidates):,}, rejected {rejected:,} non-UTF-8")
    return valid_candidates


def _compute_hash32(arr: np.ndarray, L: int, n_end: int) -> np.ndarray:
    """Compute big-endian uint32 hash of each L-gram."""
    h = np.zeros(n_end, dtype=np.uint32)
    for j in range(L):
        h |= arr[j : j + n_end].astype(np.uint32) << (8 * (L - 1 - j))
    return h


def _count_hash32(
    arr: np.ndarray, L: int, valid: np.ndarray, n_end: int, min_freq: int
) -> dict[bytes, int]:
    """Count L-grams (L<=4) using uint32 hashes + np.unique.
    No UTF-8 filter here — applied in generate_candidates at the end."""
    h = _compute_hash32(arr, L, n_end)
    unique, counts = np.unique(h[valid], return_counts=True)
    result: dict[bytes, int] = {}
    for u, c in zip(unique, counts):
        if c < min_freq:
            continue
        result[int(u).to_bytes(L, "big")] = int(c)
    return result


def _count_hash64(
    arr: np.ndarray, L: int, valid: np.ndarray, n_end: int, min_freq: int
) -> dict[bytes, int]:
    """Count L-grams (5<=L<=8) using uint64 hashes + np.unique.
    No UTF-8 filter here — applied in generate_candidates at the end."""
    h = np.zeros(n_end, dtype=np.uint64)
    for j in range(L):
        h |= arr[j : j + n_end].astype(np.uint64) << (8 * (L - 1 - j))
    unique, counts = np.unique(h[valid], return_counts=True)
    result: dict[bytes, int] = {}
    for u, c in zip(unique, counts):
        if c < min_freq:
            continue
        result[int(u).to_bytes(L, "big")] = int(c)
    return result


def _count_python(
    raw: bytes, L: int, valid: np.ndarray, n_end: int, min_freq: int
) -> dict[bytes, int]:
    """Count L-grams (L>=9) using Python Counter on valid positions.
    No UTF-8 filter here — applied in generate_candidates at the end."""
    indices = np.where(valid)[0]
    counter: Counter[bytes] = Counter()
    for i in indices:
        counter[raw[i : i + L]] += 1
    return {key: cnt for key, cnt in counter.items() if cnt >= min_freq}


def _update_survived(
    arr: np.ndarray,
    L: int,
    valid: np.ndarray,
    n_end: int,
    level: dict[bytes, int],
    raw: bytes,
    N: int,
) -> np.ndarray:
    """Update the survived-position bitmap after counting level L."""
    new_survived = np.zeros(N, dtype=np.bool_)

    if L <= 3:
        # Boolean LUT (fits in memory: 2^(8*L) entries, max 16 MB for L=3)
        max_hash = 1 << (8 * L)
        lut = np.zeros(max_hash, dtype=np.bool_)
        for key in level:
            lut[int.from_bytes(key, "big")] = True
        h = _compute_hash32(arr, L, n_end)
        new_survived[:n_end] = valid & lut[h]
    elif L <= 8:
        # np.isin for larger hash spaces
        if not level:
            return new_survived
        if L <= 4:
            survived_arr = np.array(
                [int.from_bytes(k, "big") for k in level], dtype=np.uint32
            )
            h = _compute_hash32(arr, L, n_end)
        else:
            survived_arr = np.array(
                [int.from_bytes(k, "big") for k in level], dtype=np.uint64
            )
            h_64 = np.zeros(n_end, dtype=np.uint64)
            for j in range(L):
                h_64 |= arr[j : j + n_end].astype(np.uint64) << (8 * (L - 1 - j))
            h = h_64
        in_survived = np.isin(h, survived_arr)
        new_survived[:n_end] = valid & in_survived
    else:
        # Python loop (L>=9, positions are few)
        survived_set = set(level.keys())
        indices = np.where(valid)[0]
        for i in indices:
            if raw[i : i + L] in survived_set:
                new_survived[i] = True

    return new_survived


# ---------------------------------------------------------------------------
# Hierarchical Candidate Generation (chunked, for large corpora)
# ---------------------------------------------------------------------------

# Maximum positions to process per numpy chunk.  Keeps peak memory per chunk
# to ~2 GB even for uint64 hashes + np.unique sort workspace.
_CHUNK_POS = 100_000_000


def save_candidates(candidates: dict[bytes, int], path: Path | str,
                    survived_set: set[bytes] | None = None,
                    max_level: int = 0):
    """Save candidates + state to a JSON file for later reuse/resumption.

    Format: {"max_level": N, "candidates": {"hex_key": freq, ...},
             "survived_last": ["hex_key", ...]}

    Uses atomic write (temp file + rename) to prevent data loss on crash.
    """
    path = Path(path)
    data = {
        "max_level": max_level,
        "candidates": {k.hex(): v for k, v in candidates.items()},
    }
    if survived_set is not None:
        data["survived_last"] = [k.hex() for k in survived_set]
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)
    n_cand = len(data["candidates"])
    n_surv = len(data.get("survived_last", []))
    print(f"  Saved {n_cand:,} candidates + {n_surv:,} survived keys to {path}")


def load_candidates(path: Path | str) -> tuple[dict[bytes, int], set[bytes] | None, int]:
    """Load saved candidates. Returns (candidates, survived_set, max_level)."""
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)
    candidates = {bytes.fromhex(k): v for k, v in data["candidates"].items()}
    survived_set = None
    if "survived_last" in data:
        survived_set = {bytes.fromhex(k) for k in data["survived_last"]}
    max_level = data.get("max_level", 0)
    print(f"  Loaded {len(candidates):,} candidates from {path} (max_level={max_level})")
    if survived_set:
        print(f"  Survived set: {len(survived_set):,} keys (for resuming from L={max_level + 1})")
    return candidates, survived_set, max_level


def generate_candidates_large(
    corpus_path: Path | str,
    max_len: int = 16,
    min_freq: int = 500,
    chunk_positions: int = _CHUNK_POS,
    save_path: Path | str | None = None,
    resume_from: Path | str | None = None,
) -> dict[bytes, int]:
    """Memory-efficient candidate generation using a memory-mapped corpus file.

    Processes the corpus in *position chunks* of ``chunk_positions`` bytes.
    Memory usage is O(chunk_positions) regardless of total corpus size.

    Args:
        corpus_path: Binary file written by ``write_normalized_corpus``.
        max_len: Maximum n-gram length to consider.
        min_freq: Minimum frequency for survival.
        chunk_positions: Positions per numpy chunk (default 100M → ~2 GB peak).
        save_path: If set, save all candidates + survived set to this JSON
            file after completion (for reuse with different vocab sizes or
            resumption to higher max_len).
        resume_from: If set, load previously saved candidates and resume from
            the next level after the saved max_level.

    Returns:
        dict mapping byte sequence → frequency (only valid UTF-8 keys).
    """
    corpus_path = Path(corpus_path)
    total_bytes = corpus_path.stat().st_size
    if total_bytes == 0:
        return {}

    N = total_bytes
    # Open file handle for direct reads (NOT memmap — memmap caches all
    # accessed pages in physical RAM, consuming ~25 GB on a 25 GB file.
    # Direct reads + explicit del give us control over working set).
    corpus_fh = open(corpus_path, "rb")
    print(f"  Corpus opened: {N:,} bytes ({N / 1e9:.2f} GB)")

    all_candidates: dict[bytes, int] = {}
    start_L = 2
    # Instead of a full-corpus boolean bitmap, carry the *set* of survived
    # byte patterns between levels.  Much smaller than N booleans.
    prev_survived_set: set[bytes] | None = None

    # Resume from saved state if provided
    if resume_from is not None:
        all_candidates, prev_survived_set, prev_max_level = load_candidates(resume_from)
        start_L = prev_max_level + 1
        print(f"  Resuming from L={start_L} (loaded L=2..{prev_max_level})")

    t_total = time.time()
    for L in range(start_L, max_len + 1):
        t_level = time.time()
        n_end = N - L + 1
        if n_end <= 0:
            break

        # For L≤8, merge as int-keyed dict (avoid creating bytes per chunk).
        # For L>8, merge as Counter[bytes] (FNV pipeline needs bytes for
        # hash collision resolution via position recovery).
        use_int_keys = L <= 8
        level_counts: Counter[bytes] | dict[int, int] = {} if use_int_keys else Counter()
        total_valid = 0

        # Pre-build anti-mono lookup structure ONCE per level (reused across chunks)
        lut_obj = None
        if prev_survived_set is not None:
            lut_obj = _build_anti_mono_lut(prev_survived_set, L - 1)

        n_chunks = (n_end + chunk_positions - 1) // chunk_positions
        for chunk_idx, chunk_start in enumerate(range(0, n_end, chunk_positions)):
            chunk_len = min(chunk_positions, n_end - chunk_start)

            # Read chunk directly into numpy array
            corpus_fh.seek(chunk_start)
            chunk = np.fromfile(corpus_fh, dtype=np.uint8, count=chunk_len + L - 1)

            # Valid = no SEP byte inside the L-gram
            valid = np.empty(chunk_len, dtype=np.bool_)
            _sep_check(chunk, chunk_len, L, valid)

            # Anti-monotonicity
            if lut_obj is not None:
                valid = _anti_mono_chunk(chunk, chunk_len, L, valid, lut_obj)

            nv = int(valid.sum())
            total_valid += nv
            if nv == 0:
                continue

            # Count L-grams
            result = _count_chunk(chunk, chunk_len, L, valid)

            if use_int_keys:
                # L≤8: result is (unique_packed, counts) numpy arrays
                unique_arr, count_arr = result
                # Merge into int-keyed dict (fast: no bytes creation)
                for i in range(len(unique_arr)):
                    k = int(unique_arr[i])
                    level_counts[k] = level_counts.get(k, 0) + int(count_arr[i])
            else:
                # L>8: result is Counter[bytes]
                level_counts.update(result)

            # Progress every 25% of chunks
            if n_chunks >= 8 and (chunk_idx + 1) % max(1, n_chunks // 4) == 0:
                pct = 100 * (chunk_idx + 1) / n_chunks
                elapsed = time.time() - t_level
                print(f"    L={L:2d}: chunk {chunk_idx+1}/{n_chunks} ({pct:.0f}%, {elapsed:.0f}s)")

        del lut_obj  # free bitmap / hash arrays between levels

        # Apply global min_freq filter and convert int keys to bytes
        if use_int_keys:
            level: dict[bytes, int] = {}
            for k, v in level_counts.items():
                if v >= min_freq:
                    level[k.to_bytes(L, "big")] = v
        else:
            level = {k: v for k, v in level_counts.items() if v >= min_freq}
        n_surv = len(level)
        elapsed = time.time() - t_level
        print(
            f"  L={L:2d}: {total_valid:>14,} valid positions "
            f"-> {n_surv:>8,} survivors  ({elapsed:.1f}s)"
        )

        if n_surv == 0:
            break

        all_candidates.update(level)
        prev_survived_set = set(level.keys())

        # Incremental save after each level (crash recovery)
        if save_path is not None:
            save_candidates(all_candidates, save_path, prev_survived_set, L)

    corpus_fh.close()  # release file handle

    # Final save (includes survived set from last level for resumption)
    if save_path is not None:
        save_candidates(all_candidates, save_path, prev_survived_set, max_len)

    # Final UTF-8 validity filter
    valid_candidates: dict[bytes, int] = {}
    rejected = 0
    for key, freq in all_candidates.items():
        try:
            key.decode("utf-8")
            valid_candidates[key] = freq
        except UnicodeDecodeError:
            rejected += 1
    if rejected:
        print(
            f"  UTF-8 filter: kept {len(valid_candidates):,}, "
            f"rejected {rejected:,} non-UTF-8"
        )

    total_elapsed = time.time() - t_total
    print(f"  Candidate generation: {total_elapsed:.0f}s total")
    return valid_candidates


def _build_anti_mono_lut(survived_set: set[bytes], L_prev: int):
    """Pre-build a lookup structure for anti-monotonicity checks.

    Called ONCE per level, then reused across all chunks.  Returns an
    opaque object that ``_anti_mono_chunk`` can use for O(1) lookups.

    Strategy — ALL levels use O(1) bitmap lookups:
        L_prev ≤ 3  →  boolean ndarray LUT (≤ 16M entries, exact)
        L_prev = 4  →  512 MB bitmap over uint32 (exact, no hash collisions)
        L_prev 5-8  →  512 MB bitmap via XOR-folded uint64 → uint32
                        (tiny FP rate ≈ m/2^32, zero FN — safe for anti-mono)
        L_prev ≥ 9  →  512 MB bitmap via XOR-folded FNV-1a → uint32
                        (same FP/FN properties)

    False positives just keep a few extra positions that counting will filter.
    False negatives would lose candidates — but XOR folding is deterministic
    so there are zero FN (every survived key maps to a set bit).
    """
    if L_prev <= 3:
        max_hash = 1 << (8 * L_prev)
        lut = np.zeros(max_hash, dtype=np.bool_)
        for key in survived_set:
            lut[int.from_bytes(key, "big")] = True
        return ("lut", lut, L_prev)

    elif L_prev == 4:
        # Bitmap: 2^32 bits = 512 MB.  Exact — no hash folding needed.
        bitmap = np.zeros(1 << 29, dtype=np.uint8)  # 2^32 / 8 = 2^29 bytes
        for key in survived_set:
            val = int.from_bytes(key, "big")
            bitmap[val >> 3] |= np.uint8(1 << (val & 7))
        return ("bitmap32", bitmap, L_prev)

    elif L_prev <= 8:
        # Pack into uint64, XOR-fold to uint32, then bitmap
        bitmap = np.zeros(1 << 29, dtype=np.uint8)
        for key in survived_set:
            h64 = int.from_bytes(key, "big")
            h32 = (h64 ^ (h64 >> 32)) & 0xFFFFFFFF
            bitmap[h32 >> 3] |= np.uint8(1 << (h32 & 7))
        return ("bitmap_fold64", bitmap, L_prev)

    else:
        # FNV-1a → uint64, XOR-fold to uint32, then bitmap
        bitmap = np.zeros(1 << 29, dtype=np.uint8)
        for key in survived_set:
            h_val = int(_FNV_BASIS)
            for b in key:
                h_val = ((h_val ^ b) * int(_FNV_PRIME)) & 0xFFFFFFFFFFFFFFFF
            h32 = (h_val ^ (h_val >> 32)) & 0xFFFFFFFF
            bitmap[h32 >> 3] |= np.uint8(1 << (h32 & 7))
        return ("bitmap_fnv", bitmap, L_prev)


def _bitmap_lookup(bitmap: np.ndarray, h32: np.ndarray) -> np.ndarray:
    """O(1) bitmap lookup: returns bool array of which h32 values are set."""
    byte_idx = (h32 >> np.uint32(3)).astype(np.int64)
    bit_mask = np.uint8(1) << (h32 & np.uint32(7)).astype(np.uint8)
    return (bitmap[byte_idx] & bit_mask) > 0


def _anti_mono_chunk(
    chunk: np.ndarray,
    chunk_len: int,
    L: int,
    valid: np.ndarray,
    lut_obj,
) -> np.ndarray:
    """Apply anti-monotonicity within a chunk using a pre-built bitmap/LUT.

    Checks that both the (L-1)-prefix and (L-1)-suffix of each L-gram at
    valid positions are in the survived set.  Returns a masked ``valid``.

    All paths are O(1) per position — no searchsorted or np.isin.
    ``lut_obj`` is created by ``_build_anti_mono_lut`` once per level.
    """
    L_prev = L - 1
    n_check = chunk_len + 1
    kind = lut_obj[0]

    if kind == "lut":
        lut = lut_obj[1]
        h = np.empty(n_check, dtype=np.uint32)
        _pack_bytes_to_u32(chunk, n_check, L_prev, h)
        survived = lut[h]

    elif kind == "bitmap32":
        bitmap = lut_obj[1]
        h = np.empty(n_check, dtype=np.uint32)
        _pack_bytes_to_u32(chunk, n_check, 4, h)
        survived = _bitmap_lookup(bitmap, h)

    elif kind == "bitmap_fold64":
        bitmap = lut_obj[1]
        # Numba JIT: pack L_prev bytes → uint64, XOR-fold → uint32
        h32 = np.empty(n_check, dtype=np.uint32)
        _pack_fold64_to_u32(chunk, n_check, L_prev, h32)
        survived = _bitmap_lookup(bitmap, h32)

    elif kind == "bitmap_fnv":
        bitmap = lut_obj[1]
        # Numba JIT FNV-1a hash, XOR-folded to uint32 for bitmap lookup
        h32 = np.empty(n_check, dtype=np.uint32)
        _fnv_anti_mono_hash(chunk, n_check, L_prev, h32)
        survived = _bitmap_lookup(bitmap, h32)

    return valid & survived[:chunk_len] & survived[1 : chunk_len + 1]


def _count_chunk(
    chunk: np.ndarray,
    chunk_len: int,
    L: int,
    valid: np.ndarray,
) -> Counter | tuple[np.ndarray, np.ndarray]:
    """Count L-grams at valid positions within a single chunk.

    For L ≤ 8: returns (unique_packed, counts) as numpy arrays to avoid
    creating millions of Python bytes objects per chunk.  The caller merges
    these as integer dicts and converts to bytes only once at the end.

    For L > 8: returns a Counter[bytes] (FNV hash pipeline with position
    recovery — bytes are needed for hash collision resolution).
    """
    if L <= 4:
        h = np.empty(chunk_len, dtype=np.uint32)
        _pack_bytes_to_u32(chunk, chunk_len, L, h)
        valid_h = h[valid]
        if len(valid_h) == 0:
            return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.int64)
        unique, ucounts = np.unique(valid_h, return_counts=True)
        return unique, ucounts.astype(np.int64)

    elif L <= 8:
        h = np.empty(chunk_len, dtype=np.uint64)
        _pack_bytes_to_u64(chunk, chunk_len, L, h)
        valid_h = h[valid]
        if len(valid_h) == 0:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
        unique, ucounts = np.unique(valid_h, return_counts=True)
        return unique, ucounts.astype(np.int64)

    else:
        # Vectorized FNV-1a hash for L ≥ 9
        return _count_chunk_fnv(chunk, chunk_len, L, valid)


# FNV-1a constants (64-bit)
_FNV_PRIME = np.uint64(1099511628211)
_FNV_BASIS = np.uint64(14695981039346656037)
_FNV_PRIME_INT = np.uint64(1099511628211)
_FNV_BASIS_INT = np.uint64(14695981039346656037)


# ---------------------------------------------------------------------------
# Numba JIT-compiled FNV-1a hash kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fnv_hash_array(chunk, chunk_len, L, out):
    """Compute FNV-1a hash for each L-gram starting at positions 0..chunk_len-1.

    Single pass over the data — no temporary arrays, ~10-50x faster than numpy.
    ``out`` is a pre-allocated uint64 array of length chunk_len.
    """
    prime = np.uint64(1099511628211)
    basis = np.uint64(14695981039346656037)
    for i in range(chunk_len):
        h = basis
        for j in range(L):
            h = (h ^ np.uint64(chunk[i + j])) * prime
        out[i] = h


@njit(cache=True, parallel=True)
def _fnv_hash_array_parallel(chunk, chunk_len, L, out):
    """Parallel version — uses prange for multi-core FNV hashing."""
    prime = np.uint64(1099511628211)
    basis = np.uint64(14695981039346656037)
    for i in prange(chunk_len):
        h = basis
        for j in range(L):
            h = (h ^ np.uint64(chunk[i + j])) * prime
        out[i] = h


@njit(cache=True, parallel=True)
def _pack_fold64_to_u32(chunk, n_check, L_prev, out):
    """Pack L_prev bytes into uint64, XOR-fold to uint32.  For bitmap_fold64."""
    for i in prange(n_check):
        h = np.uint64(0)
        for j in range(L_prev):
            h |= np.uint64(chunk[i + j]) << np.uint64(8 * (L_prev - 1 - j))
        out[i] = np.uint32(h ^ (h >> np.uint64(32)))


@njit(cache=True, parallel=True)
def _sep_check(chunk, chunk_len, L, out):
    """Check for SEP (0x00) byte within each L-gram.  out[i] = True if no SEP."""
    for i in prange(chunk_len):
        ok = True
        for j in range(L):
            if chunk[i + j] == 0:
                ok = False
                break
        out[i] = ok


@njit(cache=True, parallel=True)
def _fnv_anti_mono_hash(chunk, n_check, L_prev, out):
    """Compute FNV-1a hash for anti-monotonicity check positions.

    Computes hash of L_prev-grams at positions 0..n_check-1.
    XOR-folds to uint32 for bitmap lookup.
    """
    prime = np.uint64(1099511628211)
    basis = np.uint64(14695981039346656037)
    for i in prange(n_check):
        h = basis
        for j in range(L_prev):
            h = (h ^ np.uint64(chunk[i + j])) * prime
        # XOR-fold 64-bit → 32-bit
        out[i] = np.uint32(h ^ (h >> np.uint64(32)))


@njit(cache=True, parallel=True)
def _pack_bytes_to_u64(chunk, chunk_len, L, out):
    """Pack L bytes (L≤8) into uint64 at each position.  Replaces numpy loop."""
    for i in prange(chunk_len):
        h = np.uint64(0)
        for j in range(L):
            h |= np.uint64(chunk[i + j]) << np.uint64(8 * (L - 1 - j))
        out[i] = h


@njit(cache=True, parallel=True)
def _pack_bytes_to_u32(chunk, chunk_len, L, out):
    """Pack L bytes (L≤4) into uint32 at each position.  Replaces numpy loop."""
    for i in prange(chunk_len):
        h = np.uint32(0)
        for j in range(L):
            h |= np.uint32(chunk[i + j]) << np.uint32(8 * (L - 1 - j))
        out[i] = h


@njit(cache=True)
def _count_sorted_hashes(sorted_h, sorted_idx, out_hash, out_pos, out_count):
    """Count runs in a sorted hash array and record first position.

    Returns the number of unique hashes found (n_unique).
    Results are stored in pre-allocated out_hash/out_pos/out_count arrays.

    This replaces np.unique(return_index=True, return_counts=True) and is
    ~2-3x faster because it avoids Python object creation overhead.
    """
    n = len(sorted_h)
    if n == 0:
        return 0
    n_unique = 0
    out_hash[0] = sorted_h[0]
    out_pos[0] = sorted_idx[0]
    out_count[0] = 1
    for i in range(1, n):
        if sorted_h[i] != sorted_h[i - 1]:
            n_unique += 1
            out_hash[n_unique] = sorted_h[i]
            out_pos[n_unique] = sorted_idx[i]
            out_count[n_unique] = 1
        else:
            out_count[n_unique] += 1
    return n_unique + 1


@njit(cache=True)
def _hashtable_count(valid_h, indices, n_valid, table_keys, table_counts,
                     table_first_pos, table_size_u64):
    """Open-addressing hash table: count unique uint64 keys and record first pos.

    Inserts n_valid (hash, position) pairs into a pre-allocated open-addressing
    table.  Linear probing with ~50% load factor gives ~1.5 probes/insert on
    average → O(N) total vs O(N log N) for sort-based approaches.

    Returns the number of occupied slots.
    """
    n_occupied = np.int64(0)
    for i in range(n_valid):
        key = valid_h[i]
        # Fibonacci hashing for better distribution than modulo
        slot = np.int64(np.uint64(key * np.uint64(11400714819323198485)) >> np.uint64(64 - 30))
        slot = slot % np.int64(table_size_u64)
        while True:
            if table_counts[slot] == 0:
                # Empty slot — insert new entry
                table_keys[slot] = key
                table_counts[slot] = 1
                table_first_pos[slot] = indices[i]
                n_occupied += 1
                break
            elif table_keys[slot] == key:
                # Existing entry — increment count
                table_counts[slot] += 1
                break
            else:
                # Collision — linear probe
                slot += 1
                if slot >= np.int64(table_size_u64):
                    slot = 0
    return n_occupied


def _count_chunk_fnv(
    chunk: np.ndarray,
    chunk_len: int,
    L: int,
    valid: np.ndarray,
) -> Counter:
    """Count L-grams (L>8) using Numba JIT FNV-1a hash + JIT hash table.

    Pipeline: JIT hash → extract valid → JIT hash table (O(N)) → recover bytes.

    The JIT hash table replaces np.unique (which sorts, O(N log N)) with
    open-addressing linear probing at ~50% load factor.  For ~80M valid
    positions, this is ~13× faster (1.2s vs 16s per chunk).

    Collision probability ≈ n² / 2^64 ≈ 0 for practical n-gram counts.
    """
    counts: Counter[bytes] = Counter()

    # Compute FNV-1a hash at each position using Numba JIT kernel
    h = np.empty(chunk_len, dtype=np.uint64)
    _fnv_hash_array_parallel(chunk, chunk_len, L, h)

    # Extract valid hashes and their original indices
    indices = np.where(valid)[0]
    valid_h = h[indices]
    del h
    n_valid = len(valid_h)
    if n_valid == 0:
        return counts

    # Allocate hash table at ~50% load factor (next power of 2 above 2*n_valid)
    table_size = 1
    while table_size < 2 * n_valid:
        table_size <<= 1
    table_keys = np.empty(table_size, dtype=np.uint64)
    table_counts = np.zeros(table_size, dtype=np.int64)
    table_first_pos = np.empty(table_size, dtype=np.int64)

    # JIT hash table insertion — O(N) amortized
    _hashtable_count(valid_h, indices, n_valid, table_keys, table_counts,
                     table_first_pos, np.uint64(table_size))
    del valid_h, indices

    # Extract occupied slots and recover byte sequences
    occupied = np.where(table_counts > 0)[0]
    for slot_idx in range(len(occupied)):
        slot = occupied[slot_idx]
        pos = int(table_first_pos[slot])
        cnt = int(table_counts[slot])
        key = bytes(chunk[pos : pos + L])
        counts[key] = cnt

    return counts


# ---------------------------------------------------------------------------
# Vocabulary Builder (greedy selection with DP re-scoring)
# ---------------------------------------------------------------------------

def _is_within_word(token_bytes: bytes) -> bool:
    """Check if a token does NOT contain an interior ▁ (U+2581 = E2 96 81).

    A leading ▁ at position 0 is allowed (word-initial subword).
    Cross-word tokens (▁ at position 1+) return False.
    """
    if len(token_bytes) < 4:
        return True
    # Search for E2 96 81 starting at byte index 1 (position 0 is allowed)
    marker = b"\xe2\x96\x81"
    pos = token_bytes.find(marker, 1)
    return pos == -1


def build_vocabulary(
    candidates: dict[bytes, int],
    corpus_bytes: bytes,
    num_tokens: int = 764,
    rescore_interval: int = 50,
    rescore_sample_bytes: int = 10_000_000,
    max_token_len: int = 32,
    phase1_fraction: float = 0.0,
) -> list[bytes]:
    """Select tokens greedily by (len-1)*freq score, re-scoring periodically.

    After every rescore_interval selections, DP-encode a sample with the current
    vocabulary and re-extract candidates from multi-token spans. This captures
    how selected tokens change the value of remaining candidates.

    If phase1_fraction > 0, uses a two-phase SuperBPE-style curriculum:
      Phase 1: only within-word candidates (no interior ▁)
      Phase 2: all candidates (cross-word tokens become eligible)

    Returns: list of selected token byte sequences in selection order.
    """
    trie = make_base_trie()
    selected: list[bytes] = []
    selected_set: set[bytes] = set()

    phase1_count = round(num_tokens * phase1_fraction) if phase1_fraction > 0 else 0

    # Working candidates: bytes -> freq
    working = dict(candidates)

    def _build_heap(work: dict[bytes, int], within_word_only: bool) -> list[tuple[int, bytes]]:
        h: list[tuple[int, bytes]] = []
        for tb, freq in work.items():
            if within_word_only and not _is_within_word(tb):
                continue
            score = (len(tb) - 1) * freq
            if score > 0:
                heapq.heappush(h, (-score, tb))
        return h

    in_phase1 = phase1_count > 0
    heap = _build_heap(working, within_word_only=in_phase1)

    print(f"  Starting vocabulary selection: {num_tokens} tokens to select")
    if phase1_count > 0:
        within_word = sum(1 for tb in candidates if _is_within_word(tb))
        print(f"  Two-phase: phase1={phase1_count} within-word, phase2={num_tokens - phase1_count} all")
        print(f"  Within-word candidates: {within_word}/{len(candidates)}")
    print(f"  Initial candidates in heap: {len(heap):,}")

    # JIT warm-up: build flat trie and do one tiny encode to trigger compilation
    flat = FlatTrie(trie)
    dp_encode(b"\xe2\x96\x81x", trie, flat=flat)  # warm up JIT
    print(f"  JIT DP encoder warmed up ({flat.n_nodes} trie nodes)")

    for step in range(num_tokens):
        # Check phase transition
        if in_phase1 and step >= phase1_count:
            in_phase1 = False
            print(f"  --- Phase 2 start (step {step}) — all candidates now eligible ---")
            heap = _build_heap(working, within_word_only=False)

        # Pop best valid candidate
        chosen = None
        while heap:
            neg_score, tb = heapq.heappop(heap)
            if tb in working and tb not in selected_set:
                if in_phase1 and not _is_within_word(tb):
                    continue  # skip cross-word in phase 1
                chosen = tb
                break

        if chosen is None:
            if in_phase1:
                print(f"  Phase 1 exhausted at step {step}. Switching to phase 2 early.")
                in_phase1 = False
                heap = _build_heap(working, within_word_only=False)
                # Retry
                while heap:
                    neg_score, tb = heapq.heappop(heap)
                    if tb in working and tb not in selected_set:
                        chosen = tb
                        break
            if chosen is None:
                print(f"  WARNING: ran out of candidates at step {step}")
                break

        # Add to vocabulary
        token_id = LEARNED_OFFSET + step
        trie.insert(chosen, token_id)
        selected.append(chosen)
        selected_set.add(chosen)
        working.pop(chosen, None)

        score = -neg_score
        phase_tag = "P1" if in_phase1 else "P2"
        if step < 5 or (step + 1) % 50 == 0 or step == num_tokens - 1:
            piece_repr = chosen.decode("utf-8", errors="replace").encode(
                "ascii", errors="backslashreplace"
            ).decode("ascii")
            print(f"  [{step+1:>4}/{num_tokens}] score={score:>12,} "
                  f"len={len(chosen)} {phase_tag} piece={piece_repr}")

        # Periodic re-scoring
        if (step + 1) % rescore_interval == 0 and step + 1 < num_tokens:
            print(f"  --- Re-scoring (step {step+1}) ---")
            # Rebuild flat trie with newly added tokens
            flat = FlatTrie(trie)
            working = _rescore(
                trie, selected, selected_set, corpus_bytes,
                rescore_sample_bytes, max_token_len,
                original_candidates=candidates,
                corpus_total_bytes=len(corpus_bytes),
                flat=flat,
            )
            # Rebuild heap with phase filter
            heap = _build_heap(working, within_word_only=in_phase1)
            print(f"  Re-scored: {len(heap):,} candidates remaining")

    # Summary
    within_word_selected = sum(1 for tb in selected if _is_within_word(tb))
    cross_word_selected = len(selected) - within_word_selected
    print(f"  Final: {within_word_selected} within-word ({within_word_selected*100/max(len(selected),1):.1f}%), "
          f"{cross_word_selected} cross-word ({cross_word_selected*100/max(len(selected),1):.1f}%)")

    return selected


def build_vocabulary_augment(
    candidates: dict[bytes, int],
    corpus_bytes: bytes,
    base_tokens: list[bytes],
    num_new_tokens: int = 79,
    rescore_interval: int = 10,
    rescore_sample_bytes: int = 50_000_000,
    max_token_len: int = 32,
) -> list[bytes]:
    """Add new tokens on top of an existing base vocabulary.

    Starts with base_tokens already in the trie, then greedily selects
    num_new_tokens from candidates (excluding base tokens).  Uses DP
    re-scoring to capture how each new token interacts with the existing
    vocabulary.

    Returns: base_tokens + newly selected tokens (in selection order).
    """
    trie = make_base_trie()
    selected_set: set[bytes] = set()

    # Install base vocabulary
    for idx, tb in enumerate(base_tokens):
        trie.insert(tb, LEARNED_OFFSET + idx)
        selected_set.add(tb)

    # Filter candidates: remove those already in base vocab, single bytes
    working = {k: v for k, v in candidates.items()
               if k not in selected_set and len(k) >= 2}

    # Initial heap
    heap: list[tuple[int, bytes]] = []
    for tb, freq in working.items():
        score = (len(tb) - 1) * freq
        if score > 0:
            heapq.heappush(heap, (-score, tb))

    print(f"  Augmenting {len(base_tokens)} base tokens with {num_new_tokens} new tokens")
    print(f"  Candidate pool: {len(heap):,}")

    # JIT warm-up
    flat = FlatTrie(trie)
    dp_encode(b"\xe2\x96\x81x", trie, flat=flat)
    print(f"  JIT DP encoder warmed up ({flat.n_nodes} trie nodes)")

    new_selected: list[bytes] = []
    next_id = LEARNED_OFFSET + len(base_tokens)

    for step in range(num_new_tokens):
        chosen = None
        while heap:
            neg_score, tb = heapq.heappop(heap)
            if tb in working and tb not in selected_set:
                chosen = tb
                break

        if chosen is None:
            print(f"  WARNING: ran out of candidates at step {step}")
            break

        trie.insert(chosen, next_id + step)
        new_selected.append(chosen)
        selected_set.add(chosen)
        working.pop(chosen, None)

        score = -neg_score
        if step < 5 or (step + 1) % 10 == 0 or step == num_new_tokens - 1:
            piece_repr = chosen.decode("utf-8", errors="replace").encode(
                "ascii", errors="backslashreplace"
            ).decode("ascii")
            print(f"  [{step+1:>4}/{num_new_tokens}] score={score:>12,} "
                  f"len={len(chosen)} piece={piece_repr}")

        # Periodic re-scoring
        if (step + 1) % rescore_interval == 0 and step + 1 < num_new_tokens:
            print(f"  --- Re-scoring (step {step+1}) ---")
            flat = FlatTrie(trie)
            working = _rescore(
                trie, list(base_tokens) + new_selected, selected_set,
                corpus_bytes, rescore_sample_bytes, max_token_len,
                original_candidates=candidates,
                corpus_total_bytes=len(corpus_bytes),
                flat=flat,
            )
            heap = []
            for tb, freq in working.items():
                s = (len(tb) - 1) * freq
                if s > 0:
                    heapq.heappush(heap, (-s, tb))
            print(f"  Re-scored: {len(heap):,} candidates remaining")

    print(f"  Selected {len(new_selected)} new tokens")
    return list(base_tokens) + new_selected


def _rescore(
    trie: ByteTrie,
    selected: list[bytes],
    selected_set: set[bytes],
    corpus_bytes: bytes,
    sample_size: int,
    max_token_len: int,
    original_candidates: dict[bytes, int],
    corpus_total_bytes: int,
    flat: FlatTrie | None = None,
) -> dict[bytes, int]:
    """Re-tokenize a sample and update candidate frequencies.

    After DP-encoding with the current vocabulary, count how often each
    *original* candidate appears as a multi-token span.  This captures how
    selected tokens change the marginal value of remaining candidates.

    Key design: only count spans that exist in ``original_candidates``.
    This prevents the candidate set from exploding with arbitrary long
    multi-word phrases that score high due to large (len-1) but have
    diminishing marginal value.

    Frequencies are scaled from sample to full corpus proportionally.

    Uses JIT-accelerated DP encoder when ``flat`` is provided.
    """
    sample = corpus_bytes[:sample_size]
    docs = sample.split(bytes([SEP]))
    scale = corpus_total_bytes / max(len(sample), 1)

    # Only count candidates from the original set (minus already selected)
    eligible = set(original_candidates.keys()) - selected_set

    span_counts: Counter[bytes] = Counter()
    # Max tokens in a candidate span — kept small for performance.
    max_span_tokens = 4
    # Cap byte length based on what's actually in the candidate pool.
    # Most valuable candidates at this stage are 2-6 bytes.
    max_span_bytes = min(max_token_len, 24)

    for doc in docs:
        if len(doc) < 2:
            continue
        tokens = dp_encode(doc, trie, flat=flat)
        if not tokens:
            continue

        # Reconstruct byte positions
        n_tok = len(tokens)
        positions = [0] * (n_tok + 1)
        for idx in range(n_tok):
            positions[idx + 1] = positions[idx] + trie.token_len(tokens[idx])

        # Extract multi-token spans, but only count ones in eligible set
        for start in range(n_tok):
            byte_start = positions[start]
            for end in range(start + 2, min(start + max_span_tokens + 1, n_tok + 1)):
                byte_end = positions[end]
                span_len = byte_end - byte_start
                if span_len > max_span_bytes:
                    break
                candidate = doc[byte_start:byte_end]
                if candidate in eligible:
                    span_counts[candidate] += 1

    # Build result: update frequencies for candidates found in sample,
    # keep original freq (scaled) for candidates not found in sample
    # (they may appear later in the corpus).
    result: dict[bytes, int] = {}
    for tb in eligible:
        if tb in span_counts:
            # Use observed frequency, scaled to full corpus
            result[tb] = int(span_counts[tb] * scale)
        # Candidates not observed in sample are dropped — if they don't
        # appear in the sample, they're too rare to be worth a vocab slot.

    return result


# ---------------------------------------------------------------------------
# SentencePiece Model Creation (protobuf injection)
# ---------------------------------------------------------------------------

def create_sp_model(
    selected_tokens: list[bytes],
    baseline_model_path: str,
    output_path: str,
    required_bytes: set[int] | None = None,
) -> None:
    """Create a valid SentencePiece .model file with our hybrid vocabulary.

    Uses the baseline model as a template: copies normalizer_spec and trainer_spec
    to ensure identical text normalization. Rebuilds the piece list with our
    selected tokens.

    If ``required_bytes`` is provided (a set of byte values 0-255), only those
    byte values get TYPE_BYTE fallback tokens. This enables byte_fallback=False
    mode where unused byte values are excluded from the vocabulary entirely,
    freeing slots for learned tokens. When None, all 256 byte fallbacks are
    included (byte_fallback=True behavior).
    """
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2

    # Load baseline model as template
    baseline = sp_pb2.ModelProto()
    with open(baseline_model_path, "rb") as f:
        baseline.ParseFromString(f.read())

    byte_values = sorted(required_bytes) if required_bytes is not None else list(range(256))
    num_byte_tokens = len(byte_values)
    num_fixed = 4 + num_byte_tokens  # 4 control + byte tokens

    model = sp_pb2.ModelProto()

    # Copy normalizer and trainer specs (ensures identical normalization)
    model.normalizer_spec.CopyFrom(baseline.normalizer_spec)
    if baseline.HasField("denormalizer_spec"):
        model.denormalizer_spec.CopyFrom(baseline.denormalizer_spec)
    model.trainer_spec.CopyFrom(baseline.trainer_spec)
    model.trainer_spec.vocab_size = num_fixed + len(selected_tokens)
    # Switch from BPE (type=2) to Unigram (type=1). BPE encoding relies on
    # merge hierarchies our top-down vocab doesn't have. Unigram uses Viterbi
    # decoding which naturally finds minimum-token segmentation with uniform scores.
    model.trainer_spec.model_type = 1  # UNIGRAM
    if required_bytes is not None and len(required_bytes) == 0:
        model.trainer_spec.byte_fallback = False
    else:
        # byte_fallback=True requires all 256 <0xHH> tokens
        byte_values = list(range(256))
        num_byte_tokens = 256
        num_fixed = 4 + 256
        model.trainer_spec.byte_fallback = True
        model.trainer_spec.vocab_size = num_fixed + len(selected_tokens)

    # --- Build piece list ---
    # Score strategy: all tokens get -1.0. With uniform scores, SP Unigram's
    # Viterbi maximizes Σ score = Σ(-1) = -(num_tokens), i.e. minimizes token count.
    # This matches our DP encoder's objective exactly.
    UNIFORM_SCORE = -1.0

    # ID 0: <pad> (CONTROL) — pad_id=0
    _add_piece(model, "<pad>", 0.0, TYPE_CONTROL)
    # ID 1: <s> (CONTROL) — bos_id=1
    _add_piece(model, "<s>", 0.0, TYPE_CONTROL)
    # ID 2: </s> (CONTROL) — eos_id=2
    _add_piece(model, "</s>", 0.0, TYPE_CONTROL)
    # ID 3: <unk> (UNKNOWN) — unk_id=3
    _add_piece(model, "<unk>", 0.0, TYPE_UNKNOWN)

    # Byte fallback tokens (all 256 or only required subset)
    for i in byte_values:
        _add_piece(model, f"<0x{i:02X}>", UNIFORM_SCORE, TYPE_BYTE)

    if required_bytes is not None and len(required_bytes) == 0:
        print(f"  Byte fallback: DISABLED (0 byte tokens, byte_fallback=False)")
    else:
        print(f"  Byte fallback: all 256 bytes (byte_fallback=True)")

    # Learned tokens (NORMAL)
    for rank, token_bytes in enumerate(selected_tokens):
        piece_str = token_bytes.decode("utf-8")
        _add_piece(model, piece_str, UNIFORM_SCORE, TYPE_NORMAL)

    # Write
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(model.SerializeToString())

    print(f"  SP model written: {output_path} ({len(model.pieces)} pieces, "
          f"{num_byte_tokens} byte + {len(selected_tokens)} learned)")


def _add_piece(model, piece: str, score: float, piece_type: int):
    p = model.pieces.add()
    p.piece = piece
    p.score = score
    p.type = piece_type


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_compression(
    texts: list[str],
    trie: ByteTrie,
    use_dp: bool = True,
    sp_processor=None,
    flat: FlatTrie | None = None,
) -> dict:
    """Measure tokens/byte on a set of texts.

    If use_dp=True: uses our DP encoder (optimal).  Pass ``flat`` for JIT speed.
    If sp_processor is provided and use_dp=False: uses SP greedy encoder.
    """
    total_tokens = 0
    total_bytes = 0
    for text in texts:
        raw_bytes = len(text.encode("utf-8"))
        total_bytes += raw_bytes

        if use_dp:
            normalized = normalize_text(text)
            data = normalized.encode("utf-8")
            tokens = dp_encode(data, trie, flat=flat)
            total_tokens += len(tokens)
        elif sp_processor is not None:
            ids = sp_processor.encode(text)
            total_tokens += len(ids)

    tok_per_byte = total_tokens / total_bytes if total_bytes > 0 else 0
    return {
        "tokens": total_tokens,
        "bytes": total_bytes,
        "tokens_per_byte": tok_per_byte,
        "bytes_per_token": total_bytes / total_tokens if total_tokens > 0 else 0,
    }
