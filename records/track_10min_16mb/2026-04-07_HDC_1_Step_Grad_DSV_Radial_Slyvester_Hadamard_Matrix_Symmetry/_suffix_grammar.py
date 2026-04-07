"""Suffix Grammar Table — morphological grammar learning from corpus statistics.

Learns the mapping: suffix_hypervector → grammatical context signature.

Built during Phase 3.5 in one corpus scan. At inference, provides a
suffix_grammar_score(candidate, S_p) that answers:
    "Is this candidate token's suffix grammatically consistent with the
     current semantic context S[p]?"

This is the subatomic-grounded grammar layer. It learns rules like:
    -ed  → past tense / past participle  (high score after "yesterday", "had", "was")
    -ing → present participle            (high score after "is", "are", "keep")
    -s   → plural noun OR 3rd-person sg  (disambiguated by S[p] context)
    -ly  → adverb                        (high score after verbs, adjectives)
    -er  → comparative                   (high score after "than", "more")

No hand-written grammar rules — all learned from corpus co-occurrence statistics.

Key design invariant:
    The suffix grammar score is a GATE, not a generator. It is applied AFTER
    the semantic rolling hash S[p] has identified candidate tokens. It reweights
    candidates by morphological consistency, preventing tense/number/POS confusion
    when the subatomic expansion adds morphological neighbors.

Storage: ~260 KB total
    suffix_hvs:         vocab_size × W_UINT64 × 8 bytes = 128 KB
    suffix_gram_bundle: vocab_size × W_UINT64 × 8 bytes = 128 KB
    suffix_gram_count:  vocab_size × 4 bytes             =   4 KB

Compute: one corpus scan (~5-10s for 500M tokens)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hamming similarity (local copy to avoid circular imports)
# ---------------------------------------------------------------------------

def _hamming_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Hamming similarity in [0, 1]: 1.0 = identical, 0.5 = random."""
    xor = a ^ b
    bits = int(np.unpackbits(xor.view(np.uint8)).sum())
    total = len(a) * 64
    return 1.0 - bits / total


def _hamming_sim_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Vectorised Hamming similarity: query (W,) vs matrix (N, W).

    Returns (N,) float32 similarities.
    """
    xor = query[None, :] ^ matrix                          # (N, W) uint64
    bits = np.unpackbits(xor.view(np.uint8), axis=1).sum(axis=1)  # (N,) int
    total = matrix.shape[1] * 64
    return 1.0 - bits.astype(np.float32) / float(total)


# ---------------------------------------------------------------------------
# SuffixGrammarTable
# ---------------------------------------------------------------------------

class SuffixGrammarTable:
    """Suffix-to-grammar-role table learned from corpus statistics.

    Parameters
    ----------
    vocab_size  : int — number of tokens (e.g. 1024)
    W_UINT64    : int — uint64 blocks per hypervector (e.g. 16)
    char_hv     : CharacterHypervector — for encoding suffix strings
    tokenizer   : sentencepiece model — for decoding token IDs to strings
    suffix_len  : int — number of trailing characters to use as suffix (default 3)
    sim_threshold : float — minimum suffix similarity to consider "same suffix class"
    """

    def __init__(
        self,
        vocab_size: int,
        W_UINT64: int,
        char_hv,                  # CharacterHypervector instance
        tokenizer,                # sentencepiece model
        suffix_len: int = 3,
        sim_threshold: float = 0.70,
    ) -> None:
        self.vocab_size    = vocab_size
        self.W_UINT64      = W_UINT64
        self.dim           = W_UINT64 * 64
        self.char_hv       = char_hv
        self.tokenizer     = tokenizer
        self.suffix_len    = suffix_len
        self.sim_threshold = sim_threshold

        # Precompute suffix hypervectors for all vocab tokens
        self.suffix_hvs = self._build_suffix_hvs()

        # Grammar context bundles — built from corpus in build_from_corpus()
        self.suffix_gram_bundle = np.zeros((vocab_size, W_UINT64), dtype=np.uint64)
        self.suffix_gram_count  = np.zeros(vocab_size, dtype=np.int32)

        self._built = False

    # ------------------------------------------------------------------
    # Suffix HV precomputation
    # ------------------------------------------------------------------

    def _build_suffix_hvs(self) -> np.ndarray:
        """Precompute suffix hypervectors for all vocab tokens.

        suffix_hvs[t] = char_hv.encode_string(token_str[-suffix_len:])

        Tokens sharing the same suffix (e.g. "jumped" and "walked" both
        ending in "-ed") will have high Hamming similarity.

        Returns
        -------
        (vocab_size, W_UINT64) uint64
        """
        suffix_hvs = np.zeros((self.vocab_size, self.W_UINT64), dtype=np.uint64)

        for t in range(self.vocab_size):
            try:
                token_str = self.tokenizer.decode([t])
                # Strip leading space (SentencePiece adds ▁ prefix)
                token_str = token_str.lstrip('▁').lstrip(' ')
                if not token_str:
                    token_str = ' '
                suffix = token_str[-self.suffix_len:] if len(token_str) >= self.suffix_len else token_str
                suffix_hvs[t] = self.char_hv.encode_string(suffix)
            except Exception:
                # Unknown token — leave as zeros (will have low similarity to everything)
                pass

        return suffix_hvs

    # ------------------------------------------------------------------
    # Corpus scan — build grammar context bundles
    # ------------------------------------------------------------------

    def build_from_corpus(
        self,
        tokens: np.ndarray,
        S_states_or_srh,
        srh=None,
        sem_fwd_matrix: Optional[np.ndarray] = None,
        keys: Optional[np.ndarray] = None,
        checkpoints: Optional[Dict] = None,
        time_budget_s: float = 10.0,
        chunk_size: int = 2_000_000,
        label: str = "SuffixGrammar",
    ) -> None:
        """Build suffix_gram_bundle from one corpus scan.

        For each position p, bundles S[p] into suffix_gram_bundle[tokens[p]].
        After the scan, suffix_gram_bundle[t] encodes:
            "What grammatical contexts does token t's suffix appear in?"

        Parameters
        ----------
        tokens          : (N,) uint16 — full token array
        S_states_or_srh : Either (N, W_UINT64) precomputed states OR a
                          SemanticRollingHash instance (recomputes on the fly)
        srh             : SemanticRollingHash — required if S_states_or_srh is SRH
        sem_fwd_matrix  : (vocab_size, W_UINT64) — required if recomputing S[p]
        keys            : (N,) uint64 — required if recomputing S[p]
        checkpoints     : Dict[int, np.ndarray] — required if recomputing S[p]
        time_budget_s   : float — soft wall-clock limit
        chunk_size      : int — tokens per chunk
        label           : str — log prefix
        """
        N = len(tokens)
        start = time.time()
        print(f"\n[{label}] Building suffix grammar table (N={N:,}, suffix_len={self.suffix_len})")

        # Determine how to get S[p] states
        have_precomputed = (
            isinstance(S_states_or_srh, np.ndarray)
            and S_states_or_srh.ndim == 2
            and S_states_or_srh.shape[1] == self.W_UINT64
        )

        total_processed = 0

        for chunk_start in range(0, N, chunk_size):
            if time.time() - start > time_budget_s:
                print(f"[{label}] Time budget reached at {chunk_start:,}")
                break

            chunk_end = min(chunk_start + chunk_size, N)
            chunk_n   = chunk_end - chunk_start

            # Get S[p] states for this chunk
            if have_precomputed:
                # Direct slice from precomputed array
                if chunk_end <= len(S_states_or_srh):
                    chunk_states = S_states_or_srh[chunk_start:chunk_end]
                else:
                    break
            elif srh is not None and sem_fwd_matrix is not None and checkpoints is not None:
                # Recompute on the fly
                chunk_states = srh.recompute_chunk(
                    chunk_start, chunk_end, tokens, sem_fwd_matrix, keys, checkpoints
                )
            else:
                # No S[p] available — use zeros (grammar table will be uninformative)
                chunk_states = np.zeros((chunk_n, self.W_UINT64), dtype=np.uint64)

            # Accumulate: for each position, XOR S[p] into suffix_gram_bundle[token]
            chunk_tokens = tokens[chunk_start:chunk_end].astype(np.int32)

            for i in range(chunk_n):
                t = int(chunk_tokens[i])
                if t < 0 or t >= self.vocab_size:
                    continue
                self.suffix_gram_bundle[t] ^= chunk_states[i]
                self.suffix_gram_count[t]  += 1

            total_processed += chunk_n

        elapsed = time.time() - start
        filled = int(np.sum(self.suffix_gram_count > 0))
        print(f"[{label}] Done. {total_processed:,} tokens processed in {elapsed:.2f}s "
              f"| {filled}/{self.vocab_size} suffix slots filled")
        self._built = True

    # ------------------------------------------------------------------
    # Inference: suffix grammar score
    # ------------------------------------------------------------------

    def suffix_grammar_score(self, candidate: int, S_p: np.ndarray) -> float:
        """Score: how grammatically consistent is candidate's suffix with S_p?

        Algorithm:
        1. Find all tokens with similar suffixes (suffix_sim > sim_threshold)
        2. Bundle their suffix_gram_bundle entries (XOR-reduce)
        3. Hamming similarity of bundle to S_p

        High score → candidate's suffix typically appears in contexts like S_p.
        Low score  → candidate's suffix is grammatically inconsistent with S_p.

        Parameters
        ----------
        candidate : int — token ID to score
        S_p       : (W_UINT64,) uint64 — current semantic state

        Returns
        -------
        float in [0, 1]
        """
        if not self._built:
            return 0.5   # uninformative prior

        if candidate < 0 or candidate >= self.vocab_size:
            return 0.5

        # Step 1: find tokens with similar suffixes
        cand_suffix_hv = self.suffix_hvs[candidate]
        suffix_sims = _hamming_sim_batch(cand_suffix_hv, self.suffix_hvs)  # (vocab_size,)
        similar_mask = suffix_sims > self.sim_threshold

        if not np.any(similar_mask):
            # No similar suffixes found — use candidate's own bundle
            similar_mask[candidate] = True

        # Step 2: bundle grammar signatures of similar-suffix tokens
        similar_indices = np.where(similar_mask)[0]
        gram_bundle = np.zeros(self.W_UINT64, dtype=np.uint64)
        for idx in similar_indices:
            if self.suffix_gram_count[idx] > 0:
                gram_bundle ^= self.suffix_gram_bundle[idx]

        # If bundle is all zeros (no data), return neutral score
        if not np.any(gram_bundle):
            return 0.5

        # Step 3: similarity of grammar bundle to current S_p
        score = _hamming_sim(gram_bundle, S_p)
        return float(score)

    def batch_suffix_grammar_scores(
        self, candidates: np.ndarray, S_p: np.ndarray
    ) -> np.ndarray:
        """Vectorised suffix grammar scoring for multiple candidates.

        Parameters
        ----------
        candidates : (k,) int — token IDs to score
        S_p        : (W_UINT64,) uint64 — current semantic state

        Returns
        -------
        (k,) float32 — grammar scores for each candidate
        """
        scores = np.full(len(candidates), 0.5, dtype=np.float32)
        for i, cand in enumerate(candidates):
            scores[i] = self.suffix_grammar_score(int(cand), S_p)
        return scores

    # ------------------------------------------------------------------
    # Subatomic gradient: morphological neighbors
    # ------------------------------------------------------------------

    def subatomic_gradient_neighbors(
        self,
        token_id: int,
        max_neighbors: int = 16,
        min_sim: float = 0.65,
    ) -> List[Tuple[int, float]]:
        """Find tokens that are morphologically similar via suffix structure.

        Uses one-step gradient in suffix hypervector space: finds tokens
        whose suffix HV is most similar to token_id's suffix HV.

        This is NOT a bit-flip search in token ID space — it searches in
        the character-encoding space, finding morphological variants like:
            "jump" → "jumps", "jumped", "jumping", "jumper"

        Parameters
        ----------
        token_id      : int — source token
        max_neighbors : int — maximum neighbors to return
        min_sim       : float — minimum suffix similarity threshold

        Returns
        -------
        List of (neighbor_token_id, similarity) sorted by similarity descending
        """
        if token_id < 0 or token_id >= self.vocab_size:
            return []

        source_suffix_hv = self.suffix_hvs[token_id]
        sims = _hamming_sim_batch(source_suffix_hv, self.suffix_hvs)  # (vocab_size,)

        # Exclude the token itself
        sims[token_id] = 0.0

        # Filter by minimum similarity
        valid_mask = sims >= min_sim
        if not np.any(valid_mask):
            return []

        valid_indices = np.where(valid_mask)[0]
        valid_sims    = sims[valid_indices]

        # Sort by similarity descending
        order = np.argsort(valid_sims)[::-1]
        top_indices = valid_indices[order[:max_neighbors]]
        top_sims    = valid_sims[order[:max_neighbors]]

        return [(int(idx), float(sim)) for idx, sim in zip(top_indices, top_sims)]

    # ------------------------------------------------------------------
    # Grammar-gated subatomic expansion
    # ------------------------------------------------------------------

    def grammar_gated_expansion(
        self,
        candidates: np.ndarray,
        S_p: np.ndarray,
        grammar_threshold: float = 0.52,
        max_neighbors_per_candidate: int = 8,
    ) -> List[int]:
        """Expand candidates with morphological neighbors, gated by grammar score.

        This is the safe way to use the subatomic table:
        1. For each candidate, find morphological neighbors
        2. Only add neighbors that ALSO pass the grammar gate
        3. Return expanded candidate set

        This prevents tense/number/POS confusion: "jumped" is only added
        if sem_bwd confirms it's grammatically consistent with S_p.

        Parameters
        ----------
        candidates         : (k,) int — initial semantic candidates
        S_p                : (W_UINT64,) uint64 — current semantic state
        grammar_threshold  : float — minimum grammar score to include neighbor
        max_neighbors_per_candidate : int

        Returns
        -------
        List[int] — expanded candidate set (original + grammar-consistent neighbors)
        """
        expanded = set(int(c) for c in candidates)

        for cand in candidates:
            neighbors = self.subatomic_gradient_neighbors(
                int(cand), max_neighbors=max_neighbors_per_candidate
            )
            for neighbor_tok, morph_sim in neighbors:
                if neighbor_tok in expanded:
                    continue
                gram_score = self.suffix_grammar_score(neighbor_tok, S_p)
                if gram_score > grammar_threshold:
                    expanded.add(neighbor_tok)

        return list(expanded)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def top_grammar_contexts(self, token_id: int, codebook: np.ndarray,
                              top_k: int = 5) -> List[Tuple[int, float]]:
        """Return the top-k tokens whose codebook vectors best match this
        token's grammar context bundle.

        Useful for debugging: shows what grammatical role the table has
        learned for a given token's suffix.

        Parameters
        ----------
        token_id : int
        codebook : (vocab_size, W_UINT64) uint64
        top_k    : int

        Returns
        -------
        List of (token_id, similarity) sorted by similarity descending
        """
        if token_id < 0 or token_id >= self.vocab_size:
            return []

        bundle = self.suffix_gram_bundle[token_id]
        if not np.any(bundle):
            return []

        sims = _hamming_sim_batch(bundle, codebook)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    def summary(self) -> str:
        """Return a human-readable summary of the table state."""
        filled = int(np.sum(self.suffix_gram_count > 0))
        total_obs = int(np.sum(self.suffix_gram_count))
        return (
            f"SuffixGrammarTable("
            f"vocab={self.vocab_size}, "
            f"suffix_len={self.suffix_len}, "
            f"filled={filled}/{self.vocab_size}, "
            f"total_observations={total_obs:,}, "
            f"built={self._built})"
        )
