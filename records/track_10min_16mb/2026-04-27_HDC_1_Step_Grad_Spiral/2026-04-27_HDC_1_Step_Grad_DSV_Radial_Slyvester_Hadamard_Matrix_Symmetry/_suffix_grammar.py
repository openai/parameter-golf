
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

def _hamming_sim(a: np.ndarray, b: np.ndarray) -> float:
    xor = a ^ b
    bits = int(np.unpackbits(xor.view(np.uint8)).sum())
    total = len(a) * 64
    return 1.0 - bits / total

def _hamming_sim_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    xor = query[None, :] ^ matrix
    bits = np.unpackbits(xor.view(np.uint8), axis=1).sum(axis=1)
    total = matrix.shape[1] * 64
    return 1.0 - bits.astype(np.float32) / float(total)

class SuffixGrammarTable:

    def __init__(
        self,
        vocab_size: int,
        W_UINT64: int,
        char_hv,
        tokenizer,
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

        self.suffix_hvs = self._build_suffix_hvs()

        self.suffix_gram_bundle = np.zeros((vocab_size, W_UINT64), dtype=np.uint64)
        self.suffix_gram_count  = np.zeros(vocab_size, dtype=np.int32)

        self._built = False

    def _build_suffix_hvs(self) -> np.ndarray:
        suffix_hvs = np.zeros((self.vocab_size, self.W_UINT64), dtype=np.uint64)

        for t in range(self.vocab_size):
            try:
                token_str = self.tokenizer.decode([t])
                token_str = token_str.lstrip('▁').lstrip(' ')
                if not token_str:
                    token_str = ' '
                suffix = token_str[-self.suffix_len:] if len(token_str) >= self.suffix_len else token_str
                suffix_hvs[t] = self.char_hv.encode_string(suffix)
            except Exception:
                pass

        return suffix_hvs

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
        N = len(tokens)
        start = time.time()
        print(f"\n[{label}] Building suffix grammar table (N={N:,}, suffix_len={self.suffix_len})")

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

            if have_precomputed:
                if chunk_end <= len(S_states_or_srh):
                    chunk_states = S_states_or_srh[chunk_start:chunk_end]
                else:
                    break
            elif srh is not None and sem_fwd_matrix is not None and checkpoints is not None:
                chunk_states = srh.recompute_chunk(
                    chunk_start, chunk_end, tokens, sem_fwd_matrix, keys, checkpoints
                )
            else:
                chunk_states = np.zeros((chunk_n, self.W_UINT64), dtype=np.uint64)

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

    def suffix_grammar_score(self, candidate: int, S_p: np.ndarray) -> float:
        if not self._built:
            return 0.5

        if candidate < 0 or candidate >= self.vocab_size:
            return 0.5

        cand_suffix_hv = self.suffix_hvs[candidate]
        suffix_sims = _hamming_sim_batch(cand_suffix_hv, self.suffix_hvs)
        similar_mask = suffix_sims > self.sim_threshold

        if not np.any(similar_mask):
            similar_mask[candidate] = True

        similar_indices = np.where(similar_mask)[0]
        gram_bundle = np.zeros(self.W_UINT64, dtype=np.uint64)
        for idx in similar_indices:
            if self.suffix_gram_count[idx] > 0:
                gram_bundle ^= self.suffix_gram_bundle[idx]

        if not np.any(gram_bundle):
            return 0.5

        score = _hamming_sim(gram_bundle, S_p)
        return float(score)

    def batch_suffix_grammar_scores(
        self, candidates: np.ndarray, S_p: np.ndarray
    ) -> np.ndarray:
        scores = np.full(len(candidates), 0.5, dtype=np.float32)
        for i, cand in enumerate(candidates):
            scores[i] = self.suffix_grammar_score(int(cand), S_p)
        return scores

    def subatomic_gradient_neighbors(
        self,
        token_id: int,
        max_neighbors: int = 16,
        min_sim: float = 0.65,
    ) -> List[Tuple[int, float]]:
        if token_id < 0 or token_id >= self.vocab_size:
            return []

        source_suffix_hv = self.suffix_hvs[token_id]
        sims = _hamming_sim_batch(source_suffix_hv, self.suffix_hvs)

        sims[token_id] = 0.0

        valid_mask = sims >= min_sim
        if not np.any(valid_mask):
            return []

        valid_indices = np.where(valid_mask)[0]
        valid_sims    = sims[valid_indices]

        order = np.argsort(valid_sims)[::-1]
        top_indices = valid_indices[order[:max_neighbors]]
        top_sims    = valid_sims[order[:max_neighbors]]

        return [(int(idx), float(sim)) for idx, sim in zip(top_indices, top_sims)]

    def grammar_gated_expansion(
        self,
        candidates: np.ndarray,
        S_p: np.ndarray,
        grammar_threshold: float = 0.52,
        max_neighbors_per_candidate: int = 8,
    ) -> List[int]:
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

    def top_grammar_contexts(self, token_id: int, codebook: np.ndarray,
                              top_k: int = 5) -> List[Tuple[int, float]]:
        if token_id < 0 or token_id >= self.vocab_size:
            return []

        bundle = self.suffix_gram_bundle[token_id]
        if not np.any(bundle):
            return []

        sims = _hamming_sim_batch(bundle, codebook)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    def summary(self) -> str:
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
