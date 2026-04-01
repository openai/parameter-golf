"""
Shared context hashing for correction table.

Single source of truth — used by both build_correction_table.py and eval_final.py.
Uses polynomial rolling hash with uint64 arithmetic, truncated to uint32.
"""
from __future__ import annotations

import numpy as np

# Hash constants
HASH_PRIME = 16777619


def context_hash_one(tokens: np.ndarray, context_len: int = 8) -> int:
    """Hash a single context window (last `context_len` tokens before target).
    
    Args:
        tokens: 1D array of token IDs, length >= context_len
        context_len: number of tokens to hash
    
    Returns:
        32-bit hash as Python int
    """
    assert len(tokens) >= context_len
    ctx = tokens[-context_len:]
    h = 0
    p = 1
    for t in ctx:
        h += int(t) * p
        p *= HASH_PRIME
    return h & 0xFFFFFFFF  # Truncate to uint32


def context_hash_all(tokens: np.ndarray, context_len: int = 8) -> np.ndarray:
    """Compute 32-bit context hashes for ALL positions in token array.
    
    For each position pos >= context_len, computes:
        hash(tokens[pos-context_len : pos]) → uint32
    
    Positions < context_len get hash = 0.
    
    Args:
        tokens: 1D array of token IDs
        context_len: number of preceding tokens to hash
    
    Returns:
        np.ndarray of shape [len(tokens)], dtype=uint32
    """
    n = len(tokens)
    hashes = np.zeros(n, dtype=np.uint32)
    
    # Precompute powers of HASH_PRIME (Python ints = arbitrary precision)
    powers = [1]
    for i in range(1, context_len):
        powers.append(powers[-1] * HASH_PRIME)
    
    for pos in range(context_len, n):
        h = 0
        for i in range(context_len):
            h += int(tokens[pos - context_len + i]) * powers[i]
        hashes[pos] = h & 0xFFFFFFFF
    
    return hashes
