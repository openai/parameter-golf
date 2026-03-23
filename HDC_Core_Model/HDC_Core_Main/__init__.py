"""
HDC Core Main - Main HDC sparse core implementation.

This package provides:
- hdc_sparse_core: Core HDC sparse operations including BLAKE3-based vector generation
"""

from .hdc_sparse_core import (
    seed_to_hypervector_blake3,
    seed_string_to_int,
    _BLAKE3_AVAILABLE,
    SparseBinaryHDC,
    SparseBinaryConfig
)

__all__ = [
    'seed_to_hypervector_blake3',
    'seed_string_to_int',
    '_BLAKE3_AVAILABLE',
    'SparseBinaryHDC',
    'SparseBinaryConfig',
]
