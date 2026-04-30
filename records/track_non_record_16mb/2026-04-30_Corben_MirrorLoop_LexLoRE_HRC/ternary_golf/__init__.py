"""Small ternary-training helpers for sub-4MB Parameter Golf experiments."""

from .layers import TernaryLinear, dequantize_ternary_groups_with_shrinkage, ternary_weight_cache

__all__ = ["TernaryLinear", "dequantize_ternary_groups_with_shrinkage", "ternary_weight_cache"]
