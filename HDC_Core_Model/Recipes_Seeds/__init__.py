"""
Recipes and Seeds - Recipe storage and seed-based HDC operations.

This package provides:
- walsh_hadamard_core: Walsh-Hadamard basis for orthogonal projection
- recipe_storage: Identity recipe storage
- seed_recipe_storage: Seed-based recipe storage
- resonator_network: Resonator network for parallel factorization
- xor_peeling_search: XOR peeling search for recipe discovery
- difficulty_learning: Difficulty learning for adaptive time budgeting
"""

from .walsh_hadamard_core import (
    WalshHadamardBasis,
    TernaryHadamardEncoder,
    DEFAULT_HDC_DIM,
    HDC_DIM_LEGACY
)

from .recipe_storage import IdentityRecipe, RecipeStorage

from .seed_recipe_storage import PatternRecipe, RecipeOperation, RecipeOperationType

__all__ = [
    'WalshHadamardBasis',
    'TernaryHadamardEncoder',
    'DEFAULT_HDC_DIM',
    'HDC_DIM_LEGACY',
    'IdentityRecipe',
    'RecipeStorage',
    'PatternRecipe',
    'RecipeOperation',
    'RecipeOperationType',
]
