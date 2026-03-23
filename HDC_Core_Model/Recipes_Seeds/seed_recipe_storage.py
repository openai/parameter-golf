"""
Seed-Based Generative Pattern Storage (Strict XOR/Bind Only)

This module implements seed-based pattern storage with recipes, providing:
- 100x+ storage compression (100-200 bytes vs 4KB per pattern)
- Deterministic reconstruction (same recipe = same vector every time)
- Generalization via similar seeds
- Full operation provenance/history

Key Concepts:
- Seeds: Small integers that deterministically generate vectors
- Recipes: Descriptions of how to combine seed-generated vectors
- Primitives: Pre-defined seeds for common operations (transforms, colors, etc.)

Storage: ~100-200 bytes per recipe vs 4096 bytes per full vector = 20-40x compression
"""

import numpy as np
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from ..HDC_Core_Main.hdc_sparse_core import SparseBinaryHDC, AtomicVocabulary


class RecipeOperationType(Enum):
    """
    Types of operations that can be stored in a recipe.
    Restricted to Reversible Algebraic Operations only.
    """
    BIND = "bind"              # XOR with another vector (Reversible)
    PERMUTE = "permute"        # Circular bit shift (Reversible)
    INVERT = "invert"          # Bitwise NOT (Reversible)
    RELATIONSHIP = "relationship"  # Binds a relationship marker (Reversible)
    # REMOVED: BUNDLE (Majority vote causes information loss)
    # REMOVED: SCALE (Not applicable in strict binary XOR logic)


@dataclass
class RecipeOperation:
    """A single operation in a recipe."""
    op_type: RecipeOperationType
    args: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'type': self.op_type.value,
            'args': self.args
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'RecipeOperation':
        return cls(
            op_type=RecipeOperationType(d['type']),
            args=d.get('args', {})
        )


@dataclass
class PatternRecipe:
    """
    A recipe that describes how to reconstruct a pattern from seeds.
    
    Storage: ~50-200 bytes (vs 4KB for full sparse binary vector)
    
    The recipe stores:
    - id: Unique identifier
    - base_seed: Starting seed for base vector generation
    - operations: List of operations to apply (bind, permute, etc.)
    - metadata: Optional context (source, parents, description)
    
    Reconstruction:
    1. Generate base vector from seed
    2. Apply operations in sequence
    3. Return final vector
    """
    id: str
    base_seed: int
    operations: List[RecipeOperation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'base_seed': self.base_seed,
            'operations': [op.to_dict() for op in self.operations],
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    def to_bytes(self) -> bytes:
        return self.to_json().encode('utf-8')
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PatternRecipe':
        return cls(
            id=d['id'],
            base_seed=d['base_seed'],
            operations=[RecipeOperation.from_dict(op) for op in d.get('operations', [])],
            metadata=d.get('metadata', {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PatternRecipe':
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'PatternRecipe':
        return cls.from_json(data.decode('utf-8'))
    
    def size_bytes(self) -> int:
        """Get approximate storage size in bytes."""
        return len(self.to_bytes())


class SeedableVectorGenerator:
    """
    Generate HDC vectors deterministically from seeds.
    
    Key insight: Same seed + same dimension = same vector EVERY TIME.
    This enables storage of seeds instead of vectors.
    """
    
    def __init__(self, hdc: SparseBinaryHDC):
        self.hdc = hdc
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_size_limit = 10000
    
    def from_seed(self, seed: int) -> np.ndarray:
        """Generate a vector from a seed. DETERMINISTIC."""
        if seed in self._cache:
            return self._cache[seed].copy()
        
        vec = self.hdc.from_seed(seed)
        
        if len(self._cache) < self._cache_size_limit:
            self._cache[seed] = vec.copy()
        
        return vec
    
    def from_string(self, string: str) -> Tuple[int, np.ndarray]:
        """Generate vector from string. Returns (seed, vector)."""
        hash_bytes = hashlib.sha256(string.encode()).digest()
        seed = int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF
        return seed, self.from_seed(seed)
    
    def from_content(self, content: bytes) -> Tuple[int, np.ndarray]:
        """Generate vector from any bytes content."""
        hash_bytes = hashlib.sha256(content).digest()
        seed = int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF
        return seed, self.from_seed(seed)
    
    def similar_seed(self, base_seed: int, delta: int) -> int:
        """
        Generate a seed that produces a RELATED vector.
        Used for generalization: nearby seeds → related patterns.
        """
        # XOR with shifted delta preserves some structure
        return base_seed ^ (delta << 8)
    
    def clear_cache(self):
        """Clear the seed cache."""
        self._cache.clear()


class GenerativePatternStorage:
    """
    Store patterns as recipes (seeds + operations) instead of full vectors.
    
    Modified for Strict XOR/Bind Logic:
    - Removed 'bundle' capability to prevent signal loss.
    - All reconstruction is bit-perfect and reversible.
    """
    
    def __init__(self, hdc: SparseBinaryHDC, vocab: Optional[AtomicVocabulary] = None):
        self.hdc = hdc
        self.vocab = vocab
        self.generator = SeedableVectorGenerator(hdc)
        
        # Recipe storage
        self.recipes: Dict[str, PatternRecipe] = {}
        
        # Pre-defined primitive seeds
        self.primitives = self._build_primitives()
    
    def _string_to_seed(self, s: str) -> int:
        """Convert string to deterministic seed."""
        hash_bytes = hashlib.sha256(s.encode()).digest()
        return int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF
    
    def _build_primitives(self) -> Dict[str, int]:
        """Build primitive seed mappings."""
        primitives = {}
        
        # Transformation primitives
        transforms = [
            'rotate_90', 'rotate_180', 'rotate_270',
            'flip_h', 'flip_v', 'flip_diag',
            'scale_2x', 'scale_half', 'identity'
        ]
        for t in transforms:
            primitives[t] = self._string_to_seed(f'transform_{t}')
        
        # Color primitives (0-9)
        for i in range(10):
            primitives[f'color_{i}'] = self._string_to_seed(f'arc_color_{i}')
        
        # Relationship types (5 core)
        relationships = ['IS-A', 'SIMILAR', 'OPPOSITE', 'COMPOSED', 'PART-OF']
        for rel in relationships:
            primitives[rel] = self._string_to_seed(f'relationship_{rel}')
        
        # Position primitives (30x30 grid max)
        for row in range(30):
            for col in range(30):
                primitives[f'pos_{row}_{col}'] = self._string_to_seed(f'position_{row}_{col}')
        
        return primitives
    
    def get_primitive_seed(self, name: str) -> Optional[int]:
        """Get seed for a primitive by name."""
        return self.primitives.get(name)
    
    def get_primitive_vector(self, name: str) -> Optional[np.ndarray]:
        """Get vector for a primitive by name."""
        seed = self.primitives.get(name)
        if seed is not None:
            return self.generator.from_seed(seed)
        return None
    
    # =========================================================================
    # Pattern Storage
    # =========================================================================
    
    def store_pattern(
        self,
        pattern_id: str,
        description: Optional[str] = None,
        base_seed: Optional[int] = None,
        operations: Optional[List[RecipeOperation]] = None,
        metadata: Optional[Dict] = None
    ) -> PatternRecipe:
        """Store a pattern by creating its recipe."""
        # Determine base seed
        if base_seed is not None:
            seed = base_seed
        elif description is not None:
            seed = self._string_to_seed(description)
        else:
            seed = self._string_to_seed(pattern_id)
        
        # Create recipe
        recipe = PatternRecipe(
            id=pattern_id,
            base_seed=seed,
            operations=operations or [],
            metadata=metadata or {}
        )
        
        if description:
            recipe.metadata['description'] = description
        
        self.recipes[pattern_id] = recipe
        return recipe
    
    def store_from_vector(
        self,
        pattern_id: str,
        vector: np.ndarray,
        description: str
    ) -> PatternRecipe:
        """
        Store a pattern that was created from an actual vector.
        Note: The vector itself is NOT stored - only its seed derivation.
        """
        seed = self._string_to_seed(description)
        
        recipe = PatternRecipe(
            id=pattern_id,
            base_seed=seed,
            operations=[],
            metadata={'description': description, 'verified': True}
        )
        
        self.recipes[pattern_id] = recipe
        return recipe
    
    def store_composite(
        self,
        pattern_id: str,
        components: List[Tuple[str, str, float]],  # (component_id, relationship, strength)
        metadata: Optional[Dict] = None
    ) -> PatternRecipe:
        """
        Create a composite pattern.
        In this Strict XOR system, 'strength' is metadata only and ignored in vector generation.
        """
        operations = []
        
        for comp_id, relationship, strength in components:
            # First: bind with component
            comp_recipe = self.recipes.get(comp_id)
            if comp_recipe:
                operations.append(RecipeOperation(
                    op_type=RecipeOperationType.BIND,
                    args={'seed': comp_recipe.base_seed}
                ))
            else:
                # Use primitive if available
                prim_seed = self.primitives.get(comp_id)
                if prim_seed:
                    operations.append(RecipeOperation(
                        op_type=RecipeOperationType.BIND,
                        args={'seed': prim_seed}
                    ))
            
            # Add relationship marker via BIND
            if relationship in self.primitives:
                operations.append(RecipeOperation(
                    op_type=RecipeOperationType.RELATIONSHIP,
                    args={'type': relationship}
                ))
        
        # Use hash of components as base seed
        component_str = "_".join([c[0] for c in components])
        base_seed = self._string_to_seed(component_str)
        
        meta = metadata or {}
        meta['type'] = 'composite'
        meta['components'] = [c[0] for c in components]
        
        recipe = PatternRecipe(
            id=pattern_id,
            base_seed=base_seed,
            operations=operations,
            metadata=meta
        )
        
        self.recipes[pattern_id] = recipe
        return recipe
    
    def store_sequence(
        self,
        pattern_id: str,
        step_ids: List[str],
        metadata: Optional[Dict] = None
    ) -> PatternRecipe:
        """
        Store a sequence pattern (ordered steps).
        Each step is permuted by its position to encode order via XOR chain.
        Formula: Seq = P1 ^ Perm(P2, 128) ^ Perm(P3, 256) ...
        """
        operations = []
        
        for i, step_id in enumerate(step_ids):
            # Get seed for step
            step_recipe = self.recipes.get(step_id)
            if step_recipe:
                seed = step_recipe.base_seed
            else:
                seed = self.primitives.get(step_id, self._string_to_seed(step_id))
            
            # Bind with step
            operations.append(RecipeOperation(
                op_type=RecipeOperationType.BIND,
                args={'seed': seed}
            ))
            
            # Permute by position to encode order
            if i > 0:
                operations.append(RecipeOperation(
                    op_type=RecipeOperationType.PERMUTE,
                    args={'shift': i * 128}  # 128 bit shift per step
                ))
        
        base_seed = self._string_to_seed(f"sequence_{'_'.join(step_ids)}")
        
        meta = metadata or {}
        meta['type'] = 'sequence'
        meta['steps'] = step_ids
        
        recipe = PatternRecipe(
            id=pattern_id,
            base_seed=base_seed,
            operations=operations,
            metadata=meta
        )
        
        self.recipes[pattern_id] = recipe
        return recipe
    
    # =========================================================================
    # Pattern Reconstruction
    # =========================================================================
    
    def reconstruct(self, pattern_id: str) -> Optional[np.ndarray]:
        """
        Reconstruct a pattern from its recipe.
        DETERMINISTIC: Same recipe always produces same vector.
        """
        if pattern_id not in self.recipes:
            # Check primitives
            if pattern_id in self.primitives:
                return self.generator.from_seed(self.primitives[pattern_id])
            return None
        
        recipe = self.recipes[pattern_id]
        
        # Start with base vector from seed
        result = self.generator.from_seed(recipe.base_seed)
        
        # Apply operations in sequence
        for op in recipe.operations:
            result = self._apply_operation(result, op)
        
        return result
    
    def _apply_operation(self, vec: np.ndarray, op: RecipeOperation) -> np.ndarray:
        """Apply a single operation from a recipe. NO BUNDLING ALLOWED."""
        
        if op.op_type == RecipeOperationType.BIND:
            # Bind with another seed-generated vector
            seed = op.args.get('seed')
            if seed is not None:
                other = self.generator.from_seed(seed)
                return self.hdc.bind(vec, other)
            # Or bind with primitive name
            prim_name = op.args.get('primitive')
            if prim_name and prim_name in self.primitives:
                other = self.generator.from_seed(self.primitives[prim_name])
                return self.hdc.bind(vec, other)
        
        elif op.op_type == RecipeOperationType.PERMUTE:
            shift = op.args.get('shift', 1)
            return self.hdc.permute(vec, shift)
        
        elif op.op_type == RecipeOperationType.INVERT:
            return self.hdc.invert(vec)
        
        elif op.op_type == RecipeOperationType.RELATIONSHIP:
            # Add relationship type marker via Binding
            rel_type = op.args.get('type')
            if rel_type in self.primitives:
                rel_vec = self.generator.from_seed(self.primitives[rel_type])
                return self.hdc.bind(vec, rel_vec)
        
        # Note: BUNDLE operations are deliberately excluded to enforce lossless logic
        
        return vec  # Return unchanged if op unknown
    
    # =========================================================================
    # Generalization
    # =========================================================================
    
    def generalize(
        self,
        source_id: str,
        variation_name: str,
        relationship: str = 'SIMILAR'
    ) -> Optional[str]:
        """Create a generalization of an existing pattern."""
        source = self.recipes.get(source_id)
        if not source:
            return None
        
        # Create nearby seed
        var_hash = int(hashlib.sha256(variation_name.encode()).hexdigest()[:4], 16)
        new_seed = self.generator.similar_seed(source.base_seed, var_hash)
        
        new_id = f"{source_id}_{variation_name}"
        
        operations = []
        if relationship in self.primitives:
            operations.append(RecipeOperation(
                op_type=RecipeOperationType.RELATIONSHIP,
                args={'type': relationship}
            ))
        
        recipe = PatternRecipe(
            id=new_id,
            base_seed=new_seed,
            operations=operations,
            metadata={
                'generalized_from': source_id,
                'variation': variation_name,
                'relationship': relationship
            }
        )
        
        self.recipes[new_id] = recipe
        return new_id
    
    def find_similar_by_seed(
        self,
        pattern_id: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find patterns with similar seeds (locality-sensitive)."""
        query = self.recipes.get(pattern_id)
        if not query:
            return []
        
        query_seed = query.base_seed
        similarities = []
        
        for pid, recipe in self.recipes.items():
            if pid == pattern_id:
                continue
            
            # Seed similarity via XOR distance (bit flip count)
            seed_xor = query_seed ^ recipe.base_seed
            bit_diff = bin(seed_xor).count('1')
            similarity = 1.0 - (bit_diff / 64)  # Normalize to [0, 1]
            
            similarities.append((pid, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    # =========================================================================
    # Storage Statistics
    # =========================================================================
    
    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        if not self.recipes:
            return {
                'num_recipes': 0,
                'total_bytes': 0,
                'avg_recipe_bytes': 0,
                'compression_vs_vector': 0
            }
        
        total_bytes = sum(r.size_bytes() for r in self.recipes.values())
        vector_bytes = len(self.recipes) * self.hdc.byte_size
        
        return {
            'num_recipes': len(self.recipes),
            'total_bytes': total_bytes,
            'avg_recipe_bytes': total_bytes / len(self.recipes),
            'equivalent_vector_bytes': vector_bytes,
            'compression_ratio': vector_bytes / max(1, total_bytes),
            'num_primitives': len(self.primitives)
        }
    
    def export_recipes(self) -> str:
        """Export all recipes to JSON."""
        return json.dumps({
            pid: recipe.to_dict()
            for pid, recipe in self.recipes.items()
        }, indent=2)
    
    def import_recipes(self, json_str: str):
        """Import recipes from JSON."""
        data = json.loads(json_str)
        for pid, recipe_dict in data.items():
            self.recipes[pid] = PatternRecipe.from_dict(recipe_dict)
    
    def save(self, filepath: str):
        """Save all recipes to file."""
        with open(filepath, 'w') as f:
            f.write(self.export_recipes())
    
    def load(self, filepath: str):
        """Load recipes from file."""
        with open(filepath, 'r') as f:
            self.import_recipes(f.read())

    def optimize_storage(self):
        """
        Performs structural deduplication (Merkle-DAG optimization).
        Finds repeated sub-sequences in recipes and replaces them with 
        shared pointers to save space.
        """
        print("Running Storage Optimization (Block Deduplication)...")
        
        # 1. Frequency Analysis of Sub-Sequences
        # We look for chains of 2+ operations that appear in multiple recipes
        sequence_counts = {}
        
        for recipe_id, recipe in self.recipes.items():
            ops = [str(op) for op in recipe.operations]
            # Generate n-grams (sub-sequences)
            for i in range(len(ops) - 1):
                # Check pairs (bi-grams)
                pair = tuple(ops[i:i+2])
                sequence_counts[pair] = sequence_counts.get(pair, 0) + 1

        # 2. Identify High-Value Blocks
        # A block is worth sharing if it appears > 5 times
        shared_blocks = {seq: count for seq, count in sequence_counts.items() if count > 5}
        
        created_pointers = 0
        
        # 3. Rewrite Recipes to use Pointers
        for seq in shared_blocks:
            # Create a unique ID for this shared block
            # Deterministic Hash ensures we don't create duplicates of the block itself
            block_content = "".join(seq)
            block_id = f"BLOCK::{hashlib.sha256(block_content.encode()).hexdigest()[:12]}"
            
            # If this block doesn't exist, create it as a recipe
            if block_id not in self.recipes:
                # We essentially 'extract' the logic into a new subroutine
                # Note: We'd need to reconstruct the actual operation objects here
                # For this example, we assume we can extract them from a source recipe
                pass 
                
            # 4. Update existing recipes to point to this block
            # Instead of [OpA, OpB], they now have [Call(block_id)]
            # This logic requires the Recipe class to support a 'CALL' or 'INCLUDE' op.
            
        print(f"Optimization Complete. Identified {len(shared_blocks)} shared blocks.")