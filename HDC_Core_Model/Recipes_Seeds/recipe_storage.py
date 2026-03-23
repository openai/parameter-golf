"""
Recipe Storage System - 4KB Identity Recipes for Instant Model Merging

This module implements the recipe-based memory storage as specified in the
Universal XOR 8K Engine architecture.

Key Features:
- 4KB per learned recipe (8KB for full 32768D with 2-bit packing)
- Instant model merging via file concatenation (no retraining!)
- Checksum verification for bit-perfect integrity
- NVMe-optimized I/O for high-throughput recipe access

Architecture Notes (from FULLINTEGRATION_NEW_ARCHITECTURE.md):
- Recipe = bound vector (Input ⊕ Label) stored as binary file
- No weights are modified during learning
- New knowledge is simply appended as logical links
- Merging = copy recipe files to same folder (universal Hadamard basis)

NOTE: THIS IS NOT CORRECT. THIS FILE SHOULD NOT USE BUNDLING AT ALL TO AVOID BLURRING AND ACCURACY ERRORS. INSTEAD, IT SHOULD MERGE MODELS TOGETHER BY STORING SEEDS WITH THE SAME INDEXING BUT DIFFERENT CONTENT IN DIFFERENT LOCATIONS. IF THE SEEDS HAVE SIMILAR CONTENT, THEN THEIR INDEXING NEEDS TO BE MODIFIED TO POINT TO THE SHORTEST REPRESENATION OF THE SIMILAR CONTENT.
"""

import numpy as np
import hashlib
import json
import struct
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple, Dict, Any, Iterator
from datetime import datetime
import os

# Try to import our Walsh-Hadamard components
try:
    from ..HDC_Core_Main.Recipes_Seeds.walsh_hadamard_core import TernaryHadamardEncoder
    from ..HDC_Core_Main.hdc_sparse_core import DEFAULT_HDC_DIM
except ImportError:
    TernaryHadamardEncoder = None
    DEFAULT_HDC_DIM = 1048576  # 2^20 - Default for 8K video processing


# =============================================================================
# Recipe Data Structures
# =============================================================================

@dataclass
class IdentityRecipe:
    """
    A single learned memory - the atomic unit of knowledge in this architecture.
    
    Each recipe represents a binding: Input ⊕ Label = Recipe
    
    ZERO-WEIGHT PHILOSOPHY:
    We do NOT store the full 32KB vector. 
    We ONLY store the GENERATIVE SEED/INDEX (~100 bytes).
    The vector is materialized on-the-fly via the DeterministicFlowEngine.
    
    Attributes:
        recipe_id: Unique identifier (SHA256 hash)
        hadamard_index: Which Hadamard row was used (The "Address")
        base_seed: The procedural seed for the content (if applicable)
        operation: The operation performed (e.g., "bind", "peel")
        source_id: ID of the source recipe (if derived)
        metadata: Additional key-value metadata
    """
    recipe_id: str
    hadamard_index: int
    base_seed: int = 0
    operation: str = "identity" # "identity", "bind", "peel"
    source_id: str = ""
    name: str = ""
    recipe_type: str = "general"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_bytes(self) -> bytes:
        """Serialize recipe to compact binary format."""
        # Header: magic number + version
        header = struct.pack('>4sH', b'XORR', 2)  # XOR Recipe v2 (Seed-Based)
        
        # Core data (Fixed size struct for speed)
        # Q = unsigned long long (8 bytes)
        # i = int (4 bytes)
        core = struct.pack('>QQi', 
            self.hadamard_index, 
            self.base_seed,
            0 # Reserved/Flags
        )
        
        # Variable string data (JSON for flexibility)
        meta = {
            'id': self.recipe_id,
            'op': self.operation,
            'src': self.source_id,
            'name': self.name,
            'type': self.recipe_type,
            'ts': self.created_at,
            'meta': self.metadata
        }
        meta_json = json.dumps(meta, separators=(',', ':')).encode('utf-8')
        
        return (
            header +
            core +
            struct.pack('>I', len(meta_json)) +
            meta_json
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'IdentityRecipe':
        """Deserialize recipe from binary format."""
        magic, version = struct.unpack('>4sH', data[:6])
        if magic != b'XORR':
            raise ValueError(f"Invalid recipe format: magic={magic}")
        
        offset = 6
        
        if version == 2:
            # Current Seed-Based Format
            hadamard_index, base_seed, flags = struct.unpack('>QQi', data[offset:offset+20])
            offset += 20
            
            meta_len = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            
            meta_json = data[offset:offset+meta_len].decode('utf-8')
            meta = json.loads(meta_json)
            
            return cls(
                recipe_id=meta['id'],
                hadamard_index=hadamard_index,
                base_seed=base_seed,
                operation=meta.get('op', 'identity'),
                source_id=meta.get('src', ''),
                name=meta.get('name', ''),
                recipe_type=meta.get('type', 'general'),
                created_at=meta.get('ts', ''),
                metadata=meta.get('meta', {})
            )
        elif version == 1:
            # Legacy Vector-Based Format (Auto-upgrade logic could go here)
            # For now, we raise error or return partial
            raise ValueError("Version 1 (Vector-Based) recipes should be migrated.")
        else:
            raise ValueError(f"Unsupported recipe version: {version}")

    @property
    def size_bytes(self) -> int:
        """Total size in bytes."""
        return len(self.to_bytes())



    def verify_integrity(self) -> bool:
        """
        Verify recipe integrity.
        
        For PROCEDURAL recipes (H...), checks if ID matches 'H{idx}.S{seed}'.
        For BINDING recipes (BIND...), checks if ID matches 'BIND({ids})'.
        """
        if self.operation == "identity":
            expected = f"H{self.hadamard_index}.S{self.base_seed}"
            return self.recipe_id == expected
            
        elif self.operation == "bind":
            # We can't easily verify BIND IDs without parsing the string components,
            # but we can check the prefix.
            return self.recipe_id.startswith("BIND(")
            
        return True

# =============================================================================
# Recipe Storage System
# =============================================================================

class RecipeStorage:
    """
    File-based recipe storage for instant model merging.
    
    Storage Layout:
        base_path/
        ├── recipes/
        │   ├── {recipe_id}.xorr    # Individual recipe files
        │   └── ...
        ├── index.json              # Fast lookup index
        └── manifest.json           # Storage metadata
    
    Key Methods:
        save_recipe(): Store a single recipe
        load_recipe(): Load a recipe by ID
        list_recipes(): List all recipe IDs
        merge_from(): Merge recipes from another storage location
    
    Merging is trivial:
        To merge Model A and Model B, simply copy their recipe files
        into the same folder.
        
        Indices define identity:
        - If Model A has H5.S100 and Model B has H5.S100, they are identical.
        - If Model A has H5.S100 and Model B has H5.S200, they are distinct
          variants of the same abstract address. Both are kept.
    """
    
    RECIPES_DIR = "recipes"
    INDEX_FILE = "index.json"
    MANIFEST_FILE = "manifest.json"
    RECIPE_EXT = ".xorr"
    
    def __init__(self, base_path: Path, create: bool = True):
        """
        Initialize recipe storage.
        
        Args:
            base_path: Root directory for storage
            create: Whether to create directories if missing
        """
        self.base_path = Path(base_path)
        self.recipes_path = self.base_path / self.RECIPES_DIR
        self.index_path = self.base_path / self.INDEX_FILE
        self.manifest_path = self.base_path / self.MANIFEST_FILE
        
        if create:
            self._ensure_directories()
        
        self._index: Dict[str, Dict] = {}
        self._load_index()
        
    def _ensure_directories(self):
        """Create storage directories if needed."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.recipes_path.mkdir(exist_ok=True)
        
        # Create manifest if missing
        if not self.manifest_path.exists():
            manifest = {
                "version": 1,
                "created_at": datetime.utcnow().isoformat(),
                "description": "Universal XOR 8K Engine Recipe Storage",
                "hadamard_dim": DEFAULT_HDC_DIM
            }
            self.manifest_path.write_text(json.dumps(manifest, indent=2))
    
    def _load_index(self):
        """Load recipe index for fast lookups."""
        if self.index_path.exists():
            self._index = json.loads(self.index_path.read_text())
        else:
            self._index = {}
    
    def _save_index(self):
        """Save recipe index."""
        self.index_path.write_text(json.dumps(self._index, indent=2))
    
    def _recipe_path(self, recipe_id: str) -> Path:
        """Get path for a recipe file."""
        return self.recipes_path / f"{recipe_id}{self.RECIPE_EXT}"
    
    # =========================================================================
    # Recipe Operations
    # =========================================================================
    
    def save_recipe(self, recipe: IdentityRecipe) -> str:
        """
        Save a recipe to storage.
        
        Args:
            recipe: Recipe to save
        
        Returns:
            Recipe ID string
        """
        path = self._recipe_path(recipe.recipe_id)
        path.write_bytes(recipe.to_bytes())
        
        # Update index
        self._index[recipe.recipe_id] = {
            "name": recipe.name,
            "type": recipe.recipe_type,
            "hadamard_index": recipe.hadamard_index,
            "base_seed": recipe.base_seed, # Added for v2
            "created_at": recipe.created_at,
            "size_bytes": recipe.size_bytes
        }
        self._save_index()
        
        return recipe.recipe_id
    
    def load_recipe(self, recipe_id: str) -> Optional[IdentityRecipe]:
        """
        Load a recipe by ID.
        
        Args:
            recipe_id: Recipe identifier
        
        Returns:
            Recipe if found, None otherwise
        """
        path = self._recipe_path(recipe_id)
        if not path.exists():
            return None
        
        data = path.read_bytes()
        return IdentityRecipe.from_bytes(data)
    
    def delete_recipe(self, recipe_id: str) -> bool:
        """
        Delete a recipe.
        
        Args:
            recipe_id: Recipe to delete
        
        Returns:
            True if deleted, False if not found
        """
        path = self._recipe_path(recipe_id)
        if not path.exists():
            return False
        
        path.unlink()
        
        if recipe_id in self._index:
            del self._index[recipe_id]
            self._save_index()
        
        return True
    
    def list_recipes(self, recipe_type: Optional[str] = None) -> List[str]:
        """
        List all recipe IDs in storage.
        
        Args:
            recipe_type: Optional filter by type
        
        Returns:
            List of recipe IDs
        """
        if recipe_type:
            return [
                rid for rid, info in self._index.items()
                if info.get("type") == recipe_type
            ]
        return list(self._index.keys())
        
    def find_recipes_by_metadata(self, key: str, value: Any = None) -> List[str]:
        """
        Find recipes where metadata contains a specific key/value pair.
        
        This enables the storage to act as a lightweight semantic graph.
        Example: storage.find_recipes_by_metadata("relation", "IS-A")
        
        Args:
            key: Metadata key to check
            value: Optional value to match. If None, just checks for key existence.
            
        Returns:
            List of matching recipe IDs
        """
        matches = []
        for rid, info in self._index.items():
            # We need to load the recipe to check deep metadata if it's not in index?
            # Ideally index should contain searchable metadata.
            # Current Implementation of save_recipe puts metadata in the ITEM logic, 
            # but only writes specific fields to _index. 
            # To support fast search without loading every file, we should 
            # promote commonly used metadata keys to the index or implement a lazy scan.
            
            # For now, let's assume we implement a lazy scan which is slower but correct.
            # OPTIMIZATION: In V2, promote specific keys to self._index
            recipe = self.load_recipe(rid)
            if recipe and key in recipe.metadata:
                if value is None or recipe.metadata[key] == value:
                    matches.append(rid)
        return matches
    
    def iter_recipes(self, recipe_type: Optional[str] = None) -> Iterator[IdentityRecipe]:
        """
        Iterate over all recipes (lazy loading).
        
        Args:
            recipe_type: Optional filter by type
        
        Yields:
            IdentityRecipe objects
        """
        for recipe_id in self.list_recipes(recipe_type):
            recipe = self.load_recipe(recipe_id)
            if recipe:
                yield recipe
    
    # =========================================================================
    # Model Merging (The Key Feature!)
    # =========================================================================
    
    def merge_from(self, other_path: Path, overwrite: bool = False) -> Dict[str, int]:
        """
        Merge recipes from another storage location.
        
        This is the "instant model merging" feature from the architecture.
        Because both models use the universal Hadamard basis, their recipes
        are instantly compatible - no retraining or weight averaging needed!
        
        Args:
            other_path: Path to other recipe storage
            overwrite: If True, overwrite existing recipes with same ID
        
        Returns:
            Dict with merge statistics: {"added": N, "skipped": M, "errors": E}
        """
        other = RecipeStorage(other_path, create=False)
        
        stats = {"added": 0, "skipped": 0, "errors": 0}
        
        for recipe_id in other.list_recipes():
            if recipe_id in self._index and not overwrite:
                stats["skipped"] += 1
                continue
            
            try:
                recipe = other.load_recipe(recipe_id)
                if recipe:
                    self.save_recipe(recipe)
                    stats["added"] += 1
            except Exception as e:
                stats["errors"] += 1
                print(f"Error merging recipe {recipe_id}: {e}")
        
        return stats
    
    def merge_folders(self, folder_a: Path, folder_b: Path, 
                     output: Path) -> Dict[str, int]:
        """
        Merge two model folders into a new output folder.
        
        Static method for merging without modifying source folders.
        
        Args:
            folder_a: First model's recipe folder
            folder_b: Second model's recipe folder
            output: Output folder for merged model
        
        Returns:
            Merge statistics
        """
        # Create output storage
        merged = RecipeStorage(output, create=True)
        
        # Merge both sources
        stats_a = merged.merge_from(folder_a)
        stats_b = merged.merge_from(folder_b)
        
        # Combine stats
        return {
            "added": stats_a["added"] + stats_b["added"],
            "skipped": stats_a["skipped"] + stats_b["skipped"],
            "errors": stats_a["errors"] + stats_b["errors"]
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = sum(
            info.get("size_bytes", 0) for info in self._index.values()
        )
        
        type_counts = {}
        for info in self._index.values():
            t = info.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "total_recipes": len(self._index),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "type_distribution": type_counts
        }
    
    def verify_all(self) -> Tuple[int, int]:
        """
        Verify integrity of all recipes.
        
        Returns:
            Tuple of (valid_count, invalid_count)
        """
        valid = 0
        invalid = 0
        
        for recipe in self.iter_recipes():
            if recipe.verify_integrity():
                valid += 1
            else:
                invalid += 1
        
        return valid, invalid


# =============================================================================
# Recipe Factory Functions
# =============================================================================

def create_procedural_recipe(
    hadamard_index: int,
    base_seed: int,
    name: str = "",
    recipe_type: str = "general",
    metadata: Optional[Dict] = None
) -> IdentityRecipe:
    """
    Create a new procedural recipe (Zero-Weight).
    
    Args:
        hadamard_index: The universal address in Hadamard space
        base_seed: The seed that generates the content
        name: Human-readable name
        recipe_type: Type categorization
        metadata: Additional metadata
    
    Returns:
        IdentityRecipe ready for storage
    """
    # DETERMINISTIC IDENTITY:
    # Instead of SHA256 content hash, we use the unique coordinate in Hadamard space.
    # Format: H{hadamard_index}.S{base_seed}
    # This guarantees that the same "Address" always equals the same "Idea".
    recipe_id = f"H{hadamard_index}.S{base_seed}"
    
    return IdentityRecipe(
        recipe_id=recipe_id,
        hadamard_index=hadamard_index,
        base_seed=base_seed,
        operation="identity",
        name=name,
        recipe_type=recipe_type,
        metadata=metadata or {}
    )

def create_binding_recipe(
    target_recipe_id: str,
    bind_with_recipe_id: str,
    name: str = "", 
    recipe_type: str = "binding"
) -> IdentityRecipe:
    """
    Create a recipe that represents the BINDING of two other recipes.
    
    Note: This stores the OPERATION, not the result.
    Result = Recipe(A) ⊕ Recipe(B)
    """
    # BINDING IDENTITY:
    # The ID is the deterministic XOR composition of the two source IDs.
    # Since IDs are strings, we combine them deterministically.
    # Format: BIND({id_a}+{id_b})
    # We sort them to ensure commutativity if desired (A+B = B+A), 
    # but strict ordering (A+B != B+A) is often safer for non-commutative binding.
    # Assuming standard XOR binding is commutative:
    ids = sorted([target_recipe_id, bind_with_recipe_id])
    recipe_id = f"BIND({ids[0]}+{ids[1]})"
    
    return IdentityRecipe(
        recipe_id=recipe_id,
        hadamard_index=0, # Virtual index for composite
        base_seed=0,
        operation="bind",
        source_id=target_recipe_id, 
        name=name,
        recipe_type=recipe_type,
        metadata={"operand": bind_with_recipe_id}
    )
