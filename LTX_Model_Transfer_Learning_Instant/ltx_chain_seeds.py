"""
LTX Chain Seeds - Store Audio-Video Generation Chains as Reproducible Seed Sequences

This module provides storage and manipulation of LTX generation chains as seed sequences,
enabling 500x compression ratio and perfect reproducibility.

Key Features:
1. Store complete generation trajectories as seed sequences
2. Reconstruct vectors from seeds using BLAKE3 deterministic generation
3. Synthesize new chains from existing seeds
4. Encode timestep weights and attention patterns
5. Support audio-video joint generation chains

Architecture Integration:
- Each seed = 8 bytes (vs 16KB for full vector)
- 500x compression ratio
- Perfect reproducibility (same seed = same vector)
- Circular temporal encoding for unlimited depth
"""

import numpy as np
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import threading

# Import unified HDC integration (new architecture features)
try:
    from .ltx_unified_integration import (
        LTXUnifiedIntegration,
        LTXRoleType,
        get_ltx_integration
    )
    UNIFIED_INTEGRATION_AVAILABLE = True
except ImportError:
    UNIFIED_INTEGRATION_AVAILABLE = False


class LTXChainOperation(Enum):
    """Operations for LTX chain steps."""
    BIND = "bind"              # XOR bind
    UNBIND = "unbind"          # XOR unbind
    BUNDLE = "bundle"          # Superposition
    CIRCULAR_SHIFT = "circular_shift"  # Temporal encoding
    ATTENTION_WEIGHT = "attention_weight"  # Apply attention weight
    CROSS_MODAL_BIND = "cross_modal_bind"  # Bind audio-video


@dataclass
class LTXSeedStep:
    """A single step in an LTX generation chain."""
    step_id: str
    seed: int
    hadamard_index: int
    operation: LTXChainOperation
    weight: float = 1.0
    
    # LTX-specific fields
    layer_name: str = ""
    timestep: int = 0
    modality: str = "joint"  # 'video', 'audio', 'joint'
    
    # Attention pattern (optional)
    attention_pattern: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step_id': self.step_id,
            'seed': self.seed,
            'hadamard_index': self.hadamard_index,
            'operation': self.operation.value,
            'weight': self.weight,
            'layer_name': self.layer_name,
            'timestep': self.timestep,
            'modality': self.modality,
            'attention_pattern': self.attention_pattern,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LTXSeedStep':
        """Create from dictionary."""
        return cls(
            step_id=data['step_id'],
            seed=data['seed'],
            hadamard_index=data['hadamard_index'],
            operation=LTXChainOperation(data['operation']),
            weight=data.get('weight', 1.0),
            layer_name=data.get('layer_name', ''),
            timestep=data.get('timestep', 0),
            modality=data.get('modality', 'joint'),
            attention_pattern=data.get('attention_pattern'),
            metadata=data.get('metadata', {})
        )


@dataclass
class LTXChainSeed:
    """A complete LTX generation chain stored as seed sequence."""
    chain_id: str
    model_name: str
    generation_mode: str  # LTXGenerationMode value
    steps: List[LTXSeedStep]
    
    # Chain metadata
    total_timesteps: int = 0
    video_resolution: Tuple[int, int] = (768, 512)
    audio_duration_sec: float = 0.0
    
    # Creation info
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'chain_id': self.chain_id,
            'model_name': self.model_name,
            'generation_mode': self.generation_mode,
            'steps': [s.to_dict() for s in self.steps],
            'total_timesteps': self.total_timesteps,
            'video_resolution': list(self.video_resolution),
            'audio_duration_sec': self.audio_duration_sec,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LTXChainSeed':
        """Create from dictionary."""
        return cls(
            chain_id=data['chain_id'],
            model_name=data['model_name'],
            generation_mode=data['generation_mode'],
            steps=[LTXSeedStep.from_dict(s) for s in data['steps']],
            total_timesteps=data.get('total_timesteps', 0),
            video_resolution=tuple(data.get('video_resolution', [768, 512])),
            audio_duration_sec=data.get('audio_duration_sec', 0.0),
            created_at=data.get('created_at', datetime.now().isoformat()),
            modified_at=data.get('modified_at', datetime.now().isoformat()),
            metadata=data.get('metadata', {})
        )
    
    def get_storage_size_bytes(self) -> int:
        """Calculate storage size in bytes."""
        # Each step: seed (8) + hadamard_index (4) + operation (1) + weight (4) + metadata
        step_size = 8 + 4 + 1 + 4 + 50  # Approximate
        return len(self.steps) * step_size + 200  # Header overhead


class LTXChainStorage:
    """
    Persistent storage for LTX generation chain seeds.
    
    Features:
    - File-based JSON persistence
    - In-memory caching for fast access
    - Index by model, generation mode, and content hash
    - Chain versioning support
    """
    
    def __init__(self, storage_path: str, hdc_dim: int = 131072):
        """
        Initialize chain storage.
        
        Args:
            storage_path: Path to storage directory
            hdc_dim: HDC dimension for vector reconstruction
        """
        self.storage_path = Path(storage_path)
        self.hdc_dim = hdc_dim
        self.uint64_count = hdc_dim // 64
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, LTXChainSeed] = {}
        self._cache_lock = threading.Lock()
        
        # Indexes
        self._by_model: Dict[str, List[str]] = {}
        self._by_mode: Dict[str, List[str]] = {}
        self._by_timestep: Dict[int, List[str]] = {}
        
        # Load existing chains
        self._load_index()
    
    def _load_index(self):
        """Load index of existing chains."""
        for chain_file in self.storage_path.glob("*.json"):
            try:
                with open(chain_file, 'r') as f:
                    data = json.load(f)
                chain = LTXChainSeed.from_dict(data)
                self._index_chain(chain)
            except Exception as e:
                print(f"Warning: Failed to load chain {chain_file}: {e}")
    
    def _index_chain(self, chain: LTXChainSeed):
        """Add chain to indexes."""
        # By model
        if chain.model_name not in self._by_model:
            self._by_model[chain.model_name] = []
        self._by_model[chain.model_name].append(chain.chain_id)
        
        # By mode
        if chain.generation_mode not in self._by_mode:
            self._by_mode[chain.generation_mode] = []
        self._by_mode[chain.generation_mode].append(chain.chain_id)
        
        # By timestep
        for step in chain.steps:
            if step.timestep not in self._by_timestep:
                self._by_timestep[step.timestep] = []
            if chain.chain_id not in self._by_timestep[step.timestep]:
                self._by_timestep[step.timestep].append(chain.chain_id)
    
    def save_chain(self, chain: LTXChainSeed) -> str:
        """
        Save a chain to storage.
        
        Args:
            chain: Chain to save
            
        Returns:
            Chain ID
        """
        with self._cache_lock:
            self._cache[chain.chain_id] = chain
        
        # Update modified timestamp
        chain.modified_at = datetime.now().isoformat()
        
        # Save to file
        chain_file = self.storage_path / f"{chain.chain_id}.json"
        with open(chain_file, 'w') as f:
            json.dump(chain.to_dict(), f, indent=2)
        
        # Update indexes
        self._index_chain(chain)
        
        return chain.chain_id
    
    def load_chain(self, chain_id: str) -> Optional[LTXChainSeed]:
        """
        Load a chain from storage.
        
        Args:
            chain_id: Chain ID to load
            
        Returns:
            Chain or None if not found
        """
        # Check cache first
        with self._cache_lock:
            if chain_id in self._cache:
                return self._cache[chain_id]
        
        # Load from file
        chain_file = self.storage_path / f"{chain_id}.json"
        if not chain_file.exists():
            return None
        
        with open(chain_file, 'r') as f:
            data = json.load(f)
        
        chain = LTXChainSeed.from_dict(data)
        
        # Add to cache
        with self._cache_lock:
            self._cache[chain_id] = chain
        
        return chain
    
    def delete_chain(self, chain_id: str) -> bool:
        """
        Delete a chain from storage.
        
        Args:
            chain_id: Chain ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        # Remove from cache
        with self._cache_lock:
            if chain_id in self._cache:
                del self._cache[chain_id]
        
        # Remove file
        chain_file = self.storage_path / f"{chain_id}.json"
        if chain_file.exists():
            chain_file.unlink()
            return True
        
        return False
    
    def get_chains_by_model(self, model_name: str) -> List[LTXChainSeed]:
        """Get all chains for a specific model."""
        chain_ids = self._by_model.get(model_name, [])
        return [self.load_chain(cid) for cid in chain_ids if self.load_chain(cid) is not None]
    
    def get_chains_by_mode(self, generation_mode: str) -> List[LTXChainSeed]:
        """Get all chains for a specific generation mode."""
        chain_ids = self._by_mode.get(generation_mode, [])
        return [self.load_chain(cid) for cid in chain_ids if self.load_chain(cid) is not None]
    
    def get_chains_by_timestep(self, timestep: int) -> List[LTXChainSeed]:
        """Get all chains containing a specific timestep."""
        chain_ids = self._by_timestep.get(timestep, [])
        return [self.load_chain(cid) for cid in chain_ids if self.load_chain(cid) is not None]
    
    def list_all_chains(self) -> List[str]:
        """List all chain IDs."""
        return list(self._by_model.get('LTX-2.3', [])) + list(self._by_model.get('LTX-2', []))


class LTXChainReconstructor:
    """
    Reconstructs HDC vectors from LTX chain seeds.
    
    Uses BLAKE3 deterministic generation to reconstruct vectors from seeds,
    then applies chain operations to produce final vectors.
    """
    
    def __init__(self, hdc_dim: int = 131072):
        """
        Initialize reconstructor.
        
        Args:
            hdc_dim: HDC dimension
        """
        self.hdc_dim = hdc_dim
        self.uint64_count = hdc_dim // 64
        
        # Try to import BLAKE3
        try:
            import blake3
            self._blake3 = blake3
            self._use_blake3 = True
        except ImportError:
            self._blake3 = None
            self._use_blake3 = False
    
    def seed_to_vector(self, seed: int) -> np.ndarray:
        """
        Convert seed to HDC vector.
        
        Args:
            seed: Integer seed
            
        Returns:
            uint64 HDC vector
        """
        seed_bytes = seed.to_bytes(8, 'little', signed=True)
        
        if self._use_blake3:
            hash_bytes = self._blake3.blake3(seed_bytes).digest(length=self.uint64_count * 8)
            return np.frombuffer(hash_bytes, dtype=np.uint64).copy()
        else:
            # Fallback to SHA256
            extended = bytearray()
            counter = 0
            while len(extended) < self.uint64_count * 8:
                h = hashlib.sha256(seed_bytes + counter.to_bytes(4, 'little'))
                extended.extend(h.digest())
                counter += 1
            return np.frombuffer(extended[:self.uint64_count * 8], dtype=np.uint64).copy()
    
    def reconstruct_step(self, step: LTXSeedStep) -> np.ndarray:
        """
        Reconstruct vector for a single step.
        
        Args:
            step: Step to reconstruct
            
        Returns:
            HDC vector for the step
        """
        return self.seed_to_vector(step.seed)
    
    def reconstruct_chain(self, 
                          chain: LTXChainSeed, 
                          strategy: str = "sequential") -> Union[List[np.ndarray], np.ndarray]:
        """
        Reconstruct vectors from chain.
        
        Args:
            chain: Chain to reconstruct
            strategy: Reconstruction strategy:
                - 'sequential': Return list of vectors
                - 'bind': XOR all steps together
                - 'weighted': Weighted superposition
                - 'temporal': Apply circular temporal encoding
                
        Returns:
            Reconstructed vector(s)
        """
        vectors = [self.reconstruct_step(step) for step in chain.steps]
        
        if strategy == "sequential":
            return vectors
        
        elif strategy == "bind":
            result = vectors[0].copy()
            for v in vectors[1:]:
                result = np.bitwise_xor(result, v)
            return result
        
        elif strategy == "weighted":
            # Weighted superposition (convert to float, weight, sum, threshold)
            float_result = np.zeros(self.hdc_dim, dtype=np.float32)
            for v, step in zip(vectors, chain.steps):
                # Convert uint64 to bipolar
                bipolar = np.where(
                    np.unpackbits(v.view(np.uint8)).reshape(-1,) > 0.5,
                    1.0, -1.0
                )[:self.hdc_dim]
                float_result += bipolar * step.weight
            
            # Threshold back to binary
            threshold = 0.0
            binary_result = (float_result > threshold).astype(np.uint8)
            
            # Pack back to uint64
            packed = np.packbits(binary_result)
            return packed.view(np.uint64)[:self.uint64_count]
        
        elif strategy == "temporal":
            # Circular temporal encoding: ρ^0(v0) ⊕ ρ^1(v1) ⊕ ρ^2(v2) ⊕ ...
            result = np.zeros(self.uint64_count, dtype=np.uint64)
            for i, (v, step) in enumerate(zip(vectors, chain.steps)):
                # Apply circular shift based on position
                shift_amount = i % (self.uint64_count * 64)  # Wrap around
                shifted = np.roll(v, shift_amount // 64)
                result = np.bitwise_xor(result, shifted)
            return result
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def reconstruct_generation_trajectory(self, 
                                          chain: LTXChainSeed) -> Dict[int, np.ndarray]:
        """
        Reconstruct generation trajectory organized by timestep.
        
        Args:
            chain: Chain to reconstruct
            
        Returns:
            Dictionary mapping timestep to combined vector
        """
        trajectory = {}
        
        # Group steps by timestep
        timestep_steps: Dict[int, List[LTXSeedStep]] = {}
        for step in chain.steps:
            if step.timestep not in timestep_steps:
                timestep_steps[step.timestep] = []
            timestep_steps[step.timestep].append(step)
        
        # Reconstruct each timestep
        for timestep, steps in timestep_steps.items():
            vectors = [self.reconstruct_step(s) for s in steps]
            # Combine vectors for this timestep
            combined = vectors[0].copy()
            for v in vectors[1:]:
                combined = np.bitwise_xor(combined, v)
            trajectory[timestep] = combined
        
        return trajectory
    
    def reconstruct_by_modality(self,
                                chain: LTXChainSeed) -> Dict[str, np.ndarray]:
        """
        Reconstruct vectors organized by modality (video/audio/joint).
        
        Args:
            chain: Chain to reconstruct
            
        Returns:
            Dictionary mapping modality to combined vector
        """
        modality_vectors: Dict[str, List[np.ndarray]] = {
            'video': [],
            'audio': [],
            'joint': []
        }
        
        for step in chain.steps:
            v = self.reconstruct_step(step)
            modality = step.modality if step.modality in modality_vectors else 'joint'
            modality_vectors[modality].append(v)
        
        result = {}
        for modality, vectors in modality_vectors.items():
            if vectors:
                combined = vectors[0].copy()
                for v in vectors[1:]:
                    combined = np.bitwise_xor(combined, v)
                result[modality] = combined
        
        return result


class LTXChainSynthesizer:
    """
    Creates new chains from existing LTX seeds.
    
    Operations:
    - Merge chains: Combine multiple chains
    - Branch from step: Create branch from existing step
    - Mutate chain: Create mutated variant
    - Prune chain: Remove low-weight steps
    - Interpolate chains: Create interpolation between chains
    """
    
    def __init__(self, storage: LTXChainStorage, hdc_dim: int = 131072):
        """
        Initialize synthesizer.
        
        Args:
            storage: Chain storage
            hdc_dim: HDC dimension
        """
        self.storage = storage
        self.hdc_dim = hdc_dim
        self.reconstructor = LTXChainReconstructor(hdc_dim)
    
    def merge_chains(self,
                     chain_ids: List[str],
                     merge_strategy: str = "concatenate") -> LTXChainSeed:
        """
        Merge multiple chains into a new chain.
        
        Args:
            chain_ids: List of chain IDs to merge
            merge_strategy: Merge strategy:
                - 'concatenate': Concatenate all steps
                - 'interleave': Interleave steps from each chain
                - 'bind': XOR bind chains together
                
        Returns:
            New merged chain
        """
        chains = [self.storage.load_chain(cid) for cid in chain_ids]
        chains = [c for c in chains if c is not None]
        
        if not chains:
            raise ValueError("No valid chains to merge")
        
        if merge_strategy == "concatenate":
            # Concatenate all steps
            all_steps = []
            for chain in chains:
                all_steps.extend(chain.steps)
            
            merged = LTXChainSeed(
                chain_id=f"merged_{hashlib.md5(str(chain_ids).encode()).hexdigest()[:12]}",
                model_name=chains[0].model_name,
                generation_mode=chains[0].generation_mode,
                steps=all_steps,
                metadata={'merged_from': chain_ids, 'strategy': merge_strategy}
            )
        
        elif merge_strategy == "interleave":
            # Interleave steps from each chain
            max_steps = max(len(c.steps) for c in chains)
            interleaved = []
            for i in range(max_steps):
                for chain in chains:
                    if i < len(chain.steps):
                        step = chain.steps[i]
                        # Create new step with updated ID
                        new_step = LTXSeedStep(
                            step_id=f"merged_{step.step_id}",
                            seed=step.seed,
                            hadamard_index=len(interleaved),
                            operation=step.operation,
                            weight=step.weight,
                            layer_name=step.layer_name,
                            timestep=step.timestep,
                            modality=step.modality,
                            metadata=step.metadata
                        )
                        interleaved.append(new_step)
            
            merged = LTXChainSeed(
                chain_id=f"merged_{hashlib.md5(str(chain_ids).encode()).hexdigest()[:12]}",
                model_name=chains[0].model_name,
                generation_mode=chains[0].generation_mode,
                steps=interleaved,
                metadata={'merged_from': chain_ids, 'strategy': merge_strategy}
            )
        
        elif merge_strategy == "bind":
            # XOR bind chains together
            # Reconstruct each chain and bind
            bound_vectors = []
            for chain in chains:
                v = self.reconstructor.reconstruct_chain(chain, strategy="bind")
                bound_vectors.append(v)
            
            # XOR all together
            result = bound_vectors[0].copy()
            for v in bound_vectors[1:]:
                result = np.bitwise_xor(result, v)
            
            # Create new chain with single step representing the bound result
            seed = int(hashlib.md5(result.tobytes()).hexdigest()[:16], 16)
            step = LTXSeedStep(
                step_id="bound_result",
                seed=seed,
                hadamard_index=0,
                operation=LTXChainOperation.BIND,
                weight=1.0,
                metadata={'bound_chains': chain_ids}
            )
            
            merged = LTXChainSeed(
                chain_id=f"bound_{hashlib.md5(str(chain_ids).encode()).hexdigest()[:12]}",
                model_name=chains[0].model_name,
                generation_mode=chains[0].generation_mode,
                steps=[step],
                metadata={'merged_from': chain_ids, 'strategy': merge_strategy}
            )
        
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        return merged
    
    def branch_from_step(self,
                         chain_id: str,
                         step_index: int,
                         branch_steps: List[LTXSeedStep]) -> LTXChainSeed:
        """
        Create a branch from an existing chain at a specific step.
        
        Args:
            chain_id: Source chain ID
            step_index: Index to branch from
            branch_steps: New steps for the branch
            
        Returns:
            New branched chain
        """
        chain = self.storage.load_chain(chain_id)
        if chain is None:
            raise ValueError(f"Chain not found: {chain_id}")
        
        if step_index >= len(chain.steps):
            raise ValueError(f"Step index out of range: {step_index}")
        
        # Copy steps up to branch point
        prefix_steps = chain.steps[:step_index + 1]
        
        # Add new branch steps
        all_steps = prefix_steps + branch_steps
        
        # Update hadamard indices
        for i, step in enumerate(all_steps):
            step.hadamard_index = i
        
        branched = LTXChainSeed(
            chain_id=f"branch_{chain_id}_{step_index}",
            model_name=chain.model_name,
            generation_mode=chain.generation_mode,
            steps=all_steps,
            metadata={'branched_from': chain_id, 'branch_point': step_index}
        )
        
        return branched
    
    def mutate_chain(self,
                     chain_id: str,
                     mutation_rate: float = 0.1,
                     mutation_type: str = "random") -> LTXChainSeed:
        """
        Create a mutated variant of a chain.
        
        Args:
            chain_id: Chain to mutate
            mutation_rate: Rate of mutation (0-1)
            mutation_type: Type of mutation:
                - 'random': Random seed changes
                - 'weight': Weight perturbation
                - 'shuffle': Shuffle step order
                
        Returns:
            Mutated chain
        """
        chain = self.storage.load_chain(chain_id)
        if chain is None:
            raise ValueError(f"Chain not found: {chain_id}")
        
        import random
        
        mutated_steps = []
        for i, step in enumerate(chain.steps):
            if mutation_type == "random":
                # Randomly change some seeds
                if random.random() < mutation_rate:
                    new_seed = random.randint(0, 2**63 - 1)
                    mutated_step = LTXSeedStep(
                        step_id=f"mutated_{step.step_id}",
                        seed=new_seed,
                        hadamard_index=step.hadamard_index,
                        operation=step.operation,
                        weight=step.weight,
                        layer_name=step.layer_name,
                        timestep=step.timestep,
                        modality=step.modality,
                        metadata={'mutated_from': step.seed}
                    )
                else:
                    mutated_step = step
                mutated_steps.append(mutated_step)
            
            elif mutation_type == "weight":
                # Perturb weights
                weight_delta = random.uniform(-mutation_rate, mutation_rate)
                new_weight = max(0.0, min(1.0, step.weight + weight_delta))
                mutated_step = LTXSeedStep(
                    step_id=step.step_id,
                    seed=step.seed,
                    hadamard_index=step.hadamard_index,
                    operation=step.operation,
                    weight=new_weight,
                    layer_name=step.layer_name,
                    timestep=step.timestep,
                    modality=step.modality,
                    metadata=step.metadata
                )
                mutated_steps.append(mutated_step)
        
        if mutation_type == "shuffle":
            # Shuffle step order
            mutated_steps = chain.steps.copy()
            random.shuffle(mutated_steps)
            # Update indices
            for i, step in enumerate(mutated_steps):
                step.hadamard_index = i
        
        mutated = LTXChainSeed(
            chain_id=f"mutated_{chain_id}",
            model_name=chain.model_name,
            generation_mode=chain.generation_mode,
            steps=mutated_steps,
            metadata={'mutated_from': chain_id, 'mutation_rate': mutation_rate, 'mutation_type': mutation_type}
        )
        
        return mutated
    
    def prune_chain(self,
                    chain_id: str,
                    min_weight: float = 0.1) -> LTXChainSeed:
        """
        Remove low-weight steps from a chain.
        
        Args:
            chain_id: Chain to prune
            min_weight: Minimum weight threshold
            
        Returns:
            Pruned chain
        """
        chain = self.storage.load_chain(chain_id)
        if chain is None:
            raise ValueError(f"Chain not found: {chain_id}")
        
        # Filter steps by weight
        pruned_steps = [s for s in chain.steps if s.weight >= min_weight]
        
        # Update indices
        for i, step in enumerate(pruned_steps):
            step.hadamard_index = i
        
        pruned = LTXChainSeed(
            chain_id=f"pruned_{chain_id}",
            model_name=chain.model_name,
            generation_mode=chain.generation_mode,
            steps=pruned_steps,
            metadata={'pruned_from': chain_id, 'min_weight': min_weight}
        )
        
        return pruned
    
    def interpolate_chains(self,
                           chain_id_1: str,
                           chain_id_2: str,
                           alpha: float = 0.5) -> LTXChainSeed:
        """
        Create an interpolation between two chains.
        
        Args:
            chain_id_1: First chain ID
            chain_id_2: Second chain ID
            alpha: Interpolation factor (0 = chain1, 1 = chain2)
            
        Returns:
            Interpolated chain
        """
        chain1 = self.storage.load_chain(chain_id_1)
        chain2 = self.storage.load_chain(chain_id_2)
        
        if chain1 is None or chain2 is None:
            raise ValueError("One or both chains not found")
        
        # Reconstruct vectors
        v1 = self.reconstructor.reconstruct_chain(chain1, strategy="bind")
        v2 = self.reconstructor.reconstruct_chain(chain2, strategy="bind")
        
        # Interpolate in bipolar space
        bipolar1 = np.where(np.unpackbits(v1.view(np.uint8)).reshape(-1,) > 0.5, 1.0, -1.0)[:self.hdc_dim]
        bipolar2 = np.where(np.unpackbits(v2.view(np.uint8)).reshape(-1,) > 0.5, 1.0, -1.0)[:self.hdc_dim]
        
        interpolated = (1 - alpha) * bipolar1 + alpha * bipolar2
        
        # Threshold back to binary
        binary = (interpolated > 0).astype(np.uint8)
        packed = np.packbits(binary)
        result_vec = packed.view(np.uint64)[:self.hdc_dim // 64]
        
        # Create new chain with interpolated result
        seed = int(hashlib.md5(result_vec.tobytes()).hexdigest()[:16], 16)
        step = LTXSeedStep(
            step_id="interpolated",
            seed=seed,
            hadamard_index=0,
            operation=LTXChainOperation.BIND,
            weight=1.0,
            metadata={'interpolated_from': [chain_id_1, chain_id_2], 'alpha': alpha}
        )
        
        interpolated_chain = LTXChainSeed(
            chain_id=f"interp_{chain_id_1}_{chain_id_2}",
            model_name=chain1.model_name,
            generation_mode=chain1.generation_mode,
            steps=[step],
            metadata={'interpolated_from': [chain_id_1, chain_id_2], 'alpha': alpha}
        )
        
        return interpolated_chain


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ltx_chain_system(storage_path: str = "./ltx_chains",
                            hdc_dim: int = 131072) -> Tuple[LTXChainStorage, LTXChainReconstructor, LTXChainSynthesizer]:
    """
    Create a complete LTX chain system.
    
    Args:
        storage_path: Path for chain storage
        hdc_dim: HDC dimension
        
    Returns:
        Tuple of (storage, reconstructor, synthesizer)
    """
    storage = LTXChainStorage(storage_path, hdc_dim)
    reconstructor = LTXChainReconstructor(hdc_dim)
    synthesizer = LTXChainSynthesizer(storage, hdc_dim)
    
    return storage, reconstructor, synthesizer


def create_ltx_chain_from_trajectory(model_name: str,
                                     generation_mode: str,
                                     timesteps: List[int],
                                     seeds: List[int],
                                     hadamard_indices: List[int],
                                     weights: Optional[List[float]] = None,
                                     layer_names: Optional[List[str]] = None,
                                     modalities: Optional[List[str]] = None) -> LTXChainSeed:
    """
    Create an LTX chain from trajectory data.
    
    Args:
        model_name: Model name
        generation_mode: Generation mode
        timesteps: List of timesteps
        seeds: List of seeds
        hadamard_indices: List of Hadamard indices
        weights: Optional list of weights
        layer_names: Optional list of layer names
        modalities: Optional list of modalities
        
    Returns:
        LTX chain seed
    """
    if weights is None:
        weights = [1.0] * len(seeds)
    if layer_names is None:
        layer_names = [""] * len(seeds)
    if modalities is None:
        modalities = ["joint"] * len(seeds)
    
    steps = []
    for i, (timestep, seed, h_idx, weight, layer, modality) in enumerate(
        zip(timesteps, seeds, hadamard_indices, weights, layer_names, modalities)
    ):
        step = LTXSeedStep(
            step_id=f"step_{i}",
            seed=seed,
            hadamard_index=h_idx,
            operation=LTXChainOperation.BIND,
            weight=weight,
            layer_name=layer,
            timestep=timestep,
            modality=modality
        )
        steps.append(step)
    
    chain = LTXChainSeed(
        chain_id=f"ltx_{hashlib.md5(str(seeds).encode()).hexdigest()[:12]}",
        model_name=model_name,
        generation_mode=generation_mode,
        steps=steps,
        total_timesteps=max(timesteps) if timesteps else 0
    )
    
    return chain
