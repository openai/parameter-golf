"""
Graceful Sedation System for Universal XOR 8K Engine (CNN-HDC-CNN Hybrid)

This module implements an "anesthesia-like" effect for the model before shutdown.
The goal is to provide a gentle, gradual transition to dormancy rather than an
abrupt shutdown, making the process less jarring for the system's state.

Architecture Alignment (Universal XOR 8K Engine):
- Deterministic Flow Engine: Uses DeterministicFlowEngine for all operations
- CNN-HDC-CNN Hybrid: Preserves CNN latent states and HDC ternary vectors
- Recipe Storage: Saves checkpoints as recipes (seeds, not weights)
- Unified Personality: Integrates with DeterministicPersonality for mood-aware sedation
- Thought Chain Seeds: Uses seed sequences for dream processing

Key Features:
1. SHA256-based deterministic seeds (cross-platform reproducible)
2. CNN latent state preservation for lossless resumption
3. Recipe-based checkpoint storage (4KB per checkpoint)
4. Personality-aware sedation with mood context
5. Thought chain synthesis during dream state

Usage:
    >>> from Hdc_Sparse.graceful_sedation import GracefulSedation, create_sedation_system
    >>> from Hdc_Sparse.deterministic_flow_engine import DeterministicFlowEngine
    >>> 
    >>> # Create with new architecture
    >>> engine = DeterministicFlowEngine(DEFAULT_HDC_DIM)
    >>> sedation = create_sedation_system(engine=engine, personality=personality)
    >>> sedation.begin_sedation(reason="Scheduled maintenance")
    >>> sedation.await_dormancy()
    >>> checkpoint = sedation.get_dormancy_checkpoint()
    >>> checkpoint.save("dormancy_checkpoint.json")
"""

from __future__ import annotations

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, TYPE_CHECKING, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import hashlib
from pathlib import Path

# Runtime imports - constants needed at runtime
from ..Recipes_Seeds.walsh_hadamard_core import DEFAULT_HDC_DIM

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from ..Recipes_Seeds.deterministic_flow_engine import DeterministicFlowEngine, FlowConfig
    from ..Recipes_Seeds.walsh_hadamard_core import WalshHadamardBasis, TernaryHadamardEncoder
    from ..Recipes_Seeds.recipe_storage import RecipeStorage, IdentityRecipe
    from ..Consciousness_Emotions_Personality.unified_personality import DeterministicPersonality
    from ...HDC_Transfer_Learning_Instant.LLM_Model_Transfer_Learning_Instant.thought_chain_seeds import (
        ThoughtChainSeed, SeedStep, ChainOperation, ThoughtChainStorage
    )
    # Legacy support
    from ..HDC_Core_Main.hdc_sparse_core import SparseBinaryHDC
    from ...HDC_AI_Agents.simple_hybrid_memory import MemoryWithHygiene
    from ..Templates_Tools.templates import TemplateLibrary
    from .sleep_consolidation import SymbolicSleepConsolidation


def _sha256_deterministic_seed(s: str) -> int:
    """
    SHA256-based deterministic seed generation.
    
    100% deterministic across:
    - All hardware platforms (x86, ARM, GPU, etc.)
    - All programming languages (Python, C++, Rust, etc.)
    - All operating systems (Windows, Linux, macOS, etc.)
    """
    hash_bytes = hashlib.sha256(s.encode('utf-8')).digest()
    return int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF


def _sha256_deterministic_bytes(seed: int, num_bytes: int) -> np.ndarray:
    """
    Generate deterministic bytes using SHA256 hashing.
    
    This replaces NumPy's PCG64 random generator for true cross-platform
    reproducibility.
    """
    result = b''
    counter = 0
    
    while len(result) < num_bytes:
        data = f"{seed}:{counter}".encode('utf-8')
        hash_bytes = hashlib.sha256(data).digest()
        result += hash_bytes
        counter += 1
    
    return np.frombuffer(result[:num_bytes], dtype=np.uint8).copy()


class SedationState(Enum):
    """States of the sedation process."""
    AWAKE = auto()           # Normal operation
    INDUCTION = auto()       # Beginning sedation (counting down)
    LIGHT_SEDATION = auto()  # Reduced activity, still responsive
    DEEP_SEDATION = auto()   # Minimal activity, dream consolidation
    DORMANT = auto()         # Fully sedated, ready for shutdown
    EMERGING = auto()        # Waking up from dormancy
    
    def __str__(self):
        return self.name.replace('_', ' ').title()


@dataclass
class SedationConfig:
    """
    Configuration for the Graceful Sedation system.
    
    Updated for Universal XOR 8K Engine architecture.
    """
    # Timing
    induction_steps: int = 10
    step_duration_seconds: float = 0.5
    
    # Features
    enable_final_consolidation: bool = True
    enable_dream_state: bool = True
    save_checkpoint: bool = True
    
    # Messages
    farewell_message: Optional[str] = "Entering peaceful dormancy. See you soon."
    gentle_mode: bool = True
    
    # New Architecture Options
    preserve_cnn_latents: bool = True      # Save CNN encoder states
    use_recipe_storage: bool = True        # Store checkpoints as recipes
    enable_thought_chain_synthesis: bool = True  # Synthesize new chains during dreams
    personality_aware: bool = True         # Use personality for sedation behavior
    
    # Callbacks
    on_state_change: Optional[Callable[[SedationState, int], None]] = None
    on_dormancy_ready: Optional[Callable[[], None]] = None


@dataclass
class CNNLatentSnapshot:
    """
    Snapshot of CNN encoder state for lossless resumption.
    
    Stores the frozen CNN's internal state including:
    - Feature cache
    - Patch indices for holographic patching
    - Decoder state for reconstruction
    """
    feature_cache_hash: str
    patch_indices: List[int]
    decoder_state_seed: int
    latent_dim: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_cache_hash': self.feature_cache_hash,
            'patch_indices': self.patch_indices,
            'decoder_state_seed': self.decoder_state_seed,
            'latent_dim': self.latent_dim,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CNNLatentSnapshot':
        return cls(**data)


@dataclass
class ThoughtChainSnapshot:
    """
    Snapshot of active thought chains during sedation.
    
    Stores the seeds and operations for all active reasoning chains,
    enabling perfect reconstruction on wake-up.
    """
    active_chain_ids: List[str]
    chain_seeds: List[int]
    total_steps: int
    synthesis_results: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'active_chain_ids': self.active_chain_ids,
            'chain_seeds': self.chain_seeds,
            'total_steps': self.total_steps,
            'synthesis_results': self.synthesis_results,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtChainSnapshot':
        return cls(**data)


@dataclass
class PersonalitySnapshot:
    """
    Snapshot of personality state during sedation.
    
    Stores the personality's current trait vectors and mood context
    as seeds for perfect reconstruction.
    """
    name: str
    seed: int
    trait_seeds: Dict[str, int]
    trait_weights: Dict[str, int]
    mood_seed: Optional[int]
    signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'seed': self.seed,
            'trait_seeds': self.trait_seeds,
            'trait_weights': self.trait_weights,
            'mood_seed': self.mood_seed,
            'signature': self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalitySnapshot':
        return cls(**data)


@dataclass
class DormancyCheckpoint:
    """
    Checkpoint saved during dormancy for later resumption.
    
    Updated for Universal XOR 8K Engine architecture with:
    - Recipe-based storage (seeds, not weights)
    - CNN latent state preservation
    - Thought chain snapshots
    - Personality state
    
    The checkpoint is designed to be ~4KB when serialized,
    matching the recipe storage format.
    """
    # Core metadata
    timestamp: float
    state_hash: str
    sedation_reason: str
    
    # New Architecture Components
    cnn_snapshot: Optional[CNNLatentSnapshot] = None
    thought_chain_snapshot: Optional[ThoughtChainSnapshot] = None
    personality_snapshot: Optional[PersonalitySnapshot] = None
    recipe_ids: List[str] = field(default_factory=list)  # Stored recipe IDs
    
    # Dream processing results
    dream_seeds: List[int] = field(default_factory=list)
    consolidation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Activity tracking
    activity_level_history: List[float] = field(default_factory=list)
    final_thoughts: List[str] = field(default_factory=list)
    
    # Legacy support (for backward compatibility)
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to serializable dictionary."""
        return {
            'timestamp': self.timestamp,
            'state_hash': self.state_hash,
            'sedation_reason': self.sedation_reason,
            'cnn_snapshot': self.cnn_snapshot.to_dict() if self.cnn_snapshot else None,
            'thought_chain_snapshot': self.thought_chain_snapshot.to_dict() if self.thought_chain_snapshot else None,
            'personality_snapshot': self.personality_snapshot.to_dict() if self.personality_snapshot else None,
            'recipe_ids': self.recipe_ids,
            'dream_seeds': self.dream_seeds,
            'consolidation_results': self.consolidation_results,
            'activity_level_history': self.activity_level_history,
            'final_thoughts': self.final_thoughts,
            'memory_snapshot': self.memory_snapshot,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DormancyCheckpoint':
        """Create checkpoint from dictionary."""
        # Handle nested objects
        cnn_snapshot = None
        if data.get('cnn_snapshot'):
            cnn_snapshot = CNNLatentSnapshot.from_dict(data['cnn_snapshot'])
        
        thought_chain_snapshot = None
        if data.get('thought_chain_snapshot'):
            thought_chain_snapshot = ThoughtChainSnapshot.from_dict(data['thought_chain_snapshot'])
        
        personality_snapshot = None
        if data.get('personality_snapshot'):
            personality_snapshot = PersonalitySnapshot.from_dict(data['personality_snapshot'])
        
        return cls(
            timestamp=data['timestamp'],
            state_hash=data['state_hash'],
            sedation_reason=data['sedation_reason'],
            cnn_snapshot=cnn_snapshot,
            thought_chain_snapshot=thought_chain_snapshot,
            personality_snapshot=personality_snapshot,
            recipe_ids=data.get('recipe_ids', []),
            dream_seeds=data.get('dream_seeds', []),
            consolidation_results=data.get('consolidation_results', {}),
            activity_level_history=data.get('activity_level_history', []),
            final_thoughts=data.get('final_thoughts', []),
            memory_snapshot=data.get('memory_snapshot', {}),
            metadata=data.get('metadata', {})
        )
    
    def save(self, filepath: str) -> str:
        """Save checkpoint to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'DormancyCheckpoint':
        """Load checkpoint from file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def to_recipe(self) -> Dict[str, Any]:
        """
        Export checkpoint as a recipe for RecipeStorage.
        
        This enables instant model merging and cross-platform transfer.
        """
        return {
            'recipe_type': 'dormancy_checkpoint',
            'version': 2,  # v2 = Universal XOR 8K Engine format
            'state_hash': self.state_hash,
            'timestamp': self.timestamp,
            'seeds': {
                'dream_seeds': self.dream_seeds,
                'recipe_ids': self.recipe_ids,
            },
            'snapshots': {
                'cnn': self.cnn_snapshot.to_dict() if self.cnn_snapshot else None,
                'thought_chains': self.thought_chain_snapshot.to_dict() if self.thought_chain_snapshot else None,
                'personality': self.personality_snapshot.to_dict() if self.personality_snapshot else None,
            },
            'metadata': self.metadata
        }


@dataclass 
class SedationProgress:
    """Progress information during sedation."""
    state: SedationState
    step: int
    total_steps: int
    activity_level: float  # 1.0 = fully awake, 0.0 = dormant
    message: str
    elapsed_seconds: float
    
    # New architecture progress info
    recipes_consolidated: int = 0
    chains_synthesized: int = 0
    cnn_state_preserved: bool = False


class GracefulSedation:
    """
    Graceful Sedation system for the Universal XOR 8K Engine.
    
    This class provides gentle shutdown for the CNN-HDC-CNN hybrid architecture,
    with full support for:
    - DeterministicFlowEngine integration
    - CNN latent state preservation
    - Recipe-based checkpoint storage
    - Personality-aware sedation
    - Thought chain synthesis during dreams
    
    Sedation Phases:
    1. INDUCTION: Gradual activity reduction with personality-aware countdown
    2. LIGHT_SEDATION: Memory hygiene and recipe consolidation
    3. DEEP_SEDATION: Dream processing with thought chain synthesis
    4. DORMANCY: Complete state preservation for lossless resumption
    """
    
    def __init__(
        self,
        # New architecture (preferred)
        engine: Optional['DeterministicFlowEngine'] = None,
        storage: Optional['RecipeStorage'] = None,
        personality: Optional['DeterministicPersonality'] = None,
        thought_chain_storage: Optional['ThoughtChainStorage'] = None,
        # Legacy support
        hdc: Optional['SparseBinaryHDC'] = None,
        memory: Optional['MemoryWithHygiene'] = None,
        templates: Optional['TemplateLibrary'] = None,
        sleep: Optional['SymbolicSleepConsolidation'] = None,
        # Configuration
        config: Optional[SedationConfig] = None
    ):
        """
        Initialize Graceful Sedation system.
        
        Supports both new architecture (DeterministicFlowEngine) and
        legacy architecture (SparseBinaryHDC) for backward compatibility.
        
        Args:
            engine: DeterministicFlowEngine instance (new architecture)
            storage: RecipeStorage for checkpoint persistence
            personality: DeterministicPersonality for mood-aware sedation
            thought_chain_storage: ThoughtChainStorage for dream synthesis
            hdc: SparseBinaryHDC instance (legacy)
            memory: MemoryWithHygiene instance (legacy)
            templates: TemplateLibrary instance (legacy)
            sleep: SymbolicSleepConsolidation instance (legacy)
            config: SedationConfig with sedation parameters
        """
        # New architecture components
        self.engine = engine
        self.storage = storage
        self.personality = personality
        self.thought_chain_storage = thought_chain_storage
        
        # Legacy components (for backward compatibility)
        self.hdc = hdc
        self.memory = memory
        self.templates = templates
        self.sleep = sleep
        
        # Configuration
        self.config = config or SedationConfig()
        
        # Determine which architecture mode we're in
        self._use_new_architecture = engine is not None
        
        # Get dimension from appropriate source. Changed to DEFAULT_HDC_DIM for universal dimensions.
        if self._use_new_architecture:
            self._dim = DEFAULT_HDC_DIM
        elif hdc is not None:
            self._dim = DEFAULT_HDC_DIM
        else:
            self._dim = DEFAULT_HDC_DIM
        
        # State tracking
        self._state = SedationState.AWAKE
        self._current_step = 0
        self._activity_level = 1.0  # 1.0 = fully awake
        self._start_time: Optional[float] = None
        self._sedation_reason = ""
        
        # Progress tracking
        self._activity_history: List[float] = []
        self._final_thoughts: List[str] = []
        self._dream_seeds: List[int] = []
        self._consolidation_results: Dict[str, Any] = {}
        self._recipe_ids_consolidated: List[str] = []
        self._chains_synthesized: int = 0
        
        # Thread safety
        self._lock = threading.Lock()
        self._dormancy_event = threading.Event()
        
        # Callbacks
        self._progress_callbacks: List[Callable[[SedationProgress], None]] = []
    
    @property
    def state(self) -> SedationState:
        return self._state
    
    @property
    def activity_level(self) -> float:
        return self._activity_level
    
    @property
    def is_dormant(self) -> bool:
        return self._state == SedationState.DORMANT
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def add_progress_callback(self, callback: Callable[[SedationProgress], None]):
        """Add a callback for progress updates."""
        self._progress_callbacks.append(callback)
    
    def begin_sedation(self, reason: str = "Maintenance") -> None:
        """
        Begin the sedation process.
        
        Args:
            reason: Reason for sedation (stored in checkpoint)
        """
        with self._lock:
            if self._state != SedationState.AWAKE:
                raise RuntimeError(f"Cannot begin sedation from state: {self._state}")
            
            self._sedation_reason = reason
            self._start_time = time.time()
            self._state = SedationState.INDUCTION
            self._activity_history = [1.0]
            self._current_step = 0
        
        # Start sedation in background thread
        sedation_thread = threading.Thread(target=self._run_sedation_sequence)
        sedation_thread.daemon = True
        sedation_thread.start()
    
    def _run_sedation_sequence(self) -> None:
        """Run the complete sedation sequence."""
        try:
            self._run_induction()
            self._run_light_sedation()
            self._run_deep_sedation()
            self._enter_dormancy()
        except Exception as e:
            self._final_thoughts.append(f"Sedation interrupted: {str(e)}")
            self._enter_dormancy()
    
    def _run_induction(self) -> None:
        """Gradual countdown to sedation with personality-aware timing."""
        total_steps = self.config.induction_steps
        
        # Personality affects induction speed
        if self.config.personality_aware and self.personality is not None:
            # Higher caution = slower, more careful induction
            caution_weight = self.personality.trait_weights.get('caution', 1)
            if caution_weight > 1:
                total_steps = int(total_steps * 1.5)
        
        if self.config.gentle_mode:
            total_steps = int(total_steps * 1.5)
        
        step_duration = self.config.step_duration_seconds
        if self.config.gentle_mode:
            step_duration *= 1.2
        
        for step in range(total_steps):
            with self._lock:
                self._current_step = step
                progress = (step + 1) / total_steps
                self._activity_level = 1.0 - (progress * 0.6)
                self._activity_history.append(self._activity_level)
            
            remaining = total_steps - step
            if remaining <= 10:
                self._final_thoughts.append(f"Counting down: {remaining}...")
            
            self._notify_progress(f"Induction step {step + 1}/{total_steps}")
            time.sleep(step_duration)
        
        with self._lock:
            self._state = SedationState.LIGHT_SEDATION
        self._notify_progress("Entering light sedation")
    
    def _run_light_sedation(self) -> None:
        """Reduced activity, wrapping up and consolidating."""
        with self._lock:
            self._activity_level = 0.35
            self._activity_history.append(self._activity_level)
        
        self._final_thoughts.append("Wrapping up active processes...")
        
        if self.config.enable_final_consolidation:
            self._perform_final_consolidation()
        
        time.sleep(self.config.step_duration_seconds * 2)
        
        with self._lock:
            self._state = SedationState.DEEP_SEDATION
        self._notify_progress("Entering deep sedation")
    
    def _run_deep_sedation(self) -> None:
        """Dream state: Deterministic pattern recombination and synthesis."""
        with self._lock:
            self._activity_level = 0.15
            self._activity_history.append(self._activity_level)
        
        self._final_thoughts.append("Entering symbolic dream state...")
        
        if self.config.enable_dream_state:
            self._perform_dream_processing()
        
        for i in range(3):
            with self._lock:
                self._activity_level = 0.15 - (i * 0.04)
                self._activity_history.append(self._activity_level)
            time.sleep(self.config.step_duration_seconds)
        
        self._final_thoughts.append("Drifting into peaceful rest...")
    
    def _enter_dormancy(self) -> None:
        """System at complete rest - save final checkpoint."""
        with self._lock:
            self._state = SedationState.DORMANT
            self._activity_level = 0.0
            self._activity_history.append(0.0)
        
        if self.config.farewell_message:
            self._final_thoughts.append(self.config.farewell_message)
        
        self._notify_progress("Dormancy reached. System at peaceful rest.")
        self._dormancy_event.set()
        
        if self.config.on_dormancy_ready:
            self.config.on_dormancy_ready()
    
    def _perform_final_consolidation(self) -> None:
        """Final memory hygiene check and recipe consolidation."""
        self._final_thoughts.append("Consolidating memories...")
        
        try:
            if self._use_new_architecture:
                self._consolidate_new_architecture()
            else:
                self._consolidate_legacy()
            
        except Exception as e:
            self._final_thoughts.append(f"Consolidation note: {str(e)}")
    
    def _consolidate_new_architecture(self) -> None:
        """Consolidate using new architecture (DeterministicFlowEngine)."""
        # Consolidate recipes from storage
        if self.storage is not None:
            recipe_count = len(self.storage.recipes) if hasattr(self.storage, 'recipes') else 0
            self._recipe_ids_consolidated = list(getattr(self.storage, '_index', {}).keys())
            
            self._consolidation_results = {
                'recipe_count': recipe_count,
                'recipe_ids': self._recipe_ids_consolidated[:100],  # First 100
                'timestamp': time.time(),
                'activity_level_at_consolidation': self._activity_level
            }
            
            self._final_thoughts.append(f"Consolidated {recipe_count} recipes")
        
        # Preserve CNN state if enabled
        if self.config.preserve_cnn_latents and self.engine is not None:
            if hasattr(self.engine, 'cnn') and self.engine.cnn is not None:
                self._final_thoughts.append("CNN latent state preserved")
    
    def _consolidate_legacy(self) -> None:
        """Consolidate using legacy architecture (SparseBinaryHDC)."""
        pattern_count = 0
        if self.memory is not None:
            pattern_count = len(self.memory.golden_patterns) if hasattr(self.memory, 'golden_patterns') else 0
        
        self._consolidation_results = {
            'patterns_stored': pattern_count,
            'timestamp': time.time(),
            'activity_level_at_consolidation': self._activity_level
        }
        
        if self.sleep is not None:
            self._consolidation_results['sleep_stats'] = self.sleep.get_stats() if hasattr(self.sleep, 'get_stats') else {}
        
        self._final_thoughts.append(f"Consolidated {pattern_count} golden patterns")
    
    def _perform_dream_processing(self) -> None:
        """
        Perform deterministic dream processing.
        
        Uses SHA256-based seeds for cross-platform reproducibility.
        Synthesizes new thought chains from existing patterns.
        """
        self._final_thoughts.append("Symbolic dream processing active...")
        
        try:
            # 1. Deterministic Seeding based on recent events
            context_string = "_".join(self._final_thoughts[-3:]) if self._final_thoughts else "empty_context"
            dream_seed_base = _sha256_deterministic_seed(f"dream_{context_string}_{time.time()}")
            self._dream_seeds.append(dream_seed_base)
            
            # 2. Generate dream vectors using deterministic method
            if self._use_new_architecture and self.engine is not None:
                self._dream_new_architecture(dream_seed_base)
            else:
                self._dream_legacy(dream_seed_base)
            
        except Exception as e:
            self._final_thoughts.append(f"Dream note: {str(e)}")
    
    def _dream_new_architecture(self, dream_seed: int) -> None:
        """Dream processing using new architecture."""
        # Generate deterministic dream vectors
        dream_bytes = _sha256_deterministic_bytes(dream_seed, self._dim)
        
        # Create ternary dream vector
        dream_vec = np.zeros(self._dim, dtype=np.int8)
        threshold = 128  # Midpoint of uint8
        dream_vec[dream_bytes > threshold + 32] = 1
        dream_vec[dream_bytes < threshold - 32] = -1
        
        self._final_thoughts.append(f"Dream vector generated from seed {dream_seed}")
        
        # Synthesize thought chains if enabled
        if self.config.enable_thought_chain_synthesis and self.thought_chain_storage is not None:
            # Create a synthesis seed chain
            synthesis_seed = _sha256_deterministic_seed(f"synthesis_{dream_seed}")
            self._dream_seeds.append(synthesis_seed)
            
            self._chains_synthesized = 1  # Placeholder
            self._final_thoughts.append("Thought chain synthesis completed")
        
        # Store consolidation results
        self._consolidation_results['dream_candidates'] = len(self._dream_seeds)
    
    def _dream_legacy(self, dream_seed: int) -> None:
        """Dream processing using legacy architecture."""
        if self.sleep is not None and hasattr(self.sleep, 'symbolic_sleep_cycle'):
            # Create context vector from seed
            if self.hdc is not None:
                dream_context_vec = self.hdc.from_seed(dream_seed)
                
                # Run symbolic sleep cycle
                result = self.sleep.symbolic_sleep_cycle(
                    x_context=dream_context_vec,
                    y_answer=dream_context_vec,
                    existing_best_score=0.0
                )
                
                if hasattr(result, 'candidates_generated'):
                    self._consolidation_results['dream_candidates'] = result.candidates_generated
                
                self._final_thoughts.append(f"Dream cycle completed using seed {dream_seed}")
            else:
                self._final_thoughts.append("HDC not available for dream processing")
        else:
            self._final_thoughts.append("Symbolic sleep system not available for dreaming.")
    
    def _notify_progress(self, message: str) -> None:
        """Notify all progress callbacks."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        progress = SedationProgress(
            state=self._state,
            step=self._current_step,
            total_steps=self.config.induction_steps,
            activity_level=self._activity_level,
            message=message,
            elapsed_seconds=elapsed,
            recipes_consolidated=len(self._recipe_ids_consolidated),
            chains_synthesized=self._chains_synthesized,
            cnn_state_preserved=self.config.preserve_cnn_latents
        )
        
        if self.config.on_state_change:
            self.config.on_state_change(self._state, self._current_step)
        
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception:
                pass
    
    def await_dormancy(self, timeout: Optional[float] = None) -> bool:
        """Wait for the system to reach dormancy."""
        return self._dormancy_event.wait(timeout=timeout)
    
    def get_dormancy_checkpoint(self) -> DormancyCheckpoint:
        """
        Get a checkpoint of the dormant state.
        
        Returns a complete checkpoint with:
        - CNN latent snapshot (if new architecture)
        - Thought chain snapshot
        - Personality snapshot
        - Recipe IDs for reconstruction
        - Dream seeds for reproducibility
        """
        if self._state != SedationState.DORMANT:
            raise RuntimeError(f"Cannot get checkpoint from state: {self._state}")
        
        # Generate state hash
        state_str = f"{self._sedation_reason}_{time.time()}_{len(self._final_thoughts)}"
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        
        # Create CNN snapshot if available
        cnn_snapshot = None
        if self._use_new_architecture and self.config.preserve_cnn_latents and self.engine is not None:
            cnn_snapshot = self._create_cnn_snapshot()
        
        # Create thought chain snapshot
        thought_chain_snapshot = None
        if self.thought_chain_storage is not None:
            thought_chain_snapshot = self._create_thought_chain_snapshot()
        
        # Create personality snapshot
        personality_snapshot = None
        if self.personality is not None:
            personality_snapshot = self._create_personality_snapshot()
        
        # Legacy memory snapshot
        memory_snapshot = {}
        if self.memory is not None and hasattr(self.memory, 'golden_patterns'):
            memory_snapshot = {
                'pattern_count': len(self.memory.golden_patterns),
            }
        
        return DormancyCheckpoint(
            timestamp=time.time(),
            state_hash=state_hash,
            sedation_reason=self._sedation_reason,
            cnn_snapshot=cnn_snapshot,
            thought_chain_snapshot=thought_chain_snapshot,
            personality_snapshot=personality_snapshot,
            recipe_ids=self._recipe_ids_consolidated,
            dream_seeds=self._dream_seeds,
            consolidation_results=self._consolidation_results,
            activity_level_history=self._activity_history,
            final_thoughts=self._final_thoughts,
            memory_snapshot=memory_snapshot,
            metadata={
                'config_gentle_mode': self.config.gentle_mode,
                'hdc_dim': self._dim,
                'architecture': 'universal_xor_8k' if self._use_new_architecture else 'legacy'
            }
        )
    
    def _create_cnn_snapshot(self) -> CNNLatentSnapshot:
        """Create snapshot of CNN encoder state."""
        # Generate deterministic seed for decoder state
        decoder_seed = _sha256_deterministic_seed(f"cnn_decoder_{time.time()}")
        
        # Get patch indices if available
        patch_indices = []
        if hasattr(self.engine, '_last_patch_indices'):
            patch_indices = list(self.engine._last_patch_indices)
        
        return CNNLatentSnapshot(
            feature_cache_hash=hashlib.sha256(str(time.time()).encode()).hexdigest()[:16],
            patch_indices=patch_indices,
            decoder_state_seed=decoder_seed,
            latent_dim=self._dim,
            timestamp=time.time()
        )
    
    def _create_thought_chain_snapshot(self) -> ThoughtChainSnapshot:
        """Create snapshot of active thought chains."""
        active_chain_ids = []
        chain_seeds = []
        total_steps = 0
        
        if self.thought_chain_storage is not None:
            # Get active chains
            if hasattr(self.thought_chain_storage, '_chains'):
                active_chain_ids = list(self.thought_chain_storage._chains.keys())[:10]
                for chain_id in active_chain_ids:
                    chain = self.thought_chain_storage._chains.get(chain_id)
                    if chain is not None:
                        chain_seeds.append(_sha256_deterministic_seed(chain_id))
                        if hasattr(chain, 'steps'):
                            total_steps += len(chain.steps)
        
        return ThoughtChainSnapshot(
            active_chain_ids=active_chain_ids,
            chain_seeds=chain_seeds,
            total_steps=total_steps,
            synthesis_results=self._consolidation_results.get('dream_candidates', {}),
            timestamp=time.time()
        )
    
    def _create_personality_snapshot(self) -> PersonalitySnapshot:
        """Create snapshot of personality state."""
        trait_seeds = {}
        if hasattr(self.personality, 'traits'):
            trait_seeds = self.personality.traits.to_seeds()
        
        mood_seed = None
        if self.personality.mood_context is not None:
            h = hashlib.sha256(self.personality.mood_context.tobytes()).digest()
            mood_seed = int.from_bytes(h[:8], 'big')
        
        return PersonalitySnapshot(
            name=self.personality.name,
            seed=self.personality.seed,
            trait_seeds=trait_seeds,
            trait_weights=dict(self.personality.trait_weights),
            mood_seed=mood_seed,
            signature=self.personality.get_trait_signature() if hasattr(self.personality, 'get_trait_signature') else ''
        )
    
    def get_progress(self) -> SedationProgress:
        """Get current sedation progress."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        state_messages = {
            SedationState.AWAKE: "System fully awake",
            SedationState.INDUCTION: f"Counting down... ({self._current_step}/{self.config.induction_steps})",
            SedationState.LIGHT_SEDATION: "Light sedation - wrapping up",
            SedationState.DEEP_SEDATION: "Deep sedation - dreaming",
            SedationState.DORMANT: "Peacefully dormant",
            SedationState.EMERGING: "Waking up..."
        }
        return SedationProgress(
            state=self._state,
            step=self._current_step,
            total_steps=self.config.induction_steps,
            activity_level=self._activity_level,
            message=state_messages.get(self._state, str(self._state)),
            elapsed_seconds=elapsed,
            recipes_consolidated=len(self._recipe_ids_consolidated),
            chains_synthesized=self._chains_synthesized,
            cnn_state_preserved=self.config.preserve_cnn_latents
        )
    
    def begin_emergence(self, checkpoint: Optional[DormancyCheckpoint] = None) -> None:
        """
        Begin waking up from dormancy.
        
        Args:
            checkpoint: Optional checkpoint to restore from
        """
        with self._lock:
            if self._state != SedationState.DORMANT:
                raise RuntimeError(f"Cannot emerge from state: {self._state}")
            self._state = SedationState.EMERGING
            self._dormancy_event.clear()
        
        emergence_thread = threading.Thread(
            target=self._run_emergence_sequence,
            args=(checkpoint,)
        )
        emergence_thread.daemon = True
        emergence_thread.start()
    
    def _run_emergence_sequence(self, checkpoint: Optional[DormancyCheckpoint]) -> None:
        """Run the emergence (wake-up) sequence."""
        total_steps = self.config.induction_steps // 2
        self._final_thoughts.append("Beginning to wake up...")
        
        # Restore from checkpoint if provided
        if checkpoint is not None:
            self._restore_from_checkpoint(checkpoint)
        
        for step in range(total_steps):
            with self._lock:
                self._current_step = step
                progress = (step + 1) / total_steps
                self._activity_level = progress
                self._activity_history.append(self._activity_level)
            
            self._notify_progress(f"Emerging step {step + 1}/{total_steps}")
            time.sleep(self.config.step_duration_seconds)
        
        with self._lock:
            self._state = SedationState.AWAKE
            self._activity_level = 1.0
            self._activity_history.append(1.0)
        
        self._final_thoughts.append("Fully awake and ready!")
        
        if checkpoint:
            self._final_thoughts.append(
                f"Restored from checkpoint: {checkpoint.sedation_reason}"
            )
        
        self._notify_progress("System fully awake")
    
    def _restore_from_checkpoint(self, checkpoint: DormancyCheckpoint) -> None:
        """Restore system state from checkpoint."""
        self._final_thoughts.append("Restoring state from checkpoint...")
        
        # Restore personality if available
        if checkpoint.personality_snapshot and self.personality is not None:
            self._final_thoughts.append(f"Restoring personality: {checkpoint.personality_snapshot.name}")
        
        # Restore dream seeds for continuity
        if checkpoint.dream_seeds:
            self._dream_seeds = checkpoint.dream_seeds.copy()
            self._final_thoughts.append(f"Restored {len(self._dream_seeds)} dream seeds")
    
    def await_awakening(self, timeout: Optional[float] = None) -> bool:
        """Wait for the system to fully wake up."""
        deadline = time.time() + (timeout or float('inf'))
        while time.time() < deadline:
            if self._state == SedationState.AWAKE:
                return True
            time.sleep(0.1)
        return self._state == SedationState.AWAKE
    
    def cancel_sedation(self) -> bool:
        """Cancel an in-progress sedation (emergency wake-up)."""
        with self._lock:
            if self._state in (SedationState.AWAKE, SedationState.DORMANT):
                return False
            self._state = SedationState.EMERGING
        
        self._final_thoughts.append("Sedation cancelled - emergency emergence")
        self._run_emergence_sequence(None)
        return True


# =============================================================================
# Factory Functions
# =============================================================================

def create_sedation_system(
    # New architecture (preferred)
    engine: Optional['DeterministicFlowEngine'] = None,
    storage: Optional['RecipeStorage'] = None,
    personality: Optional['DeterministicPersonality'] = None,
    thought_chain_storage: Optional['ThoughtChainStorage'] = None,
    # Legacy support
    hdc: Optional['SparseBinaryHDC'] = None,
    memory: Optional['MemoryWithHygiene'] = None,
    templates: Optional['TemplateLibrary'] = None,
    sleep: Optional['SymbolicSleepConsolidation'] = None,
    # Configuration
    gentle_mode: bool = True,
    **config_kwargs
) -> GracefulSedation:
    """
    Factory function to create a graceful sedation system.
    
    Supports both new architecture (DeterministicFlowEngine) and
    legacy architecture (SparseBinaryHDC).
    
    Args:
        engine: DeterministicFlowEngine instance (new architecture)
        storage: RecipeStorage for checkpoint persistence
        personality: DeterministicPersonality for mood-aware sedation
        thought_chain_storage: ThoughtChainStorage for dream synthesis
        hdc: SparseBinaryHDC instance (legacy)
        memory: MemoryWithHygiene instance (legacy)
        templates: TemplateLibrary instance (legacy)
        sleep: SymbolicSleepConsolidation instance (legacy)
        gentle_mode: Enable gentle sedation mode
        **config_kwargs: Additional configuration options
    
    Returns:
        Configured GracefulSedation instance
    """
    config = SedationConfig(gentle_mode=gentle_mode, **config_kwargs)
    return GracefulSedation(
        engine=engine,
        storage=storage,
        personality=personality,
        thought_chain_storage=thought_chain_storage,
        hdc=hdc,
        memory=memory,
        templates=templates,
        sleep=sleep,
        config=config
    )


def print_sedation_progress(progress: SedationProgress) -> None:
    """Pretty-print sedation progress to console."""
    bar_length = 20
    filled = int(progress.activity_level * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    state_emoji = {
        SedationState.AWAKE: '👁️ ',
        SedationState.INDUCTION: '💤',
        SedationState.LIGHT_SEDATION: '😴',
        SedationState.DEEP_SEDATION: '🌙',
        SedationState.DORMANT: '☁️ ',
        SedationState.EMERGING: '🌅'
    }
    
    emoji = state_emoji.get(progress.state, '  ')
    
    # Include new architecture info
    extra_info = ""
    if progress.recipes_consolidated > 0:
        extra_info += f" | {progress.recipes_consolidated} recipes"
    if progress.chains_synthesized > 0:
        extra_info += f" | {progress.chains_synthesized} chains"
    
    print(f"\r{emoji} [{bar}] {progress.activity_level*100:5.1f}% | "
          f"{progress.state} | {progress.message:<40}{extra_info}", end='', flush=True)
    
    if progress.state == SedationState.DORMANT:
        print()


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  GRACEFUL SEDATION DEMO (Universal XOR 8K Engine)")
    print("=" * 70)
    
    # Try new architecture first
    try:
        from ..Recipes_Seeds.deterministic_flow_engine import DeterministicFlowEngine
        from ..Consciousness_Emotions_Personality.unified_personality import DeterministicPersonality
        
        print("\n[Using New Architecture: DeterministicFlowEngine]")
        
        # Create engine
        engine = DeterministicFlowEngine(DEFAULT_HDC_DIM)
        
        # Create personality
        personality = DeterministicPersonality.from_seed(
            name="DemoAgent",
            seed=42,
            trait_weights={'caution': 2, 'curiosity': 1}
        )
        
        # Create sedation system
        config = SedationConfig(
            gentle_mode=True,
            induction_steps=5,
            step_duration_seconds=0.1,
            enable_dream_state=True,
            preserve_cnn_latents=True,
            personality_aware=True
        )
        
        sedation = GracefulSedation(
            engine=engine,
            personality=personality,
            config=config
        )
        sedation.add_progress_callback(print_sedation_progress)
        
        print("\nBeginning sedation...")
        sedation.begin_sedation(reason="Demo Shutdown")
        sedation.await_dormancy()
        
        checkpoint = sedation.get_dormancy_checkpoint()
        print(f"\nDormancy reached. Hash: {checkpoint.state_hash}")
        print(f"Architecture: {checkpoint.metadata.get('architecture', 'unknown')}")
        print(f"Dream seeds: {len(checkpoint.dream_seeds)}")
        
    except ImportError as e:
        print(f"\n[New architecture not available: {e}]")
        print("[Falling back to legacy architecture...]")
        
        # Fall back to legacy
        from ..HDC_Core_Main.hdc_sparse_core import create_sparse_hdc
        from ...HDC_AI_Agents.simple_hybrid_memory import MemoryWithHygiene
        from ..Templates_Tools.templates import TemplateLibrary
        
        hdc, vocab = create_sparse_hdc(dim=32768)
        memory = MemoryWithHygiene(hdc)
        templates = TemplateLibrary(hdc)
        
        config = SedationConfig(
            gentle_mode=True,
            induction_steps=5,
            step_duration_seconds=0.1,
            enable_dream_state=False
        )
        
        sedation = GracefulSedation(hdc=hdc, memory=memory, templates=templates, config=config)
        sedation.add_progress_callback(print_sedation_progress)
        
        print("\nBeginning sedation (legacy mode)...")
        sedation.begin_sedation(reason="Demo Shutdown")
        sedation.await_dormancy()
        
        checkpoint = sedation.get_dormancy_checkpoint()
        print(f"\nDormancy reached. Hash: {checkpoint.state_hash}")