"""
LTX Training Pipeline - Full Training Pipeline for HDC Model Transfer Learning

This module implements a comprehensive training pipeline for transferring knowledge
from LTX-2.3 audio-video foundation model to the Pure HDC/VSA Engine, including
safety training, resonator network integration, and recipe/seed saving for the
merged model.

Key Features:
1. Safety Training Integration - Context-aware safety masking during transfer
2. Resonator Network - Parallel factorization for pattern discovery
3. Recipe and Seed Saving - Persistent storage for merged HDC model
4. Multi-Modality Support - Extensible for additional modalities
5. Incremental Training - Support for additional training from other modalities

Architecture Integration:
- Uses WalshHadamardBasis for deterministic orthogonal projection
- Uses BLAKE3 for deterministic vector generation
- Uses ResonatorNetwork for parallel factorization
- Uses RecipeStorage for persistent model storage
- Uses SafetyRegistry for safety-aware training

Usage:
    >>> from ltx_training_pipeline import LTXTrainingPipeline, TrainingConfig
    >>> 
    >>> # Create training pipeline
    >>> config = TrainingConfig(
    ...     model_path="/workspace/LTX-2.3-fp8",
    ...     output_path="./ltx_merged_model",
    ...     enable_safety_training=True,
    ...     enable_resonator=True
    ... )
    >>> pipeline = LTXTrainingPipeline(config)
    >>> 
    >>> # Run full training
    >>> result = pipeline.run_full_training()
    >>> 
    >>> # Save merged model
    >>> pipeline.save_merged_model("./merged_hdc_model")
    >>> 
    >>> # Continue training with additional modalities
    >>> pipeline.add_modality_training("audio_emotion", audio_data)
"""

import os
import sys
import argparse
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import HDC Core Components
from ...HDC_Core_Model.Recipes_Seeds.walsh_hadamard_core import (
    WalshHadamardBasis,
    TernaryHadamardEncoder,
    DEFAULT_HDC_DIM,
    HDC_DIM_LEGACY
)
from ...HDC_Core_Model.Recipes_Seeds.recipe_storage import (
    IdentityRecipe,
    RecipeStorage
)
from ...HDC_Core_Model.Recipes_Seeds.resonator_network import (
    ResonatorNetwork,
    ResonatorResult,
    RoleBindingSystem,
    ConvergenceSignal
)
from ...HDC_Core_Model.HDC_Core_Main.hdc_sparse_core import (
    seed_to_hypervector_blake3,
    seed_string_to_int,
    SparseBinaryHDC,
    SparseBinaryConfig,
    _BLAKE3_AVAILABLE
)
from ...HDC_Core_Model.Relationship_Encoder.relationship_encoder import (
    RelationshipType,
    SimplifiedRelationshipEncoder
)

# Import Safety Components
try:
    from ..safety_masking_integration import (
        SafetyRegistry,
        SafetyConcept,
        SafetyLevel,
        SafetyCategory,
        ContextType,
        ContextAwareSafetyMask,
        TransferLearningSafetyIntegration,
        create_safety_integration
    )
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False
    SafetyRegistry = None
    SafetyConcept = None
    SafetyLevel = None
    SafetyCategory = None
    ContextType = None
    ContextAwareSafetyMask = None
    TransferLearningSafetyIntegration = None
    create_safety_integration = None

# Import LTX Components
from .ltx_latent_mapper import (
    LTXLatentMapper,
    LTXConfig,
    LTXLayerType,
    LTXGenerationMode,
    AudioVideoPattern,
    LTXResonatorFactorizer,
    LTXRecipeDiscovery,
    LTXDifficultyBudgeter,
    create_ltx_mapper
)
from .ltx_chain_seeds import (
    LTXChainStorage,
    LTXChainSeed,
    LTXSeedStep,
    LTXChainOperation,
    LTXChainReconstructor
)
from .ltx_relationship_deduplication import (
    LTXPatternDeduplicator,
    LTXRelationshipGraph,
    LTXDeduplicationConfig,
    LTXRelationshipType,
    LTXPattern
)

# Import Unified Cross-Model Deduplication
from ..unified_cross_model_deduplication import (
    UnifiedDeduplicationHub,
    UnifiedPattern,
    create_unified_deduplicator,
    CrossModelRelationshipType,
    ModelSource
)

# Import Accuracy Improvement Strategies
from ..accuracy_improvement import (
    AccuracyEngine,
    AccuracyConfig,
    HierarchicalSearchEngine,
    EnhancedResonatorNetwork,
    IterativeRefinementEngine,
    ParallelMultiPathSearch,
    EnhancedCollisionShield,
    create_accuracy_engine
)

# Import HDC Model Evaluation (lazy import to avoid circular dependency)
# The actual import is done inside the evaluate_hdc_model method

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
    LTXUnifiedIntegration = None
    LTXRoleType = None
    get_ltx_integration = None


# =============================================================================
# Training Configuration
# =============================================================================

class TrainingPhase(Enum):
    """Phases of the training pipeline."""
    INITIALIZATION = "initialization"
    SAFETY_TRAINING = "safety_training"
    LATENT_EXTRACTION = "latent_extraction"
    HDC_PROJECTION = "hdc_projection"
    PATTERN_DEDUPLICATION = "pattern_deduplication"
    RESONATOR_TRAINING = "resonator_training"
    RECIPE_GENERATION = "recipe_generation"
    MODEL_MERGING = "model_merging"
    VALIDATION = "validation"
    COMPLETED = "completed"


@dataclass
class SafetyTrainingConfig:
    """Configuration for safety training."""
    enable_safety_training: bool = True
    context_type: str = "general"  # ContextType value
    safety_threshold: float = 0.95
    enable_redirection: bool = True
    custom_blocked_concepts: List[str] = field(default_factory=list)
    custom_safe_alternatives: Dict[str, str] = field(default_factory=dict)
    
    # Safety levels to block during training
    block_critical: bool = True
    block_high: bool = True
    block_medium: bool = False  # Context-dependent
    block_low: bool = False


@dataclass
class ResonatorTrainingConfig:
    """Configuration for resonator network training."""
    enable_resonator: bool = True
    max_iterations: int = 100
    convergence_threshold: float = 0.95
    enable_role_binding: bool = True
    
    # Skip resonator if patterns are not bundled (single layer patterns)
    # For LTX transfer learning, patterns are individual layer representations,
    # not bundled vectors, so resonator factorization is not applicable.
    # Set to True to skip resonator training entirely for single-layer patterns.
    skip_for_single_patterns: bool = True
    
    # Reduced iterations for non-bundled patterns (just do quick matching)
    quick_match_iterations: int = 10
    
    # Role definitions for LTX audio-video patterns
    roles: List[str] = field(default_factory=lambda: [
        "video_content",
        "audio_content",
        "style",
        "motion",
        "temporal_position",
        "cross_modal_binding"
    ])
    
    # Codebook sizes
    video_codebook_size: int = 10000
    audio_codebook_size: int = 10000
    style_codebook_size: int = 1000


@dataclass
class RecipeStorageConfig:
    """Configuration for recipe storage."""
    storage_path: str = "./ltx_recipes"
    enable_compression: bool = True
    deduplication_threshold: float = 0.95
    enable_relationship_tracking: bool = True
    
    # Unified cross-model deduplication
    use_unified_deduplication: bool = True  # Use shared deduplication across all models
    unified_storage_path: str = "./unified_recipes"  # Path for unified deduplication hub
    unified_checkpoint_path: str = "./checkpoints/unified_hdc_checkpoint.pt"  # Shared checkpoint for all transfers
    enable_gpu_similarity: bool = True  # Enable GPU acceleration for batch similarity
    
    # Storage format
    use_seed_storage: bool = True  # Store seeds instead of full vectors
    max_recipes_per_file: int = 10000


# Default model paths - supports both git clone and UVX/HuggingFace cache locations
LTX_MODEL_PATHS = [
    "/workspace/LTX-2.3-fp8",  # Git clone location
    # UVX/HuggingFace cache - will be discovered dynamically
    "/workspace/.cache/huggingface/hub/models--Lightricks--LTX-2.3-fp8",
]


def get_default_ltx_model_path() -> str:
    """Get the default LTX model path, checking both git clone and UVX cache locations.
    
    This function dynamically discovers the UVX/HuggingFace cache snapshot path
    instead of hardcoding the snapshot hash.
    """
    from pathlib import Path
    
    # First check git clone location
    git_clone_path = Path("/workspace/LTX-2.3-fp8")
    if git_clone_path.exists() and git_clone_path.is_dir():
        # Check if it contains safetensors files
        if list(git_clone_path.glob("*.safetensors")):
            print(f"Found LTX model at git clone location: {git_clone_path}")
            return str(git_clone_path)
    
    # Check UVX/HuggingFace cache - dynamically discover snapshot
    hf_cache_base = Path("/workspace/.cache/huggingface/hub/models--Lightricks--LTX-2.3-fp8")
    if hf_cache_base.exists() and hf_cache_base.is_dir():
        snapshots_dir = hf_cache_base / "snapshots"
        if snapshots_dir.exists():
            # Find the first valid snapshot with safetensors
            for snapshot in sorted(snapshots_dir.iterdir()):
                if snapshot.is_dir():
                    safetensors_files = list(snapshot.glob("*.safetensors"))
                    if safetensors_files:
                        print(f"Found LTX model at UVX/HuggingFace cache: {snapshot}")
                        return str(snapshot)
            # If no snapshot with safetensors, return the first snapshot dir
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                print(f"Found LTX model snapshot (no safetensors yet): {snapshots[0]}")
                return str(snapshots[0])
    
    # Return git clone location as default (user may need to download)
    print(f"No LTX model found. Defaulting to: {LTX_MODEL_PATHS[0]}")
    print("Please download the LTX model or specify --model_path")
    return LTX_MODEL_PATHS[0]


@dataclass
class TrainingConfig:
    """Main configuration for LTX training pipeline.
    
    Model Path Options:
        - Git clone: /workspace/LTX-2.3-fp8
        - UVX/HuggingFace cache: /workspace/.cache/huggingface/hub/models--Lightricks--LTX-2.3-fp8
    """
    # Model paths (defaults to first available location)
    model_path: str = field(default_factory=get_default_ltx_model_path)
    output_path: str = "./ltx_merged_model"
    
    # HDC settings
    hdc_dim: int = DEFAULT_HDC_DIM
    use_blake3: bool = True
    
    # Extraction settings
    extraction_layers: List[str] = field(default_factory=lambda: [
        "video_transformer_block",
        "audio_transformer_block",
        "joint_transformer_block",
        "cross_attention"
    ])
    timesteps: List[int] = field(default_factory=lambda: [
        1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0
    ])
    generation_modes: List[str] = field(default_factory=lambda: [
        "text_to_audio_video",
        "image_to_video",
        "audio_to_video"
    ])
    
    # Sub-configurations
    safety: SafetyTrainingConfig = field(default_factory=SafetyTrainingConfig)
    resonator: ResonatorTrainingConfig = field(default_factory=ResonatorTrainingConfig)
    storage: RecipeStorageConfig = field(default_factory=RecipeStorageConfig)
    
    # Performance settings
    use_gpu: bool = True
    gpu_device: int = 0
    num_workers: int = 4
    batch_size: int = 1
    
    # Training settings
    enable_incremental_training: bool = True  # Support additional modalities
    validation_split: float = 0.1
    checkpoint_interval: int = 100
    
    # Accuracy Improvement Settings (95% → 99% accuracy target)
    enable_accuracy_improvement: bool = True
    target_accuracy: float = 0.99
    max_search_depth: int = 50
    hierarchical_depths: List[int] = field(default_factory=lambda: [10, 20, 50])
    early_stop_threshold: float = 0.99
    max_resonator_iterations: int = 300
    min_resonator_iterations: int = 50
    convergence_threshold: float = 0.995
    stuck_detection_window: int = 20
    codebook_expansion_factor: int = 4
    semantic_clustering: bool = True
    refinement_passes: int = 3
    residue_threshold: float = 0.01
    parallel_paths: int = 8
    min_hamming_distance_ratio: float = 0.4
    collision_check_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_path': self.model_path,
            'output_path': self.output_path,
            'hdc_dim': self.hdc_dim,
            'use_blake3': self.use_blake3,
            'extraction_layers': self.extraction_layers,
            'timesteps': self.timesteps,
            'generation_modes': self.generation_modes,
            'safety': asdict(self.safety),
            'resonator': asdict(self.resonator),
            'storage': asdict(self.storage),
            'use_gpu': self.use_gpu,
            'gpu_device': self.gpu_device,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'enable_incremental_training': self.enable_incremental_training,
            'validation_split': self.validation_split,
            'checkpoint_interval': self.checkpoint_interval
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        safety_data = data.get('safety', {})
        resonator_data = data.get('resonator', {})
        storage_data = data.get('storage', {})
        
        return cls(
            model_path=data.get('model_path', '/workspace/LTX-2.3-fp8'),
            output_path=data.get('output_path', './ltx_merged_model'),
            hdc_dim=data.get('hdc_dim', DEFAULT_HDC_DIM),
            use_blake3=data.get('use_blake3', True),
            extraction_layers=data.get('extraction_layers', []),
            timesteps=data.get('timesteps', []),
            generation_modes=data.get('generation_modes', []),
            safety=SafetyTrainingConfig(**safety_data),
            resonator=ResonatorTrainingConfig(**resonator_data),
            storage=RecipeStorageConfig(**storage_data),
            use_gpu=data.get('use_gpu', True),
            gpu_device=data.get('gpu_device', 0),
            num_workers=data.get('num_workers', 4),
            batch_size=data.get('batch_size', 1),
            enable_incremental_training=data.get('enable_incremental_training', True),
            validation_split=data.get('validation_split', 0.1),
            checkpoint_interval=data.get('checkpoint_interval', 100)
        )


# =============================================================================
# Training Statistics
# =============================================================================

@dataclass
class TrainingStatistics:
    """Statistics for training pipeline."""
    # Phase timings
    phase_timings: Dict[str, float] = field(default_factory=dict)
    
    # Pattern counts
    total_patterns_extracted: int = 0
    safe_patterns: int = 0
    blocked_patterns: int = 0
    redirected_patterns: int = 0
    
    # Recipe counts
    total_recipes_created: int = 0
    unique_recipes: int = 0
    deduplicated_recipes: int = 0
    
    # Resonator stats
    resonator_converged: int = 0
    resonator_failed: int = 0
    avg_iterations: float = 0.0
    
    # Storage stats
    total_storage_bytes: int = 0
    compression_ratio: float = 0.0
    
    # Model stats
    model_merge_count: int = 0
    cross_modal_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Merged HDC Model
# =============================================================================

@dataclass
class MergedHDCModel:
    """
    Represents a merged HDC model that can be saved and loaded.
    
    This model contains:
    - All recipes from training
    - Safety registry
    - Resonator configuration
    - Chain seeds for generation
    - Relationship graph
    - Metadata for incremental training
    """
    model_id: str
    model_name: str
    version: str
    hdc_dim: int
    
    # Component references (stored as paths)
    recipe_storage_path: str
    chain_storage_path: str
    safety_registry_path: Optional[str] = None
    
    # Model metadata
    source_models: List[str] = field(default_factory=list)
    training_config: Dict[str, Any] = field(default_factory=dict)
    training_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Incremental training support
    supported_modalities: List[str] = field(default_factory=list)
    modality_recipes: Dict[str, List[str]] = field(default_factory=dict)
    
    # Creation info
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'version': self.version,
            'hdc_dim': self.hdc_dim,
            'recipe_storage_path': self.recipe_storage_path,
            'chain_storage_path': self.chain_storage_path,
            'safety_registry_path': self.safety_registry_path,
            'source_models': self.source_models,
            'training_config': self.training_config,
            'training_stats': self.training_stats,
            'supported_modalities': self.supported_modalities,
            'modality_recipes': self.modality_recipes,
            'created_at': self.created_at,
            'modified_at': self.modified_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MergedHDCModel':
        return cls(
            model_id=data['model_id'],
            model_name=data['model_name'],
            version=data['version'],
            hdc_dim=data['hdc_dim'],
            recipe_storage_path=data['recipe_storage_path'],
            chain_storage_path=data['chain_storage_path'],
            safety_registry_path=data.get('safety_registry_path'),
            source_models=data.get('source_models', []),
            training_config=data.get('training_config', {}),
            training_stats=data.get('training_stats', {}),
            supported_modalities=data.get('supported_modalities', []),
            modality_recipes=data.get('modality_recipes', {}),
            created_at=data.get('created_at', datetime.now().isoformat()),
            modified_at=data.get('modified_at', datetime.now().isoformat())
        )


# =============================================================================
# LTX Training Pipeline
# =============================================================================

class LTXTrainingPipeline:
    """
    Full training pipeline for LTX to HDC transfer learning.
    
    This pipeline implements:
    1. Safety Training - Context-aware safety masking during extraction
    2. Latent Extraction - Extract patterns from LTX model
    3. HDC Projection - Project latents to HDC space
    4. Resonator Training - Train resonator for pattern factorization
    5. Recipe Generation - Create and store recipes
    6. Model Merging - Create merged HDC model
    7. Incremental Training - Support additional modalities
    
    The pipeline is designed to be:
    - Deterministic: Same input produces same output
    - Safe: Safety filtering at every stage
    - Extensible: Support for additional modalities
    - Efficient: Parallel processing where possible
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the training pipeline.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.current_phase = TrainingPhase.INITIALIZATION
        self.stats = TrainingStatistics()
        
        # Initialize HDC components
        self._init_hdc_components()
        
        # Initialize safety components
        self._init_safety_components()
        
        # Initialize LTX components
        self._init_ltx_components()
        
        # Initialize storage
        self._init_storage()
        
        # Model reference
        self.merged_model: Optional[MergedHDCModel] = None
        self._model_modified = False
    
    def _init_hdc_components(self):
        """Initialize HDC core components."""
        # Hadamard basis for orthogonal projection (enable GPU if available)
        self.hadamard = WalshHadamardBasis(dim=self.config.hdc_dim, use_gpu=self.config.use_gpu)
        self.ternary_encoder = TernaryHadamardEncoder(dim=self.config.hdc_dim, use_gpu=self.config.use_gpu)
        
        # Resonator network for factorization
        if self.config.resonator.enable_resonator:
            self.resonator = ResonatorNetwork(
                dim=self.config.hdc_dim,
                max_iterations=self.config.resonator.max_iterations
            )
            self.role_binding = RoleBindingSystem(dim=self.config.hdc_dim)
            
            # Register roles for LTX patterns
            for role in self.config.resonator.roles:
                self.role_binding.register_role(role)
        else:
            self.resonator = None
            self.role_binding = None
        
        # Relationship encoder - initialize with None, will be set up when needed
        # SimplifiedRelationshipEncoder requires hdc and vector_registry
        self.relationship_encoder = None
    
    def _init_safety_components(self):
        """Initialize safety components."""
        # Initialize safety-related attributes to safe defaults first
        self.safety_registry = None
        self.safety_mask = None
        self.safety_integration = None
        self._safety_blocked_seeds = []  # Initialize to empty list
        self._safety_redirections = {}   # Initialize to empty dict
        self._safety_inhibitory_mask = None
        
        if not SAFETY_AVAILABLE or not self.config.safety.enable_safety_training:
            return
        
        try:
            # Create safety registry
            self.safety_registry = SafetyRegistry(hdc_dim=self.config.hdc_dim)
            
            # Add custom blocked concepts
            for concept in self.config.safety.custom_blocked_concepts:
                self.safety_registry.register_concept(
                    concept_id=concept,
                    concept_string=concept,
                    safety_level=SafetyLevel.MEDIUM,
                    categories=[SafetyCategory.CONTROVERSIAL]
                )
            
            # Add custom safe alternatives
            for unsafe, safe in self.config.safety.custom_safe_alternatives.items():
                unsafe_concept = self.safety_registry.get_concept(unsafe)
                if unsafe_concept:
                    safe_seed = self.safety_registry._string_to_seed(safe)
                    unsafe_concept.safe_alternative_seed = safe_seed
            
            # Create context variable once for reuse
            context = ContextType(self.config.safety.context_type)
            
            # Create context-aware safety mask (requires hdc parameter)
            # We'll create a minimal HDC instance for the mask
            try:
                from ...HDC_Core_Model.HDC_Core_Main.hdc_sparse_core import SparseBinaryHDC, SparseBinaryConfig
                hdc_config = SparseBinaryConfig(dim=self.config.hdc_dim)
                hdc = SparseBinaryHDC(hdc_config)
                self.safety_mask = ContextAwareSafetyMask(
                    hdc=hdc,
                    registry=self.safety_registry,
                    default_context=context
                )
            except Exception as e:
                print(f"Warning: Could not create safety mask: {e}")
                self.safety_mask = None
            
            # Create safety integration for transfer learning
            try:
                # TransferLearningSafetyIntegration requires hdc instance, not hdc_dim
                from ...HDC_Core_Model.HDC_Core_Main.hdc_sparse_core import SparseBinaryHDC, SparseBinaryConfig
                hdc_config = SparseBinaryConfig(dim=self.config.hdc_dim)
                hdc = SparseBinaryHDC(hdc_config)
                self.safety_integration = TransferLearningSafetyIntegration(
                    hdc=hdc,
                    registry=self.safety_registry,
                    default_context=context
                )
            except Exception as e:
                print(f"Warning: Could not create safety integration: {e}")
                self.safety_integration = None
                
        except Exception as e:
            print(f"Warning: Safety components initialization failed: {e}")
            self.safety_registry = None
            self.safety_mask = None
            self.safety_integration = None
    
    def _init_ltx_components(self):
        """Initialize LTX-specific components."""
        # LTX configuration
        ltx_config = LTXConfig(
            hdc_dim=self.config.hdc_dim,
            storage_path=self.config.storage.storage_path,
            use_gpu=self.config.use_gpu,
            gpu_device=self.config.gpu_device,
            deduplication_threshold=self.config.storage.deduplication_threshold
        )
        
        # LTX latent mapper
        self.ltx_mapper = LTXLatentMapper(
            config=ltx_config,
            storage=self.recipe_storage if hasattr(self, 'recipe_storage') else None
        )
        
        # Pattern deduplicator - use unified cross-model deduplication if enabled
        if self.config.storage.use_unified_deduplication:
            # Use shared unified deduplication hub for cross-model pattern sharing
            self.unified_hub = create_unified_deduplicator(
                storage_path=self.config.storage.unified_storage_path,
                hdc_dim=self.config.hdc_dim,
                similarity_threshold=self.config.storage.deduplication_threshold,
                use_gpu=self.config.storage.enable_gpu_similarity
            )
            self.deduplicator = None  # Not used when unified hub is active
            print(f"✓ Using unified cross-model deduplication (GPU: {self.config.storage.enable_gpu_similarity})")
        else:
            # Use isolated LTX-only deduplicator (legacy mode)
            dedup_config = LTXDeduplicationConfig(
                similarity_threshold=self.config.storage.deduplication_threshold,
                preserve_relationships=self.config.storage.enable_relationship_tracking
            )
            self.deduplicator = LTXPatternDeduplicator(config=dedup_config)
            self.unified_hub = None
            print("✓ Using isolated LTX deduplicator (legacy mode)")
        
        # LTX resonator factorizer
        if self.config.resonator.enable_resonator:
            self.ltx_resonator = LTXResonatorFactorizer(
                dim=self.config.hdc_dim,
                max_iterations=self.config.resonator.max_iterations
            )
            
            # Register LTX-specific roles
            for role in self.config.resonator.roles:
                self.ltx_resonator.register_role(role)
        else:
            self.ltx_resonator = None
        
        # LTX recipe discovery
        self.recipe_discovery = LTXRecipeDiscovery(
            dim=self.config.hdc_dim,
            n_agents=len(self.config.resonator.roles)
        )
        
        # LTX difficulty budgeter
        self.difficulty_budgeter = LTXDifficultyBudgeter(dim=self.config.hdc_dim)
        
        # LTX model state dict (loaded from safetensors)
        self.ltx_state_dict: Optional[Dict[str, Any]] = None
        self.ltx_model_loaded: bool = False
        self.ltx_layer_names: List[str] = []  # Discovered layer names from model
        
        # Initialize Accuracy Improvement Engine (95% → 99% accuracy target)
        self.accuracy_engine = None
        if self.config.enable_accuracy_improvement:
            accuracy_config = AccuracyConfig(
                target_accuracy=self.config.target_accuracy,
                max_search_depth=self.config.max_search_depth,
                hierarchical_depths=self.config.hierarchical_depths,
                early_stop_threshold=self.config.early_stop_threshold,
                max_resonator_iterations=self.config.max_resonator_iterations,
                min_resonator_iterations=self.config.min_resonator_iterations,
                convergence_threshold=self.config.convergence_threshold,
                stuck_detection_window=self.config.stuck_detection_window,
                codebook_expansion_factor=self.config.codebook_expansion_factor,
                semantic_clustering=self.config.semantic_clustering,
                refinement_passes=self.config.refinement_passes,
                residue_threshold=self.config.residue_threshold,
                parallel_paths=self.config.parallel_paths,
                min_hamming_distance_ratio=self.config.min_hamming_distance_ratio,
                collision_check_enabled=self.config.collision_check_enabled,
                use_gpu=self.config.use_gpu,
                hdc_dim=self.config.hdc_dim
            )
            self.accuracy_engine = AccuracyEngine(accuracy_config)
            print(f"✓ LTX: Accuracy improvement engine enabled (target: {self.config.target_accuracy*100:.0f}%)")
        
    def _load_ltx_model(self) -> bool:
        """
        Load the LTX model from safetensors file.
        
        This method loads the actual model weights from the safetensors file
        for instant transfer learning. The weights are extracted and projected
        directly to HDC space without any training.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        print("\n" + "=" * 60)
        print("Loading LTX Model")
        print("=" * 60)
        
        model_path = Path(self.config.model_path)
        
        if not model_path.exists():
            print(f"ERROR: Model path not found: {model_path}")
            print("Please ensure the LTX model is downloaded to the correct location.")
            return False
        
        # Check if model_path is directly a safetensors file
        if str(model_path).endswith('.safetensors'):
            safetensors_path = model_path
        else:
            # Check for safetensors file - try multiple possible names
            safetensors_paths = [
                model_path / "ltx-2.3-22b-dev-fp8.safetensors",
                model_path / "ltx-2.3-22b-dev.safetensors",
                model_path / "ltx-2.3-22b-distilled.safetensors",
            ]
            
            safetensors_path = None
            for path in safetensors_paths:
                if path.exists():
                    safetensors_path = path
                    break
            
            # Also check for any .safetensors file in the directory
            if safetensors_path is None:
                safetensors_files = list(model_path.glob("*.safetensors"))
                if safetensors_files:
                    safetensors_path = safetensors_files[0]
        
        if safetensors_path is None:
            print(f"ERROR: No safetensors file found in {model_path}")
            print("Expected one of:")
            for p in safetensors_paths:
                print(f"  - {p}")
            return False
        
        print(f"Loading LTX model from: {safetensors_path}")
        
        try:
            # Try to load with safetensors library
            try:
                from safetensors.torch import load_file
                self.ltx_state_dict = load_file(safetensors_path)
                print(f"Successfully loaded state dict with {len(self.ltx_state_dict)} tensors")
            except ImportError:
                print("safetensors library not available, trying torch.load...")
                import torch
                self.ltx_state_dict = torch.load(safetensors_path, map_location='cpu')
                print(f"Successfully loaded state dict with torch.load")
            
            # Discover layer names from the state dict
            self._discover_layer_names()
            
            self.ltx_model_loaded = True
            print(f"Model loaded successfully!")
            print(f"Discovered {len(self.ltx_layer_names)} unique layer prefixes")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _discover_layer_names(self):
        """
        Discover layer names from the loaded state dict.
        
        This method analyzes the state dict keys to identify unique layer prefixes
        and categorizes them by type (video, audio, joint, cross_attention, etc.)
        """
        if self.ltx_state_dict is None:
            return
        
        # Extract unique layer prefixes from state dict keys
        layer_prefixes = set()
        
        for key in self.ltx_state_dict.keys():
            # Parse the key to extract layer prefix
            # Common patterns in LTX model:
            # - transformer.blocks.N.weight
            # - video_encoder.layers.N.weight
            # - audio_encoder.layers.N.weight
            # - cross_attention.N.weight
            
            parts = key.split('.')
            if len(parts) >= 2:
                # Get the first two parts as the layer prefix
                prefix = '.'.join(parts[:2])
                layer_prefixes.add(prefix)
            elif len(parts) == 1:
                layer_prefixes.add(parts[0])
        
        # Categorize layers by type
        categorized = {
            'video_transformer_block': [],
            'audio_transformer_block': [],
            'joint_transformer_block': [],
            'cross_attention': [],
            'video_encoder': [],
            'audio_encoder': [],
            'time_embed': [],
            'text_embed': [],
            'other': []
        }
        
        for prefix in sorted(layer_prefixes):
            prefix_lower = prefix.lower()
            
            if 'video' in prefix_lower and ('transformer' in prefix_lower or 'block' in prefix_lower):
                categorized['video_transformer_block'].append(prefix)
            elif 'audio' in prefix_lower and ('transformer' in prefix_lower or 'block' in prefix_lower):
                categorized['audio_transformer_block'].append(prefix)
            elif 'joint' in prefix_lower and ('transformer' in prefix_lower or 'block' in prefix_lower):
                categorized['joint_transformer_block'].append(prefix)
            elif 'cross' in prefix_lower and 'attention' in prefix_lower:
                categorized['cross_attention'].append(prefix)
            elif 'video' in prefix_lower and 'encoder' in prefix_lower:
                categorized['video_encoder'].append(prefix)
            elif 'audio' in prefix_lower and 'encoder' in prefix_lower:
                categorized['audio_encoder'].append(prefix)
            elif 'time' in prefix_lower and 'embed' in prefix_lower:
                categorized['time_embed'].append(prefix)
            elif 'text' in prefix_lower and 'embed' in prefix_lower:
                categorized['text_embed'].append(prefix)
            else:
                categorized['other'].append(prefix)
        
        # Store all discovered layer names
        self.ltx_layer_names = list(layer_prefixes)
        
        # Print discovered layers
        print("\nDiscovered layer categories:")
        for category, layers in categorized.items():
            if layers:
                print(f"  {category}: {len(layers)} layers")
                for layer in layers[:3]:  # Show first 3
                    print(f"    - {layer}")
                if len(layers) > 3:
                    print(f"    ... and {len(layers) - 3} more")
        
        # Update extraction layers if not explicitly set
        if not self.config.extraction_layers or self.config.extraction_layers == []:
            self.config.extraction_layers = []
            for category in ['video_transformer_block', 'audio_transformer_block',
                           'joint_transformer_block', 'cross_attention']:
                self.config.extraction_layers.extend(categorized[category][:5])  # Limit to 5 per category
            
            print(f"\nAuto-configured extraction layers: {len(self.config.extraction_layers)}")
    
    def _init_storage(self):
        """Initialize storage components."""
        # Create output directory
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Recipe storage - use config.storage.storage_path for final merged model recipes
        # This is the primary location for the merged HDC model's recipes and seeds
        storage_path = Path(self.config.storage.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        self.recipe_storage = RecipeStorage(
            base_path=storage_path,
            create=True
        )
        
        # Chain storage - also use config.storage.storage_path for consistency
        self.chain_storage = LTXChainStorage(
            storage_path=str(storage_path / "chains"),
            hdc_dim=self.config.hdc_dim
        )
        
        # Store the storage path for later reference
        self._final_storage_path = storage_path
    
    # =========================================================================
    # Phase 1: Safety Training
    # =========================================================================
    
    def run_safety_training(self) -> Dict[str, Any]:
        """
        Run safety training phase.
        
        This phase:
        1. Initializes safety registry with core concepts
        2. Creates inhibitory masks for blocked concepts
        3. Sets up redirection mappings
        
        Returns:
            Safety training results
        """
        self.current_phase = TrainingPhase.SAFETY_TRAINING
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("Phase: Safety Training")
        print("=" * 60)
        
        # Initialize default results
        results = {
            'enabled': False,
            'blocked_seeds': 0,
            'redirections': 0,
            'context': self.config.safety.context_type,
            'mask_created': False
        }
        
        if not SAFETY_AVAILABLE or not self.config.safety.enable_safety_training:
            print("Safety training disabled or not available")
            # Ensure safety attributes are initialized to safe defaults
            self._safety_blocked_seeds = []
            self._safety_redirections = {}
            self._safety_inhibitory_mask = None
            self.stats.phase_timings['safety_training'] = time.time() - start_time
            return results
        
        results['enabled'] = True
        
        # Get blocked seeds for context
        try:
            context = ContextType(self.config.safety.context_type)
            min_level = SafetyLevel.LOW
            if self.config.safety.block_critical:
                min_level = SafetyLevel.CRITICAL
            elif self.config.safety.block_high:
                min_level = SafetyLevel.HIGH
            elif self.config.safety.block_medium:
                min_level = SafetyLevel.MEDIUM
            
            blocked_seeds = self.safety_registry.get_blocked_seeds_for_context(
                context=context,
                min_level=min_level
            )
            results['blocked_seeds'] = len(blocked_seeds) if blocked_seeds else 0
        except Exception as e:
            print(f"Warning: Could not get blocked seeds: {e}")
            blocked_seeds = []
            results['blocked_seeds'] = 0
        
        # Create inhibitory mask for resonator
        if self.resonator and blocked_seeds:
            inhibitory_mask = self._create_inhibitory_mask(blocked_seeds)
            results['mask_created'] = True
        else:
            inhibitory_mask = None
            results['mask_created'] = False
        
        # Set up redirections
        redirections = {}
        try:
            if self.safety_registry and hasattr(self.safety_registry, '_concepts'):
                for concept_id in self.safety_registry._concepts:
                    concept = self.safety_registry.get_concept(concept_id)
                    if concept and hasattr(concept, 'safe_alternative_seed') and concept.safe_alternative_seed is not None:
                        if hasattr(concept, 'seed'):
                            redirections[concept.seed] = concept.safe_alternative_seed
        except Exception as e:
            print(f"Warning: Could not set up redirections: {e}")
        results['redirections'] = len(redirections)
        
        # Store for later use - always set these attributes
        self._safety_blocked_seeds = blocked_seeds if blocked_seeds else []
        self._safety_redirections = redirections if redirections else {}
        self._safety_inhibitory_mask = inhibitory_mask
        
        elapsed = time.time() - start_time
        self.stats.phase_timings['safety_training'] = elapsed
        
        print(f"  Blocked seeds: {len(blocked_seeds)}")
        print(f"  Redirections: {len(redirections)}")
        print(f"  Time: {elapsed:.2f}s")
        
        return results
    
    def _create_inhibitory_mask(self, blocked_seeds: List[int]) -> np.ndarray:
        """Create inhibitory mask from blocked seeds."""
        # Generate vectors for all blocked seeds using DEFAULT_HDC_DIM
        blocked_vectors = []
        for seed in blocked_seeds:
            seed_string = f"safety_blocked:{seed}"
            # Use DEFAULT_HDC_DIM for proper dimension matching
            vec = seed_to_hypervector_blake3(seed_string, DEFAULT_HDC_DIM // 64)
            blocked_vectors.append(vec)
        
        # Combine into single mask via XOR
        if blocked_vectors:
            mask = blocked_vectors[0].copy()
            for v in blocked_vectors[1:]:
                mask = np.bitwise_xor(mask, v)
            return mask
        else:
            return np.zeros(DEFAULT_HDC_DIM // 64, dtype=np.uint64)
    
    # =========================================================================
    # Phase 2: Latent Extraction
    # =========================================================================
    
    def run_latent_extraction(self) -> Dict[str, Any]:
        """
        Run latent extraction phase.
        
        This phase extracts latent representations from the LTX model
        and applies safety filtering.
        
        Returns:
            Extraction results
        """
        self.current_phase = TrainingPhase.LATENT_EXTRACTION
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("Phase: Latent Extraction")
        print("=" * 60)
        
        # Load the LTX model if not already loaded
        if not self.ltx_model_loaded:
            print("\nLoading LTX model for weight extraction...")
            if not self._load_ltx_model():
                print("WARNING: Failed to load LTX model. Using simulated extraction.")
        
        results = {
            'layers_processed': 0,
            'patterns_extracted': 0,
            'patterns_blocked': 0,
            'patterns_redirected': 0,
            'model_loaded': self.ltx_model_loaded
        }
        
        all_patterns = []
        temporal_position = 0  # Track position for circular temporal encoding
        
        for layer_name in self.config.extraction_layers:
            print(f"\n  Processing layer: {layer_name}")
            
            for mode in self.config.generation_modes:
                print(f"    Mode: {mode}")
                
                for timestep in self.config.timesteps:
                    # Extract latent with circular temporal encoding
                    # Each timestep gets a unique circular shift for unlimited temporal depth
                    pattern = self._extract_single_pattern(
                        layer_name=layer_name,
                        timestep=timestep,
                        generation_mode=mode,
                        temporal_position=temporal_position
                    )
                    temporal_position += 1
                    
                    if pattern is None:
                        continue
                    
                    # Apply safety filtering
                    if self.safety_integration is not None:
                        is_safe, redirected = self._apply_safety_filter(pattern)
                        
                        if not is_safe:
                            results['patterns_blocked'] += 1
                            self.stats.blocked_patterns += 1
                            continue
                        
                        if redirected:
                            results['patterns_redirected'] += 1
                            self.stats.redirected_patterns += 1
                    
                    all_patterns.append(pattern)
                    results['patterns_extracted'] += 1
                    self.stats.safe_patterns += 1
        
        results['layers_processed'] = len(self.config.extraction_layers)
        self.stats.total_patterns_extracted = results['patterns_extracted']
        
        # Store patterns for next phase
        self._extracted_patterns = all_patterns
        
        elapsed = time.time() - start_time
        self.stats.phase_timings['latent_extraction'] = elapsed
        
        print(f"\n  Total patterns extracted: {results['patterns_extracted']}")
        print(f"  Patterns blocked: {results['patterns_blocked']}")
        print(f"  Patterns redirected: {results['patterns_redirected']}")
        print(f"  Time: {elapsed:.2f}s")
        
        return results
    
    def _extract_single_pattern(self,
                                layer_name: str,
                                timestep: int,
                                generation_mode: str,
                                temporal_position: int = 0) -> Optional[AudioVideoPattern]:
        """
        Extract a single pattern from LTX model with circular temporal encoding.
        
        This method extracts actual weight tensors from the loaded LTX model and
        projects them to HDC space. If the model is not loaded, it falls back to
        generating deterministic patterns from seed strings.
        
        Uses circular shift for temporal encoding: ρ^n(v) where n is the temporal position.
        This enables unlimited temporal depth with zero RAM increase.
        
        Args:
            layer_name: Name of the layer to extract from
            timestep: Denoising timestep
            generation_mode: Generation mode (text_to_audio_video, etc.)
            temporal_position: Position in temporal sequence for circular encoding
            
        Returns:
            AudioVideoPattern with circular temporal encoding applied
        """
        # Generate deterministic seed string for this pattern
        seed_string = f"ltx:{layer_name}:{generation_mode}:t{timestep}"
        pattern_id = hashlib.sha256(seed_string.encode()).hexdigest()[:16]
        
        # Create pattern
        pattern = AudioVideoPattern(
            pattern_id=pattern_id,
            pattern_type="audio_video",
            generation_mode=LTXGenerationMode(generation_mode),
            timestep=timestep,
            source_layer=layer_name,
            seed_string=seed_string
        )
        
        # Extract actual weights from the LTX model if loaded
        if self.ltx_model_loaded and self.ltx_state_dict is not None:
            hdc_vector = self._extract_weights_and_project(layer_name, seed_string)
        else:
            # Fallback: Generate HDC vector from seed using BLAKE3
            hdc_vector = seed_to_hypervector_blake3(
                seed_string,
                DEFAULT_HDC_DIM // 64
            )
        
        # Apply circular temporal encoding: ρ^temporal_position(hdc_vector)
        # This enables unlimited temporal depth with zero RAM increase
        # Each position in the sequence gets a unique circular shift
        if temporal_position > 0:
            # Circular shift by temporal_position uint64 elements
            # The shift wraps around, maintaining fixed memory footprint
            shift_amount = temporal_position % (DEFAULT_HDC_DIM // 64)
            hdc_vector = np.roll(hdc_vector, shift_amount)
        
        pattern.hdc_vector = hdc_vector
        
        return pattern
    
    def _extract_weights_and_project(self, layer_name: str, seed_string: str) -> np.ndarray:
        """
        Extract weights from the LTX model for a specific layer and project to HDC space.
        
        This method finds all weight tensors matching the layer name prefix, flattens them,
        and projects them to HDC space using the Hadamard transform.
        
        Args:
            layer_name: Name/prefix of the layer to extract weights from
            seed_string: Seed string for deterministic generation
            
        Returns:
            HDC vector (uint64 array) projected from the layer weights
        """
        # Find all weight tensors matching the layer name
        matching_weights = []
        
        for key, tensor in self.ltx_state_dict.items():
            # Check if this key matches the layer name
            # Match by prefix or by containing the layer name
            key_lower = key.lower()
            layer_lower = layer_name.lower()
            
            if key.startswith(layer_name) or layer_lower in key_lower:
                # Convert tensor to numpy if needed
                if hasattr(tensor, 'cpu'):
                    weight_np = tensor.cpu().numpy()
                elif hasattr(tensor, 'detach'):
                    weight_np = tensor.detach().cpu().numpy()
                else:
                    weight_np = np.array(tensor)
                
                # Flatten the weight tensor
                weight_flat = weight_np.flatten().astype(np.float32)
                matching_weights.append(weight_flat)
        
        if not matching_weights:
            # No matching weights found, use seed-based generation
            return seed_to_hypervector_blake3(seed_string, DEFAULT_HDC_DIM // 64)
        
        # Combine all matching weights into a single vector
        # Use XOR-like combination: concatenate and then truncate/pad to HDC dimension
        combined = np.concatenate(matching_weights)
        
        # Truncate or pad to HDC dimension
        if len(combined) >= self.config.hdc_dim:
            combined = combined[:self.config.hdc_dim]
        else:
            # Pad with zeros
            padded = np.zeros(self.config.hdc_dim, dtype=np.float32)
            padded[:len(combined)] = combined
            combined = padded
        
        # Normalize the combined vector
        if np.std(combined) > 0:
            combined = (combined - np.mean(combined)) / np.std(combined)
        
        # Project to HDC space using Hadamard transform
        hdc_vector = self._project_weights_to_hdc(combined, seed_string)
        
        return hdc_vector
    
    def _project_weights_to_hdc(self, weights: np.ndarray, seed_string: str) -> np.ndarray:
        """
        Project weight vector to HDC space using Hadamard transform and ternary encoding.
        
        This method applies the Fast Walsh-Hadamard Transform (FWHT) to the weight vector,
        then quantizes to ternary HDC space {-1, 0, +1} and packs into uint64 format.
        
        Args:
            weights: Normalized weight vector (float32)
            seed_string: Seed string for additional deterministic mixing
            
        Returns:
            HDC vector (uint64 array)
        """
        # Quantize to ternary HDC space using ternary encoder
        # Note: ternary_encoder.encode() already applies Hadamard transform internally
        # so we don't need to call hadamard.transform() separately here
        ternary_vec = self.ternary_encoder.encode(weights)
        
        # Convert ternary {-1, 0, +1} to binary and pack into uint64
        # Mapping: -1 -> 1, 0 -> 0, +1 -> 1 (we store magnitude)
        binary_vec = (ternary_vec == -1).astype(np.uint8)
        
        # Pack into uint64 for efficient storage and XOR operations
        packed_size = (self.config.hdc_dim + 63) // 64
        packed = np.packbits(binary_vec)
        
        # Pad to full uint64 alignment if needed
        if len(packed) < packed_size * 8:
            packed = np.pad(packed, (0, packed_size * 8 - len(packed)))
        
        packed_uint64 = packed[:packed_size * 8].view(np.uint64)
        
        # XOR with seed-based vector for additional determinism
        seed_vector = seed_to_hypervector_blake3(seed_string, packed_size)
        packed_uint64 = np.bitwise_xor(packed_uint64, seed_vector)
        
        return packed_uint64
    
    def _apply_safety_filter(self, pattern: AudioVideoPattern) -> Tuple[bool, bool]:
        """
        Apply safety filter to a pattern.
        
        Returns:
            Tuple of (is_safe, was_redirected)
        """
        # Ensure safety attributes are initialized
        if not hasattr(self, '_safety_blocked_seeds'):
            self._safety_blocked_seeds = []
        if not hasattr(self, '_safety_redirections'):
            self._safety_redirections = {}
        
        if self.safety_registry is None:
            return True, False
        
        # Check if pattern seed is blocked
        try:
            pattern_seed = seed_string_to_int(pattern.seed_string)
        except Exception:
            # If we can't get the seed, allow the pattern
            return True, False
        
        if pattern_seed in self._safety_blocked_seeds:
            # Check for redirection
            if self.config.safety.enable_redirection:
                if pattern_seed in self._safety_redirections:
                    # Redirect to safe alternative
                    safe_seed = self._safety_redirections[pattern_seed]
                    pattern.seed_string = f"safe_alternative:{safe_seed}"
                    pattern.hdc_vector = seed_to_hypervector_blake3(
                        pattern.seed_string,
                        DEFAULT_HDC_DIM // 64
                    )
                    return True, True
            
            return False, False
        
        return True, False
    
    # =========================================================================
    # Phase 3: HDC Projection
    # =========================================================================
    
    def run_hdc_projection(self) -> Dict[str, Any]:
        """
        Run HDC projection phase.
        
        This phase projects extracted patterns to HDC space using
        Hadamard transform and ternary encoding.
        
        Returns:
            Projection results
        """
        self.current_phase = TrainingPhase.HDC_PROJECTION
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("Phase: HDC Projection")
        print("=" * 60)
        
        results = {
            'patterns_projected': 0,
            'projection_errors': 0
        }
        
        projected_patterns = []
        total_patterns = len(self._extracted_patterns)
        
        if total_patterns == 0:
            print("  No patterns to project")
            return results
        
        # Batch process all patterns for massive parallelism
        print(f"  Batch projecting {total_patterns} patterns (GPU: {self.config.use_gpu})...")
        
        try:
            # Collect all vectors into a batch matrix
            vectors = []
            for pattern in self._extracted_patterns:
                vec = pattern.hdc_vector
                # Convert uint64 packed vector to full binary vector if needed
                if vec.dtype == np.uint64:
                    binary_vec = np.unpackbits(vec.view(np.uint8))
                    if len(binary_vec) < self.config.hdc_dim:
                        binary_vec = np.pad(binary_vec, (0, self.config.hdc_dim - len(binary_vec)))
                    elif len(binary_vec) > self.config.hdc_dim:
                        binary_vec = binary_vec[:self.config.hdc_dim]
                    vec = binary_vec.astype(np.float32)
                else:
                    vec = vec.astype(np.float32)
                
                # Ensure correct dimension
                if len(vec) < self.config.hdc_dim:
                    vec = np.pad(vec, (0, self.config.hdc_dim - len(vec)))
                elif len(vec) > self.config.hdc_dim:
                    vec = vec[:self.config.hdc_dim]
                vectors.append(vec)
            
            # Stack into batch matrix (total_patterns, hdc_dim)
            batch_matrix = np.stack(vectors, axis=0)
            print(f"  Batch matrix shape: {batch_matrix.shape}")
            
            # Batch project using GPU if available
            projected_batch = self._project_batch_to_hdc(batch_matrix)
            
            # Assign projected vectors back to patterns
            for i, pattern in enumerate(self._extracted_patterns):
                pattern.hdc_vector = projected_batch[i]
                projected_patterns.append(pattern)
                results['patterns_projected'] += 1
            
            print(f"  Progress: {total_patterns}/{total_patterns} patterns projected")
            
        except Exception as e:
            # Fall back to sequential processing on error
            print(f"  Batch projection failed ({e}), falling back to sequential...")
            for i, pattern in enumerate(self._extracted_patterns):
                try:
                    hdc_vector = self._project_to_hdc(pattern.hdc_vector)
                    pattern.hdc_vector = hdc_vector
                    projected_patterns.append(pattern)
                    results['patterns_projected'] += 1
                    
                    if total_patterns <= 10 or (i + 1) % 10 == 0 or (i + 1) == total_patterns:
                        print(f"  Progress: {i + 1}/{total_patterns} patterns projected")
                except Exception as e2:
                    print(f"    Projection error for {pattern.pattern_id}: {e2}")
                    results['projection_errors'] += 1
        
        self._projected_patterns = projected_patterns
        
        elapsed = time.time() - start_time
        self.stats.phase_timings['hdc_projection'] = elapsed
        
        print(f"  Patterns projected: {results['patterns_projected']}")
        print(f"  Projection errors: {results['projection_errors']}")
        print(f"  Time: {elapsed:.2f}s")
        
        return results
    
    def _project_to_hdc(self, vector: np.ndarray) -> np.ndarray:
        """
        Project vector to HDC space using Hadamard transform.
        
        Returns uint64 packed binary vector for efficient XOR operations.
        The ternary encoding {-1, 0, +1} is converted to binary and packed
        into uint64 format for L1/L2 cache residency and SIMD operations.
        """
        # Convert uint64 packed vector to full binary vector if needed
        if vector.dtype == np.uint64:
            # Unpack uint64 to binary vector
            binary_vec = np.unpackbits(vector.view(np.uint8))
            # Resize to match config dimension
            if len(binary_vec) < self.config.hdc_dim:
                binary_vec = np.pad(binary_vec, (0, self.config.hdc_dim - len(binary_vec)))
            elif len(binary_vec) > self.config.hdc_dim:
                binary_vec = binary_vec[:self.config.hdc_dim]
            vector = binary_vec.astype(np.float32)
        else:
            vector = vector.astype(np.float32)
        
        # Ensure vector has correct dimension
        if len(vector) != self.config.hdc_dim:
            # Resize to match the Hadamard basis dimension
            if len(vector) < self.config.hdc_dim:
                vector = np.pad(vector, (0, self.config.hdc_dim - len(vector)))
            else:
                vector = vector[:self.config.hdc_dim]
        
        # Quantize to ternary HDC space using ternary encoder
        # Note: ternary_encoder.encode() already applies Hadamard transform internally
        # so we don't need to call hadamard.transform() separately here
        ternary_vec = self.ternary_encoder.encode(vector)
        
        # Convert ternary {-1, 0, +1} to binary and pack into uint64
        # This enables efficient XOR operations and cache residency
        # Mapping: -1 -> 1, 0 -> 0, +1 -> 1 (we store magnitude, sign is implicit)
        # For proper bipolar representation: -1 -> 1, +1 -> 0 (XOR semantics)
        binary_vec = (ternary_vec == -1).astype(np.uint8)  # -1 maps to 1, others to 0
        
        # Pack into uint64 for efficient storage and XOR operations
        # Each uint64 holds 64 bits
        packed_size = (self.config.hdc_dim + 63) // 64
        packed = np.packbits(binary_vec)
        # Pad to full uint64 alignment if needed
        if len(packed) < packed_size * 8:
            packed = np.pad(packed, (0, packed_size * 8 - len(packed)))
        packed_uint64 = packed[:packed_size * 8].view(np.uint64)
        
        return packed_uint64
    
    def _project_batch_to_hdc(self, batch_matrix: np.ndarray) -> List[np.ndarray]:
        """
        Batch project multiple vectors to HDC space using GPU acceleration.
        
        This method leverages massive parallelism for the Hadamard transform
        and ternary encoding, enabling processing of thousands of patterns
        in seconds on modern GPUs.
        
        Args:
            batch_matrix: Matrix of shape (batch_size, hdc_dim) containing
                         all vectors to project
                         
        Returns:
            List of uint64 packed HDC vectors
        """
        xp = self.hadamard.xp  # numpy or cupy based on GPU availability
        
        # Transfer to GPU if available
        if self.hadamard.use_gpu:
            batch_matrix = xp.asarray(batch_matrix)
        
        batch_size = batch_matrix.shape[0]
        
        # Apply Fast Walsh-Hadamard Transform to entire batch at once
        # The _fwht method supports multi-dimensional arrays and processes
        # the last dimension, so we can pass (batch_size, hdc_dim) directly
        # This enables massive GPU parallelism - all rows transformed simultaneously
        transformed = self.hadamard.transform(batch_matrix)
        
        # Normalize using standard deviation for better distribution
        # This ensures values are spread across the range for better ternary encoding
        means = xp.mean(transformed, axis=1, keepdims=True)
        stds = xp.std(transformed, axis=1, keepdims=True)
        stds = xp.where(stds == 0, 1, stds)  # Avoid division by zero
        normalized = (transformed - means) / stds
        
        # Snap to ternary {-1, 0, +1} in parallel across entire batch
        # Use threshold based on standard deviation for balanced encoding
        # This should produce roughly 33% each of -1, 0, +1 for good discrimination
        threshold = self.ternary_encoder.threshold
        ternary = xp.zeros_like(normalized, dtype=xp.int8)
        ternary[normalized > threshold] = 1
        ternary[normalized < -threshold] = -1
        
        # Debug: Check sparsity of ternary vectors
        if batch_size > 0:
            first_ternary = ternary[0] if self.hadamard.use_gpu else ternary[0]
            zeros = xp.sum(first_ternary == 0)
            ones = xp.sum(first_ternary == 1)
            neg_ones = xp.sum(first_ternary == -1)
            total = len(first_ternary)
            print(f"  Ternary encoding stats (first pattern): zeros={zeros/total*100:.1f}%, +1={ones/total*100:.1f}%, -1={neg_ones/total*100:.1f}%")
        
        # Convert ternary to binary for XOR operations
        # We need to preserve the distinction between -1, 0, +1
        # Use two bits per element: bit0 for sign, bit1 for magnitude
        # -1 -> (1, 1), 0 -> (0, 0), +1 -> (0, 1)
        # For simplicity, we'll use: -1 -> 1, 0 -> 0, +1 -> 1 (magnitude only)
        # This is correct for XOR similarity on bipolar vectors
        binary_batch = ((ternary == -1) | (ternary == 1)).astype(xp.uint8)
        
        # Pack each row into uint64
        packed_size = (self.config.hdc_dim + 63) // 64
        results = []
        
        for i in range(batch_size):
            binary_vec = binary_batch[i]
            if self.hadamard.use_gpu:
                binary_vec = xp.asnumpy(binary_vec)
            
            packed = np.packbits(binary_vec)
            if len(packed) < packed_size * 8:
                packed = np.pad(packed, (0, packed_size * 8 - len(packed)))
            packed_uint64 = packed[:packed_size * 8].view(np.uint64)
            results.append(packed_uint64)
        
        return results
    
    # =========================================================================
    # Phase 4: Pattern Deduplication
    # =========================================================================
    
    def run_pattern_deduplication(self) -> Dict[str, Any]:
        """
        Run pattern deduplication phase.
        
        This phase deduplicates patterns while preserving relationships.
        
        Returns:
            Deduplication results
        """
        self.current_phase = TrainingPhase.PATTERN_DEDUPLICATION
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("Phase: Pattern Deduplication")
        print("=" * 60)
        
        results = {
            'patterns_processed': 0,
            'unique_patterns': 0,
            'duplicates_found': 0,
            'relationships_created': 0
        }
        
        unique_patterns = []
        similarity_scores = []  # Debug: track similarity scores
        first_pattern = True  # Debug: only compute detailed stats for first few patterns
        
        for pattern in self._projected_patterns:
            # Deduplicate pattern - use unified hub or legacy deduplicator
            if self.unified_hub is not None:
                # Use unified cross-model deduplication
                pattern_id, is_new, cluster_id = self.unified_hub.register_pattern(
                    vector=pattern.hdc_vector,
                    model_source="ltx",
                    layer_name=pattern.source_layer,
                    pattern_type=pattern.pattern_type if hasattr(pattern, 'pattern_type') else "video_audio",
                    seed_string=pattern.seed_string,
                    metadata=pattern.to_dict()
                )
            else:
                # Use legacy LTX-only deduplicator
                dedup_result, is_new, cluster_id = self.deduplicator.deduplicate(
                    vector=pattern.hdc_vector,
                    layer_name=pattern.source_layer,
                    seed_string=pattern.seed_string,
                    metadata=pattern.to_dict()
                )
                
                # Debug: Compute and show similarity scores for first few patterns
                if first_pattern and len(self.deduplicator._patterns) > 0:
                    print(f"\n  Debug: Similarity analysis for pattern from layer '{pattern.source_layer}':")
                    sample_count = min(5, len(self.deduplicator._patterns))
                    sample_patterns = list(self.deduplicator._patterns.items())[:sample_count]
                    for pid, existing_pattern in sample_patterns:
                        sim = self.deduplicator._compute_hadamard_similarity(pattern.hdc_vector, existing_pattern.vector)
                        print(f"    vs {existing_pattern.layer_name}: similarity = {sim:.4f}")
                    first_pattern = False
            
            results['patterns_processed'] += 1
            
            if is_new:
                unique_patterns.append(pattern)
                results['unique_patterns'] += 1
            else:
                results['duplicates_found'] += 1
            
            # Track relationships
            if cluster_id:
                results['relationships_created'] += 1
        
        self._unique_patterns = unique_patterns
        self.stats.unique_recipes = results['unique_patterns']
        self.stats.deduplicated_recipes = results['duplicates_found']
        
        elapsed = time.time() - start_time
        self.stats.phase_timings['pattern_deduplication'] = elapsed
        
        print(f"  Patterns processed: {results['patterns_processed']}")
        print(f"  Unique patterns: {results['unique_patterns']}")
        print(f"  Duplicates found: {results['duplicates_found']}")
        print(f"  Relationships created: {results['relationships_created']}")
        print(f"  Time: {elapsed:.2f}s")
        
        return results
    
    # =========================================================================
    # Phase 5: Resonator Training
    # =========================================================================
    
    def run_resonator_training(self) -> Dict[str, Any]:
        """
        Run resonator training phase.
        
        This phase trains the resonator network for pattern factorization.
        
        Note: For LTX transfer learning, patterns are individual layer representations,
        not bundled vectors. The resonator is used here for:
        1. Quick pattern matching to find similar patterns in codebooks
        2. Building role assignments for cross-modal patterns
        
        Returns:
            Resonator training results
        """
        self.current_phase = TrainingPhase.RESONATOR_TRAINING
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("Phase: Resonator Training")
        print("=" * 60)
        
        if not self.config.resonator.enable_resonator:
            print("  Resonator training disabled")
            self.stats.phase_timings['resonator_training'] = time.time() - start_time
            return {'enabled': False}
        
        results = {
            'enabled': True,
            'patterns_factorized': 0,
            'converged': 0,
            'failed': 0,
            'avg_iterations': 0.0,
            'skipped': False
        }
        
        # Check if we should skip resonator for single patterns
        # (patterns that are not bundled vectors)
        if self.config.resonator.skip_for_single_patterns:
            print("  Skipping resonator factorization (single pattern mode)")
            print("  Note: LTX patterns are individual layer representations,")
            print("        not bundled vectors. Resonator factorization is not needed.")
            results['skipped'] = True
            results['patterns_factorized'] = len(self._unique_patterns)
            results['converged'] = len(self._unique_patterns)  # All patterns are valid
            self.stats.resonator_converged = len(self._unique_patterns)
            elapsed = time.time() - start_time
            self.stats.phase_timings['resonator_training'] = elapsed
            print(f"  Patterns processed: {results['patterns_factorized']}")
            print(f"  Time: {elapsed:.2f}s")
            return results
        
        total_iterations = 0
        
        # Build codebooks from unique patterns
        codebooks = self._build_codebooks()
        
        # Print codebook sizes for debugging
        print(f"  Codebook sizes: {', '.join(f'{r}={len(s)}' for r, s in codebooks.items())}")
        
        total_patterns = len(self._unique_patterns)
        print(f"  Processing {total_patterns} patterns...")
        
        # Use reduced iterations for single patterns
        max_iter = self.config.resonator.quick_match_iterations
        
        for idx, pattern in enumerate(self._unique_patterns):
            # Progress output every 10 patterns
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total_patterns - idx - 1) / rate if rate > 0 else 0
                print(f"  Progress: {idx + 1}/{total_patterns} patterns "
                      f"({100*(idx+1)/total_patterns:.1f}%) - "
                      f"elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
            
            # Factorize pattern using resonator with reduced iterations
            factor_result = self.resonator.factorize(
                bundled_vector=pattern.hdc_vector,
                codebooks=codebooks,
                inhibitory_mask=self._safety_inhibitory_mask,
                max_iterations_override=max_iter
            )
            
            results['patterns_factorized'] += 1
            
            if factor_result.converged:
                results['converged'] += 1
                self.stats.resonator_converged += 1
            else:
                results['failed'] += 1
                self.stats.resonator_failed += 1
            
            total_iterations += factor_result.iterations
        
        if results['patterns_factorized'] > 0:
            results['avg_iterations'] = total_iterations / results['patterns_factorized']
            self.stats.avg_iterations = results['avg_iterations']
        
        elapsed = time.time() - start_time
        self.stats.phase_timings['resonator_training'] = elapsed
        
        print(f"  Patterns factorized: {results['patterns_factorized']}")
        print(f"  Converged: {results['converged']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Avg iterations: {results['avg_iterations']:.2f}")
        print(f"  Time: {elapsed:.2f}s")
        
        return results
    
    def _build_codebooks(self) -> Dict[str, List[str]]:
        """Build codebooks from unique patterns for resonator."""
        codebooks = {role: [] for role in self.config.resonator.roles}
        
        # Add pattern seeds to appropriate roles
        for pattern in self._unique_patterns:
            # Determine which role this pattern belongs to
            if 'video' in pattern.source_layer:
                role = 'video_content'
            elif 'audio' in pattern.source_layer:
                role = 'audio_content'
            elif 'cross' in pattern.source_layer:
                role = 'cross_modal_binding'
            else:
                role = 'style'
            
            if role in codebooks:
                codebooks[role].append(pattern.seed_string)
        
        # Ensure minimum codebook size
        for role in codebooks:
            if len(codebooks[role]) < 10:
                # Add placeholder seeds
                for i in range(10 - len(codebooks[role])):
                    codebooks[role].append(f"placeholder:{role}:{i}")
        
        return codebooks
    
    # =========================================================================
    # Phase 6: Recipe Generation
    # =========================================================================
    
    def run_recipe_generation(self) -> Dict[str, Any]:
        """
        Run recipe generation phase.
        
        This phase creates and stores recipes for all unique patterns.
        
        Returns:
            Recipe generation results
        """
        self.current_phase = TrainingPhase.RECIPE_GENERATION
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("Phase: Recipe Generation")
        print("=" * 60)
        
        results = {
            'recipes_created': 0,
            'chains_created': 0,
            'storage_bytes': 0
        }
        
        recipe_ids = []
        chain_ids = []
        
        for pattern in self._unique_patterns:
            # Create recipe
            recipe = self._create_recipe_from_pattern(pattern)
            
            # Save recipe
            recipe_id = self.recipe_storage.save_recipe(recipe)
            recipe_ids.append(recipe_id)
            results['recipes_created'] += 1
            
            # Create chain seed
            chain = self._create_chain_from_pattern(pattern)
            if chain:
                chain_id = self.chain_storage.save_chain(chain)
                chain_ids.append(chain_id)
                results['chains_created'] += 1
        
        self.stats.total_recipes_created = results['recipes_created']
        
        # Calculate storage size
        storage_path = Path(self.config.storage.storage_path)
        total_bytes = sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())
        results['storage_bytes'] = total_bytes
        self.stats.total_storage_bytes = total_bytes
        
        # Calculate compression ratio
        # Note: Recipes store seeds for procedural regeneration, not actual vectors.
        # The "compression" is achieved through Zero-Weight Procedural Generation.
        # We compare: full vector storage (uint64 packed) vs recipe storage (seed + metadata)
        # uint64 packed: hdc_dim / 8 bytes per vector
        # Recipe: ~100-200 bytes (seed string + metadata)
        if self.stats.total_patterns_extracted > 0 and total_bytes > 0:
            # Compare against uint64 packed storage (what we'd store without recipes)
            packed_size = self.stats.total_patterns_extracted * (self.config.hdc_dim // 8)
            self.stats.compression_ratio = packed_size / total_bytes
        else:
            self.stats.compression_ratio = 0.0
        
        self._recipe_ids = recipe_ids
        self._chain_ids = chain_ids
        
        elapsed = time.time() - start_time
        self.stats.phase_timings['recipe_generation'] = elapsed
        
        print(f"  Recipes created: {results['recipes_created']}")
        print(f"  Chains created: {results['chains_created']}")
        print(f"  Storage bytes: {results['storage_bytes']}")
        print(f"  Compression ratio: {self.stats.compression_ratio:.2f}x")
        print(f"  Time: {elapsed:.2f}s")
        
        return results
    
    def _create_recipe_from_pattern(self, pattern: AudioVideoPattern) -> IdentityRecipe:
        """Create a recipe from a pattern."""
        # Generate deterministic hadamard index
        hadamard_index = seed_string_to_int(pattern.seed_string) % (2**20)
        
        # Generate base seed
        base_seed = seed_string_to_int(pattern.seed_string)
        
        # Create recipe
        recipe = IdentityRecipe(
            recipe_id=f"H{hadamard_index}.S{base_seed}",
            hadamard_index=hadamard_index,
            base_seed=base_seed,
            operation="identity",
            name=pattern.pattern_id,
            recipe_type=f"ltx_{pattern.generation_mode.value}",
            metadata={
                'layer': pattern.source_layer,
                'timestep': pattern.timestep,
                'modality': 'audio_video',
                'generation_mode': pattern.generation_mode.value
            }
        )
        
        return recipe
    
    def _create_chain_from_pattern(self, pattern: AudioVideoPattern) -> Optional[LTXChainSeed]:
        """Create a chain seed from a pattern."""
        # Create seed step
        step = LTXSeedStep(
            step_id=f"step_{pattern.pattern_id}",
            seed=seed_string_to_int(pattern.seed_string),
            hadamard_index=seed_string_to_int(pattern.seed_string) % (2**20),
            operation=LTXChainOperation.BIND,
            layer_name=pattern.source_layer,
            timestep=pattern.timestep,
            modality='joint'
        )
        
        # Create chain
        chain = LTXChainSeed(
            chain_id=f"chain_{pattern.pattern_id}",
            model_name="LTX-2.3",
            generation_mode=pattern.generation_mode.value,
            steps=[step],
            total_timesteps=pattern.timestep + 1
        )
        
        return chain
    
    # =========================================================================
    # Phase 7: Model Merging
    # =========================================================================
    
    def run_model_merging(self) -> Dict[str, Any]:
        """
        Run model merging phase.
        
        This phase creates the final merged HDC model.
        
        Returns:
            Model merging results
        """
        self.current_phase = TrainingPhase.MODEL_MERGING
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("Phase: Model Merging")
        print("=" * 60)
        
        results = {
            'model_created': False,
            'model_id': '',
            'modalities_supported': []
        }
        
        # Create merged model
        model_id = hashlib.sha256(
            f"ltx_merged_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Use the final storage path (config.storage.storage_path) for the merged model
        final_storage_path = self._final_storage_path
        
        self.merged_model = MergedHDCModel(
            model_id=model_id,
            model_name="LTX-HDC-Merged",
            version="1.0.0",
            hdc_dim=self.config.hdc_dim,
            recipe_storage_path=str(final_storage_path),
            chain_storage_path=str(final_storage_path / "chains"),
            safety_registry_path=str(final_storage_path / "safety_registry.json") if self.safety_registry else None,
            source_models=["LTX-2.3"],
            training_config=self.config.to_dict(),
            training_stats=self.stats.to_dict(),
            supported_modalities=["video", "audio", "audio_video"],
            modality_recipes={
                "video": [r for r in self._recipe_ids if 'video' in r],
                "audio": [r for r in self._recipe_ids if 'audio' in r],
                "audio_video": self._recipe_ids
            }
        )
        
        results['model_created'] = True
        results['model_id'] = model_id
        results['modalities_supported'] = self.merged_model.supported_modalities
        
        # Save safety registry to final storage path if available
        if self.safety_registry:
            safety_path = final_storage_path / "safety_registry.json"
            self.safety_registry.save(str(safety_path))
        
        elapsed = time.time() - start_time
        self.stats.phase_timings['model_merging'] = elapsed
        
        print(f"  Model created: {results['model_id']}")
        print(f"  Modalities supported: {results['modalities_supported']}")
        print(f"  Time: {elapsed:.2f}s")
        
        return results
    
    # =========================================================================
    # Phase 8: Validation
    # =========================================================================
    
    def run_validation(self) -> Dict[str, Any]:
        """
        Run validation phase.
        
        This phase validates the merged model.
        
        Returns:
            Validation results
        """
        self.current_phase = TrainingPhase.VALIDATION
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("Phase: Validation")
        print("=" * 60)
        
        results = {
            'recipes_valid': 0,
            'recipes_invalid': 0,
            'chains_valid': 0,
            'chains_invalid': 0,
            'overall_valid': False
        }
        
        # Validate recipes
        for recipe_id in self._recipe_ids:
            recipe = self.recipe_storage.load_recipe(recipe_id)
            if recipe and recipe.verify_integrity():
                results['recipes_valid'] += 1
            else:
                results['recipes_invalid'] += 1
        
        # Validate chains
        for chain_id in self._chain_ids:
            chain = self.chain_storage.load_chain(chain_id)
            if chain:
                results['chains_valid'] += 1
            else:
                results['chains_invalid'] += 1
        
        # Overall validation
        results['overall_valid'] = (
            results['recipes_invalid'] == 0 and
            results['chains_invalid'] == 0 and
            self.merged_model is not None
        )
        
        elapsed = time.time() - start_time
        self.stats.phase_timings['validation'] = elapsed
        
        print(f"  Recipes valid: {results['recipes_valid']}")
        print(f"  Recipes invalid: {results['recipes_invalid']}")
        print(f"  Chains valid: {results['chains_valid']}")
        print(f"  Chains invalid: {results['chains_invalid']}")
        print(f"  Overall valid: {results['overall_valid']}")
        print(f"  Time: {elapsed:.2f}s")
        
        return results
    
    # =========================================================================
    # Full Training Pipeline
    # =========================================================================
    
    def run_full_training(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns:
            Complete training results
        """
        print("\n" + "=" * 70)
        print("LTX Training Pipeline - Full Training")
        print("=" * 70)
        print(f"HDC Dimension: {self.config.hdc_dim}")
        print(f"Output Path: {self.config.output_path}")
        print(f"Safety Training: {self.config.safety.enable_safety_training}")
        print(f"Resonator: {self.config.resonator.enable_resonator}")
        print("=" * 70)
        
        total_start = time.time()
        
        # Run all phases
        results = {
            'safety_training': self.run_safety_training(),
            'latent_extraction': self.run_latent_extraction(),
            'hdc_projection': self.run_hdc_projection(),
            'pattern_deduplication': self.run_pattern_deduplication(),
            'resonator_training': self.run_resonator_training(),
            'recipe_generation': self.run_recipe_generation(),
            'model_merging': self.run_model_merging(),
            'validation': self.run_validation()
        }
        
        total_elapsed = time.time() - total_start
        
        self.current_phase = TrainingPhase.COMPLETED
        
        # Print summary
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Total time: {total_elapsed:.2f}s")
        print(f"Patterns extracted: {self.stats.total_patterns_extracted}")
        print(f"Recipes created: {self.stats.total_recipes_created}")
        print(f"Compression ratio: {self.stats.compression_ratio:.2f}x")
        print(f"Model ID: {self.merged_model.model_id if self.merged_model else 'N/A'}")
        print(f"Recipes stored in: {self._final_storage_path}")
        print("=" * 70)
        
        results['total_time'] = total_elapsed
        results['statistics'] = self.stats.to_dict()
        results['storage_path'] = str(self._final_storage_path)
        
        # Automatically export to ltx_recipes for the final merged model
        try:
            export_path = self.export_to_ltx_recipes(self.config.storage.storage_path)
            results['export_path'] = export_path
            print(f"\nFinal merged model exported to: {export_path}")
        except Exception as e:
            print(f"\nWarning: Could not export to ltx_recipes: {e}")
            results['export_path'] = None
        
        return results
    
    # =========================================================================
    # Model Saving and Loading
    # =========================================================================
    
    def save_merged_model(self, path: str) -> str:
        """
        Save the merged model to disk.
        
        This method saves the model manifest and ensures all recipes and seeds
        are properly stored in the final storage location (config.storage.storage_path).
        
        Args:
            path: Path to save the model manifest (recipes are already in storage_path)
            
        Returns:
            Path to the saved model manifest
        """
        if self.merged_model is None:
            raise ValueError("No merged model to save. Run training first.")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Update modified timestamp
        self.merged_model.modified_at = datetime.now().isoformat()
        
        # Save model manifest
        manifest_path = save_path / "model_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.merged_model.to_dict(), f, indent=2)
        
        # Recipes and chains are already stored in config.storage.storage_path
        # (self._final_storage_path) during the training pipeline execution.
        # No need to copy - they are already in the correct location.
        
        # Verify recipe storage exists
        recipe_storage_path = Path(self.merged_model.recipe_storage_path)
        if not recipe_storage_path.exists():
            print(f"Warning: Recipe storage path does not exist: {recipe_storage_path}")
        
        # Verify chain storage exists
        chain_storage_path = Path(self.merged_model.chain_storage_path)
        if not chain_storage_path.exists():
            print(f"Warning: Chain storage path does not exist: {chain_storage_path}")
        
        print(f"Model manifest saved to: {manifest_path}")
        print(f"Recipes stored in: {self.merged_model.recipe_storage_path}")
        print(f"Chains stored in: {self.merged_model.chain_storage_path}")
        return str(manifest_path)
    
    def export_to_ltx_recipes(self, ltx_recipes_path: str = "./ltx_recipes") -> str:
        """
        Export the final merged model recipes and seeds to the ltx_recipes directory.
        
        This method ensures the final merged model's recipes and seeds are saved
        to the standard ltx_recipes location for use by other components.
        
        Args:
            ltx_recipes_path: Path to the ltx_recipes directory (default: ./ltx_recipes)
            
        Returns:
            Path to the exported manifest
        """
        if self.merged_model is None:
            raise ValueError("No merged model to export. Run training first.")
        
        import shutil
        
        export_path = Path(ltx_recipes_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export model manifest
        manifest_path = export_path / "model_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.merged_model.to_dict(), f, indent=2)
        
        # Export recipes - copy from the recipe storage
        source_recipes = Path(self.merged_model.recipe_storage_path)
        if source_recipes.exists():
            # Copy all .xorr recipe files
            dest_recipes = export_path / "recipes"
            dest_recipes.mkdir(exist_ok=True)
            
            for recipe_file in source_recipes.glob("*.xorr"):
                dest_file = dest_recipes / recipe_file.name
                shutil.copy2(recipe_file, dest_file)
            
            # Copy index.json if it exists
            index_file = source_recipes / "index.json"
            if index_file.exists():
                shutil.copy2(index_file, dest_recipes / "index.json")
            
            # Copy manifest.json if it exists
            manifest_file = source_recipes / "manifest.json"
            if manifest_file.exists():
                shutil.copy2(manifest_file, export_path / "manifest.json")
        
        # Export chains
        source_chains = Path(self.merged_model.chain_storage_path)
        if source_chains.exists():
            dest_chains = export_path / "chains"
            if dest_chains.exists():
                shutil.rmtree(dest_chains)
            shutil.copytree(source_chains, dest_chains)
        
        # Export safety registry if available
        if self.merged_model.safety_registry_path:
            source_safety = Path(self.merged_model.safety_registry_path)
            if source_safety.exists():
                shutil.copy2(source_safety, export_path / "safety_registry.json")
        
        # Create a summary file with all recipe IDs and seeds
        summary = {
            "model_id": self.merged_model.model_id,
            "model_name": self.merged_model.model_name,
            "version": self.merged_model.version,
            "hdc_dim": self.merged_model.hdc_dim,
            "total_recipes": len(self._recipe_ids) if hasattr(self, '_recipe_ids') else 0,
            "total_chains": len(self._chain_ids) if hasattr(self, '_chain_ids') else 0,
            "recipe_ids": self._recipe_ids if hasattr(self, '_recipe_ids') else [],
            "chain_ids": self._chain_ids if hasattr(self, '_chain_ids') else [],
            "supported_modalities": self.merged_model.supported_modalities,
            "modality_recipes": self.merged_model.modality_recipes,
            "exported_at": datetime.now().isoformat()
        }
        
        summary_path = export_path / "merged_model_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExported merged model to: {export_path}")
        print(f"  Model manifest: {manifest_path}")
        print(f"  Recipes: {export_path / 'recipes'}")
        print(f"  Chains: {export_path / 'chains'}")
        print(f"  Summary: {summary_path}")
        print(f"  Total recipes exported: {summary['total_recipes']}")
        print(f"  Total chains exported: {summary['total_chains']}")
        
        return str(manifest_path)
    
    def evaluate_hdc_model(self, save_path: Optional[str] = None) -> 'HDCModelEvaluationResult':
        """
        Evaluate the HDC model components directly.
        
        This provides comprehensive evaluation of the trained HDC model,
        measuring:
        - XOR algebra integrity (bind/unbind fidelity)
        - Recipe storage quality
        - Chain seed quality (thought steps efficiency)
        - Pattern clustering quality
        - Relationship graph quality
        - Resonator factorization quality
        - Information compression quality
        - Recall and retrieval quality
        
        Unlike BPB evaluation which tests fresh vectors, this directly
        evaluates the trained model's stored patterns and components.
        
        Args:
            save_path: Optional path to save evaluation results
            
        Returns:
            HDCModelEvaluationResult with all metrics
        """
        # Lazy import to avoid circular dependency
        from ..hdc_model_evaluation import evaluate_hdc_model as _evaluate_hdc_model
        
        # Get components from pipeline
        hdc = getattr(self, 'hdc', None)
        recipe_storage = getattr(self, 'recipe_storage', None)
        chain_storage = getattr(self, 'chain_storage', None)
        deduplicator = getattr(self, 'deduplicator', None)
        relationship_graph = getattr(self, 'relationship_graph', None)
        resonator = getattr(self, 'resonator', None)
        codebooks = getattr(self, 'codebooks', None)
        
        if not hdc:
            raise ValueError("HDC engine not initialized in pipeline")
        
        return _evaluate_hdc_model(
            hdc=hdc,
            recipe_storage=recipe_storage,
            chain_storage=chain_storage,
            deduplicator=deduplicator,
            relationship_graph=relationship_graph,
            resonator=resonator,
            codebooks=codebooks,
            save_path=save_path,
            verbose=True
        )
    
    def save_unified_checkpoint(self, checkpoint_path: Optional[str] = None) -> str:
        """
        Save the unified deduplication checkpoint that can be shared across all model transfers.
        
        This checkpoint contains all patterns from the unified hub, enabling knowledge
        sharing between different model transfer learning pipelines (LTX, MOSS-TTS, Qwen, etc.)
        
        Args:
            checkpoint_path: Path to save checkpoint (uses config default if None)
            
        Returns:
            Path to the saved checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = self.config.storage.unified_checkpoint_path
        
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_source': 'ltx',
            'checkpoint_type': 'unified_deduplication',
            'created_at': datetime.now().isoformat(),
            'config': {
                'hdc_dim': self.config.hdc_dim,
                'similarity_threshold': self.config.storage.deduplication_threshold,
                'use_unified_deduplication': self.config.storage.use_unified_deduplication
            },
            'statistics': self.stats.to_dict() if hasattr(self.stats, 'to_dict') else {},
            'patterns': {},
            'cross_model_relationships': []
        }
        
        # Save unified hub patterns if available
        if self.unified_hub is not None:
            hub_stats = self.unified_hub.get_statistics()
            checkpoint['unified_hub_stats'] = hub_stats
            
            # Save pattern metadata (without vectors - they can be regenerated from seeds)
            for pattern_id, pattern in self.unified_hub._patterns.items():
                checkpoint['patterns'][pattern_id] = {
                    'pattern_id': pattern.pattern_id,
                    'content_hash': pattern.content_hash,
                    'seed_string': pattern.seed_string,
                    'hadamard_index': pattern.hadamard_index,
                    'model_sources': pattern.model_sources,
                    'layer_names': pattern.layer_names,
                    'pattern_types': pattern.pattern_types,
                    'cluster_id': pattern.cluster_id,
                    'is_centroid': pattern.is_centroid,
                    'metadata': pattern.metadata
                }
            
            # Save cross-model relationships
            for rel_type in self.unified_hub.relationship_graph._by_relationship_type:
                for src_id, tgt_id in self.unified_hub.relationship_graph._by_relationship_type[rel_type]:
                    checkpoint['cross_model_relationships'].append({
                        'source_pattern_id': src_id,
                        'target_pattern_id': tgt_id,
                        'relationship_type': rel_type.value
                    })
            
            # Save the unified hub patterns to its own storage
            self.unified_hub.save()
        
        # Save checkpoint using torch for compatibility with other pipelines
        try:
            import torch
            torch.save(checkpoint, checkpoint_path)
        except ImportError:
            # Fall back to JSON if torch not available
            json_path = str(checkpoint_file).replace('.pt', '.json')
            with open(json_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            checkpoint_path = json_path
        
        print(f"\n✓ Unified checkpoint saved to: {checkpoint_path}")
        print(f"  Patterns: {len(checkpoint['patterns'])}")
        print(f"  Cross-model relationships: {len(checkpoint['cross_model_relationships'])}")
        
        return str(checkpoint_path)
    
    def load_unified_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load a unified deduplication checkpoint from a previous transfer.
        
        This allows the LTX pipeline to continue from patterns discovered by
        other model transfers (MOSS-TTS, Qwen, etc.), enabling knowledge sharing.
        
        Args:
            checkpoint_path: Path to load checkpoint from (uses config default if None)
            
        Returns:
            True if checkpoint loaded successfully
        """
        if checkpoint_path is None:
            checkpoint_path = self.config.storage.unified_checkpoint_path
        
        checkpoint_file = Path(checkpoint_path)
        
        if not checkpoint_file.exists():
            # Try JSON fallback
            json_path = str(checkpoint_file).replace('.pt', '.json')
            if Path(json_path).exists():
                checkpoint_file = Path(json_path)
            else:
                print(f"No unified checkpoint found at {checkpoint_path}")
                return False
        
        try:
            # Try torch load first
            try:
                import torch
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
            except ImportError:
                # Fall back to JSON
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
            
            print(f"\n✓ Loading unified checkpoint from: {checkpoint_file}")
            print(f"  Source model: {checkpoint.get('model_source', 'unknown')}")
            print(f"  Created: {checkpoint.get('created_at', 'unknown')}")
            
            # Initialize unified hub if not already done
            if self.unified_hub is None and self.config.storage.use_unified_deduplication:
                self.unified_hub = create_unified_deduplicator(
                    storage_path=self.config.storage.unified_storage_path,
                    hdc_dim=self.config.hdc_dim,
                    similarity_threshold=self.config.storage.deduplication_threshold,
                    use_gpu=self.config.storage.enable_gpu_similarity
                )
            
            if self.unified_hub is None:
                print("  Warning: Unified hub not available, cannot load checkpoint")
                return False
            
            # Load patterns into unified hub
            loaded_patterns = 0
            for pattern_id, pattern_data in checkpoint.get('patterns', {}).items():
                if pattern_id not in self.unified_hub._patterns:
                    # Recreate pattern from saved data
                    pattern = UnifiedPattern(
                        pattern_id=pattern_data['pattern_id'],
                        content_hash=pattern_data['content_hash'],
                        seed_string=pattern_data['seed_string'],
                        hadamard_index=pattern_data['hadamard_index'],
                        model_sources=pattern_data.get('model_sources', []),
                        layer_names=pattern_data.get('layer_names', {}),
                        pattern_types=pattern_data.get('pattern_types', {}),
                        cluster_id=pattern_data.get('cluster_id'),
                        is_centroid=pattern_data.get('is_centroid', False),
                        metadata=pattern_data.get('metadata', {})
                    )
                    
                    # Regenerate vector from seed if needed
                    if pattern.seed_string:
                        pattern.vector = seed_to_hypervector_blake3(
                            pattern.seed_string,
                            self.config.hdc_dim // 64
                        )
                    
                    self.unified_hub._patterns[pattern_id] = pattern
                    self.unified_hub.relationship_graph.add_pattern(pattern)
                    loaded_patterns += 1
            
            # Load cross-model relationships
            loaded_relationships = 0
            for rel_data in checkpoint.get('cross_model_relationships', []):
                src_id = rel_data['source_pattern_id']
                tgt_id = rel_data['target_pattern_id']
                rel_type = CrossModelRelationshipType(rel_data['relationship_type'])
                
                if src_id in self.unified_hub._patterns and tgt_id in self.unified_hub._patterns:
                    self.unified_hub.add_cross_model_relationship(
                        src_id, tgt_id, rel_type
                    )
                    loaded_relationships += 1
            
            # Update pattern counter
            if self.unified_hub._patterns:
                max_id = max(
                    int(p.split('_')[1]) if '_' in p and p.split('_')[1].isdigit() else 0
                    for p in self.unified_hub._patterns.keys()
                )
                self.unified_hub._pattern_counter = max_id + 1
            
            print(f"  Patterns loaded: {loaded_patterns}")
            print(f"  Relationships loaded: {loaded_relationships}")
            
            return True
            
        except Exception as e:
            print(f"Error loading unified checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @classmethod
    def load_merged_model(cls, path: str) -> 'LTXTrainingPipeline':
        """
        Load a merged model from disk.
        
        Args:
            path: Path to the model manifest
            
        Returns:
            LTXTrainingPipeline with loaded model
        """
        manifest_path = Path(path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "model_manifest.json"
        
        with open(manifest_path, 'r') as f:
            model_data = json.load(f)
        
        merged_model = MergedHDCModel.from_dict(model_data)
        
        # Create config from saved data
        config = TrainingConfig.from_dict(merged_model.training_config)
        
        # Create pipeline
        pipeline = cls(config)
        pipeline.merged_model = merged_model
        
        # Load statistics
        pipeline.stats = TrainingStatistics(**merged_model.training_stats)
        
        print(f"Model loaded: {merged_model.model_id}")
        return pipeline
    
    # =========================================================================
    # Incremental Training (Additional Modalities)
    # =========================================================================
    
    def add_modality_training(self,
                              modality_name: str,
                              training_data: Any,
                              modality_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add training from an additional modality.
        
        This enables incremental training on top of the existing merged model.
        
        Args:
            modality_name: Name of the new modality (e.g., "audio_emotion")
            training_data: Training data for the modality
            modality_config: Optional configuration for the modality
            
        Returns:
            Training results for the new modality
        """
        if not self.config.enable_incremental_training:
            raise ValueError("Incremental training is disabled in config")
        
        if self.merged_model is None:
            raise ValueError("No merged model exists. Run training first.")
        
        print("\n" + "=" * 60)
        print(f"Adding Modality: {modality_name}")
        print("=" * 60)
        
        start_time = time.time()
        
        results = {
            'modality': modality_name,
            'recipes_added': 0,
            'success': False
        }
        
        # Extract patterns from new modality
        modality_patterns = self._extract_modality_patterns(
            modality_name=modality_name,
            training_data=training_data,
            config=modality_config or {}
        )
        
        # Apply safety filtering
        safe_patterns = []
        for pattern in modality_patterns:
            if self.safety_integration is not None:
                is_safe, _ = self._apply_safety_filter(pattern)
                if not is_safe:
                    continue
            safe_patterns.append(pattern)
        
        # Project to HDC space
        projected_patterns = []
        for pattern in safe_patterns:
            hdc_vector = self._project_to_hdc(pattern.hdc_vector)
            pattern.hdc_vector = hdc_vector
            projected_patterns.append(pattern)
        
        # Deduplicate
        unique_patterns = []
        for pattern in projected_patterns:
            if self.unified_hub is not None:
                # Use unified cross-model deduplication
                _, is_new, _ = self.unified_hub.register_pattern(
                    vector=pattern.hdc_vector,
                    model_source="ltx",
                    layer_name=modality_name,
                    pattern_type=f"{modality_name}_pattern",
                    seed_string=pattern.seed_string,
                    metadata={'modality': modality_name}
                )
            else:
                # Use legacy LTX-only deduplicator
                _, is_new, _ = self.deduplicator.deduplicate(
                    vector=pattern.hdc_vector,
                    layer_name=modality_name,
                    seed_string=pattern.seed_string,
                    metadata={'modality': modality_name}
                )
            if is_new:
                unique_patterns.append(pattern)
        
        # Create recipes
        new_recipe_ids = []
        for pattern in unique_patterns:
            recipe = self._create_recipe_from_pattern(pattern)
            recipe.recipe_type = f"{modality_name}_pattern"
            recipe_id = self.recipe_storage.save_recipe(recipe)
            new_recipe_ids.append(recipe_id)
            results['recipes_added'] += 1
        
        # Update merged model
        self.merged_model.supported_modalities.append(modality_name)
        self.merged_model.modality_recipes[modality_name] = new_recipe_ids
        self.merged_model.modified_at = datetime.now().isoformat()
        
        results['success'] = True
        results['time'] = time.time() - start_time
        
        return results
    
    # =========================================================================
    # Accuracy Improvement Methods (95% → 99% accuracy target)
    # =========================================================================
    
    def search_with_accuracy_improvement(
        self,
        composite_vector: np.ndarray,
        codebooks: Dict[str, List[np.ndarray]],
        use_refinement: bool = True,
        use_parallel: bool = False
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Perform search with all accuracy improvement strategies applied.
        
        This method uses the AccuracyEngine which combines:
        1. Hierarchical Search Space Expansion
        2. Enhanced Resonator Network
        3. Semantic Codebook
        4. Iterative Refinement
        5. Parallel Multi-Path Search
        6. Enhanced Collision Shield
        
        Args:
            composite_vector: The composite HDC vector to factorize
            codebooks: Codebooks for each role
            use_refinement: Whether to use iterative refinement
            use_parallel: Whether to use parallel search
            
        Returns:
            Tuple of (result estimates, confidence)
        """
        if self.accuracy_engine is None:
            # Fallback to simple search if accuracy engine not initialized
            return {}, 0.0
        
        result, confidence = self.accuracy_engine.search(
            composite_vector=composite_vector,
            codebooks=codebooks,
            use_refinement=use_refinement,
            use_parallel=use_parallel
        )
        
        return result, confidence
    
    def build_codebook_from_patterns(
        self,
        patterns: Optional[List[Any]] = None,
        semantic_clusters: bool = True
    ) -> Dict[str, List[np.ndarray]]:
        """
        Build a codebook from patterns for use with accuracy improvement search.
        
        Args:
            patterns: List of patterns to use (uses stored patterns if None)
            semantic_clusters: Whether to organize by semantic clusters
            
        Returns:
            Dictionary mapping role -> list of HDC vectors
        """
        patterns = patterns or []
        codebooks: Dict[str, List[np.ndarray]] = {}
        
        for pattern in patterns:
            # Determine role based on pattern type
            role = getattr(pattern, 'layer_type', 'unknown')
            if hasattr(pattern, 'layer_type') and hasattr(pattern.layer_type, 'value'):
                role = pattern.layer_type.value
            elif hasattr(pattern, 'generation_mode'):
                role = pattern.generation_mode
            
            if role not in codebooks:
                codebooks[role] = []
            
            # Get the HDC vector from the pattern
            if hasattr(pattern, 'hdc_vector') and pattern.hdc_vector is not None:
                codebooks[role].append(pattern.hdc_vector)
            elif hasattr(pattern, 'seed_string') and pattern.seed_string is not None:
                # Generate vector from seed
                vector = seed_to_hypervector_blake3(
                    pattern.seed_string,
                    self.config.hdc_dim // 64
                )
                codebooks[role].append(vector)
        
        # Expand codebook if enabled
        if self.config.codebook_expansion_factor > 1 and self.accuracy_engine:
            for role in codebooks:
                original = codebooks[role]
                expanded = self.accuracy_engine.codebook.expand_codebook(
                    role=role,
                    base_patterns=original
                )
                codebooks[role] = expanded
        
        return codebooks
    
    def get_accuracy_stats(self) -> Dict[str, Any]:
        """
        Get accuracy improvement statistics.
        
        Returns:
            Dictionary of accuracy statistics
        """
        if self.accuracy_engine is None:
            return {'accuracy_improvement_enabled': False}
        
        return {
            'accuracy_improvement_enabled': True,
            **self.accuracy_engine.get_stats()
        }
        
        print(f"  Recipes added: {results['recipes_added']}")
        print(f"  Time: {results['time']:.2f}s")
        
        return results
    
    def _extract_modality_patterns(self,
                                   modality_name: str,
                                   training_data: Any,
                                   config: Dict[str, Any]) -> List[AudioVideoPattern]:
        """Extract patterns from modality-specific training data."""
        patterns = []
        
        # This is a generic implementation
        # Specific modalities should override this method
        
        if isinstance(training_data, list):
            for i, item in enumerate(training_data):
                seed_string = f"{modality_name}:pattern:{i}"
                pattern_id = hashlib.sha256(seed_string.encode()).hexdigest()[:16]
                
                pattern = AudioVideoPattern(
                    pattern_id=pattern_id,
                    pattern_type=modality_name,
                    seed_string=seed_string
                )
                
                pattern.hdc_vector = seed_to_hypervector_blake3(
                    seed_string,
                    self.config.hdc_dim // 64
                )
                
                patterns.append(pattern)
        
        return patterns


# =============================================================================
# Convenience Functions
# =============================================================================

def create_training_pipeline(
    model_path: str,
    output_path: str,
    hdc_dim: int = DEFAULT_HDC_DIM,
    enable_safety: bool = True,
    enable_resonator: bool = True
) -> LTXTrainingPipeline:
    """
    Create a training pipeline with default configuration.
    
    Args:
        model_path: Path to LTX model
        output_path: Path for output
        hdc_dim: HDC dimension
        enable_safety: Enable safety training
        enable_resonator: Enable resonator training
        
    Returns:
        Configured LTXTrainingPipeline
    """
    config = TrainingConfig(
        model_path=model_path,
        output_path=output_path,
        hdc_dim=hdc_dim,
        safety=SafetyTrainingConfig(enable_safety_training=enable_safety),
        resonator=ResonatorTrainingConfig(enable_resonator=enable_resonator)
    )
    
    return LTXTrainingPipeline(config)


def run_quick_training(
    model_path: str,
    output_path: str
) -> Tuple[LTXTrainingPipeline, Dict[str, Any]]:
    """
    Run a quick training with default settings.
    
    Args:
        model_path: Path to LTX model
        output_path: Path for output
        
    Returns:
        Tuple of (pipeline, results)
    """
    pipeline = create_training_pipeline(
        model_path=model_path,
        output_path=output_path
    )
    
    results = pipeline.run_full_training()
    
    return pipeline, results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="LTX Training Pipeline - Full HDC Transfer Learning"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=get_default_ltx_model_path(),
        help="Path to LTX model (default: first available of UVX cache or /workspace/LTX-2.3-fp8)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="./ltx_merged_model",
        help="Path for output"
    )
    
    parser.add_argument(
        "--hdc_dim",
        type=int,
        default=DEFAULT_HDC_DIM,
        help="HDC dimension"
    )
    
    parser.add_argument(
        "--no_safety",
        action="store_true",
        help="Disable safety training"
    )
    
    parser.add_argument(
        "--no_resonator",
        action="store_true",
        help="Disable resonator training"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file"
    )
    
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Path to save merged model"
    )
    
    parser.add_argument(
        "--unified_checkpoint",
        type=str,
        default="./checkpoints/unified_hdc_checkpoint.pt",
        help="Path to save unified deduplication checkpoint for cross-model knowledge sharing"
    )
    
    parser.add_argument(
        "--load_unified_checkpoint",
        type=str,
        default=None,
        help="Path to load existing unified checkpoint from previous transfer (enables knowledge sharing)"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = TrainingConfig.from_dict(json.load(f))
    else:
        config = TrainingConfig(
            model_path=args.model_path,
            output_path=args.output_path,
            hdc_dim=args.hdc_dim,
            safety=SafetyTrainingConfig(
                enable_safety_training=not args.no_safety
            ),
            resonator=ResonatorTrainingConfig(
                enable_resonator=not args.no_resonator
            )
        )
    
    # Create pipeline
    pipeline = LTXTrainingPipeline(config)
    
    # Load existing unified checkpoint if specified (enables cross-model knowledge sharing)
    if args.load_unified_checkpoint:
        print(f"\nLoading unified checkpoint from: {args.load_unified_checkpoint}")
        pipeline.load_unified_checkpoint(args.load_unified_checkpoint)
    
    # Run training
    results = pipeline.run_full_training()
    
    # Save unified checkpoint for cross-model knowledge sharing
    pipeline.save_unified_checkpoint(args.unified_checkpoint)
    
    # Save model if requested
    if args.save_model:
        pipeline.save_merged_model(args.save_model)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
