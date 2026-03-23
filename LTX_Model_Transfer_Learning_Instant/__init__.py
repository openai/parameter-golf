"""
LTX Model Transfer Learning Instant - Package Initialization

This module provides instant transfer learning for LTX-2.3 (Lightweight Transformer for
Audio-Video Generation) models to HDC (Hyperdimensional Computing) space.

Key Features:
- Instant Transfer Learning: Extract knowledge from LTX-2.3 into HDC recipes
- Deterministic Encoding: All patterns encoded as reproducible seed sequences
- Audio-Video Joint Understanding: Handle synchronized audio-video generation patterns
- Real-time Inference: 60+ FPS inference using HDC pattern matching
- 100-Year Memory: Unlimited audio-video pattern storage with zero RAM increase
- Safety Training: Context-aware safety masking during transfer learning
- Resonator Network: Parallel factorization for pattern discovery
- Incremental Training: Support for additional modalities

Architecture Integration (Pure HDC - No CNN):
- Uses WalshHadamardBasis for deterministic orthogonal projection
- Uses BLAKE3 for deterministic vector generation
- Stores extracted patterns as IdentityRecipes in RecipeStorage
- Preserves audio-video generation chains as seed sequences
- Enables cross-model knowledge transfer via universal Hadamard basis
- Integrates ResonatorNetwork for parallel factorization
- Integrates SafetyRegistry for safety-aware training
"""

from .ltx_latent_mapper import (
    LTXLatentMapper,
    LTXConfig,
    LTXLayerType,
    LTXModalityType,
    AudioVideoPattern,
    create_ltx_mapper,
    extract_and_map_ltx
)

from .ltx_chain_seeds import (
    LTXChainStorage,
    LTXChainSeed,
    LTXSeedStep,
    LTXChainOperation,
    create_ltx_chain_system,
    create_ltx_chain_from_trajectory
)

from .ltx_relationship_deduplication import (
    LTXPatternDeduplicator,
    LTXRelationshipGraph,
    LTXDeduplicationConfig,
    LTXRelationshipType,
    create_ltx_deduplicator
)

from .ltx_training_pipeline import (
    LTXTrainingPipeline,
    TrainingConfig,
    TrainingPhase,
    TrainingStatistics,
    SafetyTrainingConfig,
    ResonatorTrainingConfig,
    RecipeStorageConfig,
    MergedHDCModel,
    create_training_pipeline,
    run_quick_training
)

__all__ = [
    # Latent Mapper
    'LTXLatentMapper',
    'LTXConfig',
    'LTXLayerType',
    'LTXModalityType',
    'AudioVideoPattern',
    'create_ltx_mapper',
    'extract_and_map_ltx',
    
    # Chain Seeds
    'LTXChainStorage',
    'LTXChainSeed',
    'LTXSeedStep',
    'LTXChainOperation',
    'create_ltx_chain_system',
    'create_ltx_chain_from_trajectory',
    
    # Deduplication
    'LTXPatternDeduplicator',
    'LTXRelationshipGraph',
    'LTXDeduplicationConfig',
    'LTXRelationshipType',
    'create_ltx_deduplicator',
    
    # Training Pipeline
    'LTXTrainingPipeline',
    'TrainingConfig',
    'TrainingPhase',
    'TrainingStatistics',
    'SafetyTrainingConfig',
    'ResonatorTrainingConfig',
    'RecipeStorageConfig',
    'MergedHDCModel',
    'create_training_pipeline',
    'run_quick_training'
]

__version__ = '1.1.0'
__author__ = 'HDC Team'
__description__ = 'Instant Transfer Learning for LTX-2.3 Audio-Video Foundation Model to HDC Space'
