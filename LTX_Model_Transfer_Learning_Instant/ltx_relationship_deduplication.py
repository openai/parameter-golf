"""
LTX Relationship Deduplication - Deduplicate Audio-Video Patterns While Preserving Relationships

This module provides pattern deduplication for LTX audio-video generation patterns,
ensuring efficient storage while preserving semantic relationships between patterns.

Key Features:
1. Content hash for O(1) exact duplicate detection (BLAKE3 with SHA256 fallback)
2. Seed-based storage - vectors generated on demand from BLAKE3 seeds
3. Circular temporal encoding for sequence tracking (replaces diffusion timesteps)
4. Relationship graph for semantic preservation
5. Audio-video modality-aware clustering
6. Generation trajectory tracking via circular shifts
7. Hadamard similarity for near-duplicate detection
8. Unified cross-model integration for multi-modal knowledge transfer

Architecture Integration:
- Uses HDC Hamming distance for similarity computation
- XOR-based relationship encoding
- BLAKE3 seeds for pattern identification and on-demand vector generation
- Zero-weight procedural generation (no vector storage)
- Efficient uint64 bit-packed operations
- Circular temporal encoding: unlimited sequence depth with zero RAM increase
"""

import numpy as np
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import threading
from collections import defaultdict

# BLAKE3 for faster hashing (falls back to SHA256 if not available)
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Import HDC components
from ...HDC_Core_Model.Recipes_Seeds.walsh_hadamard_core import (
    WalshHadamardBasis,
    TernaryHadamardEncoder,
    DEFAULT_HDC_DIM
)


def seed_to_hypervector(seed_string: str, uint64_count: int = 2048) -> np.ndarray:
    """
    Deterministically generate a hypervector from any string.
    Identical output on every machine, every OS, forever.
    
    Uses BLAKE3 for:
    - Unlimited seed generation (extendable output)
    - Single API call (no counter loop needed)
    - ~3x faster than SHA256
    - Cross-platform reproducibility
    """
    num_bytes = uint64_count * 8  # 8 bytes per uint64
    
    if HAS_BLAKE3:
        import blake3 as blake3_module
        hash_bytes = blake3_module.blake3(seed_string.encode()).digest(length=num_bytes)
    else:
        # Fallback to SHA256 with counter (slower but works)
        hash_bytes = b''
        counter = 0
        while len(hash_bytes) < num_bytes:
            h = hashlib.sha256(f"{seed_string}:{counter}".encode()).digest()
            hash_bytes += h
        hash_bytes = hash_bytes[:num_bytes]
    
    return np.frombuffer(hash_bytes, dtype=np.uint64).copy()


class LTXRelationshipType(Enum):
    """Types of relationships between LTX patterns."""
    # Temporal relationships
    PRECEDES = "precedes"           # Temporal ordering
    FOLLOWS = "follows"             # Reverse temporal ordering
    DENOISES_TO = "denoises_to"     # Denoising trajectory
    
    # Modal relationships
    AUDIO_SYNC = "audio_sync"       # Audio-video synchronization
    VIDEO_SYNC = "video_sync"       # Video-audio synchronization
    AUDIO_VIDEO_BIND = "audio_video_bind"  # Cross-modal binding
    
    # Semantic relationships
    SIMILAR = "similar"             # Similar content
    VARIANT = "variant"             # Variant of same content
    COMPOSED = "composed"           # Composed from other patterns
    SEMANTIC_SIMILAR = "semantic_similar"
    CONCEPT_INHERITANCE = "concept_inheritance"
    SKILL_TRANSFER = "skill_transfer"
    
    # Generation relationships
    CONDITIONED_BY = "conditioned_by"  # Text/image conditioning
    GENERATES = "generates"             # Output generation
    
    # Frame relationships
    FRAME_PREV = "frame_prev"       # Previous frame
    FRAME_NEXT = "frame_next"       # Next frame
    SCENE_SIMILAR = "scene_similar" # Similar scene


@dataclass
class LTXRelationshipEdge:
    """
    An edge in the LTX relationship graph.
    
    Provides rich metadata for relationship tracking including
    strength, sequence positions (circular temporal encoding), and arbitrary metadata.
    
    Note: Uses circular temporal encoding instead of diffusion timesteps for
    unlimited sequence depth with zero RAM increase.
    """
    source_id: str
    target_id: str
    relationship_type: LTXRelationshipType
    strength: float = 1.0
    sequence_position: int = 0  # Position in circular temporal encoding
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge to dictionary."""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship_type': self.relationship_type.value,
            'strength': self.strength,
            'sequence_position': self.sequence_position,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LTXRelationshipEdge':
        """Deserialize edge from dictionary."""
        return cls(
            source_id=data['source_id'],
            target_id=data['target_id'],
            relationship_type=LTXRelationshipType(data['relationship_type']),
            strength=data.get('strength', 1.0),
            sequence_position=data.get('sequence_position', data.get('timestep_from', 0)),  # Backward compat
            metadata=data.get('metadata', {})
        )


@dataclass
class LTXDeduplicationConfig:
    """Configuration for LTX pattern deduplication."""
    similarity_threshold: float = 0.95
    video_similarity_threshold: float = 0.92
    audio_similarity_threshold: float = 0.90
    joint_similarity_threshold: float = 0.93
    
    max_patterns_per_video: int = 100
    max_patterns_per_audio: int = 50
    
    enable_cross_modal: bool = True
    enable_streaming_dedup: bool = True
    
    # Performance settings
    use_gpu: bool = True
    batch_size: int = 32
    num_workers: int = 4
    
    # Video-specific settings
    frame_similarity_tolerance: float = 0.1
    video_sequence_window: int = 5
    
    # Clustering settings
    auto_merge_exact: bool = True
    create_variants: bool = True
    max_variants: int = 10
    preserve_relationships: bool = True
    track_trajectories: bool = True
    
    # Timestep-aware settings
    cross_timestep_threshold: float = 0.85
    temporal_smoothing: bool = True
    
    # Modality settings
    separate_modalities: bool = False
    audio_video_sync_threshold: float = 0.9


@dataclass
class LTXPattern:
    """
    Represents an LTX audio-video pattern.
    
    Uses seed-based storage - vectors are generated on demand from seed_string.
    This follows the Zero-Weight Procedural Generation architecture.
    """
    pattern_id: str
    pattern_type: str  # 'video', 'audio', 'joint', 'frame'
    seed_string: str  # BLAKE3 seed for on-demand vector generation
    hadamard_index: int = 0
    content_hash: str = ""  # For O(1) exact duplicate detection
    
    # Pattern characteristics
    modality: str = "joint"  # 'video', 'audio', 'joint'
    layer_name: str = ""
    generation_mode: str = "text_to_audio_video"  # 'text_to_audio_video', 'image_to_video', etc.
    
    # Video characteristics
    frame_count: int = 0
    frame_rate: float = 24.0
    resolution: Tuple[int, int] = (512, 512)
    video_duration: float = 0.0
    
    # Audio characteristics
    sample_rate: int = 44100
    audio_duration: float = 0.0
    audio_channels: int = 2
    
    # Timestep for trajectory tracking
    timestep: int = 0
    
    # Cluster information
    cluster_id: Optional[str] = None
    is_centroid: bool = False
    
    # Characteristics
    characteristics: Dict[str, float] = field(default_factory=dict)
    
    # Relationships
    related_patterns: List[str] = field(default_factory=list)
    relationship_types: Dict[str, LTXRelationshipType] = field(default_factory=dict)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Cache for generated vector (optional, for performance)
    _cached_vector: Optional[np.ndarray] = field(default=None, repr=False)
    _cache_valid: bool = field(default=False, repr=False)
    
    def get_vector(self, uint64_count: int = 2048, use_cache: bool = True) -> np.ndarray:
        """
        Generate vector on demand from seed_string.
        
        Args:
            uint64_count: Number of uint64 elements (HDC_DIM // 64)
            use_cache: Whether to use cached vector if available
            
        Returns:
            Generated hypervector as uint64 array
        """
        if use_cache and self._cache_valid and self._cached_vector is not None:
            return self._cached_vector
        
        vector = seed_to_hypervector(self.seed_string, uint64_count)
        
        if use_cache:
            self._cached_vector = vector
            self._cache_valid = True
        
        return vector
    
    def invalidate_cache(self):
        """Invalidate the cached vector."""
        self._cache_valid = False
        self._cached_vector = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'seed_string': self.seed_string,
            'hadamard_index': self.hadamard_index,
            'content_hash': self.content_hash,
            'modality': self.modality,
            'layer_name': self.layer_name,
            'generation_mode': self.generation_mode,
            'frame_count': self.frame_count,
            'frame_rate': self.frame_rate,
            'resolution': list(self.resolution),
            'video_duration': self.video_duration,
            'sample_rate': self.sample_rate,
            'audio_duration': self.audio_duration,
            'audio_channels': self.audio_channels,
            'timestep': self.timestep,
            'cluster_id': self.cluster_id,
            'is_centroid': self.is_centroid,
            'characteristics': self.characteristics,
            'related_patterns': self.related_patterns,
            'relationship_types': {k: v.value for k, v in self.relationship_types.items()},
            'created_at': self.created_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LTXPattern':
        """Create from dictionary."""
        resolution = data.get('resolution', [512, 512])
        if isinstance(resolution, list):
            resolution = tuple(resolution)
            
        return cls(
            pattern_id=data['pattern_id'],
            pattern_type=data['pattern_type'],
            seed_string=data['seed_string'],
            hadamard_index=data.get('hadamard_index', 0),
            content_hash=data.get('content_hash', ''),
            modality=data.get('modality', 'joint'),
            layer_name=data.get('layer_name', ''),
            generation_mode=data.get('generation_mode', 'text_to_audio_video'),
            frame_count=data.get('frame_count', 0),
            frame_rate=data.get('frame_rate', 24.0),
            resolution=resolution,
            video_duration=data.get('video_duration', 0.0),
            sample_rate=data.get('sample_rate', 44100),
            audio_duration=data.get('audio_duration', 0.0),
            audio_channels=data.get('audio_channels', 2),
            timestep=data.get('timestep', 0),
            cluster_id=data.get('cluster_id'),
            is_centroid=data.get('is_centroid', False),
            characteristics=data.get('characteristics', {}),
            related_patterns=data.get('related_patterns', []),
            relationship_types={k: LTXRelationshipType(v) for k, v in data.get('relationship_types', {}).items()},
            created_at=data.get('created_at', datetime.now().isoformat()),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', datetime.now().isoformat()),
            metadata=data.get('metadata', {})
        )


class LTXRelationshipGraph:
    """
    Manages relationships between LTX audio-video patterns.
    
    Features:
    - Rich edge metadata via LTXRelationshipEdge
    - Denoising trajectory tracking
    - Path finding
    - Subgraph extraction
    - Build and maintain relationship graphs
    - Fast pattern lookup by relationship type
    - Support for transitive relationships
    - Efficient graph traversal
    """
    
    def __init__(self):
        """Initialize the relationship graph."""
        self._patterns: Dict[str, LTXPattern] = {}
        
        # Edge storage
        self._edges: Dict[str, List[LTXRelationshipEdge]] = defaultdict(list)
        self._reverse_edges: Dict[str, List[LTXRelationshipEdge]] = defaultdict(list)
        
        # Legacy edge storage for backward compatibility
        self._legacy_edges: Dict[str, Set[str]] = {}
        self._edge_types: Dict[Tuple[str, str], LTXRelationshipType] = {}
        
        # Indexes for fast lookup
        self._by_type: Dict[str, Set[str]] = {}
        self._by_modality: Dict[str, Set[str]] = {}
        self._by_generation_mode: Dict[str, Set[str]] = {}
        self._by_relationship: Dict[LTXRelationshipType, Set[Tuple[str, str]]] = {}
        
        self._lock = threading.Lock()
    
    def add_pattern(self, pattern: LTXPattern):
        """Add a pattern to the graph."""
        with self._lock:
            self._patterns[pattern.pattern_id] = pattern
            
            # Index by type
            if pattern.pattern_type not in self._by_type:
                self._by_type[pattern.pattern_type] = set()
            self._by_type[pattern.pattern_type].add(pattern.pattern_id)
            
            # Index by modality
            if pattern.modality not in self._by_modality:
                self._by_modality[pattern.modality] = set()
            self._by_modality[pattern.modality].add(pattern.pattern_id)
            
            # Index by generation mode
            if pattern.generation_mode not in self._by_generation_mode:
                self._by_generation_mode[pattern.generation_mode] = set()
            self._by_generation_mode[pattern.generation_mode].add(pattern.pattern_id)
            
            # Initialize edge set (legacy)
            if pattern.pattern_id not in self._legacy_edges:
                self._legacy_edges[pattern.pattern_id] = set()
    
    def add_edge(self, edge: LTXRelationshipEdge):
        """
        Add a relationship edge.
        
        Args:
            edge: The relationship edge to add
        """
        with self._lock:
            self._edges[edge.source_id].append(edge)
            self._reverse_edges[edge.target_id].append(edge)
            
            # Also update legacy indexes
            if edge.source_id not in self._legacy_edges:
                self._legacy_edges[edge.source_id] = set()
            self._legacy_edges[edge.source_id].add(edge.target_id)
            self._edge_types[(edge.source_id, edge.target_id)] = edge.relationship_type
            
            if edge.relationship_type not in self._by_relationship:
                self._by_relationship[edge.relationship_type] = set()
            self._by_relationship[edge.relationship_type].add((edge.source_id, edge.target_id))
    
    def add_relationship(
        self,
        pattern_id1: str,
        pattern_id2: str,
        relationship_type: LTXRelationshipType,
        bidirectional: bool = False,
        strength: float = 1.0,
        sequence_position: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a relationship between two patterns.
        
        Uses circular temporal encoding for sequence tracking instead of diffusion timesteps.
        This provides unlimited temporal depth with zero RAM increase.
        
        Args:
            pattern_id1: Source pattern ID
            pattern_id2: Target pattern ID
            relationship_type: Type of relationship
            bidirectional: Whether to create reverse edge
            strength: Relationship strength
            sequence_position: Position in circular temporal encoding (for trajectory tracking)
            metadata: Additional metadata
        """
        with self._lock:
            # Create edge with circular temporal encoding
            edge = LTXRelationshipEdge(
                source_id=pattern_id1,
                target_id=pattern_id2,
                relationship_type=relationship_type,
                strength=strength,
                sequence_position=sequence_position,
                metadata=metadata or {}
            )
            self._edges[pattern_id1].append(edge)
            self._reverse_edges[pattern_id2].append(edge)
            
            # Update legacy indexes
            if pattern_id1 not in self._legacy_edges:
                self._legacy_edges[pattern_id1] = set()
            self._legacy_edges[pattern_id1].add(pattern_id2)
            self._edge_types[(pattern_id1, pattern_id2)] = relationship_type
            
            if relationship_type not in self._by_relationship:
                self._by_relationship[relationship_type] = set()
            self._by_relationship[relationship_type].add((pattern_id1, pattern_id2))
            
            # Add reverse edge if bidirectional
            if bidirectional:
                reverse_edge = LTXRelationshipEdge(
                    source_id=pattern_id2,
                    target_id=pattern_id1,
                    relationship_type=relationship_type,
                    strength=strength,
                    sequence_position=sequence_position + 1,  # Next position in sequence
                    metadata=metadata or {}
                )
                self._edges[pattern_id2].append(reverse_edge)
                self._reverse_edges[pattern_id1].append(reverse_edge)
                
                if pattern_id2 not in self._legacy_edges:
                    self._legacy_edges[pattern_id2] = set()
                self._legacy_edges[pattern_id2].add(pattern_id1)
                self._edge_types[(pattern_id2, pattern_id1)] = relationship_type
                self._by_relationship[relationship_type].add((pattern_id2, pattern_id1))
    
    def remove_edge(self, source_id: str, target_id: str, 
                    relationship_type: Optional[LTXRelationshipType] = None):
        """
        Remove a relationship edge.
        
        Args:
            source_id: Source pattern ID
            target_id: Target pattern ID
            relationship_type: Optional relationship type filter
        """
        with self._lock:
            # Remove from storage
            self._edges[source_id] = [
                e for e in self._edges[source_id]
                if not (e.target_id == target_id and 
                       (relationship_type is None or e.relationship_type == relationship_type))
            ]
            self._reverse_edges[target_id] = [
                e for e in self._reverse_edges[target_id]
                if not (e.source_id == source_id and
                       (relationship_type is None or e.relationship_type == relationship_type))
            ]
            
            # Remove from legacy storage
            self._legacy_edges[source_id] = self._legacy_edges.get(source_id, set())
            self._legacy_edges[source_id].discard(target_id)
            
            edge_key = (source_id, target_id)
            if edge_key in self._edge_types:
                rel_type = self._edge_types.pop(edge_key)
                if rel_type in self._by_relationship:
                    self._by_relationship[rel_type].discard(edge_key)
    
    def get_outgoing(self, node_id: str, 
                     relationship_type: Optional[LTXRelationshipType] = None) -> List[LTXRelationshipEdge]:
        """
        Get outgoing edges from a node.
        
        Args:
            node_id: Pattern ID
            relationship_type: Optional filter by relationship type
            
        Returns:
            List of outgoing edges
        """
        edges = self._edges.get(node_id, [])
        if relationship_type:
            edges = [e for e in edges if e.relationship_type == relationship_type]
        return edges
    
    def get_incoming(self, node_id: str,
                     relationship_type: Optional[LTXRelationshipType] = None) -> List[LTXRelationshipEdge]:
        """
        Get incoming edges to a node.
        
        Args:
            node_id: Pattern ID
            relationship_type: Optional filter by relationship type
            
        Returns:
            List of incoming edges
        """
        edges = self._reverse_edges.get(node_id, [])
        if relationship_type:
            edges = [e for e in edges if e.relationship_type == relationship_type]
        return edges
    
    def get_pattern(self, pattern_id: str) -> Optional[LTXPattern]:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)
    
    def get_related_patterns(
        self,
        pattern_id: str,
        relationship_type: Optional[LTXRelationshipType] = None
    ) -> List[Tuple[str, LTXRelationshipType]]:
        """Get patterns related to a given pattern."""
        with self._lock:
            if pattern_id not in self._legacy_edges:
                return []
            
            related = []
            for related_id in self._legacy_edges[pattern_id]:
                edge_type = self._edge_types.get((pattern_id, related_id))
                if edge_type is None:
                    continue
                if relationship_type is None or edge_type == relationship_type:
                    related.append((related_id, edge_type))
            
            return related
    
    def get_denoising_trajectory(self, pattern_id: str,
                                  target_position: int = 0) -> List[str]:
        """
        Get the denoising trajectory from a pattern to target sequence position.
        
        Uses circular temporal encoding for trajectory tracking instead of diffusion timesteps.
        
        Args:
            pattern_id: Starting pattern ID
            target_position: Target sequence position (default 0 = start of sequence)
            
        Returns:
            List of pattern IDs in the trajectory
        """
        trajectory = [pattern_id]
        current = pattern_id
        visited = {pattern_id}
        
        while True:
            # Find next step in denoising trajectory
            outgoing = self.get_outgoing(current, LTXRelationshipType.DENOISES_TO)
            if not outgoing:
                break
            
            # Find edge with lowest sequence position (closest to target)
            next_edge = min(outgoing, key=lambda e: e.sequence_position)
            
            if next_edge.target_id in visited:
                break
            
            trajectory.append(next_edge.target_id)
            visited.add(next_edge.target_id)
            current = next_edge.target_id
            
            # Check if we've reached target position
            if next_edge.sequence_position <= target_position:
                break
        
        return trajectory
    
    def get_audio_video_sync_patterns(self, pattern_id: str) -> Dict[str, List[str]]:
        """
        Get synchronized audio/video patterns for a given pattern.
        
        Args:
            pattern_id: Pattern ID to find sync patterns for
            
        Returns:
            Dictionary with 'audio' and 'video' lists of synchronized pattern IDs
        """
        result = {'audio': [], 'video': []}
        
        # Get audio sync relationships
        audio_sync = self.get_outgoing(pattern_id, LTXRelationshipType.AUDIO_SYNC)
        audio_sync += self.get_incoming(pattern_id, LTXRelationshipType.AUDIO_SYNC)
        for edge in audio_sync:
            other_id = edge.target_id if edge.source_id == pattern_id else edge.source_id
            other_pattern = self._patterns.get(other_id)
            if other_pattern:
                if other_pattern.modality == 'audio':
                    result['audio'].append(other_id)
                elif other_pattern.modality == 'video':
                    result['video'].append(other_id)
        
        # Get video sync relationships
        video_sync = self.get_outgoing(pattern_id, LTXRelationshipType.VIDEO_SYNC)
        video_sync += self.get_incoming(pattern_id, LTXRelationshipType.VIDEO_SYNC)
        for edge in video_sync:
            other_id = edge.target_id if edge.source_id == pattern_id else edge.source_id
            other_pattern = self._patterns.get(other_id)
            if other_pattern and other_pattern.modality == 'video':
                result['video'].append(other_id)
        
        # Get audio-video bindings
        av_bind = self.get_outgoing(pattern_id, LTXRelationshipType.AUDIO_VIDEO_BIND)
        av_bind += self.get_incoming(pattern_id, LTXRelationshipType.AUDIO_VIDEO_BIND)
        for edge in av_bind:
            other_id = edge.target_id if edge.source_id == pattern_id else edge.source_id
            other_pattern = self._patterns.get(other_id)
            if other_pattern:
                if other_pattern.modality == 'audio':
                    result['audio'].append(other_id)
                elif other_pattern.modality == 'video':
                    result['video'].append(other_id)
        
        # Deduplicate
        for key in result:
            result[key] = list(set(result[key]))
        
        return result
    
    def get_cross_modal_patterns(self, pattern_id: str) -> Dict[str, List[str]]:
        """
        Get cross-modal patterns for a given pattern.
        
        Args:
            pattern_id: Pattern ID to find cross-modal patterns for
            
        Returns:
            Dictionary with 'audio', 'video', 'joint' lists
        """
        result = {'audio': [], 'video': [], 'joint': []}
        
        # Get all cross-modal bindings
        for rel_type in [LTXRelationshipType.AUDIO_SYNC, LTXRelationshipType.VIDEO_SYNC, 
                         LTXRelationshipType.AUDIO_VIDEO_BIND]:
            bindings = self.get_outgoing(pattern_id, rel_type)
            bindings += self.get_incoming(pattern_id, rel_type)
            
            for edge in bindings:
                other_id = edge.target_id if edge.source_id == pattern_id else edge.source_id
                other_pattern = self._patterns.get(other_id)
                if other_pattern and other_pattern.modality in result:
                    result[other_pattern.modality].append(other_id)
        
        # Deduplicate
        for key in result:
            result[key] = list(set(result[key]))
        
        return result
    
    def find_path(self, source_id: str, target_id: str, 
                  max_depth: int = 10) -> Optional[List[str]]:
        """
        Find a path between two patterns.
        
        Args:
            source_id: Source pattern ID
            target_id: Target pattern ID
            max_depth: Maximum search depth
            
        Returns:
            List of pattern IDs forming the path, or None if no path found
        """
        if source_id == target_id:
            return [source_id]
        
        visited = {source_id}
        queue = [(source_id, [source_id])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for edge in self.get_outgoing(current):
                if edge.target_id == target_id:
                    return path + [target_id]
                
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, path + [edge.target_id]))
        
        return None
    
    def get_subgraph(self, node_ids: Set[str]) -> Dict[str, List[LTXRelationshipEdge]]:
        """
        Extract a subgraph containing only the specified nodes.
        
        Args:
            node_ids: Set of node IDs to include
            
        Returns:
            Dictionary mapping node IDs to their edges within the subgraph
        """
        subgraph = {}
        for node_id in node_ids:
            edges = self.get_outgoing(node_id)
            subgraph[node_id] = [e for e in edges if e.target_id in node_ids]
        return subgraph
    
    def get_patterns_by_type(self, pattern_type: str) -> List[LTXPattern]:
        """Get all patterns of a specific type."""
        with self._lock:
            pattern_ids = self._by_type.get(pattern_type, set())
            return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]
    
    def get_patterns_by_modality(self, modality: str) -> List[LTXPattern]:
        """Get all patterns of a specific modality."""
        with self._lock:
            pattern_ids = self._by_modality.get(modality, set())
            return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]
    
    def get_patterns_by_generation_mode(self, generation_mode: str) -> List[LTXPattern]:
        """Get all patterns of a specific generation mode."""
        with self._lock:
            pattern_ids = self._by_generation_mode.get(generation_mode, set())
            return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]
    
    def find_similar_video_patterns(
        self,
        frame_count: int,
        tolerance: int = 5
    ) -> List[LTXPattern]:
        """Find patterns with similar frame counts."""
        with self._lock:
            similar = []
            for pattern in self._patterns.values():
                if pattern.modality in ['video', 'joint']:
                    if abs(pattern.frame_count - frame_count) <= tolerance:
                        similar.append(pattern)
            return similar
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            return {
                'total_patterns': len(self._patterns),
                'total_edges': sum(len(edges) for edges in self._edges.values()),
                'pattern_types': {k: len(v) for k, v in self._by_type.items()},
                'modalities': {k: len(v) for k, v in self._by_modality.items()},
                'generation_modes': {k: len(v) for k, v in self._by_generation_mode.items()},
                'relationship_types': {k.value: len(v) for k, v in self._by_relationship.items()}
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        all_edges = []
        for edges in self._edges.values():
            all_edges.extend([e.to_dict() for e in edges])
        return {'edges': all_edges}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LTXRelationshipGraph':
        """Deserialize graph from dictionary."""
        graph = cls()
        for edge_data in data.get('edges', []):
            edge = LTXRelationshipEdge.from_dict(edge_data)
            graph.add_edge(edge)
        return graph


class LTXPatternDeduplicator:
    """
    Deduplicates LTX audio-video patterns using HDC similarity.
    
    Uses seed-based storage - vectors are generated on demand from BLAKE3 seeds.
    This follows the Zero-Weight Procedural Generation architecture where:
    - No vectors are stored, only seed strings
    - Vectors are generated deterministically on demand
    - Content hash provides O(1) exact duplicate detection
    - Clustering groups similar patterns
    - Relationship graph preserves semantic connections
    
    Features:
    - Efficient similarity-based deduplication
    - Cross-modal pattern matching (audio-video)
    - Video sequence deduplication
    - Generation-mode-specific pattern management
    - Timestep-aware trajectory tracking
    - Streaming deduplication support
    """
    
    def __init__(
        self,
        config: Optional[LTXDeduplicationConfig] = None,
        hdc_dim: int = DEFAULT_HDC_DIM
    ):
        """
        Initialize the deduplicator.
        
        Args:
            config: Deduplication configuration
            hdc_dim: HDC dimension for similarity computation
        """
        self.config = config or LTXDeduplicationConfig()
        self.hdc_dim = hdc_dim
        self.uint64_count = hdc_dim // 64
        
        # Initialize components
        self.hadamard = WalshHadamardBasis(dim=hdc_dim)
        self.ternary_encoder = TernaryHadamardEncoder(dim=hdc_dim)
        self.relationship_graph = LTXRelationshipGraph()
        
        # Pattern storage (seed-based, no vectors stored)
        self._patterns: Dict[str, LTXPattern] = {}
        self._content_hash_index: Dict[str, str] = {}  # hash -> pattern_id
        
        # Cluster storage
        self._clusters: Dict[str, List[str]] = defaultdict(list)
        self._pattern_to_cluster: Dict[str, str] = {}
        
        # Indexes for fast lookup
        self._by_seed_string: Dict[str, str] = {}  # seed_string -> pattern_id
        self._by_generation_mode: Dict[str, Set[str]] = {}
        self._by_layer: Dict[str, Set[str]] = {}
        
        # Vector cache for performance (optional, can be disabled)
        self._vector_cache: Dict[str, np.ndarray] = {}
        self._cache_enabled: bool = True
        self._max_cache_size: int = 10000
        
        # Statistics
        self._stats = {
            'patterns_added': 0,
            'duplicates_found': 0,
            'exact_duplicates_merged': 0,
            'near_duplicates_found': 0,
            'relationships_created': 0,
            'relationships_preserved': 0,
            'similarity_checks': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self._lock = threading.Lock()
    
    def _compute_content_hash(self, seed_string: str) -> str:
        """
        Compute content hash for a seed string using BLAKE3 (faster) with SHA256 fallback.
        
        Provides O(1) exact duplicate detection.
        """
        if HAS_BLAKE3:
            import blake3 as blake3_module
            return blake3_module.blake3(seed_string.encode()).hexdigest()
        else:
            return hashlib.sha256(seed_string.encode()).hexdigest()
    
    def _get_vector(self, pattern: LTXPattern) -> np.ndarray:
        """
        Get vector for a pattern, using cache if available.
        
        Args:
            pattern: Pattern to get vector for
            
        Returns:
            Generated hypervector
        """
        if self._cache_enabled and pattern.pattern_id in self._vector_cache:
            self._stats['cache_hits'] += 1
            return self._vector_cache[pattern.pattern_id]
        
        self._stats['cache_misses'] += 1
        vector = pattern.get_vector(self.uint64_count, use_cache=False)
        
        if self._cache_enabled:
            # Manage cache size
            if len(self._vector_cache) >= self._max_cache_size:
                # Remove oldest entries (simple LRU approximation)
                keys_to_remove = list(self._vector_cache.keys())[:self._max_cache_size // 2]
                for key in keys_to_remove:
                    del self._vector_cache[key]
            
            self._vector_cache[pattern.pattern_id] = vector
        
        return vector
    
    def _compute_hadamard_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute Hadamard similarity between two vectors.
        
        Handles ternary-encoded vectors (-1, 0, +1) properly by using
        cosine similarity or normalized dot product.
        
        For binary vectors (0/1 or -1/+1), uses Hamming distance.
        """
        # Check if vectors are ternary-encoded (contain -1, 0, +1)
        unique_vals = np.unique(np.concatenate([np.unique(vec1), np.unique(vec2)]))
        is_ternary = np.any(unique_vals < 0) or (0 in unique_vals and np.any(unique_vals > 0))
        
        if is_ternary:
            # Use cosine similarity for ternary vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity: dot product / (norm1 * norm2)
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            
            # Normalize from [-1, 1] to [0, 1] for threshold comparison
            return (similarity + 1.0) / 2.0
        else:
            # Use Hamming distance for binary/uint64 vectors
            v1 = vec1.astype(np.uint64) if vec1.dtype != np.uint64 else vec1
            v2 = vec2.astype(np.uint64) if vec2.dtype != np.uint64 else vec2
            
            xored = np.bitwise_xor(v1, v2)
            differences = np.unpackbits(xored.view(np.uint8)).sum()
            return 1.0 - (differences / (len(v1) * 64))
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute similarity between two HDC vectors.
        
        Uses Hadamard similarity for better accuracy with ternary vectors.
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        # Try Hadamard similarity first
        try:
            return self._compute_hadamard_similarity(vec1, vec2)
        except Exception:
            # Fallback to simple Hamming similarity
            matches = np.sum(vec1 == vec2)
            return float(matches / len(vec1))
    
    def add_pattern(self, pattern: LTXPattern) -> bool:
        """
        Add a pattern to the deduplicator.
        
        Args:
            pattern: Pattern to add
            
        Returns:
            True if pattern was added (not a duplicate)
        """
        with self._lock:
            # Check for exact seed match
            if pattern.seed_string in self._by_seed_string:
                existing_id = self._by_seed_string[pattern.seed_string]
                self._stats['duplicates_found'] += 1
                
                # Add relationship
                self.relationship_graph.add_relationship(
                    existing_id,
                    pattern.pattern_id,
                    LTXRelationshipType.SEMANTIC_SIMILAR,
                    bidirectional=True
                )
                
                return False
            
            # Store pattern
            self._patterns[pattern.pattern_id] = pattern
            
            # Update indexes
            self._by_seed_string[pattern.seed_string] = pattern.pattern_id
            
            if pattern.generation_mode:
                if pattern.generation_mode not in self._by_generation_mode:
                    self._by_generation_mode[pattern.generation_mode] = set()
                self._by_generation_mode[pattern.generation_mode].add(pattern.pattern_id)
            
            if pattern.layer_name:
                if pattern.layer_name not in self._by_layer:
                    self._by_layer[pattern.layer_name] = set()
                self._by_layer[pattern.layer_name].add(pattern.pattern_id)
            
            # Add to relationship graph
            self.relationship_graph.add_pattern(pattern)
            
            self._stats['patterns_added'] += 1
            
            return True
    
    def check_duplicate(
        self,
        seed_string: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a seed string is a duplicate of existing patterns.
        
        Args:
            seed_string: Seed string to check
            threshold: Optional similarity threshold override
            
        Returns:
            Tuple of (is_duplicate, existing_pattern_id)
        """
        threshold = threshold or self.config.similarity_threshold
        
        with self._lock:
            # Check for exact seed match first (O(1))
            if seed_string in self._by_seed_string:
                return True, self._by_seed_string[seed_string]
            
            # Check content hash
            content_hash = self._compute_content_hash(seed_string)
            if content_hash in self._content_hash_index:
                return True, self._content_hash_index[content_hash]
            
            return False, None
    
    def deduplicate(
        self,
        vector: np.ndarray,
        layer_name: str,
        seed_string: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[LTXPattern, bool, Optional[str]]:
        """
        Deduplicate a pattern and return the result.
        
        Uses seed-based storage - the seed_string is stored, not the vector.
        The vector is used for similarity comparison but not persisted.
        
        Args:
            vector: HDC vector for similarity comparison (not stored)
            layer_name: Name of the source layer
            seed_string: Seed string for the pattern (stored for on-demand generation)
            metadata: Optional metadata
            
        Returns:
            Tuple of (pattern, is_new, cluster_id)
        """
        if metadata is None:
            metadata = {}
        
        with self._lock:
            # Compute content hash for O(1) exact duplicate detection
            content_hash = self._compute_content_hash(seed_string)
            
            # Check for exact duplicate by seed string (O(1))
            if seed_string in self._by_seed_string:
                existing_id = self._by_seed_string[seed_string]
                existing_pattern = self._patterns[existing_id]
                existing_pattern.access_count += 1
                existing_pattern.last_accessed = datetime.now().isoformat()
                
                if self.config.auto_merge_exact:
                    self._stats['exact_duplicates_merged'] += 1
                    return existing_pattern, False, existing_pattern.cluster_id
            
            # Check for exact duplicate by content hash (O(1))
            if content_hash in self._content_hash_index:
                existing_id = self._content_hash_index[content_hash]
                existing_pattern = self._patterns[existing_id]
                existing_pattern.access_count += 1
                existing_pattern.last_accessed = datetime.now().isoformat()
                
                if self.config.auto_merge_exact:
                    self._stats['exact_duplicates_merged'] += 1
                    return existing_pattern, False, existing_pattern.cluster_id
            
            # Check for near-duplicate using Hadamard similarity
            similar_pattern_id = None
            similar_cluster_id = None
            
            if self.config.separate_modalities:
                # Only search within same modality
                modality = metadata.get('modality', 'joint')
                search_patterns = [
                    (pid, p) for pid, p in self._patterns.items()
                    if p.modality == modality
                ]
            else:
                search_patterns = list(self._patterns.items())
            
            for pid, pattern in search_patterns:
                # Timestep-aware comparison
                if self.config.track_trajectories:
                    timestep_diff = abs(pattern.timestep - metadata.get('timestep', 0))
                    threshold = self.config.cross_timestep_threshold if timestep_diff > 0 else self.config.similarity_threshold
                else:
                    threshold = self.config.similarity_threshold
                
                # Get vector for comparison (generated on demand)
                stored_vector = self._get_vector(pattern)
                similarity = self._compute_hadamard_similarity(vector, stored_vector)
                self._stats['similarity_checks'] += 1
                
                if similarity >= threshold:
                    similar_pattern_id = pid
                    similar_cluster_id = pattern.cluster_id
                    self._stats['near_duplicates_found'] += 1
                    break
            
            # Create new pattern (seed-based, no vector stored)
            pattern_id = f"ltx_{layer_name}_{content_hash[:12]}"
            pattern = LTXPattern(
                pattern_id=pattern_id,
                pattern_type=metadata.get('pattern_type', 'joint'),
                seed_string=seed_string,  # Store seed, not vector
                hadamard_index=metadata.get('hadamard_index', 0),
                content_hash=content_hash,
                modality=metadata.get('modality', 'joint'),
                layer_name=layer_name,
                generation_mode=metadata.get('generation_mode', 'text_to_audio_video'),
                frame_count=metadata.get('frame_count', 0),
                frame_rate=metadata.get('frame_rate', 24.0),
                resolution=metadata.get('resolution', (512, 512)),
                video_duration=metadata.get('video_duration', 0.0),
                sample_rate=metadata.get('sample_rate', 44100),
                audio_duration=metadata.get('audio_duration', 0.0),
                audio_channels=metadata.get('audio_channels', 2),
                timestep=metadata.get('timestep', 0),
                metadata=metadata
            )
            
            # Handle clustering
            if similar_cluster_id:
                # Add to existing cluster
                pattern.cluster_id = similar_cluster_id
                self._clusters[similar_cluster_id].append(pattern_id)
                self._pattern_to_cluster[pattern_id] = similar_cluster_id
                
                # Add similarity relationship
                if similar_pattern_id and self.config.preserve_relationships:
                    edge = LTXRelationshipEdge(
                        source_id=pattern_id,
                        target_id=similar_pattern_id,
                        relationship_type=LTXRelationshipType.SEMANTIC_SIMILAR,
                        strength=self._compute_hadamard_similarity(vector, self._get_vector(self._patterns[similar_pattern_id]))
                    )
                    self.relationship_graph.add_edge(edge)
                    self._stats['relationships_preserved'] += 1
            else:
                # Create new cluster
                cluster_id = f"cluster_{pattern_id}"
                pattern.cluster_id = cluster_id
                pattern.is_centroid = True
                self._clusters[cluster_id] = [pattern_id]
                self._pattern_to_cluster[pattern_id] = cluster_id
            
            # Store pattern (seed-based)
            self._patterns[pattern_id] = pattern
            self._content_hash_index[content_hash] = pattern_id
            self._stats['patterns_added'] += 1
            
            # Update indexes
            self._by_seed_string[seed_string] = pattern_id
            
            if pattern.generation_mode:
                if pattern.generation_mode not in self._by_generation_mode:
                    self._by_generation_mode[pattern.generation_mode] = set()
                self._by_generation_mode[pattern.generation_mode].add(pattern_id)
            
            if pattern.layer_name:
                if pattern.layer_name not in self._by_layer:
                    self._by_layer[pattern.layer_name] = set()
                self._by_layer[pattern.layer_name].add(pattern_id)
            
            # Add to relationship graph
            self.relationship_graph.add_pattern(pattern)
            
            return pattern, True, pattern.cluster_id
    
    def deduplicate_from_seed(
        self,
        seed_string: str,
        layer_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[LTXPattern, bool, Optional[str]]:
        """
        Deduplicate a pattern using only a seed string (vector generated on demand).
        
        This is the preferred method for seed-based architecture.
        
        Args:
            seed_string: Seed string for the pattern
            layer_name: Name of the source layer
            metadata: Optional metadata
            
        Returns:
            Tuple of (pattern, is_new, cluster_id)
        """
        # Generate vector on demand for similarity comparison
        vector = seed_to_hypervector(seed_string, self.uint64_count)
        return self.deduplicate(vector, layer_name, seed_string, metadata)
    
    def add_trajectory_relationship(
        self,
        source_id: str,
        target_id: str,
        sequence_position: int,
        relationship_type: LTXRelationshipType = LTXRelationshipType.DENOISES_TO,
        strength: float = 1.0
    ):
        """
        Add a trajectory relationship (denoising, frame sequence, etc.).
        
        Uses circular temporal encoding for sequence tracking instead of diffusion timesteps.
        
        Args:
            source_id: Source pattern ID
            target_id: Target pattern ID
            sequence_position: Position in circular temporal encoding
            relationship_type: Type of trajectory relationship
            strength: Relationship strength
        """
        edge = LTXRelationshipEdge(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            sequence_position=sequence_position
        )
        self.relationship_graph.add_edge(edge)
        self._stats['relationships_preserved'] += 1
    
    def add_audio_video_sync(
        self,
        video_pattern_id: str,
        audio_pattern_id: str,
        strength: float = 1.0
    ):
        """
        Add audio-video synchronization relationship.
        
        Args:
            video_pattern_id: Video pattern ID
            audio_pattern_id: Audio pattern ID
            strength: Sync strength
        """
        # Add bidirectional binding
        edge1 = LTXRelationshipEdge(
            source_id=video_pattern_id,
            target_id=audio_pattern_id,
            relationship_type=LTXRelationshipType.AUDIO_SYNC,
            strength=strength
        )
        edge2 = LTXRelationshipEdge(
            source_id=audio_pattern_id,
            target_id=video_pattern_id,
            relationship_type=LTXRelationshipType.VIDEO_SYNC,
            strength=strength
        )
        self.relationship_graph.add_edge(edge1)
        self.relationship_graph.add_edge(edge2)
        self._stats['relationships_preserved'] += 2
    
    def add_cross_modal_binding(
        self,
        pattern_id1: str,
        pattern_id2: str,
        binding_type: LTXRelationshipType,
        strength: float = 1.0
    ):
        """
        Add cross-modal binding between patterns.
        
        Args:
            pattern_id1: First pattern ID
            pattern_id2: Second pattern ID
            binding_type: Type of cross-modal binding
            strength: Binding strength
        """
        # Add bidirectional binding
        edge1 = LTXRelationshipEdge(
            source_id=pattern_id1,
            target_id=pattern_id2,
            relationship_type=binding_type,
            strength=strength
        )
        edge2 = LTXRelationshipEdge(
            source_id=pattern_id2,
            target_id=pattern_id1,
            relationship_type=binding_type,
            strength=strength
        )
        self.relationship_graph.add_edge(edge1)
        self.relationship_graph.add_edge(edge2)
        self._stats['relationships_preserved'] += 2
    
    def get_all_pattern_ids(self) -> List[str]:
        """Get all pattern IDs."""
        with self._lock:
            return list(self._patterns.keys())
    
    def get_pattern(self, pattern_id: str) -> Optional[LTXPattern]:
        """Get a pattern by ID."""
        with self._lock:
            return self._patterns.get(pattern_id)
    
    def get_pattern_vector(self, pattern_id: str) -> Optional[np.ndarray]:
        """
        Get vector for a pattern by ID (generated on demand from seed).
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Generated hypervector or None if pattern not found
        """
        with self._lock:
            pattern = self._patterns.get(pattern_id)
            if pattern is None:
                return None
            return self._get_vector(pattern)
    
    def get_cluster_patterns(self, cluster_id: str) -> List[LTXPattern]:
        """Get all patterns in a cluster."""
        pattern_ids = self._clusters.get(cluster_id, [])
        return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]
    
    def get_denoising_trajectory(self, pattern_id: str, target_timestep: int = 0) -> List[LTXPattern]:
        """
        Get the denoising trajectory for a pattern.
        
        Args:
            pattern_id: Starting pattern ID
            target_timestep: Target timestep
            
        Returns:
            List of patterns in the trajectory
        """
        trajectory_ids = self.relationship_graph.get_denoising_trajectory(pattern_id, target_timestep)
        return [self._patterns[pid] for pid in trajectory_ids if pid in self._patterns]
    
    def get_audio_video_sync_patterns(self, pattern_id: str) -> Dict[str, List[LTXPattern]]:
        """
        Get synchronized audio/video patterns.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Dictionary with 'audio' and 'video' pattern lists
        """
        sync_ids = self.relationship_graph.get_audio_video_sync_patterns(pattern_id)
        return {
            'audio': [self._patterns[pid] for pid in sync_ids['audio'] if pid in self._patterns],
            'video': [self._patterns[pid] for pid in sync_ids['video'] if pid in self._patterns]
        }
    
    def get_cross_modal_patterns_for_pattern(self, pattern_id: str) -> Dict[str, List[LTXPattern]]:
        """
        Get cross-modal patterns for a given pattern.
        
        Returns:
            Dictionary with 'audio', 'video', 'joint' pattern lists
        """
        cross_modal_ids = self.relationship_graph.get_cross_modal_patterns(pattern_id)
        return {
            modality: [self._patterns[pid] for pid in ids if pid in self._patterns]
            for modality, ids in cross_modal_ids.items()
        }
    
    def find_similar(
        self,
        query_vector: np.ndarray,
        threshold: float = 0.8,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar patterns to the given vector.
        
        Args:
            query_vector: Query HDC vector
            threshold: Minimum similarity threshold
            limit: Maximum number of results
            
        Returns:
            List of dictionaries with pattern_id and similarity
        """
        with self._lock:
            results = []
            
            for pattern_id, pattern in self._patterns.items():
                stored_vector = self._get_vector(pattern)
                similarity = self._compute_similarity(query_vector, stored_vector)
                
                if similarity >= threshold:
                    results.append({
                        'pattern_id': pattern_id,
                        'similarity': similarity,
                        'pattern_type': pattern.pattern_type,
                        'modality': pattern.modality,
                        'generation_mode': pattern.generation_mode,
                        'timestep': pattern.timestep
                    })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
    
    def find_similar_by_seed(
        self,
        seed_string: str,
        threshold: float = 0.8,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar patterns using a seed string (vector generated on demand).
        
        Args:
            seed_string: Seed string to generate query vector
            threshold: Minimum similarity threshold
            limit: Maximum number of results
            
        Returns:
            List of dictionaries with pattern_id and similarity
        """
        query_vector = seed_to_hypervector(seed_string, self.uint64_count)
        return self.find_similar(query_vector, threshold, limit)
    
    def find_similar_patterns(
        self,
        query_vector: np.ndarray,
        pattern_type: Optional[str] = None,
        modality: Optional[str] = None,
        limit: int = 10
    ) -> List[Tuple[LTXPattern, float]]:
        """
        Find patterns similar to the given vector.
        
        Args:
            query_vector: Query HDC vector
            pattern_type: Optional pattern type filter
            modality: Optional modality filter
            limit: Maximum number of results
            
        Returns:
            List of (pattern, similarity) tuples
        """
        with self._lock:
            results = []
            
            for pattern_id, pattern in self._patterns.items():
                # Apply filters
                if pattern_type and pattern.pattern_type != pattern_type:
                    continue
                if modality and pattern.modality != modality:
                    continue
                
                # Compute similarity
                stored_vector = self._get_vector(pattern)
                similarity = self._compute_similarity(query_vector, stored_vector)
                results.append((pattern, similarity))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:limit]
    
    def find_video_sequence_patterns(
        self,
        frame_sequence: np.ndarray,
        modality: Optional[str] = None
    ) -> List[Tuple[LTXPattern, float]]:
        """
        Find patterns with similar video sequences.
        
        Args:
            frame_sequence: Video frame sequence to match
            modality: Optional modality filter
            
        Returns:
            List of (pattern, similarity) tuples
        """
        with self._lock:
            results = []
            
            for pattern in self._patterns.values():
                if pattern.modality not in ['video', 'joint']:
                    continue
                
                if modality and pattern.modality != modality:
                    continue
                
                # Check frame count similarity
                seq_frames = len(frame_sequence) if len(frame_sequence.shape) > 1 else 1
                if abs(pattern.frame_count - seq_frames) > self.config.frame_similarity_tolerance * seq_frames:
                    continue
                
                # Compute pattern similarity
                stored_vector = self._get_vector(pattern)
                video_hdc = self._encode_video_sequence(frame_sequence)
                similarity = self._compute_similarity(video_hdc, stored_vector)
                results.append((pattern, similarity))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:10]
    
    def _encode_video_sequence(self, frame_sequence: np.ndarray) -> np.ndarray:
        """Encode a video frame sequence as an HDC vector."""
        # Simple encoding: use frame statistics
        if len(frame_sequence.shape) == 1:
            frame_sequence = frame_sequence.reshape(1, -1)
        
        # Compute statistics
        mean_frame = np.mean(frame_sequence, axis=0)
        std_frame = np.std(frame_sequence, axis=0)
        
        # Create feature vector
        features = np.concatenate([mean_frame, std_frame])
        
        # Pad or truncate to HDC dimension
        if len(features) < self.hdc_dim:
            features = np.pad(features, (0, self.hdc_dim - len(features)))
        else:
            features = features[:self.hdc_dim]
        
        # Ternarize
        hdc_vector = np.sign(features).astype(np.int8)
        hdc_vector[hdc_vector == 0] = 1  # Avoid zeros
        
        return hdc_vector
    
    def get_patterns_by_generation_mode(self, generation_mode: str) -> List[LTXPattern]:
        """Get all patterns for a specific generation mode."""
        with self._lock:
            pattern_ids = self._by_generation_mode.get(generation_mode, set())
            return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]
    
    def get_cross_modal_patterns(
        self,
        modality1: str,
        modality2: str
    ) -> List[Tuple[LTXPattern, LTXPattern]]:
        """Get pairs of patterns from different modalities that are related."""
        with self._lock:
            pairs = []
            
            # Get patterns from each modality
            patterns1 = [p for p in self._patterns.values() if p.modality == modality1]
            patterns2 = [p for p in self._patterns.values() if p.modality == modality2]
            
            # Find related pairs
            for p1 in patterns1:
                for related_id, rel_type in self.relationship_graph.get_related_patterns(p1.pattern_id):
                    p2 = self._patterns.get(related_id)
                    if p2 and p2.modality == modality2:
                        pairs.append((p1, p2))
            
            return pairs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplicator statistics."""
        with self._lock:
            stats: Dict[str, Any] = dict(self._stats)
            stats['total_patterns'] = len(self._patterns)
            stats['unique_seeds'] = len(self._by_seed_string)
            stats['unique_generation_modes'] = len(self._by_generation_mode)
            stats['unique_layers'] = len(self._by_layer)
            stats['total_clusters'] = len(self._clusters)
            stats['cache_size'] = len(self._vector_cache)
            stats['graph_stats'] = self.relationship_graph.get_statistics()
            return stats
    
    # =========================================================================
    # UNIFIED CROSS-MODEL INTEGRATION METHODS
    # =========================================================================
    
    def export_to_unified_hub(self) -> Dict[str, Any]:
        """
        Export patterns for integration with the unified cross-model deduplication hub.
        
        This enables cross-modal relationships between LTX audio-video patterns
        and patterns from other models (MOSS-TTS, Qwen, Ponder V3, etc.).
        
        Returns:
            Dictionary with pattern data compatible with UnifiedDeduplicationHub
        """
        with self._lock:
            patterns_data = {}
            
            for pattern_id, pattern in self._patterns.items():
                patterns_data[pattern_id] = {
                    'pattern_id': pattern_id,
                    'seed_string': pattern.seed_string,
                    'content_hash': pattern.content_hash,
                    'hadamard_index': pattern.hadamard_index,
                    'model_source': 'ltx',
                    'layer_name': pattern.layer_name,
                    'pattern_type': pattern.pattern_type,
                    'modality': pattern.modality,
                    'generation_mode': pattern.generation_mode,
                    'frame_count': pattern.frame_count,
                    'frame_rate': pattern.frame_rate,
                    'video_duration': pattern.video_duration,
                    'audio_duration': pattern.audio_duration,
                    'timestep': pattern.timestep,
                    'cluster_id': pattern.cluster_id,
                    'is_centroid': pattern.is_centroid,
                    'metadata': pattern.metadata
                }
            
            # Export relationships with circular temporal encoding
            relationships_data = []
            for edges in self.relationship_graph._edges.values():
                for edge in edges:
                    relationships_data.append({
                        'source_id': edge.source_id,
                        'target_id': edge.target_id,
                        'relationship_type': edge.relationship_type.value,
                        'strength': edge.strength,
                        'sequence_position': edge.sequence_position,
                        'metadata': edge.metadata
                    })
            
            return {
                'model_source': 'ltx',
                'patterns': patterns_data,
                'relationships': relationships_data,
                'clusters': dict(self._clusters),
                'statistics': self.get_statistics()
            }
    
    def import_from_unified_hub(self, unified_data: Dict[str, Any]):
        """
        Import patterns from the unified cross-model deduplication hub.
        
        This enables loading patterns from other models (MOSS-TTS, Qwen, Ponder V3, etc.)
        for cross-modal knowledge transfer to LTX.
        
        Args:
            unified_data: Dictionary with pattern data from UnifiedDeduplicationHub
        """
        with self._lock:
            # Import patterns
            for pattern_id, pdata in unified_data.get('patterns', {}).items():
                # Skip if already exists
                if pattern_id in self._patterns:
                    continue
                
                # Create pattern from imported data
                pattern = LTXPattern(
                    pattern_id=pattern_id,
                    pattern_type=pdata.get('pattern_type', 'joint'),
                    seed_string=pdata.get('seed_string', ''),
                    hadamard_index=pdata.get('hadamard_index', 0),
                    content_hash=pdata.get('content_hash', ''),
                    modality=pdata.get('modality', 'joint'),
                    layer_name=pdata.get('layer_name', ''),
                    generation_mode=pdata.get('generation_mode', 'text_to_audio_video'),
                    timestep=pdata.get('timestep', 0),
                    cluster_id=pdata.get('cluster_id'),
                    is_centroid=pdata.get('is_centroid', False),
                    metadata=pdata.get('metadata', {})
                )
                
                # Store pattern
                self._patterns[pattern_id] = pattern
                
                # Update indexes
                if pattern.seed_string:
                    self._by_seed_string[pattern.seed_string] = pattern_id
                if pattern.content_hash:
                    self._content_hash_index[pattern.content_hash] = pattern_id
                if pattern.generation_mode:
                    if pattern.generation_mode not in self._by_generation_mode:
                        self._by_generation_mode[pattern.generation_mode] = set()
                    self._by_generation_mode[pattern.generation_mode].add(pattern_id)
                
                # Add to relationship graph
                self.relationship_graph.add_pattern(pattern)
            
            # Import relationships
            for rel_data in unified_data.get('relationships', []):
                try:
                    rel_type = LTXRelationshipType(rel_data['relationship_type'])
                except ValueError:
                    # Skip unknown relationship types
                    continue
                
                edge = LTXRelationshipEdge(
                    source_id=rel_data['source_id'],
                    target_id=rel_data['target_id'],
                    relationship_type=rel_type,
                    strength=rel_data.get('strength', 1.0),
                    sequence_position=rel_data.get('sequence_position', rel_data.get('timestep_from', 0)),  # Backward compat
                    metadata=rel_data.get('metadata', {})
                )
                self.relationship_graph.add_edge(edge)
            
            # Import clusters
            for cluster_id, pattern_ids in unified_data.get('clusters', {}).items():
                if cluster_id not in self._clusters:
                    self._clusters[cluster_id] = []
                for pid in pattern_ids:
                    if pid not in self._clusters[cluster_id]:
                        self._clusters[cluster_id].append(pid)
                    self._pattern_to_cluster[pid] = cluster_id
    
    def find_cross_model_candidates(
        self,
        pattern: LTXPattern,
        unified_hub: Any,
        threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Find patterns in the unified hub that could be related to an LTX pattern.
        
        This enables discovering cross-modal relationships, e.g.:
        - Audio patterns from MOSS-TTS that match audio tracks
        - Vision patterns from Qwen that match video frames
        - Text patterns from GLM-5 that match prompts
        
        Args:
            pattern: LTX pattern to find cross-model matches for
            unified_hub: UnifiedDeduplicationHub instance
            threshold: Similarity threshold for matching
            
        Returns:
            List of candidate patterns from other models
        """
        # Get vector for pattern
        vector = self._get_vector(pattern)
        
        # Query unified hub for similar patterns
        candidates = []
        
        # Use unified hub's find_similar_patterns if available
        if hasattr(unified_hub, 'find_similar_patterns'):
            similar = unified_hub.find_similar_patterns(
                vector=vector,
                top_k=20,
                model_filter=None  # Search all models
            )
            
            for candidate_pattern, similarity in similar:
                if similarity >= threshold:
                    # Skip patterns from same model
                    if 'ltx' in candidate_pattern.model_sources:
                        continue
                    
                    candidates.append({
                        'pattern_id': candidate_pattern.pattern_id,
                        'model_sources': candidate_pattern.model_sources,
                        'similarity': similarity,
                        'pattern_type': candidate_pattern.pattern_types,
                        'modality': getattr(candidate_pattern, 'modality', 'unknown'),
                        'suggested_relationship': self._suggest_cross_model_relationship(
                            pattern, candidate_pattern
                        )
                    })
        
        return candidates
    
    def _suggest_cross_model_relationship(
        self,
        ltx_pattern: LTXPattern,
        candidate_pattern: Any
    ) -> str:
        """
        Suggest a relationship type based on pattern characteristics.
        
        Args:
            ltx_pattern: LTX pattern
            candidate_pattern: Candidate pattern from another model
            
        Returns:
            Suggested relationship type string
        """
        # Get candidate modality
        candidate_modality = getattr(candidate_pattern, 'modality', 'unknown')
        if hasattr(candidate_pattern, 'pattern_types'):
            candidate_types = candidate_pattern.pattern_types
        else:
            candidate_types = {}
        
        # Suggest based on modality combinations
        if ltx_pattern.modality == 'video' and candidate_modality in ['image', 'vision']:
            return 'video_frame_bind'
        elif ltx_pattern.modality == 'audio' and candidate_modality == 'audio':
            return 'audio_sync'
        elif ltx_pattern.modality == 'joint' and candidate_modality in ['video', 'image']:
            return 'multimodal_vision_bind'
        elif 'moss_tts' in candidate_pattern.model_sources and ltx_pattern.modality in ['audio', 'joint']:
            return 'audio_track_bind'
        elif 'qwen' in candidate_pattern.model_sources and ltx_pattern.modality in ['video', 'joint']:
            return 'vision_multimodal_bind'
        elif 'ponder_v3' in candidate_pattern.model_sources and ltx_pattern.modality == 'video':
            return 'robotics_vision_bind'
        elif 'glm5' in candidate_pattern.model_sources:
            return 'text_condition_bind'
        else:
            return 'semantic_similar'
    
    def create_cross_model_binding(
        self,
        ltx_pattern_id: str,
        external_pattern_id: str,
        relationship_type: str,
        unified_hub: Any
    ) -> bool:
        """
        Create a binding between an LTX pattern and a pattern from another model.
        
        Args:
            ltx_pattern_id: LTX pattern ID
            external_pattern_id: Pattern ID from another model
            relationship_type: Type of relationship
            unified_hub: UnifiedDeduplicationHub to register the relationship
            
        Returns:
            True if binding was created successfully
        """
        with self._lock:
            if ltx_pattern_id not in self._patterns:
                return False
            
            try:
                rel_type = LTXRelationshipType(relationship_type)
            except ValueError:
                rel_type = LTXRelationshipType.SEMANTIC_SIMILAR
            
            # Add local relationship edge
            edge = LTXRelationshipEdge(
                source_id=ltx_pattern_id,
                target_id=external_pattern_id,
                relationship_type=rel_type,
                metadata={'cross_model': True}
            )
            self.relationship_graph.add_edge(edge)
            
            # Register with unified hub if available
            if hasattr(unified_hub, 'add_cross_model_relationship'):
                from ..unified_cross_model_deduplication import CrossModelRelationshipType
                
                # Map relationship type
                unified_rel_type = CrossModelRelationshipType.SEMANTIC_SIMILAR
                if relationship_type in ['audio_sync', 'audio_track_bind']:
                    unified_rel_type = CrossModelRelationshipType.AUDIO_VIDEO_SYNC
                elif relationship_type in ['video_frame_bind', 'vision_multimodal_bind', 'robotics_vision_bind']:
                    unified_rel_type = CrossModelRelationshipType.MULTIMODAL_FUSION
                elif relationship_type in ['text_condition_bind', 'text_image_bind']:
                    unified_rel_type = CrossModelRelationshipType.TEXT_IMAGE_BIND
                
                unified_hub.add_cross_model_relationship(
                    source_pattern_id=ltx_pattern_id,
                    target_pattern_id=external_pattern_id,
                    relationship_type=unified_rel_type
                )
            
            self._stats['relationships_preserved'] += 1
            return True
    
    def get_unified_checkpoint(self) -> Dict[str, Any]:
        """
        Get a checkpoint compatible with the unified cross-model system.
        
        Returns:
            Checkpoint dictionary that can be loaded by other models
        """
        return {
            'model_source': 'ltx',
            'checkpoint_type': 'unified_deduplication',
            'created_at': datetime.now().isoformat(),
            'config': {
                'hdc_dim': self.hdc_dim,
                'uint64_count': self.uint64_count,
                'similarity_threshold': self.config.similarity_threshold
            },
            'statistics': self.get_statistics(),
            'patterns': {
                pid: p.to_dict() for pid, p in self._patterns.items()
            },
            'relationships': self.relationship_graph.to_dict(),
            'clusters': dict(self._clusters)
        }
    
    def load_unified_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Load a checkpoint from the unified cross-model system.
        
        Args:
            checkpoint: Checkpoint dictionary from another model or unified hub
        """
        with self._lock:
            # Load config
            config = checkpoint.get('config', {})
            if 'hdc_dim' in config:
                self.hdc_dim = config['hdc_dim']
                self.uint64_count = self.hdc_dim // 64
            
            # Load patterns
            for pid, pdata in checkpoint.get('patterns', {}).items():
                pattern = LTXPattern.from_dict(pdata)
                self._patterns[pid] = pattern
                
                if pattern.seed_string:
                    self._by_seed_string[pattern.seed_string] = pid
                if pattern.content_hash:
                    self._content_hash_index[pattern.content_hash] = pid
                if pattern.generation_mode:
                    if pattern.generation_mode not in self._by_generation_mode:
                        self._by_generation_mode[pattern.generation_mode] = set()
                    self._by_generation_mode[pattern.generation_mode].add(pid)
                
                self.relationship_graph.add_pattern(pattern)
            
            # Load relationships
            rel_data = checkpoint.get('relationships', {})
            for edge_data in rel_data.get('edges', []):
                try:
                    edge = LTXRelationshipEdge.from_dict(edge_data)
                    self.relationship_graph.add_edge(edge)
                except Exception:
                    pass
            
            # Load clusters
            for cid, pids in checkpoint.get('clusters', {}).items():
                self._clusters[cid] = pids
                for pid in pids:
                    self._pattern_to_cluster[pid] = cid
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics (alias)."""
        return self.get_statistics()
    
    def clear_cache(self):
        """Clear the vector cache."""
        with self._lock:
            self._vector_cache.clear()
    
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable the vector cache."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()
    
    def save(self, path: str):
        """
        Save deduplicator state to disk.
        
        Only saves seed strings and metadata - no vectors stored.
        Vectors are regenerated on demand when loaded.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Save patterns metadata (seed-based, no vectors)
            patterns_data = {
                pid: p.to_dict() for pid, p in self._patterns.items()
            }
            with open(save_path / "patterns.json", 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save relationship graph
            with open(save_path / "relationships.json", 'w') as f:
                json.dump(self.relationship_graph.to_dict(), f, indent=2)
            
            # Save clusters
            with open(save_path / "clusters.json", 'w') as f:
                json.dump(dict(self._clusters), f, indent=2)
            
            # Save statistics
            with open(save_path / "stats.json", 'w') as f:
                json.dump(self._stats, f, indent=2)
            
            # Save config
            config_data = {
                'hdc_dim': self.hdc_dim,
                'uint64_count': self.uint64_count,
                'cache_enabled': self._cache_enabled,
                'max_cache_size': self._max_cache_size
            }
            with open(save_path / "config.json", 'w') as f:
                json.dump(config_data, f, indent=2)
    
    def load(self, path: str):
        """
        Load deduplicator state from disk.
        
        Loads seed strings and metadata - vectors are generated on demand.
        """
        load_path = Path(path)
        
        if not load_path.exists():
            return
        
        with self._lock:
            # Load config first
            config_file = load_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                self.hdc_dim = config_data.get('hdc_dim', self.hdc_dim)
                self.uint64_count = config_data.get('uint64_count', self.uint64_count)
                self._cache_enabled = config_data.get('cache_enabled', True)
                self._max_cache_size = config_data.get('max_cache_size', 10000)
            
            # Load patterns metadata
            patterns_file = load_path / "patterns.json"
            if not patterns_file.exists():
                return
                
            with open(patterns_file, 'r') as f:
                patterns_data = json.load(f)
            
            # Reconstruct patterns (vectors generated on demand)
            for pid, pdata in patterns_data.items():
                pattern = LTXPattern.from_dict(pdata)
                self._patterns[pid] = pattern
                
                # Update content hash index
                if pattern.content_hash:
                    self._content_hash_index[pattern.content_hash] = pid
                
                # Update indexes
                self._by_seed_string[pattern.seed_string] = pid
                if pattern.generation_mode:
                    if pattern.generation_mode not in self._by_generation_mode:
                        self._by_generation_mode[pattern.generation_mode] = set()
                    self._by_generation_mode[pattern.generation_mode].add(pid)
                if pattern.layer_name:
                    if pattern.layer_name not in self._by_layer:
                        self._by_layer[pattern.layer_name] = set()
                    self._by_layer[pattern.layer_name].add(pid)
                
                self.relationship_graph.add_pattern(pattern)
            
            # Load relationship graph
            relationships_file = load_path / "relationships.json"
            if relationships_file.exists():
                with open(relationships_file, 'r') as f:
                    graph_data = json.load(f)
                # Reconstruct edges
                for edge_data in graph_data.get('edges', []):
                    edge = LTXRelationshipEdge.from_dict(edge_data)
                    self.relationship_graph.add_edge(edge)
            
            # Load clusters
            clusters_file = load_path / "clusters.json"
            if clusters_file.exists():
                with open(clusters_file, 'r') as f:
                    clusters_data = json.load(f)
                self._clusters = defaultdict(list, clusters_data)
                self._pattern_to_cluster = {
                    pid: cid for cid, pids in self._clusters.items() for pid in pids
                }
            
            # Load stats
            stats_file = load_path / "stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self._stats = json.load(f)


# =============================================================================
# Factory Functions
# =============================================================================

def create_ltx_deduplicator(
    similarity_threshold: float = 0.95,
    hdc_dim: int = DEFAULT_HDC_DIM,
    **kwargs
) -> LTXPatternDeduplicator:
    """
    Factory function to create an LTX pattern deduplicator.
    
    Args:
        similarity_threshold: Threshold for pattern similarity
        hdc_dim: HDC dimension
        **kwargs: Additional configuration options
    
    Returns:
        Configured LTXPatternDeduplicator instance
    """
    config = LTXDeduplicationConfig(
        similarity_threshold=similarity_threshold,
        **kwargs
    )
    
    return LTXPatternDeduplicator(config=config, hdc_dim=hdc_dim)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Functions
    'seed_to_hypervector',
    
    # Enums
    'LTXRelationshipType',
    
    # Config
    'LTXDeduplicationConfig',
    
    # Pattern
    'LTXPattern',
    
    # Edge
    'LTXRelationshipEdge',
    
    # Classes
    'LTXRelationshipGraph',
    'LTXPatternDeduplicator',
    
    # Factory
    'create_ltx_deduplicator',
    
    # Constants
    'HAS_BLAKE3'
]
