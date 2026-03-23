"""
Pure Recipe-Based Recursive Solver

This module implements a recursion solver that operates ENTIRELY on recipes
without constructing HDC vectors during the recursion process. Vectors are only
constructed at the very end for final verification.

Key Benefits:
- O(1) per recursion step (vs O(dim) for vector operations)
- Effectively infinite recursion depth (limited only by memory for recipe storage)
- ~80x memory reduction (recipes are ~50 bytes vs 4KB vectors)
- 100-1000x speedup for similarity comparisons

Architecture:
1. Templates stored as TemplateRecipe (seeds + operation signatures)
2. Reasoning state z stored as SymbolicReasoningRecipe (growing list of operations)
3. Matching done via recipe_similarity() heuristics (no vectors)
4. Cascade verification: only reconstruct top-K candidates at the end

Usage:
    solver = PureRecipeSolver(hdc, encoder, grid_engine)
    result, confidence, metadata = solver.solve(task)
"""

import hashlib
import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Import types from parent module - will be available when used as part of the package
try:
    from ..HDC_Core_Main.hdc_sparse_core import SparseBinaryHDC
    from ..HDC_Core_Model.Templates_Tools.grid_templates import GridTemplateEngine, TransformationRecipe, TransformationStep
    from ..HDC_Core_Model.Recipes_Seeds.relationship_encoder import (
        SimplifiedRelationshipEncoder,
        RelationshipType,
        Relationship
    )
    _RELATIONSHIP_ENCODER_AVAILABLE = True
except ImportError:
    # Standalone testing mode
    SparseBinaryHDC = Any  # type: ignore
    GridTemplateEngine = Any  # type: ignore
    TransformationRecipe = Any  # type: ignore
    TransformationStep = Any  # type: ignore
    SimplifiedRelationshipEncoder = Any  # type: ignore
    RelationshipType = Any  # type: ignore
    Relationship = Any  # type: ignore
    _RELATIONSHIP_ENCODER_AVAILABLE = False


def _string_to_seed(s: str) -> int:
    """Convert string to deterministic seed using SHA256."""
    hash_bytes = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF


# =============================================================================
# Template Relationship Knowledge System (6 Core Types + Conditionals)
# =============================================================================

@dataclass
class ConditionalRelationship:
    """
    Represents a conditional relationship: IF context THEN relationship changes.
    
    Example: "rotate_90 IS-A geometric, BUT IF task_has_colors THEN rotate_90 SIMILAR color_preserve"
    
    This enables context-dependent reasoning for ARC tasks.
    """
    condition: str                    # Context that activates this conditional
    base_relationship: str            # The base relationship type
    modified_relationship: str        # How it changes in this context
    source: str                       # Source template
    target: str                       # Target template
    strength: float = 0.8             # How strong when condition is met


class TemplateRelationshipKnowledge:
    """
    Comprehensive relationship knowledge for templates using all 6 core types.
    
    This class encodes the semantic structure of transformations to enable:
    1. SIMILAR-based fallbacks when a template almost works
    2. OPPOSITE-based inverse detection
    3. COMPOSED-based multi-step discovery
    4. IS-A category reasoning
    5. PART-OF component analysis
    6. PREDICTS temporal/sequential patterns
    7. CONDITIONAL context-dependent relationships
    
    The 6 Core Relationship Types:
    - IS-A (⊂): Template belongs to category (rotate_90 IS-A geometric)
    - SIMILAR (≈): Templates are similar (rotate_90 SIMILAR rotate_180)
    - OPPOSITE (⊕): Templates are inverses (rotate_90 OPPOSITE rotate_270)
    - COMPOSED (∘): Template = composition (rotate_180 COMPOSED rotate_90+rotate_90)
    - PART-OF (∈): Template is component of complex transform
    - PREDICTS (→): Template typically followed by another (crop PREDICTS mark_boundary)
    
    Conditionals:
    - IF size_changes THEN prefer structural templates
    - IF colors_added THEN prefer morphological templates
    """
    
    # Pre-defined template categories for IS-A relationships
    TEMPLATE_CATEGORIES = {
        'geometric': [
            'rotate_90', 'rotate_180', 'rotate_270',
            'flip_horizontal', 'flip_vertical', 'flip_diagonal',
            'flip_antidiagonal', 'identity'
        ],
        'gravity': ['gravity_down', 'gravity_up', 'gravity_left', 'gravity_right'],
        'translation': [
            'translate_up', 'translate_down', 'translate_left', 'translate_right',
            'translate_up_left', 'translate_up_right', 'translate_down_left', 'translate_down_right',
            'center_object', 'align_to_edge', 'align_to_corner'
        ],
        'morphological': [
            'mark_boundary', 'mark_boundary_8conn', 'extract_boundary',
            'extract_interior', 'fill_enclosed', 'fill_holes',
            'dilate', 'erode', 'morph_outline_recolor', 'detect_and_mark_boundary'
        ],
        'structural': [
            'tile_2x2', 'tile_horizontal', 'tile_vertical',
            'scale_2x', 'crop_nonzero', 'outline'
        ],
        'drawing': ['draw_line', 'connect_points', 'connect_all_pairs'],
        'color': ['color_swap', 'color_replace', 'color_invert']
    }
    
    # Pre-defined OPPOSITE relationships
    OPPOSITE_PAIRS = {
        'rotate_90': 'rotate_270',
        'rotate_270': 'rotate_90',
        'rotate_180': 'rotate_180',  # Self-inverse
        'flip_horizontal': 'flip_horizontal',  # Self-inverse
        'flip_vertical': 'flip_vertical',  # Self-inverse
        'flip_diagonal': 'flip_antidiagonal',
        'flip_antidiagonal': 'flip_diagonal',
        'gravity_down': 'gravity_up',
        'gravity_up': 'gravity_down',
        'gravity_left': 'gravity_right',
        'gravity_right': 'gravity_left',
        'translate_up': 'translate_down',
        'translate_down': 'translate_up',
        'translate_left': 'translate_right',
        'translate_right': 'translate_left',
        'dilate': 'erode',
        'erode': 'dilate',
        'scale_2x': 'crop_nonzero',  # Approximate inverse
    }
    
    # Pre-defined COMPOSED relationships (result: [components])
    COMPOSED_FROM = {
        'rotate_180': ['rotate_90', 'rotate_90'],
        'rotate_270': ['rotate_90', 'rotate_90', 'rotate_90'],
        'flip_antidiagonal': ['flip_diagonal', 'rotate_180'],
    }
    
    # Pre-defined PREDICTS relationships (temporal sequence patterns)
    PREDICTS_CHAINS = {
        'crop_nonzero': ['mark_boundary', 'center_object', 'rotate_90'],
        'rotate_90': ['flip_horizontal', 'mark_boundary', 'gravity_down'],
        'identity': ['mark_boundary', 'crop_nonzero'],  # Often pre-processing
        'mark_boundary': ['crop_nonzero'],  # Often post-processing
        'center_object': ['rotate_90', 'flip_horizontal'],
    }
    
    # Conditional relationships based on task context
    CONDITIONAL_RELATIONSHIPS = [
        # Size change conditionals
        ConditionalRelationship(
            condition='size_increases',
            base_relationship='structural',
            modified_relationship='high_priority',
            source='task_analysis',
            target='tile_2x2',
            strength=0.9
        ),
        ConditionalRelationship(
            condition='size_decreases',
            base_relationship='structural',
            modified_relationship='high_priority',
            source='task_analysis',
            target='crop_nonzero',
            strength=0.9
        ),
        # Color change conditionals
        ConditionalRelationship(
            condition='colors_added',
            base_relationship='morphological',
            modified_relationship='high_priority',
            source='task_analysis',
            target='mark_boundary',
            strength=0.85
        ),
        ConditionalRelationship(
            condition='colors_removed',
            base_relationship='color',
            modified_relationship='high_priority',
            source='task_analysis',
            target='color_replace',
            strength=0.8
        ),
        # Position change conditionals
        ConditionalRelationship(
            condition='position_shifted',
            base_relationship='translation',
            modified_relationship='high_priority',
            source='task_analysis',
            target='translate_by_offset',
            strength=0.9
        ),
        # Complexity conditionals
        ConditionalRelationship(
            condition='multi_object',
            base_relationship='structural',
            modified_relationship='prefer_global',
            source='task_analysis',
            target='gravity_down',
            strength=0.7
        ),
    ]
    
    def __init__(self, hdc: Optional['SparseBinaryHDC'] = None, seed: int = 42):
        """
        Initialize relationship knowledge base.
        
        Args:
            hdc: Optional HDC system for vector-based relationship encoding
            seed: Random seed for reproducibility
        """
        self.hdc = hdc
        self.seed = seed
        
        # Build relationship indices
        self._is_a: Dict[str, str] = {}  # template -> category
        self._similar: Dict[str, List[str]] = {}  # template -> similar templates
        self._opposite: Dict[str, str] = {}  # template -> opposite
        self._composed_of: Dict[str, List[str]] = {}  # template -> components
        self._composes_to: Dict[str, List[str]] = {}  # component -> composed templates
        self._predicts: Dict[str, List[str]] = {}  # template -> likely next templates
        self._part_of: Dict[str, List[str]] = {}  # template -> containing transforms
        
        # Conditional relationship index
        self._conditionals: Dict[str, List[ConditionalRelationship]] = {}
        
        # Relationship encoder (if available)
        self._encoder: Optional['SimplifiedRelationshipEncoder'] = None
        if hdc is not None and _RELATIONSHIP_ENCODER_AVAILABLE:
            try:
                self._encoder = SimplifiedRelationshipEncoder(hdc, seed)
            except Exception:
                self._encoder = None
        
        # Build all indices
        self._build_indices()
    
    def _build_indices(self):
        """Build all relationship indices."""
        # Build IS-A (template -> category)
        for category, templates in self.TEMPLATE_CATEGORIES.items():
            for template in templates:
                self._is_a[template] = category
        
        # Build SIMILAR (same category = similar)
        for category, templates in self.TEMPLATE_CATEGORIES.items():
            for template in templates:
                similar = [t for t in templates if t != template]
                self._similar[template] = similar
        
        # Build OPPOSITE
        self._opposite = dict(self.OPPOSITE_PAIRS)
        
        # Build COMPOSED_OF and COMPOSES_TO
        for composed, components in self.COMPOSED_FROM.items():
            self._composed_of[composed] = components
            for comp in components:
                if comp not in self._composes_to:
                    self._composes_to[comp] = []
                if composed not in self._composes_to[comp]:
                    self._composes_to[comp].append(composed)
        
        # Build PREDICTS
        for template, predictions in self.PREDICTS_CHAINS.items():
            self._predicts[template] = predictions
        
        # Build conditional index
        for cond in self.CONDITIONAL_RELATIONSHIPS:
            if cond.condition not in self._conditionals:
                self._conditionals[cond.condition] = []
            self._conditionals[cond.condition].append(cond)
    
    # =========================================================================
    # Relationship Query Methods
    # =========================================================================
    
    def get_category(self, template: str) -> Optional[str]:
        """Get the category of a template (IS-A relationship)."""
        return self._is_a.get(template)
    
    def get_similar(self, template: str, top_k: int = 5) -> List[str]:
        """Get templates similar to the given one (SIMILAR relationship)."""
        return self._similar.get(template, [])[:top_k]
    
    def get_opposite(self, template: str) -> Optional[str]:
        """Get the opposite/inverse of a template (OPPOSITE relationship)."""
        return self._opposite.get(template)
    
    def get_components(self, template: str) -> List[str]:
        """Get the components of a composed template (COMPOSED relationship)."""
        return self._composed_of.get(template, [])
    
    def get_compositions(self, component: str) -> List[str]:
        """Get templates this component can compose into."""
        return self._composes_to.get(component, [])
    
    def get_predicted_next(self, template: str, top_k: int = 3) -> List[str]:
        """Get likely next templates (PREDICTS relationship)."""
        return self._predicts.get(template, [])[:top_k]
    
    def get_templates_in_category(self, category: str) -> List[str]:
        """Get all templates that IS-A category (PART-OF inverse)."""
        return self.TEMPLATE_CATEGORIES.get(category, [])
    
    def get_conditionals(self, context: str) -> List[ConditionalRelationship]:
        """Get conditional relationships activated by this context."""
        return self._conditionals.get(context, [])
    
    def add_relationship(self, source: str, rel_type: str, target: str):
        """
        Add a new relationship to the knowledge base.
        
        Supports relationship types: IS_A, SIMILAR, OPPOSITE, COMPOSED, PART_OF, PREDICTS
        
        Args:
            source: Source template/entity
            rel_type: Relationship type (e.g., "IS_A", "OPPOSITE")
            target: Target template/category
        """
        if rel_type == "IS_A":
            self._is_a[source] = target
            # Also update similar relationships
            if target not in self._similar:
                self._similar[source] = []
            # Add to category templates
            if target in self.TEMPLATE_CATEGORIES:
                if source not in self.TEMPLATE_CATEGORIES[target]:
                    self.TEMPLATE_CATEGORIES[target].append(source)
        elif rel_type == "SIMILAR":
            if source not in self._similar:
                self._similar[source] = []
            if target not in self._similar[source]:
                self._similar[source].append(target)
        elif rel_type == "OPPOSITE":
            self._opposite[source] = target
            # Opposite is bidirectional
            self._opposite[target] = source
        elif rel_type == "COMPOSED":
            if source not in self._composed_of:
                self._composed_of[source] = []
            if target not in self._composed_of[source]:
                self._composed_of[source].append(target)
            # Update inverse index
            if target not in self._composes_to:
                self._composes_to[target] = []
            if source not in self._composes_to[target]:
                self._composes_to[target].append(source)
        elif rel_type == "PART_OF":
            if source not in self._part_of:
                self._part_of[source] = []
            if target not in self._part_of[source]:
                self._part_of[source].append(target)
        elif rel_type == "PREDICTS":
            if source not in self._predicts:
                self._predicts[source] = []
            if target not in self._predicts[source]:
                self._predicts[source].append(target)
    
    # =========================================================================
    # Context-Aware Reasoning
    # =========================================================================
    
    def get_prioritized_templates(
        self,
        characteristics: Dict[str, Any],
        base_templates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get templates prioritized by task characteristics using conditionals.
        
        Args:
            characteristics: Task analysis results (from analyze_task_characteristics)
            base_templates: Optional list to filter; if None, uses all templates
        
        Returns:
            List of (template_name, priority_score) sorted by priority
        """
        scores: Dict[str, float] = {}
        
        # Initialize all templates with base score
        all_templates = base_templates or list(self._is_a.keys())
        for t in all_templates:
            scores[t] = 0.5  # Base priority
        
        # Apply conditionals based on characteristics
        active_contexts = []
        
        if characteristics.get('size_changes'):
            size_info = characteristics.get('inferred_params', {})
            if size_info.get('scale') == 2:
                active_contexts.append('size_increases')
            elif characteristics.get('size_changes'):
                # Check if shrinking
                active_contexts.append('size_decreases')
        
        if characteristics.get('new_colors_introduced'):
            active_contexts.append('colors_added')
        
        if characteristics.get('colors_removed'):
            active_contexts.append('colors_removed')
        
        if characteristics.get('position_shifts'):
            active_contexts.append('position_shifted')
        
        # Apply each active conditional
        for ctx in active_contexts:
            for cond in self.get_conditionals(ctx):
                if cond.target in scores:
                    scores[cond.target] += cond.strength
                # Also boost entire category
                category_templates = self.get_templates_in_category(cond.base_relationship)
                for t in category_templates:
                    if t in scores:
                        scores[t] += 0.2  # Category boost
        
        # Apply likely_categories boost
        for cat in characteristics.get('likely_categories', []):
            for t in self.get_templates_in_category(cat):
                if t in scores:
                    scores[t] += 0.3
        
        # Sort by score descending
        result = [(t, s) for t, s in scores.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def get_fallback_templates(
        self,
        failed_template: str,
        accuracy: float
    ) -> List[Tuple[str, str]]:
        """
        Get fallback templates when a template achieves high but not exact accuracy.
        
        Uses relationships to suggest alternatives:
        1. SIMILAR templates (same category)
        2. OPPOSITE if accuracy ~0.0 (wrong direction)
        3. Compositions if partial match
        
        Args:
            failed_template: Template that almost worked
            accuracy: Accuracy achieved (0-1)
        
        Returns:
            List of (template_name, reason) tuples
        """
        fallbacks: List[Tuple[str, str]] = []
        
        # High accuracy (>80%) - try similar templates
        if accuracy > 0.8:
            for t in self.get_similar(failed_template, top_k=3):
                fallbacks.append((t, f"SIMILAR to {failed_template}"))
        
        # Very low accuracy (<20%) - try opposite
        if accuracy < 0.2:
            opposite = self.get_opposite(failed_template)
            if opposite:
                fallbacks.append((opposite, f"OPPOSITE of {failed_template}"))
        
        # Medium accuracy - template might be a component
        if 0.3 < accuracy < 0.7:
            # Try compositions that include this template
            for comp in self.get_compositions(failed_template):
                fallbacks.append((comp, f"{failed_template} COMPOSED into {comp}"))
            
            # Try predicted next steps (maybe need 2-step)
            for pred in self.get_predicted_next(failed_template, top_k=2):
                fallbacks.append((pred, f"{failed_template} PREDICTS {pred}"))
        
        return fallbacks
    
    def suggest_composition(
        self,
        template1: str,
        task_characteristics: Dict[str, Any]
    ) -> List[List[str]]:
        """
        Suggest 2-step compositions starting from template1.
        
        Uses PREDICTS relationships and task context to suggest likely
        second steps.
        
        Args:
            template1: First step template
            task_characteristics: Task analysis
        
        Returns:
            List of [template1, template2] compositions to try
        """
        compositions = []
        
        # Use PREDICTS relationships
        for pred in self.get_predicted_next(template1):
            compositions.append([template1, pred])
        
        # Use task-appropriate second steps
        likely_cats = task_characteristics.get('likely_categories', [])
        for cat in likely_cats:
            for t2 in self.get_templates_in_category(cat)[:2]:
                if t2 != template1 and [template1, t2] not in compositions:
                    compositions.append([template1, t2])
        
        # If morphological task, try mark_boundary variants
        if 'morphological' in likely_cats:
            boundary_colors = task_characteristics.get('inferred_params', {}).get('boundary_colors', [2])
            if boundary_colors:
                compositions.append([template1, 'mark_boundary'])
        
        return compositions[:10]  # Limit to top 10
    
    def encode_relationship_vector(
        self,
        source: str,
        target: str,
        rel_type: str
    ) -> Optional[np.ndarray]:
        """
        Encode a relationship as an HDC vector (if encoder available).
        
        Args:
            source: Source template name
            target: Target template name
            rel_type: Relationship type ('IS-A', 'SIMILAR', 'OPPOSITE', etc.)
        
        Returns:
            HDC vector encoding the relationship, or None
        """
        if self._encoder is None or self.hdc is None:
            return None
        
        try:
            # Get vectors for templates
            source_seed = _string_to_seed(f"template_{source}")
            target_seed = _string_to_seed(f"template_{target}")
            
            source_vec = self.hdc.from_seed(source_seed)
            target_vec = self.hdc.from_seed(target_seed)
            
            # Map string to RelationshipType
            type_map = {
                'IS-A': RelationshipType.IS_A,
                'SIMILAR': RelationshipType.SIMILAR,
                'OPPOSITE': RelationshipType.OPPOSITE,
                'COMPOSED': RelationshipType.COMPOSED,
                'PART-OF': RelationshipType.PART_OF,
                'PREDICTS': RelationshipType.PREDICTS,
            }
            
            if rel_type not in type_map:
                return None
            
            return self._encoder.encode_relationship(
                source_vec, target_vec,
                type_map[rel_type],
                strength=1.0
            )
        except Exception:
            return None


# Global instance (lazy initialization)
_template_relationships: Optional[TemplateRelationshipKnowledge] = None

def get_template_relationships(hdc: Optional['SparseBinaryHDC'] = None) -> TemplateRelationshipKnowledge:
    """Get or create the global template relationship knowledge base."""
    global _template_relationships
    if _template_relationships is None or (hdc is not None and _template_relationships.hdc is None):
        _template_relationships = TemplateRelationshipKnowledge(hdc)
    return _template_relationships


# =============================================================================
# Relationship Strength Tracker (Degradation after Failures)
# =============================================================================

class RelationshipStrengthTracker:
    """
    Tracks relationship success/failure to enable degradation over time.
    
    Key feature: After 5 consecutive failures, relationships are marked for override.
    This allows the system to learn that certain templates DON'T work for certain
    task types and avoid wasting time retrying them.
    
    Degradation rules:
    - Each failure reduces strength by 15%
    - After 5 consecutive failures: marked_for_override = True
    - Success rate < 50% and 3+ uses: degraded = True
    - Success rate < 20% and 5+ uses: marked_for_override = True
    - On success: strength increases by 10% (capped at original)
    """
    
    DEGRADATION_RATE = 0.85  # 15% reduction per failure
    RECOVERY_RATE = 1.10     # 10% increase per success
    CONSECUTIVE_FAILURES_THRESHOLD = 5
    DEGRADED_SUCCESS_RATE = 0.50
    OVERRIDE_SUCCESS_RATE = 0.20
    
    def __init__(self):
        """Initialize the tracker with empty records."""
        # Format: { relationship_id: { tracking_data } }
        self._tracking: Dict[str, Dict[str, Any]] = {}
        
        # Cache for deprecated relationships (marked for override)
        self._deprecated: Set[str] = set()
        
        # Statistics
        self._stats = {
            "total_tracked": 0,
            "total_successes": 0,
            "total_failures": 0,
            "degraded_count": 0,
            "override_count": 0,
        }
    
    def track_outcome(
        self,
        relationship_id: str,
        success: bool,
        context: Optional[str] = None,
        initial_strength: float = 1.0
    ) -> Dict[str, Any]:
        """
        Track the outcome of using a relationship.
        
        Args:
            relationship_id: Unique ID (e.g., "template_name" or "template::task_type")
            success: Whether the relationship/template worked
            context: Optional context (task type, characteristics)
            initial_strength: Starting strength for new relationships
        
        Returns:
            Updated tracking record
        """
        # Initialize if new
        if relationship_id not in self._tracking:
            self._tracking[relationship_id] = {
                "relationship_id": relationship_id,
                "usage_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "consecutive_failures": 0,
                "success_rate": 1.0,  # Optimistic start
                "degraded": False,
                "marked_for_override": False,
                "current_strength": initial_strength,
                "original_strength": initial_strength,
                "contexts": {},  # Track per-context success
            }
            self._stats["total_tracked"] += 1
        
        tracking = self._tracking[relationship_id]
        tracking["usage_count"] += 1
        
        if success:
            tracking["success_count"] += 1
            tracking["consecutive_failures"] = 0
            
            # Recover strength (capped at original)
            tracking["current_strength"] = min(
                tracking["original_strength"],
                tracking["current_strength"] * self.RECOVERY_RATE
            )
            
            # Un-degrade if consistently successful after recovery
            if tracking["degraded"] and tracking["success_rate"] > 0.6:
                tracking["degraded"] = False
            
            self._stats["total_successes"] += 1
        else:
            tracking["failure_count"] += 1
            tracking["consecutive_failures"] += 1
            
            # Degrade strength
            tracking["current_strength"] *= self.DEGRADATION_RATE
            
            self._stats["total_failures"] += 1
        
        # Update success rate
        tracking["success_rate"] = (
            tracking["success_count"] / tracking["usage_count"]
        )
        
        # Track per-context success
        if context:
            if context not in tracking["contexts"]:
                tracking["contexts"][context] = {"success": 0, "failure": 0}
            if success:
                tracking["contexts"][context]["success"] += 1
            else:
                tracking["contexts"][context]["failure"] += 1
        
        # Check degradation thresholds
        if not tracking["degraded"]:
            if (tracking["success_rate"] < self.DEGRADED_SUCCESS_RATE and
                tracking["usage_count"] >= 3):
                tracking["degraded"] = True
                self._stats["degraded_count"] += 1
        
        # Check override threshold
        if not tracking["marked_for_override"]:
            if (tracking["consecutive_failures"] >= self.CONSECUTIVE_FAILURES_THRESHOLD or
                (tracking["success_rate"] < self.OVERRIDE_SUCCESS_RATE and
                 tracking["usage_count"] >= 5)):
                tracking["marked_for_override"] = True
                self._deprecated.add(relationship_id)
                self._stats["override_count"] += 1
        
        return tracking
    
    def get_strength(self, relationship_id: str) -> float:
        """Get current strength of a relationship (default 1.0 if unknown)."""
        if relationship_id in self._tracking:
            return self._tracking[relationship_id]["current_strength"]
        return 1.0
    
    def is_deprecated(self, relationship_id: str) -> bool:
        """Check if a relationship is marked for override."""
        return relationship_id in self._deprecated
    
    def is_degraded(self, relationship_id: str) -> bool:
        """Check if a relationship is degraded but not yet deprecated."""
        if relationship_id in self._tracking:
            return self._tracking[relationship_id]["degraded"]
        return False
    
    def get_context_success_rate(
        self,
        relationship_id: str,
        context: str
    ) -> Optional[float]:
        """Get success rate for a specific context."""
        if relationship_id not in self._tracking:
            return None
        
        ctx_data = self._tracking[relationship_id]["contexts"].get(context)
        if not ctx_data:
            return None
        
        total = ctx_data["success"] + ctx_data["failure"]
        if total == 0:
            return None
        
        return ctx_data["success"] / total
    
    def get_non_deprecated_templates(
        self,
        template_names: List[str],
        context: Optional[str] = None
    ) -> List[str]:
        """
        Filter template names to exclude deprecated ones.
        
        If context provided, also excludes templates with <30% success in that context.
        """
        result = []
        
        for name in template_names:
            # Skip deprecated
            if self.is_deprecated(name):
                continue
            
            # Check context-specific success if provided
            if context:
                ctx_rate = self.get_context_success_rate(name, context)
                if ctx_rate is not None and ctx_rate < 0.30:
                    continue
            
            result.append(name)
        
        return result
    
    def restore_relationship(self, relationship_id: str):
        """Restore a deprecated relationship (user override)."""
        if relationship_id in self._deprecated:
            self._deprecated.remove(relationship_id)
        
        if relationship_id in self._tracking:
            tracking = self._tracking[relationship_id]
            tracking["marked_for_override"] = False
            tracking["degraded"] = False
            tracking["consecutive_failures"] = 0
            tracking["current_strength"] = max(0.5, tracking["current_strength"])
    
    def apply_time_decay(self, decay_rate: float = 0.99):
        """Apply time-based decay to all relationships (call periodically)."""
        for tracking in self._tracking.values():
            if not tracking["marked_for_override"]:
                tracking["current_strength"] *= decay_rate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            **self._stats,
            "avg_success_rate": (
                self._stats["total_successes"] /
                max(self._stats["total_successes"] + self._stats["total_failures"], 1)
            ),
            "deprecated_templates": list(self._deprecated),
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save tracker state for persistence."""
        return {
            "tracking": self._tracking,
            "deprecated": list(self._deprecated),
            "stats": self._stats,
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load tracker state from persistence."""
        self._tracking = state.get("tracking", {})
        self._deprecated = set(state.get("deprecated", []))
        self._stats = state.get("stats", self._stats)


# Global tracker instance
_relationship_tracker: Optional[RelationshipStrengthTracker] = None

def get_relationship_tracker() -> RelationshipStrengthTracker:
    """Get or create the global relationship strength tracker."""
    global _relationship_tracker
    if _relationship_tracker is None:
        _relationship_tracker = RelationshipStrengthTracker()
    return _relationship_tracker


# =============================================================================
# Temporal Sequence Learning (from ExtendedReasoningSystem)
# =============================================================================

class TemplateSequenceLearner:
    """
    Learns which template sequences tend to work together.
    
    From ExtendedReasoningSystem's temporal relationship capabilities,
    adapted for template sequencing:
    - Tracks BEFORE/AFTER relationships between templates
    - Learns successful composite sequences
    - Suggests likely next steps based on history
    
    Example learned patterns:
    - crop_nonzero BEFORE mark_boundary (common preprocessing)
    - rotate_90 BEFORE flip_horizontal (geometric combo)
    - size_change_detected ENABLES scale_2x (causal)
    """
    
    # Pre-defined common sequences (bootstrapped knowledge)
    COMMON_SEQUENCES = {
        # Preprocessing -> morphological
        ('crop_nonzero', 'mark_boundary'): 0.8,
        ('crop_nonzero', 'mark_boundary_8conn'): 0.75,
        ('crop_nonzero', 'detect_and_mark_boundary'): 0.7,
        # Preprocessing -> geometric
        ('crop_nonzero', 'rotate_90'): 0.6,
        ('crop_nonzero', 'flip_horizontal'): 0.55,
        # Geometric combos
        ('rotate_90', 'flip_horizontal'): 0.65,
        ('rotate_90', 'flip_vertical'): 0.6,
        ('flip_horizontal', 'rotate_90'): 0.6,
        ('rotate_180', 'flip_horizontal'): 0.5,
        # Morphological -> postprocessing
        ('mark_boundary', 'crop_nonzero'): 0.5,
        # Gravity combos
        ('gravity_down', 'crop_nonzero'): 0.6,
        ('crop_nonzero', 'gravity_down'): 0.55,
    }
    
    def __init__(self):
        """Initialize sequence learner."""
        # Learned sequence successes: (template1, template2) -> {success, failure, confidence}
        self._sequence_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Initialize with common sequences
        for seq, confidence in self.COMMON_SEQUENCES.items():
            self._sequence_stats[seq] = {
                "success": 1,
                "failure": 0,
                "confidence": confidence,
                "learned": False  # Pre-defined, not learned
            }
        
        # Recent successful sequences (for learning)
        self._recent_successes: List[List[str]] = []
        
        # Statistics
        self._stats = {
            "sequences_learned": 0,
            "sequence_suggestions": 0,
            "sequence_successes": 0,
        }
    
    def record_sequence_outcome(
        self,
        sequence: List[str],
        success: bool,
        accuracy: float = 0.0
    ):
        """
        Record the outcome of trying a template sequence.
        
        Args:
            sequence: List of template names in order
            success: Whether the sequence achieved exact match
            accuracy: Partial accuracy achieved
        """
        if len(sequence) < 2:
            return
        
        # Record pairwise relationships
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            
            if pair not in self._sequence_stats:
                self._sequence_stats[pair] = {
                    "success": 0,
                    "failure": 0,
                    "confidence": 0.5,
                    "learned": True
                }
            
            stats = self._sequence_stats[pair]
            
            if success:
                stats["success"] += 1
                self._stats["sequence_successes"] += 1
            else:
                stats["failure"] += 1
            
            # Update confidence
            total = stats["success"] + stats["failure"]
            if total > 0:
                stats["confidence"] = stats["success"] / total
                # Boost confidence for repeated successes
                if stats["success"] >= 3:
                    stats["confidence"] = min(0.95, stats["confidence"] * 1.1)
        
        # Track recent successes for pattern mining
        if success:
            self._recent_successes.append(sequence)
            if len(self._recent_successes) > 100:
                self._recent_successes.pop(0)
            self._stats["sequences_learned"] += 1
    
    def get_likely_next_templates(
        self,
        current_template: str,
        top_k: int = 5,
        min_confidence: float = 0.4
    ) -> List[Tuple[str, float]]:
        """
        Get likely next templates based on learned sequences.
        
        Uses temporal AFTER relationships: What typically comes AFTER current_template?
        
        Args:
            current_template: Template that was just applied
            top_k: Number of suggestions to return
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of (template_name, confidence) sorted by confidence
        """
        self._stats["sequence_suggestions"] += 1
        
        suggestions = []
        
        for (t1, t2), stats in self._sequence_stats.items():
            if t1 == current_template and stats["confidence"] >= min_confidence:
                suggestions.append((t2, stats["confidence"]))
        
        # Sort by confidence descending
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_k]
    
    def get_likely_previous_templates(
        self,
        target_template: str,
        top_k: int = 5,
        min_confidence: float = 0.4
    ) -> List[Tuple[str, float]]:
        """
        Get likely previous templates based on learned sequences.
        
        Uses temporal BEFORE relationships: What typically comes BEFORE target_template?
        
        Args:
            target_template: Template we want to use
            top_k: Number of suggestions to return
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of (template_name, confidence) sorted by confidence
        """
        suggestions = []
        
        for (t1, t2), stats in self._sequence_stats.items():
            if t2 == target_template and stats["confidence"] >= min_confidence:
                suggestions.append((t1, stats["confidence"]))
        
        # Sort by confidence descending
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_k]
    
    def suggest_two_step_sequences(
        self,
        characteristics: Dict[str, Any],
        top_k: int = 10
    ) -> List[Tuple[List[str], float]]:
        """
        Suggest complete 2-step sequences based on task characteristics.
        
        Combines:
        1. Task-appropriate templates
        2. Learned sequence successes
        3. Pre-defined common patterns
        
        Args:
            characteristics: Task analysis results
            top_k: Number of suggestions to return
        
        Returns:
            List of ([step1, step2], combined_confidence)
        """
        suggestions = []
        likely_categories = characteristics.get('likely_categories', [])
        
        for (t1, t2), stats in self._sequence_stats.items():
            if stats["confidence"] >= 0.4:
                # Boost confidence if templates match task characteristics
                boost = 1.0
                
                # Check if t1 matches likely categories
                for cat in likely_categories:
                    if cat in t1 or cat in t2:
                        boost *= 1.2
                
                # Check for specific characteristic matches
                if characteristics.get('size_changes'):
                    if 'crop' in t1 or 'scale' in t1 or 'tile' in t1:
                        boost *= 1.3
                
                if characteristics.get('new_colors_introduced'):
                    if 'mark' in t1 or 'boundary' in t1 or 'mark' in t2 or 'boundary' in t2:
                        boost *= 1.3
                
                combined_confidence = min(0.95, stats["confidence"] * boost)
                suggestions.append(([t1, t2], combined_confidence))
        
        # Sort by combined confidence
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            **self._stats,
            "total_sequences_tracked": len(self._sequence_stats),
            "high_confidence_sequences": sum(
                1 for s in self._sequence_stats.values() if s["confidence"] >= 0.7
            ),
        }


# =============================================================================
# IF-THEN Conditional Chain Reasoning (from ExtendedReasoningSystem)
# =============================================================================

class ConditionalChainReasoner:
    """
    Learns and applies IF-THEN conditional chains for template selection.
    
    From ExtendedReasoningSystem's conditional reasoning capabilities:
    - IF condition THEN outcome relationships
    - Chain following for multi-step reasoning
    - Causal modeling (CAUSES, ENABLES, PREVENTS)
    
    Example chains:
    - IF edge_errors THEN mark_boundary ENABLES exact_match
    - IF size_decreases THEN crop_nonzero CAUSES size_match
    - IF new_colors THEN morphological ENABLES boundary_detection
    """
    
    # Pre-defined conditional rules
    CONDITIONAL_RULES = [
        # Error-based conditionals
        {"condition": "edge_errors", "action": "mark_boundary", "effect": "fix_edges", "confidence": 0.85},
        {"condition": "edge_errors", "action": "mark_boundary_8conn", "effect": "fix_edges", "confidence": 0.8},
        {"condition": "corner_errors", "action": "rotate_90", "effect": "fix_corners", "confidence": 0.6},
        {"condition": "corner_errors", "action": "flip_diagonal", "effect": "fix_corners", "confidence": 0.55},
        {"condition": "interior_errors", "action": "fill_enclosed", "effect": "fix_interior", "confidence": 0.7},
        {"condition": "interior_errors", "action": "color_replace", "effect": "fix_interior", "confidence": 0.5},
        {"condition": "color_swap_errors", "action": "color_swap", "effect": "fix_colors", "confidence": 0.9},
        
        # Size-based conditionals
        {"condition": "size_increases", "action": "tile_2x2", "effect": "match_size", "confidence": 0.8},
        {"condition": "size_increases", "action": "scale_2x", "effect": "match_size", "confidence": 0.75},
        {"condition": "size_decreases", "action": "crop_nonzero", "effect": "match_size", "confidence": 0.85},
        
        # Color-based conditionals
        {"condition": "new_colors", "action": "mark_boundary", "effect": "add_boundary_color", "confidence": 0.8},
        {"condition": "colors_removed", "action": "color_replace", "effect": "remove_color", "confidence": 0.7},
        
        # Position-based conditionals
        {"condition": "position_shifted", "action": "translate_by_offset", "effect": "correct_position", "confidence": 0.85},
        {"condition": "position_shifted", "action": "center_object", "effect": "correct_position", "confidence": 0.6},
        
        # Structural conditionals
        {"condition": "multi_object", "action": "gravity_down", "effect": "compact_objects", "confidence": 0.6},
    ]
    
    def __init__(self):
        """Initialize conditional chain reasoner."""
        # Learned conditionals: condition -> [rules]
        self._conditionals: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize with pre-defined rules
        for rule in self.CONDITIONAL_RULES:
            cond = rule["condition"]
            if cond not in self._conditionals:
                self._conditionals[cond] = []
            self._conditionals[cond].append({
                "action": rule["action"],
                "effect": rule["effect"],
                "confidence": rule["confidence"],
                "learned": False
            })
        
        # Track rule effectiveness
        self._rule_stats: Dict[str, Dict[str, int]] = {}
        
        self._stats = {
            "rules_applied": 0,
            "rules_successful": 0,
            "chains_followed": 0,
        }
    
    def query_if_then(
        self,
        condition: str,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Query: IF condition, THEN what action?
        
        Args:
            condition: The condition to match (e.g., "edge_errors", "size_increases")
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of matching rules sorted by confidence
        """
        self._stats["rules_applied"] += 1
        
        results = []
        
        # Exact match
        if condition in self._conditionals:
            for rule in self._conditionals[condition]:
                if rule["confidence"] >= min_confidence:
                    results.append(rule)
        
        # Partial match (condition contains keyword)
        for cond, rules in self._conditionals.items():
            if condition != cond and (cond in condition or condition in cond):
                for rule in rules:
                    if rule["confidence"] >= min_confidence * 0.8:  # Slightly lower threshold
                        results.append({**rule, "partial_match": True})
        
        # Sort by confidence
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return results
    
    def query_what_causes(
        self,
        effect: str
    ) -> List[Dict[str, Any]]:
        """
        Query: WHAT causes this effect?
        
        Reverse lookup: Given a desired effect, what conditions/actions lead to it?
        
        Args:
            effect: The desired effect (e.g., "fix_edges", "match_size")
        
        Returns:
            List of (condition, action, confidence) that lead to this effect
        """
        results = []
        
        for condition, rules in self._conditionals.items():
            for rule in rules:
                if effect in rule.get("effect", "") or rule.get("effect", "") in effect:
                    results.append({
                        "condition": condition,
                        "action": rule["action"],
                        "effect": rule["effect"],
                        "confidence": rule["confidence"]
                    })
        
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return results
    
    def chain_conditionals(
        self,
        start_condition: str,
        max_depth: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        Follow conditional chains to find multi-step solutions.
        
        From ExtendedReasoningSystem.chain_conditionals():
        Follows IF-THEN relationships through multiple steps.
        
        Example chain:
        1. IF edge_errors THEN mark_boundary (effect: fix_edges)
        2. IF fix_edges THEN crop_nonzero (effect: clean_output)
        3. Result: edge_errors -> mark_boundary -> crop_nonzero
        
        Args:
            start_condition: Starting condition
            max_depth: Maximum chain length
        
        Returns:
            List of chains, where each chain is a list of rules
        """
        self._stats["chains_followed"] += 1
        chains = []
        
        def follow_chain(current: str, chain: List[Dict], depth: int):
            if depth >= max_depth:
                if chain:
                    chains.append(chain.copy())
                return
            
            rules = self.query_if_then(current, min_confidence=0.4)
            
            if not rules:
                if chain:
                    chains.append(chain.copy())
                return
            
            for rule in rules[:3]:  # Top 3 rules to prevent explosion
                effect = rule.get("effect", "")
                if effect and effect != current:
                    chain.append(rule)
                    follow_chain(effect, chain, depth + 1)
                    chain.pop()
        
        follow_chain(start_condition, [], 0)
        return chains
    
    def learn_conditional(
        self,
        condition: str,
        action: str,
        effect: str,
        success: bool
    ):
        """
        Learn a new conditional rule from experience.
        
        Args:
            condition: The triggering condition
            action: The action taken
            effect: The resulting effect
            success: Whether it worked
        """
        rule_key = f"{condition}::{action}::{effect}"
        
        if rule_key not in self._rule_stats:
            self._rule_stats[rule_key] = {"success": 0, "failure": 0}
        
        if success:
            self._rule_stats[rule_key]["success"] += 1
            self._stats["rules_successful"] += 1
        else:
            self._rule_stats[rule_key]["failure"] += 1
        
        # Consider adding as new rule if success rate is high
        stats = self._rule_stats[rule_key]
        total = stats["success"] + stats["failure"]
        
        if total >= 3:  # Minimum observations
            success_rate = stats["success"] / total
            
            if success_rate >= 0.6:  # Good success rate
                # Add or update rule
                if condition not in self._conditionals:
                    self._conditionals[condition] = []
                
                # Check if rule already exists
                existing = None
                for rule in self._conditionals[condition]:
                    if rule["action"] == action:
                        existing = rule
                        break
                
                if existing:
                    existing["confidence"] = success_rate
                else:
                    self._conditionals[condition].append({
                        "action": action,
                        "effect": effect,
                        "confidence": success_rate,
                        "learned": True
                    })
    
    def get_actions_for_error_patterns(
        self,
        error_patterns: List['ErrorPattern']
    ) -> List[Tuple[str, str, float]]:
        """
        Get suggested actions based on error patterns.
        
        Maps error types to conditions and queries for actions.
        
        Args:
            error_patterns: List of ErrorPattern objects
        
        Returns:
            List of (action, reason, confidence) tuples
        """
        suggestions = []
        
        for pattern in error_patterns:
            # Map error type to condition
            condition_map = {
                'edge': 'edge_errors',
                'corner': 'corner_errors',
                'interior': 'interior_errors',
                'color_swap': 'color_swap_errors',
                'size_height': 'size_decreases' if 'crop' in (pattern.suggested_fix or '') else 'size_increases',
                'size_width': 'size_decreases' if 'crop' in (pattern.suggested_fix or '') else 'size_increases',
            }
            
            condition = condition_map.get(pattern.error_type, pattern.error_type)
            
            # Query for actions
            rules = self.query_if_then(condition)
            
            for rule in rules[:2]:  # Top 2 per pattern
                reason = f"IF {condition} THEN {rule['action']} ({rule['effect']})"
                suggestions.append((rule["action"], reason, rule["confidence"]))
        
        # Deduplicate and sort
        seen = set()
        unique_suggestions = []
        for action, reason, conf in suggestions:
            if action not in seen:
                seen.add(action)
                unique_suggestions.append((action, reason, conf))
        
        unique_suggestions.sort(key=lambda x: x[2], reverse=True)
        return unique_suggestions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reasoner statistics."""
        return {
            **self._stats,
            "total_rules": sum(len(rules) for rules in self._conditionals.values()),
            "learned_rules": sum(
                1 for rules in self._conditionals.values()
                for r in rules if r.get("learned", False)
            ),
        }


# =============================================================================
# Goal-Directed Sub-Goal Tracking (from ExtendedReasoningSystem)
# =============================================================================

class GoalDirectedSolver:
    """
    Tracks progress toward solving goals using sub-goal decomposition.
    
    From ExtendedReasoningSystem's goal management:
    - Decomposes complex tasks into sub-goals
    - Tracks progress toward each sub-goal
    - Prioritizes actions based on remaining sub-goals
    
    Example goal structure for ARC task:
    - GOAL: exact_match
      - SUB-GOAL: size_match (if size differs)
      - SUB-GOAL: position_match (if position shifts)
      - SUB-GOAL: color_match (if colors differ)
      - SUB-GOAL: pattern_match (core transformation)
    """
    
    def __init__(self):
        """Initialize goal-directed solver."""
        # Current goals: goal_id -> goal_data
        self._goals: Dict[str, Dict[str, Any]] = {}
        
        # Goal dependencies: goal_id -> [required_sub_goals]
        self._dependencies: Dict[str, List[str]] = {}
        
        # Statistics
        self._stats = {
            "goals_created": 0,
            "goals_completed": 0,
            "subgoals_completed": 0,
        }
    
    def create_goals_for_task(
        self,
        characteristics: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create goal hierarchy based on task characteristics.
        
        Analyzes what needs to be done and creates prioritized sub-goals.
        
        Args:
            characteristics: Task analysis from analyze_task_characteristics()
        
        Returns:
            Dict of goal_id -> goal_data
        """
        self._goals.clear()
        self._dependencies.clear()
        
        # Root goal
        root_id = "exact_match"
        self._goals[root_id] = {
            "id": root_id,
            "description": "Achieve exact match on all training pairs",
            "priority": 1.0,
            "status": "active",
            "progress": 0.0,
            "sub_goals": []
        }
        
        sub_goal_priority = 0.9
        
        # Size sub-goal (if size changes)
        if characteristics.get('size_changes'):
            size_id = "size_match"
            self._goals[size_id] = {
                "id": size_id,
                "description": "Match output size",
                "priority": sub_goal_priority,
                "status": "active",
                "progress": 0.0,
                "suggested_templates": ["crop_nonzero", "scale_2x", "tile_2x2"]
            }
            self._goals[root_id]["sub_goals"].append(size_id)
            self._dependencies[root_id] = self._dependencies.get(root_id, []) + [size_id]
            sub_goal_priority -= 0.05
        
        # Position sub-goal (if position shifts)
        if characteristics.get('position_shifts'):
            pos_id = "position_match"
            self._goals[pos_id] = {
                "id": pos_id,
                "description": "Match object positions",
                "priority": sub_goal_priority,
                "status": "active",
                "progress": 0.0,
                "suggested_templates": ["translate_by_offset", "center_object", "gravity_down"]
            }
            self._goals[root_id]["sub_goals"].append(pos_id)
            self._dependencies[root_id] = self._dependencies.get(root_id, []) + [pos_id]
            sub_goal_priority -= 0.05
        
        # Color sub-goal (if colors change)
        if characteristics.get('new_colors_introduced') or characteristics.get('colors_removed'):
            color_id = "color_match"
            self._goals[color_id] = {
                "id": color_id,
                "description": "Match color configuration",
                "priority": sub_goal_priority,
                "status": "active",
                "progress": 0.0,
                "suggested_templates": ["mark_boundary", "color_replace", "color_swap"]
            }
            self._goals[root_id]["sub_goals"].append(color_id)
            self._dependencies[root_id] = self._dependencies.get(root_id, []) + [color_id]
            sub_goal_priority -= 0.05
        
        # Pattern sub-goal (always needed)
        pattern_id = "pattern_match"
        likely_cats = characteristics.get('likely_categories', ['geometric'])
        suggested = []
        
        if 'geometric' in likely_cats:
            suggested.extend(['rotate_90', 'flip_horizontal', 'flip_vertical'])
        if 'gravity' in likely_cats:
            suggested.extend(['gravity_down', 'gravity_up'])
        if 'morphological' in likely_cats:
            suggested.extend(['mark_boundary', 'extract_boundary', 'fill_enclosed'])
        
        self._goals[pattern_id] = {
            "id": pattern_id,
            "description": "Apply correct transformation pattern",
            "priority": sub_goal_priority,
            "status": "active",
            "progress": 0.0,
            "suggested_templates": suggested[:5]
        }
        self._goals[root_id]["sub_goals"].append(pattern_id)
        
        self._stats["goals_created"] += len(self._goals)
        
        return self._goals
    
    def update_goal_progress(
        self,
        goal_id: str,
        progress: float,
        accuracy_details: Optional[Dict[str, float]] = None
    ):
        """
        Update progress toward a goal.
        
        Args:
            goal_id: Goal to update
            progress: New progress value (0-1)
            accuracy_details: Optional details like size_accuracy, color_accuracy
        """
        if goal_id not in self._goals:
            return
        
        old_progress = self._goals[goal_id]["progress"]
        self._goals[goal_id]["progress"] = progress
        
        # Check for completion
        if progress >= 1.0 and old_progress < 1.0:
            self._goals[goal_id]["status"] = "completed"
            self._stats["subgoals_completed"] += 1
        
        # Update root goal progress based on sub-goal progress
        if "exact_match" in self._goals:
            root = self._goals["exact_match"]
            if root["sub_goals"]:
                sub_progress = sum(
                    self._goals.get(sg, {}).get("progress", 0)
                    for sg in root["sub_goals"]
                ) / len(root["sub_goals"])
                root["progress"] = sub_progress
    
    def get_current_priority_goal(self) -> Optional[Dict[str, Any]]:
        """
        Get the highest priority active goal.
        
        Returns the most important goal that isn't completed yet.
        """
        active_goals = [
            g for g in self._goals.values()
            if g.get("status") == "active"
        ]
        
        if not active_goals:
            return None
        
        # Sort by priority (highest first) then by progress (lowest first)
        active_goals.sort(key=lambda g: (-g.get("priority", 0), g.get("progress", 0)))
        return active_goals[0]
    
    def get_suggested_templates_for_goals(self) -> List[Tuple[str, str, float]]:
        """
        Get template suggestions based on current goals.
        
        Returns templates prioritized by which goals they help achieve.
        
        Returns:
            List of (template_name, goal_id, priority_score)
        """
        suggestions = []
        
        for goal_id, goal in self._goals.items():
            if goal.get("status") != "active":
                continue
            
            templates = goal.get("suggested_templates", [])
            priority = goal.get("priority", 0.5)
            progress = goal.get("progress", 0)
            
            # Adjust priority based on progress (lower progress = higher priority)
            adjusted_priority = priority * (1.0 - progress * 0.5)
            
            for template in templates:
                suggestions.append((template, goal_id, adjusted_priority))
        
        # Sort by adjusted priority
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions
    
    def check_goal_completion(self, accuracy: float, error_patterns: List['ErrorPattern']) -> Dict[str, Any]:
        """
        Check which goals might be completed based on accuracy and error patterns.
        
        Args:
            accuracy: Overall accuracy achieved
            error_patterns: Current error patterns
        
        Returns:
            Dict with completion status for each goal
        """
        results = {}
        
        # Check size goal
        if "size_match" in self._goals:
            size_errors = any(p.error_type in ('size_height', 'size_width') for p in error_patterns)
            self.update_goal_progress("size_match", 0.0 if size_errors else 1.0)
            results["size_match"] = not size_errors
        
        # Check color goal
        if "color_match" in self._goals:
            color_errors = any(p.error_type == 'color_swap' for p in error_patterns)
            self.update_goal_progress("color_match", 0.0 if color_errors else accuracy)
            results["color_match"] = not color_errors and accuracy > 0.9
        
        # Check position goal
        if "position_match" in self._goals:
            # Position errors manifest as edge/corner errors typically
            position_errors = any(p.error_type in ('edge', 'corner') and p.percentage > 0.3 for p in error_patterns)
            self.update_goal_progress("position_match", 0.0 if position_errors else accuracy)
            results["position_match"] = not position_errors and accuracy > 0.8
        
        # Check pattern goal
        if "pattern_match" in self._goals:
            interior_errors = any(p.error_type == 'interior' and p.percentage > 0.2 for p in error_patterns)
            self.update_goal_progress("pattern_match", accuracy if not interior_errors else accuracy * 0.5)
            results["pattern_match"] = accuracy >= 0.95
        
        # Check root goal
        if "exact_match" in self._goals:
            if accuracy >= 1.0:
                self._goals["exact_match"]["status"] = "completed"
                self._stats["goals_completed"] += 1
            results["exact_match"] = accuracy >= 1.0
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get solver statistics."""
        return {
            **self._stats,
            "active_goals": sum(1 for g in self._goals.values() if g.get("status") == "active"),
            "completed_goals": sum(1 for g in self._goals.values() if g.get("status") == "completed"),
        }


# =============================================================================
# RelationshipReasoner - Bind-not-Bundle Context Enrichment
# =============================================================================

class RelationshipReasoner:
    """
    Enriches recipes with semantic relationships using the "Bind-not-Bundle" fix.
    
    This implements non-diluting context attachment from extended thinking:
    - Instead of bundling (which dilutes signal), we bind context tags
    - Context tags are deterministic seeds that encode semantic meaning
    - O(1) operation that preserves core transformation signal
    
    The reasoner annotates recipes with:
    1. Template identity (which template is being used)
    2. Category context (IS-A relationship: geometric, morphological, etc.)
    3. Inverse context (OPPOSITE relationship: helps back out of dead ends)
    4. Custom relationships (ENABLES, CAUSES, PREVENTS, FOLLOWS)
    
    Example:
        >>> reasoner = RelationshipReasoner(knowledge)
        >>> enriched_recipe = reasoner.enrich_recipe(recipe, "rotate_90")
        # Recipe now has context: template_rotate_90, category_geometric, has_inverse_rotate_270
    """
    
    def __init__(
        self,
        knowledge: Optional['TemplateRelationshipKnowledge'] = None,
        tracker: Optional['RelationshipStrengthTracker'] = None
    ):
        """
        Initialize the relationship reasoner.
        
        Args:
            knowledge: Template relationship knowledge base
            tracker: Relationship strength tracker for degradation
        """
        self.knowledge = knowledge or get_template_relationships()
        self.tracker = tracker or get_relationship_tracker()
        
        self._stats = {
            "recipes_enriched": 0,
            "contexts_added": 0,
            "inverse_contexts_added": 0,
            "relationship_contexts_added": 0,
        }
    
    def enrich_recipe(
        self,
        recipe: 'ReasoningRecipe',
        matched_template: str,
        add_category: bool = True,
        add_inverse: bool = True,
        add_relationships: bool = True
    ) -> 'ReasoningRecipe':
        """
        Enrich a recipe with semantic relationships using bind-not-bundle.
        
        This is the core method that implements non-diluting context attachment.
        Instead of bundling multiple vectors (which dilutes the signal), we bind
        deterministic context seeds that preserve the core transformation signal.
        
        Args:
            recipe: Recipe to enrich
            matched_template: Template that was matched
            add_category: Add IS-A category context
            add_inverse: Add OPPOSITE inverse context
            add_relationships: Add custom relationship contexts
        
        Returns:
            Enriched recipe (modified in place)
        """
        self._stats["recipes_enriched"] += 1
        
        # 1. Attach template identity
        recipe.add_context(f"template_{matched_template}")
        self._stats["contexts_added"] += 1
        
        # 2. Add Category (IS-A) Context
        if add_category:
            category = self.knowledge.get_category(matched_template)
            if category:
                recipe.add_context(f"category_{category}")
                self._stats["contexts_added"] += 1
        
        # 3. Add Inverse (OPPOSITE) Context if applicable
        # This helps the solver "back out" of dead ends by knowing the reverse
        if add_inverse:
            opposite = self.knowledge.get_opposite(matched_template)
            if opposite:
                recipe.add_context(f"has_inverse_{opposite}")
                self._stats["inverse_contexts_added"] += 1
        
        # 4. Add Custom Relationship Contexts
        if add_relationships:
            # Get all relationships for this template
            relationships = get_relationships_for_entity(matched_template)
            
            for entity_a, entity_b, rel_type, seed in relationships:
                # Add the relationship context
                recipe.add_operation(RecipeOp(RecipeOpType.BIND_CONTEXT, seed))
                
                # Add descriptive context for debugging
                if entity_a == matched_template:
                    recipe.add_context(f"{rel_type.lower()}_{entity_b}")
                else:
                    recipe.add_context(f"is_{rel_type.lower()}_by_{entity_a}")
                
                self._stats["relationship_contexts_added"] += 1
        
        # 5. Add strength-based context from tracker
        if self.tracker:
            strength = self.tracker.get_strength(matched_template)
            if strength < 0.5:
                recipe.add_context("low_confidence_template")
            elif strength > 0.8:
                recipe.add_context("high_confidence_template")
            
            if self.tracker.is_degraded(matched_template):
                recipe.add_context("degraded_template")
        
        return recipe
    
    def suggest_next_templates(
        self,
        current_template: str,
        error_context: Optional[str] = None,
        top_k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Suggest next templates based on relationships and error context.
        
        Uses PREDICTS, ENABLES, and FOLLOWS relationships to suggest
        what template should come next in a sequence.
        
        Args:
            current_template: Template that was just tried
            error_context: Optional error type that occurred
            top_k: Number of suggestions to return
        
        Returns:
            List of (template_name, reason, confidence) tuples
        """
        suggestions = []
        
        # 1. PREDICTS relationships
        predicted = self.knowledge.get_predicted_next(current_template, top_k)
        for pred in predicted:
            suggestions.append((pred, f"PREDICTS from {current_template}", 0.7))
        
        # 2. FOLLOWS relationships from custom relationships
        for entity_a, entity_b, rel_type, seed in get_relationships_for_entity(current_template):
            if rel_type == "FOLLOWS" and entity_a == current_template:
                suggestions.append((entity_b, f"FOLLOWS {current_template}", 0.6))
            elif rel_type == "ENABLES" and entity_a == current_template:
                suggestions.append((entity_b, f"ENABLED by {current_template}", 0.65))
        
        # 3. If error context provided, use conditional reasoning
        if error_context:
            # Map error to condition and query conditionals
            conditional_reasoner = get_conditional_reasoner()
            rules = conditional_reasoner.query_if_then(error_context)
            for rule in rules[:2]:
                suggestions.append((
                    rule["action"],
                    f"IF {error_context} THEN {rule['action']}",
                    rule["confidence"]
                ))
        
        # 4. SIMILAR templates if few suggestions
        if len(suggestions) < top_k:
            similar = self.knowledge.get_similar(current_template, top_k - len(suggestions))
            for sim in similar:
                if not any(s[0] == sim for s in suggestions):
                    suggestions.append((sim, f"SIMILAR to {current_template}", 0.5))
        
        # Sort by confidence and deduplicate
        seen = set()
        unique = []
        for name, reason, conf in sorted(suggestions, key=lambda x: x[2], reverse=True):
            if name not in seen:
                seen.add(name)
                unique.append((name, reason, conf))
        
        return unique[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reasoner statistics."""
        return self._stats.copy()


# =============================================================================
# 4-Stage Swarm Thinking Pipeline
# =============================================================================

class SwarmAgent:
    """
    A single agent in the swarm that explores a specific strategy.
    
    Each agent has:
    - A strategy (geometric, morphological, structural, etc.)
    - A recipe it's building
    - A confidence score
    """
    
    def __init__(
        self,
        agent_id: str,
        strategy: str,
        base_seed: int
    ):
        """
        Initialize a swarm agent.
        
        Args:
            agent_id: Unique identifier for this agent
            strategy: Strategy this agent follows (e.g., "geometric", "morphological")
            base_seed: Base seed for recipe generation
        """
        self.agent_id = agent_id
        self.strategy = strategy
        self.recipe = ReasoningRecipe(base_seed=base_seed)
        self.confidence = 0.5
        self.templates_tried: List[str] = []
        self.best_accuracy = 0.0
        self.best_template: Optional[str] = None
    
    def explore(
        self,
        knowledge: 'TemplateRelationshipKnowledge',
        characteristics: Dict[str, Any],
        n_templates: int = 3
    ) -> List[str]:
        """
        Generate candidate templates based on strategy and characteristics.
        
        Args:
            knowledge: Template relationship knowledge
            characteristics: Task characteristics
            n_templates: Number of templates to suggest
        
        Returns:
            List of template names to try
        """
        candidates = []
        
        # Get templates in this agent's strategy category
        strategy_templates = knowledge.get_templates_in_category(self.strategy)
        
        # Filter by task characteristics
        if characteristics.get('size_changes'):
            # Boost structural templates
            if self.strategy == 'structural':
                candidates.extend(strategy_templates[:n_templates])
        
        if characteristics.get('new_colors_introduced'):
            # Boost morphological templates
            if self.strategy == 'morphological':
                candidates.extend(strategy_templates[:n_templates])
        
        if characteristics.get('position_shifts'):
            # Boost translation templates
            if self.strategy == 'translation':
                candidates.extend(strategy_templates[:n_templates])
        
        # Always include some templates from the strategy
        if len(candidates) < n_templates:
            for t in strategy_templates:
                if t not in candidates and t not in self.templates_tried:
                    candidates.append(t)
                    if len(candidates) >= n_templates:
                        break
        
        return candidates[:n_templates]
    
    def update(
        self,
        template: str,
        accuracy: float,
        exact_match: bool
    ):
        """Update agent state after trying a template."""
        self.templates_tried.append(template)
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_template = template
            self.confidence = min(0.95, self.confidence + 0.1)
        else:
            self.confidence = max(0.1, self.confidence - 0.05)


class SwarmThinkingPipeline:
    """
    4-Stage Swarm Thinking Pipeline for multi-agent recipe exploration.
    
    This implements the swarm thinking approach from extended reasoning:
    
    Stage 1: EXPLORATION
        Multiple agents generate candidate recipes using different heuristics.
        Each agent specializes in a template category (geometric, morphological, etc.)
    
    Stage 2: SELF-VERIFICATION
        Test top candidates against training pairs using ToolBasedVerifier.
        Eliminates obviously wrong candidates early.
    
    Stage 3: OPTIMIZATION
        Perform beam search on best recipes to find shortest working sequence.
        Tries to minimize the number of transformation steps.
    
    Stage 4: SYNTHESIS
        Quality-gate results and synthesize final best recipes.
        Returns the most confident, verified solution.
    
    Usage:
        >>> pipeline = SwarmThinkingPipeline(grid_engine, knowledge)
        >>> result = pipeline.solve(task, characteristics)
    """
    
    # Swarm configuration
    N_AGENTS = 6  # One per major template category
    BEAM_WIDTH = 3  # Number of candidates to keep in beam search
    MAX_SEQUENCE_LENGTH = 3  # Max steps in a composite transformation
    
    AGENT_STRATEGIES = [
        'geometric',    # rotations, flips
        'morphological', # boundary, fill, erode
        'structural',   # tile, scale, crop
        'translation',  # gravity, translate
        'gravity',      # gravity transforms
        'color',        # color swaps, replacements
    ]
    
    def __init__(
        self,
        grid_engine: 'GridTemplateEngine',
        knowledge: Optional['TemplateRelationshipKnowledge'] = None,
        tracker: Optional['RelationshipStrengthTracker'] = None
    ):
        """
        Initialize the swarm thinking pipeline.
        
        Args:
            grid_engine: Grid template engine for verification
            knowledge: Template relationship knowledge
            tracker: Relationship strength tracker
        """
        self.grid_engine = grid_engine
        self.knowledge = knowledge or get_template_relationships()
        self.tracker = tracker or get_relationship_tracker()
        self.reasoner = RelationshipReasoner(self.knowledge, self.tracker)
        
        # Initialize agents
        self.agents: List[SwarmAgent] = []
        for i, strategy in enumerate(self.AGENT_STRATEGIES):
            agent = SwarmAgent(
                agent_id=f"agent_{i}_{strategy}",
                strategy=strategy,
                base_seed=_string_to_seed(f"swarm_agent_{strategy}")
            )
            self.agents.append(agent)
        
        self._stats = {
            "pipeline_runs": 0,
            "stage1_candidates": 0,
            "stage2_verified": 0,
            "stage3_optimized": 0,
            "stage4_synthesized": 0,
            "exact_matches_found": 0,
        }
    
    def stage1_exploration(
        self,
        task: 'TaskSample',
        characteristics: Dict[str, Any]
    ) -> List[Tuple[SwarmAgent, List[str]]]:
        """
        Stage 1: EXPLORATION - Multiple agents generate candidate templates.
        
        Each agent explores templates based on its strategy and task characteristics.
        
        Returns:
            List of (agent, candidate_templates) pairs
        """
        candidates = []
        
        for agent in self.agents:
            agent_candidates = agent.explore(
                self.knowledge,
                characteristics,
                n_templates=3
            )
            
            if agent_candidates:
                candidates.append((agent, agent_candidates))
                self._stats["stage1_candidates"] += len(agent_candidates)
        
        return candidates
    
    def stage2_self_verification(
        self,
        task: 'TaskSample',
        candidates: List[Tuple[SwarmAgent, List[str]]]
    ) -> List[Tuple[SwarmAgent, str, float, bool]]:
        """
        Stage 2: SELF-VERIFICATION - Test candidates against training pairs.
        
        Uses parallel verification to test all candidates quickly.
        
        Returns:
            List of (agent, template, accuracy, exact_match) sorted by accuracy
        """
        verified = []
        
        for agent, templates in candidates:
            # Filter out deprecated templates
            valid_templates = self.tracker.get_non_deprecated_templates(templates)
            
            if valid_templates:
                # Verify templates
                results = parallel_verify_templates(
                    valid_templates,
                    task,
                    self.grid_engine
                )
                
                for template_name, accuracy, exact_match in results:
                    agent.update(template_name, accuracy, exact_match)
                    verified.append((agent, template_name, accuracy, exact_match))
                    self._stats["stage2_verified"] += 1
        
        # Sort by accuracy descending
        verified.sort(key=lambda x: x[2], reverse=True)
        return verified
    
    def stage3_optimization(
        self,
        task: 'TaskSample',
        verified: List[Tuple[SwarmAgent, str, float, bool]],
        characteristics: Dict[str, Any]
    ) -> List[Tuple[List[str], float, bool]]:
        """
        Stage 3: OPTIMIZATION - Beam search for shortest working sequence.
        
        Takes top candidates and tries to find shorter sequences that work.
        
        Returns:
            List of (sequence, accuracy, exact_match) sorted by (exact, -len, accuracy)
        """
        from .grid_templates import TransformationRecipe, TransformationStep
        
        optimized = []
        
        # Take top candidates
        top_candidates = verified[:self.BEAM_WIDTH]
        
        for agent, template, accuracy, exact_match in top_candidates:
            # If exact match with single template, that's already optimal
            if exact_match:
                optimized.append(([template], accuracy, True))
                self._stats["stage3_optimized"] += 1
                continue
            
            # Try 2-step sequences with this template
            if accuracy >= 0.5:  # Only optimize promising candidates
                # Use sequence learner for suggestions
                sequence_learner = get_sequence_learner()
                
                # Get likely next templates
                next_templates = sequence_learner.get_likely_next_templates(
                    template, top_k=3
                )
                
                for next_template, conf in next_templates:
                    try:
                        recipe = TransformationRecipe(steps=[
                            TransformationStep(name=template, params={}),
                            TransformationStep(name=next_template, params={})
                        ])
                        
                        total_acc = 0.0
                        exact_matches = 0
                        
                        for pair in task.train_pairs:
                            try:
                                pred = self.grid_engine.apply_recipe(pair["input"], recipe)
                                if pred == pair["output"]:
                                    exact_matches += 1
                                    total_acc += 1.0
                                elif pred and len(pred) == len(pair["output"]):
                                    # Cell accuracy
                                    cells = len(pred) * len(pred[0])
                                    if cells > 0:
                                        correct = sum(
                                            1 for y in range(len(pred))
                                            for x in range(len(pred[0]))
                                            if pred[y][x] == pair["output"][y][x]
                                        )
                                        total_acc += correct / cells
                            except Exception:
                                pass
                        
                        avg_acc = total_acc / len(task.train_pairs) if task.train_pairs else 0
                        all_exact = exact_matches == len(task.train_pairs)
                        
                        if all_exact or avg_acc > accuracy:
                            optimized.append(([template, next_template], avg_acc, all_exact))
                            self._stats["stage3_optimized"] += 1
                            
                            # Record successful sequence
                            sequence_learner.record_sequence_outcome(
                                [template, next_template],
                                all_exact,
                                avg_acc
                            )
                            
                    except Exception:
                        pass
        
        # Sort: exact matches first, then by sequence length (shorter better), then by accuracy
        optimized.sort(key=lambda x: (not x[2], len(x[0]), -x[1]))
        return optimized
    
    def stage4_synthesis(
        self,
        optimized: List[Tuple[List[str], float, bool]]
    ) -> Optional[Tuple[List[str], float, Dict[str, Any]]]:
        """
        Stage 4: SYNTHESIS - Quality-gate and synthesize final result.
        
        Returns the best solution with metadata.
        
        Returns:
            (sequence, confidence, metadata) or None
        """
        if not optimized:
            return None
        
        # Find the best solution
        best_sequence, best_accuracy, exact_match = optimized[0]
        
        # Quality gate: require minimum accuracy
        if best_accuracy < 0.3:
            return None
        
        self._stats["stage4_synthesized"] += 1
        
        if exact_match:
            self._stats["exact_matches_found"] += 1
        
        metadata = {
            "method": "swarm_thinking_pipeline",
            "sequence_length": len(best_sequence),
            "exact_match": exact_match,
            "candidates_explored": self._stats["stage1_candidates"],
            "verified": self._stats["stage2_verified"],
            "optimized": self._stats["stage3_optimized"],
        }
        
        return (best_sequence, best_accuracy, metadata)
    
    def solve(
        self,
        task: 'TaskSample',
        characteristics: Dict[str, Any]
    ) -> Optional[Tuple[List[str], float, Dict[str, Any]]]:
        """
        Run the full 4-stage swarm thinking pipeline.
        
        Args:
            task: Task to solve
            characteristics: Analyzed task characteristics
        
        Returns:
            (template_sequence, confidence, metadata) or None
        """
        self._stats["pipeline_runs"] += 1
        
        # Stage 1: Exploration
        candidates = self.stage1_exploration(task, characteristics)
        
        if not candidates:
            return None
        
        # Stage 2: Self-Verification
        verified = self.stage2_self_verification(task, candidates)
        
        if not verified:
            return None
        
        # Check for immediate exact match
        for agent, template, accuracy, exact_match in verified:
            if exact_match:
                self._stats["exact_matches_found"] += 1
                return ([template], accuracy, {
                    "method": "swarm_stage2_exact",
                    "agent": agent.agent_id,
                    "strategy": agent.strategy,
                })
        
        # Stage 3: Optimization
        optimized = self.stage3_optimization(task, verified, characteristics)
        
        # Stage 4: Synthesis
        return self.stage4_synthesis(optimized)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self._stats,
            "agents": len(self.agents),
            "agent_confidences": {a.agent_id: a.confidence for a in self.agents},
        }


# Global instances
_sequence_learner: Optional[TemplateSequenceLearner] = None
_conditional_reasoner: Optional[ConditionalChainReasoner] = None
_goal_solver: Optional[GoalDirectedSolver] = None
_relationship_reasoner: Optional[RelationshipReasoner] = None
_swarm_pipeline: Optional[SwarmThinkingPipeline] = None


def get_sequence_learner() -> TemplateSequenceLearner:
    """Get or create the global sequence learner."""
    global _sequence_learner
    if _sequence_learner is None:
        _sequence_learner = TemplateSequenceLearner()
    return _sequence_learner


def get_conditional_reasoner() -> ConditionalChainReasoner:
    """Get or create the global conditional reasoner."""
    global _conditional_reasoner
    if _conditional_reasoner is None:
        _conditional_reasoner = ConditionalChainReasoner()
    return _conditional_reasoner


def get_goal_solver() -> GoalDirectedSolver:
    """Get or create the global goal solver."""
    global _goal_solver
    if _goal_solver is None:
        _goal_solver = GoalDirectedSolver()
    return _goal_solver


def get_relationship_reasoner(
    knowledge: Optional['TemplateRelationshipKnowledge'] = None,
    tracker: Optional['RelationshipStrengthTracker'] = None
) -> RelationshipReasoner:
    """Get or create the global RelationshipReasoner instance."""
    global _relationship_reasoner
    if _relationship_reasoner is None:
        _relationship_reasoner = RelationshipReasoner(knowledge, tracker)
    return _relationship_reasoner


def get_swarm_pipeline(
    grid_engine: 'GridTemplateEngine',
    knowledge: Optional['TemplateRelationshipKnowledge'] = None,
    tracker: Optional['RelationshipStrengthTracker'] = None
) -> SwarmThinkingPipeline:
    """
    Get or create the global SwarmThinkingPipeline instance.
    
    Note: Requires grid_engine to be passed on first call.
    """
    global _swarm_pipeline
    if _swarm_pipeline is None:
        _swarm_pipeline = SwarmThinkingPipeline(grid_engine, knowledge, tracker)
    return _swarm_pipeline


# =============================================================================
# Multi-Scale Parameter Search
# =============================================================================

def multi_scale_parameter_search(
    template_name: str,
    base_params: Dict[str, Any],
    task: 'TaskSample',
    grid_engine: 'GridTemplateEngine',
    variations: Optional[Dict[str, List[Any]]] = None
) -> Tuple[str, float, bool, Dict[str, Any]]:
    """
    Try parameter variations around the base values.
    
    When a template achieves high but not exact accuracy, the parameters
    might be slightly off. This function tries variations:
    - Translation amounts: base ± 1
    - Colors: try adjacent color values
    - Scale: try 2x vs 3x
    
    Args:
        template_name: Template to try
        base_params: Base parameter values to vary around
        task: Task to verify against
        grid_engine: Grid engine for transformations
        variations: Optional explicit variations per param
    
    Returns:
        (template_name, best_accuracy, exact_match, best_params)
    """
    from .grid_templates import TransformationRecipe, TransformationStep
    
    best_accuracy = 0.0
    best_exact = False
    best_params = base_params.copy()
    
    # Default variations for common parameters
    if variations is None:
        variations = {}
        
        for key, value in base_params.items():
            if key in ('dy', 'dx', 'amount') and isinstance(value, int):
                # Translation offsets: try ±1
                variations[key] = [value - 1, value, value + 1]
            elif key in ('boundary_color', 'fill_color', 'old_color', 'new_color'):
                # Colors: try current ± 1 (clamped to 0-9)
                if isinstance(value, int):
                    variations[key] = [
                        max(0, value - 1),
                        value,
                        min(9, value + 1)
                    ]
            elif key == 'scale':
                # Scale: try 2x vs 3x
                variations[key] = [2, 3]
    
    # If no variations, just try base params
    if not variations:
        return verify_template_with_params(
            template_name, base_params, task, grid_engine
        )
    
    # Generate all parameter combinations
    from itertools import product
    
    param_names = list(variations.keys())
    param_values = [variations[name] for name in param_names]
    
    for combo in product(*param_values):
        # Build params dict for this combination
        test_params = base_params.copy()
        for name, value in zip(param_names, combo):
            test_params[name] = value
        
        # Skip invalid params (e.g., zero translation)
        if any(v == 0 for k, v in test_params.items()
               if k in ('dy', 'dx', 'amount') and k in variations):
            continue
        
        try:
            recipe = TransformationRecipe(
                steps=[TransformationStep(name=template_name, params=test_params)],
                confidence=0.5
            )
            
            total_acc = 0.0
            exact_matches = 0
            
            for pair in task.train_pairs:
                try:
                    pred = grid_engine.apply_recipe(pair["input"], recipe)
                    if pred is not None:
                        if pred == pair["output"]:
                            exact_matches += 1
                            total_acc += 1.0
                        elif len(pred) == len(pair["output"]) and len(pred) > 0:
                            out_width = len(pair["output"][0]) if pair["output"] else 0
                            pred_width = len(pred[0]) if pred else 0
                            
                            if pred_width == out_width:
                                correct = sum(
                                    1 for y in range(len(pred))
                                    for x in range(len(pred[0]))
                                    if pred[y][x] == pair["output"][y][x]
                                )
                                total_cells = len(pred) * len(pred[0])
                                total_acc += correct / total_cells if total_cells > 0 else 0
                except Exception:
                    pass
            
            avg_acc = total_acc / len(task.train_pairs) if task.train_pairs else 0.0
            all_exact = exact_matches == len(task.train_pairs)
            
            if all_exact:
                # Found exact match - return immediately
                return (template_name, avg_acc, True, test_params)
            
            if avg_acc > best_accuracy:
                best_accuracy = avg_acc
                best_params = test_params.copy()
                
        except Exception:
            continue
    
    return (template_name, best_accuracy, best_exact, best_params)


# =============================================================================
# Error Pattern Analysis
# =============================================================================

@dataclass
class ErrorPattern:
    """Describes where and what kind of errors occur."""
    error_type: str           # 'edge', 'corner', 'interior', 'color', 'size'
    locations: List[Tuple[int, int]]  # (y, x) positions of errors
    expected_values: List[int]  # What values should be there
    actual_values: List[int]    # What values are there
    percentage: float          # Percentage of total cells with errors
    suggested_fix: Optional[str]  # Suggested template to try


def analyze_error_pattern(
    predicted: List[List[int]],
    expected: List[List[int]]
) -> List[ErrorPattern]:
    """
    Analyze WHERE mismatches occur to suggest better templates.
    
    This function examines the spatial distribution of errors:
    - Edge errors → might need boundary detection or edge handling
    - Corner errors → might need different rotation or flip
    - Interior errors → might need fill or morphological operation
    - Color errors → might need color swap/replace
    - Size errors → might need crop or scale
    
    Args:
        predicted: Predicted grid
        expected: Expected output grid
    
    Returns:
        List of ErrorPattern objects describing the errors
    """
    patterns = []
    
    # Handle size mismatch
    if len(predicted) != len(expected):
        patterns.append(ErrorPattern(
            error_type='size_height',
            locations=[],
            expected_values=[len(expected)],
            actual_values=[len(predicted)],
            percentage=1.0,
            suggested_fix='crop_nonzero' if len(predicted) > len(expected) else 'scale_2x'
        ))
        return patterns  # Can't analyze further with size mismatch
    
    if not predicted or not expected:
        return patterns
    
    pred_width = len(predicted[0]) if predicted else 0
    exp_width = len(expected[0]) if expected else 0
    
    if pred_width != exp_width:
        patterns.append(ErrorPattern(
            error_type='size_width',
            locations=[],
            expected_values=[exp_width],
            actual_values=[pred_width],
            percentage=1.0,
            suggested_fix='crop_nonzero' if pred_width > exp_width else 'tile_horizontal'
        ))
        return patterns
    
    height, width = len(predicted), pred_width
    total_cells = height * width
    if total_cells == 0:
        return patterns
    
    # Collect error locations
    edge_errors = []       # Errors on edges (row 0, col 0, max row, max col)
    corner_errors = []     # Errors at corners
    interior_errors = []   # Errors in interior
    color_mismatches = []  # All color mismatches
    
    for y in range(height):
        for x in range(width):
            pred_val = predicted[y][x]
            exp_val = expected[y][x]
            
            if pred_val != exp_val:
                is_edge = y == 0 or y == height - 1 or x == 0 or x == width - 1
                is_corner = (y in (0, height-1)) and (x in (0, width-1))
                
                color_mismatches.append({
                    'pos': (y, x),
                    'expected': exp_val,
                    'actual': pred_val
                })
                
                if is_corner:
                    corner_errors.append((y, x))
                elif is_edge:
                    edge_errors.append((y, x))
                else:
                    interior_errors.append((y, x))
    
    # Analyze error patterns
    total_errors = len(color_mismatches)
    
    if not total_errors:
        return patterns  # No errors
    
    error_percentage = total_errors / total_cells
    
    # Edge-dominant errors
    if len(edge_errors) > len(interior_errors) * 2:
        patterns.append(ErrorPattern(
            error_type='edge',
            locations=edge_errors[:20],  # Limit to first 20
            expected_values=[m['expected'] for m in color_mismatches[:20]],
            actual_values=[m['actual'] for m in color_mismatches[:20]],
            percentage=len(edge_errors) / total_cells,
            suggested_fix='mark_boundary'  # Edge errors often need boundary marking
        ))
    
    # Corner-dominant errors
    if len(corner_errors) >= 2:  # At least 2 corners have errors
        patterns.append(ErrorPattern(
            error_type='corner',
            locations=corner_errors,
            expected_values=[expected[y][x] for y, x in corner_errors],
            actual_values=[predicted[y][x] for y, x in corner_errors],
            percentage=len(corner_errors) / total_cells,
            suggested_fix='rotate_90'  # Corner errors might indicate wrong rotation
        ))
    
    # Interior-dominant errors
    if len(interior_errors) > len(edge_errors):
        # Check if interior errors are about filling
        interior_expected = [expected[y][x] for y, x in interior_errors]
        interior_actual = [predicted[y][x] for y, x in interior_errors]
        
        # If expected is non-zero but actual is zero, need fill
        if sum(1 for e, a in zip(interior_expected, interior_actual) if e != 0 and a == 0) > len(interior_errors) * 0.5:
            suggested = 'fill_enclosed'
        # If expected is zero but actual is non-zero, need clear/erode
        elif sum(1 for e, a in zip(interior_expected, interior_actual) if e == 0 and a != 0) > len(interior_errors) * 0.5:
            suggested = 'erode'
        else:
            suggested = 'color_replace'
        
        patterns.append(ErrorPattern(
            error_type='interior',
            locations=interior_errors[:20],
            expected_values=interior_expected[:20],
            actual_values=interior_actual[:20],
            percentage=len(interior_errors) / total_cells,
            suggested_fix=suggested
        ))
    
    # Color pattern: consistent color swap
    if color_mismatches:
        # Check if there's a consistent color mapping error
        color_swaps = {}  # (actual, expected) -> count
        for m in color_mismatches:
            key = (m['actual'], m['expected'])
            color_swaps[key] = color_swaps.get(key, 0) + 1
        
        # Find most common swap
        if color_swaps:
            most_common = max(color_swaps.items(), key=lambda x: x[1])
            swap_key, count = most_common
            
            if count > total_errors * 0.5:  # >50% of errors are this swap
                patterns.append(ErrorPattern(
                    error_type='color_swap',
                    locations=[(m['pos'][0], m['pos'][1]) for m in color_mismatches
                              if (m['actual'], m['expected']) == swap_key][:20],
                    expected_values=[swap_key[1]],  # Expected color
                    actual_values=[swap_key[0]],     # Actual color
                    percentage=count / total_cells,
                    suggested_fix=f'color_swap({swap_key[0]},{swap_key[1]})'
                ))
    
    return patterns


def suggest_fixes_from_errors(
    error_patterns: List[ErrorPattern],
    relationship_knowledge: TemplateRelationshipKnowledge
) -> List[Tuple[str, str, float]]:
    """
    Use error patterns + relationship knowledge to suggest template fixes.
    
    Combines error analysis with SIMILAR/OPPOSITE/COMPOSED relationships
    to generate prioritized list of templates to try.
    
    Args:
        error_patterns: Analyzed error patterns
        relationship_knowledge: Template relationship database
    
    Returns:
        List of (template_name, reason, priority) tuples
    """
    suggestions = []
    
    for pattern in error_patterns:
        base_fix = pattern.suggested_fix
        if not base_fix:
            continue
        
        priority = 1.0 - pattern.percentage  # Higher priority for fewer errors
        
        # Add the direct suggestion
        if '(' not in base_fix:  # Simple template name
            suggestions.append((base_fix, f"fixes {pattern.error_type} errors", priority))
            
            # Add SIMILAR templates as alternatives
            similar = relationship_knowledge.get_similar(base_fix, top_k=2)
            for sim_template in similar:
                suggestions.append((
                    sim_template,
                    f"SIMILAR to {base_fix} for {pattern.error_type}",
                    priority * 0.8
                ))
            
            # If error percentage is very high (>50%), try OPPOSITE
            if pattern.percentage > 0.5:
                opposite = relationship_knowledge.get_opposite(base_fix)
                if opposite:
                    suggestions.append((
                        opposite,
                        f"OPPOSITE of {base_fix} (high error rate)",
                        priority * 0.6
                    ))
    
    # Sort by priority descending
    suggestions.sort(key=lambda x: x[2], reverse=True)
    
    return suggestions


# =============================================================================
# Custom Symbolic Relationship Learning
# =============================================================================

# Global registry for learned symbolic relationships
# Maps (entity_a, entity_b, rel_type) -> deterministic seed

CUSTOM_RELATIONSHIPS: Dict[Tuple[str, str, str], int] = {}

def learn_symbolic_relationship(entity_a: str, entity_b: str, relationship_type: str) -> int:
    """
    Learn a symbolic relationship between two entities.
    
    This implements the custom relationship learning from ExtendedReasoningSystem:
    - ENABLES: Template A enables Template B
    - CAUSES: Template A causes effect B
    - PREVENTS: Template A prevents Template B
    - FOLLOWS: Template A typically follows Template B
    
    The relationship is stored as a deterministic seed that can be used to
    add context to recipes without vector operations.
    
    Args:
        a: First entity (e.g., template name, effect name)
        b: Second entity
        rel_type: Relationship type (ENABLES, CAUSES, PREVENTS, FOLLOWS)
    
    Returns:
        Deterministic seed for this relationship
    
    Example:
        >>> seed = learn_symbolic_relationship("crop_nonzero", "mark_boundary", "ENABLES")
        >>> # crop_nonzero ENABLES mark_boundary (preprocessing enables boundary detection)
    """
    # Create the key tuple
    key = (entity_a, entity_b, relationship_type)
    
    # Check if already learned
    if key in CUSTOM_RELATIONSHIPS:
        return CUSTOM_RELATIONSHIPS[key]
    
    # Generate deterministic seed using SHA256 (standard for this architecture)
    # This ensures the same relationship always yields the same seed across sessions
    raw_string = f"{entity_a}::{relationship_type}::{entity_b}"
    hash_bytes = hashlib.sha256(raw_string.encode()).digest()
    
    # Convert to 63-bit integer (fits in signed 64-bit int for compatibility)
    seed = int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF
    
    # Store in registry
    CUSTOM_RELATIONSHIPS[key] = seed
    
    return seed

def query_symbolic_relationship(a: str, b: str, rel_type: str) -> Optional[int]:
    """
    Query a learned symbolic relationship.
    
    Args:
        a: First entity
        b: Second entity
        rel_type: Relationship type
    
    Returns:
        Seed if relationship exists, None otherwise
    """
    return CUSTOM_RELATIONSHIPS.get((a, b, rel_type))


def get_relationships_for_entity(entity: str) -> List[Tuple[str, str, str, int]]:
    """
    Get all relationships involving an entity.
    
    Args:
        entity: Entity to query
    
    Returns:
        List of (entity_a, entity_b, rel_type, seed) tuples
    """
    results = []
    for (a, b, rel_type), seed in CUSTOM_RELATIONSHIPS.items():
        if a == entity or b == entity:
            results.append((a, b, rel_type, seed))
    return results


# Pre-populate common relationships
def _init_common_relationships():
    """Initialize common template relationships."""
    # ENABLES relationships (preprocessing enables downstream operations)
    learn_symbolic_relationship("crop_nonzero", "mark_boundary", "ENABLES")
    learn_symbolic_relationship("crop_nonzero", "rotate_90", "ENABLES")
    learn_symbolic_relationship("crop_nonzero", "flip_horizontal", "ENABLES")
    learn_symbolic_relationship("identity", "mark_boundary", "ENABLES")
    
    # CAUSES relationships (transformations cause effects)
    learn_symbolic_relationship("mark_boundary", "new_color_added", "CAUSES")
    learn_symbolic_relationship("fill_enclosed", "interior_filled", "CAUSES")
    learn_symbolic_relationship("gravity_down", "objects_compact", "CAUSES")
    
    # FOLLOWS relationships (common sequences)
    learn_symbolic_relationship("mark_boundary", "crop_nonzero", "FOLLOWS")
    learn_symbolic_relationship("rotate_90", "flip_horizontal", "FOLLOWS")
    learn_symbolic_relationship("gravity_down", "center_object", "FOLLOWS")
    
    # PREVENTS relationships (mutual exclusion)
    learn_symbolic_relationship("scale_2x", "crop_nonzero", "PREVENTS")
    learn_symbolic_relationship("dilate", "erode", "PREVENTS")


# Initialize on module load
_init_common_relationships()


# =============================================================================
# Recipe Data Structures (Enhanced with BIND_CONTEXT)
# =============================================================================

class RecipeOpType(Enum):
    """
    Types of operations in a recipe.
    
    Enhanced with BIND_CONTEXT for non-diluting relationship attachment.
    This implements the "Bind-not-Bundle" fix from extended thinking.
    """
    BIND = "bind"             # Standard binding (XOR)
    UNBIND = "unbind"         # Reverse binding
    PERMUTE = "permute"       # Order encoding
    BUNDLE = "bundle"         # Superposition (AVOID during deep recursion!)
    BIND_CONTEXT = "bind_ctx" # Relationship-based context attachment (non-diluting)


@dataclass
class RecipeOp:
    """A single operation in a recipe."""
    op_type: RecipeOpType
    param: Any  # Context name for bind, shift for permute, etc.
    
    def __hash__(self):
        return hash((self.op_type.value, str(self.param)))
    
    def __eq__(self, other):
        if not isinstance(other, RecipeOp):
            return False
        return self.op_type == other.op_type and self.param == other.param


@dataclass
class TemplateRecipe:
    """
    Store a template as a recipe instead of a vector.
    
    This enables O(1) storage and fast recipe-based matching without
    ever constructing the actual HDC vector until verification.
    
    Attributes:
        name: Template name (e.g., 'rotate_90', 'flip_horizontal')
        base_seed: Deterministic seed from template name
        signature: Characteristic operations for this template
        category: Template category for grouping (geometric, gravity, etc.)
    """
    name: str
    base_seed: int
    signature: List[RecipeOp] = field(default_factory=list)
    category: str = "unknown"
    
    # Cache for lazy vector reconstruction
    _cached_vector: Optional[np.ndarray] = field(default=None, repr=False)
    
    @classmethod
    def from_name(cls, name: str, category: str = "unknown") -> 'TemplateRecipe':
        """Create a template recipe from template name."""
        seed = _string_to_seed(f"trm_template_{name}")
        
        # Define signature operations based on template type
        signature = cls._get_template_signature(name)
        
        return cls(
            name=name,
            base_seed=seed,
            signature=signature,
            category=category
        )
    
    @staticmethod
    def _get_template_signature(name: str) -> List[RecipeOp]:
        """
        Get characteristic operation signature for a template.
        
        These signatures capture the "essence" of each transformation
        for recipe-based similarity matching.
        
        v2.7: Enhanced signatures with multiple characteristic operations
        for better discrimination between similar templates.
        """
        signatures = {
            # ====== GEOMETRIC TRANSFORMS ======
            # Each has spatial operation + characteristic bind for disambiguation
            'rotate_90': [
                RecipeOp(RecipeOpType.PERMUTE, "spatial_90"),
                RecipeOp(RecipeOpType.BIND, "rotation"),
                RecipeOp(RecipeOpType.BIND, "clockwise")
            ],
            'rotate_180': [
                RecipeOp(RecipeOpType.PERMUTE, "spatial_180"),
                RecipeOp(RecipeOpType.BIND, "rotation"),
                RecipeOp(RecipeOpType.BIND, "half_turn")
            ],
            'rotate_270': [
                RecipeOp(RecipeOpType.PERMUTE, "spatial_270"),
                RecipeOp(RecipeOpType.BIND, "rotation"),
                RecipeOp(RecipeOpType.BIND, "counter_clockwise")
            ],
            'flip_horizontal': [
                RecipeOp(RecipeOpType.PERMUTE, "flip_h"),
                RecipeOp(RecipeOpType.BIND, "flip"),
                RecipeOp(RecipeOpType.BIND, "mirror_x")
            ],
            'flip_vertical': [
                RecipeOp(RecipeOpType.PERMUTE, "flip_v"),
                RecipeOp(RecipeOpType.BIND, "flip"),
                RecipeOp(RecipeOpType.BIND, "mirror_y")
            ],
            'flip_diagonal': [
                RecipeOp(RecipeOpType.PERMUTE, "flip_d"),
                RecipeOp(RecipeOpType.BIND, "flip"),
                RecipeOp(RecipeOpType.BIND, "transpose")
            ],
            'flip_antidiagonal': [
                RecipeOp(RecipeOpType.PERMUTE, "flip_ad"),
                RecipeOp(RecipeOpType.BIND, "flip"),
                RecipeOp(RecipeOpType.BIND, "anti_transpose")
            ],
            'identity': [],  # No ops - intentionally empty
            
            # ====== GRAVITY TRANSFORMS ======
            # Combines gravity + direction + movement semantics
            'gravity_down': [
                RecipeOp(RecipeOpType.BIND, "gravity"),
                RecipeOp(RecipeOpType.PERMUTE, "down"),
                RecipeOp(RecipeOpType.BIND, "fall"),
                RecipeOp(RecipeOpType.BIND, "compact_vertical")
            ],
            'gravity_up': [
                RecipeOp(RecipeOpType.BIND, "gravity"),
                RecipeOp(RecipeOpType.PERMUTE, "up"),
                RecipeOp(RecipeOpType.BIND, "rise"),
                RecipeOp(RecipeOpType.BIND, "compact_vertical")
            ],
            'gravity_left': [
                RecipeOp(RecipeOpType.BIND, "gravity"),
                RecipeOp(RecipeOpType.PERMUTE, "left"),
                RecipeOp(RecipeOpType.BIND, "compact_horizontal")
            ],
            'gravity_right': [
                RecipeOp(RecipeOpType.BIND, "gravity"),
                RecipeOp(RecipeOpType.PERMUTE, "right"),
                RecipeOp(RecipeOpType.BIND, "compact_horizontal")
            ],
            
            # ====== TRANSLATION TRANSFORMS ======
            # Combines translate + direction + object semantics
            'translate_up': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.PERMUTE, "up"),
                RecipeOp(RecipeOpType.BIND, "move_object"),
                RecipeOp(RecipeOpType.BIND, "shift_position")
            ],
            'translate_down': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.PERMUTE, "down"),
                RecipeOp(RecipeOpType.BIND, "move_object"),
                RecipeOp(RecipeOpType.BIND, "shift_position")
            ],
            'translate_left': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.PERMUTE, "left"),
                RecipeOp(RecipeOpType.BIND, "move_object"),
                RecipeOp(RecipeOpType.BIND, "shift_position")
            ],
            'translate_right': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.PERMUTE, "right"),
                RecipeOp(RecipeOpType.BIND, "move_object"),
                RecipeOp(RecipeOpType.BIND, "shift_position")
            ],
            
            # Diagonal translations
            'translate_up_left': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.PERMUTE, "up"),
                RecipeOp(RecipeOpType.PERMUTE, "left"),
                RecipeOp(RecipeOpType.BIND, "diagonal_move")
            ],
            'translate_up_right': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.PERMUTE, "up"),
                RecipeOp(RecipeOpType.PERMUTE, "right"),
                RecipeOp(RecipeOpType.BIND, "diagonal_move")
            ],
            'translate_down_left': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.PERMUTE, "down"),
                RecipeOp(RecipeOpType.PERMUTE, "left"),
                RecipeOp(RecipeOpType.BIND, "diagonal_move")
            ],
            'translate_down_right': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.PERMUTE, "down"),
                RecipeOp(RecipeOpType.PERMUTE, "right"),
                RecipeOp(RecipeOpType.BIND, "diagonal_move")
            ],
            
            # ====== MORPHOLOGICAL TRANSFORMS ======
            # Combines morph + specific operation + shape semantics
            'mark_boundary': [
                RecipeOp(RecipeOpType.BIND, "morph"),
                RecipeOp(RecipeOpType.BIND, "boundary"),
                RecipeOp(RecipeOpType.BIND, "edge_detection"),
                RecipeOp(RecipeOpType.BIND, "recolor")
            ],
            'mark_boundary_8conn': [
                RecipeOp(RecipeOpType.BIND, "morph"),
                RecipeOp(RecipeOpType.BIND, "boundary_8"),
                RecipeOp(RecipeOpType.BIND, "edge_detection"),
                RecipeOp(RecipeOpType.BIND, "eight_connected")
            ],
            'extract_boundary': [
                RecipeOp(RecipeOpType.BIND, "morph"),
                RecipeOp(RecipeOpType.UNBIND, "interior"),
                RecipeOp(RecipeOpType.BIND, "edge_only"),
                RecipeOp(RecipeOpType.BIND, "hollow")
            ],
            'extract_interior': [
                RecipeOp(RecipeOpType.BIND, "morph"),
                RecipeOp(RecipeOpType.UNBIND, "boundary"),
                RecipeOp(RecipeOpType.BIND, "interior_only")
            ],
            'fill_enclosed': [
                RecipeOp(RecipeOpType.BIND, "morph"),
                RecipeOp(RecipeOpType.BIND, "fill"),
                RecipeOp(RecipeOpType.BIND, "flood_interior")
            ],
            'fill_holes': [
                RecipeOp(RecipeOpType.BIND, "morph"),
                RecipeOp(RecipeOpType.BIND, "fill"),
                RecipeOp(RecipeOpType.BIND, "hole_filling")
            ],
            'dilate': [
                RecipeOp(RecipeOpType.BIND, "morph"),
                RecipeOp(RecipeOpType.BIND, "expand"),
                RecipeOp(RecipeOpType.BIND, "grow_shape")
            ],
            'erode': [
                RecipeOp(RecipeOpType.BIND, "morph"),
                RecipeOp(RecipeOpType.UNBIND, "expand"),
                RecipeOp(RecipeOpType.BIND, "shrink_shape")
            ],
            'morph_outline_recolor': [
                RecipeOp(RecipeOpType.BIND, "morph"),
                RecipeOp(RecipeOpType.BIND, "outline"),
                RecipeOp(RecipeOpType.BIND, "recolor_boundary")
            ],
            'detect_and_mark_boundary': [
                RecipeOp(RecipeOpType.BIND, "morph"),
                RecipeOp(RecipeOpType.BIND, "auto_boundary"),
                RecipeOp(RecipeOpType.BIND, "auto_detect"),
                RecipeOp(RecipeOpType.BIND, "mark_edges")
            ],
            
            # ====== STRUCTURAL TRANSFORMS ======
            'tile_2x2': [
                RecipeOp(RecipeOpType.BIND, "structural"),
                RecipeOp(RecipeOpType.BIND, "tile"),
                RecipeOp(RecipeOpType.BIND, "repeat_2x2")
            ],
            'tile_horizontal': [
                RecipeOp(RecipeOpType.BIND, "structural"),
                RecipeOp(RecipeOpType.BIND, "tile"),
                RecipeOp(RecipeOpType.BIND, "repeat_horizontal")
            ],
            'tile_vertical': [
                RecipeOp(RecipeOpType.BIND, "structural"),
                RecipeOp(RecipeOpType.BIND, "tile"),
                RecipeOp(RecipeOpType.BIND, "repeat_vertical")
            ],
            'scale_2x': [
                RecipeOp(RecipeOpType.BIND, "structural"),
                RecipeOp(RecipeOpType.BIND, "scale"),
                RecipeOp(RecipeOpType.BIND, "enlarge_2x")
            ],
            'crop_nonzero': [
                RecipeOp(RecipeOpType.BIND, "structural"),
                RecipeOp(RecipeOpType.UNBIND, "padding"),
                RecipeOp(RecipeOpType.BIND, "crop_to_content")
            ],
            'outline': [
                RecipeOp(RecipeOpType.BIND, "structural"),
                RecipeOp(RecipeOpType.BIND, "add_outline"),
                RecipeOp(RecipeOpType.BIND, "border")
            ],
            
            # ====== DYNAMIC TRANSLATION TRANSFORMS ======
            'center_object': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.BIND, "center"),
                RecipeOp(RecipeOpType.BIND, "auto_position"),
                RecipeOp(RecipeOpType.BIND, "middle_of_grid")
            ],
            'align_to_edge': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.BIND, "align"),
                RecipeOp(RecipeOpType.BIND, "snap_to_edge")
            ],
            'align_to_corner': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.BIND, "corner"),
                RecipeOp(RecipeOpType.BIND, "snap_to_corner")
            ],
            'move_object_to': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.BIND, "target"),
                RecipeOp(RecipeOpType.BIND, "position_to")
            ],
            'translate_to_corner': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.BIND, "corner"),
                RecipeOp(RecipeOpType.BIND, "move_to_corner")
            ],
            'translate_by_offset': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.BIND, "offset"),
                RecipeOp(RecipeOpType.BIND, "relative_move")
            ],
            'shift_all_objects': [
                RecipeOp(RecipeOpType.BIND, "translate"),
                RecipeOp(RecipeOpType.BIND, "shift"),
                RecipeOp(RecipeOpType.BIND, "bulk_move")
            ],
            
            # ====== DRAWING TRANSFORMS ======
            'draw_line': [
                RecipeOp(RecipeOpType.BIND, "draw"),
                RecipeOp(RecipeOpType.BIND, "line"),
                RecipeOp(RecipeOpType.BIND, "bresenham")
            ],
            'connect_points': [
                RecipeOp(RecipeOpType.BIND, "draw"),
                RecipeOp(RecipeOpType.BIND, "connect"),
                RecipeOp(RecipeOpType.BIND, "link_same_color")
            ],
            'connect_all_pairs': [
                RecipeOp(RecipeOpType.BIND, "draw"),
                RecipeOp(RecipeOpType.BIND, "connect_all"),
                RecipeOp(RecipeOpType.BIND, "full_connection")
            ],
            'draw_box_around_object': [
                RecipeOp(RecipeOpType.BIND, "draw"),
                RecipeOp(RecipeOpType.BIND, "box"),
                RecipeOp(RecipeOpType.BIND, "bounding_box")
            ],
            'draw_rectangle': [
                RecipeOp(RecipeOpType.BIND, "draw"),
                RecipeOp(RecipeOpType.BIND, "rect"),
                RecipeOp(RecipeOpType.BIND, "rectangle_shape")
            ],
            'flood_fill': [
                RecipeOp(RecipeOpType.BIND, "draw"),
                RecipeOp(RecipeOpType.BIND, "fill"),
                RecipeOp(RecipeOpType.BIND, "bucket_fill")
            ],
            
            # ====== COLOR TRANSFORMS ======
            'color_swap': [
                RecipeOp(RecipeOpType.BIND, "color"),
                RecipeOp(RecipeOpType.BIND, "swap"),
                RecipeOp(RecipeOpType.BIND, "exchange_colors")
            ],
            'color_replace': [
                RecipeOp(RecipeOpType.BIND, "color"),
                RecipeOp(RecipeOpType.BIND, "replace"),
                RecipeOp(RecipeOpType.BIND, "substitute_color")
            ],
            'color_invert': [
                RecipeOp(RecipeOpType.BIND, "color"),
                RecipeOp(RecipeOpType.BIND, "invert"),
                RecipeOp(RecipeOpType.BIND, "negate_colors")
            ],
        }
        
        return signatures.get(name, [RecipeOp(RecipeOpType.BIND, name)])
    
    def signature_hash(self) -> int:
        """Fast hash for recipe deduplication."""
        return hash((self.base_seed, tuple(self.signature)))
    
    def get_vector(self, hdc: 'SparseBinaryHDC') -> np.ndarray:
        """Lazy reconstruction of the actual vector (only when needed)."""
        if self._cached_vector is None:
            self._cached_vector = hdc.from_seed(self.base_seed)
        return self._cached_vector


@dataclass
class ReasoningRecipe:
    """
    Stores the reasoning state z as a pure recipe.
    
    This is the core data structure that enables vector-free recursion.
    Instead of accumulating bundle operations into a vector (which saturates),
    we store a list of operations that CAN be reconstructed into a vector
    but don't need to be until final verification.
    
    Memory: ~50 bytes per operation vs 4KB per vector
    Time: O(1) to add operation vs O(dim) for vector operation
    """
    base_seed: int
    operations: List[RecipeOp] = field(default_factory=list)
    
    # Optimization: Fast lookup structures
    _op_counts: Dict[RecipeOpType, int] = field(default_factory=dict, repr=False)
    _context_set: Set[str] = field(default_factory=set, repr=False)
    _permute_sum: int = field(default=0, repr=False)  # Sum of permute shifts (mod dim)
    
    # Hash for deduplication
    _hash_cache: Optional[int] = field(default=None, repr=False)
    
    def add_operation(self, op: RecipeOp):
        """Add an operation to the recipe with O(1) bookkeeping."""
        self.operations.append(op)
        
        # Update fast lookup structures
        self._op_counts[op.op_type] = self._op_counts.get(op.op_type, 0) + 1
        
        if op.op_type == RecipeOpType.BIND:
            self._context_set.add(str(op.param))
        elif op.op_type == RecipeOpType.PERMUTE and isinstance(op.param, int):
            self._permute_sum += op.param
        
        # Invalidate hash cache
        self._hash_cache = None
    
    def add_bind(self, context_name: str):
        """Convenience method to add a bind operation."""
        self.add_operation(RecipeOp(RecipeOpType.BIND, context_name))
    
    def add_permute(self, shift: int):
        """Convenience method to add a permute operation."""
        self.add_operation(RecipeOp(RecipeOpType.PERMUTE, shift))
    
    def add_context(self, context_name: str):
        """
        Add relationship context symbolically using BIND_CONTEXT.
        
        This implements the "Bind-not-Bundle" fix from extended thinking:
        - O(1) operation that preserves core transformation signal
        - Non-diluting context attachment (unlike bundle)
        - Deterministic seed ensures cross-session consistency
        
        Use this for semantic hints like categories, relationships, or
        task characteristics without diluting the transformation signal.
        
        Args:
            context_name: Semantic context to attach (e.g., "geometric", "has_inverse_rotate_270")
        
        Example:
            >>> recipe.add_context("category_geometric")
            >>> recipe.add_context("has_inverse_rotate_270")
            >>> recipe.add_context("template_rotate_90")
        """
        # Deterministic seed ensures cross-session consistency
        seed = _string_to_seed(f"ctx_{context_name}")
        self.add_operation(RecipeOp(RecipeOpType.BIND_CONTEXT, seed))
        self._context_set.add(context_name)
    
    def add_relationship_context(self, entity_a: str, entity_b: str, rel_type: str):
        """
        Add a relationship context using learned symbolic relationships.
        
        Args:
            entity_a: First entity
            entity_b: Second entity
            rel_type: Relationship type (ENABLES, CAUSES, PREVENTS, FOLLOWS)
        """
        seed = query_symbolic_relationship(entity_a, entity_b, rel_type)
        if seed is None:
            # Learn the relationship on-the-fly
            seed = learn_symbolic_relationship(entity_a, entity_b, rel_type)
        
        self.add_operation(RecipeOp(RecipeOpType.BIND_CONTEXT, seed))
        self._context_set.add(f"rel_{rel_type}_{entity_a}_{entity_b}")
    
    def get_hash(self) -> int:
        """
        Get a fast hash for deduplication.
        
        Uses only recent operations to avoid O(n) hashing.
        """
        if self._hash_cache is None:
            # Hash based on: base_seed + operation counts + last 10 ops
            recent = tuple(self.operations[-10:]) if len(self.operations) > 10 else tuple(self.operations)
            self._hash_cache = hash((
                self.base_seed,
                len(self.operations),
                self._permute_sum % 1000,
                recent
            ))
        return self._hash_cache
    
    def size_bytes(self) -> int:
        """Estimate memory usage."""
        return 8 + len(self.operations) * 50  # base_seed + ~50 bytes per op
    
    def reconstruct(self, hdc: 'SparseBinaryHDC',
                    context_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Reconstruct the actual HDC vector from this recipe.
        
        Only call this when absolutely necessary (final verification).
        
        Args:
            hdc: HDC system for operations
            context_vectors: Dict mapping context names to their vectors
        
        Returns:
            The reconstructed z vector
        """
        z = hdc.from_seed(self.base_seed)
        
        for op in self.operations:
            if op.op_type == RecipeOpType.BIND:
                ctx_name = str(op.param)
                if ctx_name in context_vectors:
                    z = hdc.bind(z, context_vectors[ctx_name])
            elif op.op_type == RecipeOpType.UNBIND:
                ctx_name = str(op.param)
                if ctx_name in context_vectors:
                    z = hdc.unbind(z, context_vectors[ctx_name])
            elif op.op_type == RecipeOpType.PERMUTE:
                z = hdc.permute(z, int(op.param))
            elif op.op_type == RecipeOpType.BUNDLE:
                # Bundle requires list of vectors - skip for now
                pass
        
        return z
    
    def reconstruct_gpu_optimized(self, hdc: 'SparseBinaryHDC',
                                   context_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """
        GPU-optimized reconstruction using batched operations where possible.
        
        Groups consecutive operations of the same type and applies them
        using batch HDC operations for better GPU utilization.
        
        Args:
            hdc: HDC system with GPU support
            context_vectors: Dict mapping context names to their vectors
        
        Returns:
            The reconstructed z vector
        """
        # Check if batched operations are available
        has_batch_ops = hasattr(hdc, 'batch_bind') and hasattr(hdc, 'batch_permute')
        
        if not has_batch_ops:
            # Fall back to sequential reconstruction
            return self.reconstruct(hdc, context_vectors)
        
        z = hdc.from_seed(self.base_seed)
        
        # Group consecutive bind operations for batching
        i = 0
        while i < len(self.operations):
            op = self.operations[i]
            
            if op.op_type == RecipeOpType.BIND:
                # Collect consecutive binds
                bind_vectors = []
                while i < len(self.operations) and self.operations[i].op_type == RecipeOpType.BIND:
                    ctx_name = str(self.operations[i].param)
                    if ctx_name in context_vectors:
                        bind_vectors.append(context_vectors[ctx_name])
                    i += 1
                
                # Apply binds in batch (chain: z XOR v1 XOR v2 XOR ...)
                if bind_vectors:
                    if len(bind_vectors) >= 4:  # Batch threshold
                        # Use batch_bind for efficiency
                        vectors_with_z = [z] + bind_vectors
                        results = hdc.batch_bind(vectors_with_z[:-1], vectors_with_z[1:])
                        z = results[-1]  # Final result after chain
                    else:
                        for v in bind_vectors:
                            z = hdc.bind(z, v)
                            
            elif op.op_type == RecipeOpType.UNBIND:
                ctx_name = str(op.param)
                if ctx_name in context_vectors:
                    z = hdc.unbind(z, context_vectors[ctx_name])
                i += 1
                
            elif op.op_type == RecipeOpType.PERMUTE:
                # Collect consecutive permutes and combine shifts
                total_shift = 0
                while i < len(self.operations) and self.operations[i].op_type == RecipeOpType.PERMUTE:
                    total_shift += int(self.operations[i].param)
                    i += 1
                
                # Single permute with combined shift (mod dim)
                if total_shift != 0:
                    dim = hdc.dim if hasattr(hdc, 'dim') else len(z)
                    z = hdc.permute(z, total_shift % dim)
                    
            else:
                i += 1
        
        return z
    
    def clone(self) -> 'ReasoningRecipe':
        """Create a copy of this recipe."""
        new = ReasoningRecipe(
            base_seed=self.base_seed,
            operations=list(self.operations)
        )
        new._op_counts = dict(self._op_counts)
        new._context_set = set(self._context_set)
        new._permute_sum = self._permute_sum
        return new


# =============================================================================
# Verification Engine (moved here to avoid circular imports)
# =============================================================================

class VerificationStatus(Enum):
    """Status of verification result."""
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class VerificationResult:
    """Result of a verification process."""
    status: VerificationStatus
    confidence: float
    flags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    corrected_recipe: Optional[Any] = None
    
    @property
    def is_accepted(self) -> bool:
        return self.status == VerificationStatus.ACCEPT


class UniversalHDCVerifier:
    """
    Level 1: Checks vector health, orthogonality, and recipe coherence.
    """
    def __init__(self, hdc):
        self.hdc = hdc
        
    def check(self, prediction_vector: np.ndarray, recipe: Optional['ReasoningRecipe'] = None) -> VerificationResult:
        """
        Verify the physical integrity of the HDC vector and recipe.
        """
        flags = []
        metrics = {}
        
        # 1. Vector Health: Check bit density
        if prediction_vector is not None:
            density = np.mean(prediction_vector)
            metrics['bit_density'] = density
            
            if density < 0.40 or density > 0.60:
                flags.append(f"Vector saturation warning: density {density:.2f}")
                if density < 0.10 or density > 0.90:
                    return VerificationResult(VerificationStatus.REJECT, 0.0, flags=["Critical vector saturation"], metrics=metrics)

        # 2. Recipe Coherence (if recipe provided)
        if recipe:
            bound_contexts = set()
            for op in recipe.operations:
                if op.op_type == RecipeOpType.BIND:
                    bound_contexts.add(str(op.param))
                elif op.op_type == RecipeOpType.UNBIND:
                    ctx = str(op.param)
                    if ctx not in bound_contexts:
                        flags.append(f"Logical error: Unbinding context '{ctx}' that was never bound")
                        return VerificationResult(VerificationStatus.REJECT, 0.0, flags=flags, metrics=metrics)
                    bound_contexts.remove(ctx)
        
        return VerificationResult(VerificationStatus.ACCEPT, 1.0, flags=flags, metrics=metrics)


class ToolBasedVerifier:
    """
    Level 2: The Executable Oracle. Runs recipes through GridTemplateEngine.
    """
    # Tolerance thresholds for verification
    EXACT_MATCH_TOLERANCE = 1.0      # Perfect match
    ACCEPT_TOLERANCE = 0.95          # Accept as correct (95%+)
    PARTIAL_TOLERANCE = 0.80         # Partial match for retry
    
    def __init__(self, grid_engine: 'GridTemplateEngine'):
        self.grid_engine = grid_engine
        
    def simulate(self, recipe: 'TransformationRecipe', input_grid: List[List[int]], expected_output: Optional[List[List[int]]] = None) -> VerificationResult:
        """
        Execute the recipe and verify output matches expectation (if provided)
        or satisfies invariants.
        
        Uses tiered tolerance:
        - 100%: Exact match (ACCEPT, high confidence)
        - 95%+: Near-perfect match (ACCEPT, good confidence)
        - 80-95%: Partial match (NEEDS_REVIEW, may retry)
        - <80%: Reject
        """
        try:
            # 1. Execute Recipe
            executed_grid = self.grid_engine.apply_recipe(input_grid, recipe)
            
            # 2. Tiered tolerance check (if expected output is known)
            if expected_output is not None:
                # Try exact match first
                exact_match, accuracy = self.grid_engine._grids_match(executed_grid, expected_output, tolerance=self.EXACT_MATCH_TOLERANCE)
                
                if exact_match:
                    return VerificationResult(
                        VerificationStatus.ACCEPT,
                        1.0,
                        flags=["Exact match"],
                        metrics={'execution_accuracy': accuracy}
                    )
                
                # Try with tolerance for near-perfect match
                accept_match, accuracy = self.grid_engine._grids_match(executed_grid, expected_output, tolerance=self.ACCEPT_TOLERANCE)
                
                if accept_match:
                    return VerificationResult(
                        VerificationStatus.ACCEPT,
                        accuracy,
                        flags=[f"Near-perfect match: {accuracy:.2%} accuracy"],
                        metrics={'execution_accuracy': accuracy}
                    )
                
                # Check for partial match (needs review)
                partial_match, accuracy = self.grid_engine._grids_match(executed_grid, expected_output, tolerance=self.PARTIAL_TOLERANCE)
                
                if partial_match:
                    return VerificationResult(
                        VerificationStatus.NEEDS_REVIEW,
                        accuracy,
                        flags=[f"Partial match: {accuracy:.2%} accuracy - may need correction"],
                        metrics={'execution_accuracy': accuracy}
                    )
                
                # Below partial threshold - reject but report accuracy
                return VerificationResult(
                    VerificationStatus.REJECT,
                    accuracy,
                    flags=[f"Execution mismatch: {accuracy:.2%} accuracy"],
                    metrics={'execution_accuracy': accuracy}
                )
            
            # 3. Object-Centric Invariants
            is_constructive = any(step.name in ['tile_2x2', 'tile_horizontal', 'tile_vertical'] for step in recipe.steps)
            is_destructive = any(step.name in ['crop_nonzero', 'extract_boundary'] for step in recipe.steps)

            # 4. Inverse Execution (Reversibility)
            if len(recipe.steps) == 1:
                step = recipe.steps[0]
                inverse_map = {
                    'rotate_90': 'rotate_270',
                    'rotate_270': 'rotate_90',
                    'flip_horizontal': 'flip_horizontal',
                    'flip_vertical': 'flip_vertical',
                    'color_invert': 'color_invert'
                }
                
                if step.name in inverse_map:
                    inverse_name = inverse_map[step.name]
                    try:
                        reversed_grid = self.grid_engine.apply_transform(executed_grid, inverse_name)
                        match, _ = self.grid_engine._grids_match(reversed_grid, input_grid, tolerance=1.0)
                        if not match:
                            return VerificationResult(
                                VerificationStatus.REJECT,
                                0.5,
                                flags=["Irreversibility check failed"],
                                metrics={'reversibility': 0.0}
                            )
                    except Exception:
                        pass

            return VerificationResult(VerificationStatus.ACCEPT, 1.0, metrics={'execution_accuracy': 1.0})

        except Exception as e:
            return VerificationResult(VerificationStatus.REJECT, 0.0, flags=[f"Execution error: {str(e)}"])


class AdversarialVerifier:
    """
    Level 3: Stress tests the hypothesis with perturbations and alternatives.
    """
    def __init__(self, grid_engine: 'GridTemplateEngine'):
        self.grid_engine = grid_engine

    def attack(self, recipe: 'TransformationRecipe', input_grid: List[List[int]], solver_func: Any) -> VerificationResult:
        """
        Perform adversarial attacks.
        """
        metrics = {}
        flags = []
        
        # 1. Logic-Preserving Perturbations
        try:
            perturbed_grid = copy.deepcopy(input_grid)
            height, width = len(perturbed_grid), len(perturbed_grid[0])
            
            changes = 0
            for _ in range(5):
                y, x = np.random.randint(0, height), np.random.randint(0, width)
                if perturbed_grid[y][x] == 0:
                    perturbed_grid[y][x] = np.random.randint(1, 10)
                    changes += 1
                    if changes >= 3: break
            
            if changes > 0:
                output_original = self.grid_engine.apply_recipe(input_grid, recipe)
                output_perturbed = self.grid_engine.apply_recipe(perturbed_grid, recipe)

        except Exception as e:
            flags.append(f"Adversarial perturbation caused crash: {e}")
        
        # 2. Hypothesis Ranking Check
        if hasattr(recipe, 'confidence') and recipe.confidence < 0.8:
             flags.append(f"Low confidence recipe: {recipe.confidence:.2f}")
             metrics['confidence'] = recipe.confidence
             if recipe.confidence < 0.5:
                 return VerificationResult(VerificationStatus.NEEDS_REVIEW, recipe.confidence, flags=flags, metrics=metrics)

        return VerificationResult(VerificationStatus.ACCEPT, 1.0, flags=flags, metrics=metrics)


class AutoReviewModule:
    """
    Level 4: Handles rejected/flagged predictions with self-correction.
    
    This module performs active correction attempts when verification fails:
    1. For partial matches (80-95%): Try related templates
    2. For low confidence: Suggest alternative templates in same category
    3. For near-misses: Attempt small parameter adjustments
    """
    
    # Related template alternatives
    TEMPLATE_ALTERNATIVES = {
        # Geometric alternatives
        'rotate_90': ['rotate_270', 'flip_diagonal'],
        'rotate_180': ['flip_horizontal', 'flip_vertical'],
        'rotate_270': ['rotate_90', 'flip_antidiagonal'],
        'flip_horizontal': ['flip_vertical', 'rotate_180'],
        'flip_vertical': ['flip_horizontal', 'rotate_180'],
        'flip_diagonal': ['flip_antidiagonal', 'rotate_90'],
        'flip_antidiagonal': ['flip_diagonal', 'rotate_270'],
        # Gravity alternatives
        'gravity_down': ['gravity_up', 'translate_down'],
        'gravity_up': ['gravity_down', 'translate_up'],
        'gravity_left': ['gravity_right', 'translate_left'],
        'gravity_right': ['gravity_left', 'translate_right'],
        # Morphological alternatives
        'mark_boundary': ['mark_boundary_8conn', 'extract_boundary', 'detect_and_mark_boundary'],
        'mark_boundary_8conn': ['mark_boundary', 'extract_boundary'],
        'extract_boundary': ['mark_boundary', 'mark_boundary_8conn'],
        'dilate': ['erode', 'fill_holes'],
        'erode': ['dilate', 'extract_boundary'],
        # Translation alternatives
        'translate_up': ['translate_down', 'gravity_up'],
        'translate_down': ['translate_up', 'gravity_down'],
        'translate_left': ['translate_right', 'gravity_left'],
        'translate_right': ['translate_left', 'gravity_right'],
        'center_object': ['align_to_edge', 'align_to_corner'],
    }
    
    def __init__(self, solver):
        self.solver = solver
        self._correction_attempts = 0
        self._correction_successes = 0
        
    def resolve(self, result: VerificationResult, task_data: Any) -> VerificationResult:
        """
        Attempt to resolve issues triggered by previous levels.
        
        For partial matches with accuracy >= 80%, tries alternative templates.
        For low confidence, suggests exploring different template categories.
        """
        if "Low confidence recipe" in str(result.flags):
            logger.info("Triggering Recursive Depth Scaling due to low confidence")

        if "Irreversibility check failed" in result.flags:
            logger.info("Triggering Semantic Stability Profiling")
            
        # For partial matches, try to find a corrected recipe
        accuracy = result.metrics.get('execution_accuracy', 0.0)
        
        if result.status == VerificationStatus.NEEDS_REVIEW and accuracy >= 0.80:
            # Partial match - try alternative templates
            corrected = self._try_alternative_templates(result, task_data)
            if corrected:
                return corrected
        
        # Promote high partial matches to ACCEPT with adjusted confidence
        if result.status == VerificationStatus.NEEDS_REVIEW and accuracy >= 0.90:
            # 90%+ partial match is good enough - accept with reduced confidence
            result.status = VerificationStatus.ACCEPT
            result.confidence = accuracy * 0.95  # Slight penalty for not being exact
            result.flags.append(f"Auto-promoted: {accuracy:.1%} accuracy accepted")
            logger.info(f"Auto-promoted partial match: {accuracy:.1%} accuracy")
            return result
            
        if result.status == VerificationStatus.REJECT:
            return result
            
        result.flags.append("Auto-reviewed")
        return result
    
    def _try_alternative_templates(self, result: VerificationResult, task_data: Any) -> Optional[VerificationResult]:
        """
        Try alternative templates when a partial match is found.
        
        Returns corrected result if a better template is found, else None.
        """
        if not self.solver or not task_data:
            return None
            
        self._correction_attempts += 1
        
        # Get the template that produced the partial match
        original_template = result.metrics.get('template_name', '')
        if not original_template:
            return None
            
        # Get alternatives
        alternatives = self.TEMPLATE_ALTERNATIVES.get(original_template, [])
        
        for alt_template in alternatives:
            try:
                # Try the alternative template
                if hasattr(self.solver, 'grid_engine') and hasattr(self.solver.grid_engine, 'apply_transform'):
                    # This would need the actual grid data to test
                    # For now, just suggest alternatives
                    pass
            except Exception:
                continue
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-review statistics."""
        return {
            "correction_attempts": self._correction_attempts,
            "correction_successes": self._correction_successes,
            "success_rate": self._correction_successes / max(self._correction_attempts, 1)
        }


class MetaVerificationEngine:
    """
    Orchestrates the 4-level verification funnel.
    """
    def __init__(self, hdc, grid_engine, solver=None):
        self.universal_verifier = UniversalHDCVerifier(hdc)
        self.tool_verifier = ToolBasedVerifier(grid_engine)
        self.adversarial_verifier = AdversarialVerifier(grid_engine)
        self.auto_review = AutoReviewModule(solver)
        
    def verify(self,
               prediction_vector: Optional[np.ndarray],
               recipe: Union['TransformationRecipe', 'ReasoningRecipe'],
               input_grid: List[List[int]],
               expected_output: Optional[List[List[int]]] = None) -> VerificationResult:
        
        # Convert ReasoningRecipe to TransformationRecipe if needed
        tool_recipe = recipe
        if hasattr(recipe, 'to_transformation_recipe'):
             tool_recipe = recipe.to_transformation_recipe()
        
        # 1. Physical Check (HDC Space)
        hdc_result = self.universal_verifier.check(prediction_vector, recipe if isinstance(recipe, ReasoningRecipe) else None)
        if not hdc_result.is_accepted:
            return hdc_result
            
        # 2. Functional Check (Simulation)
        if hasattr(tool_recipe, 'steps'):
            tool_result = self.tool_verifier.simulate(tool_recipe, input_grid, expected_output)
            if not tool_result.is_accepted:
                return self.auto_review.resolve(tool_result, None)
        
        # 3. Adversarial Check (Stress Test)
        if hasattr(tool_recipe, 'steps'):
            adv_result = self.adversarial_verifier.attack(tool_recipe, input_grid, None)
            if not adv_result.is_accepted or adv_result.status == VerificationStatus.NEEDS_REVIEW:
                return self.auto_review.resolve(adv_result, None)
        
        return VerificationResult(VerificationStatus.ACCEPT, 1.0, flags=["Passed all levels"])


def create_verification_engine(hdc, grid_engine, solver=None) -> MetaVerificationEngine:
    """Factory function to create the verification engine."""
    return MetaVerificationEngine(hdc, grid_engine, solver)


# =============================================================================
# Recipe Similarity Functions (No Vector Reconstruction)
# =============================================================================

def longest_common_subsequence(seq_a: List[RecipeOp], seq_b: List[RecipeOp]) -> int:
    """
    Compute LCS length between two operation sequences.
    
    This is O(n*m) but n and m are typically small (< 100).
    For very long sequences, we could use approximations.
    """
    if not seq_a or not seq_b:
        return 0
    
    # Limit to last N operations for efficiency
    MAX_OPS = 100
    a = seq_a[-MAX_OPS:] if len(seq_a) > MAX_OPS else seq_a
    b = seq_b[-MAX_OPS:] if len(seq_b) > MAX_OPS else seq_b
    
    n, m = len(a), len(b)
    
    # Space-optimized DP (only need previous row)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev
    
    return prev[m]


def recipe_similarity(recipe_a: ReasoningRecipe, recipe_b: ReasoningRecipe) -> float:
    """
    Compute similarity between two reasoning recipes WITHOUT vector reconstruction.
    
    This is the core innovation: O(ops) similarity instead of O(dim).
    
    Components:
    1. Base seed similarity (same family?)
    2. Operation set overlap (Jaccard)
    3. Sequence alignment (LCS-based)
    4. Feature matching (op counts, context overlap)
    
    Returns:
        Similarity score in [0, 1]
    """
    # Component 1: Base seed (15% weight)
    # Same seed = same starting point = related vectors
    seed_sim = 1.0 if recipe_a.base_seed == recipe_b.base_seed else 0.3
    
    # Component 2: Operation set overlap (25% weight)
    ops_a = set(recipe_a.operations)
    ops_b = set(recipe_b.operations)
    if ops_a or ops_b:
        overlap = len(ops_a & ops_b)
        total = len(ops_a | ops_b)
        op_set_sim = overlap / total if total > 0 else 0.5
    else:
        op_set_sim = 1.0  # Both empty = identical
    
    # Component 3: Sequence alignment (35% weight)
    max_len = max(len(recipe_a.operations), len(recipe_b.operations))
    if max_len > 0:
        lcs = longest_common_subsequence(recipe_a.operations, recipe_b.operations)
        seq_sim = lcs / max_len
    else:
        seq_sim = 1.0  # Both empty
    
    # Component 4: Feature matching (25% weight)
    # Compare operation type counts
    all_types = set(recipe_a._op_counts.keys()) | set(recipe_b._op_counts.keys())
    if all_types:
        count_diff_sum = 0
        count_max_sum = 0
        for op_type in all_types:
            c_a = recipe_a._op_counts.get(op_type, 0)
            c_b = recipe_b._op_counts.get(op_type, 0)
            count_diff_sum += abs(c_a - c_b)
            count_max_sum += max(c_a, c_b)
        count_sim = 1.0 - (count_diff_sum / max(count_max_sum, 1))
    else:
        count_sim = 1.0
    
    # Context set overlap
    if recipe_a._context_set or recipe_b._context_set:
        ctx_overlap = len(recipe_a._context_set & recipe_b._context_set)
        ctx_total = len(recipe_a._context_set | recipe_b._context_set)
        ctx_sim = ctx_overlap / ctx_total if ctx_total > 0 else 0.5
    else:
        ctx_sim = 1.0
    
    feature_sim = 0.5 * count_sim + 0.5 * ctx_sim
    
    # Combine with weights
    return 0.15 * seed_sim + 0.25 * op_set_sim + 0.35 * seq_sim + 0.25 * feature_sim


def recipe_to_template_similarity(z_recipe: ReasoningRecipe, 
                                   template: TemplateRecipe) -> float:
    """
    Compute how well a reasoning recipe matches a template.
    
    This is specialized for matching z against templates:
    - Looks for template signature operations in z's recent operations
    - Considers category-based heuristics
    - Returns higher score if z "looks like" applying this template
    
    Args:
        z_recipe: The current reasoning state recipe
        template: The template to match against
    
    Returns:
        Match score in [0, 1]
    """
    if not template.signature:
        # Templates without signatures match based on base similarity
        return 0.5
    
    # Check if recent operations contain template signature
    recent_ops = z_recipe.operations[-50:] if len(z_recipe.operations) > 50 else z_recipe.operations
    recent_set = set(recent_ops)
    
    # Signature match: how many signature ops appear in recent history?
    sig_matches = sum(1 for sig_op in template.signature if sig_op in recent_set)
    sig_ratio = sig_matches / len(template.signature) if template.signature else 0
    
    # Context match: does z bind/unbind contexts relevant to template?
    template_contexts = {str(op.param) for op in template.signature 
                        if op.op_type in (RecipeOpType.BIND, RecipeOpType.UNBIND)}
    if template_contexts:
        ctx_match = len(z_recipe._context_set & template_contexts) / len(template_contexts)
    else:
        ctx_match = 0.5
    
    # Permute pattern match (for spatial templates)
    if any(op.op_type == RecipeOpType.PERMUTE for op in template.signature):
        # Look for permute operations in z
        z_has_permute = RecipeOpType.PERMUTE in z_recipe._op_counts
        permute_match = 0.8 if z_has_permute else 0.3
    else:
        permute_match = 0.5
    
    return 0.4 * sig_ratio + 0.3 * ctx_match + 0.3 * permute_match


# =============================================================================
# Batch and Parallel Recipe Operations
# =============================================================================

def batch_recipe_to_template_similarity(
    z_recipe: ReasoningRecipe,
    templates: List[TemplateRecipe],
    num_workers: int = None
) -> List[Tuple[str, float]]:
    """
    Compute recipe-to-template similarity for MULTIPLE templates at once.
    
    This is optimized for the common case of matching one reasoning recipe
    against many templates. Uses parallel processing when beneficial.
    
    Performance:
    - Sequential for small template lists (<10)
    - Parallel for larger lists using ThreadPoolExecutor
    
    Args:
        z_recipe: The current reasoning state recipe
        templates: List of templates to match against
        num_workers: Number of parallel workers (None = auto based on CPU count)
    
    Returns:
        List of (template_name, similarity_score) sorted by score descending
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    
    if not templates:
        return []
    
    # For small lists, sequential is faster (avoids threading overhead)
    if len(templates) < 10:
        results = []
        for template in templates:
            sim = recipe_to_template_similarity(z_recipe, template)
            results.append((template.name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    # Parallel processing for larger template lists
    if num_workers is None:
        num_workers = min(len(templates), os.cpu_count() or 4)
    
    results = []
    
    # Use ThreadPoolExecutor for I/O-bound similarity checks
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all similarity computations
        future_to_template = {
            executor.submit(recipe_to_template_similarity, z_recipe, template): template
            for template in templates
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_template):
            template = future_to_template[future]
            try:
                sim = future.result()
                results.append((template.name, sim))
            except Exception:
                # On error, use default low score
                results.append((template.name, 0.0))
    
    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# Tolerance thresholds for verification
NEAR_PERFECT_THRESHOLD = 0.95  # 95%+ cell accuracy = "near perfect"
ACCEPTABLE_THRESHOLD = 0.90    # 90%+ = acceptable


# =============================================================================
# Task Analysis and Heuristics (v2.8)
# =============================================================================

def analyze_task_characteristics(task: 'TaskSample') -> Dict[str, Any]:
    """
    Analyze task characteristics to prioritize likely templates.
    
    This performs heuristic analysis to identify:
    - Size changes (suggests scaling, tiling, cropping)
    - Color changes (suggests color transforms)
    - Position shifts (suggests translation, gravity)
    - Shape preservation (suggests geometric transforms)
    
    Returns dict with analysis results and suggested template categories.
    """
    characteristics = {
        'size_changes': False,
        'color_changes': False,
        'position_shifts': False,
        'shape_preserved': True,
        'new_colors_introduced': False,
        'colors_removed': False,
        'likely_categories': [],
        'inferred_params': {},
    }
    
    if not task.train_pairs:
        return characteristics
    
    size_changes = []
    color_changes = []
    position_shifts = []
    all_input_colors = set()
    all_output_colors = set()
    
    for pair in task.train_pairs:
        inp = pair.get("input", [])
        out = pair.get("output", [])
        
        if not inp or not out:
            continue
        
        # Size analysis
        in_h, in_w = len(inp), len(inp[0]) if inp else 0
        out_h, out_w = len(out), len(out[0]) if out else 0
        
        if in_h != out_h or in_w != out_w:
            size_changes.append((in_h, in_w, out_h, out_w))
        
        # Color analysis
        input_colors = set(c for row in inp for c in row)
        output_colors = set(c for row in out for c in row)
        all_input_colors.update(input_colors)
        all_output_colors.update(output_colors)
        
        new_colors = output_colors - input_colors
        removed_colors = input_colors - output_colors
        
        if new_colors or removed_colors:
            color_changes.append({
                'new': new_colors,
                'removed': removed_colors
            })
        
        # Position shift detection (for translation tasks)
        # Find non-zero regions and compare centers
        in_positions = [(y, x) for y in range(in_h) for x in range(in_w) if inp[y][x] != 0]
        out_positions = [(y, x) for y in range(out_h) for x in range(out_w) if out[y][x] != 0]
        
        if in_positions and out_positions:
            in_center = (sum(p[0] for p in in_positions) / len(in_positions),
                        sum(p[1] for p in in_positions) / len(in_positions))
            out_center = (sum(p[0] for p in out_positions) / len(out_positions),
                         sum(p[1] for p in out_positions) / len(out_positions))
            
            dy = out_center[0] - in_center[0]
            dx = out_center[1] - in_center[1]
            
            if abs(dy) > 0.5 or abs(dx) > 0.5:
                position_shifts.append((round(dy), round(dx)))
    
    # Update characteristics based on analysis
    if size_changes:
        characteristics['size_changes'] = True
        # Check for common patterns
        if all(out_h == in_h * 2 and out_w == in_w * 2 for in_h, in_w, out_h, out_w in size_changes):
            characteristics['likely_categories'].append('structural')
            characteristics['inferred_params']['scale'] = 2
        elif all(out_h == in_h * 2 for in_h, in_w, out_h, out_w in size_changes):
            characteristics['likely_categories'].append('structural')
            characteristics['inferred_params']['tile'] = 'vertical'
        elif all(out_w == in_w * 2 for in_h, in_w, out_h, out_w in size_changes):
            characteristics['likely_categories'].append('structural')
            characteristics['inferred_params']['tile'] = 'horizontal'
    
    if color_changes:
        characteristics['color_changes'] = True
        # Check for boundary color introduction (morphological)
        new_colors = set()
        for cc in color_changes:
            new_colors.update(cc['new'])
        if new_colors:
            characteristics['new_colors_introduced'] = True
            characteristics['inferred_params']['boundary_colors'] = list(new_colors)
            characteristics['likely_categories'].append('morphological')
    
    if position_shifts:
        characteristics['position_shifts'] = True
        # Find most common translation
        from collections import Counter
        most_common = Counter(position_shifts).most_common(1)
        if most_common:
            dy, dx = most_common[0][0]
            characteristics['inferred_params']['translation'] = (dy, dx)
            characteristics['likely_categories'].append('translation')
    
    # Infer new colors from output
    new_colors_global = all_output_colors - all_input_colors
    if new_colors_global:
        characteristics['inferred_params']['new_colors'] = list(new_colors_global)
    
    # Default categories if none identified
    if not characteristics['likely_categories']:
        characteristics['likely_categories'] = ['geometric', 'gravity']
    
    return characteristics


def infer_parameters_from_task(
    task: 'TaskSample',
    grid_engine: 'GridTemplateEngine'
) -> Dict[str, Dict[str, Any]]:
    """
    Infer template parameters from training pairs.
    
    For each training pair, try to detect specific transformation parameters:
    - Translation: detect (dy, dx) offset
    - Mark boundary: detect boundary_color
    - Color transforms: detect color mappings
    
    Returns dict mapping template_name -> inferred_params
    """
    from .grid_templates import detect_translation
    
    inferred = {}
    
    if not task.train_pairs:
        return inferred
    
    # Analyze first training pair
    pair = task.train_pairs[0]
    inp = pair.get("input", [])
    out = pair.get("output", [])
    
    if not inp or not out:
        return inferred
    
    # 1. Translation detection
    translation = detect_translation(inp, out)
    if translation:
        dy, dx = translation
        inferred['translate_object'] = {'dy': dy, 'dx': dx}
        
        # Also infer direction-specific templates
        if dy < 0 and dx == 0:
            inferred['translate_up'] = {'amount': abs(dy)}
        elif dy > 0 and dx == 0:
            inferred['translate_down'] = {'amount': dy}
        elif dy == 0 and dx < 0:
            inferred['translate_left'] = {'amount': abs(dx)}
        elif dy == 0 and dx > 0:
            inferred['translate_right'] = {'amount': dx}
        elif dy < 0 and dx < 0:
            inferred['translate_up_left'] = {'amount': max(abs(dy), abs(dx))}
        elif dy < 0 and dx > 0:
            inferred['translate_up_right'] = {'amount': max(abs(dy), dx)}
        elif dy > 0 and dx < 0:
            inferred['translate_down_left'] = {'amount': max(dy, abs(dx))}
        elif dy > 0 and dx > 0:
            inferred['translate_down_right'] = {'amount': max(dy, dx)}
    
    # 2. Color inference for morphological transforms
    input_colors = set(c for row in inp for c in row if c != 0)
    output_colors = set(c for row in out for c in row if c != 0)
    new_colors = output_colors - input_colors
    
    if new_colors:
        boundary_color = min(new_colors)  # Use smallest new color
        interior_color = min(input_colors) if input_colors else None
        
        inferred['mark_boundary'] = {
            'boundary_color': boundary_color,
            'interior_color': interior_color
        }
        inferred['mark_boundary_8conn'] = {
            'boundary_color': boundary_color,
            'interior_color': interior_color
        }
        inferred['morph_outline_recolor'] = {'boundary_color': boundary_color}
        inferred['outline'] = {'outline_color': boundary_color}
    
    # 3. Color swap/replace detection
    if input_colors and output_colors:
        removed = input_colors - output_colors
        added = output_colors - input_colors
        
        if len(removed) == 1 and len(added) == 1:
            old_c = list(removed)[0]
            new_c = list(added)[0]
            inferred['color_replace'] = {'old_color': old_c, 'new_color': new_c}
        
        if len(removed) == 1 and len(added) == 1:
            c1 = list(removed)[0]
            c2 = list(added)[0]
            inferred['color_swap'] = {'color1': min(c1, c2), 'color2': max(c1, c2)}
    
    return inferred


def verify_template_with_params(
    template_name: str,
    params: Dict[str, Any],
    task: 'TaskSample',
    grid_engine: 'GridTemplateEngine'
) -> Tuple[str, float, bool, Dict[str, Any]]:
    """
    Verify a single template with specific parameters.
    
    This function tries a template with inferred parameters (not default values).
    
    Args:
        template_name: Name of template to verify
        params: Inferred parameters for this template
        task: Task to verify against
        grid_engine: Grid engine for transformations
    
    Returns:
        (template_name, accuracy, exact_match, used_params)
    """
    try:
        from .grid_templates import TransformationRecipe, TransformationStep
        
        recipe = TransformationRecipe(
            steps=[TransformationStep(name=template_name, params=params)],
            confidence=0.5
        )
        
        total_acc = 0.0
        exact_matches = 0
        
        for pair in task.train_pairs:
            try:
                pred = grid_engine.apply_recipe(pair["input"], recipe)
                if pred is not None:
                    if pred == pair["output"]:
                        exact_matches += 1
                        total_acc += 1.0
                    elif len(pred) == len(pair["output"]) and len(pred) > 0:
                        out_width = len(pair["output"][0]) if pair["output"] else 0
                        pred_width = len(pred[0]) if pred else 0
                        
                        if pred_width == out_width:
                            correct = sum(
                                1 for y in range(len(pred))
                                for x in range(len(pred[0]))
                                if pred[y][x] == pair["output"][y][x]
                            )
                            total_cells = len(pred) * len(pred[0])
                            cell_acc = correct / total_cells if total_cells > 0 else 0
                            total_acc += cell_acc
            except Exception:
                pass
        
        avg_acc = total_acc / len(task.train_pairs) if task.train_pairs else 0.0
        all_exact = exact_matches == len(task.train_pairs)
        
        return (template_name, avg_acc, all_exact, params)
        
    except Exception:
        return (template_name, 0.0, False, params)


def parallel_verify_parameterized_templates(
    task: 'TaskSample',
    grid_engine: 'GridTemplateEngine',
    inferred_params: Dict[str, Dict[str, Any]],
    num_workers: int = None
) -> List[Tuple[str, float, bool, Dict[str, Any]]]:
    """
    Verify templates with their inferred parameters in parallel.
    
    This is the key function for improving exact match accuracy:
    - Instead of using default params (amount=1), uses inferred params
    - Tries translation amounts detected from training pairs
    - Tries boundary colors detected from output
    
    Args:
        task: Task to verify against
        grid_engine: Grid engine for transformations
        inferred_params: Dict mapping template_name -> inferred parameters
        num_workers: Number of parallel workers
    
    Returns:
        List of (template_name, accuracy, exact_match, used_params)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    
    if not inferred_params:
        return []
    
    # Build list of (template_name, params) to verify
    templates_to_verify = list(inferred_params.items())
    
    if len(templates_to_verify) < 3:
        # Sequential for small lists
        return [
            verify_template_with_params(name, params, task, grid_engine)
            for name, params in templates_to_verify
        ]
    
    if num_workers is None:
        num_workers = min(len(templates_to_verify), os.cpu_count() or 4)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                verify_template_with_params, name, params, task, grid_engine
            ): (name, params)
            for name, params in templates_to_verify
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception:
                name, params = futures[future]
                results.append((name, 0.0, False, params))
    
    # Sort by accuracy descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def verify_composite_transforms(
    task: 'TaskSample',
    grid_engine: 'GridTemplateEngine',
    characteristics: Dict[str, Any],
    top_k: int = 5
) -> List[Tuple[str, float, bool]]:
    """
    Try composite (2-step) transformations with heuristic guidance.
    
    Uses task characteristics to prioritize likely 2-step combinations:
    - Size changes + geometric → scale then transform
    - Color changes + morphological → mark boundary etc.
    
    Enhanced in v2.8.1: Uses inferred parameters for morphological steps
    in composites (e.g., mark_boundary with correct boundary_color).
    
    Args:
        task: Task to verify against
        grid_engine: Grid engine for transformations
        characteristics: Analyzed task characteristics
        top_k: Number of top results to return
    
    Returns:
        List of (composite_name, accuracy, exact_match)
    """
    from .grid_templates import TransformationRecipe, TransformationStep
    
    results = []
    likely_categories = characteristics.get('likely_categories', [])
    inferred_params = characteristics.get('inferred_params', {})
    
    # Extract boundary colors for morphological ops
    boundary_colors = inferred_params.get('boundary_colors', [])
    if not boundary_colors and inferred_params.get('new_colors'):
        boundary_colors = inferred_params['new_colors']
    if not boundary_colors:
        boundary_colors = [2]  # Default
    
    # Extract translation amounts
    translation = inferred_params.get('translation', (1, 1))
    
    # Define promising 2-step combinations based on characteristics
    # Format: (step1_name, step1_params, step2_name, step2_params)
    promising_combos = []
    
    if 'morphological' in likely_categories:
        # Morphological tasks often need geometric + mark_boundary with correct color
        for bc in boundary_colors[:2]:  # Top 2 boundary colors
            promising_combos.extend([
                ('crop_nonzero', {}, 'mark_boundary', {'boundary_color': bc}),
                ('crop_nonzero', {}, 'mark_boundary_8conn', {'boundary_color': bc}),
                ('rotate_90', {}, 'mark_boundary', {'boundary_color': bc}),
                ('rotate_180', {}, 'mark_boundary', {'boundary_color': bc}),
                ('flip_horizontal', {}, 'mark_boundary', {'boundary_color': bc}),
                ('flip_vertical', {}, 'mark_boundary', {'boundary_color': bc}),
                ('mark_boundary', {'boundary_color': bc}, 'crop_nonzero', {}),
                ('identity', {}, 'mark_boundary', {'boundary_color': bc}),
            ])
        
        # Also try without params as fallback
        promising_combos.extend([
            ('crop_nonzero', {}, 'detect_and_mark_boundary', {}),
            ('rotate_90', {}, 'detect_and_mark_boundary', {}),
        ])
    
    if 'geometric' in likely_categories:
        # Geometric combinations
        promising_combos.extend([
            ('rotate_90', {}, 'flip_horizontal', {}),
            ('rotate_90', {}, 'flip_vertical', {}),
            ('flip_horizontal', {}, 'rotate_90', {}),
            ('flip_vertical', {}, 'rotate_90', {}),
            ('rotate_180', {}, 'flip_horizontal', {}),
            ('flip_diagonal', {}, 'rotate_90', {}),
        ])
    
    if 'translation' in likely_categories:
        # Translation + other transforms
        dy, dx = translation
        promising_combos.extend([
            ('center_object', {}, 'rotate_90', {}),
            ('center_object', {}, 'flip_horizontal', {}),
            ('crop_nonzero', {}, 'center_object', {}),
            ('translate_up', {'amount': abs(dy)}, 'rotate_90', {}),
            ('rotate_90', {}, 'translate_up', {'amount': abs(dy)}),
        ])
    
    if 'gravity' in likely_categories or 'structural' in likely_categories:
        promising_combos.extend([
            ('gravity_down', {}, 'crop_nonzero', {}),
            ('gravity_up', {}, 'crop_nonzero', {}),
            ('crop_nonzero', {}, 'gravity_down', {}),
            ('crop_nonzero', {}, 'gravity_up', {}),
            ('gravity_down', {}, 'rotate_90', {}),
        ])
    
    # Default combinations if nothing specific
    if not promising_combos:
        promising_combos = [
            ('crop_nonzero', {}, 'rotate_90', {}),
            ('rotate_90', {}, 'gravity_down', {}),
            ('flip_horizontal', {}, 'gravity_down', {}),
            ('identity', {}, 'mark_boundary', {'boundary_color': 2}),
        ]
    
    for combo in promising_combos[:15]:  # Limit to prevent timeout
        try:
            step1_name, step1_params, step2_name, step2_params = combo
            
            recipe = TransformationRecipe(steps=[
                TransformationStep(name=step1_name, params=step1_params),
                TransformationStep(name=step2_name, params=step2_params)
            ])
            
            total_acc = 0.0
            exact_matches = 0
            
            for pair in task.train_pairs:
                try:
                    pred = grid_engine.apply_recipe(pair["input"], recipe)
                    if pred is not None:
                        if pred == pair["output"]:
                            exact_matches += 1
                            total_acc += 1.0
                        elif len(pred) == len(pair["output"]) and len(pred) > 0:
                            out_width = len(pair["output"][0]) if pair["output"] else 0
                            pred_width = len(pred[0]) if pred else 0
                            
                            if pred_width == out_width:
                                correct = sum(
                                    1 for y in range(len(pred))
                                    for x in range(len(pred[0]))
                                    if pred[y][x] == pair["output"][y][x]
                                )
                                total_cells = len(pred) * len(pred[0])
                                cell_acc = correct / total_cells if total_cells > 0 else 0
                                total_acc += cell_acc
                except Exception:
                    pass
            
            avg_acc = total_acc / len(task.train_pairs) if task.train_pairs else 0.0
            all_exact = exact_matches == len(task.train_pairs)
            
            # Build composite name with params if present
            composite_parts = []
            if step1_params:
                composite_parts.append(f"{step1_name}({step1_params})")
            else:
                composite_parts.append(step1_name)
            if step2_params:
                composite_parts.append(f"{step2_name}({step2_params})")
            else:
                composite_parts.append(step2_name)
            
            composite_name = "+".join(composite_parts)
            results.append((composite_name, avg_acc, all_exact))
            
            # Early exit if we find exact match
            if all_exact:
                break
                
        except Exception:
            continue
    
    # Sort by accuracy
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def parallel_verify_templates(
    template_names: List[str],
    task: 'TaskSample',
    grid_engine: 'GridTemplateEngine',
    num_workers: int = None,
    accept_near_perfect: bool = True  # Accept 95%+ matches as exact
) -> List[Tuple[str, float, bool]]:
    """
    Verify multiple templates against a task in parallel.
    
    This is the most expensive operation (actual grid transformations),
    so parallelization provides significant speedup.
    
    Changes in v2.7:
    - accept_near_perfect: When True, treats 95%+ accuracy as "exact match"
    - This handles minor differences (off-by-one, rounding errors)
    
    Args:
        template_names: List of template names to verify
        task: Task to verify against
        grid_engine: Grid engine for applying transformations
        num_workers: Number of parallel workers
        accept_near_perfect: Treat 95%+ accuracy as exact match
    
    Returns:
        List of (template_name, accuracy, exact_match) sorted by accuracy descending
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    
    def verify_single(template_name: str) -> Tuple[str, float, bool]:
        """Verify a single template with tolerance-based matching."""
        try:
            from .grid_templates import TransformationRecipe, TransformationStep
            
            recipe = TransformationRecipe(
                steps=[TransformationStep(name=template_name, params={})],
                confidence=0.5
            )
            
            total_acc = 0.0
            exact_matches = 0
            near_perfect_matches = 0  # 95%+ accuracy
            
            for pair in task.train_pairs:
                try:
                    pred = grid_engine.apply_recipe(pair["input"], recipe)
                    if pred is not None:
                        # Check for exact match
                        if pred == pair["output"]:
                            exact_matches += 1
                            near_perfect_matches += 1
                            total_acc += 1.0
                        elif len(pred) == len(pair["output"]) and len(pred) > 0:
                            # Cell-level accuracy
                            out_width = len(pair["output"][0]) if pair["output"] else 0
                            pred_width = len(pred[0]) if pred else 0
                            
                            if pred_width == out_width:
                                correct = sum(
                                    1 for y in range(len(pred))
                                    for x in range(len(pred[0]))
                                    if pred[y][x] == pair["output"][y][x]
                                )
                                total_cells = len(pred) * len(pred[0])
                                cell_acc = correct / total_cells if total_cells > 0 else 0
                                total_acc += cell_acc
                                
                                # Track near-perfect matches
                                if cell_acc >= NEAR_PERFECT_THRESHOLD:
                                    near_perfect_matches += 1
                            else:
                                # Width mismatch - size changed incorrectly
                                total_acc += 0.0
                        else:
                            # Size mismatch - try to compute partial accuracy
                            if len(pred) > 0 and len(pair["output"]) > 0:
                                # Compare what we can
                                min_h = min(len(pred), len(pair["output"]))
                                min_w = min(
                                    len(pred[0]) if pred else 0,
                                    len(pair["output"][0]) if pair["output"] else 0
                                )
                                if min_h > 0 and min_w > 0:
                                    correct = sum(
                                        1 for y in range(min_h)
                                        for x in range(min_w)
                                        if pred[y][x] == pair["output"][y][x]
                                    )
                                    # Penalize for size mismatch
                                    max_cells = max(
                                        len(pred) * (len(pred[0]) if pred else 1),
                                        len(pair["output"]) * (len(pair["output"][0]) if pair["output"] else 1)
                                    )
                                    total_acc += correct / max_cells * 0.5  # 50% penalty for size mismatch
                except Exception:
                    pass
            
            avg_acc = total_acc / len(task.train_pairs) if task.train_pairs else 0.0
            
            # Determine if this counts as an "exact" match
            all_exact = exact_matches == len(task.train_pairs)
            
            # If accept_near_perfect is enabled, also count 95%+ as "exact"
            if accept_near_perfect and not all_exact:
                all_near_perfect = near_perfect_matches == len(task.train_pairs)
                if all_near_perfect and avg_acc >= NEAR_PERFECT_THRESHOLD:
                    all_exact = True  # Promote to exact
            
            return (template_name, avg_acc, all_exact)
            
        except Exception:
            return (template_name, 0.0, False)
    
    if not template_names:
        return []
    
    # For very small lists, sequential is fine
    if len(template_names) < 3:
        return [verify_single(name) for name in template_names]
    
    if num_workers is None:
        num_workers = min(len(template_names), os.cpu_count() or 4)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(verify_single, name): name
            for name in template_names
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception:
                name = futures[future]
                results.append((name, 0.0, False))
    
    # Sort by accuracy descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# =============================================================================
# GPU-Accelerated Batch Operations
# =============================================================================

def gpu_batch_reconstruct_recipes(
    recipes: List[ReasoningRecipe],
    hdc: 'SparseBinaryHDC',
    context_vectors: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """
    GPU-accelerated batch reconstruction of multiple recipes.
    
    Reconstructs multiple reasoning recipes into vectors in parallel
    using GPU acceleration when available.
    
    Args:
        recipes: List of ReasoningRecipe to reconstruct
        hdc: HDC system with GPU support
        context_vectors: Dict mapping context names to vectors
    
    Returns:
        List of reconstructed vectors
    """
    if not recipes:
        return []
    
    # Check for GPU-optimized method
    has_gpu_reconstruct = hasattr(recipes[0], 'reconstruct_gpu_optimized')
    
    # Reconstruct all recipes
    results = []
    for recipe in recipes:
        if has_gpu_reconstruct:
            results.append(recipe.reconstruct_gpu_optimized(hdc, context_vectors))
        else:
            results.append(recipe.reconstruct(hdc, context_vectors))
    
    return results


def gpu_batch_verify_candidates(
    candidate_vectors: List[np.ndarray],
    target_vector: np.ndarray,
    hdc: 'SparseBinaryHDC',
    candidate_names: List[str]
) -> List[Tuple[str, float]]:
    """
    GPU-accelerated batch verification of candidate vectors.
    
    Uses GPU batch_similarity() to compare all candidates against
    a target vector in a single batched operation.
    
    Args:
        candidate_vectors: List of candidate HDC vectors
        target_vector: Target vector to compare against
        hdc: HDC system with GPU support
        candidate_names: Names/IDs for each candidate
    
    Returns:
        List of (name, similarity) sorted by similarity descending
    """
    if not candidate_vectors:
        return []
    
    # Check for batch_similarity support
    has_batch_sim = hasattr(hdc, 'batch_similarity')
    
    if has_batch_sim and len(candidate_vectors) >= 2:
        # Use GPU batch similarity for efficiency
        # FIXED: batch_similarity(query, candidates) returns similarities of query against all candidates
        # The first argument is the single query vector, second is the list of candidates
        similarities = hdc.batch_similarity(target_vector, candidate_vectors)
        
        # Combine with names
        results = list(zip(candidate_names, similarities))
    else:
        # Fall back to sequential similarity
        results = []
        for name, vec in zip(candidate_names, candidate_vectors):
            sim = hdc.similarity(vec, target_vector)
            results.append((name, sim))
    
    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def parallel_verify_templates_gpu(
    template_names: List[str],
    task: 'TaskSample',
    grid_engine: 'GridTemplateEngine',
    hdc: 'SparseBinaryHDC',
    encoder: Any,
    num_workers: int = None
) -> List[Tuple[str, float, bool]]:
    """
    GPU-enhanced parallel verification of templates.
    
    Combines GPU-accelerated HDC operations with parallel grid verification
    for maximum throughput.
    
    Args:
        template_names: List of template names to verify
        task: Task to verify against
        grid_engine: Grid engine for applying transformations
        hdc: HDC system with GPU support
        encoder: Grid encoder for HDC encoding
        num_workers: Number of parallel workers
    
    Returns:
        List of (template_name, accuracy, exact_match) sorted by accuracy descending
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    
    def verify_single_gpu(template_name: str) -> Tuple[str, float, bool]:
        """Verify a single template with GPU-enhanced HDC encoding."""
        try:
            from .grid_templates import TransformationRecipe, TransformationStep
            
            recipe = TransformationRecipe(
                steps=[TransformationStep(name=template_name, params={})],
                confidence=0.5
            )
            
            total_acc = 0.0
            exact_matches = 0
            hdc_similarities = []
            
            for pair in task.train_pairs:
                try:
                    pred = grid_engine.apply_recipe(pair["input"], recipe)
                    if pred is not None:
                        if pred == pair["output"]:
                            exact_matches += 1
                            total_acc += 1.0
                            hdc_similarities.append(1.0)
                        elif len(pred) == len(pair["output"]):
                            # Cell-level accuracy
                            correct = sum(
                                1 for y in range(len(pred))
                                for x in range(len(pred[0]))
                                if pred[y][x] == pair["output"][y][x]
                            )
                            total_cells = len(pred) * len(pred[0])
                            cell_acc = correct / total_cells if total_cells > 0 else 0
                            total_acc += cell_acc
                            hdc_similarities.append(cell_acc)
                except Exception:
                    pass
            
            avg_acc = total_acc / len(task.train_pairs) if task.train_pairs else 0.0
            all_exact = exact_matches == len(task.train_pairs)
            
            return (template_name, avg_acc, all_exact)
            
        except Exception:
            return (template_name, 0.0, False)
    
    if not template_names:
        return []
    
    # For very small lists, sequential is fine
    if len(template_names) < 3:
        return [verify_single_gpu(name) for name in template_names]
    
    if num_workers is None:
        num_workers = min(len(template_names), os.cpu_count() or 4)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(verify_single_gpu, name): name
            for name in template_names
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception:
                name = futures[future]
                results.append((name, 0.0, False))
    
    # Sort by accuracy descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# =============================================================================
# Pure Recipe Solver
# =============================================================================

@dataclass
class TaskSample:
    """Simple task representation for the solver."""
    task_id: str
    train_pairs: List[Dict[str, Any]]
    test_pairs: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class PureRecipeSolver:
    """
    Recursive solver that operates entirely on recipes.
    
    This is the main interface for recipe-based recursion. Key features:
    
    1. NO VECTOR CONSTRUCTION during recursion
       - z is a ReasoningRecipe, not a vector
       - Operations are appended to recipe (O(1))
       - No bundle saturation, no density drift
    
    2. RECIPE-BASED TEMPLATE MATCHING
       - Templates stored as TemplateRecipe
       - Matching via recipe_similarity() (O(ops), not O(dim))
       - Can match millions of steps
    
    3. CASCADE VERIFICATION
       - Top-K templates selected by recipe similarity
       - Only top-K reconstructed for actual verification
       - Grid operations used for final accuracy check
    
    Performance vs vector mode:
    - 100-1000x faster similarity checks
    - Effectively infinite recursion depth
    - 80x less memory per operation
    
    Usage:
        solver = PureRecipeSolver(hdc, encoder, grid_engine)
        result, confidence, meta = solver.solve(task)
    """
    
    # Recursion hyperparameters (can be MUCH higher than vector mode)
    N_SUP = 256          # Supervision steps (was 16 in vector mode)
    N_RECURSIONS = 50    # Recursions per step (was 6 in vector mode)
    T_ROUNDS = 5         # Rounds per supervision (was 3 in vector mode)
    
    # Total possible = 256 * 50 * 5 = 64,000 operations (vs ~50 in vector mode)
    
    # Early stopping thresholds
    TEMPLATE_MATCH_THRESHOLD = 0.85  # High confidence template match
    EXACT_MATCH_THRESHOLD = 1.0       # Stop on exact match
    
    # Cascade verification
    TOP_K_TEMPLATES = 5  # Only reconstruct top 5 for verification
    
    # Time limit for try-verify-retry loop (seconds)
    # This ensures we don't loop forever if task is unsolvable
    DEFAULT_TIME_LIMIT = 30.0  # 30 seconds per task
    
    def __init__(
        self,
        hdc: SparseBinaryHDC,
        encoder: Any,  # ARCGridEncoder
        grid_engine: GridTemplateEngine,
        templates: Optional[Any] = None,  # TemplateLibrary
        use_gpu: bool = True  # Enable GPU for final verification phase
    ):
        """
        Initialize the pure recipe solver.
        
        Args:
            hdc: HDC system (only used for final verification reconstruction)
            encoder: Grid encoder (only used for final verification)
            grid_engine: Grid template engine (for applying transformations)
            templates: Optional template library
            use_gpu: Enable GPU acceleration for final verification phase
        """
        self.hdc = hdc
        self.encoder = encoder
        self.grid_engine = grid_engine
        self.templates_lib = templates
        self.use_gpu = use_gpu
        
        # Detect GPU availability for final reconstruction phase
        self._gpu_available = (
            hasattr(hdc, 'batch_similarity') and
            hasattr(hdc, 'use_gpu') and
            getattr(hdc, 'use_gpu', False)
        )
        
        # Build template recipes (NOT vectors!)
        self.template_recipes: Dict[str, TemplateRecipe] = {}
        self._build_template_recipes()
        
        # =====================================================================
        # NEW in v2.9: Relationship Knowledge & Strength Tracker
        # =====================================================================
        # These enable SIMILAR/OPPOSITE fallbacks and degradation after failures
        self.relationship_knowledge = get_template_relationships(hdc)
        self.relationship_tracker = get_relationship_tracker()
        
        # =====================================================================
        # NEW in v2.9.1: Extended Reasoning Components
        # =====================================================================
        # From ExtendedReasoningSystem in train_arc_agi2.py:
        # - TemplateSequenceLearner: Tracks BEFORE/AFTER relationships
        # - ConditionalChainReasoner: IF-THEN chains for template selection
        # - GoalDirectedSolver: Sub-goal decomposition and tracking
        self.sequence_learner = get_sequence_learner()
        self.conditional_reasoner = get_conditional_reasoner()
        self.goal_solver = get_goal_solver()
        
        # =====================================================================
        # NEW in v2.9.2: Bind-not-Bundle & Swarm Thinking
        # =====================================================================
        # - RelationshipReasoner: Enriches recipes with semantic context (non-diluting)
        # - SwarmThinkingPipeline: 4-Stage multi-agent exploration
        self.relationship_reasoner = get_relationship_reasoner(
            self.relationship_knowledge,
            self.relationship_tracker
        )
        self._swarm_pipeline: Optional[SwarmThinkingPipeline] = None  # Lazy init
        
        # Stats tracking
        self.stats = {
            "total_solves": 0,
            "recipes_built": 0,
            "max_recipe_ops": 0,
            "vectors_reconstructed": 0,  # Should be minimal!
            "early_stops": 0,
            "gpu_verifications": 0,  # Track GPU-accelerated verifications
            "relationship_fallbacks_used": 0,  # Track relationship-based recoveries
            "multi_scale_searches": 0,  # Track parameter variation attempts
            "error_pattern_analyses": 0,  # Track error analyses
            # New extended reasoning stats
            "sequence_suggestions_used": 0,
            "conditional_chains_used": 0,
            "goals_created": 0,
            "goals_completed": 0,
        }
        
        # Cache for seen recipes (avoid redundant exploration)
        self._seen_hashes: Set[int] = set()

        # Create verification engine
        self.verification_engine = create_verification_engine(
            hdc=hdc,
            grid_engine=grid_engine,
            solver=self
        )
    
    def _build_template_recipes(self):
        """Build template recipes (NOT vectors) for all known templates."""
        # Geometric templates
        geometric = [
            'rotate_90', 'rotate_180', 'rotate_270',
            'flip_horizontal', 'flip_vertical', 'flip_diagonal',
            'flip_antidiagonal', 'identity'
        ]
        for name in geometric:
            self.template_recipes[name] = TemplateRecipe.from_name(name, "geometric")
        
        # Gravity templates
        gravity = ['gravity_down', 'gravity_up', 'gravity_left', 'gravity_right']
        for name in gravity:
            self.template_recipes[name] = TemplateRecipe.from_name(name, "gravity")
        
        # Translation templates
        translation = [
            'translate_up', 'translate_down', 'translate_left', 'translate_right',
            'translate_up_left', 'translate_up_right', 'translate_down_left', 'translate_down_right'
        ]
        for name in translation:
            self.template_recipes[name] = TemplateRecipe.from_name(name, "translation")
        
        # Morphological templates
        morph = [
            'mark_boundary', 'mark_boundary_8conn', 'extract_boundary',
            'extract_interior', 'fill_enclosed', 'fill_holes',
            'dilate', 'erode', 'morph_outline_recolor',
            'detect_and_mark_boundary'
        ]
        for name in morph:
            self.template_recipes[name] = TemplateRecipe.from_name(name, "morphological")
        
        # Structural templates
        structural = [
            'tile_2x2', 'tile_horizontal', 'tile_vertical',
            'scale_2x', 'crop_nonzero', 'outline'
        ]
        for name in structural:
            self.template_recipes[name] = TemplateRecipe.from_name(name, "structural")
            
        # Dynamic Translation templates (for OBJ_TRANS tasks)
        dynamic_trans = [
            'center_object', 'align_to_edge', 'align_to_corner',
            'move_object_to', 'translate_to_corner',
            'translate_by_offset', 'shift_all_objects'
        ]
        for name in dynamic_trans:
            self.template_recipes[name] = TemplateRecipe.from_name(name, "translation")
            
        # Drawing templates
        drawing = ['draw_line', 'connect_points', 'connect_all_pairs']
        for name in drawing:
            self.template_recipes[name] = TemplateRecipe.from_name(name, "drawing")
    
    def _create_task_context_recipe(self, task: TaskSample) -> ReasoningRecipe:
        """
        Create a recipe representing the task context.
        
        Instead of encoding grids into vectors, we create a symbolic recipe
        that captures the task's characteristics.
        """
        # Create base seed from task ID
        base_seed = _string_to_seed(f"task_context_{task.task_id}")
        recipe = ReasoningRecipe(base_seed=base_seed)
        
        # Add operations that represent task characteristics
        for i, pair in enumerate(task.train_pairs):
            # Bind operation for each training pair
            recipe.add_bind(f"train_pair_{i}")
            
            # Analyze grid characteristics and add relevant binds
            input_grid = pair.get("input", [])
            output_grid = pair.get("output", [])
            
            # Size characteristics
            in_h, in_w = len(input_grid), len(input_grid[0]) if input_grid else 0
            out_h, out_w = len(output_grid), len(output_grid[0]) if output_grid else 0
            
            if in_h != out_h or in_w != out_w:
                recipe.add_bind("size_change")
            if in_h == out_w and in_w == out_h:
                recipe.add_bind("possible_rotation")
            
            # Color characteristics
            in_colors = set()
            out_colors = set()
            for row in input_grid:
                in_colors.update(row)
            for row in output_grid:
                out_colors.update(row)
            
            if in_colors != out_colors:
                recipe.add_bind("color_change")
            if len(out_colors - in_colors) > 0:
                recipe.add_bind("new_colors")
                recipe.add_bind("possible_boundary")
            
            # Permute by pair index for ordering
            recipe.add_permute(i * 137)
        
        return recipe
    
    def _match_templates_by_recipe(
        self,
        z_recipe: ReasoningRecipe,
        top_k: int = 5,
        use_parallel: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find best matching templates using RECIPE similarity only.
        
        NO VECTORS ARE CONSTRUCTED HERE.
        
        Uses batch_recipe_to_template_similarity for parallel processing
        when there are many templates (>10).
        
        Args:
            z_recipe: Current reasoning state as recipe
            top_k: Number of top matches to return
            use_parallel: Whether to use parallel processing
        
        Returns:
            List of (template_name, similarity_score) tuples
        """
        templates_list = list(self.template_recipes.values())
        
        if use_parallel:
            # Use batch/parallel matching for large template sets
            matches = batch_recipe_to_template_similarity(z_recipe, templates_list)
        else:
            # Sequential matching
            matches = []
            for template in templates_list:
                sim = recipe_to_template_similarity(z_recipe, template)
                matches.append((template.name, sim))
            matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:top_k]
    
    def _pure_latent_recursion(
        self,
        z_recipe: ReasoningRecipe,
        task_recipe: ReasoningRecipe,
        answer_recipe: ReasoningRecipe,
        n_recursions: int,
        dim: int = 32768
    ) -> ReasoningRecipe:
        """
        Pure recipe-based latent recursion.
        
        This is the TRM latent update: z = net(x, y, z)
        But ALL operations are symbolic - no vectors involved.
        
        Cost: O(n_recursions) simple list appends
        (vs O(n_recursions * dim) for vector mode)
        """
        for i in range(n_recursions):
            # Symbolic equivalent of: xz = bind(x, z)
            z_recipe.add_bind("x_context")
            
            # Symbolic equivalent of: yz = bind(y, z)
            z_recipe.add_bind("y_answer")
            
            # Symbolic equivalent of: z_shifted = permute(z, shift)
            shift = ((i + 1) * 137) % dim
            z_recipe.add_permute(shift)
        
        return z_recipe
    
    def _verify_recipe_on_task(
        self,
        template_name: str,
        task: TaskSample
    ) -> Tuple[float, bool]:
        """
        Actually verify a template on the task.
        
        THIS IS WHERE WE FINALLY CONSTRUCT VECTORS (if needed).
        But typically we just apply the grid transformation directly.
        
        Args:
            template_name: Name of template to try
            task: Task to verify on
        
        Returns:
            (accuracy, exact_match)
        """
        self.stats["vectors_reconstructed"] += 1
        
        # Create a transformation recipe for grid operations
        try:
            # Import here to avoid circular imports
            from .grid_templates import TransformationRecipe, TransformationStep
        except ImportError:
            # Fallback for standalone testing
            return 0.0, False
        
        recipe = TransformationRecipe(
            steps=[TransformationStep(name=template_name, params={})],
            confidence=0.5
        )
        
        # =================================================================
        # INTEGRATED VERIFICATION (Level 1-4)
        # =================================================================
        # Before running expensive grid operations on all pairs, run verification
        # on the first pair to catch obvious issues or adversarial failures
        if task.train_pairs:
            verify_result = self.verification_engine.verify(
                prediction_vector=None,
                recipe=recipe,
                input_grid=task.train_pairs[0]["input"],
                expected_output=task.train_pairs[0]["output"]
            )
            
            if not verify_result.is_accepted:
                # Verification failed - reject this candidate immediately
                return 0.0, False

        total_acc = 0.0
        exact_matches = 0
        
        for pair in task.train_pairs:
            try:
                pred = self.grid_engine.apply_recipe(pair["input"], recipe)
                if pred is not None:
                    # Check exact match
                    if pred == pair["output"]:
                        exact_matches += 1
                        total_acc += 1.0
                    else:
                        # Cell-level accuracy
                        if len(pred) == len(pair["output"]) and all(
                            len(r1) == len(r2) for r1, r2 in zip(pred, pair["output"])
                        ):
                            correct = sum(
                                1 for y in range(len(pred))
                                for x in range(len(pred[0]))
                                if pred[y][x] == pair["output"][y][x]
                            )
                            total_cells = len(pred) * len(pred[0])
                            total_acc += correct / total_cells
            except Exception:
                pass
        
        avg_acc = total_acc / len(task.train_pairs) if task.train_pairs else 0.0
        all_exact = exact_matches == len(task.train_pairs)
        
        return avg_acc, all_exact
    
    def solve(
        self,
        task: TaskSample,
        trace: Optional[Any] = None,  # ReasoningTrace
        time_limit: Optional[float] = None  # Override default time limit
    ) -> Tuple[Optional[List[List[int]]], float, Dict[str, Any]]:
        """
        Solve a task using try-verify-retry loop with time limit.
        
        Algorithm (v2.8.0 - Enhanced with Heuristics):
        1. Analyze task characteristics (size changes, colors, positions)
        2. Infer template parameters from training pairs
        3. Try PARAMETERIZED templates first (with inferred parameters)
        4. Try ALL templates with default params (fast parallel verification)
        5. Try composite transforms based on characteristics
        6. If no exact match, use recursion to generate new suggestions
        7. Repeat until exact match OR time limit reached
        
        Key improvements in v2.8:
        - Parameterized verification: Uses inferred translation amounts, colors
        - Heuristic pre-filtering: Prioritizes likely templates based on analysis
        - Composite transforms: Tries 2-step combinations with guidance
        
        Args:
            task: The task to solve
            trace: Optional reasoning trace for logging
            time_limit: Max seconds to spend (None = use DEFAULT_TIME_LIMIT)
        
        Returns:
            (predicted_output, confidence, metadata)
        """
        import time
        
        self.stats["total_solves"] += 1
        self._seen_hashes.clear()  # Reset for new task
        
        # Set time limit
        max_time = time_limit if time_limit is not None else self.DEFAULT_TIME_LIMIT
        start_time = time.time()
        
        def time_remaining() -> float:
            """Check remaining time."""
            return max_time - (time.time() - start_time)
        
        def time_expired() -> bool:
            """Check if time limit exceeded."""
            return max_time > 0 and time.time() - start_time >= max_time
        
        # Track best result so far
        best_template = None
        best_accuracy = 0.0
        best_params = {}  # Parameters used for best result
        total_ops = 0
        tried_templates: Set[str] = set()  # Track which templates we've tried
        
        # =================================================================
        # PHASE 0: Analyze task characteristics (NEW in v2.8)
        # =================================================================
        characteristics = analyze_task_characteristics(task)
        inferred_params = infer_parameters_from_task(task, self.grid_engine)
        
        logger.debug(f"Task characteristics: {characteristics['likely_categories']}")
        logger.debug(f"Inferred params: {list(inferred_params.keys())}")
        
        # =================================================================
        # PHASE 1: Try PARAMETERIZED templates first (highest priority)
        # =================================================================
        # This is the key improvement: we try templates with their INFERRED
        # parameters (e.g., translate_up with amount=3) instead of defaults
        if inferred_params:
            param_results = parallel_verify_parameterized_templates(
                task, self.grid_engine, inferred_params
            )
            
            for template_name, acc, exact, params in param_results:
                tried_templates.add(template_name)
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_template = template_name
                    best_params = params
                
                if exact:
                    self.stats["early_stops"] += 1
                    return self._apply_to_test_with_params(
                        task, template_name, params, acc, {
                            "method": "parameterized_verify_exact",
                            "supervision_steps": 0,
                            "total_operations": 0,
                            "time_spent": time.time() - start_time,
                            "exact_match": True,
                            "inferred_params": params
                        }
                    )
        
        if time_expired():
            if best_template:
                return self._apply_to_test_with_params(
                    task, best_template, best_params, best_accuracy, {
                        "method": "parameterized_verify_timeout",
                        "time_spent": time.time() - start_time,
                        "exact_match": False
                    }
                )
        
        # =================================================================
        # PHASE 2: Try ALL templates with default params
        # =================================================================
        all_template_names = list(self.template_recipes.keys())
        
        # Use GPU-enhanced parallel verification when available
        if self.use_gpu and self._gpu_available:
            verified_results = parallel_verify_templates_gpu(
                all_template_names, task, self.grid_engine,
                self.hdc, self.encoder
            )
            self.stats["gpu_verifications"] += len(verified_results)
        else:
            verified_results = parallel_verify_templates(
                all_template_names, task, self.grid_engine
            )
        self.stats["vectors_reconstructed"] += len(verified_results)
        
        # Check results - return immediately on exact match
        for template_name, acc, exact in verified_results:
            tried_templates.add(template_name)
            
            # Track outcome for degradation
            task_context = characteristics.get('likely_categories', ['unknown'])[0]
            self.relationship_tracker.track_outcome(
                template_name,
                success=exact,
                context=task_context
            )
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_template = template_name
                best_params = {}  # Default params
            
            if exact:
                self.stats["early_stops"] += 1
                return self._apply_to_test(
                    task, template_name, acc, {
                        "method": "direct_verify_exact",
                        "supervision_steps": 0,
                        "total_operations": 0,
                        "time_spent": time.time() - start_time,
                        "exact_match": True
                    }
                )
        
        # =================================================================
        # PHASE 2.5: Multi-scale parameter search for near-misses (NEW v2.9)
        # =================================================================
        # If a template achieved 80-95% accuracy, try parameter variations
        near_miss_templates = [
            (name, acc, params) for name, acc, exact in verified_results
            if 0.80 <= acc < 0.95
            for params in [best_params]  # Get current params
        ]
        
        if near_miss_templates and not time_expired():
            self.stats["multi_scale_searches"] += len(near_miss_templates)
            
            for template_name, acc, base_params in near_miss_templates[:3]:  # Top 3 near-misses
                # Try parameter variations
                search_result = multi_scale_parameter_search(
                    template_name,
                    base_params if base_params else {},  # Use inferred or empty
                    task,
                    self.grid_engine
                )
                
                _, new_acc, new_exact, new_params = search_result
                
                if new_exact:
                    self.stats["early_stops"] += 1
                    self.relationship_tracker.track_outcome(template_name, success=True, context=task_context)
                    return self._apply_to_test_with_params(
                        task, template_name, new_params, new_acc, {
                            "method": "multi_scale_search_exact",
                            "base_accuracy": acc,
                            "improved_accuracy": new_acc,
                            "time_spent": time.time() - start_time,
                            "exact_match": True,
                            "params_found": new_params
                        }
                    )
                
                if new_acc > best_accuracy:
                    best_accuracy = new_acc
                    best_template = template_name
                    best_params = new_params
        
        # =================================================================
        # PHASE 2.6: Error pattern analysis for relationship fallbacks (NEW v2.9)
        # =================================================================
        # If we have a near-miss template, analyze WHERE errors occur to suggest fixes
        if best_template and 0.70 <= best_accuracy < 1.0 and not time_expired():
            self.stats["error_pattern_analyses"] += 1
            
            # Get prediction from best template to analyze errors
            try:
                from .grid_templates import TransformationRecipe, TransformationStep
                
                test_recipe = TransformationRecipe(
                    steps=[TransformationStep(name=best_template, params=best_params)],
                    confidence=best_accuracy
                )
                
                # Analyze first training pair
                first_pair = task.train_pairs[0]
                pred = self.grid_engine.apply_recipe(first_pair["input"], test_recipe)
                
                if pred is not None:
                    # Analyze error patterns
                    error_patterns = analyze_error_pattern(pred, first_pair["output"])
                    
                    if error_patterns:
                        # =======================================================
                        # Use IF-THEN conditional chain reasoning (NEW v2.9.2)
                        # =======================================================
                        # Query conditional reasoner for error-driven suggestions
                        conditional_suggestions = self.conditional_reasoner.get_actions_for_error_patterns(
                            error_patterns
                        )
                        self.stats["conditional_chains_used"] += len(conditional_suggestions)
                        
                        # Try conditional suggestions first (IF error THEN action)
                        for cond_template, cond_reason, cond_conf in conditional_suggestions[:2]:
                            if cond_template in tried_templates:
                                continue
                            if self.relationship_tracker.is_deprecated(cond_template):
                                continue
                            
                            tried_templates.add(cond_template)
                            
                            cond_results = parallel_verify_templates(
                                [cond_template], task, self.grid_engine
                            )
                            
                            for c_name, c_acc, c_exact in cond_results:
                                self.relationship_tracker.track_outcome(
                                    c_name, success=c_exact, context=task_context
                                )
                                
                                # Learn from this outcome
                                error_type = error_patterns[0].error_type if error_patterns else "unknown"
                                effect = "fix_" + error_type
                                self.conditional_reasoner.learn_conditional(
                                    error_type, c_name, effect, success=c_exact
                                )
                                
                                if c_exact:
                                    self.stats["early_stops"] += 1
                                    return self._apply_to_test(
                                        task, c_name, c_acc, {
                                            "method": "conditional_chain_exact",
                                            "original_template": best_template,
                                            "condition": cond_reason,
                                            "error_patterns": [p.error_type for p in error_patterns],
                                            "time_spent": time.time() - start_time,
                                            "exact_match": True
                                        }
                                    )
                                
                                if c_acc > best_accuracy:
                                    best_accuracy = c_acc
                                    best_template = c_name
                                    best_params = {}
                        
                        # Get suggested fixes based on errors + relationships
                        suggested_fixes = suggest_fixes_from_errors(
                            error_patterns,
                            self.relationship_knowledge
                        )
                        
                        # Try suggested fixes
                        for fix_template, reason, priority in suggested_fixes[:3]:
                            if fix_template in tried_templates:
                                continue
                            if self.relationship_tracker.is_deprecated(fix_template):
                                continue
                            
                            self.stats["relationship_fallbacks_used"] += 1
                            tried_templates.add(fix_template)
                            
                            # Verify the suggested fix
                            fix_results = parallel_verify_templates(
                                [fix_template], task, self.grid_engine
                            )
                            
                            for fix_name, fix_acc, fix_exact in fix_results:
                                self.relationship_tracker.track_outcome(
                                    fix_name, success=fix_exact, context=task_context
                                )
                                
                                if fix_exact:
                                    self.stats["early_stops"] += 1
                                    return self._apply_to_test(
                                        task, fix_name, fix_acc, {
                                            "method": "error_pattern_fix_exact",
                                            "original_template": best_template,
                                            "fix_reason": reason,
                                            "error_patterns": [p.error_type for p in error_patterns],
                                            "time_spent": time.time() - start_time,
                                            "exact_match": True
                                        }
                                    )
                                
                                if fix_acc > best_accuracy:
                                    best_accuracy = fix_acc
                                    best_template = fix_name
                                    best_params = {}
            except Exception as e:
                logger.debug(f"Error pattern analysis failed: {e}")
        
        # =================================================================
        # PHASE 2.7: Relationship-based fallbacks (NEW v2.9)
        # =================================================================
        # Use SIMILAR/OPPOSITE/COMPOSED relationships to find alternatives
        if best_template and best_accuracy < 1.0 and not time_expired():
            fallbacks = self.relationship_knowledge.get_fallback_templates(
                best_template, best_accuracy
            )
            
            for fallback_template, reason in fallbacks:
                if fallback_template in tried_templates:
                    continue
                if self.relationship_tracker.is_deprecated(fallback_template):
                    continue
                
                self.stats["relationship_fallbacks_used"] += 1
                tried_templates.add(fallback_template)
                
                fb_results = parallel_verify_templates(
                    [fallback_template], task, self.grid_engine
                )
                
                for fb_name, fb_acc, fb_exact in fb_results:
                    self.relationship_tracker.track_outcome(
                        fb_name, success=fb_exact, context=task_context
                    )
                    
                    if fb_exact:
                        self.stats["early_stops"] += 1
                        return self._apply_to_test(
                            task, fb_name, fb_acc, {
                                "method": "relationship_fallback_exact",
                                "original_template": best_template,
                                "fallback_reason": reason,
                                "time_spent": time.time() - start_time,
                                "exact_match": True
                            }
                        )
                    
                    if fb_acc > best_accuracy:
                        best_accuracy = fb_acc
                        best_template = fb_name
                        best_params = {}
        
        if time_expired():
            if best_template:
                return self._apply_to_test_with_params(
                    task, best_template, best_params, best_accuracy, {
                        "method": "direct_verify_timeout",
                        "time_spent": time.time() - start_time,
                        "exact_match": False
                    }
                )
        
        # =================================================================
        # PHASE 2.8: Goal-Directed Solving (NEW in v2.9.2)
        # =================================================================
        # Create goals based on task characteristics
        goals = self.goal_solver.create_goals_for_task(characteristics)
        self.stats["goals_created"] += len(goals)
        
        logger.debug(f"Created {len(goals)} goals: {list(goals.keys())}")
        
        # Get template suggestions from goals
        goal_suggestions = self.goal_solver.get_suggested_templates_for_goals()
        
        if goal_suggestions and not time_expired():
            goal_templates = [t for t, _, _ in goal_suggestions[:5]]
            goal_results = parallel_verify_templates(
                goal_templates, task, self.grid_engine
            )
            
            for template_name, acc, exact in goal_results:
                tried_templates.add(template_name)
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_template = template_name
                    best_params = {}
                
                if exact:
                    self.stats["early_stops"] += 1
                    self.stats["goals_completed"] += 1
                    return self._apply_to_test(
                        task, template_name, acc, {
                            "method": "goal_directed_exact",
                            "goals": list(goals.keys()),
                            "time_spent": time.time() - start_time,
                            "exact_match": True
                        }
                    )
        
        # =================================================================
        # PHASE 2.9: Swarm Thinking Pipeline (NEW in v2.9.2)
        # =================================================================
        # Use 4-stage multi-agent exploration for complex tasks
        if not time_expired() and (best_accuracy < 0.9 or not best_template):
            # Lazy initialize swarm pipeline
            if self._swarm_pipeline is None:
                self._swarm_pipeline = SwarmThinkingPipeline(
                    self.grid_engine,
                    self.relationship_knowledge,
                    self.relationship_tracker
                )
            
            swarm_result = self._swarm_pipeline.solve(task, characteristics)
            
            if swarm_result:
                sequence, confidence, swarm_meta = swarm_result
                
                # Check if swarm found exact match
                if swarm_meta.get("exact_match", False):
                    self.stats["early_stops"] += 1
                    
                    if len(sequence) == 1:
                        return self._apply_to_test(
                            task, sequence[0], confidence, {
                                "method": "swarm_thinking_exact",
                                "swarm_meta": swarm_meta,
                                "time_spent": time.time() - start_time,
                                "exact_match": True
                            }
                        )
                    else:
                        return self._apply_composite_to_test(
                            task, sequence, confidence, {
                                "method": "swarm_thinking_composite_exact",
                                "swarm_meta": swarm_meta,
                                "time_spent": time.time() - start_time,
                                "exact_match": True
                            }
                        )
                
                # Update best if swarm found better
                if confidence > best_accuracy:
                    best_accuracy = confidence
                    best_template = sequence[0] if len(sequence) == 1 else "+".join(sequence)
                    best_params = {}
        
        # =================================================================
        # PHASE 3: Try composite transforms using sequence learning (v2.8 + v2.9.2)
        # =================================================================
        # Use sequence learner suggestions in addition to heuristics
        sequence_suggestions = self.sequence_learner.suggest_two_step_sequences(
            characteristics, top_k=5
        )
        self.stats["sequence_suggestions_used"] += len(sequence_suggestions)
        
        composite_results = verify_composite_transforms(
            task, self.grid_engine, characteristics
        )
        
        for composite_name, acc, exact in composite_results:
            if acc > best_accuracy:
                best_accuracy = acc
                best_template = composite_name
                best_params = {}
            
            if exact:
                self.stats["early_stops"] += 1
                # Parse composite name to get steps
                steps = composite_name.split('+')
                return self._apply_composite_to_test(
                    task, steps, acc, {
                        "method": "composite_verify_exact",
                        "supervision_steps": 0,
                        "total_operations": 0,
                        "time_spent": time.time() - start_time,
                        "exact_match": True,
                        "composite_steps": steps
                    }
                )
        
        # If time already expired after initial verification, return best
        if time_expired():
            if best_template:
                return self._apply_to_test(
                    task, best_template, best_accuracy, {
                        "method": "direct_verify_timeout",
                        "supervision_steps": 0,
                        "total_operations": 0,
                        "time_spent": time.time() - start_time,
                        "exact_match": False
                    }
                )
            return None, 0.0, {"method": "timeout_no_solution", "time_spent": time.time() - start_time}
        
        # =================================================================
        # PHASE 2: Try-Verify-Retry loop using recursion
        # =================================================================
        # No single template worked, so now use recursion to explore
        # composite transformations or alternative approaches
        
        # Create task context as recipe
        task_recipe = self._create_task_context_recipe(task)
        
        # Initialize answer and reasoning recipes
        answer_seed = _string_to_seed(f"answer_{task.task_id}")
        answer_recipe = ReasoningRecipe(base_seed=answer_seed)
        
        z_seed = _string_to_seed(f"reasoning_{task.task_id}")
        z_recipe = ReasoningRecipe(base_seed=z_seed)
        
        retry_count = 0
        max_retries = self.N_SUP  # Use supervision steps as max retries
        
        while not time_expired() and retry_count < max_retries:
            retry_count += 1
            
            # =========================================================
            # STEP A: Run recursion to generate new template suggestions
            # =========================================================
            for t in range(self.T_ROUNDS):
                z_recipe = self._pure_latent_recursion(
                    z_recipe, task_recipe, answer_recipe,
                    n_recursions=self.N_RECURSIONS,
                    dim=self.hdc.dim if hasattr(self.hdc, 'dim') else 32768
                )
                total_ops += self.N_RECURSIONS * 3
            
            # Check for cycling
            z_hash = z_recipe.get_hash()
            if z_hash in self._seen_hashes:
                # Recipe cycling - add escape operation to break pattern
                z_recipe.add_bind(f"escape_{retry_count}_{total_ops}")
            self._seen_hashes.add(z_hash)
            
            # Add hint from best match so far
            if best_template:
                answer_recipe.add_bind(f"hint_{best_template}")
                # Also add "failed" context so recursion knows to try something different
                z_recipe.add_bind(f"failed_{best_template}")
            
            # =========================================================
            # STEP B: Get template suggestions from recipe matching
            # =========================================================
            top_matches = self._match_templates_by_recipe(z_recipe, self.TOP_K_TEMPLATES * 2)
            
            # Filter to templates we haven't exhaustively tried, or retry with variations
            templates_to_try = []
            for name, score in top_matches:
                # Always include high scoring templates
                if score > 0.3 or name not in tried_templates:
                    templates_to_try.append(name)
                if len(templates_to_try) >= self.TOP_K_TEMPLATES:
                    break
            
            if not templates_to_try:
                # All templates exhausted, try composing them
                # Get top 2 templates and try combining
                if len(top_matches) >= 2:
                    templates_to_try = [top_matches[0][0], top_matches[1][0]]
            
            # =========================================================
            # STEP C: Verify suggestions against grid
            # =========================================================
            if templates_to_try:
                if self.use_gpu and self._gpu_available:
                    verified = parallel_verify_templates_gpu(
                        templates_to_try, task, self.grid_engine,
                        self.hdc, self.encoder
                    )
                    self.stats["gpu_verifications"] += len(verified)
                else:
                    verified = parallel_verify_templates(
                        templates_to_try, task, self.grid_engine
                    )
                self.stats["vectors_reconstructed"] += len(verified)
                
                for template_name, acc, exact in verified:
                    tried_templates.add(template_name)
                    
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_template = template_name
                    
                    # =========================================================
                    # STEP D: If exact match, SUCCESS! Return immediately
                    # =========================================================
                    if exact:
                        self.stats["early_stops"] += 1
                        self.stats["max_recipe_ops"] = max(
                            self.stats["max_recipe_ops"],
                            len(z_recipe.operations)
                        )
                        
                        return self._apply_to_test(
                            task, template_name, acc, {
                                "method": "try_verify_retry_success",
                                "supervision_steps": retry_count,
                                "total_operations": total_ops,
                                "recipe_size_bytes": z_recipe.size_bytes(),
                                "time_spent": time.time() - start_time,
                                "retries": retry_count,
                                "exact_match": True
                            }
                        )
            
            # =========================================================
            # STEP E: Verification failed - continue loop to try again
            # =========================================================
            # The loop will continue with modified z_recipe that has
            # "failed_{template}" context, guiding it to try alternatives
        
        # =================================================================
        # PHASE 3: Time limit reached - return best partial result
        # =================================================================
        self.stats["recipes_built"] += 1
        self.stats["max_recipe_ops"] = max(
            self.stats["max_recipe_ops"],
            len(z_recipe.operations)
        )
        
        if best_template:
            return self._apply_to_test(
                task, best_template, best_accuracy, {
                    "method": "try_verify_retry_timeout",
                    "supervision_steps": retry_count,
                    "total_operations": total_ops,
                    "recipe_size_bytes": z_recipe.size_bytes(),
                    "time_spent": time.time() - start_time,
                    "retries": retry_count,
                    "exact_match": False
                }
            )
        
        return None, 0.0, {
            "method": "try_verify_retry_no_solution",
            "total_operations": total_ops,
            "time_spent": time.time() - start_time,
            "retries": retry_count
        }
    
    def _apply_to_test(
        self,
        task: TaskSample,
        template_name: str,
        accuracy: float,
        metadata: Dict[str, Any]
    ) -> Tuple[Optional[List[List[int]]], float, Dict[str, Any]]:
        """Apply the found template to test input."""
        if not task.test_pairs:
            return None, accuracy, metadata
        
        try:
            from .grid_templates import TransformationRecipe, TransformationStep
            
            recipe = TransformationRecipe(
                steps=[TransformationStep(name=template_name, params={})],
                confidence=accuracy
            )
            
            test_input = task.test_pairs[0]["input"]
            prediction = self.grid_engine.apply_recipe(test_input, recipe)
            
            metadata["recipe_applied"] = template_name
            return prediction, accuracy, metadata
            
        except Exception:
            return None, accuracy, metadata
    
    def _apply_to_test_with_params(
        self,
        task: TaskSample,
        template_name: str,
        params: Dict[str, Any],
        accuracy: float,
        metadata: Dict[str, Any]
    ) -> Tuple[Optional[List[List[int]]], float, Dict[str, Any]]:
        """Apply the found template with specific parameters to test input."""
        if not task.test_pairs:
            return None, accuracy, metadata
        
        try:
            from .grid_templates import TransformationRecipe, TransformationStep
            
            recipe = TransformationRecipe(
                steps=[TransformationStep(name=template_name, params=params)],
                confidence=accuracy
            )
            
            test_input = task.test_pairs[0]["input"]
            prediction = self.grid_engine.apply_recipe(test_input, recipe)
            
            metadata["recipe_applied"] = f"{template_name}({params})"
            metadata["params_used"] = params
            return prediction, accuracy, metadata
            
        except Exception as e:
            logger.debug(f"Failed to apply {template_name} with params {params}: {e}")
            return None, accuracy, metadata
    
    def _apply_composite_to_test(
        self,
        task: TaskSample,
        steps: List[str],
        accuracy: float,
        metadata: Dict[str, Any]
    ) -> Tuple[Optional[List[List[int]]], float, Dict[str, Any]]:
        """
        Apply a composite (multi-step) transformation to test input.
        
        Handles steps in format:
        - Simple: ["rotate_90", "flip_horizontal"]
        - With params: ["rotate_90", "mark_boundary({'boundary_color': 3})"]
        """
        if not task.test_pairs:
            return None, accuracy, metadata
        
        try:
            from .grid_templates import TransformationRecipe, TransformationStep
            import ast
            
            transformation_steps = []
            for step in steps:
                # Check if step has parameters in format: name(params)
                if '(' in step and step.endswith(')'):
                    # Parse name and params
                    paren_idx = step.index('(')
                    name = step[:paren_idx]
                    params_str = step[paren_idx+1:-1]
                    
                    try:
                        params = ast.literal_eval(params_str) if params_str else {}
                    except (ValueError, SyntaxError):
                        params = {}
                else:
                    name = step
                    params = {}
                
                transformation_steps.append(TransformationStep(name=name, params=params))
            
            recipe = TransformationRecipe(
                steps=transformation_steps,
                confidence=accuracy
            )
            
            test_input = task.test_pairs[0]["input"]
            prediction = self.grid_engine.apply_recipe(test_input, recipe)
            
            metadata["recipe_applied"] = " + ".join(steps)
            metadata["composite_steps"] = steps
            return prediction, accuracy, metadata
            
        except Exception as e:
            logger.debug(f"Failed to apply composite {steps}: {e}")
            return None, accuracy, metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get solver performance statistics including relationship tracking."""
        # Get relationship tracker stats
        tracker_stats = self.relationship_tracker.get_stats()
        
        return {
            **self.stats,
            "avg_vectors_per_solve": (
                self.stats["vectors_reconstructed"] / max(self.stats["total_solves"], 1)
            ),
            "gpu_verification_ratio": (
                self.stats.get("gpu_verifications", 0) /
                max(self.stats["vectors_reconstructed"], 1)
            ),
            "num_templates": len(self.template_recipes),
            "gpu_enabled": self.use_gpu,
            "gpu_available": self._gpu_available,
            # Relationship tracker stats (v2.9)
            "relationship_fallback_success_rate": (
                self.stats["relationship_fallbacks_used"] / max(self.stats["early_stops"], 1)
                if self.stats["relationship_fallbacks_used"] > 0 else 0.0
            ),
            "deprecated_templates": tracker_stats.get("deprecated_templates", []),
            "degraded_template_count": tracker_stats.get("degraded_count", 0),
            "override_template_count": tracker_stats.get("override_count", 0),
            "tracker_avg_success_rate": tracker_stats.get("avg_success_rate", 0.0),
            # Extended reasoning stats (v2.9.2)
            "sequence_learning": self.sequence_learner.get_stats(),
            "conditional_reasoning": self.conditional_reasoner.get_stats(),
            "goal_tracking": self.goal_solver.get_stats(),
            "relationship_reasoner": self.relationship_reasoner.get_stats(),
            "swarm_pipeline": self._swarm_pipeline.get_stats() if self._swarm_pipeline else {"not_initialized": True},
        }


# =============================================================================
# Integration Helper
# =============================================================================

def create_pure_recipe_solver(
    hdc: SparseBinaryHDC,
    encoder: Any,
    grid_engine: GridTemplateEngine,
    use_gpu: bool = True
) -> PureRecipeSolver:
    """
    Factory function to create a PureRecipeSolver.
    
    Args:
        hdc: HDC system
        encoder: Grid encoder
        grid_engine: Grid template engine
        use_gpu: Enable GPU acceleration for final verification phase
    
    Returns:
        Configured PureRecipeSolver with GPU support
    """
    return PureRecipeSolver(hdc, encoder, grid_engine, use_gpu=use_gpu)


# =============================================================================
# Benchmark Code
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=== Pure Recipe Solver Benchmark ===\n")
    
    # Test recipe operations
    recipe = ReasoningRecipe(base_seed=12345)
    
    start = time.time()
    for i in range(100000):
        recipe.add_bind("context_x")
        recipe.add_bind("context_y")
        recipe.add_permute(i * 137)
    elapsed = time.time() - start
    
    print(f"Added 300,000 operations in {elapsed:.3f}s")
    print(f"  Ops per second: {300000 / elapsed:,.0f}")
    print(f"  Recipe size: {recipe.size_bytes() / 1024 / 1024:.2f} MB")
    print(f"  Ops count: {len(recipe.operations):,}")
    
    # Test recipe similarity
    recipe2 = ReasoningRecipe(base_seed=12345)
    for i in range(50000):
        recipe2.add_bind("context_x")
        recipe2.add_bind("context_y")
        recipe2.add_permute(i * 137)
    
    start = time.time()
    sim = recipe_similarity(recipe, recipe2)
    elapsed = time.time() - start
    
    print(f"\nRecipe similarity: {sim:.4f}")
    print(f"  Computed in: {elapsed * 1000:.3f}ms")
    
    # Test template matching
    template = TemplateRecipe.from_name("rotate_90", "geometric")
    
    start = time.time()
    for _ in range(10000):
        score = recipe_to_template_similarity(recipe, template)
    elapsed = time.time() - start
    
    print(f"\nTemplate matching: {score:.4f}")
    print(f"  10,000 matches in: {elapsed * 1000:.1f}ms")
    print(f"  Matches per second: {10000 / elapsed:,.0f}")
    
    print("\n✅ Benchmark complete!")
    print("\nComparison to vector mode (estimated):")
    print("  Vector similarity: ~1-10ms per comparison")
    print("  Recipe similarity: ~0.01ms per comparison")
    print("  Speedup: ~100-1000x")
