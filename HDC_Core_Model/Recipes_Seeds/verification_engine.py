"""
Verification Engine for HDC-Sparse System

This module re-exports the verification engine classes from pure_recipe_solver
to maintain backward compatibility. The actual implementation has been moved
to pure_recipe_solver.py to avoid circular imports.

The 4-level verification funnel:
- Level 1: HDC space verification (vector health, orthogonality)
- Level 2: Tool-based simulation (executable oracle)
- Level 3: Adversarial stress testing
- Level 4: Auto-review module

Enhanced in v2.9 with:
- Error pattern analysis (analyze WHERE mismatches occur)
- Relationship-based fix suggestions (SIMILAR/OPPOSITE templates)
- Relationship strength tracking (degradation after failures)
- Multi-scale parameter search (±1 variations for near-misses)

Enhanced in v2.9.2 with:
- TemplateSequenceLearner: Temporal BEFORE/AFTER relationships between templates
- ConditionalChainReasoner: IF-THEN chains for template selection
- GoalDirectedSolver: Sub-goal decomposition and tracking
- RelationshipReasoner: Bind-not-bundle context enrichment
- SwarmThinkingPipeline: 4-stage multi-agent exploration
- Custom symbolic relationships: ENABLES, CAUSES, PREVENTS, FOLLOWS

For new code, prefer importing directly from pure_recipe_solver.
"""

# Re-export verification classes from pure_recipe_solver to avoid circular imports
from ..HDC_Core_Model.Recipes_Seeds.pure_recipe_solver import (
    # Core verification classes
    VerificationStatus,
    VerificationResult,
    UniversalHDCVerifier,
    ToolBasedVerifier,
    AdversarialVerifier,
    AutoReviewModule,
    MetaVerificationEngine,
    create_verification_engine,
    # v2.9: Error pattern analysis
    ErrorPattern,
    analyze_error_pattern,
    suggest_fixes_from_errors,
    # v2.9: Relationship-based improvements
    TemplateRelationshipKnowledge,
    get_template_relationships,
    RelationshipStrengthTracker,
    get_relationship_tracker,
    # v2.9: Multi-scale parameter search
    multi_scale_parameter_search,
    verify_template_with_params,
    # v2.9.2: Extended Reasoning Components
    # Temporal sequence learning (BEFORE/AFTER patterns)
    TemplateSequenceLearner,
    get_sequence_learner,
    # IF-THEN conditional chain reasoning
    ConditionalChainReasoner,
    get_conditional_reasoner,
    # Goal-directed sub-goal tracking
    GoalDirectedSolver,
    get_goal_solver,
    # Bind-not-bundle relationship reasoning
    RelationshipReasoner,
    get_relationship_reasoner,
    # Swarm thinking pipeline
    SwarmThinkingPipeline,
    SwarmAgent,
    get_swarm_pipeline,
    # Custom symbolic relationships
    learn_symbolic_relationship,
    query_symbolic_relationship,
    get_relationships_for_entity,
    CUSTOM_RELATIONSHIPS,
    # Recipe operation types (includes BIND_CONTEXT)
    RecipeOpType,
    RecipeOp,
    ReasoningRecipe,
    TemplateRecipe,
)

# Export all verification-related symbols
__all__ = [
    # Core verification
    'VerificationStatus',
    'VerificationResult',
    'UniversalHDCVerifier',
    'ToolBasedVerifier',
    'AdversarialVerifier',
    'AutoReviewModule',
    'MetaVerificationEngine',
    'create_verification_engine',
    # Error pattern analysis
    'ErrorPattern',
    'analyze_error_pattern',
    'suggest_fixes_from_errors',
    # Relationship-based improvements
    'TemplateRelationshipKnowledge',
    'get_template_relationships',
    'RelationshipStrengthTracker',
    'get_relationship_tracker',
    # Multi-scale parameter search
    'multi_scale_parameter_search',
    'verify_template_with_params',
    # v2.9.2: Extended Reasoning Components
    'TemplateSequenceLearner',
    'get_sequence_learner',
    'ConditionalChainReasoner',
    'get_conditional_reasoner',
    'GoalDirectedSolver',
    'get_goal_solver',
    'RelationshipReasoner',
    'get_relationship_reasoner',
    'SwarmThinkingPipeline',
    'SwarmAgent',
    'get_swarm_pipeline',
    # Custom symbolic relationships
    'learn_symbolic_relationship',
    'query_symbolic_relationship',
    'get_relationships_for_entity',
    'CUSTOM_RELATIONSHIPS',
    # Recipe structures
    'RecipeOpType',
    'RecipeOp',
    'ReasoningRecipe',
    'TemplateRecipe',
]
