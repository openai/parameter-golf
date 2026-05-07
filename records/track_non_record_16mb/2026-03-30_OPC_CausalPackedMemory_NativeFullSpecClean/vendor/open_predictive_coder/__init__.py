"""Public API for the open_predictive_coder kernel.

The package is intentionally organized in layers:

1. foundational types and configs
2. reusable kernel primitives for substrates, control, memory, views, readouts, and runtime
3. the shared adapter layer plus the first concrete byte-latent adapter

The full package map and layer boundary are documented in `docs/architecture.md`.
"""

# Foundation and configuration surfaces.
from .artifacts import (
    ArtifactAccounting,
    ArtifactMetadata,
    coerce_artifact_metadata,
    make_artifact_accounting,
    make_replay_span,
    ReplaySpan,
)
from .artifacts_audits import ArtifactAuditRecord, ArtifactAuditSummary, audit_artifact, summarize_artifact_audits
from .codecs import ByteCodec, ensure_byte_tokens, ensure_tokens
from .config import (
    ByteLatentPredictiveCoderConfig,
    DelayLineConfig,
    HierarchicalSubstrateConfig,
    LatentConfig,
    LatentControllerConfig,
    LinearMemoryConfig,
    MemoryMergeMode,
    MixedMemoryConfig,
    OscillatoryMemoryConfig,
    OpenPredictiveCoderConfig,
    ReservoirConfig,
    ReservoirTopology,
    SampledReadoutBandConfig,
    SampledReadoutConfig,
    SegmenterConfig,
    SegmenterMode,
    SubstrateKind,
)

# Control, routing, modulation, and predictive side channels.
from .control import (
    ControllerSummary,
    ControllerSummaryBuilder,
    ControllerSummaryConfig,
    stack_summaries,
)
from .controllers import (
    PredictiveController,
    PredictiveObservation,
    PredictiveState,
)
from .gating import PathwayGateConfig, PathwayGateController, PathwayGateState, PathwayGateValues
from .modulation import HormoneModulationConfig, HormoneModulator, HormoneState
from .oracle_analysis import (
    OracleAnalysisAdapter,
    OracleAnalysisConfig,
    OracleAnalysisFitReport,
    OracleAnalysisPoint,
    OracleAnalysisReport,
)
from .predictive_surprise import PredictionState, PredictiveSurpriseConfig, PredictiveSurpriseController, SummaryMode
from .routing import RoutingConfig, RoutingDecision, RoutingMode, SummaryRouter

# Memory, latent, and feature-view primitives.
from .bidirectional_context import (
    BidirectionalContextConfig,
    BidirectionalContextLeaveOneOutStats,
    BidirectionalContextNeighborhood,
    BidirectionalContextProbe,
    BidirectionalContextStats,
)
from .bridge_export import (
    BridgeExportAdapter,
    BridgeExportConfig,
    BridgeExportFitReport,
    BridgeExportReport,
)
from .bridge_features import BridgeFeatureArrays, BridgeFeatureConfig, bridge_feature_arrays
from .exact_context import (
    ExactContextConfig,
    ExactContextFitReport,
    ExactContextMemory,
    ExactContextPrediction,
    SupportBlend,
    SupportMixConfig,
    SupportWeightedMixer,
)
from .latents import LatentCommitter, LatentObservation, LatentState
from .learned_segmentation import (
    BoundaryDecision,
    BoundaryFeatures,
    BoundaryScorerConfig,
    LearnedBoundaryScorer,
    LearnedSegmenter,
    LearnedSegmenterConfig,
)
from .hierarchical_views import HierarchicalFeatureView, HierarchicalSummary
from .linear_views import LinearMemoryFeatureView
from .ngram_memory import NgramMemory, NgramMemoryConfig, NgramMemoryReport
from .noncausal_reconstructive import (
    NoncausalReconstructiveAdapter,
    NoncausalReconstructiveConfig,
    NoncausalReconstructiveFitReport,
    NoncausalReconstructiveReport,
    NoncausalReconstructiveTrace,
)
from .patch_latent_blocks import (
    GlobalLocalBridge,
    GlobalLocalBridgeConfig,
    LocalByteEncoder,
    LocalByteEncoderConfig,
    PatchPooler,
    PatchPoolerConfig,
)
from .probability_diagnostics import (
    ProbabilityDiagnostics,
    ProbabilityDiagnosticsConfig,
    normalized_entropy,
    overlap_mass,
    probability_diagnostics,
    shared_top_k_mass,
    top1_agreement,
    top1_peak,
    top2_margin,
    top_k_mass,
)
from .sampled_readout import SampledBandSummary, SampledMultiscaleReadout
from .span_selection import ScoredSpan, SpanSelectionConfig, replay_spans_from_scores, select_scored_spans
from .views import ByteLatentFeatureView

# Substrates, factories, and presets.
from .factories import (
    create_delay_line_substrate,
    create_echo_state_substrate,
    create_hierarchical_substrate,
    create_mixed_memory_substrate,
    create_oscillatory_memory_substrate,
    create_substrate,
    create_substrate_for_model,
)
from .presets import delay_small, echo_state_small, hierarchical_small, mixed_memory_small
from .segmenters import AdaptiveSegmenter, SegmentStats
from .substrates import (
    DelayLineSubstrate,
    EchoStateSubstrate,
    HierarchicalSubstrate,
    LinearMemorySubstrate,
    MixedMemorySubstrate,
    OscillatoryMemorySubstrate,
    TokenSubstrate,
)

# Readouts, experts, datasets, and runtime surfaces.
from .datasets import ByteSequenceDataset
from .eval import NextStepScore, RolloutEvaluation, RolloutMode, evaluate_rollout, score_next_step
from .experts import ExpertFitReport, ExpertScore, FrozenReadoutExpert
from .metrics import (
    bits_per_byte_from_logits,
    bits_per_byte_from_probabilities,
    bits_per_token_from_logits,
    bits_per_token_from_probabilities,
)
from .readouts import RidgeReadout
from .runtime import (
    CausalFitReport,
    CausalSequenceReport,
    CausalTrace,
    FitReport,
    SequenceReport,
    SequenceTrace,
    tag_metadata,
)
from .train_modes import TrainModeConfig, TrainStateMode
from .train_eval import (
    DatasetEvaluation,
    RolloutCheckpoint,
    RolloutCurve,
    RolloutCurveMode,
    RolloutCurveEvaluation,
    RolloutCurvePoint,
    TransferEvaluation,
    TransferProbeReport,
    evaluate_dataset,
    evaluate_rollout_curve,
    evaluate_transfer_probe,
    score_dataset,
)

# Concrete adapter surface.
from .causal_predictive import CausalPredictiveAdapter, CausalPredictiveFitReport, CausalPredictiveScore
from .model import ByteLatentPredictiveCoder, OpenPredictiveCoder
from .teacher_export import TeacherExportAdapter, TeacherExportConfig, TeacherExportRecord, TeacherExportReport

__all__ = [
    "AdaptiveSegmenter",
    "ArtifactAccounting",
    "ArtifactAuditRecord",
    "ArtifactAuditSummary",
    "ArtifactMetadata",
    "audit_artifact",
    "coerce_artifact_metadata",
    "ByteCodec",
    "ensure_byte_tokens",
    "ByteLatentFeatureView",
    "ByteLatentPredictiveCoder",
    "ByteLatentPredictiveCoderConfig",
    "ByteSequenceDataset",
    "bits_per_byte_from_logits",
    "bits_per_byte_from_probabilities",
    "bits_per_token_from_logits",
    "bits_per_token_from_probabilities",
    "BidirectionalContextConfig",
    "BidirectionalContextLeaveOneOutStats",
    "BidirectionalContextNeighborhood",
    "BidirectionalContextProbe",
    "BidirectionalContextStats",
    "BridgeExportAdapter",
    "BridgeExportConfig",
    "BridgeExportFitReport",
    "BridgeExportReport",
    "BoundaryDecision",
    "BoundaryFeatures",
    "BoundaryScorerConfig",
    "BridgeFeatureArrays",
    "BridgeFeatureConfig",
    "bridge_feature_arrays",
    "CausalFitReport",
    "CausalPredictiveAdapter",
    "CausalPredictiveFitReport",
    "CausalPredictiveScore",
    "CausalSequenceReport",
    "CausalTrace",
    "ControllerSummary",
    "ControllerSummaryBuilder",
    "ControllerSummaryConfig",
    "SubstrateKind",
    "create_delay_line_substrate",
    "create_echo_state_substrate",
    "create_hierarchical_substrate",
    "create_mixed_memory_substrate",
    "create_oscillatory_memory_substrate",
    "create_substrate",
    "create_substrate_for_model",
    "DelayLineConfig",
    "DelayLineSubstrate",
    "delay_small",
    "DatasetEvaluation",
    "EchoStateSubstrate",
    "echo_state_small",
    "ExpertFitReport",
    "ExpertScore",
    "ExactContextConfig",
    "ExactContextFitReport",
    "ExactContextMemory",
    "ExactContextPrediction",
    "evaluate_rollout",
    "FitReport",
    "FrozenReadoutExpert",
    "HormoneModulationConfig",
    "HormoneModulator",
    "HormoneState",
    "HierarchicalFeatureView",
    "HierarchicalSummary",
    "HierarchicalSubstrate",
    "HierarchicalSubstrateConfig",
    "hierarchical_small",
    "GlobalLocalBridge",
    "GlobalLocalBridgeConfig",
    "LatentConfig",
    "LatentCommitter",
    "LatentControllerConfig",
    "LatentObservation",
    "LatentState",
    "LearnedBoundaryScorer",
    "LearnedSegmenter",
    "LearnedSegmenterConfig",
    "LinearMemoryConfig",
    "LinearMemoryFeatureView",
    "LinearMemorySubstrate",
    "LocalByteEncoder",
    "LocalByteEncoderConfig",
    "make_artifact_accounting",
    "make_replay_span",
    "MemoryMergeMode",
    "MixedMemoryConfig",
    "MixedMemorySubstrate",
    "mixed_memory_small",
    "NextStepScore",
    "NgramMemory",
    "NgramMemoryConfig",
    "NgramMemoryReport",
    "NoncausalReconstructiveAdapter",
    "NoncausalReconstructiveConfig",
    "NoncausalReconstructiveFitReport",
    "NoncausalReconstructiveReport",
    "NoncausalReconstructiveTrace",
    "OracleAnalysisAdapter",
    "OracleAnalysisConfig",
    "OracleAnalysisFitReport",
    "OracleAnalysisPoint",
    "OracleAnalysisReport",
    "OpenPredictiveCoder",
    "OpenPredictiveCoderConfig",
    "OscillatoryMemoryConfig",
    "OscillatoryMemorySubstrate",
    "PathwayGateConfig",
    "PathwayGateController",
    "PathwayGateState",
    "PathwayGateValues",
    "PatchPooler",
    "PatchPoolerConfig",
    "ProbabilityDiagnostics",
    "ProbabilityDiagnosticsConfig",
    "probability_diagnostics",
    "normalized_entropy",
    "overlap_mass",
    "shared_top_k_mass",
    "top1_agreement",
    "top1_peak",
    "top2_margin",
    "top_k_mass",
    "PredictiveController",
    "PredictiveObservation",
    "PredictiveState",
    "PredictionState",
    "PredictiveSurpriseConfig",
    "PredictiveSurpriseController",
    "ReplaySpan",
    "ReservoirConfig",
    "ReservoirTopology",
    "SampledBandSummary",
    "SampledMultiscaleReadout",
    "SampledReadoutBandConfig",
    "SampledReadoutConfig",
    "ScoredSpan",
    "RidgeReadout",
    "RolloutCurveEvaluation",
    "RolloutCurve",
    "RolloutCheckpoint",
    "RolloutCurveMode",
    "RolloutCurvePoint",
    "RolloutEvaluation",
    "RolloutMode",
    "RoutingConfig",
    "RoutingDecision",
    "RoutingMode",
    "SegmenterConfig",
    "SegmenterMode",
    "SegmentStats",
    "SequenceReport",
    "SequenceTrace",
    "score_next_step",
    "tag_metadata",
    "TeacherExportAdapter",
    "TeacherExportConfig",
    "TeacherExportRecord",
    "TeacherExportReport",
    "TransferProbeReport",
    "TransferEvaluation",
    "TokenSubstrate",
    "ensure_tokens",
    "evaluate_dataset",
    "evaluate_rollout_curve",
    "evaluate_transfer_probe",
    "score_dataset",
    "select_scored_spans",
    "SpanSelectionConfig",
    "stack_summaries",
    "SupportBlend",
    "SupportMixConfig",
    "SupportWeightedMixer",
    "summarize_artifact_audits",
    "SummaryMode",
    "SummaryRouter",
    "TrainModeConfig",
    "TrainStateMode",
    "replay_spans_from_scores",
]
