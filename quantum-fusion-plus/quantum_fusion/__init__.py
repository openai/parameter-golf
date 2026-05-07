"""
QUANTUM-FUSION-PLUS: 融合量化与递归架构的参数优化方案
"""

from .models import QuantumFusionGPT, TransformerBlock, MultiHeadAttention, FeedForward
from .quantization import (
    HadamardRotation,
    AWQQuantizer,
    LayerWiseQuantizer,
    HessianAwareCalibrator,
    QuantizationAwareTraining
)
from .training import (
    MuonOptimizer,
    WarmdownScheduler,
    EMAManager,
    QATTrainer,
    Trainer
)
from .inference import LegalTTT, KVLinCCache, InferenceEngine
from .data import FineWebDataset, DataLoaderFactory, create_dummy_batch
from .utils import (
    load_config,
    DotDict,
    setup_logging,
    calculate_model_size,
    calculate_parameters,
    calculate_flops,
    calculate_perplexity,
    calculate_bpb,
    MetricsTracker,
    set_seed
)

__version__ = "1.0.0"
__author__ = "Manus AI"

__all__ = [
    # Models
    'QuantumFusionGPT',
    'TransformerBlock',
    'MultiHeadAttention',
    'FeedForward',
    
    # Quantization
    'HadamardRotation',
    'AWQQuantizer',
    'LayerWiseQuantizer',
    'HessianAwareCalibrator',
    'QuantizationAwareTraining',
    
    # Training
    'MuonOptimizer',
    'WarmdownScheduler',
    'EMAManager',
    'QATTrainer',
    'Trainer',
    
    # Inference
    'LegalTTT',
    'KVLinCCache',
    'InferenceEngine',
    
    # Data
    'FineWebDataset',
    'DataLoaderFactory',
    'create_dummy_batch',
    
    # Utils
    'load_config',
    'DotDict',
    'setup_logging',
    'calculate_model_size',
    'calculate_parameters',
    'calculate_flops',
    'calculate_perplexity',
    'calculate_bpb',
    'MetricsTracker',
    'set_seed',
]
