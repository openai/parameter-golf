"""
Test LTX Training Pipeline - Comprehensive Tests for HDC Transfer Learning

This test module validates the full training pipeline including:
1. Safety training integration
2. Resonator network training
3. Recipe and seed saving
4. Model merging and loading
5. Incremental training for additional modalities

Usage:
    python test_ltx_training_pipeline.py
    python test_ltx_training_pipeline.py --quick
    python test_ltx_training_pipeline.py --full
"""

import os
import sys
import json
import time
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add workspace to path for imports
workspace_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(workspace_path))

# Add current directory for direct imports
sys.path.insert(0, str(Path(__file__).parent))

# Try to import from package, fall back to direct imports
try:
    from HDC_ONLY_Model.Hdc_Sparse.HDC_Transfer_Learning_Instant.LTX_Model_Transfer_Learning_Instant.ltx_training_pipeline import (
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
    from HDC_ONLY_Model.Hdc_Sparse.HDC_Core_Model.Recipes_Seeds.walsh_hadamard_core import (
        WalshHadamardBasis,
        TernaryHadamardEncoder,
        DEFAULT_HDC_DIM
    )
    from HDC_ONLY_Model.Hdc_Sparse.HDC_Core_Model.Recipes_Seeds.recipe_storage import (
        IdentityRecipe,
        RecipeStorage
    )
    from HDC_ONLY_Model.Hdc_Sparse.HDC_Core_Model.HDC_Core_Main.hdc_sparse_core import (
        seed_to_hypervector_blake3,
        seed_string_to_int
    )
except ImportError:
    # Fall back to direct imports for testing
    from ltx_training_pipeline import (
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


class TestResults:
    """Container for test results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.timings = {}
    
    def add_pass(self, test_name: str, timing: float = 0.0):
        self.passed += 1
        self.timings[test_name] = timing
        print(f"  ✓ {test_name} ({timing:.2f}s)")
    
    def add_fail(self, test_name: str, error: str, timing: float = 0.0):
        self.failed += 1
        self.errors.append((test_name, error))
        self.timings[test_name] = timing
        print(f"  ✗ {test_name} ({timing:.2f}s)")
        print(f"    Error: {error}")
    
    def summary(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Summary: {self.passed}/{total} passed")
        print(f"{'='*60}")
        
        if self.errors:
            print("\nFailed Tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        
        return self.failed == 0


def test_config_creation():
    """Test configuration creation and serialization."""
    print("\nTest: Configuration Creation")
    
    # Test default config
    config = TrainingConfig()
    assert config.hdc_dim == DEFAULT_HDC_DIM
    assert config.safety.enable_safety_training == True
    assert config.resonator.enable_resonator == True
    
    # Test custom config
    config = TrainingConfig(
        model_path="/custom/path",
        output_path="/custom/output",
        hdc_dim=262144,
        safety=SafetyTrainingConfig(
            enable_safety_training=False
        ),
        resonator=ResonatorTrainingConfig(
            max_iterations=50
        )
    )
    assert config.hdc_dim == 262144
    assert config.safety.enable_safety_training == False
    assert config.resonator.max_iterations == 50
    
    # Test serialization
    config_dict = config.to_dict()
    assert 'safety' in config_dict
    assert 'resonator' in config_dict
    
    # Test deserialization
    config2 = TrainingConfig.from_dict(config_dict)
    assert config2.hdc_dim == config.hdc_dim
    assert config2.safety.enable_safety_training == config.safety.enable_safety_training
    
    return True


def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("\nTest: Pipeline Initialization")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,  # Use DEFAULT_HDC_DIM for consistency
            safety=SafetyTrainingConfig(enable_safety_training=True),
            resonator=ResonatorTrainingConfig(enable_resonator=True)
        )
        
        pipeline = LTXTrainingPipeline(config)
        
        # Check initialization
        assert pipeline.current_phase == TrainingPhase.INITIALIZATION
        assert pipeline.hadamard is not None
        assert pipeline.ternary_encoder is not None
        assert pipeline.resonator is not None
        assert pipeline.role_binding is not None
        assert pipeline.recipe_storage is not None
        assert pipeline.chain_storage is not None
        
        # Check HDC components
        assert pipeline.hadamard.dim == config.hdc_dim
        
        # Check storage paths
        assert Path(pipeline.recipe_storage.base_path).exists()
        assert Path(pipeline.chain_storage.storage_path).exists()
        
    return True


def test_safety_training():
    """Test safety training phase."""
    print("\nTest: Safety Training Phase")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            safety=SafetyTrainingConfig(
                enable_safety_training=True,
                context_type="general",
                block_critical=True,
                block_high=True
            )
        )
        
        pipeline = LTXTrainingPipeline(config)
        results = pipeline.run_safety_training()
        
        # Check results
        assert 'enabled' in results
        if results['enabled']:
            assert 'blocked_seeds' in results
            assert 'redirections' in results
            assert pipeline._safety_blocked_seeds is not None
        
    return True


def test_latent_extraction():
    """Test latent extraction phase."""
    print("\nTest: Latent Extraction Phase")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block", "audio_transformer_block"],
            timesteps=[1000, 500, 0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=False)  # Disable for simpler test
        )
        
        pipeline = LTXTrainingPipeline(config)
        
        # Run safety training first (required phase)
        pipeline.run_safety_training()
        
        # Run latent extraction
        results = pipeline.run_latent_extraction()
        
        # Check results
        assert 'patterns_extracted' in results
        assert 'layers_processed' in results
        assert results['layers_processed'] == 2
        assert len(pipeline._extracted_patterns) > 0
        
        # Check pattern structure
        pattern = pipeline._extracted_patterns[0]
        assert pattern.hdc_vector is not None
        assert pattern.seed_string is not None
        
    return True


def test_hdc_projection():
    """Test HDC projection phase."""
    print("\nTest: HDC Projection Phase")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block"],
            timesteps=[0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=False)
        )
        
        pipeline = LTXTrainingPipeline(config)
        pipeline.run_safety_training()
        pipeline.run_latent_extraction()
        
        # Run HDC projection
        results = pipeline.run_hdc_projection()
        
        # Check results
        assert 'patterns_projected' in results
        assert results['patterns_projected'] > 0
        assert len(pipeline._projected_patterns) > 0
        
        # Check projection
        pattern = pipeline._projected_patterns[0]
        assert pattern.hdc_vector is not None
        # HDC vectors are now uint64 packed for efficient XOR operations
        assert pattern.hdc_vector.dtype == np.uint64 or pattern.hdc_vector.dtype == np.int8
        
    return True


def test_pattern_deduplication():
    """Test pattern deduplication phase."""
    print("\nTest: Pattern Deduplication Phase")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block"],
            timesteps=[0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=False),
            storage=RecipeStorageConfig(deduplication_threshold=0.95)
        )
        
        pipeline = LTXTrainingPipeline(config)
        pipeline.run_safety_training()
        pipeline.run_latent_extraction()
        pipeline.run_hdc_projection()
        
        # Run deduplication
        results = pipeline.run_pattern_deduplication()
        
        # Check results
        assert 'patterns_processed' in results
        assert 'unique_patterns' in results
        assert len(pipeline._unique_patterns) > 0
        
    return True


def test_resonator_training():
    """Test resonator training phase."""
    print("\nTest: Resonator Training Phase")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block", "audio_transformer_block"],
            timesteps=[0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=False),
            resonator=ResonatorTrainingConfig(
                enable_resonator=True,
                max_iterations=50
            )
        )
        
        pipeline = LTXTrainingPipeline(config)
        pipeline.run_safety_training()
        pipeline.run_latent_extraction()
        pipeline.run_hdc_projection()
        pipeline.run_pattern_deduplication()
        
        # Run resonator training
        results = pipeline.run_resonator_training()
        
        # Check results
        assert 'enabled' in results
        if results['enabled']:
            assert 'patterns_factorized' in results
            assert 'converged' in results
            assert 'avg_iterations' in results
        
    return True


def test_recipe_generation():
    """Test recipe generation phase."""
    print("\nTest: Recipe Generation Phase")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block"],
            timesteps=[0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=False)
        )
        
        pipeline = LTXTrainingPipeline(config)
        pipeline.run_safety_training()
        pipeline.run_latent_extraction()
        pipeline.run_hdc_projection()
        pipeline.run_pattern_deduplication()
        pipeline.run_resonator_training()
        
        # Run recipe generation
        results = pipeline.run_recipe_generation()
        
        # Check results
        assert 'recipes_created' in results
        assert results['recipes_created'] > 0
        assert len(pipeline._recipe_ids) > 0
        
        # Verify recipes are stored
        for recipe_id in pipeline._recipe_ids[:3]:  # Check first 3
            recipe = pipeline.recipe_storage.load_recipe(recipe_id)
            assert recipe is not None
            assert recipe.verify_integrity()
        
    return True


def test_model_merging():
    """Test model merging phase."""
    print("\nTest: Model Merging Phase")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block"],
            timesteps=[0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=False)
        )
        
        pipeline = LTXTrainingPipeline(config)
        pipeline.run_safety_training()
        pipeline.run_latent_extraction()
        pipeline.run_hdc_projection()
        pipeline.run_pattern_deduplication()
        pipeline.run_resonator_training()
        pipeline.run_recipe_generation()
        
        # Run model merging
        results = pipeline.run_model_merging()
        
        # Check results
        assert results['model_created'] == True
        assert pipeline.merged_model is not None
        assert pipeline.merged_model.model_id is not None
        assert 'video' in pipeline.merged_model.supported_modalities
        assert 'audio' in pipeline.merged_model.supported_modalities
        
    return True


def test_validation():
    """Test validation phase."""
    print("\nTest: Validation Phase")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block"],
            timesteps=[0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=False)
        )
        
        pipeline = LTXTrainingPipeline(config)
        pipeline.run_safety_training()
        pipeline.run_latent_extraction()
        pipeline.run_hdc_projection()
        pipeline.run_pattern_deduplication()
        pipeline.run_resonator_training()
        pipeline.run_recipe_generation()
        pipeline.run_model_merging()
        
        # Run validation
        results = pipeline.run_validation()
        
        # Check results
        assert 'recipes_valid' in results
        assert 'overall_valid' in results
        assert results['overall_valid'] == True
        
    return True


def test_full_training():
    """Test full training pipeline."""
    print("\nTest: Full Training Pipeline")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block", "audio_transformer_block"],
            timesteps=[1000, 500, 0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=True),
            resonator=ResonatorTrainingConfig(enable_resonator=True)
        )
        
        pipeline = LTXTrainingPipeline(config)
        results = pipeline.run_full_training()
        
        # Check all phases completed
        assert 'safety_training' in results
        assert 'latent_extraction' in results
        assert 'hdc_projection' in results
        assert 'pattern_deduplication' in results
        assert 'resonator_training' in results
        assert 'recipe_generation' in results
        assert 'model_merging' in results
        assert 'validation' in results
        
        # Check statistics
        assert 'statistics' in results
        assert results['statistics']['total_patterns_extracted'] > 0
        assert results['statistics']['total_recipes_created'] > 0
        
        # Check pipeline state
        assert pipeline.current_phase == TrainingPhase.COMPLETED
        assert pipeline.merged_model is not None
        
    return True


def test_model_save_load():
    """Test model saving and loading."""
    print("\nTest: Model Save and Load")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output"
        save_path = Path(tmpdir) / "saved_model"
        
        # Create and train model
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=str(output_path),
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block"],
            timesteps=[0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=False)
        )
        
        pipeline = LTXTrainingPipeline(config)
        pipeline.run_full_training()
        
        original_model_id = pipeline.merged_model.model_id
        
        # Save model
        manifest_path = pipeline.save_merged_model(str(save_path))
        assert Path(manifest_path).exists()
        
        # Load model
        loaded_pipeline = LTXTrainingPipeline.load_merged_model(str(save_path))
        assert loaded_pipeline.merged_model.model_id == original_model_id
        assert loaded_pipeline.merged_model.hdc_dim == config.hdc_dim
        
    return True


def test_incremental_training():
    """Test incremental training with additional modalities."""
    print("\nTest: Incremental Training")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block"],
            timesteps=[0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=False),
            enable_incremental_training=True
        )
        
        pipeline = LTXTrainingPipeline(config)
        pipeline.run_full_training()
        
        original_recipe_count = pipeline.stats.total_recipes_created
        
        # Add new modality
        training_data = [
            {'pattern': f'emotion_{i}', 'intensity': 0.5}
            for i in range(10)
        ]
        
        results = pipeline.add_modality_training(
            modality_name="audio_emotion",
            training_data=training_data
        )
        
        # Check results
        assert results['success'] == True
        assert results['recipes_added'] > 0
        assert 'audio_emotion' in pipeline.merged_model.supported_modalities
        
    return True


def test_convenience_functions():
    """Test convenience functions."""
    print("\nTest: Convenience Functions")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test create_training_pipeline
        pipeline = create_training_pipeline(
            model_path="/dummy/model",
            output_path=tmpdir,
            hdc_dim=DEFAULT_HDC_DIM,
            enable_safety=True,
            enable_resonator=True
        )
        
        assert pipeline is not None
        # Use DEFAULT_HDC_DIM for consistency (1048576)
        assert pipeline.config.hdc_dim == DEFAULT_HDC_DIM
        
        # Test run_quick_training
        pipeline2, results = run_quick_training(
            model_path="/dummy/model",
            output_path=str(Path(tmpdir) / "quick")
        )
        
        assert pipeline2 is not None
        assert results is not None
        assert 'total_time' in results
        
    return True


def test_determinism():
    """Test that the pipeline is deterministic."""
    print("\nTest: Determinism")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run training twice with same config
        config = TrainingConfig(
            model_path="/dummy/model",
            output_path=str(Path(tmpdir) / "run1"),
            hdc_dim=DEFAULT_HDC_DIM,
            extraction_layers=["video_transformer_block"],
            timesteps=[0],
            generation_modes=["text_to_audio_video"],
            safety=SafetyTrainingConfig(enable_safety_training=False)
        )
        
        pipeline1 = LTXTrainingPipeline(config)
        results1 = pipeline1.run_full_training()
        
        # Second run with same config
        config.output_path = str(Path(tmpdir) / "run2")
        pipeline2 = LTXTrainingPipeline(config)
        results2 = pipeline2.run_full_training()
        
        # Check determinism
        assert results1['statistics']['total_patterns_extracted'] == results2['statistics']['total_patterns_extracted']
        assert results1['statistics']['total_recipes_created'] == results2['statistics']['total_recipes_created']
        
    return True


def run_tests(quick_mode: bool = False):
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LTX Training Pipeline Tests")
    print("=" * 60)
    
    results = TestResults()
    
    # Core tests
    tests = [
        ("Config Creation", test_config_creation),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Safety Training", test_safety_training),
        ("Latent Extraction", test_latent_extraction),
        ("HDC Projection", test_hdc_projection),
        ("Pattern Deduplication", test_pattern_deduplication),
        ("Resonator Training", test_resonator_training),
        ("Recipe Generation", test_recipe_generation),
        ("Model Merging", test_model_merging),
        ("Validation", test_validation),
    ]
    
    # Add extended tests if not in quick mode
    if not quick_mode:
        tests.extend([
            ("Full Training", test_full_training),
            ("Model Save/Load", test_model_save_load),
            ("Incremental Training", test_incremental_training),
            ("Convenience Functions", test_convenience_functions),
            ("Determinism", test_determinism),
        ])
    
    for name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            
            if result:
                results.add_pass(name, elapsed)
            else:
                results.add_fail(name, "Test returned False", elapsed)
        except Exception as e:
            elapsed = time.time() - start_time
            results.add_fail(name, str(e), elapsed)
    
    return results.summary()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LTX Training Pipeline")
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--full', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    success = run_tests(quick_mode=args.quick and not args.full)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
