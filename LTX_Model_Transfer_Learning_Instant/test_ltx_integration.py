"""
Test LTX Integration - Tests for LTX-2.3 Instant Transfer Learning

This module provides comprehensive tests for the LTX instant transfer learning
integration with the Pure HDC/VSA Engine.

Run with:
    python -m pytest test_ltx_integration.py -v
    # or
    python test_ltx_integration.py
"""

import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Add workspace to path for imports
workspace_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(workspace_path))

# Add current directory for direct imports
sys.path.insert(0, str(Path(__file__).parent))

# Try to import from package, fall back to direct imports
try:
    from HDC_ONLY_Model.Hdc_Sparse.HDC_Transfer_Learning_Instant.LTX_Model_Transfer_Learning_Instant.ltx_latent_mapper import (
        LTXLatentMapper,
        LTXConfig,
        LTXLayerType,
        LTXGenerationMode,
        AudioVideoPattern,
        create_ltx_mapper,
        LTXResonatorFactorizer,
        LTXRecipeDiscovery,
        LTXDifficultyBudgeter,
        LTXEpisodicMemory
    )
    from HDC_ONLY_Model.Hdc_Sparse.HDC_Transfer_Learning_Instant.LTX_Model_Transfer_Learning_Instant.ltx_chain_seeds import (
        LTXChainStorage,
        LTXChainSeed,
        LTXSeedStep,
        LTXChainOperation,
        LTXChainReconstructor,
        LTXChainSynthesizer,
        create_ltx_chain_system,
        create_ltx_chain_from_trajectory
    )
    from HDC_ONLY_Model.Hdc_Sparse.HDC_Transfer_Learning_Instant.LTX_Model_Transfer_Learning_Instant.ltx_relationship_deduplication import (
        LTXPatternDeduplicator,
        LTXDeduplicationConfig,
        LTXRelationshipGraph,
        LTXRelationshipType,
        LTXRelationshipEdge,
        LTXPattern,
        create_ltx_deduplicator
    )
except ImportError:
    # Fall back to direct imports for testing
    from ltx_latent_mapper import (
        LTXLatentMapper,
        LTXConfig,
        LTXLayerType,
        LTXGenerationMode,
        AudioVideoPattern,
        create_ltx_mapper,
        LTXResonatorFactorizer,
        LTXRecipeDiscovery,
        LTXDifficultyBudgeter,
        LTXEpisodicMemory
    )
    from ltx_chain_seeds import (
        LTXChainStorage,
        LTXChainSeed,
        LTXSeedStep,
        LTXChainOperation,
        LTXChainReconstructor,
        LTXChainSynthesizer,
        create_ltx_chain_system,
        create_ltx_chain_from_trajectory
    )
    from ltx_relationship_deduplication import (
        LTXPatternDeduplicator,
        LTXDeduplicationConfig,
        LTXRelationshipGraph,
        LTXRelationshipType,
        LTXRelationshipEdge,
        LTXPattern,
        create_ltx_deduplicator
    )


class TestLTXConfig:
    """Test LTXConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LTXConfig()
        
        assert config.model_name == "LTX-2.3"
        assert config.model_size == "22B"
        assert config.hdc_dim > 0
        assert config.storage_path == "./ltx_recipes"
        assert config.deduplication_threshold == 0.95
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LTXConfig(
            hdc_dim=262144,
            model_name="LTX-Custom",
            storage_path="/custom/path"
        )
        
        assert config.hdc_dim == 262144
        assert config.model_name == "LTX-Custom"
        assert config.storage_path == "/custom/path"


class TestLTXLatentMapper:
    """Test LTXLatentMapper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = LTXConfig(
            hdc_dim=131072,  # Use smaller dimension for tests
            storage_path=self.temp_dir
        )
        self.mapper = LTXLatentMapper(config=self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_mapper_initialization(self):
        """Test mapper initializes correctly."""
        assert self.mapper is not None
        assert self.mapper.config.hdc_dim == 131072
        assert self.mapper.hadamard is not None
        assert self.mapper.ternary_encoder is not None
    
    def test_deterministic_vector_generation(self):
        """Test that seed generates deterministic vectors."""
        seed1 = b"test_seed_1"
        seed2 = b"test_seed_1"
        
        vec1 = self.mapper._generate_deterministic_features(seed1)
        vec2 = self.mapper._generate_deterministic_features(seed2)
        
        assert np.array_equal(vec1, vec2), "Same seed should produce same vector"
    
    def test_different_seeds_different_vectors(self):
        """Test that different seeds produce different vectors."""
        seed1 = b"test_seed_1"
        seed2 = b"test_seed_2"
        
        vec1 = self.mapper._generate_deterministic_features(seed1)
        vec2 = self.mapper._generate_deterministic_features(seed2)
        
        assert not np.array_equal(vec1, vec2), "Different seeds should produce different vectors"
    
    def test_hadamard_projection(self):
        """Test Hadamard projection."""
        # Create test latent
        latent = np.random.randn(1, 4096).astype(np.float32)
        
        # Project
        hdc_vec = self.mapper._project_hadamard(latent)
        
        assert hdc_vec is not None
        assert hdc_vec.dtype == np.uint64
        # The function returns shape (batch_size, uint64_count) for batch input
        # or (uint64_count,) for single 1D input
        if hdc_vec.ndim == 2:
            assert hdc_vec.shape[1] == self.config.uint64_count
        else:
            assert len(hdc_vec) == self.config.uint64_count
    
    def test_bind_audio_video(self):
        """Test audio-video binding."""
        video_vec = self.mapper._generate_deterministic_features(b"video")
        audio_vec = self.mapper._generate_deterministic_features(b"audio")
        
        bound = self.mapper.bind_audio_video(video_vec, audio_vec)
        
        assert bound is not None
        assert bound.shape == video_vec.shape
    
    def test_unbind_video(self):
        """Test unbinding video from bound vector."""
        video_vec = self.mapper._generate_deterministic_features(b"video")
        audio_vec = self.mapper._generate_deterministic_features(b"audio")
        
        bound = self.mapper.bind_audio_video(video_vec, audio_vec)
        retrieved_video = self.mapper.unbind_video(bound, audio_vec)
        
        # XOR is reversible
        assert np.array_equal(retrieved_video, video_vec)
    
    def test_unbind_audio(self):
        """Test unbinding audio from bound vector."""
        video_vec = self.mapper._generate_deterministic_features(b"video")
        audio_vec = self.mapper._generate_deterministic_features(b"audio")
        
        bound = self.mapper.bind_audio_video(video_vec, audio_vec)
        retrieved_audio = self.mapper.unbind_audio(bound, video_vec)
        
        assert np.array_equal(retrieved_audio, audio_vec)
    
    def test_similarity(self):
        """Test similarity calculation."""
        vec1 = self.mapper._generate_deterministic_features(b"test1")
        vec2 = self.mapper._generate_deterministic_features(b"test1")
        vec3 = self.mapper._generate_deterministic_features(b"test_different")
        
        sim_same = self.mapper.similarity(vec1, vec2)
        sim_diff = self.mapper.similarity(vec1, vec3)
        
        assert sim_same == 1.0, "Same vectors should have similarity 1.0"
        assert 0.4 <= sim_diff <= 0.6, "Different vectors should have ~0.5 similarity"
    
    def test_store_as_recipes(self):
        """Test storing HDC vectors as recipes."""
        hdc_vectors = {
            'layer1': self.mapper._generate_deterministic_features(b"layer1"),
            'layer2': self.mapper._generate_deterministic_features(b"layer2")
        }
        
        metadata = {
            'generation_mode': 'text_to_audio_video',
            'timestep': 500
        }
        
        recipe_ids, chain = self.mapper.store_as_recipes(hdc_vectors, metadata)
        
        assert len(recipe_ids) == 2
        assert chain is not None
        assert chain.model_name == "LTX-2.3"


class TestLTXChainSeeds:
    """Test LTX chain seed functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.hdc_dim = 131072
        self.storage = LTXChainStorage(
            storage_path=self.temp_dir,
            hdc_dim=self.hdc_dim
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_step(self):
        """Test creating a chain step."""
        step = LTXSeedStep(
            step_id="step_0",
            seed=42,
            hadamard_index=0,
            operation=LTXChainOperation.BIND,
            weight=1.0,
            layer_name="video_block_0",
            timestep=500
        )
        
        assert step.step_id == "step_0"
        assert step.seed == 42
        assert step.operation == LTXChainOperation.BIND
    
    def test_create_chain(self):
        """Test creating a chain."""
        steps = [
            LTXSeedStep(
                step_id=f"step_{i}",
                seed=i,
                hadamard_index=i,
                operation=LTXChainOperation.BIND,
                weight=1.0
            )
            for i in range(5)
        ]
        
        chain = LTXChainSeed(
            chain_id="test_chain",
            model_name="LTX-2.3",
            generation_mode="text_to_audio_video",
            steps=steps
        )
        
        assert chain.chain_id == "test_chain"
        assert len(chain.steps) == 5
    
    def test_save_and_load_chain(self):
        """Test saving and loading a chain."""
        steps = [
            LTXSeedStep(
                step_id=f"step_{i}",
                seed=i,
                hadamard_index=i,
                operation=LTXChainOperation.BIND,
                weight=1.0
            )
            for i in range(3)
        ]
        
        chain = LTXChainSeed(
            chain_id="test_chain_save",
            model_name="LTX-2.3",
            generation_mode="text_to_audio_video",
            steps=steps
        )
        
        # Save
        chain_id = self.storage.save_chain(chain)
        assert chain_id == "test_chain_save"
        
        # Load
        loaded = self.storage.load_chain(chain_id)
        assert loaded is not None
        assert loaded.chain_id == chain.chain_id
        assert len(loaded.steps) == len(chain.steps)
    
    def test_reconstruct_chain(self):
        """Test reconstructing vectors from chain."""
        reconstructor = LTXChainReconstructor(hdc_dim=self.hdc_dim)
        
        steps = [
            LTXSeedStep(
                step_id=f"step_{i}",
                seed=i * 1000,
                hadamard_index=i,
                operation=LTXChainOperation.BIND,
                weight=1.0
            )
            for i in range(3)
        ]
        
        chain = LTXChainSeed(
            chain_id="test_reconstruct",
            model_name="LTX-2.3",
            generation_mode="text_to_audio_video",
            steps=steps
        )
        
        # Reconstruct sequential
        vectors = reconstructor.reconstruct_chain(chain, strategy="sequential")
        assert len(vectors) == 3
        
        # Reconstruct bind
        bound = reconstructor.reconstruct_chain(chain, strategy="bind")
        assert bound is not None
        assert bound.shape[0] == self.hdc_dim // 64
    
    def test_merge_chains(self):
        """Test merging chains."""
        synthesizer = LTXChainSynthesizer(self.storage, self.hdc_dim)
        
        # Create two chains
        chain1 = LTXChainSeed(
            chain_id="chain1",
            model_name="LTX-2.3",
            generation_mode="text_to_audio_video",
            steps=[
                LTXSeedStep(
                    step_id=f"step_{i}",
                    seed=i,
                    hadamard_index=i,
                    operation=LTXChainOperation.BIND,
                    weight=1.0
                )
                for i in range(3)
            ]
        )
        
        chain2 = LTXChainSeed(
            chain_id="chain2",
            model_name="LTX-2.3",
            generation_mode="text_to_audio_video",
            steps=[
                LTXSeedStep(
                    step_id=f"step_{i}",
                    seed=i + 100,
                    hadamard_index=i,
                    operation=LTXChainOperation.BIND,
                    weight=1.0
                )
                for i in range(3)
            ]
        )
        
        self.storage.save_chain(chain1)
        self.storage.save_chain(chain2)
        
        # Merge
        merged = synthesizer.merge_chains(["chain1", "chain2"], merge_strategy="concatenate")
        
        assert merged is not None
        assert len(merged.steps) == 6


class TestLTXRelationshipDeduplication:
    """Test LTX relationship deduplication."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = LTXDeduplicationConfig(
            similarity_threshold=0.95
        )
        self.deduplicator = LTXPatternDeduplicator(config=self.config)
        self.hdc_dim = 131072
        self.uint64_count = self.hdc_dim // 64
    
    def _create_random_vector(self, seed: int = None) -> np.ndarray:
        """Create a random HDC vector for testing."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.randint(0, 2**64, size=self.uint64_count, dtype=np.uint64)
    
    def test_deduplicate_new_pattern(self):
        """Test deduplicating a new pattern."""
        vector = self._create_random_vector(42)
        
        pattern, is_new, cluster_id = self.deduplicator.deduplicate(
            vector=vector,
            layer_name="video_block_0",
            seed_string="test:pattern:1",
            metadata={'timestep': 500}
        )
        
        assert pattern is not None
        assert is_new == True
        assert cluster_id is not None
    
    def test_deduplicate_exact_duplicate(self):
        """Test deduplicating an exact duplicate."""
        vector = self._create_random_vector(42)
        
        # First insertion
        pattern1, is_new1, _ = self.deduplicator.deduplicate(
            vector=vector,
            layer_name="video_block_0",
            seed_string="test:pattern:1"
        )
        
        # Second insertion (exact duplicate)
        pattern2, is_new2, _ = self.deduplicator.deduplicate(
            vector=vector,
            layer_name="video_block_0",
            seed_string="test:pattern:2"
        )
        
        assert is_new1 == True
        assert is_new2 == False  # Should be detected as duplicate
        assert pattern1.content_hash == pattern2.content_hash
    
    def test_find_similar(self):
        """Test finding similar patterns."""
        # Create base vector
        base_vector = self._create_random_vector(42)
        
        # Add base pattern
        self.deduplicator.deduplicate(
            vector=base_vector,
            layer_name="video_block_0",
            seed_string="test:base"
        )
        
        # Create similar vector (flip a few bits)
        similar_vector = base_vector.copy()
        # Use proper numpy uint64 XOR operation
        similar_vector[0] = np.bitwise_xor(similar_vector[0], np.uint64(0xFF))  # Flip 8 bits
        
        # Find similar
        similar = self.deduplicator.find_similar(similar_vector, threshold=0.9)
        
        assert len(similar) > 0
    
    def test_add_denoising_relationship(self):
        """Test adding denoising relationship."""
        vector1 = self._create_random_vector(42)
        vector2 = self._create_random_vector(43)
        
        pattern1, _, _ = self.deduplicator.deduplicate(
            vector=vector1,
            layer_name="video_block_0",
            seed_string="test:pattern:1",
            metadata={'timestep': 500}
        )
        
        pattern2, _, _ = self.deduplicator.deduplicate(
            vector=vector2,
            layer_name="video_block_0",
            seed_string="test:pattern:2",
            metadata={'timestep': 400}
        )
        
        self.deduplicator.add_denoising_relationship(
            source_id=pattern1.pattern_id,
            target_id=pattern2.pattern_id,
            timestep_from=500,
            timestep_to=400
        )
        
        trajectory = self.deduplicator.get_denoising_trajectory(pattern1.pattern_id)
        
        assert len(trajectory) >= 1
    
    def test_add_audio_video_sync(self):
        """Test adding audio-video sync relationship."""
        video_vec = self._create_random_vector(42)
        audio_vec = self._create_random_vector(43)
        
        video_pattern, _, _ = self.deduplicator.deduplicate(
            vector=video_vec,
            layer_name="video_block_0",
            seed_string="test:video",
            metadata={'modality': 'video'}
        )
        
        audio_pattern, _, _ = self.deduplicator.deduplicate(
            vector=audio_vec,
            layer_name="audio_block_0",
            seed_string="test:audio",
            metadata={'modality': 'audio'}
        )
        
        self.deduplicator.add_audio_video_sync(
            video_pattern_id=video_pattern.pattern_id,
            audio_pattern_id=audio_pattern.pattern_id
        )
        
        sync_patterns = self.deduplicator.get_audio_video_sync_patterns(video_pattern.pattern_id)
        
        assert len(sync_patterns['audio']) > 0


class TestLTXRelationshipGraph:
    """Test LTX relationship graph."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph = LTXRelationshipGraph()
    
    def test_add_edge(self):
        """Test adding an edge."""
        edge = LTXRelationshipEdge(
            source_id="pattern_1",
            target_id="pattern_2",
            relationship_type=LTXRelationshipType.SIMILAR,
            strength=0.9
        )
        
        self.graph.add_edge(edge)
        
        outgoing = self.graph.get_outgoing("pattern_1")
        assert len(outgoing) == 1
        assert outgoing[0].target_id == "pattern_2"
    
    def test_get_incoming(self):
        """Test getting incoming edges."""
        edge = LTXRelationshipEdge(
            source_id="pattern_1",
            target_id="pattern_2",
            relationship_type=LTXRelationshipType.DENOISES_TO
        )
        
        self.graph.add_edge(edge)
        
        incoming = self.graph.get_incoming("pattern_2")
        assert len(incoming) == 1
        assert incoming[0].source_id == "pattern_1"
    
    def test_find_path(self):
        """Test finding a path between patterns."""
        # Create a chain: 1 -> 2 -> 3
        self.graph.add_edge(LTXRelationshipEdge(
            source_id="pattern_1",
            target_id="pattern_2",
            relationship_type=LTXRelationshipType.DENOISES_TO
        ))
        self.graph.add_edge(LTXRelationshipEdge(
            source_id="pattern_2",
            target_id="pattern_3",
            relationship_type=LTXRelationshipType.DENOISES_TO
        ))
        
        path = self.graph.find_path("pattern_1", "pattern_3")
        
        assert path is not None
        assert path == ["pattern_1", "pattern_2", "pattern_3"]
    
    def test_get_denoising_trajectory(self):
        """Test getting denoising trajectory."""
        # Create trajectory: t=1000 -> t=500 -> t=0
        self.graph.add_edge(LTXRelationshipEdge(
            source_id="pattern_t1000",
            target_id="pattern_t500",
            relationship_type=LTXRelationshipType.DENOISES_TO,
            timestep_from=1000,
            timestep_to=500
        ))
        self.graph.add_edge(LTXRelationshipEdge(
            source_id="pattern_t500",
            target_id="pattern_t0",
            relationship_type=LTXRelationshipType.DENOISES_TO,
            timestep_from=500,
            timestep_to=0
        ))
        
        trajectory = self.graph.get_denoising_trajectory("pattern_t1000", target_timestep=0)
        
        assert len(trajectory) == 3
        assert trajectory[0] == "pattern_t1000"
        assert trajectory[-1] == "pattern_t0"


class TestLTXInstantTransfer:
    """Test LTX instant transfer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_ltx_mapper(self):
        """Test creating LTX mapper with convenience function."""
        mapper = create_ltx_mapper(
            hdc_dim=131072,
            storage_path=self.temp_dir
        )
        
        assert mapper is not None
        assert mapper.config.hdc_dim == 131072
    
    def test_create_chain_system(self):
        """Test creating chain system."""
        storage, reconstructor, synthesizer = create_ltx_chain_system(
            storage_path=self.temp_dir,
            hdc_dim=131072
        )
        
        assert storage is not None
        assert reconstructor is not None
        assert synthesizer is not None
    
    def test_create_chain_from_trajectory(self):
        """Test creating chain from trajectory."""
        chain = create_ltx_chain_from_trajectory(
            model_name="LTX-2.3",
            generation_mode="text_to_audio_video",
            timesteps=[1000, 500, 0],
            seeds=[42, 43, 44],
            hadamard_indices=[0, 1, 2],
            weights=[1.0, 0.8, 0.6]
        )
        
        assert chain is not None
        assert len(chain.steps) == 3
        assert chain.total_timesteps == 1000


class TestIntegration:
    """Integration tests for the complete LTX transfer pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.hdc_dim = 131072
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_transfer_pipeline(self):
        """Test the full transfer pipeline."""
        # Create mapper
        config = LTXConfig(
            hdc_dim=self.hdc_dim,
            storage_path=self.temp_dir
        )
        mapper = LTXLatentMapper(config=config)
        
        # Simulate latent extraction
        latents = {
            'video_block_0': np.random.randn(1, 4096).astype(np.float32),
            'audio_block_0': np.random.randn(1, 2048).astype(np.float32)
        }
        
        # Project to HDC
        hdc_vectors = mapper.project_to_hdc(latents)
        
        # Store as recipes
        metadata = {
            'generation_mode': 'text_to_audio_video',
            'timestep': 500
        }
        recipe_ids, chain = mapper.store_as_recipes(hdc_vectors, metadata)
        
        assert len(recipe_ids) == 2
        assert chain is not None
        
        # Verify chain can be loaded
        loaded_chain = mapper.chain_storage.load_chain(chain.chain_id)
        assert loaded_chain is not None
    
    def test_audio_video_binding_pipeline(self):
        """Test audio-video binding pipeline."""
        config = LTXConfig(
            hdc_dim=self.hdc_dim,
            storage_path=self.temp_dir
        )
        mapper = LTXLatentMapper(config=config)
        
        # Create video and audio vectors
        video_latent = np.random.randn(1, 4096).astype(np.float32)
        audio_latent = np.random.randn(1, 2048).astype(np.float32)
        
        video_hdc = mapper._project_hadamard(video_latent)
        audio_hdc = mapper._project_hadamard(audio_latent)
        
        # Bind
        bound = mapper.bind_audio_video(video_hdc, audio_hdc)
        
        # Unbind and verify
        retrieved_video = mapper.unbind_video(bound, audio_hdc)
        retrieved_audio = mapper.unbind_audio(bound, video_hdc)
        
        assert np.array_equal(retrieved_video, video_hdc)
        assert np.array_equal(retrieved_audio, audio_hdc)


def run_tests():
    """Run all tests manually."""
    print("=" * 60)
    print("LTX Integration Tests")
    print("=" * 60)
    
    test_classes = [
        TestLTXConfig,
        TestLTXLatentMapper,
        TestLTXChainSeeds,
        TestLTXRelationshipDeduplication,
        TestLTXRelationshipGraph,
        TestLTXInstantTransfer,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)
        
        # Run tests - create fresh instance for each test to ensure isolation
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                total_tests += 1
                instance = test_class()
                
                # Run setup before each test
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {str(e)}")
                    failed_tests += 1
                
                # Run teardown after each test
                if hasattr(instance, 'teardown_method'):
                    instance.teardown_method()
    
    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} passed, {failed_tests} failed")
    print("=" * 60)
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
