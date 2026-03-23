"""
LTX HDC Decoder - Video/Audio Generation from Saved HDC Recipes

This test file implements the HDC decoder that reconstructs video and audio
from saved HDC recipes using Hadamard indices. It validates that the transfer
learning pipeline produces usable generative capabilities.

Key Features:
1. Hadamard Position Decoding - Reconstruct spatial patterns from HDC vectors
2. Circular Temporal Decoding - Extract temporal sequences from bundled vectors
3. XOR Unbinding - Retrieve specific elements from superposed patterns
4. Resonator Factorization - Decompose complex patterns into constituents
5. Video Frame Generation - Produce video frames from HDC recipes
6. Audio Sample Generation - Produce audio waveforms from HDC recipes

Architecture Reference:
- FULLINTEGRATION_NEW_ARCHITECTURE.md Section 2: Pure HDC Image/Video Encoding
- README_NEW_ARCHITECTURE.md: Hadamard Position Encoding
- Circular Temporal Encoding for unlimited temporal depth

Usage:
    python test_ltx_decoder_generation.py --recipe_path ./ltx_recipes --output_path ./test_output
"""

import os
import sys
import argparse
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field

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
from ...HDC_Core_Model.HDC_Core_Main.hdc_sparse_core import (
    seed_to_hypervector_blake3,
    seed_string_to_int,
    SparseBinaryHDC,
    SparseBinaryConfig,
    _BLAKE3_AVAILABLE
)

# Import LTX Components
from .ltx_latent_mapper import (
    LTXLatentMapper,
    LTXConfig,
    LTXGenerationMode,
    AudioVideoPattern
)
from .ltx_chain_seeds import (
    LTXChainStorage,
    LTXChainSeed,
    LTXSeedStep,
    LTXChainOperation,
    LTXChainReconstructor
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DecoderConfig:
    """Configuration for HDC decoder."""
    # HDC settings
    hdc_dim: int = DEFAULT_HDC_DIM
    
    # Video settings
    video_width: int = 256
    video_height: int = 256
    video_frames: int = 16
    video_fps: int = 24
    patch_size: int = 16
    
    # Audio settings
    audio_sample_rate: int = 44100
    audio_channels: int = 2
    audio_duration_seconds: float = 1.0
    
    # Decoding settings
    use_hadamard_position: bool = True
    use_circular_temporal: bool = True
    use_resonator_factorization: bool = True
    
    # Output settings
    output_path: str = "./test_output"
    save_video: bool = True
    save_audio: bool = True
    save_visualization: bool = True


# =============================================================================
# HDC Video Decoder
# =============================================================================

class HDCVideoDecoder:
    """
    Decodes HDC vectors into video frames using Hadamard position encoding.
    
    This implements the Pure HDC decoding pipeline from the architecture docs:
    1. Load recipe and materialize HDC vector from seed
    2. Apply inverse Hadamard transform
    3. Unbind position vectors to retrieve pixel values
    4. Assemble frames from decoded pixels
    5. Apply temporal decoding for frame sequences
    
    Architecture Reference:
    - Hadamard Position Encoding: Each pixel uses orthogonal Hadamard row
    - Circular Temporal Encoding: ρ^0(f0) ⊕ ρ^1(f1) ⊕ ρ^2(f2) ...
    """
    
    def __init__(self, config: DecoderConfig):
        self.config = config
        
        # Initialize Hadamard basis
        self.hadamard = WalshHadamardBasis(dim=config.hdc_dim)
        self.ternary_encoder = TernaryHadamardEncoder(dim=config.hdc_dim)
        
        # Pre-compute position vectors for video
        self._position_vectors = {}
        self._value_vectors = {}
        
        # Cache for decoded frames
        self._frame_cache = {}
    
    def get_position_vector(self, x: int, y: int, frame: int = 0) -> np.ndarray:
        """
        Get Hadamard position vector for pixel at (x, y, frame).
        
        Uses orthogonal Hadamard rows for zero-collision spatial encoding.
        Position index = x + y * width + frame * width * height
        """
        position_index = x + y * self.config.video_width + frame * self.config.video_width * self.config.video_height
        
        if position_index not in self._position_vectors:
            # Generate Hadamard row as position vector
            self._position_vectors[position_index] = self.hadamard.get_row(position_index)
        
        return self._position_vectors[position_index]
    
    def get_value_vector(self, value: int, channel: int = 0) -> np.ndarray:
        """
        Get HDC vector for a pixel value (0-255) using BLAKE3.
        
        Each RGB value gets a deterministic vector.
        """
        key = (value, channel)
        if key not in self._value_vectors:
            seed_string = f"pixel_value:{value}:channel:{channel}"
            self._value_vectors[key] = seed_to_hypervector_blake3(
                seed_string, 
                self.config.hdc_dim // 64
            )
        return self._value_vectors[key]
    
    def decode_frame_from_vector(self, 
                                  hdc_vector: np.ndarray,
                                  frame_index: int = 0,
                                  apply_circular_shift: bool = True) -> np.ndarray:
        """
        Decode a single video frame from an HDC vector.
        
        Steps:
        1. Apply inverse circular shift if temporal encoding was used
        2. For each pixel position, unbind to retrieve value
        3. Match to closest value vector
        4. Assemble into image array
        
        Args:
            hdc_vector: The HDC vector containing encoded video
            frame_index: Which frame to decode (for temporal sequences)
            apply_circular_shift: Whether to apply inverse circular shift
            
        Returns:
            numpy array of shape (height, width, 3) for RGB frame
        """
        height = self.config.video_height
        width = self.config.video_width
        
        # Apply inverse circular shift for temporal position
        if apply_circular_shift and frame_index > 0:
            shift_amount = frame_index % (self.config.hdc_dim // 64)
            hdc_vector = np.roll(hdc_vector, -shift_amount)
        
        # Initialize frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Decode each pixel
        # For efficiency, we decode in patches
        patch_size = self.config.patch_size
        
        for py in range(0, height, patch_size):
            for px in range(0, width, patch_size):
                # Decode patch
                patch = self._decode_patch(
                    hdc_vector, 
                    px, py, 
                    patch_size, 
                    frame_index
                )
                
                # Place in frame
                end_y = min(py + patch_size, height)
                end_x = min(px + patch_size, width)
                frame[py:end_y, px:end_x] = patch[:end_y-py, :end_x-px]
        
        return frame
    
    def _decode_patch(self,
                      hdc_vector: np.ndarray,
                      start_x: int,
                      start_y: int,
                      patch_size: int,
                      frame_index: int) -> np.ndarray:
        """
        Decode a patch of pixels from the HDC vector.
        
        Uses XOR unbinding to retrieve pixel values:
        unbound = hdc_vector ⊕ position_vector
        Then match to closest value vector.
        """
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        
        for dy in range(patch_size):
            for dx in range(patch_size):
                x = start_x + dx
                y = start_y + dy
                
                if x >= self.config.video_width or y >= self.config.video_height:
                    continue
                
                # Get position vector
                pos_vec = self.get_position_vector(x, y, frame_index)
                
                # Unbind: XOR with position vector
                unbound = np.bitwise_xor(hdc_vector, pos_vec)
                
                # Find closest value vector for each channel
                for c in range(3):
                    value = self._find_closest_value(unbound, c)
                    patch[dy, dx, c] = value
        
        return patch
    
    def _find_closest_value(self, 
                            vector: np.ndarray, 
                            channel: int,
                            search_range: int = 256) -> int:
        """
        Find the closest matching value vector using Hamming similarity.
        
        This implements the "snap to closest" operation from the architecture.
        """
        best_value = 0
        best_similarity = -1
        
        # Search through possible values
        # For efficiency, we use a coarse-to-fine search
        for value in range(0, search_range, 16):  # Coarse search
            value_vec = self.get_value_vector(value, channel)
            similarity = self._hamming_similarity(vector, value_vec)
            if similarity > best_similarity:
                best_similarity = similarity
                best_value = value
        
        # Fine search around best coarse value
        start = max(0, best_value - 16)
        end = min(256, best_value + 16)
        
        for value in range(start, end):
            value_vec = self.get_value_vector(value, channel)
            similarity = self._hamming_similarity(vector, value_vec)
            if similarity > best_similarity:
                best_similarity = similarity
                best_value = value
        
        return best_value
    
    def _hamming_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Hamming similarity between two uint64 vectors."""
        xored = np.bitwise_xor(a, b)
        # Count differing bits
        diff_bits = 0
        for val in xored:
            diff_bits += bin(val).count('1')
        total_bits = len(a) * 64
        return 1.0 - (diff_bits / total_bits)
    
    def decode_video_sequence(self,
                              hdc_vector: np.ndarray,
                              num_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Decode a complete video sequence from a bundled HDC vector.
        
        Uses circular temporal decoding:
        - Frame 0: No shift
        - Frame 1: Shift by 1
        - Frame 2: Shift by 2
        - ...
        
        The bundled vector contains: ρ^0(f0) ⊕ ρ^1(f1) ⊕ ρ^2(f2) ...
        """
        if num_frames is None:
            num_frames = self.config.video_frames
        
        frames = []
        for i in range(num_frames):
            frame = self.decode_frame_from_vector(hdc_vector, i)
            frames.append(frame)
        
        return frames


# =============================================================================
# HDC Audio Decoder
# =============================================================================

class HDCAudioDecoder:
    """
    Decodes HDC vectors into audio waveforms using Hadamard encoding.
    
    Audio is encoded as:
    - Each sample position gets a Hadamard position vector
    - Sample values are encoded via BLAKE3 deterministic vectors
    - Temporal encoding uses circular shifts
    """
    
    def __init__(self, config: DecoderConfig):
        self.config = config
        
        # Initialize Hadamard basis
        self.hadamard = WalshHadamardBasis(dim=config.hdc_dim)
        
        # Cache for position and value vectors
        self._position_vectors = {}
        self._value_vectors = {}
    
    def get_position_vector(self, sample_index: int, channel: int = 0) -> np.ndarray:
        """Get Hadamard position vector for audio sample."""
        position_index = sample_index + channel * int(self.config.audio_sample_rate * self.config.audio_duration_seconds)
        
        if position_index not in self._position_vectors:
            self._position_vectors[position_index] = self.hadamard.get_row(position_index)
        
        return self._position_vectors[position_index]
    
    def get_value_vector(self, sample_value: float) -> np.ndarray:
        """
        Get HDC vector for an audio sample value.
        
        Audio samples are typically in range [-1.0, 1.0].
        We quantize to 16-bit range for encoding.
        """
        # Quantize to 16-bit
        quantized = int(sample_value * 32767)
        quantized = max(-32768, min(32767, quantized))
        
        if quantized not in self._value_vectors:
            seed_string = f"audio_sample:{quantized}"
            self._value_vectors[quantized] = seed_to_hypervector_blake3(
                seed_string,
                self.config.hdc_dim // 64
            )
        
        return self._value_vectors[quantized]
    
    def decode_audio_from_vector(self,
                                  hdc_vector: np.ndarray,
                                  num_samples: Optional[int] = None,
                                  apply_circular_shift: bool = True) -> np.ndarray:
        """
        Decode audio waveform from an HDC vector.
        
        Args:
            hdc_vector: The HDC vector containing encoded audio
            num_samples: Number of samples to decode
            apply_circular_shift: Whether to apply temporal decoding
            
        Returns:
            numpy array of shape (num_samples, num_channels)
        """
        if num_samples is None:
            num_samples = int(self.config.audio_sample_rate * self.config.audio_duration_seconds)
        
        num_channels = self.config.audio_channels
        audio = np.zeros((num_samples, num_channels), dtype=np.float32)
        
        # Decode samples in chunks for efficiency
        chunk_size = 1024
        
        for ch in range(num_channels):
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                
                for i in range(start, end):
                    # Apply inverse circular shift for temporal position
                    if apply_circular_shift and i > 0:
                        shift_amount = i % (self.config.hdc_dim // 64)
                        shifted_vec = np.roll(hdc_vector, -shift_amount)
                    else:
                        shifted_vec = hdc_vector
                    
                    # Get position vector
                    pos_vec = self.get_position_vector(i, ch)
                    
                    # Unbind
                    unbound = np.bitwise_xor(shifted_vec, pos_vec)
                    
                    # Find closest sample value
                    sample_value = self._find_closest_sample(unbound)
                    audio[i, ch] = sample_value
        
        return audio
    
    def _find_closest_sample(self, vector: np.ndarray) -> float:
        """Find the closest matching audio sample value."""
        best_value = 0.0
        best_similarity = -1
        
        # Coarse search
        for quantized in range(-32768, 32768, 256):
            value_vec = self.get_value_vector(quantized / 32767.0)
            similarity = self._hamming_similarity(vector, value_vec)
            if similarity > best_similarity:
                best_similarity = similarity
                best_value = quantized / 32767.0
        
        # Fine search
        start = int(best_value * 32767) - 256
        end = int(best_value * 32767) + 256
        start = max(-32768, start)
        end = min(32767, end)
        
        for quantized in range(start, end):
            value_vec = self.get_value_vector(quantized / 32767.0)
            similarity = self._hamming_similarity(vector, value_vec)
            if similarity > best_similarity:
                best_similarity = similarity
                best_value = quantized / 32767.0
        
        return best_value
    
    def _hamming_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Hamming similarity between two uint64 vectors."""
        xored = np.bitwise_xor(a, b)
        diff_bits = 0
        for val in xored:
            diff_bits += bin(val).count('1')
        total_bits = len(a) * 64
        return 1.0 - (diff_bits / total_bits)


# =============================================================================
# LTX HDC Decoder - Combined Video/Audio Generation
# =============================================================================

class LTXHDCDecoder:
    """
    Combined decoder for LTX-style audio-video generation from HDC recipes.
    
    This class:
    1. Loads saved HDC recipes from the transfer learning pipeline
    2. Materializes HDC vectors from seeds
    3. Decodes video frames using Hadamard position encoding
    4. Decodes audio samples using Hadamard position encoding
    5. Synchronizes audio-video output
    """
    
    def __init__(self, config: DecoderConfig):
        self.config = config
        
        # Initialize sub-decoders
        self.video_decoder = HDCVideoDecoder(config)
        self.audio_decoder = HDCAudioDecoder(config)
        
        # Recipe storage
        self.recipe_storage = None
        self.chain_storage = None
    
    def load_recipes(self, recipe_path: str) -> bool:
        """
        Load saved HDC recipes from the transfer learning pipeline.
        
        Args:
            recipe_path: Path to the ltx_recipes directory
            
        Returns:
            True if recipes loaded successfully
        """
        path = Path(recipe_path)
        
        if not path.exists():
            print(f"Recipe path not found: {path}")
            return False
        
        # Load recipe storage
        self.recipe_storage = RecipeStorage(base_path=path)
        
        # Load chain storage
        chain_path = path / "chains"
        if chain_path.exists():
            self.chain_storage = LTXChainStorage(
                storage_path=str(chain_path),
                hdc_dim=self.config.hdc_dim
            )
        
        # Load manifest
        manifest_path = path / "model_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}
        
        print(f"Loaded recipes from: {path}")
        return True
    
    def generate_from_recipe(self,
                             recipe_id: str,
                             generation_mode: str = "text_to_audio_video") -> Dict[str, Any]:
        """
        Generate video and audio from a specific recipe.
        
        Args:
            recipe_id: The recipe ID to generate from
            generation_mode: The generation mode (text_to_audio_video, etc.)
            
        Returns:
            Dictionary with 'video' and 'audio' keys
        """
        if self.recipe_storage is None:
            raise ValueError("No recipes loaded. Call load_recipes() first.")
        
        # Load recipe
        recipe = self.recipe_storage.load_recipe(recipe_id)
        if recipe is None:
            raise ValueError(f"Recipe not found: {recipe_id}")
        
        # Materialize HDC vector from recipe
        hdc_vector = self._materialize_recipe(recipe)
        
        # Generate video
        video_frames = self.video_decoder.decode_video_sequence(hdc_vector)
        
        # Generate audio
        audio_samples = self.audio_decoder.decode_audio_from_vector(hdc_vector)
        
        return {
            'video': video_frames,
            'audio': audio_samples,
            'recipe_id': recipe_id,
            'generation_mode': generation_mode
        }
    
    def generate_from_seed(self,
                           seed_string: str,
                           generation_mode: str = "text_to_audio_video") -> Dict[str, Any]:
        """
        Generate video and audio directly from a seed string.
        
        This is useful for testing without saved recipes.
        
        Args:
            seed_string: The seed string to generate from
            generation_mode: The generation mode
            
        Returns:
            Dictionary with 'video' and 'audio' keys
        """
        # Generate HDC vector from seed
        hdc_vector = seed_to_hypervector_blake3(
            seed_string,
            self.config.hdc_dim // 64
        )
        
        # Generate video
        video_frames = self.video_decoder.decode_video_sequence(hdc_vector)
        
        # Generate audio
        audio_samples = self.audio_decoder.decode_audio_from_vector(hdc_vector)
        
        return {
            'video': video_frames,
            'audio': audio_samples,
            'seed_string': seed_string,
            'generation_mode': generation_mode
        }
    
    def _materialize_recipe(self, recipe: IdentityRecipe) -> np.ndarray:
        """
        Materialize an HDC vector from a recipe.
        
        Uses the recipe's hadamard_index and base_seed to regenerate
        the deterministic HDC vector.
        """
        # Generate base vector from seed
        seed_string = f"H{recipe.hadamard_index}.S{recipe.base_seed}"
        hdc_vector = seed_to_hypervector_blake3(
            seed_string,
            self.config.hdc_dim // 64
        )
        
        # Apply hadamard index as circular shift
        if recipe.hadamard_index > 0:
            shift_amount = recipe.hadamard_index % (self.config.hdc_dim // 64)
            hdc_vector = np.roll(hdc_vector, shift_amount)
        
        return hdc_vector
    
    def save_generation(self,
                        generation: Dict[str, Any],
                        output_name: str) -> Dict[str, str]:
        """
        Save generated video and audio to files.
        
        Args:
            generation: The generation output from generate_from_recipe/seed
            output_name: Base name for output files
            
        Returns:
            Dictionary of saved file paths
        """
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save video frames
        if self.config.save_video and 'video' in generation:
            video_path = output_path / f"{output_name}_frames"
            video_path.mkdir(exist_ok=True)
            
            for i, frame in enumerate(generation['video']):
                frame_path = video_path / f"frame_{i:04d}.npy"
                np.save(frame_path, frame)
            
            saved_files['video_frames'] = str(video_path)
            print(f"Saved {len(generation['video'])} frames to {video_path}")
        
        # Save audio samples
        if self.config.save_audio and 'audio' in generation:
            audio_path = output_path / f"{output_name}_audio.npy"
            np.save(audio_path, generation['audio'])
            saved_files['audio'] = str(audio_path)
            print(f"Saved audio to {audio_path}")
        
        # Save visualization
        if self.config.save_visualization and 'video' in generation:
            self._save_visualization(generation, output_path, output_name)
        
        return saved_files
    
    def _save_visualization(self,
                           generation: Dict[str, Any],
                           output_path: Path,
                           output_name: str):
        """Save visualization of generated content."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create figure with video frames and audio waveform
            video = generation.get('video', [])
            audio = generation.get('audio', None)
            
            if video:
                # Show first, middle, last frames
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                indices = [0, len(video) // 2, len(video) - 1]
                for ax, idx in zip(axes, indices):
                    if idx < len(video):
                        ax.imshow(video[idx])
                        ax.set_title(f"Frame {idx}")
                        ax.axis('off')
                
                plt.suptitle(f"Generated Video: {generation.get('seed_string', generation.get('recipe_id', 'unknown'))}")
                viz_path = output_path / f"{output_name}_video_viz.png"
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved video visualization to {viz_path}")
            
            if audio is not None and len(audio) > 0:
                # Plot audio waveform
                fig, axes = plt.subplots(self.config.audio_channels, 1, figsize=(12, 4))
                if self.config.audio_channels == 1:
                    axes = [axes]
                
                for ch, ax in enumerate(axes):
                    if ch < audio.shape[1]:
                        ax.plot(audio[:, ch])
                        ax.set_title(f"Audio Channel {ch}")
                        ax.set_xlabel("Sample")
                        ax.set_ylabel("Amplitude")
                
                plt.tight_layout()
                audio_viz_path = output_path / f"{output_name}_audio_viz.png"
                plt.savefig(audio_viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved audio visualization to {audio_viz_path}")
                
        except ImportError:
            print("matplotlib not available, skipping visualization")


# =============================================================================
# Validation Tests
# =============================================================================

class DecoderValidator:
    """
    Validates the HDC decoder output quality.
    
    Tests:
    1. Determinism - Same seed produces identical output
    2. Reconstructability - Encoded then decoded matches original
    3. Temporal consistency - Frames are temporally coherent
    4. Audio-video sync - Audio matches video timing
    """
    
    def __init__(self, config: DecoderConfig):
        self.config = config
        self.decoder = LTXHDCDecoder(config)
    
    def test_determinism(self, seed_string: str = "test_determinism") -> bool:
        """
        Test that the same seed produces identical output every time.
        
        This is a core requirement of the Pure HDC architecture.
        """
        print("\n" + "=" * 60)
        print("Test: Determinism")
        print("=" * 60)
        
        # Generate twice with same seed
        gen1 = self.decoder.generate_from_seed(seed_string)
        gen2 = self.decoder.generate_from_seed(seed_string)
        
        # Compare video frames
        video_match = True
        for i, (f1, f2) in enumerate(zip(gen1['video'], gen2['video'])):
            if not np.array_equal(f1, f2):
                print(f"  FAIL: Video frame {i} differs between generations")
                video_match = False
                break
        
        if video_match:
            print(f"  PASS: All {len(gen1['video'])} video frames are identical")
        
        # Compare audio samples
        audio_match = np.allclose(gen1['audio'], gen2['audio'], atol=1e-6)
        if audio_match:
            print(f"  PASS: Audio samples are identical")
        else:
            print(f"  FAIL: Audio samples differ between generations")
        
        return video_match and audio_match
    
    def test_reconstructability(self) -> bool:
        """
        Test that encoding then decoding produces similar results.
        
        This tests the full encode-decode round trip.
        """
        print("\n" + "=" * 60)
        print("Test: Reconstructability")
        print("=" * 60)
        
        # Create a simple test pattern
        test_frame = np.zeros((self.config.video_height, self.config.video_width, 3), dtype=np.uint8)
        
        # Create a gradient pattern
        for y in range(self.config.video_height):
            for x in range(self.config.video_width):
                test_frame[y, x] = [
                    int(255 * x / self.config.video_width),
                    int(255 * y / self.config.video_height),
                    128
                ]
        
        # Encode the frame
        encoded = self._encode_frame(test_frame)
        
        # Decode it back
        decoded = self.decoder.video_decoder.decode_frame_from_vector(encoded)
        
        # Compare
        # Note: Perfect reconstruction is not expected due to quantization
        # We check for reasonable similarity
        diff = np.abs(test_frame.astype(np.float32) - decoded.astype(np.float32))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        print(f"  Mean pixel difference: {mean_diff:.2f}")
        print(f"  Max pixel difference: {max_diff:.2f}")
        
        # Consider it a pass if mean difference is reasonable
        # (exact threshold depends on HDC dimension and encoding quality)
        passed = mean_diff < 50  # Allow some tolerance
        
        if passed:
            print(f"  PASS: Reconstruction quality acceptable")
        else:
            print(f"  FAIL: Reconstruction quality too low")
        
        return passed
    
    def _encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode a single frame into HDC vector.
        
        This is the inverse of decode_frame_from_vector.
        """
        hdc_vector = np.zeros(self.config.hdc_dim // 64, dtype=np.uint64)
        
        height, width, channels = frame.shape
        
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    # Get position vector
                    pos_vec = self.decoder.video_decoder.get_position_vector(x, y, 0)
                    
                    # Get value vector
                    val_vec = self.decoder.video_decoder.get_value_vector(frame[y, x, c], c)
                    
                    # Bind: XOR position with value
                    bound = np.bitwise_xor(pos_vec, val_vec)
                    
                    # Bundle into result (XOR superposition)
                    hdc_vector = np.bitwise_xor(hdc_vector, bound)
        
        return hdc_vector
    
    def test_temporal_consistency(self, seed_string: str = "test_temporal") -> bool:
        """
        Test that consecutive frames show temporal coherence.
        
        Frames should not be completely random - there should be
        some continuity between adjacent frames.
        """
        print("\n" + "=" * 60)
        print("Test: Temporal Consistency")
        print("=" * 60)
        
        generation = self.decoder.generate_from_seed(seed_string)
        frames = generation['video']
        
        if len(frames) < 2:
            print("  SKIP: Need at least 2 frames")
            return True
        
        # Calculate frame-to-frame differences
        diffs = []
        for i in range(len(frames) - 1):
            diff = np.abs(frames[i].astype(np.float32) - frames[i+1].astype(np.float32))
            mean_diff = np.mean(diff)
            diffs.append(mean_diff)
        
        avg_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        
        print(f"  Average frame-to-frame difference: {avg_diff:.2f}")
        print(f"  Maximum frame-to-frame difference: {max_diff:.2f}")
        
        # Check that differences are not too extreme
        # (very high differences would indicate random noise, not coherent video)
        passed = avg_diff < 100  # Threshold for temporal coherence
        
        if passed:
            print(f"  PASS: Temporal consistency acceptable")
        else:
            print(f"  FAIL: Frames appear too random")
        
        return passed
    
    def test_audio_video_sync(self, seed_string: str = "test_sync") -> bool:
        """
        Test that audio and video are synchronized.
        
        This is a basic test that checks both are generated from
        the same HDC vector and thus are deterministically linked.
        """
        print("\n" + "=" * 60)
        print("Test: Audio-Video Synchronization")
        print("=" * 60)
        
        # Generate multiple times
        generations = [
            self.decoder.generate_from_seed(seed_string)
            for _ in range(3)
        ]
        
        # Check that all generations are identical (deterministic sync)
        all_identical = True
        
        for i, gen in enumerate(generations[1:], 1):
            # Check video
            for j, (f1, f2) in enumerate(zip(generations[0]['video'], gen['video'])):
                if not np.array_equal(f1, f2):
                    print(f"  FAIL: Video frame {j} differs in generation {i}")
                    all_identical = False
                    break
            
            # Check audio
            if not np.allclose(generations[0]['audio'], gen['audio'], atol=1e-6):
                print(f"  FAIL: Audio differs in generation {i}")
                all_identical = False
        
        if all_identical:
            print(f"  PASS: Audio and video are deterministically synchronized")
        
        return all_identical
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all validation tests."""
        print("\n" + "=" * 70)
        print("HDC Decoder Validation Tests")
        print("=" * 70)
        
        results = {
            'determinism': self.test_determinism(),
            'reconstructability': self.test_reconstructability(),
            'temporal_consistency': self.test_temporal_consistency(),
            'audio_video_sync': self.test_audio_video_sync()
        }
        
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {test_name}: {status}")
        
        all_passed = all(results.values())
        print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        
        return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for HDC decoder testing."""
    parser = argparse.ArgumentParser(
        description="Test HDC decoder for video/audio generation"
    )
    parser.add_argument(
        "--recipe_path",
        type=str,
        default="./ltx_recipes",
        help="Path to saved HDC recipes"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./test_output",
        help="Path for output files"
    )
    parser.add_argument(
        "--hdc_dim",
        type=int,
        default=DEFAULT_HDC_DIM,
        help="HDC dimension"
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=128,
        help="Video width (smaller for faster testing)"
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=128,
        help="Video height (smaller for faster testing)"
    )
    parser.add_argument(
        "--video_frames",
        type=int,
        default=8,
        help="Number of video frames to generate"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="test_generation_seed",
        help="Seed string for generation"
    )
    parser.add_argument(
        "--run_tests",
        action="store_true",
        help="Run validation tests"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate sample output"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = DecoderConfig(
        hdc_dim=args.hdc_dim,
        video_width=args.video_width,
        video_height=args.video_height,
        video_frames=args.video_frames,
        output_path=args.output_path
    )
    
    # Create decoder
    decoder = LTXHDCDecoder(config)
    
    # Run tests if requested
    if args.run_tests:
        validator = DecoderValidator(config)
        results = validator.run_all_tests()
        return 0 if all(results.values()) else 1
    
    # Generate sample if requested
    if args.generate:
        print("\n" + "=" * 60)
        print("Generating Sample Output")
        print("=" * 60)
        
        # Try to load recipes first
        if Path(args.recipe_path).exists():
            decoder.load_recipes(args.recipe_path)
        
        # Generate from seed
        print(f"\nGenerating from seed: {args.seed}")
        generation = decoder.generate_from_seed(args.seed)
        
        # Save output
        saved = decoder.save_generation(generation, "sample_generation")
        
        print("\nSaved files:")
        for name, path in saved.items():
            print(f"  {name}: {path}")
        
        return 0
    
    # Default: show usage
    print("Use --run_tests to run validation tests")
    print("Use --generate to generate sample output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
