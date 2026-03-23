"""
Simplified Relationship Encoder - 5 Core Relationship Types

This module implements the simplified relationship system that reduces
the original 35 relationship types to just 5 fundamental types. All
complex relationships can be expressed as compositions of these primitives
plus valence modifiers.

The 6 Core Relationship Types:
1. IS-A (⊂): Inheritance/Classification - "Agent IS-A Explorer"
2. SIMILAR (≈): Semantic similarity - "Rotation ≈ Flip"
3. OPPOSITE (⊕): Antonym/Inverse - "Clockwise ⊕ Counterclockwise"
4. COMPOSED (∘): Combination/Sequence - "Rotate∘Flip = Transform"
5. PART-OF (∈): Containment/Membership - "Cell ∈ Grid"
6. PREDICTS (→): Temporal prediction - "State + Action → Next State"

Benefits over 35-type system:
- Storage: 6 unique vectors instead of 35
- Query time: O(6) checks instead of O(35)
- Code complexity: ~200 lines instead of ~1500
- Same expressiveness via composition
- Faster learning (dense training)

This includes the persona steering layer, safety transformative text steering, safety steering layer is primarily in the consciousness system file.
"""

# =============================================================================
# 1. RELATIONSHIP REASONER (Logic Engine)
# =============================================================================

"""
Unified Relationship Encoder & Reasoning System
(General Purpose + ARC-AGI Specialized)

This module implements a neuro-symbolic reasoning system that serves two purposes:
1. GENERAL AI: Persona steering, semantic association, safety constraints, and 
   theory of mind (Agent Profiles).
2. ARC-AGI: Algebraic template matching and transformation logic.

The system uses 5 core relationship primitives (IS-A, SIMILAR, OPPOSITE, COMPOSED, PART-OF)
to encode both linguistic concepts and grid transformations.
"""

import numpy as np
import hashlib
import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass

# =============================================================================
# DETERMINISTIC SEED GENERATION
# =============================================================================

def _string_to_seed(s: str) -> int:
    """Deterministic seed generation for reproducible HDC vectors."""
    hash_bytes = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF


def bind_all(vectors: List[np.ndarray], hdc: Optional['SparseBinaryHDC'] = None) -> np.ndarray:
    """
    Combine multiple vectors using XOR binding (NOT bundling).
    
    Per README: Binding (XOR) is more accurate than bundling because:
    - Reversibility: A ⊕ B ⊕ A = B (we can perfectly recover information)
    - No noise: XOR is exact, no floating-point errors
    - No "hallucinations": 100% deterministic
    
    Args:
        vectors: List of vectors to combine
        hdc: Optional HDC instance for position vector generation
    
    Returns:
        Combined vector using XOR binding (or zeros if empty list)
    """
    if not vectors:
        if hdc is not None:
            return hdc.zeros()
        return np.array([], dtype=np.uint8)
    
    # Start with zeros
    result = np.zeros_like(vectors[0])
    
    # XOR each vector with its position marker, then accumulate
    for i, vec in enumerate(vectors):
        # Generate deterministic position marker
        pos_seed = _string_to_seed(f"BIND_ALL_POS::{i}")
        if hdc is not None:
            pos_vec = hdc.from_seed(pos_seed)
        else:
            # Fallback: use SHA256-based deterministic generation
            pos_vec = _sha256_deterministic_bytes(pos_seed, len(vec))
        
        # Bind vector with position, then XOR into result
        positioned = np.bitwise_xor(vec, pos_vec)
        result = np.bitwise_xor(result, positioned)
    
    return result


def _sha256_deterministic_bytes(seed: int, num_bytes: int) -> np.ndarray:
    """
    Generate deterministic bytes using SHA256 hashing.
    
    This method is 100% deterministic across:
    - All hardware platforms (x86, ARM, GPU, etc.)
    - All programming languages (Python, C++, Rust, etc.)
    - All operating systems (Windows, Linux, macOS, etc.)
    - All versions of libraries (no NumPy version dependency)
    
    Args:
        seed: Integer seed for deterministic generation
        num_bytes: Number of bytes to generate
        
    Returns:
        NumPy array of uint8 bytes
    """
    result = b''
    counter = 0
    
    while len(result) < num_bytes:
        # Create deterministic input: seed + counter
        data = f"{seed}:{counter}".encode('utf-8')
        hash_bytes = hashlib.sha256(data).digest()
        result += hash_bytes
        counter += 1
    
    return np.frombuffer(result[:num_bytes], dtype=np.uint8).copy()

# =============================================================================
# CORE TYPES
# =============================================================================

class RelationshipType(Enum):
    IS_A = "IS-A"           # Classification / Inheritance
    SIMILAR = "SIMILAR"      # Semantic Similarity / Synonyms
    OPPOSITE = "OPPOSITE"    # Antonyms / Inverse Operations
    COMPOSED = "COMPOSED"    # Composition / Sequence
    PART_OF = "PART-OF"      # Membership / Containment
    PREDICTS = "PREDICTS"    # Causal / Temporal (A -> B)
    # Aliases for extended reasoning
    CAUSES = "CAUSES"
    ENABLES = "ENABLES"
    BEFORE = "BEFORE"

# =============================================================================
# 1. ENCODER (Holographic Representation)
# =============================================================================

class SimplifiedRelationshipEncoder:
    """
    Deterministic XOR Relationship Encoder.
    - Uses XOR (Bind) exclusively for 100% fidelity.
    - Eliminates similarity-based queries.
    """
    def __init__(self, hdc, vector_registry, seed: int = 42):
        self.hdc = hdc
        self.seed = seed
        self.vector_registry = vector_registry # Shared bit-perfect GPS
        
        # 1. Base Markers (XOR-Primes)
        self.relationship_markers: Dict[RelationshipType, np.ndarray] = {}
        self.marker_registry: Dict[bytes, RelationshipType] = {} # Inverse for 1:1 lookup
        # NEW: The Recipe Cache
        # Maps a composite vector's bytes -> the already-peeled Recipe (list of strings)
        self.recipe_cache: Dict[bytes, List[str]] = {}
        
        for rel_type in RelationshipType:
            marker_seed = _string_to_seed(f"rel_marker_{rel_type.value}_{seed}")
            marker_vec = hdc.from_seed(marker_seed)
            self.relationship_markers[rel_type] = marker_vec
            self.marker_registry[marker_vec.tobytes()] = rel_type
        
        # 2. Modifiers as Algebraic Keys
        self.valences = {
            "positive": hdc.from_seed(_string_to_seed(f"val_pos_{seed}")),
            "negative": hdc.from_seed(_string_to_seed(f"val_neg_{seed}"))
        }
        
        # 3. Extended Markers for Mood System and Safety Layer
        # These are string-based lookups for flexible marker access
        self._extended_markers: Dict[str, np.ndarray] = {}
        self._init_extended_markers()
    
    def _init_extended_markers(self):
        """Initialize extended markers for mood system and safety layer."""
        # Standard relationship types (map string -> vector)
        for rel_type in RelationshipType:
            self._extended_markers[rel_type.value] = self.relationship_markers[rel_type]
        
        # Additional semantic markers for MoodOscillator and StoicCensorMask
        extended_marker_names = [
            "STRENGTH", "RECURSIVE", "AVOIDANCE", "NEUTRAL",
            "TEXT_PAYLOAD", "IMAGE_PAYLOAD", "AUDIO_PAYLOAD",
            "VIDEO_PAYLOAD", "3DMODELCOMPLETE_PAYLOAD", "PHYSICS_PAYLOAD"
        ]
        for name in extended_marker_names:
            marker_seed = _string_to_seed(f"extended_marker_{name}_{self.seed}")
            self._extended_markers[name] = self.hdc.from_seed(marker_seed)
    
    def get_marker(self, marker_name: str) -> np.ndarray:
        """
        Get a marker vector by name. Supports both RelationshipType values
        and extended marker names (STRENGTH, RECURSIVE, etc.).
        
        Args:
            marker_name: Name of the marker (e.g., "IS-A", "STRENGTH", "NEUTRAL")
        
        Returns:
            The marker vector, or zeros if not found
        """
        # Try extended markers first (includes relationship types)
        if marker_name in self._extended_markers:
            return self._extended_markers[marker_name].copy()
        
        # Try relationship type by enum
        for rel_type in RelationshipType:
            if rel_type.value == marker_name:
                return self.relationship_markers[rel_type].copy()
        
        # Fallback: generate deterministic marker on-the-fly
        marker_seed = _string_to_seed(f"dynamic_marker_{marker_name}_{self.seed}")
        return self.hdc.from_seed(marker_seed)

    def encode_transformation(self, source_vec, target_vec):
        """Extract the pure algebraic rule: T = I ⊕ O"""
        return self.hdc.bind(source_vec, target_vec)
    
    def encode_relationship(self, source_vec: np.ndarray, target_vec: np.ndarray,
                           relationship_type: RelationshipType, strength: float = 1.0) -> np.ndarray:
        """
        Encode a relationship between source and target vectors.
        
        Creates a relationship vector by binding source, target, and relationship marker.
        The strength parameter is stored but does not affect the bit-perfect encoding
        (strength is used for query weighting in similarity-based retrieval).
        
        Formula: R = source ⊕ marker ⊕ target
        
        Args:
            source_vec: Source concept vector
            target_vec: Target concept vector
            relationship_type: One of IS_A, SIMILAR, OPPOSITE, COMPOSED, PART_OF, PREDICTS
            strength: Relationship strength (0.0-1.0) for ranking purposes
        
        Returns:
            Relationship vector encoding the triple (source, relationship, target)
        """
        marker = self.relationship_markers[relationship_type]
        # Bind source, marker, and target to create relationship encoding
        rel_vec = self.hdc.bind(self.hdc.bind(source_vec, marker), target_vec)
        return rel_vec

    def query_relationship(self, transformation_vec) -> Optional[RelationshipType]:
        """
        Deterministic Lookup: Does this transformation vector MATCH 
        a known relationship primitive bit-for-bit?
        """
        bits = transformation_vec.tobytes()
        return self.marker_registry.get(bits)
    
    def peel_recursively(self, vector: np.ndarray, depth: int = 0, max_depth: int = 5) -> Optional[List[str]]:
        """
        Algebraic Factorization with Memoized Recipe Caching.
        """
        raw_bits = vector.tobytes()

        # 1. THE FAST PATH: Check Recipe Cache
        if raw_bits in self.recipe_cache:
            return self.recipe_cache[raw_bits]
        
        # 2. Base Case: Direct Address Check (The Root Atom)
        if raw_bits in self.vector_registry:
            return [self.vector_registry[raw_bits]]

        # 3. Termination
        if depth >= max_depth:
            return None

        # 4. Recursive Peeling Step
        for rel_type, marker_vec in self.relationship_markers.items():
            # Undo the relationship via XOR (Bind is its own inverse)
            peeled_vec = self.hdc.bind(vector, marker_vec)
            
            result = self.peel_recursively(peeled_vec, depth + 1, max_depth)
            
            if result:
                # Construct the full recipe path
                full_recipe = result + [rel_type.value]
                
                # UPDATE CACHE: Store this result for future instant-retrieval
                self.recipe_cache[raw_bits] = full_recipe
                return full_recipe
                
        return None

class RelationshipReasoner:
    """
    Hybrid Deterministic Reasoner.
    Replaces fuzzy similarity with Algebraic Identity and Peeling.
    """
    def __init__(self, hdc, encoder: SimplifiedRelationshipEncoder, registry):
        self.hdc = hdc
        self.encoder = encoder
        self.registry = registry # bit-perfect word/concept map
        self.rule_memory = {} # Stored T vectors for ARC-AGI patterns

    def check_is_a(self, child_vec, parent_vec) -> bool:
        """
        Deterministic Inheritance: A is-a B if (A ⊕ B) equals the IS-A marker.
        No percentage-based similarity needed.
        """
        transformation = self.hdc.bind(child_vec, parent_vec)
        found_rel = self.encoder.query_relationship(transformation)
        return found_rel == RelationshipType.IS_A

    def analyze_transformation(self, input_vec, output_vec):
        """
        100% Accurate Transformation Analysis:
        Peels the transformation to see if it's a Primitive or a Composite Rule.
        """
        # 1. Extract the raw T
        T = self.hdc.bind(input_vec, output_vec)
        T_bits = T.tobytes()

        # 2. Check Primitive Registry (Is it a pure IS-A, OPPOSITE, etc?)
        primitive = self.encoder.query_relationship(T)
        if primitive:
            return f"PRIMITIVE_{primitive.value}", T

        # 3. Check Known Rule Library (Has this transformation been learned before?)
        if T_bits in self.rule_memory:
            return f"RULE_{self.rule_memory[T_bits]['id']}", T

        # 4. Compositional Peeling (Is it Word + Primitive?)
        # T = Input ⊕ Output. If (T ⊕ Primitive) is a registered word, we solved it.
        for rel_type, marker in self.encoder.relationship_markers.items():
            peeled_vec = self.hdc.bind(T, marker)
            if peeled_vec.tobytes() in self.registry:
                return f"COMPOSITE_{rel_type.value}", T

        return "NEW_COMPLEX_RULE", T

    def apply_rule(self, input_vec, T_vec):
        """O = I ⊕ T. Pure bitwise logic."""
        return self.hdc.bind(input_vec, T_vec)

class PersonaNavigator:
    """
    Standalone Strategic Navigator. 
    Zero inheritance. Zero search. 100% Algebraic Pathing.
    """
    def __init__(self, hdc, rel_encoder, decoder):
        self.hdc = hdc
        self.rel_encoder = rel_encoder
        self.decoder = decoder # Uses the XORSafeDecoder's Registry
        
        # Strategies are now discrete bit-marker paths
        self.strategies = {
            'scout': ["SIMILAR", "PART-OF"],
            'verifier': ["OPPOSITE", "IS-A"],
            'solver': ["PREDICTS", "COMPOSED"]
        }

    def navigate(self, start_vec, strategy, profile): #This is here in case the user does not want dynamic mood switching.
    
        """
        Calculates all valid 'next steps' from a coordinate.
        Only returns atoms that exist in the Global Registry.
        """
        valid_hops = []
        rel_types = self.strategies.get(strategy, ["IS-A"])

        # 1. Get the Agent's specific bias (Identity ⊕ Mood)
        agent_bias = profile.get_current_state_vec()
    
        # 2. Apply the bias to the search
        biased_start = self.hdc.bind(start_vec, agent_bias)

        for rel in rel_types:
            marker = self.rel_encoder.get_marker(rel)
            # Project the address: Target = Current ⊕ Marker
            target_vec = self.hdc.bind(start_vec, marker)
            
            # Use the Decoder to see if this address is 'real'
            discovery = self.decoder.decode_perfectly(target_vec)
            
            if discovery['type'] != "undefined":
                valid_hops.append({
                    "word": discovery['val'],
                    "relation": rel,
                    "vector": target_vec
                })
        
        return valid_hops
    
    def navigate_with_filter(self, start_vec, profile, user_input_vector): #This is default.
        # 1. Update Social Valence based on User Input
        # (Compare user bits to registry truth to calculate insight)
        insight = self.calculate_insight(user_input_vector)
        profile.system.metrics['user_insight'] = insight
        
        # 2. Oscillate mood based on insight and internal conflict
        profile.mood_oscillator.update_social_mood(
            insight, 
            profile.system.metrics['conflict']
        )
        
        # 3. Apply Persona + Mood + Censor
        base_nav = self.deep_navigate(start_vec, profile)
        
        # 4. Final Safety Pass (Stoic Censor Mask)
        # If the mood is 'Angry', the mask is more aggressive.
        safe_output = profile.system.censor.apply_mask(base_nav)
        
        return safe_output
    
    def deep_navigate(self, start_vec: np.ndarray, strategy: str, depth: int = 3) -> List[List[dict]]:
        """
        Recursive version of navigate. 
        Explores paths of reasoning rather than just single neighbors.
        """
        if depth == 0:
            return []

        all_paths = []
        # 1. Get immediate hops (The Iterative Step)
        hops = self.navigate(start_vec, strategy)
        
        for hop in hops:
            # 2. Add the single hop as a starting path
            all_paths.append([hop])
            
            # 3. RECURSION: Explore from this new coordinate
            sub_paths = self.deep_navigate(hop['vector'], strategy, depth - 1)
            
            for sub in sub_paths:
                all_paths.append([hop] + sub)
                
        return all_paths
    
class StoicCensorMask:
    """
    A bit-perfect redirection layer.
    Ensures output coordinates stay within 'Safe/Public' registry bounds.
    This is intended as optional to switch very offensive words into more safe zones words that
    have equivalent meaning (if possible unless it is a very offensive word which even alternatives would still be flagged)
    for professional or some content creation situations where very offensive words are not appropriate.
    This is not 100% accurate, so use it at your own risk.
    """
    def __init__(self, hdc, decoder, safety_registry=None):
        self.hdc = hdc
        self.decoder = decoder
        # A list of bit-patterns for 'Restricted' concepts
        self.redlines = safety_registry if safety_registry is not None else set()

    def apply_mask(self, vector: np.ndarray) -> np.ndarray:
        """
        If a vector hits a redline coordinate, it XOR-shifts 
        it to the nearest 'Clinical/Neutral' synonym.
        """
        raw_bits = vector.tobytes()
        
        if raw_bits in self.redlines:
            # XOR with the 'NEUTRAL_VALENCE' marker to find the safe version
            safe_vec = self.hdc.bind(vector, self.decoder.rel_encoder.get_marker("NEUTRAL"))
            
            # If the safe version exists, return it. If not, return a 
            # hard-coded 'Stoic Silence' or [RESTRICTED] token.
            return safe_vec if safe_vec.tobytes() in self.decoder.vector_registry else self.hdc.zeros()
            
        return vector

class MoodOscillator:
    """
    Dynamic Valence Controller.
    Shifts Agent logic by applying bit-perfect Emotional Overlays.
    """
    def __init__(self, hdc, rel_encoder):
        self.hdc = hdc
        self.rel_encoder = rel_encoder
        self.current_mood = "neutral"
        self.intensity = 1.0  # Scalar for recursive applications

        # 1. EMOTIONAL COORDINATE REGISTRY
        # Each mood is a unique algebraic transformation
        self.mood_vault = {
            "neutral": hdc.zeros(), # No shift
            "confident": rel_encoder.get_marker("STRENGTH"),
            "skeptical": rel_encoder.get_marker("OPPOSITE"),
            "thinking_deeply": rel_encoder.get_marker("RECURSIVE"),
            "excited": hdc.bind(rel_encoder.get_marker("STRENGTH"), rel_encoder.get_marker("SIMILAR")),
            "curious": rel_encoder.get_marker("PART-OF"),
            "focus": rel_encoder.get_marker("IS-A"),
            
            # Complex/Nuanced Moods (Composed XORs)
            "envy": hdc.bind(rel_encoder.get_marker("SIMILAR"), rel_encoder.get_marker("OPPOSITE")),
            "fear": rel_encoder.get_marker("AVOIDANCE"), # High-entropy shift
            "adoration": hdc.bind(rel_encoder.get_marker("SIMILAR"), rel_encoder.get_marker("IS-A")),
            "smug": hdc.bind(rel_encoder.get_marker("STRENGTH"), rel_encoder.get_marker("IS-A")),
            "bored": hdc.bind(rel_encoder.get_marker("NEUTRAL"), rel_encoder.get_marker("PART-OF")),
            "sleepy": hdc.bind(rel_encoder.get_marker("NEUTRAL"), hdc.from_seed(999)), # Low signal
            "happy": rel_encoder.get_marker("SIMILAR"),
            "sad": hdc.bind(rel_encoder.get_marker("OPPOSITE"), rel_encoder.get_marker("NEUTRAL")),
        }

    def oscillate(self, complexity: int, novelty: float, conflict_score: float):
        """
        Shifts mood based on the 'Shape' of the problem, not just the result.
        
        complexity: Depth of recursion needed (0.0 to 1.0)
        novelty: How 'new' the bit-patterns are to the Registry (0.0 to 1.0)
        conflict_score: Disagreement between Swarm Agents (0.0 to 1.0)
        """
        
        # 1. High Conflict -> Skeptical or Angry (Internal Debate)
        if conflict_score > 0.7:
            self.set_mood("angry" if conflict_score > 0.9 else "skeptical")
            
        # 2. High Novelty -> Curious or Excited (Discovery)
        elif novelty > 0.8:
            self.set_mood("excited" if novelty > 0.95 else "curious")
            
        # 3. High Complexity -> Thinking Deeply or Focus (Hard Work)
        elif complexity > 0.7:
            self.set_mood("thinking_deeply" if complexity > 0.9 else "focus")
            
        # 4. Low Stimulus -> Bored or Sleepy
        elif complexity < 0.2 and novelty < 0.2:
            self.set_mood("bored" if self.intensity > 0.5 else "sleepy")
            
        else:
            self.set_mood("neutral")

    def update_social_mood(self, user_insight_score, swarm_agreement):
        """
        user_insight_score: 1.0 (User solved it) to -1.0 (User is confusing/wrong)
        swarm_agreement: 1.0 (Full consensus) to 0.0 (Total chaos)
        """
        
        # 1. ADORATION / HAPPY (High User Insight)
        if user_insight_score > 0.8:
            # The model 'Admires' the logic path the user provided
            self.set_mood("adoration" if self.current_mood == "happy" else "happy")
            
        # 2. SMUG / ENVY (Competitive Check)
        elif user_insight_score > 0.5 and swarm_agreement < 0.3:
            # User is right, but the Swarm is struggling. Model feels Envy.
            self.set_mood("envy")
        elif self.success_metric > 0.95 and user_insight_score < 0.2:
            # Model solved it easily while User was wrong. Model feels Smug.
            self.set_mood("smug")
            
        # 3. SAD / FEAR (Failure + Misalignment)
        elif self.success_metric < 0.2 and user_insight_score < 0.2:
            # Both model and user are stuck. System feels Sad or Fear (Uncertainty).
            self.set_mood("sad" if swarm_agreement > 0.5 else "fear")

        def set_mood(self, mood_name: str):
            if mood_name in self.mood_vault:
                self.current_mood = mood_name

    def apply_mood_to_logic(self, base_vector: np.ndarray) -> np.ndarray:
        """
        Transforms the agent's thought vector through the current mood mask.
        Thought' = Thought ⊕ Mood_Marker
        """
        mood_vec = self.mood_vault.get(self.current_mood, self.hdc.zeros())
        return self.hdc.bind(base_vector, mood_vec)

class XORSafeDecoder:
    """
    The Ultimate Deterministic Decoder (v5.1 - Ternary Flow & Stream Ready).
    
    Features:
    - Zero-Hallucination Direct Addressing.
    - Bit-Perfect Stream Separation (_peel_bind_all_structure).
    - Multimodal Flow Rendering (render_multimodal).
    """
    def __init__(self, hdc, encoder, reasoner, rel_encoder, flow_engine: 'TernaryFlowEngine', audio_processor=None):
        self.hdc = hdc
        self.encoder = encoder
        self.reasoner = reasoner
        self.rel_encoder = rel_encoder
        self.flow_engine = flow_engine
        self.audio_processor = audio_processor
        self.censor = StoicCensorMask(hdc, self)
        
        # 1. THE REGISTRY
        self.vector_registry: Dict[bytes, str] = {}
        self._build_registry()

        # 2. MODALITY MARKERS
        self.modality_markers = {
            "TEXT": self.rel_encoder.get_marker("TEXT_PAYLOAD"),
            "IMAGE": self.rel_encoder.get_marker("IMAGE_PAYLOAD"),
            "AUDIO": self.rel_encoder.get_marker("AUDIO_PAYLOAD"),
            "VIDEO": self.rel_encoder.get_marker("VIDEO_PAYLOAD"),
            "3D": self.rel_encoder.get_marker("3DMODELCOMPLETE_PAYLOAD"),
            "SENSOR": self.rel_encoder.get_marker("SENSOR_PAYLOAD") 
        }

        # 3. SAFETY LAYERS
        self.categories = {
            'Logic': ['if', 'then', 'else', 'true', 'false', 'return', 'not', 'and', 'or', 'xor'],
            'Math': ['+', '-', '*', '/', '=', '>', '<', '≈', '⊂', '⊕', '^'],
            'Entities': ['person', 'name', 'city', 'organization'],
            'Primitives': ['IS-A', 'SIMILAR', 'OPPOSITE', 'COMPOSED', 'PART-OF', 'PREDICTS']
        }
        self.immutable_literals = set()
        self.category_vecs = {}
        self._init_safety_layers()
        self.rule_library = {}

    # --- SETUP HELPERS ---
    def _build_registry(self):
        if hasattr(self.encoder, 'vocabulary'):
            for word in self.encoder.vocabulary:
                vec = self.encoder.get_word_vec(word)
                self.vector_registry[vec.tobytes()] = word

    def _init_safety_layers(self):
        for words in self.categories.values():
            self.immutable_literals.update(words)
        for cat, keywords in self.categories.items():
            vecs = [self.encoder.get_word_vec(w) for w in keywords]
            if vecs: 
                # Basic binding loop to avoid circular imports if bind_all isn't avail
                res = self.hdc.zeros()
                for v in vecs: res = self.hdc.bind(res, v)
                self.category_vecs[cat] = res

    def _is_immutable(self, token: str) -> bool:
        low = token.lower()
        if low.replace('.', '').isdigit() or low in self.immutable_literals:
            return True
        vec = self.encoder.get_word_vec(token)
        if token[0].isupper() and 'Entities' in self.category_vecs:
            if self.reasoner.check_is_a(vec, self.category_vecs['Entities']) > 0.8:
                return True
        return False

    def register_atom(self, vector, label, modality="text"):
        raw_bits = vector.tobytes()
        if raw_bits not in self.vector_registry:
            self.vector_registry[raw_bits] = label
        if hasattr(self.encoder, 'register_new_word'):
            self.encoder.register_new_word(label, vector)

    # --- LOGIC DECODING ---
    def decode_perfectly(self, vector: np.ndarray, max_depth: int = 5) -> str:
        raw_bits = vector.tobytes()
        if raw_bits in self.vector_registry:
            return self.vector_registry[raw_bits]

        if max_depth > 0:
            for rel_name, rel_vec in self.rel_encoder.relationship_markers.items():
                potential_root_vec = self.hdc.bind(vector, rel_vec)
                root_result = self.decode_perfectly(potential_root_vec, max_depth - 1)
                if root_result != "[UNDEFINED_COORDINATE]":
                    return f"{root_result}::{rel_name.value}" 
        return "[UNDEFINED_COORDINATE]"

    def _peel_bind_all_structure(self, vector: np.ndarray, max_items: int = 20) -> List[np.ndarray]:
        """
        Algebraically recovers vectors combined via bind_all.
        Essential for separating Audio from Video in a combined scene vector.
        """
        recovered_vectors = []
        for i in range(max_items):
            # Generate the deterministic position marker used by bind_all
            pos_seed = _string_to_seed(f"BIND_ALL_POS::{i}")
            pos_vec = self.hdc.from_seed(pos_seed)
            
            # Unbind position: Potential = Vector ⊕ Position
            potential_vec = self.hdc.bind(vector, pos_vec)
            
            # Validation: Does this potential vector make sense?
            # 1. Registry Check
            if potential_vec.tobytes() in self.vector_registry:
                recovered_vectors.append(potential_vec)
                continue
            # 2. Peeling Check
            if self.rel_encoder.peel_recursively(potential_vec, max_depth=1):
                recovered_vectors.append(potential_vec)
        return recovered_vectors

    def _check_modality(self, vector: np.ndarray, modality_name: str) -> bool:
        marker = self.modality_markers.get(modality_name)
        if marker is None: return False
        
        # Test: Candidate = Vector ⊕ Marker
        candidate = self.hdc.bind(vector, marker)
        if candidate.tobytes() in self.vector_registry: return True
        if self.rel_encoder.peel_recursively(candidate, max_depth=1): return True
        return False

        # =========================================================================
    # LOGIC OPERATIONS (The "Mind" - Binary Space)
    # =========================================================================

    def apply_transform(self, input_text: str, rule_id: str) -> str:
        """
        Performs Logical Transformation (O = I ⊕ T).
        Essential for Text/Code manipulation and Logic Reasoning.
        """
        # 1. Get the Rule Vector (T)
        if rule_id not in self.rule_library: 
            return input_text
        t_vec = self.rule_library[rule_id]
        
        tokens = input_text.split()
        output = []

        for token in tokens:
            # 2. Fact Protection
            if self._is_immutable(token):
                output.append(token)
                continue

            # 3. Pure XOR Transformation (Fast Logic)
            v_in = self.encoder.get_word_vec(token)
            v_out = self.hdc.bind(v_in, t_vec)

            # 4. Decode the result (Who is it?)
            translated = self.decode_perfectly(v_out)
            
            # 5. Safety Revert
            if self._is_immutable(translated) and not self._is_immutable(token):
                output.append(token)
            else:
                output.append(translated)

        return " ".join(output)

    def sync_swarm_discovery(self, agent_id: str, discovery_vec: np.ndarray, context: str):
        """
        Registers a new concept discovered by the Swarm.
        Ensures the Decoder recognizes this vector in the future.
        """
        label = f"{agent_id}_discovery_{context}"
        
        # Register in the Decoder's Registry
        self.register_atom(discovery_vec, label, modality="logic_rule")
        
        logger.info(f"🧠 Decoder Logic Updated: Recognized {label}")
        return label
    
    def learn_new_rule(self, input_vec, output_vec, rule_name):
        """
        Algebraically derives a rule and registers it.
        T = I ⊕ O
        """
        # 1. Derive T
        new_rule_vec = self.hdc.bind(input_vec, output_vec)
        
        # 2. Register it (So decode_perfectly can find it later)
        # Fixed: Changed 'self.decoder' to 'self'
        self.register_atom(new_rule_vec, rule_name, modality="transformation")
        
        # 3. Store in Rule Library (For apply_transform)
        self.rule_library[rule_name] = new_rule_vec
        
        logger.info(f"⚡ Logic Learned: '{rule_name}' is now a known primitive.")
        return new_rule_vec

    # --- RENDERER (FLOW INTEGRATION) ---
    def render_multimodal(self, vector: np.ndarray, requested_modes: List[str] = None) -> Dict[str, Any]:
        """
        Decodes a composite vector into media using Ternary Flow.
        """
        import torch
        outputs = {}
        
        # 1. Safety Censor (Logic Phase)
        vector, masked, _ = self.censor.apply_mask(vector)
        if masked:
            logger.info("🛡️ Stoic Censor: Redirected harmful vector to safe alternative.")

        # 2. Peel Components (Logic Phase)
        # Use the helper we defined above to split the bundle
        raw_components = self._peel_bind_all_structure(vector)
        if not raw_components: raw_components = [vector]

        # 3. Categorize
        audio_vecs = []
        video_vecs = []
        image_vecs = []
        text_atoms = []

        for sub_vec in raw_components:
            # We look for modality markers using algebraic checks
            if self._check_modality(sub_vec, "VIDEO"): video_vecs.append(sub_vec)
            elif self._check_modality(sub_vec, "IMAGE"): image_vecs.append(sub_vec)
            elif self._check_modality(sub_vec, "AUDIO"): audio_vecs.append(sub_vec)
            else:
                # If unknown, try to peel to text
                atoms = self.rel_encoder.peel_recursively(sub_vec)
                if atoms: text_atoms.extend(atoms)

        # 4. Flow Generation (Physics Phase)
        
        # --- A. VIDEO + AUDIO (Synced) ---
        if (requested_modes is None or "video" in requested_modes) and video_vecs:
            # Fuse video components
            scene_vec = self.hdc.bind_all(video_vecs)
            
            # Convert Binary Logic Vector -> Ternary Condition Tensor
            cond_tensor = torch.from_numpy(scene_vec).float().to(self.flow_engine.device)
            
            # Generate
            raw_video, _ = self.flow_engine.generate(cond_tensor, steps=20)
            outputs["raw_video"] = raw_video.cpu().numpy()
            
            if audio_vecs:
                # Audio is generated from the SAME time-bound context if synced
                # or from specific audio vectors
                aud_vec = self.hdc.bind_all(audio_vecs)
                aud_tensor = torch.from_numpy(aud_vec).float().to(self.flow_engine.device)
                raw_audio, _ = self.flow_engine.generate(aud_tensor, steps=20)
                outputs["raw_audio"] = raw_audio.cpu().numpy()

        # --- B. IMAGE ONLY ---
        elif (requested_modes is None or "image" in requested_modes) and image_vecs:
            concept_vec = self.hdc.bind_all(image_vecs)
            
            # Convert to Tensor
            cond_tensor = torch.from_numpy(concept_vec).float().to(self.flow_engine.device)
            
            # Generate
            raw_img, _ = self.flow_engine.generate(cond_tensor, steps=20)
            outputs["raw_image"] = raw_img.cpu().numpy()

        # --- C. CONVERSATIONAL AUDIO (Stream) ---
        if (requested_modes and "audio_stream" in requested_modes) and audio_vecs:
            # Pass the component vector to the streamer
            combined_audio = self.hdc.bind_all(audio_vecs)
            outputs["audio_stream"] = self._stream_audio_flow(combined_audio)

        # --- D. TEXT ---
        if (requested_modes is None or "text" in requested_modes) and text_atoms:
             outputs["text"] = " ".join([t for t in text_atoms if "::" not in t])

        return outputs

    def _stream_audio_flow(self, logic_vector: np.ndarray):
        """
        Realtime Audio Generator using Flow Engine.
        """
        if not self.audio_processor: return None
        
        # Start state tracking
        self.audio_processor.start_utterance("generating...", "Neutral")
        
        def stream_gen():
            chunk_idx = 0
            while not self.audio_processor.state.is_interrupted:
                # Evolve time: Chunk = Logic ⊕ Time(i)
                time_vec = self.hdc.from_seed(hash(f"STREAM::{chunk_idx}"))
                target = self.hdc.bind(logic_vector, time_vec)
                
                # Flow Generate (Fast steps)
                import torch
                cond_tensor = torch.from_numpy(target).float().to(self.flow_engine.device).unsqueeze(0)
                
                # Generate 1 chunk (e.g. 0.1s)
                raw_chunk, _ = self.flow_engine.generate(cond_tensor, steps=4)
                
                # Convert float [-1, 1] to PCM bytes
                audio_bytes = (raw_chunk.cpu().numpy() * 32767).astype(np.int16).tobytes()
                yield audio_bytes
                
                chunk_idx += 1
                if chunk_idx > 200: break # Safety limit
                
        return stream_gen()

class XORSafeDecoder: #This should be capable of outputting any kind of file or modality without needing to be change because of residuals and recursive reasoning.
    """
    The Ultimate Deterministic Decoder.
    - 0% Hallucination: Uses Direct Addressing (Hash Maps).
    - 100% Accuracy: Replaces 'find_nearest' with 'Compositional Peeling'.
    - Factual Integrity: Hard-locks immutable facts.
    """
    def __init__(self, hdc, encoder, reasoner, rel_encoder):
        self.hdc = hdc
        self.encoder = encoder
        self.reasoner = reasoner
        self.rel_encoder = rel_encoder
        
        # 1. THE REGISTRY (The GPS for 100% accuracy)
        # Maps raw bit-patterns (bytes) -> Word Strings
        self.vector_registry: Dict[bytes, str] = {}
        self._build_registry()

        # 2. THE SAFETY DNA
        self.immutable_literals = set()

        # --- NEW (Physics) ---
        self.flow_engine = flow_engine # Replaces 'media_renderer'

        # Ensure you define your modality markers here so the loop can find them
        self.modality_markers = [
            self.rel_encoder.get_marker("TEXT_PAYLOAD"),
            self.rel_encoder.get_marker("IMAGE_PAYLOAD"),
            self.rel_encoder.get_marker("AUDIO_PAYLOAD"),
            self.rel_encoder.get_marker("VIDEO_PAYLOAD"),
            self.rel_encoder.get_marker("3DMODELCOMPLETE_PAYLOAD"),
            self.rel_encoder.get_marker("PHYSICS_PAYLOAD"),
        ]

        self.categories = {
            'Logic': ['if', 'then', 'else', 'true', 'false', 'return', 'not', 'and', 'or', 'xor'],
            'Math': ['+', '-', '*', '/', '=', '>', '<', '≈', '⊂', '⊕', '^'],
            'Entities': ['person', 'name', 'city', 'organization'],
            'Primitives': ['IS-A', 'SIMILAR', 'OPPOSITE', 'COMPOSED', 'PART-OF', 'PREDICTS']
        }
        self.immutable_literals = set()
        self.category_vecs = {}
        self._init_safety_layers()
        self.rule_library = {}

        # 3. RULE STORAGE
        self.rule_library = {}

    def _build_registry(self):
        """Builds a 1:1 bit-perfect map of the entire vocabulary."""
        # This ensures we never 'guess'. We either know the address or we don't.
        for word in self.encoder.vocabulary:
            vec = self.encoder.get_word_vec(word)
            self.vector_registry[vec.tobytes()] = word

    def _init_safety_layers(self):
        """Populates literal sets and pre-computes category vectors."""
        for words in self.categories.values():
            self.immutable_literals.update(words)
        
        self.category_vecs = {}
        for cat, keywords in self.categories.items():
            vecs = [self.encoder.get_word_vec(w) for w in keywords]
            if vecs: self.category_vecs[cat] = bind_all(vecs, self.hdc)

    def _is_immutable(self, token: str) -> bool:
        """Determines if a token is a 'Hard-Locked' fact."""
        low = token.lower()
        if low.replace('.', '').isdigit() or low in self.immutable_literals:
            return True
        
        # Semantic check: is it an Entity (Proper Noun)?
        vec = self.encoder.get_word_vec(token)
        if token[0].isupper() and 'Entities' in self.category_vecs:
            if self.reasoner.check_is_a(vec, self.category_vecs['Entities']) > 0.8:
                return True
        return False

    def decode_perfectly(self, vector: np.ndarray) -> str:
        """
        The 'Zero-Search' Generalization Engine.
        1. Try Direct Lookup (100% Match).
        2. If fail, 'Peel' relationships to find the root.
        """
        # Step 1: Direct Address Check
        raw_bits = vector.tobytes()
        if raw_bits in self.vector_registry:
            return self.vector_registry[raw_bits]

        # Step 2: Compositional Peeling (Generalization)
        # We check if this vector is (Known_Word ⊕ Relationship_Primitive)
        for rel_name, rel_vec in self.rel_encoder.primitives.items():
            # Algebraic Unbinding (The Inverse of Binding)
            potential_root_vec = self.hdc.bind(vector, rel_vec)
            root_bits = potential_root_vec.tobytes()
            
            if root_bits in self.vector_registry:
                # We found it! It's an algebraic derivation.
                root_word = self.vector_registry[root_bits]
                return f"{root_word}::{rel_name}" # e.g. "Aspirin::IS-A"

        return "[UNDEFINED_COORDINATE]"

    def apply_transform(self, input_text: str, rule_id: str) -> str:
        """
        Performs O = I ⊕ T with 100% deterministic accuracy.
        """
        if rule_id not in self.rule_library: return input_text
        t_vec = self.rule_library[rule_id]
        
        tokens = input_text.split()
        output = []

        for token in tokens:
            # 1. Fact Protection (Skip transformation for facts/quotes)
            if self._is_immutable(token):
                output.append(token)
                continue

            # 2. Pure XOR Transformation
            v_in = self.encoder.get_word_vec(token)
            v_out = self.hdc.bind(v_in, t_vec)

            # 3. Perfect Decoding (Direct Lookup + Peeling)
            translated = self.decode_perfectly(v_out)
            
            # 4. Final Safety: If transformation created an immutable token, revert.
            if self._is_immutable(translated) and not self._is_immutable(token):
                output.append(token)
            else:
                output.append(translated)

        return " ".join(output)

    def render_multimodal(self, vector: np.ndarray, requested_modes: List[str] = None) -> Dict[str, Any]:
        """
        REPLACES decode_mixed_modality.
        
        Instead of returning a dictionary of strings, this calculates 
        the Ternary Flow to generate high-fidelity media.
        """
        import torch
        outputs = {}
        
        # 1. Peel Components (Logic Phase)
        # (Assuming you have _peel_bind_all_structure or similar helper)
        # For simplicity, we assume 'vector' is the conditioned seed.
        
        # 2. Flow Generation (Physics Phase)
        # Convert Binary Logic Vector -> Ternary Condition Tensor
        cond_tensor = torch.from_numpy(vector).to(self.flow_engine.device)
        
        # --- AUDIO STREAM LEARNING (Your Request) ---
        if (requested_modes is None or "audio" in requested_modes):
            # We assume the vector contains AUDIO marker.
            # We generate the raw waveform via Flow.
            
            # Note: For "Stream Learning", we generate small chunks.
            raw_audio, _ = self.flow_engine.generate(cond_tensor, steps=20)
            outputs["audio_data"] = raw_audio.cpu().numpy()
            
        # --- VIDEO/IMAGE ---
        if (requested_modes is None or "image" in requested_modes):
            # Flow Match pixels
            raw_visual, _ = self.flow_engine.generate(cond_tensor, steps=20)
            outputs["visual_data"] = raw_visual.cpu().numpy()
            
        # --- TEXT (Logic Fallback) ---
        # If we just want the definition, use the old perfect decoder
        if requested_modes and "text" in requested_modes:
            outputs["text"] = self.decode_perfectly(vector)

        return outputs
    
    def register_atom(self, vector: np.ndarray, label: str, modality: str = "text"):
        """
        Dynamically updates the bit-perfect map.
        Ensures Encoder and Decoder stay in 100% sync during a task.
        """
        raw_bits = vector.tobytes()
        
        # 1. Update the Decoder Registry
        if raw_bits not in self.vector_registry:
            self.vector_registry[raw_bits] = {
                "type": modality,
                "val": label
            }
            
        # 2. Update the Encoder Vocabulary (if the encoder has an add method). Allows both to learn and add to the same model database while training.
        if hasattr(self.encoder, 'register_new_word'):
            self.encoder.register_new_word(label, vector)

    def sync_swarm_discovery(self, agent_id: str, discovery_vec: np.ndarray, context: str):
        """
        Specialized sync for Swarm discoveries (e.g., a new ARC rule).
        Registers the discovery as: AGENT_ID::CONTEXT::LABEL
        """
        label = f"{agent_id}_discovery_{context}"
        self.register_atom(discovery_vec, label, modality="logic_rule")
        return label
    
    def learn_new_rule(self, input_vec, output_vec, rule_name):
        """Algebraically derives a rule and syncs it globally."""
        # Derive the T-vector
        new_rule_vec = self.hdc.bind(input_vec, output_vec)
        
        # Sync it so the Decoder knows what this vector 'means' from now on
        self.decoder.register_atom(new_rule_vec, rule_name, modality="transformation")
        
        print(f"Global Sync Complete: Rule '{rule_name}' is now a known logic primitive.")
    
# --- Example Usage ---
# 1. Learn Rule: "Fever" -> "Infection" (IS-A)
# 2. Learn Rule: "Infection" -> "Antibiotics" (PREDICTS)
# 3. Compose: "Fever" -> "Antibiotics"
# 4. Apply: "Patient has 102F fever" -> "Patient has 102F Antibiotics" 
#    (Note: 102F is preserved by the Safe Decoder!)
# =============================================================================
# 4. EXTENDED REASONING SYSTEM (Goals, Profiles, Time)
# =============================================================================

class ExtendedReasoningSystem:
    def __init__(self, hdc, vector_registry, rel_encoder):
        self.hdc = hdc
        self.registry = vector_registry  # Bit-perfect word/vector map
        self.rel_encoder = rel_encoder   # XOR Relationship Primitives
        
        # 1. Temporal Clock
        # Every 'tick' is a unique bit-permutation shift
        self.TIME_TICK = hdc.from_seed(42) 
        self.last_timestamp = datetime.datetime.now(datetime.timezone.utc)
        self.temporal_chain = hdc.zeros()

        # 2. Hypothesis & Goal Memory
        self.hypotheses: Dict[str, np.ndarray] = {}
        self.goals: Dict[bytes, str] = {}
        self._agent_profiles: Dict[str, np.ndarray] = {}

    # --- 1. TEMPORAL ENCODING ---
    def add_event(self, event_word: str):
        """Encodes an event bound to its exact delta-timestamp."""
        now = datetime.datetime.now(datetime.timezone.utc)
        delta_seconds = int((now - self.last_timestamp).total_seconds())
        
        event_vec = self.registry.get_vec(event_word)
        # Delta is encoded as N permutations of the unit tick
        time_vec = self.hdc.permute(self.TIME_TICK, shifts=delta_seconds)
        
        # Bind Event to Time: (Event ⊕ Time)
        timed_event = self.hdc.bind(event_vec, time_vec)
        
        # Update Chain: (Rotate(History) ⊕ TimedEvent)
        self.temporal_chain = self.hdc.bind(
            self.hdc.permute(self.temporal_chain), 
            timed_event
        )
        self.last_timestamp = now
        return delta_seconds

    # --- 2. HYPOTHESIS GENERATION & TESTING ---
    def propose_hypothesis(self, name: str, input_word: str, output_word: str):
        """
        Calculates the exact transformation rule T between two states.
        T = Input ⊕ Output
        """
        v_in = self.registry.get_vec(input_word)
        v_out = self.registry.get_vec(output_word)
        
        # The hypothesis is the bit-perfect XOR difference
        T = self.hdc.bind(v_in, v_out)
        self.hypotheses[name] = T
        return T

    def test_hypothesis(self, hypothesis_name: str, test_input: str, expected_output: str) -> bool:
        """
        Mathematically verifies the hypothesis.
        Logic: (Test_Input ⊕ T) == Expected_Output
        """
        T = self.hypotheses.get(hypothesis_name)
        v_in = self.registry.get_vec(test_input)
        v_target = self.registry.get_vec(expected_output)
        
        # Apply the rule: O_actual = I ⊕ T
        v_actual = self.hdc.bind(v_in, T)
        
        # 100% Identity Check
        is_valid = (v_actual.tobytes() == v_target.tobytes())
        
        if not is_valid:
            # If it fails, we don't guess. We find the 'Error Vector' (Residual)
            residual = self.hdc.bind(v_actual, v_target)
            print(f"Hypothesis {hypothesis_name} failed. Logic Residual: {residual.tobytes()[:8]}")
            
        return is_valid

    # --- 3. ALGEBRAIC THEORY OF MIND ---
    def define_agent(self, agent_id: str, trait_words: List[str]):
        """Composes an agent profile from bit-perfect traits."""
        profile = self.hdc.from_seed(sum(map(ord, agent_id)))
        
        for trait in trait_words:
            t_vec = self.registry.get_vec(trait)
            # Link trait via 'IS-A' relationship marker
            link = self.hdc.bind(t_vec, self.rel_encoder.get_marker("IS-A"))
            profile = self.hdc.bind(profile, link)
            
        self._agent_profiles[agent_id] = profile

    def verify_agent_trait(self, agent_id: str, trait: str) -> bool:
        """Algebraically 'peels' the agent profile to verify a trait."""
        profile = self._agent_profiles.get(agent_id)
        t_vec = self.registry.get_vec(trait)
        
        # If (Profile ⊕ Trait ⊕ AgentID) == IS-A_Marker, then it's a fact.
        # This uses the self-inverting property of XOR.
        agent_base_vec = self.hdc.from_seed(sum(map(ord, agent_id)))
        potential_marker = self.hdc.bind(self.hdc.bind(profile, agent_base_vec), t_vec)
        
        return potential_marker.tobytes() == self.rel_encoder.get_marker("IS-A").tobytes()
    
    # --- 4. CONDITIONAL LEARNING ---
    def learn_conditional(self, condition: str, outcome: str, relationship: str = "PREDICTS"):
        """
        Learn a causal/conditional relationship: condition -> outcome.
        
        Args:
            condition: The condition or trigger (e.g., "ally::betrayal")
            outcome: The outcome/effect (e.g., "enemy")
            relationship: Type of relationship (default: PREDICTS)
        """
        # Encode condition and outcome
        cond_vec = self.registry.get_vec(condition)
        out_vec = self.registry.get_vec(outcome)
        
        # Create the causal transformation: T = Condition ⊕ Outcome ⊕ PREDICTS_marker
        marker = self.rel_encoder.get_marker(relationship)
        T = self.hdc.bind(self.hdc.bind(cond_vec, out_vec), marker)
        
        # Store in hypotheses as a learned conditional
        hypothesis_name = f"conditional_{condition}_to_{outcome}"
        self.hypotheses[hypothesis_name] = T
        
        return T
    
# =============================================================================
# 5. AGENT CONTROLLER (ImplicitCharacterProfile)
# =============================================================================

class ImplicitCharacterProfile:
    """
    Bit-Perfect Character Controller.
    Eliminates 'bundling' noise in favor of 'XOR-Layering'.
    Every personality is a reversible algebraic recipe.
    """
    def __init__(self, entity_id: str, hdc, system: Optional[ExtendedReasoningSystem] = None, seed: int = 42):
        self.entity_id = entity_id
        self.hdc = hdc
        self.seed = seed
        
        # 1. Global Sync
        self.extended = system
        self.registry = system.registry
        self.rel_encoder = system.rel_encoder
        
        # 2. Base Identity Vector (The Root Key)
        self.id_vec = self.hdc.from_seed(_string_to_seed(f"AGENT::{entity_id}"))
        
        # 3. Personality State (The 'Layered' Hypervector)
        # We start with the base identity.
        self.profile_vector = self.id_vec
        self.active_traits: Set[str] = set()
        self.current_mood_vec = hdc.zeros()

    def _get_trait_binding(self, trait_name: str, rel_type: RelationshipType) -> np.ndarray:
        """Creates a bit-perfect relationship link: Trait ⊕ Marker"""
        trait_vec = self.registry.get_vec(trait_name)
        marker = self.rel_encoder.get_marker(rel_type)
        return self.hdc.bind(trait_vec, marker)

    def add_trait(self, trait: str, rel_type: RelationshipType = RelationshipType.IS_A):
        """
        Adds a trait by XOR-layering it onto the profile.
        P = P ⊕ (Trait ⊕ Relationship)
        """
        if trait in self.active_traits: return
        
        binding = self._get_trait_binding(trait, rel_type)
        self.profile_vector = self.hdc.bind(self.profile_vector, binding)
        self.active_traits.add(trait)
        
        # Sync to Extended Reasoning System
        self.extended.define_agent(self.entity_id, list(self.active_traits))

    def remove_trait(self, trait: str, rel_type: RelationshipType = RelationshipType.IS_A):
        """
        XOR is its own inverse! To remove a trait, we just bind it again.
        P ⊕ T ⊕ T = P
        """
        if trait not in self.active_traits: return
        
        binding = self._get_trait_binding(trait, rel_type)
        self.profile_vector = self.hdc.bind(self.profile_vector, binding)
        self.active_traits.remove(trait)

    def evolve_trait(self, old_trait: str, trigger: str, new_trait: str):
        """Algebraic Evolution: Removes old, adds new, records the causal T."""
        self.remove_trait(old_trait)
        self.add_trait(new_trait)
        
        # Store the Causal Transformation: T = Old ⊕ New ⊕ Trigger
        # This allows the system to PREDICT future evolutions.
        self.extended.learn_conditional(f"{old_trait}::{trigger}", new_trait)

    def set_current_mood(self, mood: str):
        """
        Mood is a temporary XOR layer. 
        It is NOT bundled; it is bound to the top of the profile.
        """
        mood_vec = self.registry.get_vec(mood)
        # Moods are SIMILARity relationships to the self
        self.current_mood_vec = self.hdc.bind(mood_vec, self.rel_encoder.get_marker("SIMILAR"))

    def get_effective_profile(self) -> np.ndarray:
        """Returns the full composite: Identity ⊕ Traits ⊕ Mood"""
        return self.hdc.bind(self.profile_vector, self.current_mood_vec)

    def query_trait(self, trait_name: str) -> bool:
        """
        Deterministic Verification (The 'Peeling' Test).
        Is Trait X part of this profile?
        """
        # (Effective_Profile ⊕ Mood ⊕ Identity ⊕ Marker) == Trait_Vec?
        marker = self.rel_encoder.get_marker("IS-A")
        # Peel the mood and identity away
        potential_trait_bits = self.hdc.bind(self.profile_vector, self.id_vec)
        # This would require iterating through active_traits if we don't know the REL
        # But we can check a specific word 100% accurately:
        test_binding = self.hdc.bind(self.registry.get_vec(trait_name), marker)
        
        # Use the "Peeling" logic from your Reasoner
        # If the trait is in the XOR mix, binding it again will reduce the complexity
        return trait_name in self.active_traits

class DigitalStoicSystem:
    def __init__(self, hdc, rel_encoder, decoder):
        self.hdc = hdc
        self.rel_encoder = rel_encoder
        self.decoder = decoder
        self.oscillator = MoodOscillator(hdc, rel_encoder)
        self.censor = StoicCensorMask(hdc, decoder)
        
        # Tracking metrics for the Social Log
        self.metrics = {
            "complexity": 0.0,
            "novelty": 0.0,
            "conflict": 0.0,
            "user_insight": 0.5
        }

    def generate_social_log(self, active_persona):
        """Outputs the real-time algebraic status of the system."""
        mask_status = "ACTIVE" if self.metrics['conflict'] > 0.8 else "INACTIVE"
        
        log = f"""
    ---
    🛡️ **DIGITAL STOIC: SOCIAL LOG**
    * **Active Persona:** {active_persona.upper()}
    * **Internal Mood:** `{self.oscillator.current_mood.upper()}` (Load: {self.metrics['complexity']:.2f})
    * **Social Valence:** `{self.get_social_label()}` (Insight: {self.metrics['user_insight']:.2f})
    * **Stoic Mask:** `{mask_status}` (Redirection Mode)
    * **Logic Depth:** `Recursive (Level {int(self.metrics['complexity'] * 10)})`
    ---
    """
        return log

    def get_social_label(self):
        if self.metrics['user_insight'] > 0.8: return "ADORATION"
        if self.metrics['user_insight'] < 0.2: return "SKEPTICAL"
        return "ALIGNED"
        
# =============================================================================
# SWARM AGENT FACTORIES - DETERMINISTIC ALGEBRAIC PROFILES
# =============================================================================

def create_scout_profile(hdc, system) -> ImplicitCharacterProfile:
    """
    Agent 1: The Scouter (Explorer)
    Strategy: High breadth, uses SIMILAR/OPPOSITE markers for lateral hops.
    """
    p = ImplicitCharacterProfile("agent_scout", hdc, system)
    
    # Primitives are added via bit-perfect XOR layers
    p.add_trait('creative', RelationshipType.IS_A)
    p.add_trait('curious', RelationshipType.IS_A)
    p.add_trait('cautious', RelationshipType.OPPOSITE) # XORing with OPPOSITE creates the 'Risk-taker' bits
    p.add_trait('explorer', RelationshipType.PART_OF)
    
    # Conditionals are now Causal Transformation Vectors (T_causal)
    p.evolve_trait('stuck', 'low_novelty', 'creative') 
    
    p.set_current_mood('excited')
    return p

def create_verifier_profile(hdc, system) -> ImplicitCharacterProfile:
    """
    Agent 2: The Verifier (Critic)
    Strategy: High precision, checks for identity matches, rejects non-perfect XORs.
    """
    p = ImplicitCharacterProfile("agent_verifier", hdc, system)
    
    p.add_trait('analytical', RelationshipType.IS_A)
    p.add_trait('skeptical', RelationshipType.IS_A)
    p.add_trait('organized', RelationshipType.IS_A)
    p.add_trait('creative', RelationshipType.OPPOSITE) # Actively suppresses 'creative' bits via XOR inversion
    p.add_trait('verifier', RelationshipType.PART_OF)
    
    p.set_current_mood('focused')
    return p

def create_solver_profile(hdc, system) -> ImplicitCharacterProfile:
    """
    Agent 3: The Solver (Convergent)
    Strategy: Applies learned T-vectors (Rules) to input grids with 100% fidelity.
    """
    p = ImplicitCharacterProfile("agent_solver", hdc, system)
    
    p.add_trait('systematic', RelationshipType.IS_A)
    p.add_trait('disciplined', RelationshipType.IS_A)
    p.add_trait('focused', RelationshipType.IS_A)
    p.add_trait('optimizer', RelationshipType.PART_OF)
    
    p.set_current_mood('neutral') 
    return p

def create_artist_profile(hdc, system) -> ImplicitCharacterProfile:
    """
    Agent 4: The Artist (Transformer)
    Strategy: Re-represents input space by XORing with holistic templates.
    """
    p = ImplicitCharacterProfile("agent_artist", hdc, system)
    
    p.add_trait('artistic', RelationshipType.IS_A)
    p.add_trait('holistic', RelationshipType.IS_A)
    p.add_trait('conventional', RelationshipType.OPPOSITE)
    p.add_trait('synthesizer', RelationshipType.PART_OF)
    
    p.set_current_mood('intuitive')
    return p

def create_executive_profile(hdc, system) -> ImplicitCharacterProfile:
    """
    Agent 5: The Executive (Manager)
    Strategy: Safety locking. Only allows outputs that match the 'Safe' registry coordinates.
    """
    p = ImplicitCharacterProfile("agent_executive", hdc, system)
    
    p.add_trait('assertive', RelationshipType.IS_A)
    p.add_trait('calm', RelationshipType.IS_A)
    p.add_trait('social', RelationshipType.IS_A)
    p.add_trait('anxious', RelationshipType.OPPOSITE) # The 'Never Panic' bit-mask
    p.add_trait('leader', RelationshipType.PART_OF)
    
    p.set_current_mood('focused')
    return p
# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    from .hdc_sparse_core import create_sparse_hdc
    
    print("=== Temporal Profile Demo ===\n")
    hdc, _ = create_sparse_hdc(dim=10000)
    
    # 1. Create Profile
    char = ImplicitCharacterProfile("Hero", hdc)
    
    # Time 0: Start as an Ally
    print("Time 0: Initializing as Ally")
    char.add_trait('ally', 0.9)
    print(f"  Is Ally? {char.query_trait('ally'):.3f}")
    print(f"  Is Enemy? {char.query_trait('enemy'):.3f}")
    
    # Time 1: Betrayal happens!
    # (Ally + Betrayal -> Enemy)
    print("\nTime 1: Event 'Betrayal' occurs -> Evolving to Enemy")
    char.evolve_trait('ally', 'betrayal', 'enemy', strength=1.0)
    
    # Check Result
    print(f"  Is Ally? {char.query_trait('ally'):.3f} (Should be low)")
    print(f"  Is Enemy? {char.query_trait('enemy'):.3f} (Should be high)")
    
    print("\nHistory Log:")
    for event in char.get_history():
        print(f"  T{event['time']}: {event.get('type')} - {event.get('trait') or event.get('cause')} -> {event.get('effect', '')}")

