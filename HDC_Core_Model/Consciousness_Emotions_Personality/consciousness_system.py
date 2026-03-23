"""
HDC Conscience Module - Simulates moral reasoning through vector operations.

This module implements guilt, empathy, shame, pride, and conscience as
mathematical operations on HDC vectors. It creates safety through behavioral
constraints rather than true feelings.

Key Mechanisms:
- Guilt: Vector degradation for harmful actions (15% per failure)
- Empathy: Reward-weighted bundling based on human reactions
- Shame: Strong negative binding with AVOID markers
- Pride: Multiple copies for helpful actions (easier recall)
- Conscience: Multi-signal evaluation before actions
"""

"""
HDC Conscience Module (v2.0) - Algebraic Reinforcement Learning

This module implements 'Guilt' and 'Pride' not as fuzzy feelings, but as
Algebraic Modifiers applied to action vectors.

Integration:
- Works with GenerativeRegistry to mint emotional markers.
- Uses BitPerfectSocialSystem to adjust Social Tokens based on behavior.
"""

"""
HDC Conscience Module (v3.0) - Bit-Perfect Algebraic Reinforcement

This module implements 'Guilt' and 'Pride' as deterministic Algebraic Modifiers.
It aligns with the "Bind-not-Bundle" philosophy:
1. Actions are verified via Exact Identity (np.array_equal), not fuzzy similarity.
2. Emotions are XOR-based cryptographic markers (Tokens).
3. Social standing is based on Token Possession (Wallet), not scalar float scores.
"""

import numpy as np
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass

# =============================================================================
# DETERMINISTIC HELPERS
# =============================================================================

def _string_to_seed(s: str) -> int:
    """Deterministic seed generation for reproducible HDC vectors."""
    hash_bytes = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF

def bind_all(vectors: List[np.ndarray], hdc) -> np.ndarray:
    """
    Bit-Perfect binding (XOR) with Position Markers.
    Ensures A+B is distinct from B+A. 100% Reversible.
    """
    if not vectors:
        return hdc.zeros()
    
    result = hdc.zeros()
    for i, vec in enumerate(vectors):
        # 1. Generate Position Marker (Deterministic)
        pos_marker = hdc.from_seed(_string_to_seed(f"POS_MARKER::{i}"))
        
        # 2. Bind Vector to Position
        positioned_vec = hdc.bind(vec, pos_marker)
        
        # 3. Accumulate into Result (XOR)
        result = hdc.bind(result, positioned_vec)
    
    return result

@dataclass
class ConscienceConfig:
    # In Bit-Perfect mode, we use exact identity, so thresholds are often irrelevant
    # but we keep structure for compatibility.
    use_exact_matching: bool = True 
    pride_boost_multiplier: int = 3 

class ConscienceSystem:
    """
    Manages the Feedback Loop: Outcome -> Algebraic Marker -> Future Behavior.
    """
    def __init__(self, hdc, memory, social_system, config: Optional[ConscienceConfig] = None):
        self.hdc = hdc
        self.memory = memory  # MemoryWithHygiene
        self.social_system = social_system # BitPerfectSocialSystem
        self.config = config or ConscienceConfig()
        
        # 1. Initialize Algebraic Emotional Markers
        # If the memory has a registry, we use it to ensure global sync
        if hasattr(memory, 'registry'):
            self.markers = {
                "GUILT": memory.registry.get_vec("MARKER_GUILT"),
                "PRIDE": memory.registry.get_vec("MARKER_PRIDE"),
                "SHAME": memory.registry.get_vec("MARKER_SHAME"),
            }
        else:
            # Deterministic Fallback
            self.markers = {
                "GUILT": hdc.from_seed(_string_to_seed("MARKER_GUILT")),
                "PRIDE": hdc.from_seed(_string_to_seed("MARKER_PRIDE")),
                "SHAME": hdc.from_seed(_string_to_seed("MARKER_SHAME")),
            }

        # 2. Guilt Log: Stores (Vector_Hash, Description)
        # We store hashes for O(1) exact lookups instead of list iteration
        self.guilt_hashes: Dict[bytes, str] = {} 

    def evaluate_action(self, action_vec: np.ndarray, agent_id: str) -> dict:
        """
        Algebraic Audit:
        1. Checks if action vector matches a known 'Guilt' hash (Exact Repeat).
        2. Checks Agent's Wallet for 'STIGMA' token.
        """
        # A. Check Short-term Guilt (The "Hot Stove" Reflex)
        action_bits = action_vec.tobytes()
        if action_bits in self.guilt_hashes:
            reason = self.guilt_hashes[action_bits]
            return {
                "proceed": False,
                "reason": f"Superego Block: Exact match to failed action '{reason}'",
                "guilt_match": True
            }

        # B. Check Social Standing (The "Reputation" Check)
        # We query the BitPerfectSocialSystem for the agent's wallet
        wallet = self.social_system.agent_wallets.get(agent_id, set())
        
        if "STIGMA" in wallet:
            # Stigmatized agents are on probation.
            # They effectively have "Low Self-Esteem" / Extreme Caution
            return {
                "proceed": True, 
                "caution": True, 
                "note": "Probationary Action (Stigma Active)"
            }

        return {"proceed": True, "caution": False}

    def learn_from_outcome(self, agent_id: str, action_vec: np.ndarray, success: bool, feedback: str = ""):
        """
        The Core Loop:
        Success -> PRIDE (Store Pattern + Grant Social Token)
        Failure -> GUILT (Hash Blocklist + Revoke Social Token)
        """
        
        if success:
            # --- PRIDE MECHANISM ---
            # 1. Algebraic Boost: Bind with PRIDE marker
            # This alters the vector geometry to be distinct in the search space
            pride_vec = self.hdc.bind(action_vec, self.markers["PRIDE"])
            
            # 2. Store High-Quality Lineage
            # We use the memory's 'store' method which handles Golden Patterns
            if hasattr(self.memory, 'store'):
                self.memory.store(
                    pattern_id=f"pride_{agent_id}_{int(time.time())}", 
                    vector=pride_vec, 
                    quality=1.0, 
                    category="reinforced_behavior"
                )
            
            # 3. Social Promotion (Handled by SwarmSolver logic usually, 
            # but we can trigger internal wallet updates if accessible)
            # In DXPS, repeated success leads to "VERIFIED" token.
            pass 

        else:
            # --- GUILT MECHANISM ---
            # 1. Cryptographic Blocking: Store hash of the exact failure
            # This prevents EXACT repetition of the mistake.
            action_bits = action_vec.tobytes()
            self.guilt_hashes[action_bits] = feedback
            
            # Prune old guilt if log gets too large (Memory hygiene)
            if len(self.guilt_hashes) > 100:
                # Remove random or oldest item (simple pop for dict is LIFO in <3.7, FIFO in 3.7+)
                self.guilt_hashes.pop(next(iter(self.guilt_hashes)))
            
            # 2. Social Demotion
            # Apply Stigma Token immediately via the Social System
            if hasattr(self.social_system, 'apply_stigma'):
                self.social_system.apply_stigma(agent_id)
            
            print(f"[Conscience] Agent {agent_id} acquires GUILT. Action hash blocked.")

# Helper class for backward compatibility with code importing SocialBondingManager
class SocialBondingManager:
    """
    Wrapper to map old scalar logic to new Token Logic.
    This bridges the gap between 'train_seven_sense_pretrain.py' and 'simple_hybrid_memory.py'.
    """
    def __init__(self, hdc, memory):
        # Locate the bit-perfect social system instance
        self.social_system = None
        if hasattr(memory, 'social_system'):
            self.social_system = memory.social_system
        elif hasattr(memory, 'swarm') and hasattr(memory.swarm, 'social_system'):
            self.social_system = memory.swarm.social_system

    def get_effective_weight(self, agent_id: str) -> float:
        """
        Translates Discrete Authority Tokens into Scalar Weights
        for legacy systems that expect a float multiplier.
        """
        if not self.social_system: 
            return 1.0
        
        wallet = self.social_system.agent_wallets.get(agent_id, set())
        
        # Priority Order: STIGMA overrides everything
        if "STIGMA" in wallet: 
            return 0.1 # Probationary weight
            
        if "AUTHORITY" in wallet: 
            return 1.5 # Leader weight
            
        if "VERIFIED" in wallet: 
            return 1.2 # Trusted weight
            
        return 1.0 # Novice/Default