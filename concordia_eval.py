"""
CONCORDIA DISCORS EVALUATOR
============================
"Discordant Harmony" — productive tension between opposing predictors
creates something greater than any alone.

Architecture:
  DISCORDIA (tension):
    - Neural transformer: learns from data, uncertain, flexible
    - FST grammar: encodes rules, certain, rigid
    - N-gram cache: adapts in real-time, statistical, local
    - Match model: finds historical parallels, deterministic
    
  CONCORDIA (harmony):
    - Bayesian-optimal mediation via entropy-adaptive weighting
    - Each predictor's "claim" weighted by evidence strength
    - The mediator (gating function) doesn't pick winners —
      it finds the optimal compromise

This is conflict resolution applied to prediction.
The TACITUS OAG principle made concrete.

Academic basis:
  - Shannon (1948): prediction = compression
  - Willems et al. (1995): Context Tree Weighting (Bayesian optimal mixing)
  - Garcez & Lamb (2023): neurosymbolic integration
  - Veness et al. (2021): Gated Linear Networks
  - Berlin (1958): value pluralism — no single predictor has monopoly on truth
"""

import torch
import torch.nn.functional as F
import math
from fst_predictor import WebTextFST
from ngram_cache import NgramCache
from match_model import MatchModel

class ConcordiaEvaluator:
    """
    Unified hybrid evaluator combining all predictors.
    Named after Concordia Discors — the productive tension
    between discordant predictors creating harmonic prediction.
    """
    
    def __init__(self, tokenizer, vocab_size):
        self.sp = tokenizer
        self.vocab_size = vocab_size
        self.fst = WebTextFST(tokenizer)
        self.cache = NgramCache(max_order=8, vocab_size=vocab_size)  # Order 8 (upgraded from 5)
        self.match = MatchModel(vocab_size=vocab_size, max_match_len=32, min_match_len=3)
        
        # Mixing weights (tuned from our 7-config experiment)
        self.cache_max = 0.05
        self.fst_max = 0.05
        self.match_max = 0.10
        
        # Statistics
        self.stats = {
            'total': 0, 'fst_used': 0, 'cache_used': 0,
            'match_used': 0, 'neural_only': 0
        }
    
    def blend(self, neural_logits, context_tokens, context_text=None):
        """
        The Concordia function: blend discordant predictions into harmony.
        
        Args:
            neural_logits: [vocab_size] raw logits from transformer
            context_tokens: list of preceding token IDs
            context_text: decoded text (optional)
        
        Returns:
            final_log_probs: [vocab_size] blended log probabilities
        """
        self.stats['total'] += 1
        
        neural_probs = F.softmax(neural_logits.float().cpu(), dim=-1)
        
        # Measure neural uncertainty (entropy)
        entropy = -(neural_probs * torch.log(neural_probs + 1e-10)).sum()
        norm_entropy = (entropy / math.log(self.vocab_size)).item()
        
        final_probs = neural_probs.clone()
        used_any = False
        
        # === MATCH MODEL (highest priority — exact pattern match) ===
        if len(context_tokens) >= 3:
            match_probs, match_conf = self.match.predict_probs(context_tokens[-32:])
            if match_probs is not None and match_conf > 0.3:
                weight = min(self.match_max, match_conf * 0.12 * (0.5 + norm_entropy))
                final_probs = (1 - weight) * final_probs + weight * match_probs
                self.stats['match_used'] += 1
                used_any = True
        
        # === FST (structural patterns) ===
        if context_text and len(context_text) > 3:
            try:
                fst_probs, fst_conf = self.fst.predict(context_text)
                if fst_probs is not None and fst_conf > 0.4:
                    weight = min(self.fst_max, fst_conf * 0.06 * (0.5 + norm_entropy))
                    final_probs = (1 - weight) * final_probs + weight * fst_probs
                    self.stats['fst_used'] += 1
                    used_any = True
            except:
                pass
        
        # === N-GRAM CACHE (statistical patterns) ===
        if len(context_tokens) >= 2:
            cache_probs, cache_conf = self.cache.predict(context_tokens[-10:])
            if cache_probs is not None and cache_conf > 0.1:
                weight = min(self.cache_max, 0.02 + self.cache_max * norm_entropy * cache_conf)
                final_probs = (1 - weight) * final_probs + weight * cache_probs
                self.stats['cache_used'] += 1
                used_any = True
        
        if not used_any:
            self.stats['neural_only'] += 1
        
        # Ensure valid distribution
        final_probs = final_probs.clamp(min=1e-10)
        final_probs = final_probs / final_probs.sum()
        
        return torch.log(final_probs)
    
    def observe(self, token_ids):
        """Feed scored tokens to all backward-looking predictors."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        self.cache.observe(token_ids)
        self.match.observe(token_ids)
    
    def print_stats(self):
        t = max(self.stats['total'], 1)
        print(f"  Concordia stats: {t:,} positions")
        print(f"    Match model: {self.stats['match_used']:,} ({self.stats['match_used']/t*100:.1f}%)")
        print(f"    FST grammar: {self.stats['fst_used']:,} ({self.stats['fst_used']/t*100:.1f}%)")
        print(f"    N-gram cache: {self.stats['cache_used']:,} ({self.stats['cache_used']/t*100:.1f}%)")
        print(f"    Neural only: {self.stats['neural_only']:,} ({self.stats['neural_only']/t*100:.1f}%)")
    
    def reset(self):
        self.cache.reset()
        self.match.reset()
        self.stats = {k: 0 for k in self.stats}


# === STANDALONE TEST ===
if __name__ == "__main__":
    import numpy as np
    import sentencepiece as spm
    
    sp = spm.SentencePieceProcessor(model_file='data/tokenizers/fineweb_1024_bpe.model')
    V = sp.vocab_size()
    
    concordia = ConcordiaEvaluator(sp, V)
    
    raw = np.fromfile('data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin',
                       dtype=np.uint16, offset=1024)
    val_tokens = raw[:80000].tolist()
    
    chunk_size = 2048
    neural_loss = hybrid_loss = total = 0.0
    
    print("Running Concordia Discors evaluation on 80K FineWeb val tokens...")
    print("(Comparing uniform neural vs full hybrid stack)\n")
    
    for start in range(0, len(val_tokens) - chunk_size, chunk_size):
        chunk = val_tokens[start:start+chunk_size]
        
        for i in range(10, len(chunk)):
            target = chunk[i]
            context = chunk[max(0,i-50):i]
            total += 1
            
            # Uniform neural (worst case)
            neural_loss += -math.log(1.0 / V)
            
            # Concordia hybrid
            uniform_logits = torch.zeros(V)
            try:
                ctx_text = sp.decode(context[-20:])
            except:
                ctx_text = ""
            log_probs = concordia.blend(uniform_logits, context, ctx_text)
            hybrid_loss += -log_probs[target].item()
        
        concordia.observe(chunk)
    
    nb = (neural_loss / total) / math.log(2)
    hb = (hybrid_loss / total) / math.log(2)
    
    print(f"=== CONCORDIA DISCORS RESULTS ({total:,} tokens) ===")
    print(f"  Uniform baseline: {nb:.4f} BPB")
    print(f"  Concordia hybrid: {hb:.4f} BPB")
    print(f"  Improvement:      {nb - hb:.4f} BPB")
    print()
    concordia.print_stats()
