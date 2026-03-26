"""
Hybrid evaluation: blends neural model + FST + n-gram cache predictions.
This plugs into train_gpt.py's eval loop.

Usage: import and call hybrid_eval_val() instead of eval_val()
"""
import torch
import torch.nn.functional as F
import math
import sentencepiece as spm
from fst_predictor import WebTextFST
from ngram_cache import NgramCache

class HybridPredictor:
    """Combines neural model with FST structural predictor and n-gram cache."""

    def __init__(self, tokenizer_path, vocab_size):
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.fst = WebTextFST(self.sp)
        self.cache = NgramCache(max_order=5, vocab_size=vocab_size)
        self.vocab_size = vocab_size

        # Statistics tracking
        self.stats = {'total': 0, 'fst_used': 0, 'cache_used': 0, 'neural_only': 0}

    def blend_predictions(self, neural_logits, context_tokens, context_text=None):
        """
        Blend neural logits with FST and cache predictions.

        Args:
            neural_logits: [vocab_size] raw logits from transformer
            context_tokens: list of preceding token IDs
            context_text: decoded text (optional, computed if None)

        Returns:
            blended_log_probs: [vocab_size] log probabilities
        """
        self.stats['total'] += 1

        # Neural base distribution
        neural_probs = F.softmax(neural_logits.float(), dim=-1)

        # Compute neural entropy for adaptive mixing
        entropy = -(neural_probs * torch.log(neural_probs + 1e-10)).sum()
        max_entropy = math.log(self.vocab_size)
        norm_entropy = (entropy / max_entropy).item()

        # Start with neural as base
        final_probs = neural_probs.clone()
        used_fst = False
        used_cache = False

        # === FST PREDICTION ===
        if context_text is not None and len(context_text) > 5:
            try:
                fst_probs, fst_conf = self.fst.predict(context_text)
                if fst_probs is not None and fst_conf > 0.3:
                    # Scale FST weight by confidence AND neural uncertainty
                    fst_weight = min(0.5, fst_conf * 0.4 * (0.5 + norm_entropy))
                    final_probs = (1 - fst_weight) * final_probs + fst_weight * fst_probs.to(final_probs.device)
                    used_fst = True
            except:
                pass

        # === N-GRAM CACHE PREDICTION ===
        if len(context_tokens) >= 2:
            cache_probs, cache_conf = self.cache.predict(context_tokens[-10:])
            if cache_probs is not None and cache_conf > 0.1:
                # Entropy-adaptive: trust cache more when neural is uncertain
                cache_weight = min(0.4, cache_conf * (0.3 + 0.4 * norm_entropy))
                # Don't double-count if FST already adjusted
                if used_fst:
                    cache_weight *= 0.5
                final_probs = (1 - cache_weight) * final_probs + cache_weight * cache_probs.to(final_probs.device)
                used_cache = True

        # Track stats
        if used_fst:
            self.stats['fst_used'] += 1
        if used_cache:
            self.stats['cache_used'] += 1
        if not used_fst and not used_cache:
            self.stats['neural_only'] += 1

        # Ensure valid distribution
        final_probs = final_probs.clamp(min=1e-10)
        final_probs = final_probs / final_probs.sum()

        return torch.log(final_probs)

    def observe_scored_tokens(self, token_ids):
        """Feed already-scored tokens to the cache. MUST call after scoring."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        self.cache.observe(token_ids)

    def print_stats(self):
        t = max(self.stats['total'], 1)
        print(f"  Hybrid stats: {t} positions | "
              f"FST used: {self.stats['fst_used']} ({self.stats['fst_used']/t*100:.1f}%) | "
              f"Cache used: {self.stats['cache_used']} ({self.stats['cache_used']/t*100:.1f}%) | "
              f"Neural only: {self.stats['neural_only']} ({self.stats['neural_only']/t*100:.1f}%)")

    def reset(self):
        self.cache.reset()
        self.stats = {'total': 0, 'fst_used': 0, 'cache_used': 0, 'neural_only': 0}
