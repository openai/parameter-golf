"""
LZ77-style Longest Match Predictor.
Finds the longest matching substring in already-scored tokens,
predicts continuation from the match. This is how real compressors work.

The TACITUS parallel: pattern recognition across conflict precedents.
Just as a conflict analyst finds historical parallels to predict outcomes,
the match model finds textual parallels to predict continuations.

Zero artifact cost. Runs at eval time only. Backward-looking.
"""

class MatchModel:
    """Longest substring match predictor."""
    
    def __init__(self, vocab_size=1024, max_match_len=64, min_match_len=3):
        self.vocab_size = vocab_size
        self.max_match = max_match_len
        self.min_match = min_match_len
        self.history = []  # All scored tokens so far
    
    def observe(self, token_ids):
        """Add scored tokens to history."""
        if isinstance(token_ids, list):
            self.history.extend(token_ids)
        else:
            self.history.extend(token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids))
    
    def predict(self, context):
        """
        Find longest match in history, predict next token from continuation.
        
        Returns: (predicted_token, match_length, confidence) or (None, 0, 0.0)
        """
        if len(self.history) < self.min_match + 1 or len(context) < self.min_match:
            return None, 0, 0.0
        
        # Search for longest match of context suffix in history
        best_pos = -1
        best_len = 0
        
        ctx = context[-self.max_match:]  # Last N tokens of context
        hist = self.history
        
        # Search backwards through history for matches
        for match_len in range(min(len(ctx), self.max_match), self.min_match - 1, -1):
            pattern = ctx[-match_len:]
            # Search in history (not including last match_len tokens to avoid self-match)
            search_end = len(hist) - match_len
            if search_end <= 0:
                continue
            
            for pos in range(search_end - 1, max(0, search_end - 50000) - 1, -1):
                # Check if pattern matches at this position
                match = True
                for j in range(match_len):
                    if hist[pos + j] != pattern[j]:
                        match = False
                        break
                
                if match and pos + match_len < len(hist):
                    best_pos = pos
                    best_len = match_len
                    break  # Found longest match at this length
            
            if best_len > 0:
                break  # Found a match, don't search shorter patterns
        
        if best_len >= self.min_match and best_pos + best_len < len(hist):
            next_token = hist[best_pos + best_len]
            # Confidence scales with match length
            confidence = min(0.85, 0.3 + best_len * 0.05)
            return next_token, best_len, confidence
        
        return None, 0, 0.0
    
    def predict_probs(self, context, vocab_size=None):
        """Return probability distribution based on match."""
        import torch
        vs = vocab_size or self.vocab_size
        tok, mlen, conf = self.predict(context)
        if tok is None:
            return None, 0.0
        
        probs = torch.full((vs,), (1.0 - conf) / vs)
        if tok < vs:
            probs[tok] = conf
        probs = probs / probs.sum()
        return probs, conf
    
    def reset(self):
        self.history = []


if __name__ == "__main__":
    mm = MatchModel(vocab_size=1024)
    
    # Simulate: observe a document, then predict
    doc = [10, 20, 30, 40, 50, 60, 70, 80, 10, 20, 30, 40, 50, 60, 70, 80]
    mm.observe(doc[:12])
    
    # Context ends with [10, 20, 30, 40] — should match earlier occurrence
    tok, mlen, conf = mm.predict([10, 20, 30, 40])
    print(f"Match: token={tok}, length={mlen}, conf={conf:.2f}")
    print(f"Expected: token=50 (continuation after [10,20,30,40])")
    
    # Test with no match
    tok2, mlen2, conf2 = mm.predict([99, 98, 97])
    print(f"No match: token={tok2}, length={mlen2}, conf={conf2:.2f}")
    print("Match model working.")
