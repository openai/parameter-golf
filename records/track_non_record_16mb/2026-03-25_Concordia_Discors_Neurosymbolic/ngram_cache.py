import torch
import math
from collections import defaultdict

class NgramCache:
    """Backward-looking n-gram cache. Builds from already-scored tokens."""
    def __init__(self, max_order=5, vocab_size=1024):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(max_order+1)]
        self.totals = [defaultdict(int) for _ in range(max_order+1)]

    def observe(self, token_ids):
        for i in range(len(token_ids)):
            for order in range(min(i, self.max_order)+1):
                ctx = tuple(token_ids[i-order:i]) if order > 0 else ()
                self.counts[order][ctx][token_ids[i]] += 1
                self.totals[order][ctx] += 1

    def predict(self, context):
        probs = torch.zeros(self.vocab_size, dtype=torch.float32)
        best_order = -1
        for order in range(min(len(context), self.max_order), -1, -1):
            ctx = tuple(context[-order:]) if order > 0 else ()
            total = self.totals[order].get(ctx, 0)
            if total < 3: continue
            op = torch.zeros(self.vocab_size, dtype=torch.float32)
            for tok, count in self.counts[order][ctx].items():
                if tok < self.vocab_size: op[tok] = count / total
            w = min(0.95, total/(total+5)) * (0.6 + 0.4*order/max(self.max_order,1))
            if best_order < 0:
                probs = op
                best_order = order
            else:
                probs = w * op + (1-w) * probs
        if probs.sum() < 0.01: return None, 0.0
        probs = probs / probs.sum()
        total_obs = sum(self.totals[0].values())
        conf = min(0.5, total_obs / (total_obs + 500))
        return probs, conf

    def entropy_alpha(self, neural_logits):
        p = torch.softmax(neural_logits, dim=-1)
        ent = -(p * torch.log(p + 1e-10)).sum()
        norm_ent = ent / math.log(self.vocab_size)
        return min(0.4, max(0.05, 0.05 + 0.35 * norm_ent.item()))

    def reset(self):
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(self.max_order+1)]
        self.totals = [defaultdict(int) for _ in range(self.max_order+1)]
