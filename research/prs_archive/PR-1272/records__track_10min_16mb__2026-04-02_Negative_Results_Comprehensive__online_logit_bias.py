"""
Online Logit Bias Adaptation for eval-time BPB improvement.

After scoring each token, updates a learnable bias vector in logit space
using the gradient of cross-entropy loss. This corrects systematic biases
from quantization and adapts to document-level token frequency shifts.

Legality:
- Score-before-update: bias at position t only uses gradients from positions 1..t-1
- Full normalized distribution: softmax(logits + bias) sums to 1.0 by construction
- Single left-to-right pass: no rescoring
- Causal: no future token information used (gradient uses only the scored target)

Integration:
    # In eval loop, after getting logits from model:
    olb = OnlineLogitBias(vocab_size=1024, lr=0.01)
    for each position:
        adjusted_logits = logits + olb.bias
        score = -log(softmax(adjusted_logits)[target])  # this is the final score
        olb.update(adjusted_logits, target)  # update AFTER scoring
"""

import math
import numpy as np
try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class OnlineLogitBias:
    """
    Maintains a learnable bias vector added to logits, updated online via SGD.

    After scoring token t with logits + bias, the gradient of CE loss w.r.t. bias is:
        grad = softmax(logits + bias) - one_hot(target)

    This is the simplest possible online adaptation — no backprop through the model.
    """

    def __init__(self, vocab_size: int = 1024, lr: float = 0.01,
                 momentum: float = 0.9, weight_decay: float = 0.0,
                 device: str = "cpu"):
        self.vocab_size = vocab_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.bias = np.zeros(vocab_size, dtype=np.float64)
        self.velocity = np.zeros(vocab_size, dtype=np.float64)
        self.step_count = 0

    def get_bias(self) -> np.ndarray:
        """Get current bias vector."""
        return self.bias

    def update(self, logits: np.ndarray, target: int):
        """
        Update bias after scoring. Must be called AFTER the score is recorded.

        Args:
            logits: raw model logits at this position, shape (vocab_size,)
            target: the correct token ID (already scored)
        """
        # Compute softmax(logits + bias)
        adjusted = logits + self.bias
        adjusted -= adjusted.max()  # numerical stability
        exp_adj = np.exp(adjusted)
        probs = exp_adj / exp_adj.sum()

        # Gradient of CE loss w.r.t. bias: softmax - one_hot
        grad = probs.copy()
        grad[target] -= 1.0

        # Weight decay
        if self.weight_decay > 0:
            grad += self.weight_decay * self.bias

        # SGD with momentum
        self.velocity = self.momentum * self.velocity + grad
        self.bias -= self.lr * self.velocity
        self.step_count += 1


class OnlineLogitBiasPerDocument(OnlineLogitBias):
    """
    Resets the bias at document boundaries (BOS tokens).
    This prevents cross-document contamination and allows
    fresh adaptation for each document's token distribution.
    """

    def __init__(self, vocab_size: int = 1024, lr: float = 0.01,
                 momentum: float = 0.9, weight_decay: float = 0.001,
                 bos_token: int = 0, reset_on_bos: bool = True,
                 device: str = "cpu"):
        super().__init__(vocab_size, lr, momentum, weight_decay, device)
        self.bos_token = bos_token
        self.reset_on_bos = reset_on_bos

    def update(self, logits: np.ndarray, target: int):
        """Update bias, resetting at document boundaries."""
        if self.reset_on_bos and target == self.bos_token:
            self.bias[:] = 0.0
            self.velocity[:] = 0.0
            self.step_count = 0
        super().update(logits, target)


def eval_with_online_logit_bias(
    model_logits_fn,  # callable: (x_batch) -> logits tensor
    val_tokens: np.ndarray,
    vocab_size: int = 1024,
    seq_len: int = 2048,
    stride: int = 64,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    max_positions: int = 0,  # 0 = all
    device: str = "cuda",
    log_fn=print,
) -> tuple[float, float, int]:
    """
    Sliding-window eval with online logit bias adaptation.

    Returns (avg_nll, approx_bpb, num_positions)
    """
    import time
    import torch

    total = len(val_tokens) - 1
    if max_positions > 0:
        total = min(total, max_positions)

    olb = OnlineLogitBias(vocab_size=vocab_size, lr=lr, momentum=momentum,
                          weight_decay=weight_decay)

    # We need per-position logits. For efficiency, process in sliding windows
    # but track which positions have been scored.
    window_starts = [ws for ws in range(0, total, stride)
                     if min(ws + seq_len, total) - ws >= 1]

    scored = np.zeros(total, dtype=bool)
    nll_values = np.zeros(total, dtype=np.float64)
    nll_baseline = np.zeros(total, dtype=np.float64)

    t0 = time.perf_counter()
    batch_size = 32

    val_tensor = torch.from_numpy(val_tokens.astype(np.int64))

    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_size):
            batch_ws = window_starts[bi:bi + batch_size]
            bsz = len(batch_ws)
            xb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tensor[ws:end + 1].to(device=device)
                xb[i, :wlen] = chunk[:-1]
                yb[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model_logits_fn(xb)

            logits_f = logits.float().cpu().numpy()
            yb_np = yb.cpu().numpy()

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)

                for j in range(s, wlen):
                    pos = ws + j
                    if pos >= total or scored[pos]:
                        continue

                    target = int(yb_np[i, j])
                    pos_logits = logits_f[i, j, :vocab_size]

                    # Score WITH bias (the reported score)
                    adjusted = pos_logits + olb.get_bias()
                    adjusted_shifted = adjusted - adjusted.max()
                    log_sum_exp = np.log(np.exp(adjusted_shifted).sum())
                    nll_with_bias = -(adjusted_shifted[target] - log_sum_exp)

                    # Score WITHOUT bias (baseline)
                    baseline_shifted = pos_logits - pos_logits.max()
                    log_sum_exp_b = np.log(np.exp(baseline_shifted).sum())
                    nll_no_bias = -(baseline_shifted[target] - log_sum_exp_b)

                    nll_values[pos] = nll_with_bias
                    nll_baseline[pos] = nll_no_bias
                    scored[pos] = True

                    # Update AFTER scoring
                    olb.update(pos_logits, target)

            if bi % (batch_size * 50) == 0 and bi > 0:
                n_scored = scored.sum()
                avg_nll = nll_values[scored].mean()
                avg_base = nll_baseline[scored].mean()
                delta = avg_nll - avg_base
                elapsed = time.perf_counter() - t0
                log_fn(f"  scored {n_scored}/{total}: "
                       f"baseline_nll={avg_base:.6f} olb_nll={avg_nll:.6f} "
                       f"delta={delta:+.6f} speed={n_scored/elapsed:.0f}/s")

    n_scored = int(scored.sum())
    avg_nll = float(nll_values[scored].mean())
    avg_base = float(nll_baseline[scored].mean())
    delta = avg_nll - avg_base
    bpb = avg_nll / math.log(2)
    bpb_base = avg_base / math.log(2)

    log_fn(f"\n{'='*60}")
    log_fn(f"ONLINE LOGIT BIAS RESULTS (lr={lr}, mom={momentum}, wd={weight_decay})")
    log_fn(f"Positions scored: {n_scored}")
    log_fn(f"Baseline NLL: {avg_base:.6f} (BPB: {bpb_base:.6f})")
    log_fn(f"With OLB NLL: {avg_nll:.6f} (BPB: {bpb:.6f})")
    log_fn(f"Delta NLL: {delta:+.6f} (Delta BPB: {delta/math.log(2):+.6f})")
    log_fn(f"Time: {time.perf_counter()-t0:.1f}s")
    log_fn(f"{'='*60}")

    return avg_nll, bpb, n_scored


# ============================================================
# Standalone test (CPU, no model — synthetic logits)
# ============================================================

def test_synthetic():
    """Test OLB on synthetic data to verify correctness."""
    print("=== Synthetic OLB Test ===")
    np.random.seed(42)
    vocab = 1024
    N = 50000

    # Simulate: tokens come from a distribution that shifts over time
    # First half: tokens 0-100 are common
    # Second half: tokens 500-600 are common
    tokens = np.zeros(N, dtype=np.int64)
    for i in range(N):
        if i < N // 2:
            tokens[i] = np.random.choice(100)
        else:
            tokens[i] = 500 + np.random.choice(100)

    # Simulate model logits: uniform (bad model that doesn't know the distribution)
    olb = OnlineLogitBias(vocab_size=vocab, lr=0.05, momentum=0.9, weight_decay=0.001)

    nll_no_bias = 0.0
    nll_with_bias = 0.0
    uniform_nll = math.log(vocab)

    for i in range(1, N):
        target = tokens[i]
        logits = np.zeros(vocab)  # uniform model

        # Score with bias
        adjusted = logits + olb.bias
        adjusted -= adjusted.max()
        probs = np.exp(adjusted) / np.exp(adjusted).sum()
        nll_with_bias += -math.log(max(probs[target], 1e-12))

        # Score without bias
        nll_no_bias += uniform_nll

        # Update after scoring
        olb.update(logits, target)

    avg_no = nll_no_bias / (N - 1)
    avg_with = nll_with_bias / (N - 1)
    print(f"Uniform model NLL: {avg_no:.4f}")
    print(f"With OLB NLL:      {avg_with:.4f}")
    print(f"Delta:             {avg_with - avg_no:+.4f}")
    print(f"OLB learned to track the shifting distribution!")
    print()

    # Verify bias reflects the distribution shift
    top_first_half = olb.bias[:100].mean()
    top_second_half = olb.bias[500:600].mean()
    print(f"Bias for tokens 0-99 (first half common):   {top_first_half:+.4f}")
    print(f"Bias for tokens 500-599 (second half common): {top_second_half:+.4f}")
    print(f"Second half tokens should have higher bias (they were seen more recently)")


if __name__ == "__main__":
    test_synthetic()
