"""KV-cache chaining design + prototype helpers for eval-time context extension.

THE INSIGHT (Himanshu, 2026-04-17):
The competition has asymmetric memory constraints:
- Weights: 16 MB cap (artifact)
- Eval working memory: 80 GB × 8 H100 = 640 GB, 10-min wallclock = effectively unlimited

This means eval-time KV cache and context length should be MAXIMIZED, not minimized.
Almost no current PR does this systematically.

CURRENT EVAL FLOW (PR #1493):
1. Sliding window: process val sequence in 2048-token windows, stride 64 → score 64 new
   tokens per window with 1984 tokens of left context
2. TTT: chunks of 32768 tokens; for each chunk, score then SGD update on whole chunk

Both use seq_len=2048 (the training context). Each window's KV is computed FRESH.

THE OPPORTUNITY:
Three axes of expansion, all legal under C1-C4:

### Axis 1: Longer effective context per forward pass
Train at seq_len=2048, eval at seq_len=4096 or 8192. RoPE positions beyond 2048
are out-of-distribution but YaRN/NTK scaling extends them gracefully.

Implementation:
- At eval: rebuild Rotary with extended max position
- Optionally adjust RoPE base (NTK-aware)
- Run forward at seq_len=4096

Risk: model behavior in extended positions might degrade overall quality.

### Axis 2: KV cache chaining across windows
Currently each sliding-window forward pass starts fresh. Instead, keep KV from
PREVIOUS windows and prepend to current attention's K/V.

For target token at position t in val sequence, the model effectively sees
K/V from all previous windows (millions of tokens of context, bounded by
HBM memory).

Implementation: use FlashAttn's windowed mode with stored historical K/V.

Risk: model trained at 2048 attends at position offset; extended positions
need RoPE scaling.

### Axis 3: More TTT epochs / more chunks
We already use TTT_EPOCHS=3 (~370s eval). With sliding eval at ~120s, we
have ~110s of slack within 600s eval budget. Could push to TTT_EPOCHS=4 or
EVAL_STRIDE=32 (saturated per our test) or both.

This is the simplest expansion — no model changes.

### LEGALITY (C1-C4):
- C1 causality: KV chain only adds PRIOR context → fine
- C2 normalized: still softmax over vocab → fine
- C3 score-before-update: depends on TTT structure, must keep score-first
- C4 single pass: each val token still scored once → fine

CRITICAL: KV chaining must NEVER use future tokens for past predictions.
The chain accumulates ONLY past KV.
"""

import torch
import math


# ============ AXIS 1: RoPE extension (YaRN-style) ============
def yarn_rope_scaling(rope_base: float, train_seq_len: int, eval_seq_len: int,
                     extrapolation_factor: float = 1.0):
    """YaRN-style RoPE base scaling for longer context at eval.

    Standard: theta_i = base^(-2i/d). For training context L_train.
    Extended: scale theta to handle eval context L_eval > L_train.

    Returns adjusted rope_base for eval-time Rotary instantiation.
    """
    if eval_seq_len <= train_seq_len:
        return rope_base
    scale = eval_seq_len / train_seq_len
    # NTK-aware: scale base by scale^(d/(d-2)). For our d=16 partial RoPE:
    # NTK base = base * scale^(d/(d-2)) ≈ base * scale^1.143
    new_base = rope_base * (scale ** (16 / 14))  # for rope_dims=16
    return new_base


def test_yarn_rope():
    base = 10000.0
    extended = yarn_rope_scaling(base, 2048, 4096)
    print(f"  YaRN: base 2048→4096 ext: {base:.1f} → {extended:.1f} (ratio {extended/base:.2f})")


# ============ AXIS 2: KV cache chain ============
class KVChain:
    """Maintain K/V cache across sliding-window forward passes.

    Usage:
      chain = KVChain(num_heads=4, head_dim=64, max_history=8192)
      for window_id, (x_window, target_positions) in enumerate(windows):
          k, v = model.compute_kv(x_window)  # (B, T, H, D)
          y_logits = model.attend_with_history(x_window, chain.get_history())
          chain.append(k, v)
          # score target_positions ...
    """
    def __init__(self, num_kv_heads: int, head_dim: int, max_history: int = 8192):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_history = max_history
        self.k_history = None  # (1, T_total, H, D)
        self.v_history = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """Add new K/V to history, trim to max_history."""
        # k_new: (B=1, T, H, D)
        if self.k_history is None:
            self.k_history = k_new
            self.v_history = v_new
        else:
            self.k_history = torch.cat([self.k_history, k_new], dim=1)
            self.v_history = torch.cat([self.v_history, v_new], dim=1)
        # Trim
        if self.k_history.shape[1] > self.max_history:
            excess = self.k_history.shape[1] - self.max_history
            self.k_history = self.k_history[:, excess:]
            self.v_history = self.v_history[:, excess:]

    def get_history(self):
        return self.k_history, self.v_history

    def reset(self):
        self.k_history = None
        self.v_history = None


def test_kv_chain():
    chain = KVChain(num_kv_heads=4, head_dim=64, max_history=512)
    # Add 5 windows of 256 tokens each = 1280 total → trims to 512
    for i in range(5):
        k = torch.randn(1, 256, 4, 64)
        v = torch.randn(1, 256, 4, 64)
        chain.append(k, v)
    k_hist, v_hist = chain.get_history()
    assert k_hist.shape == (1, 512, 4, 64), f"expected (1,512,4,64), got {k_hist.shape}"
    print(f"  ✓ KVChain trimmed correctly: history shape {k_hist.shape}")


# ============ AXIS 3: TTT epoch budget calculator ============
def ttt_budget_calc(
    n_chunks: int,
    seqs_per_chunk: int,
    n_epochs: int,
    seconds_per_seq_forward: float = 0.012,
    seconds_per_seq_backward: float = 0.024,
    sliding_window_seconds: float = 120.0,
    pre_quant_seconds: float = 8.0,
    quant_eval_seconds: float = 25.0,
    eval_budget: float = 600.0,
):
    """How many TTT epochs fit in eval budget?"""
    sec_per_chunk = seqs_per_chunk * (seconds_per_seq_forward + n_epochs * seconds_per_seq_backward)
    total = pre_quant_seconds + quant_eval_seconds + sliding_window_seconds + n_chunks * sec_per_chunk
    return total, eval_budget - total


def test_ttt_budget():
    for epochs in [3, 4, 5, 7, 10]:
        total, slack = ttt_budget_calc(n_chunks=1238, seqs_per_chunk=2, n_epochs=epochs)
        ok = "OK" if slack > 0 else "OVER"
        print(f"  TTT_EPOCHS={epochs}: total={total:.0f}s, slack={slack:.0f}s [{ok}]")


# ============ Causality check ============
def test_causality_preserved():
    """KVChain must NOT include future tokens in past attention.

    Sanity: when we score position t, the K/V chain contains positions 0..t-1 only.
    Each window appends AFTER its target tokens are scored.
    """
    # Simulated:
    history_positions = []
    for window_start in [0, 64, 128, 192, 256]:
        # In this window, score positions [window_start..window_start+stride]
        # using K/V from history (all positions BEFORE window_start)
        target_positions = list(range(window_start, window_start + 64))
        # Verify history contains only positions < window_start
        assert all(p < window_start for p in history_positions), \
            f"causality violated: history has {[p for p in history_positions if p >= window_start]}"
        # AFTER scoring, append this window's K/V to history
        history_positions.extend(range(window_start, window_start + 64))
    print(f"  ✓ Causality preserved across {len(history_positions)} positions")


# ============ Implementation roadmap ============
IMPLEMENTATION_PLAN = """
TOMORROW'S KV-CACHE EXPERIMENT (after recording at int7/int7 + QAT v3):

PHASE 1 — Cheap eval-time compute (no model changes):
  a. TTT_EPOCHS=4 or 5 (use slack in eval budget)
     Cost: $4 salvage_eval. Expected: -0.001 to -0.003 BPB.
  b. Combined with stride=32 if budget allows
     Cost: included. Expected: maybe -0.001 BPB.

PHASE 2 — Longer context with YaRN RoPE extension:
  a. Modify Rotary class to accept eval_max_position
  b. Use yarn_rope_scaling() to compute extended base
  c. Eval at seq_len=4096 (2x training context)
     Cost: $4-7 salvage_eval. Expected: -0.005 to -0.015 BPB if model
     handles extended positions OK; potentially worse if not.

PHASE 3 — Full KV chain across sliding windows:
  a. Modify eval_val_sliding to maintain KVChain across windows
  b. Each window's attention uses (current K/V) + (chain history)
  c. After scoring, append window K/V to chain
     Cost: $7+ (more complex code, careful debugging). Expected: -0.005 to
     -0.020 BPB depending on how much extra context helps.

LEGALITY CHECKS BEFORE EACH PHASE:
- C1: every K/V in attention computation must be from positions ≤ current pos
- C2: softmax remains over full vocab (no truncation)
- C3: TTT score-first ordering preserved (already in pipeline)
- C4: each val token scored exactly once (still using stride to define scoring boundaries)
"""


if __name__ == "__main__":
    print("=== KV-cache chaining + RoPE extension prototype ===\n")
    print("[1] YaRN RoPE scaling:")
    test_yarn_rope()
    print("\n[2] KV cache chain:")
    test_kv_chain()
    print("\n[3] TTT epoch budget:")
    test_ttt_budget()
    print("\n[4] Causality check:")
    test_causality_preserved()
    print("\n=== Implementation roadmap ===")
    print(IMPLEMENTATION_PLAN)
