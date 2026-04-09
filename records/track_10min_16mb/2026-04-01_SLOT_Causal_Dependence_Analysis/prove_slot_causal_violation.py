"""
Proof: SLOT violates Rule 1 (Strict Causal Dependence) — Issue #1017
====================================================================

Rule 1: "at position t, only use tokens x_1..x_{t-1}"

SLOT optimizes a delta vector (and optionally a logit bias) using the target
tokens y = x_{t+1}, then scores those same targets with the optimized
parameters.  The gradient from target x_{t+1} flows into delta, which then
influences the prediction at position t.

This script proves the violation empirically:
  - Test A: Changing a future target changes NLL at earlier positions.
  - Test B: The model predicts x_{t+1} better when x_{t+1} is in the
            optimization targets ("self-prediction advantage").

Both tests compare against a no-SLOT baseline where the model IS causal.

No GPU required.  No flash_attn required.  Works on CPU/MPS.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal causal LM (standard attention, no external deps)
# ---------------------------------------------------------------------------

class MinimalCausalLM(nn.Module):
    """Tiny GPT for demonstration.  Architecture doesn't matter — the causal
    violation lives in the SLOT *procedure*, not the model."""

    def __init__(self, vocab_size: int = 1024, dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
                batch_first=True, norm_first=True, dropout=0.0,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.logit_softcap = 30.0

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len,
                                                              device=input_ids.device)
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.final_norm(x)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


# ---------------------------------------------------------------------------
# SLOT implementations (both our shared-delta and PR #1229 per-sample)
# ---------------------------------------------------------------------------

def run_slot(model, x, y, *, slot_steps, slot_lr, mask,
             per_sample_delta: bool, use_logit_bias: bool,
             score_targets=None):
    """Run SLOT and return per-position NLL [1, seq_len].

    per_sample_delta=False, use_logit_bias=False  =>  our PR #1209 style
    per_sample_delta=True,  use_logit_bias=True   =>  PR #1229 style

    If score_targets is given, optimization uses `y` but final NLL is
    computed against `score_targets`.  This lets us measure: "how well
    does SLOT predict token X when it optimized toward token X vs toward
    some other token Y?"
    """
    model.eval()
    bsz, seq_len = x.shape
    dim = model.tok_emb.weight.size(1)
    vocab_size = model.tok_emb.weight.size(0)
    proj_w = model.tok_emb.weight.detach().float()
    sc = model.logit_softcap

    with torch.no_grad():
        hidden = model.forward_hidden(x)
    hidden_f = hidden.detach().float()

    # Delta shape: [bsz,1,dim] or [1,1,dim]
    delta_shape = (bsz, 1, dim) if per_sample_delta else (1, 1, dim)
    delta = torch.zeros(*delta_shape, dtype=torch.float32, requires_grad=True)

    params = [delta]
    if use_logit_bias:
        logit_bias = torch.zeros(bsz, 1, vocab_size, dtype=torch.float32,
                                 requires_grad=True)
        params.append(logit_bias)
    else:
        logit_bias = None

    opt = torch.optim.AdamW(params, lr=slot_lr)
    targets_flat = y.reshape(-1)
    valid = mask.sum()

    for step in range(slot_steps):
        lr = slot_lr * 0.5 * (1 + math.cos(math.pi * step / slot_steps))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad()
        h = hidden_f + delta
        logits = F.linear(h, proj_w)
        if logit_bias is not None:
            logits = logits + logit_bias
        logits = sc * torch.tanh(logits / sc)
        nll = F.cross_entropy(logits.reshape(-1, vocab_size), targets_flat,
                              reduction="none").reshape(bsz, seq_len)
        loss = (nll * mask).sum() / valid
        loss.backward()
        opt.step()

    # Score against score_targets if given, else same as optimization targets
    final_targets = score_targets if score_targets is not None else y
    with torch.no_grad():
        h = hidden_f + delta
        logits = F.linear(h, proj_w)
        if logit_bias is not None:
            logits = logits + logit_bias
        logits = sc * torch.tanh(logits / sc)
        nll = F.cross_entropy(logits.reshape(-1, vocab_size),
                              final_targets.reshape(-1),
                              reduction="none").reshape(bsz, seq_len)
    return nll.detach()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_baseline_is_causal(model, x, y, vocab_size, scored_start, seq_len):
    """Sanity check: without SLOT the model is perfectly causal."""
    print("=" * 70)
    print("SANITY CHECK: Without SLOT, scoring is pointwise in targets")
    print("=" * 70)
    print("  (Without SLOT, NLL at position i depends only on logits[i] and")
    print("   y[i].  Changing y[j] for j != i has zero effect.  This is the")
    print("   null hypothesis that SLOT violates.)")

    model.eval()
    with torch.no_grad():
        logits = model.compute_logits(model.forward_hidden(x))
        nll_orig = F.cross_entropy(logits.reshape(-1, vocab_size),
                                   y.reshape(-1), reduction="none"
                                   ).reshape(1, seq_len)

    y_mod = y.clone()
    y_mod[0, -1] = (y[0, -1] + 1) % vocab_size

    with torch.no_grad():
        nll_mod = F.cross_entropy(logits.reshape(-1, vocab_size),
                                  y_mod.reshape(-1), reduction="none"
                                  ).reshape(1, seq_len)

    diff = (nll_mod - nll_orig).abs()
    max_change_elsewhere = diff[0, :-1].max().item()
    print(f"  Flipped target at position {seq_len - 1}")
    print(f"  Max NLL change at positions 0..{seq_len - 2}: {max_change_elsewhere:.2e}")
    assert max_change_elsewhere == 0.0
    print(f"  PASS\n")


def test_future_token_sensitivity(model, x, y, vocab_size, scored_start,
                                  seq_len, mask, label, **slot_kwargs):
    """Flip one future target, show NLL changes at earlier positions."""
    print("-" * 70)
    print(f"TEST A [{label}]: Future-token sensitivity")
    print("-" * 70)

    nll_orig = run_slot(model, x, y, mask=mask, **slot_kwargs)

    # Flip the LAST scored target
    flip_pos = seq_len - 1
    y_mod = y.clone()
    y_mod[0, flip_pos] = (y[0, flip_pos] + 1) % vocab_size
    nll_mod = run_slot(model, x, y_mod, mask=mask, **slot_kwargs)

    diff = (nll_mod - nll_orig).abs()
    # Look at ALL scored positions except the one we flipped
    other_scored = diff[0, scored_start:flip_pos]

    print(f"  Flipped target at position {flip_pos}")
    print(f"  NLL changes at OTHER scored positions ({scored_start}..{flip_pos - 1}):")
    print(f"    Max:  {other_scored.max().item():.6f}")
    print(f"    Mean: {other_scored.mean().item():.6f}")
    print(f"    Min:  {other_scored.min().item():.6f}")

    n_violated = (other_scored > 1e-4).sum().item()
    n_total = other_scored.numel()
    print(f"  Positions affected (delta > 1e-4): {n_violated}/{n_total}")

    if n_violated > 0:
        print(f"  >>> VIOLATION: changing future token x_{{{flip_pos + 1}}} affected "
              f"predictions at {n_violated} earlier positions <<<")
    else:
        print(f"  No violation detected (increase slot_steps?)")
    print()
    return other_scored.max().item()


def test_self_prediction(model, x, y, vocab_size, scored_start, seq_len,
                         mask, label, n_alternatives=64, **slot_kwargs):
    """Show P(x_{t+1}) is better when x_{t+1} is in the optimization targets.

    Key: we always SCORE the same token (the original target).  We only change
    what token SLOT optimizes toward at that position.  If the score differs,
    then the prediction of x_{t+1} depends on whether x_{t+1} was in the
    optimization — a direct Rule 1 violation.
    """
    print("-" * 70)
    print(f"TEST B [{label}]: Self-prediction advantage")
    print("-" * 70)

    probe = scored_start + (seq_len - scored_start) // 2
    original_target = y[0, probe].item()

    # Case 1: optimize toward the CORRECT target, score the correct target
    nll_correct = run_slot(model, x, y, mask=mask,
                           score_targets=y, **slot_kwargs)[0, probe].item()

    # Case 2: optimize toward WRONG targets, still score the correct target
    nlls_wrong = []
    rng = torch.Generator().manual_seed(999)
    alts = torch.randint(0, vocab_size, (n_alternatives * 2,), generator=rng)
    alts = alts[alts != original_target][:n_alternatives]

    for alt in alts:
        y_alt = y.clone()
        y_alt[0, probe] = alt.item()
        # Optimize with wrong target but SCORE against original target
        nll_alt = run_slot(model, x, y_alt, mask=mask,
                           score_targets=y, **slot_kwargs)[0, probe].item()
        nlls_wrong.append(nll_alt)

    mean_wrong = sum(nlls_wrong) / len(nlls_wrong)
    advantage = mean_wrong - nll_correct

    print(f"  Position {probe}: always scoring token {original_target}")
    print(f"  NLL(token {original_target}) when SLOT optimizes toward {original_target}:  {nll_correct:.4f}")
    print(f"  NLL(token {original_target}) when SLOT optimizes toward wrong token: {mean_wrong:.4f}  (mean/{len(nlls_wrong)})")
    print(f"  Self-prediction advantage (positive = answer leaks):  {advantage:+.4f}")

    if advantage > 0.001:
        print(f"  >>> VIOLATION: P(x_{{t+1}}) is LOWER (better) when x_{{t+1}} is in  <<<")
        print(f"      the SLOT optimization targets. The answer leaks through delta.")
    elif advantage < -0.001:
        print(f"  Note: advantage is negative — optimization toward correct token")
        print(f"  actually hurts at this position (pulled toward sum of all targets).")
        print(f"  The causal violation is still proven by Tests A and C.")
    print()
    return advantage


# ---------------------------------------------------------------------------
# Systematic perturbation: flip each scored target, measure cross-talk
# ---------------------------------------------------------------------------

def test_systematic(model, x, y, vocab_size, scored_start, seq_len, mask,
                    label, n_probes=16, **slot_kwargs):
    """Flip each of n_probes targets individually; measure effect on others."""
    print("-" * 70)
    print(f"TEST C [{label}]: Systematic cross-position leakage")
    print("-" * 70)

    nll_base = run_slot(model, x, y, mask=mask, **slot_kwargs)

    violations = 0
    checks = 0
    max_leak = 0.0
    sum_leak = 0.0

    for k in range(n_probes):
        flip_pos = scored_start + k
        y_flip = y.clone()
        y_flip[0, flip_pos] = (y[0, flip_pos] + 1) % vocab_size
        nll_flip = run_slot(model, x, y_flip, mask=mask, **slot_kwargs)

        for j in range(n_probes):
            check_pos = scored_start + j
            if check_pos == flip_pos:
                continue
            checks += 1
            leak = abs(nll_flip[0, check_pos].item() - nll_base[0, check_pos].item())
            if leak > 1e-4:
                violations += 1
            max_leak = max(max_leak, leak)
            sum_leak += leak

    pct = 100 * violations / max(1, checks)
    print(f"  Flipped {n_probes} individual targets in scored range")
    print(f"  Checked {checks} cross-position pairs")
    print(f"  Violations (|delta NLL| > 1e-4): {violations}/{checks} ({pct:.1f}%)")
    print(f"  Max cross-position NLL change:   {max_leak:.6f}")
    print(f"  Mean cross-position NLL change:  {sum_leak / max(1, checks):.6f}")

    if violations > 0:
        print(f"  >>> VIOLATION: {pct:.0f}% of position pairs show information leakage <<<")
    print()
    return violations, checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    vocab_size = 1024
    seq_len = 128
    stride = 64

    print()
    print("=" * 70)
    print("  SLOT VIOLATES RULE 1 (STRICT CAUSAL DEPENDENCE)")
    print("  Issue #1017 — Empirical proof")
    print("=" * 70)
    print()
    print("Rule 1: 'at position t, only use tokens x_1..x_{t-1}'")
    print()
    print("SLOT optimizes delta/logit_bias using target tokens, then scores")
    print("those same targets.  If the prediction at position t changes when")
    print("we modify a FUTURE target x_{t+k}, Rule 1 is violated.")
    print()

    model = MinimalCausalLM(vocab_size=vocab_size, dim=128, num_layers=2)
    tokens = torch.randint(0, vocab_size, (1, seq_len + 1))
    x = tokens[:, :-1]
    y = tokens[:, 1:]

    # Scored-position mask: last `stride` positions (matches eval protocol)
    mask = torch.zeros(1, seq_len)
    mask[0, seq_len - stride:] = 1.0
    scored_start = seq_len - stride

    # -- Baseline --
    test_baseline_is_causal(model, x, y, vocab_size, scored_start, seq_len)

    # -- Shared delta (our PR #1209) --
    shared_kwargs = dict(slot_steps=8, slot_lr=0.005,
                         per_sample_delta=False, use_logit_bias=False)

    max_a_shared = test_future_token_sensitivity(
        model, x, y, vocab_size, scored_start, seq_len, mask,
        "shared delta", **shared_kwargs)

    adv_shared = test_self_prediction(
        model, x, y, vocab_size, scored_start, seq_len, mask,
        "shared delta", n_alternatives=32, **shared_kwargs)

    v_shared, c_shared = test_systematic(
        model, x, y, vocab_size, scored_start, seq_len, mask,
        "shared delta", n_probes=16, **shared_kwargs)

    # -- Per-sample delta + logit bias (PR #1229) --
    full_kwargs = dict(slot_steps=16, slot_lr=0.008,
                       per_sample_delta=True, use_logit_bias=True)

    max_a_full = test_future_token_sensitivity(
        model, x, y, vocab_size, scored_start, seq_len, mask,
        "per-sample + logit_bias", **full_kwargs)

    adv_full = test_self_prediction(
        model, x, y, vocab_size, scored_start, seq_len, mask,
        "per-sample + logit_bias", n_alternatives=32, **full_kwargs)

    v_full, c_full = test_systematic(
        model, x, y, vocab_size, scored_start, seq_len, mask,
        "per-sample + logit_bias", n_probes=16, **full_kwargs)

    # -- Summary --
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Without SLOT (baseline):  predictions are perfectly causal.")
    print("                          Changing targets has ZERO effect on NLL.")
    print()
    print("With SLOT:                predictions depend on FUTURE targets.")
    print("                          Rule 1 is violated.")
    print()
    print(f"{'Metric':<45} {'Shared':>10} {'Per-sample':>10}")
    print(f"{'─' * 45} {'─' * 10} {'─' * 10}")
    print(f"{'Max NLL change from future token flip':<45} {max_a_shared:>10.6f} {max_a_full:>10.6f}")
    print(f"{'Self-prediction advantage':<45} {adv_shared:>10.4f} {adv_full:>10.4f}")
    print(f"{'Cross-position violations':<45} {v_shared:>10d} {v_full:>10d}")
    print(f"{'Cross-position checks':<45} {c_shared:>10d} {c_full:>10d}")
    print(f"{'Violation rate':<45} {100*v_shared/max(1,c_shared):>9.1f}% {100*v_full/max(1,c_full):>9.1f}%")
    print()
    print("The violation is STRUCTURAL — it exists in the SLOT procedure itself,")
    print("regardless of model architecture, weights, or scale.")
    print()
    print("Mathematical argument (why this holds for ANY model, not just random):")
    print("  SLOT loss: L(delta, y) = sum_t CE(f(H_t + delta), y_t)")
    print("  Gradient:  dL/d(delta) = sum_t dCE/df * df/d(delta)")
    print("  The gradient explicitly depends on every y_t.  After optimization,")
    print("  delta = g(y_1,...,y_T) for some function g.  Therefore the scored")
    print("  NLL at position t = CE(f(H_t + g(y_1,...,y_T)), y_t) depends on")
    print("  ALL targets, not just y_t.  This holds whenever the gradient is")
    print("  nonzero — i.e., for any model that isn't perfectly converged.")
    print()
    print("This applies to ALL SLOT submissions:")
    print("  - PR #1209 (shared delta, 1.1064 BPB)")
    print("  - PR #1229 (per-sample + logit_bias, 0.9300 BPB)")
    print("  - Any future variant that optimizes on targets before scoring them")


if __name__ == "__main__":
    main()
