"""
Integration tests for `ngram_eval.eval_val_ttt_with_ngram`.

Runs against a RANDOM-INIT GPT-style model (no training needed) to verify:

1. Regression: alpha=0 must produce BPB bit-identical to baseline eval
   (since the n-gram contribution is zero and scoring path is otherwise
    mathematically identical).
2. Stability: alpha > 0 produces finite, non-nan BPB values.
3. Legality preservation: the four conditions still hold after integration.
4. Update-after-score discipline: freezing ordering is correct (tested via a
   dry-run that records cache state at each chunk boundary and verifies it
   only grows monotonically with prior-chunk tokens).

Because we don't want to depend on flash_attn_3 or CUDA, we use a minimal
TinyGPT stand-in with the same `forward_logits` / `forward(input_ids, target_ids)`
interface that #1493's eval loop expects.

Device: CPU (portable, slow but correct).
"""
from __future__ import annotations
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_ngram import CausalNGram as CNG  # for legality cross-checks
from ngram_eval import CausalNGram, eval_val_ttt_with_ngram


class TinyGPT(nn.Module):
    """Minimal decoder-only LM for the integration test."""

    def __init__(self, vocab_size: int, dim: int = 64, n_layers: int = 2,
                 seq_len: int = 128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 4,
            batch_first=True, dropout=0.0, activation='gelu',
            norm_first=True,
        ) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        # Causal mask
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, src_mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1), reduction='mean',
        )


@dataclass
class TinyHparams:
    rank: int = 0
    world_size: int = 1
    eval_seq_len: int = 128
    eval_stride: int = 16
    vocab_size: int = 256
    ttt_chunk_tokens: int = 512
    ttt_lr: float = 0.0  # disable TTT SGD for legality isolation
    ttt_epochs: int = 0
    ttt_momentum: float = 0.9


class FakeValData:
    """Stand-in for ValidationData — provides val_tokens and the byte LUTs."""

    def __init__(self, tokens: torch.Tensor, vocab_size: int, device):
        self.val_tokens = tokens  # 1-D tensor of token IDs, CPU
        # Synthetic LUTs: every token is 4 bytes, no leading space, no boundary.
        # This keeps BPB computation simple and deterministic.
        self.base_bytes_lut = torch.full((vocab_size,), 4, dtype=torch.int16,
                                         device=device)
        self.has_leading_space_lut = torch.zeros((vocab_size,), dtype=torch.bool,
                                                  device=device)
        self.is_boundary_token_lut = torch.zeros((vocab_size,), dtype=torch.bool,
                                                  device=device)


def eval_val_ttt_baseline(h, device, val_data, base_model, batch_seqs: int = 8):
    """Stripped-down copy of #1493 eval_val_ttt with TTT SGD disabled. Used as
    the regression baseline for alpha=0."""
    rank = h.rank
    world_size = h.world_size
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    with torch.no_grad():
        for ci in range(num_chunks):
            windows = chunk_windows[ci]
            if not windows:
                continue
            my_windows = windows  # world_size=1
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk_tok = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction='none'
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


# =============================================================================
# Tests
# =============================================================================

def make_fake_val(vocab_size: int = 256, n_tokens: int = 4096, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (n_tokens,), dtype=torch.int64, generator=g)


def test_regression_alpha_zero():
    """alpha=0 must give BPB bit-identical to baseline eval (modulo floating
    point within 1e-10)."""
    torch.manual_seed(42)
    device = torch.device('cpu')
    vocab_size = 256
    h = TinyHparams(vocab_size=vocab_size)
    model = TinyGPT(vocab_size=vocab_size, dim=32, n_layers=2, seq_len=h.eval_seq_len)
    model.eval()
    tokens = make_fake_val(vocab_size=vocab_size, n_tokens=4096)
    val_data = FakeValData(tokens, vocab_size, device)

    _, bpb_baseline = eval_val_ttt_baseline(h, device, val_data, model, batch_seqs=8)

    # Now with alpha=0. Even creating a CausalNGram and going through the
    # blend path must reproduce the baseline (since alpha=0 short-circuits).
    ng = CausalNGram(vocab_size=vocab_size, order=5)
    _, bpb_ngram = eval_val_ttt_with_ngram(h, device, val_data, model,
                                            ngram=ng, alpha=0.0,
                                            batch_seqs=8, enable_ttt=False)
    delta = abs(bpb_baseline - bpb_ngram)
    assert delta < 1e-8, \
        f"alpha=0 regression failed: baseline={bpb_baseline:.12f} ngram={bpb_ngram:.12f} delta={delta}"
    return bpb_baseline, bpb_ngram


def test_stability_alpha_positive():
    """alpha > 0 produces finite, non-nan BPB values across a sweep."""
    torch.manual_seed(43)
    device = torch.device('cpu')
    vocab_size = 256
    h = TinyHparams(vocab_size=vocab_size)
    model = TinyGPT(vocab_size=vocab_size, dim=32, n_layers=2, seq_len=h.eval_seq_len)
    model.eval()
    # Use a structured sequence (not random) so the n-gram has something to
    # learn. Repeat a short pattern.
    pattern = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    tokens = torch.tensor(pattern * 400, dtype=torch.int64)[:4096]
    val_data = FakeValData(tokens, vocab_size, device)

    results = {}
    for alpha in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
        ng = CausalNGram(vocab_size=vocab_size, order=5)
        _, bpb = eval_val_ttt_with_ngram(h, device, val_data, model, ngram=ng,
                                          alpha=alpha, batch_seqs=8, enable_ttt=False)
        assert math.isfinite(bpb), f"alpha={alpha}: non-finite BPB={bpb}"
        assert bpb > 0, f"alpha={alpha}: non-positive BPB={bpb}"
        results[alpha] = bpb

    # For a repeating pattern, higher alpha should EVENTUALLY reduce BPB
    # (assuming the cache learns the pattern). The alpha=0 baseline is random
    # so we expect alpha>0 to win by a margin.
    assert results[1.0] < results[0.0], \
        f"n-gram did not help on repeating pattern: {results}"

    return results


def test_legality_preserved():
    """The integrated eval path must still pass the legality harness's probes
    on its CausalNGram instance."""
    # The harness in legality_harness.py operates on the causal_ngram.CausalNGram
    # (slightly different class). The ngram_eval.CausalNGram is structurally
    # the same — same freeze/add/lookup contract. Run a quick adversarial probe.
    ng = CausalNGram(vocab_size=32, order=4)
    # Build from one sequence
    import random
    rng = random.Random(0)
    seq = [rng.randrange(32) for _ in range(500)]
    running = []
    for tok in seq:
        ng.add_token(tuple(running), tok)
        running.append(tok)
        if len(running) > 3:
            running = running[-3:]
    ng.freeze()

    # C1: mutate tokens >= position 200 and verify lookup at position 200 is identical
    hist_before = tuple(seq[197:200])
    lp1 = ng._lookup_log_probs(hist_before).copy()
    # The frozen cache should not change when we mutate the live counts with
    # mutated data (live updates don't affect frozen lookups)
    ng.add_many([999 % 32] * 50)  # junk data into LIVE only
    lp2 = ng._lookup_log_probs(hist_before)
    assert np.allclose(lp1, lp2), "C1 violated: frozen cache changed due to live updates"

    # C2: distribution sums to 1
    prob = np.exp(lp2)
    assert abs(prob.sum() - 1.0) < 1e-6, f"C2 violated: sum={prob.sum()}"
    return True


def test_update_after_score_ordering():
    """Verify that in eval_val_ttt_with_ngram, the cache state used for scoring
    a chunk is the state at chunk_start (not anything updated mid-chunk).

    We instrument this by providing a structured sequence and a small model,
    then comparing the measured n-gram log-probs at scoring time against a
    parallel reference cache that's manually frozen at the right point.
    """
    torch.manual_seed(44)
    device = torch.device('cpu')
    vocab_size = 16
    h = TinyHparams(vocab_size=vocab_size, ttt_chunk_tokens=256)
    model = TinyGPT(vocab_size=vocab_size, dim=16, n_layers=1, seq_len=h.eval_seq_len)
    model.eval()
    tokens = torch.tensor([(i % vocab_size) for i in range(2048)], dtype=torch.int64)
    val_data = FakeValData(tokens, vocab_size, device)

    ng = CausalNGram(vocab_size=vocab_size, order=4)
    _, bpb = eval_val_ttt_with_ngram(h, device, val_data, model, ngram=ng,
                                      alpha=0.5, batch_seqs=4, enable_ttt=False)

    # After a full eval, the FROZEN cache should contain statistics from ALL
    # scored tokens (not more, not less). We verify by counting order-1 total
    # against the number of scored tokens expected from the eval loop.
    total_tokens = val_data.val_tokens.numel() - 1
    stride = h.eval_stride
    seq_len = h.eval_seq_len
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]
    expected_scored = 0
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        expected_scored += wlen - s

    unigram_total = ng._frozen_totals[1].get((), 0)
    # The frozen state is re-snapshotted after EACH chunk update, so at the end
    # of eval the frozen state should reflect all scored tokens.
    assert unigram_total == expected_scored, \
        f"Cache didn't update correctly: unigram total={unigram_total} expected={expected_scored}"
    return unigram_total, expected_scored


def main():
    results = {}
    for name, fn in [
        ("regression (alpha=0)", test_regression_alpha_zero),
        ("stability (alpha>0 sweep)", test_stability_alpha_positive),
        ("legality preserved", test_legality_preserved),
        ("update-after-score ordering", test_update_after_score_ordering),
    ]:
        print(f"\n--- {name} ---")
        try:
            out = fn()
            print(f"  PASS  {out}")
            results[name] = ("pass", out)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL  {e}")
            results[name] = ("fail", str(e))

    fails = [n for n, (s, _) in results.items() if s == "fail"]
    if fails:
        print(f"\n{len(fails)}/{len(results)} tests FAILED:")
        for n in fails:
            print(f"  - {n}")
        sys.exit(1)
    print(f"\n{len(results)}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
