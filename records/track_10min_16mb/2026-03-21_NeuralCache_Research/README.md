# Neural Cache: Cross-Window KV Cache for Extended Context at Eval Time

**Research proposal (no record claim)** | Base model: PR #287 reproduction (1.1284 BPB) | 8xH100 SXM

## The Idea

Standard sliding window evaluation processes each window independently. A window at position 10,000 has no memory of what happened at position 5,000 — even though those tokens were already evaluated. Neural Cache fixes this by **caching K/V pairs across windows**, extending effective context from 2,048 tokens to 50K+ tokens at zero artifact cost.

```
Standard sliding window (stride=64, seq=2048):
  Window 1: [tokens 0-2047]     -> score tokens 0-2047    (context: 2048)
  Window 2: [tokens 64-2111]    -> score tokens 2048-2111  (context: 2048)
  Window 3: [tokens 128-2175]   -> score tokens 2112-2175  (context: 2048)
  ...each window is INDEPENDENT. Token 2048 cannot see token 0.

Neural Cache (stride=64, seq=2048, cache=8192):
  Window 1: [tokens 0-2047]     -> score, cache K/V for stride tokens
  Window 2: [tokens 64-2111]    -> attend to cached K/V + current window
  Window 3: [tokens 128-2175]   -> attend to growing cache + current window
  ...token 8000 can attend to token 0 through the cache. Effective context: 10K+
```

## Why This Should Work

1. **More context = better prediction.** This is proven: seq2048 > seq1024 > seq512 (PR #136: -0.014 BPB from longer context). Neural Cache extends this principle beyond the training sequence length.

2. **Flash Attention natively supports it.** When `seqlen_k > seqlen_q`, FA3 treats the extra K/V as "earlier" context — exactly the KV-cache pattern used in LLM inference. No custom kernels needed.

3. **Backward-looking only.** The cache contains K/V from already-evaluated tokens. No future information leaks. This is the same principle as backward-looking TTT (PR #267, confirmed rule-compliant) but lighter weight — no gradient computation, just cached hidden states.

4. **Zero artifact cost.** No extra parameters, no model changes. Pure eval-time technique. ~50 lines of code.

## Implementation

The core idea: modify the attention forward pass to accept and prepend cached K/V.

```python
def attn_forward_with_cache(attn_module, x, kv_cache=None, cache_seqlen=0):
    # Compute Q, K, V for current window
    q, k, v = compute_qkv(attn_module, x)

    # Apply RoPE with position offset (critical for correctness)
    cos, sin = attn_module.rotary(cache_seqlen + seqlen, device, dtype)
    q = apply_rotary_emb(q, cos[cache_seqlen:], sin[cache_seqlen:])
    k = apply_rotary_emb(k, cos[cache_seqlen:], sin[cache_seqlen:])

    # Prepend cached K/V from previous windows
    if kv_cache is not None:
        k = torch.cat([kv_cache[0], k], dim=1)  # [B, cache+seq, H, D]
        v = torch.cat([kv_cache[1], v], dim=1)

    # Flash Attention handles seqlen_k > seqlen_q natively
    y = flash_attn_func(q, k, v, causal=True)
    return y, (new_k, new_v)  # Return current K/V for future caching
```

The eval loop maintains a per-layer cache, only storing the `stride` newest tokens per window to avoid redundancy:

```python
layer_caches = [None] * num_layers
for window in sliding_windows:
    logits, new_caches = forward_with_cache(model, window, layer_caches)
    for layer_idx in range(num_layers):
        # Only cache the NEW tokens (stride=64), not the full 2048 window
        new_k = new_caches[layer_idx][0][:, -stride:]
        # Append to existing cache, trim to max_cache_tokens
        layer_caches[layer_idx] = concat_and_trim(old_cache, new_k, max_tokens=8192)
    score_tokens(logits, window)
```

## RoPE Considerations

The model was trained with `train_seq_len=1024` and uses NTK-aware RoPE scaling (auto-scales base frequency for longer sequences). For cache positions beyond the training length, RoPE quality degrades gradually. This is a known limitation — the same issue affects any long-context evaluation.

Potential mitigations:
- **Cache only last N layers** (e.g., last 4 with XSA) — earlier layers handle local patterns that don't need extended context
- **Limit cache to 4096 tokens** — stays within 4x of training length where NTK scaling is still effective
- **Use RoPE base 50000** (as in PR #254) — extends the effective RoPE range

## Rule Compliance

Per the organizer ruling on TTT (Mar 20):
> "You can't train on the validation tokens before you evaluate on those same tokens."

Neural Cache does NOT train on anything. It caches intermediate hidden states (K/V pairs) from **already-evaluated** tokens and uses them as additional context for future tokens. This is:
- **No weight modification** (unlike TTT)
- **Backward-looking only** (only uses K/V from scored tokens)
- **Equivalent to a longer context window** — evaluation methods are explicitly unrestricted

## Status: Untested Due to Compute Constraints

We implemented the full Neural Cache eval but encountered a bug in the model state after `torch.compile` — the custom forward path produced invalid results when called on the compiled `base_model`. The fix (using a fresh `eval_model` loaded from saved weights) was identified but we ran out of compute budget before re-running.

**The code is provided below for anyone to test.** Expected cost: one 8xH100 run (~$5) to train + eval with Neural Cache.

## Estimated Impact

- **Conservative:** 0.005-0.01 BPB (from context extension alone)
- **Optimistic:** 0.01-0.03 BPB (if the model effectively leverages 10K+ context)
- **Risk:** RoPE degradation beyond training length could limit gains

For reference, sliding window eval (extending context via overlap) gave -0.034 BPB (PR #77). Neural Cache extends context further via a complementary mechanism.

## Reproduction

Base model: PR #287's recipe (XSA + EMA + 11L + SmearGate + BigramHash)

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Our reproduction: 7,009 steps @ 85.6ms/step, **1.1284 BPB** sliding window (vs PR #287's 1.1271).

## Hardware

8x NVIDIA H100 80GB SXM, RunPod. Training: 600s. Standard eval: ~30s. Sliding window: ~85s. Neural Cache eval (estimated): ~300s for 1M token subset.

## Author

Xiaoan Liu | NYU | GitHub: @sseanliu
