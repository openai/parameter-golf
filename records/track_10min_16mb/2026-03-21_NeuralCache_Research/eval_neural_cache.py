"""Neural Cache Evaluation: Cross-window KV caching for extended context.

Usage: Add this to the end of the training script's main() function,
AFTER the int6 sliding window eval creates `eval_model`.

    # --- NEURAL CACHE EVAL ---
    if master_process:
        for cache_size in [0, 2048, 4096]:
            nc_loss, nc_bpb = eval_neural_cache(
                eval_model, rank, device, val_tokens, base_bytes_lut,
                has_leading_space_lut, is_boundary_token_lut,
                seq_len=args.train_seq_len, stride=64,
                max_cache_tokens=cache_size, max_eval_tokens=1000000)
            print(f"neural_cache cache={cache_size} bpb={nc_bpb:.6f}")

IMPORTANT: Use `eval_model` (fresh model loaded from saved weights),
NOT `base_model` (which has torch.compile applied and produces invalid results).
"""

import math
import time
import torch
import torch.nn.functional as F
from flash_attn_interface import flash_attn_func as flash_attn_3_func


def attn_forward_with_cache(attn_module, x, kv_cache=None, cache_seqlen=0):
    """Attention forward with KV cache prepended for extended context.

    Args:
        attn_module: CausalSelfAttention module
        x: input [bsz, seqlen, dim] (already through attn_norm)
        kv_cache: tuple (cached_k, cached_v) or None
        cache_seqlen: number of tokens in cache (for RoPE position offset)

    Returns:
        output: [bsz, seqlen, dim]
        new_kv: tuple (k, v) for current window
    """
    # Import apply_rotary_emb from the training script
    from train_gpt import apply_rotary_emb

    bsz, seqlen, dim = x.shape
    q = attn_module.c_q(x).reshape(bsz, seqlen, attn_module.num_heads, attn_module.head_dim)
    k = attn_module.c_k(x).reshape(bsz, seqlen, attn_module.num_kv_heads, attn_module.head_dim)
    v = attn_module.c_v(x).reshape(bsz, seqlen, attn_module.num_kv_heads, attn_module.head_dim)

    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))

    # RoPE with position offset for cached context
    total_len = cache_seqlen + seqlen
    cos, sin = attn_module.rotary(total_len, x.device, q.dtype)
    q = apply_rotary_emb(q, cos[cache_seqlen:total_len], sin[cache_seqlen:total_len])
    k = apply_rotary_emb(k, cos[cache_seqlen:total_len], sin[cache_seqlen:total_len])

    q = q * attn_module.q_gain.to(dtype=q.dtype)[None, None, :, None]

    # Save current K/V before cache concatenation
    new_k, new_v = k.clone(), v.clone()

    # Prepend cached K/V from previous windows
    if kv_cache is not None:
        k = torch.cat([kv_cache[0], k], dim=1)
        v = torch.cat([kv_cache[1], v], dim=1)

    # flash_attn handles seqlen_k > seqlen_q with causal=True correctly:
    # queries attend to all cached tokens + causal portion of current window
    y = flash_attn_3_func(q, k, v, causal=True)

    if attn_module.use_xsa:
        y = attn_module._xsa_efficient(y, new_v)

    y = y.reshape(bsz, seqlen, dim)
    return attn_module.proj(y), (new_k, new_v)


def forward_logits_cached(model, input_ids, layer_caches=None, cache_seqlen=0):
    """Full forward pass with per-layer KV caches."""
    x = model.tok_emb(input_ids)
    if model.bigram is not None:
        x = x + model.bigram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = model.smear(x)
    x0 = x

    new_caches = []
    skips = []
    layer_idx = 0

    for i in range(model.num_encoder_layers):
        block = model.blocks[i]
        mix = block.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        lc = layer_caches[layer_idx] if layer_caches else None
        attn_out, new_kv = attn_forward_with_cache(
            block.attn, block.attn_norm(x), kv_cache=lc, cache_seqlen=cache_seqlen)
        new_caches.append(new_kv)
        layer_idx += 1

        x = x + block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * block.mlp(block.mlp_norm(x))
        skips.append(x)

    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        block = model.blocks[model.num_encoder_layers + i]
        mix = block.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        lc = layer_caches[layer_idx] if layer_caches else None
        attn_out, new_kv = attn_forward_with_cache(
            block.attn, block.attn_norm(x), kv_cache=lc, cache_seqlen=cache_seqlen)
        new_caches.append(new_kv)
        layer_idx += 1

        x = x + block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * block.mlp(block.mlp_norm(x))

    x = model.final_norm(x)
    if model.tie_embeddings:
        logits_proj = F.linear(x, model.tok_emb.weight)
    else:
        logits_proj = model.lm_head(x)
    logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)
    return logits, new_caches


def eval_neural_cache(
    model, rank, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    seq_len=2048, stride=64, max_cache_tokens=4096, max_eval_tokens=1000000,
):
    """Sliding window eval with cross-window KV caching.

    Args:
        model: GPT model (use eval_model, NOT base_model after torch.compile)
        rank: distributed rank (only rank 0 runs this)
        device: CUDA device
        val_tokens: validation token tensor
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut: BPB lookup tables
        seq_len: window size (default 2048)
        stride: scoring stride (default 64)
        max_cache_tokens: maximum cached K/V tokens per layer (0 = no caching)
        max_eval_tokens: subset size for quick testing

    Returns:
        (val_loss, val_bpb) tuple
    """
    if rank != 0:
        return 0.0, 0.0

    total_tokens = min(val_tokens.numel() - 1, max_eval_tokens)
    num_layers = len(model.blocks)

    loss_sum = 0.0
    token_count = 0
    byte_count = 0.0
    layer_caches = [None] * num_layers
    cache_seqlen = 0

    model.eval()
    t0 = time.perf_counter()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for ws in range(0, total_tokens, stride):
            end = min(ws + seq_len, total_tokens)
            wlen = end - ws
            if wlen < 1:
                break

            chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
            x_in = chunk[:-1].unsqueeze(0)
            y_tgt = chunk[1:].unsqueeze(0)

            logits, new_caches = forward_logits_cached(
                model, x_in, layer_caches=layer_caches, cache_seqlen=cache_seqlen)

            # Update per-layer caches: only store the stride-worth of NEW tokens
            for li in range(num_layers):
                if max_cache_tokens == 0:
                    layer_caches[li] = None
                    continue
                new_k, new_v = new_caches[li]
                cache_k = new_k[:, -stride:]
                cache_v = new_v[:, -stride:]
                if layer_caches[li] is not None:
                    old_k, old_v = layer_caches[li]
                    cache_k = torch.cat([old_k, cache_k], dim=1)
                    cache_v = torch.cat([old_v, cache_v], dim=1)
                    if cache_k.size(1) > max_cache_tokens:
                        cache_k = cache_k[:, -max_cache_tokens:]
                        cache_v = cache_v[:, -max_cache_tokens:]
                layer_caches[li] = (cache_k, cache_v)

            cache_seqlen = min(ws + wlen, max_cache_tokens) if max_cache_tokens > 0 else 0

            # Score only the NEW tokens
            nll = F.cross_entropy(logits[0].float(), y_tgt[0], reduction="none")
            s = 0 if ws == 0 else max(wlen - stride, 0)
            scored_nll = nll[s:wlen].to(torch.float64)
            loss_sum += scored_nll.sum().item()
            token_count += wlen - s

            tgt = y_tgt[0, s:wlen]
            prev = x_in[0, s:wlen]
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            byte_count += tb.sum().item()

            if ws % (stride * 500) == 0 and ws > 0:
                elapsed = time.perf_counter() - t0
                running_bpb = (loss_sum / token_count / math.log(2.0)) * (token_count / byte_count)
                print(f"  ncache pos={ws}/{total_tokens} bpb={running_bpb:.4f} "
                      f"cache={cache_seqlen} elapsed={elapsed:.0f}s")

    elapsed = time.perf_counter() - t0
    val_loss = loss_sum / token_count
    bpb = (val_loss / math.log(2.0)) * (token_count / byte_count)
    return val_loss, bpb
