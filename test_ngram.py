#!/usr/bin/env python3
"""Quick N-gram cache test on a subset of val tokens.
Uses the model from the last training run (final_model.int6.ptz)."""
import torch, torch.nn.functional as F, math, time, os, io, lzma
import numpy as np
from collections import defaultdict

# Import model and quantization utilities
from train_gpt import (
    Hyperparameters, GPT, CastedLinear, restore_low_dim_params_to_fp32,
    _rebank_state_dict, dequantize_mixed_int6,
    load_validation_tokens, build_sentencepiece_luts,
)
import sentencepiece as spm

def main():
    args = Hyperparameters()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    # Load tokenizer and val tokens
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Take first N tokens for quick test
    N_TOKENS = int(os.environ.get("TEST_TOKENS", 500_000))
    val_subset = val_tokens[:N_TOKENS + 1]
    total_tokens = val_subset.numel() - 1
    print(f"Testing N-gram on {total_tokens} tokens (of {val_tokens.numel()-1} total)")

    # Load quantized model
    ptz_file = "final_model.int6.ptz"
    if not os.path.exists(ptz_file):
        print(f"ERROR: {ptz_file} not found. Run training first.")
        return

    with open(ptz_file, "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")

    # Build template state dict for dequantization
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        gated_mlp=args.gated_mlp,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)

    # Dequantize and load weights
    from train_gpt import _unbank_state_dict
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    unbanked_template = _unbank_state_dict(sd_cpu, args.num_layers)
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_template)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    eval_model.load_state_dict(deq_state, strict=True)
    eval_model.eval()

    # Sliding window eval on subset
    seq_len = args.train_seq_len
    stride = 64
    batch_seqs = 32

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    print(f"Windows: {len(window_starts)}, stride={stride}")

    val_np = val_subset.cpu().numpy().astype(np.int64)
    bytes_lut = base_bytes_lut.cpu().numpy().astype(np.float64)
    space_lut = has_leading_space_lut.cpu().numpy()
    boundary_lut = is_boundary_token_lut.cpu().numpy()

    # Collect model NLL for all scored positions
    model_nll = np.full(total_tokens, -1.0, dtype=np.float64)
    compiled_logits = torch.compile(eval_model.forward_logits, dynamic=False, fullgraph=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_seqs):
            batch_ws = window_starts[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_subset[ws:end + 1].to(dtype=torch.int64, device=device)
                x[i, :wlen] = chunk[:-1]
                y[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                for t in range(s, wlen):
                    pos = ws + t
                    if model_nll[pos] < 0:
                        target = int(val_np[pos + 1])
                        model_nll[pos] = -log_probs[i, t, target].item()
    t_model = time.perf_counter() - t0
    scored_mask = model_nll >= 0
    n_scored = int(scored_mask.sum())
    print(f"Model inference: {n_scored} tokens in {t_model:.1f}s")

    # Compute baseline BPB (no N-gram)
    base_loss = 0.0
    base_bytes = 0.0
    for pos in range(total_tokens):
        if model_nll[pos] < 0:
            continue
        target = int(val_np[pos + 1])
        base_loss += model_nll[pos]
        tb = bytes_lut[target]
        if space_lut[target] and not boundary_lut[int(val_np[pos])]:
            tb += 1.0
        base_bytes += tb
    base_bpb = (base_loss / n_scored) / math.log(2.0) * (n_scored / base_bytes)
    print(f"Baseline (no N-gram): val_bpb={base_bpb:.4f}")

    # N-gram cache test with different orders and alphas
    for max_order in [5, 7, 9]:
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
            t_ng = time.perf_counter()
            caches = [defaultdict(lambda: defaultdict(int)) for _ in range(max_order + 1)]
            ng_loss = 0.0
            ng_bytes = 0.0
            n_hits = 0
            for pos in range(total_tokens):
                if model_nll[pos] < 0:
                    continue
                target = int(val_np[pos + 1])
                # Backoff lookup
                ngram_prob = 0.0
                for order in range(max_order, 1, -1):
                    if pos + 1 >= order:
                        ctx = tuple(val_np[pos + 2 - order:pos + 1].tolist())
                        if ctx in caches[order]:
                            counts = caches[order][ctx]
                            total_c = sum(counts.values())
                            ngram_prob = counts.get(target, 0) / total_c
                            break
                if ngram_prob > 0:
                    model_p = math.exp(-model_nll[pos])
                    combined_p = max((1 - alpha) * model_p + alpha * ngram_prob, 1e-30)
                    ng_loss += -math.log(combined_p)
                    n_hits += 1
                else:
                    ng_loss += model_nll[pos]
                # Byte count
                tb = bytes_lut[target]
                if space_lut[target] and not boundary_lut[int(val_np[pos])]:
                    tb += 1.0
                ng_bytes += tb
                # Update cache
                for order in range(2, max_order + 1):
                    if pos + 1 >= order:
                        ctx = tuple(val_np[pos + 2 - order:pos + 1].tolist())
                        caches[order][ctx][target] += 1
            ng_bpb = (ng_loss / n_scored) / math.log(2.0) * (n_scored / ng_bytes)
            delta = ng_bpb - base_bpb
            t_ng_elapsed = time.perf_counter() - t_ng
            print(f"N-gram order={max_order} alpha={alpha:.2f}: bpb={ng_bpb:.4f} delta={delta:+.4f} hits={n_hits}/{n_scored} ({100*n_hits/n_scored:.1f}%) time={t_ng_elapsed:.1f}s")

if __name__ == "__main__":
    main()
