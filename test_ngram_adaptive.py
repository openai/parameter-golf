#!/usr/bin/env python3
"""Test adaptive vs fixed alpha N-gram on a subset of val tokens."""
import torch, torch.nn.functional as F, math, time, os, io, lzma
import numpy as np
from collections import defaultdict
from train_gpt import (
    Hyperparameters, GPT, CastedLinear, restore_low_dim_params_to_fp32,
    _unbank_state_dict, _rebank_state_dict, dequantize_mixed_int6,
    load_validation_tokens, build_sentencepiece_luts,
)
import sentencepiece as spm

def run_ngram(model_nll, val_np, bytes_lut, space_lut, boundary_lut,
              total_tokens, max_order, alpha, adaptive):
    caches = [defaultdict(lambda: defaultdict(int)) for _ in range(max_order + 1)]
    loss_sum = 0.0
    byte_sum = 0.0
    n_scored = 0
    n_hits = 0
    nll_threshold = 2.5
    for pos in range(total_tokens):
        if model_nll[pos] < 0:
            continue
        n_scored += 1
        target = int(val_np[pos + 1])
        ngram_prob = 0.0
        for order in range(max_order, 1, -1):
            if pos + 1 < order:
                continue
            ctx = tuple(val_np[pos + 2 - order:pos + 1].tolist())
            if ctx in caches[order]:
                counts = caches[order][ctx]
                total_c = sum(counts.values())
                ngram_prob = counts.get(target, 0) / total_c
                break
        if ngram_prob > 0:
            if adaptive:
                a = alpha * min(2.0, max(0.1, model_nll[pos] / nll_threshold))
            else:
                a = alpha
            model_p = math.exp(-model_nll[pos])
            combined_p = max((1 - a) * model_p + a * ngram_prob, 1e-30)
            loss_sum += -math.log(combined_p)
            n_hits += 1
        else:
            loss_sum += model_nll[pos]
        tb = bytes_lut[target]
        if space_lut[target] and not boundary_lut[int(val_np[pos])]:
            tb += 1.0
        byte_sum += tb
        for order in range(2, max_order + 1):
            if pos + 1 < order:
                continue
            ctx = tuple(val_np[pos + 2 - order:pos + 1].tolist())
            caches[order][ctx][target] += 1
    val_loss = loss_sum / max(n_scored, 1)
    bpb = val_loss / math.log(2.0) * (n_scored / max(byte_sum, 1.0))
    return bpb, n_hits, n_scored

def main():
    args = Hyperparameters()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    N_TOKENS = int(os.environ.get("TEST_TOKENS", 500_000))
    val_subset = val_tokens[:N_TOKENS + 1]
    total_tokens = val_subset.numel() - 1
    val_np = val_subset.cpu().numpy().astype(np.int64)
    bytes_lut = base_bytes_lut.cpu().numpy().astype(np.float64)
    space_lut = has_leading_space_lut.cpu().numpy()
    boundary_lut = is_boundary_token_lut.cpu().numpy()

    # Load model and compute model NLL
    ptz_file = "final_model.int6.ptz"
    with open(ptz_file, "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")
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
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    unbanked = _unbank_state_dict(sd_cpu, args.num_layers)
    deq = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked)
    deq_state = _rebank_state_dict(deq, args.num_layers, sd_cpu)
    eval_model.load_state_dict(deq_state, strict=True)
    eval_model.eval()

    # Sliding window model inference
    stride = 64
    seq_len = args.train_seq_len
    batch_seqs = 32
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
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
    n_scored = int((model_nll >= 0).sum())
    print(f"Model inference: {n_scored} tokens in {time.perf_counter()-t0:.1f}s")

    # Baseline
    base_loss = sum(model_nll[pos] for pos in range(total_tokens) if model_nll[pos] >= 0)
    base_bytes = sum(
        bytes_lut[int(val_np[pos+1])] + (1.0 if space_lut[int(val_np[pos+1])] and not boundary_lut[int(val_np[pos])] else 0.0)
        for pos in range(total_tokens) if model_nll[pos] >= 0
    )
    base_bpb = (base_loss / n_scored) / math.log(2.0) * (n_scored / base_bytes)
    print(f"Baseline: bpb={base_bpb:.4f}")

    # Test fixed vs adaptive alpha
    for order in [7]:
        for alpha in [0.10, 0.15, 0.20, 0.30]:
            for adaptive in [False, True]:
                t1 = time.perf_counter()
                bpb, hits, scored = run_ngram(model_nll, val_np, bytes_lut, space_lut, boundary_lut,
                                              total_tokens, order, alpha, adaptive)
                dt = time.perf_counter() - t1
                mode = "adaptive" if adaptive else "fixed"
                delta = bpb - base_bpb
                print(f"order={order} alpha={alpha:.2f} {mode:8s}: bpb={bpb:.4f} delta={delta:+.4f} hits={hits}/{scored} time={dt:.1f}s")

if __name__ == "__main__":
    main()
