"""Standalone n-gram eval on saved pre-TTT model. No training, no GPTQ."""
import copy, math, os, sys, time, torch, torch.nn.functional as F
import numpy as np
import torch.distributed as dist

# Import model definition from the mixer script
sys.path.insert(0, os.path.dirname(__file__) or ".")
# We'll inline what we need

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Load the pre-TTT model
    model_path = os.environ.get("MODEL_PATH", "/data/backups/rganapa/parameter-golf/final_model_pre_ttt.pt")
    script_path = os.environ.get("SCRIPT_PATH", "/data/backups/rganapa/parameter-golf/clean_train_206_freeze_sweep.py")
    data_path = os.environ.get("DATA_PATH", "/data/backups/rganapa/parameter-golf/data/datasets/fineweb10B_sp1024")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "/data/backups/rganapa/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
    stride = int(os.environ.get("EVAL_STRIDE", "76"))
    seq_len = int(os.environ.get("EVAL_SEQ_LEN", "2048"))
    batch_seqs = int(os.environ.get("BATCH_SEQS", "128"))

    if rank == 0:
        print(f"Loading model from {model_path}...", flush=True)

    # Load state dict
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Import model class from the training script
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_mod", script_path)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    # Build model
    model = train_mod.GPT(
        vocab_size=1024, num_layers=14, model_dim=512,
        num_heads=8, num_kv_heads=4, mlp_mult=3,
        tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=50000, qk_gain_init=1.5,
        bigram_vocab_size=8192, bigram_dim=64,
    ).to(device).bfloat16()

    for m in model.modules():
        if isinstance(m, train_mod.CastedLinear):
            m.float()
    train_mod.restore_low_dim_params_to_fp32(model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if rank == 0:
        print(f"Model loaded. Running n-gram eval at stride={stride}...", flush=True)

    # Load val tokens (binary format: 256-int32 header + uint16 tokens)
    import glob
    from pathlib import Path
    val_pattern = os.path.join(data_path, "fineweb_val_*.bin")
    val_files = sorted(glob.glob(val_pattern))
    all_tokens = []
    for f in val_files:
        header = np.fromfile(f, dtype="<i4", count=256)
        num_tokens = int(header[2])
        header_bytes = 256 * np.dtype("<i4").itemsize
        tokens_np = np.fromfile(f, dtype="<u2", count=num_tokens, offset=header_bytes)
        all_tokens.append(torch.from_numpy(tokens_np.astype(np.int32)))
    val_tokens = torch.cat(all_tokens).contiguous()
    if rank == 0:
        print(f"val_tokens: min={val_tokens.min().item()} max={val_tokens.max().item()} n={val_tokens.numel()}", flush=True)

    # Build lookup tables for BPB
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
    vocab_size = 1024
    base_bytes_lut = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    has_leading_space_lut = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary_token_lut = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for t_id in range(vocab_size):
        piece = sp.id_to_piece(t_id)
        raw = piece.replace("\u2581", " ").encode("utf-8")
        base_bytes_lut[t_id] = max(len(raw), 1)
        has_leading_space_lut[t_id] = piece.startswith("\u2581")
        is_boundary_token_lut[t_id] = (t_id in (0, 1, 2))  # BOS, EOS, UNK

    # N-gram eval
    total_tokens = val_tokens.numel() - 1
    max_order = 5
    orders = list(range(2, max_order + 1))
    buckets = 4194304
    mask = np.uint64(buckets - 1)
    primes = np.array([np.uint64(36313), np.uint64(27191), np.uint64(51647),
                       np.uint64(81929), np.uint64(131071)], dtype=np.uint64)
    alpha_low, alpha_high, entropy_thresh = 0.05, 0.40, 4.0
    min_count = 2

    ctx_tables = {n: np.zeros(buckets, dtype=np.uint32) for n in orders}
    full_tables = {n: np.zeros(buckets, dtype=np.uint32) for n in orders}

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    my_s = (len(window_starts) * rank) // world_size
    my_e = (len(window_starts) * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    val_np = val_tokens.numpy().astype(np.int64)
    compiled_logits = model.forward_logits  # skip compile for eval-only (avoids assertion issues)

    loss_sum = 0.0; token_count = 0.0; byte_count = 0.0
    t0 = time.perf_counter()

    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws; wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            logits_float = logits.float()
            nll = F.cross_entropy(logits_float.reshape(-1, logits_float.size(-1)),
                                  y_batch.reshape(-1), reduction="none").reshape(bsz, seq_len)
            log_probs = F.log_softmax(logits_float, dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                seg_len = wlen - s
                if seg_len <= 0: continue

                seg_nll = nll[i, s:wlen].to(torch.float64).cpu().numpy()
                seg_model_p = np.exp(-seg_nll)
                seg_entropy = entropy[i, s:wlen].cpu().numpy().astype(np.float64)
                global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)

                best_p_ng = np.zeros(seg_len, dtype=np.float64)
                has_ngram = np.zeros(seg_len, dtype=bool)

                for n in reversed(orders):
                    ctx_width = n - 1
                    valid = (global_j >= n - 1) & ~has_ngram
                    if not valid.any(): continue
                    v_idx = np.nonzero(valid)[0]
                    jv = global_j[v_idx]
                    ctx_hash = np.zeros(len(jv), dtype=np.uint64)
                    for k in range(ctx_width):
                        tok = val_np[jv - (ctx_width - k)].astype(np.uint64)
                        ctx_hash ^= tok * primes[k % len(primes)]
                    ctx_key = (ctx_hash & mask).astype(np.int64)
                    tgt_np = val_np[jv].astype(np.uint64)
                    full_key = ((ctx_hash ^ (tgt_np * primes[ctx_width % len(primes)])) & mask).astype(np.int64)
                    ctx_counts = ctx_tables[n][ctx_key].astype(np.float64)
                    full_counts = full_tables[n][full_key].astype(np.float64)
                    can_mix = ctx_counts >= float(min_count)
                    if can_mix.any():
                        p_ng = np.minimum(full_counts, ctx_counts) / np.maximum(ctx_counts, 1.0)
                        p_ng = np.clip(p_ng, 0.0, 1.0)
                        mix_idx = v_idx[can_mix]
                        best_p_ng[mix_idx] = p_ng[can_mix]
                        has_ngram[mix_idx] = True

                if has_ngram.any():
                    ng_idx = np.nonzero(has_ngram)[0]
                    ent = seg_entropy[ng_idx]
                    sig = 1.0 / (1.0 + np.exp(-2.0 * (ent - entropy_thresh)))
                    alpha_vec = alpha_low + (alpha_high - alpha_low) * sig
                    mixed = (1.0 - alpha_vec) * seg_model_p[ng_idx] + alpha_vec * best_p_ng[ng_idx]
                    seg_model_p[ng_idx] = mixed

                seg_nll = -np.log(np.clip(seg_model_p, 1e-12, 1.0))
                loss_sum += float(seg_nll.sum())
                token_count += float(seg_len)
                tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += float(tb.sum().item())

                # Vectorized update
                all_global_j = np.arange(ws + 1, ws + wlen + 1, dtype=np.int64)
                for n in orders:
                    ctx_width = n - 1
                    v_mask = all_global_j >= n - 1
                    if not v_mask.any(): continue
                    vi = np.nonzero(v_mask)[0]
                    jv2 = all_global_j[vi]
                    ch = np.zeros(len(jv2), dtype=np.uint64)
                    for k in range(ctx_width):
                        tok2 = val_np[jv2 - (ctx_width - k)].astype(np.uint64)
                        ch ^= tok2 * primes[k % len(primes)]
                    ck = (ch & mask).astype(np.int64)
                    tn = val_np[jv2].astype(np.uint64)
                    fk = ((ch ^ (tn * primes[ctx_width % len(primes)])) & mask).astype(np.int64)
                    np.add.at(ctx_tables[n], ck, 1)
                    np.add.at(full_tables[n], fk, 1)

            if rank == 0 and (bi + bsz) % max(1, len(my_windows) // 5) < batch_seqs:
                elapsed = time.perf_counter() - t0
                cov = has_ngram.mean() if len(has_ngram) > 0 else 0
                cur_bpb = (loss_sum / max(token_count, 1)) / math.log(2) * (token_count / max(byte_count, 1))
                print(f"ngram: {bi+bsz}/{len(my_windows)} t={elapsed:.0f}s cov={cov*100:.0f}% bpb={cur_bpb:.6f}", flush=True)

    _loss = torch.tensor(loss_sum, device=device, dtype=torch.float64)
    _toks = torch.tensor(token_count, device=device, dtype=torch.float64)
    _bytes = torch.tensor(byte_count, device=device, dtype=torch.float64)
    dist.all_reduce(_loss); dist.all_reduce(_toks); dist.all_reduce(_bytes)
    val_loss = _loss.item() / _toks.item()
    val_bpb = val_loss / math.log(2) * (_toks.item() / _bytes.item())

    if rank == 0:
        elapsed = time.perf_counter() - t0
        print(f"ngram_eval val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} time:{elapsed:.0f}s", flush=True)
        print(f"ngram_eval_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
