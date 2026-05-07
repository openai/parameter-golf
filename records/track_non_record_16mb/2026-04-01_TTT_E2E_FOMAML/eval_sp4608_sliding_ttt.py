"""Sliding window eval with TTT on sp4608 8xH100 model.

Matches PR 1105 eval: stride=64, seq_len=2048, BPB with sentencepiece byte counting.
TTT: score window, record BPB for scored tokens, then SGD update prime MLPs.
"""
from eval_sp4608_ttt import *

STRIDE = int(os.environ.get("STRIDE", 64))


def eval_sliding_ttt(model, val_tokens, device, ttt_lr, stride,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     momentum=0.0, max_windows=0):
    """Sliding window eval with score-first TTT."""
    seq_len = SEQ_LEN
    total_tokens = val_tokens.numel() - 1

    # Generate window starts (same as PR 1105)
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + seq_len - stride < total_tokens]
    if max_windows > 0:
        window_starts = window_starts[:max_windows]
    total_windows = len(window_starts)

    prime_params = [p for _, p in model.prime_named_params()]
    optimizer = torch.optim.SGD(prime_params, lr=ttt_lr, momentum=momentum) if ttt_lr > 0 else None

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    model.eval()
    batch_seqs = 16  # process multiple windows at once for speed

    for bi in range(0, total_windows, batch_seqs):
        batch_ws = window_starts[bi:bi + batch_seqs]
        bsz = len(batch_ws)

        x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        wlens = []
        for i, ws in enumerate(batch_ws):
            end = min(ws + seq_len, total_tokens)
            wlen = end - ws
            wlens.append(wlen)
            chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
            x_batch[i, :wlen] = chunk[:-1]
            y_batch[i, :wlen] = chunk[1:]

        # ── SCORE (no grad) ──
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

        # ── TTT UPDATE on scored windows ──
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                train_loss = model(x_batch, y_batch)
            train_loss.backward()
            optimizer.step()

        if bi % (5000 * batch_seqs) == 0 or bi + batch_seqs >= total_windows:
            elapsed = time.perf_counter() - t0
            bpb = (loss_sum.item() / max(token_count.item(), 1)) / math.log(2.0) * \
                  (token_count.item() / max(byte_count.item(), 1))
            windows_done = min(bi + batch_seqs, total_windows)
            print(f"  [{windows_done}/{total_windows}] bpb={bpb:.6f} t={elapsed:.0f}s")

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    print(f"done: val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
          f"tokens={token_count.item():.0f} elapsed={time.perf_counter() - t0:.0f}s")
    return val_bpb


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    val_tokens = load_validation_tokens(os.path.join(DATA_PATH, "fineweb_val_*.bin"), SEQ_LEN)
    luts = build_sentencepiece_luts(sp, VOCAB_SIZE, device)
    print(f"val tokens: {val_tokens.numel() - 1}, stride: {STRIDE}")

    model = GPT_SP4608(prime_rank=PRIME_RANK, prime_layers=PRIME_LAYERS).to(device).bfloat16()

    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model_sd = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
    model.load_state_dict(model_sd)
    print(f"loaded {loaded}/{len(sd)} keys")
    print(f"total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"prime params: {sum(p.numel() for _, p in model.prime_named_params()):,}")

    for p in model.parameters():
        p.requires_grad_(False)
    for _, p in model.prime_named_params():
        p.requires_grad_(True)

    # Baseline (sliding window, no TTT)
    print(f"\n=== Baseline sliding window (stride={STRIDE}, no TTT) ===")
    reset_primes(model)
    bl = eval_sliding_ttt(model, val_tokens, device, 0.0, STRIDE, *luts)
    print(f"baseline: {bl:.6f}")

    # TTT LR sweep
    results = {}
    for lr in [0.003, 0.01, 0.03, 0.1]:
        print(f"\n=== TTT lr={lr} (sliding, stride={STRIDE}) ===")
        reset_primes(model)
        bpb = eval_sliding_ttt(model, val_tokens, device, lr, STRIDE, *luts)
        results[lr] = bpb
        print(f"lr={lr}: {bpb:.6f} ({bpb - bl:+.6f})")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY (sliding window stride={STRIDE}, baseline={bl:.6f})")
    print(f"{'='*50}")
    for lr, bpb in sorted(results.items()):
        print(f"  lr={lr:6.3f}: {bpb:.6f} ({bpb - bl:+.6f})")


if __name__ == "__main__":
    main()
