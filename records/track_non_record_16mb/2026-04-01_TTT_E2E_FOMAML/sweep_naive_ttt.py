"""Naive TTT with zero-init prime MLPs — no meta-learning.

Tests whether the prime MLP architecture enables useful adaptation
even without FOMAML. Sweeps TTT learning rates on the Phase 1 checkpoint.
Evaluates first 5000 chunks for fast signal, then full eval on best LR.
"""
from train_ttt_e2e import *

def eval_ttt_quick(model, val_tokens, device, ttt_lr, chunk_tokens,
                   base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                   max_chunks=0, reset_every=0):
    """Score-first TTT eval. max_chunks=0 means full eval. reset_every=0 means no reset."""
    seq_len = SEQ_LEN
    total_tokens = val_tokens.numel() - 1
    num_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens
    if max_chunks > 0:
        num_chunks = min(num_chunks, max_chunks)

    # Save init state for resets
    prime_init = {n: p.detach().clone() for n, p in model.prime_named_params()}

    prime_params = [p for _, p in model.prime_named_params()]
    optimizer = torch.optim.SGD(prime_params, lr=ttt_lr)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    model.eval()
    for ci in range(num_chunks):
        # Reset prime MLPs periodically
        if reset_every > 0 and ci > 0 and ci % reset_every == 0:
            with torch.no_grad():
                for n, p in model.prime_named_params():
                    p.copy_(prime_init[n])

        chunk_start = ci * chunk_tokens
        chunk_end = min((ci + 1) * chunk_tokens, total_tokens)
        chunk_len = chunk_end - chunk_start
        if chunk_len < 2:
            continue

        chunk_data = val_tokens[chunk_start:chunk_end + 1].to(device=device, dtype=torch.int64)
        num_seqs = chunk_len // seq_len
        if num_seqs == 0:
            x = chunk_data[:-1].unsqueeze(0)
            y = chunk_data[1:].unsqueeze(0)
        else:
            x = chunk_data[:num_seqs * seq_len].reshape(num_seqs, seq_len)
            y = chunk_data[1:num_seqs * seq_len + 1].reshape(num_seqs, seq_len)

        # SCORE (no grad)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  y.reshape(-1), reduction="none")
            loss_sum += nll.to(torch.float64).sum()
            token_count += float(y.numel())
            tgt = y.reshape(-1)
            prev = x.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            byte_count += tb.sum()

        # TRAIN prime MLPs
        if ci < num_chunks - 1:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                train_loss = model(x, y)
            train_loss.backward()
            optimizer.step()

        if ci % 1000 == 0 or ci == num_chunks - 1:
            elapsed = time.perf_counter() - t0
            bpb = (loss_sum.item() / max(token_count.item(), 1)) / math.log(2.0) * \
                  (token_count.item() / max(byte_count.item(), 1))
            print(f"  [{ci+1}/{num_chunks}] bpb={bpb:.6f} t={elapsed:.0f}s")

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_bpb


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    val_tokens = load_validation_tokens(
        os.path.join(DATA_PATH, "fineweb_val_*.bin"), SEQ_LEN)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, VOCAB_SIZE, device)
    print(f"val tokens: {val_tokens.numel() - 1}")

    # Build model with zero-init prime MLPs
    model = GPT_TTT(
        vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, model_dim=MODEL_DIM,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, mlp_mult=MLP_MULT,
        logit_softcap=LOGIT_SOFTCAP, rope_base=ROPE_BASE, qk_gain_init=QK_GAIN_INIT,
        bigram_vocab_size=BIGRAM_VOCAB, bigram_dim=BIGRAM_DIM,
        xsa_last_n=XSA_LAST_N, rope_dims=ROPE_DIMS, ln_scale=True,
        ve_enabled=VE_ENABLED, ve_dim=VE_DIM, ve_layers=VE_LAYERS,
        prime_rank=PRIME_RANK, prime_layers=PRIME_LAYERS,
    ).to(device).bfloat16()

    # Load Phase 1 checkpoint
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model_sd = model.state_dict()
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
    model.load_state_dict(model_sd)
    print(f"loaded checkpoint: {CHECKPOINT}")

    # Only prime params get gradients
    for p in model.parameters():
        p.requires_grad_(False)
    for _, p in model.prime_named_params():
        p.requires_grad_(True)

    # Baseline (no TTT, no prime MLPs active since down=0)
    print("\n=== Baseline (prime MLPs zero, no TTT) ===")
    bl = eval_ttt_quick(model, val_tokens, device, 0.0, TTT_CHUNK,
                        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                        max_chunks=5000)
    print(f"  baseline 5K chunks: {bl:.6f}")

    # LR sweep on first 5000 chunks
    lrs = [0.001, 0.003, 0.01, 0.03, 0.1]
    results = {}
    for lr in lrs:
        print(f"\n=== Naive TTT lr={lr} (5K chunks) ===")
        # Reset prime MLPs to zero
        with torch.no_grad():
            for n, p in model.prime_named_params():
                if "down" in n:
                    p.zero_()
                else:
                    p.data = torch.nn.init.orthogonal_(torch.empty_like(p, dtype=torch.float32)).to(p.dtype)
        bpb = eval_ttt_quick(model, val_tokens, device, lr, TTT_CHUNK,
                             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                             max_chunks=5000)
        results[lr] = bpb
        print(f"  lr={lr}: {bpb:.6f}")

    # Also test chunk=4096 with best LR
    best_lr = min(results, key=results.get)
    def reset_primes():
        with torch.no_grad():
            for n, p in model.prime_named_params():
                if "down" in n:
                    p.zero_()
                else:
                    p.data = torch.nn.init.orthogonal_(torch.empty_like(p, dtype=torch.float32)).to(p.dtype)

    print(f"\n=== Best LR={best_lr}, trying chunk=4096 (5K chunks) ===")
    reset_primes()
    bpb_4k = eval_ttt_quick(model, val_tokens, device, best_lr, 4096,
                            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                            max_chunks=5000)
    print(f"  chunk=4096: {bpb_4k:.6f}")

    # Test with periodic reset (best LR, chunk=1024, reset every 1000)
    print(f"\n=== Best LR={best_lr}, chunk=1024, reset every 1000 (5K chunks) ===")
    reset_primes()
    bpb_reset = eval_ttt_quick(model, val_tokens, device, best_lr, TTT_CHUNK,
                               base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                               max_chunks=5000, reset_every=1000)
    print(f"  reset_every=1000: {bpb_reset:.6f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY (5K chunks)")
    print(f"{'='*50}")
    print(f"Baseline (no TTT):     {bl:.6f}")
    for lr, bpb in sorted(results.items()):
        delta = bpb - bl
        print(f"TTT lr={lr:6.3f}:        {bpb:.6f}  ({delta:+.6f})")
    print(f"TTT lr={best_lr} chunk=4096: {bpb_4k:.6f}  ({bpb_4k - bl:+.6f})")
    print(f"TTT lr={best_lr} reset=1000: {bpb_reset:.6f}  ({bpb_reset - bl:+.6f})")

    # Full eval on best config if it beat baseline
    best_bpb = min(results.values(), default=bl)
    best_bpb = min(best_bpb, bpb_4k, bpb_reset)
    if best_bpb < bl - 0.0001:
        print(f"\n=== Full eval with best config ===")
        reset_primes()
        # Determine best config
        if best_bpb == bpb_4k:
            chunk, reset = 4096, 0
        elif best_bpb == bpb_reset:
            chunk, reset = TTT_CHUNK, 1000
        else:
            chunk, reset = TTT_CHUNK, 0
        full_bpb = eval_ttt_quick(model, val_tokens, device, best_lr, chunk,
                                  base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                                  reset_every=reset)
        print(f"  FULL: {full_bpb:.6f} (baseline: {bl:.6f}, delta: {full_bpb - bl:+.6f})")
    else:
        print(f"\nNo config beat baseline — skipping full eval.")


if __name__ == "__main__":
    main()
