"""Sweep v2: LR extension, rank, layers, momentum.

All eval-only on Phase 1 checkpoint. 5K chunks for fast signal.
"""
from train_ttt_e2e import *

def reset_primes(model, rank=None, layers=None):
    """Reset prime MLPs to zero-init state."""
    with torch.no_grad():
        for n, p in model.prime_named_params():
            if "down" in n:
                p.zero_()
            else:
                p.data = torch.nn.init.orthogonal_(
                    torch.empty_like(p, dtype=torch.float32)).to(p.dtype)

def eval_ttt_quick(model, val_tokens, device, ttt_lr, chunk_tokens,
                   base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                   max_chunks=5000, momentum=0.0):
    seq_len = SEQ_LEN
    total_tokens = val_tokens.numel() - 1
    num_chunks = min((total_tokens + chunk_tokens - 1) // chunk_tokens,
                     max_chunks) if max_chunks > 0 else \
                 (total_tokens + chunk_tokens - 1) // chunk_tokens

    prime_params = [p for _, p in model.prime_named_params()]
    if ttt_lr > 0:
        optimizer = torch.optim.SGD(prime_params, lr=ttt_lr, momentum=momentum)
    else:
        optimizer = None

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    model.eval()
    for ci in range(num_chunks):
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

        if optimizer is not None and ci < num_chunks - 1:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                train_loss = model(x, y)
            train_loss.backward()
            optimizer.step()

    val_bpb = (loss_sum / token_count).item() / math.log(2.0) * \
              (token_count.item() / byte_count.item())
    elapsed = time.perf_counter() - t0
    return val_bpb, elapsed


def build_model(prime_rank, prime_layers, device, checkpoint):
    model = GPT_TTT(
        vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, model_dim=MODEL_DIM,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, mlp_mult=MLP_MULT,
        logit_softcap=LOGIT_SOFTCAP, rope_base=ROPE_BASE, qk_gain_init=QK_GAIN_INIT,
        bigram_vocab_size=BIGRAM_VOCAB, bigram_dim=BIGRAM_DIM,
        xsa_last_n=XSA_LAST_N, rope_dims=ROPE_DIMS, ln_scale=True,
        ve_enabled=VE_ENABLED, ve_dim=VE_DIM, ve_layers=VE_LAYERS,
        prime_rank=prime_rank, prime_layers=prime_layers,
    ).to(device).bfloat16()

    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_sd = model.state_dict()
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
    model.load_state_dict(model_sd)

    for p in model.parameters():
        p.requires_grad_(False)
    for _, p in model.prime_named_params():
        p.requires_grad_(True)
    return model


def run_one(label, model, val_tokens, device, luts, lr, chunk=1024, momentum=0.0, max_chunks=5000):
    reset_primes(model)
    bpb, elapsed = eval_ttt_quick(model, val_tokens, device, lr, chunk,
                                  *luts, max_chunks=max_chunks, momentum=momentum)
    print(f"  {label:40s} bpb={bpb:.6f}  t={elapsed:.0f}s")
    return bpb


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    val_tokens = load_validation_tokens(
        os.path.join(DATA_PATH, "fineweb_val_*.bin"), SEQ_LEN)
    luts = build_sentencepiece_luts(sp, VOCAB_SIZE, device)
    print(f"val tokens: {val_tokens.numel() - 1}\n")

    results = {}

    # ── 1. LR extension (rank=256, layers=8,9,10) ──
    print("=== LR extension (rank=256, layers=8,9,10) ===")
    model = build_model(256, [8, 9, 10], device, CHECKPOINT)
    for lr in [0.0, 0.03, 0.1, 0.3, 1.0]:
        label = f"lr={lr}"
        results[label] = run_one(label, model, val_tokens, device, luts, lr)
    del model; torch.cuda.empty_cache()

    # ── 2. Rank sweep (lr=0.1, layers=8,9,10) ──
    print("\n=== Rank sweep (lr=0.1, layers=8,9,10) ===")
    baseline = results["lr=0.0"]
    for rank in [64, 128, 256, 512]:
        model = build_model(rank, [8, 9, 10], device, CHECKPOINT)
        label = f"rank={rank}"
        results[label] = run_one(label, model, val_tokens, device, luts, 0.1)
        del model; torch.cuda.empty_cache()

    # ── 3. Layer sweep (lr=0.1, rank=256) ──
    print("\n=== Layer sweep (lr=0.1, rank=256) ===")
    layer_configs = {
        "layer=[10]": [10],
        "layer=[9,10]": [9, 10],
        "layer=[8,9,10]": [8, 9, 10],
        "layer=[7,8,9,10]": [7, 8, 9, 10],
        "layer=[6..10]": [6, 7, 8, 9, 10],
        "layer=all": list(range(11)),
    }
    for label, layers in layer_configs.items():
        model = build_model(256, layers, device, CHECKPOINT)
        results[label] = run_one(label, model, val_tokens, device, luts, 0.1)
        del model; torch.cuda.empty_cache()

    # ── 4. Momentum sweep (lr=0.1, rank=256, layers=8,9,10) ──
    print("\n=== Momentum sweep (lr=0.1, rank=256, layers=8,9,10) ===")
    model = build_model(256, [8, 9, 10], device, CHECKPOINT)
    for mom in [0.0, 0.5, 0.9]:
        label = f"momentum={mom}"
        results[label] = run_one(label, model, val_tokens, device, luts, 0.1, momentum=mom)
    del model; torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY (5K chunks, baseline={baseline:.6f})")
    print(f"{'='*60}")
    for label, bpb in results.items():
        delta = bpb - baseline
        marker = " ***" if delta < -0.003 else ""
        print(f"  {label:40s} {bpb:.6f}  ({delta:+.6f}){marker}")

    # ── Full eval on best ──
    best_label = min(results, key=results.get)
    best_bpb = results[best_label]
    if best_bpb < baseline - 0.001:
        print(f"\n=== Full eval: {best_label} ===")
        # Parse config from best label
        # Determine rank and layers from best
        rank = 256
        layers = [8, 9, 10]
        lr = 0.1
        mom = 0.0
        if "rank=" in best_label:
            rank = int(best_label.split("=")[1])
        if "layer=" in best_label:
            layers = layer_configs.get(best_label, [8, 9, 10])
        if "lr=" in best_label:
            lr = float(best_label.split("=")[1])
        if "momentum=" in best_label:
            mom = float(best_label.split("=")[1])

        model = build_model(rank, layers, device, CHECKPOINT)
        full_bpb = run_one(f"FULL {best_label}", model, val_tokens, device, luts,
                           lr, momentum=mom, max_chunks=0)
        print(f"  FULL: {full_bpb:.6f} (baseline: {baseline:.6f}, delta: {full_bpb - baseline:+.6f})")


if __name__ == "__main__":
    main()
