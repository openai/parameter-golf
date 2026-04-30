"""Proper E2E TTT training: Phase 1 standard → Phase 2 FOMAML, ALL params joint.

Phase 1: Standard training with zero-init prime MLPs (no effect on forward).
Phase 2: FOMAML meta-fine-tuning. Base model at low LR, prime MLPs at high LR.
         Both inner and outer loops use the full model.
Eval:    Score-first TTT on val shard.

This is the proper two-phase training where the base model co-adapts with prime MLPs.
"""
from train_ttt_e2e import *

# Override defaults for proper training
PHASE1_STEPS   = int(os.environ.get("PHASE1_STEPS", 7000))
PHASE2_STEPS   = int(os.environ.get("PHASE2_STEPS", 1500))
PHASE1_LR      = float(os.environ.get("PHASE1_LR", 0.025))
PHASE2_BASE_LR = float(os.environ.get("PHASE2_BASE_LR", 0.001))
PHASE2_PRIME_LR = float(os.environ.get("PHASE2_PRIME_LR", 0.003))
PHASE2_INNER_LR = float(os.environ.get("PHASE2_INNER_LR", 0.01))
PHASE2_INNER_K = int(os.environ.get("PHASE2_INNER_K", 1))
BATCH_SEQS     = int(os.environ.get("BATCH_SEQS", 4))
TTT_LR         = float(os.environ.get("TTT_LR", 0.1))
TTT_CHUNK      = int(os.environ.get("TTT_CHUNK", 1024))
WARMDOWN_FRAC  = float(os.environ.get("WARMDOWN_FRAC", 0.3))


def fomaml_step_joint(model, x_inner, y_inner, x_outer, y_outer, inner_lr, K=1):
    """FOMAML step where base model also gets outer gradients."""
    # 1. Detach prime MLP weights
    adapted = {}
    for name, p in model.prime_named_params():
        adapted[name] = p.detach().clone().requires_grad_(True)

    # 2. Inner loop: K SGD steps on prime weights only
    for _k in range(K):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inner_loss = model(x_inner, y_inner, prime_overrides=adapted)
        grads = torch.autograd.grad(inner_loss, list(adapted.values()))
        adapted = {n: p - inner_lr * g
                   for (n, p), g in zip(adapted.items(), grads)}

    # Mark for gradient capture
    for v in adapted.values():
        v.retain_grad()

    # 3. Outer loss — base model AND adapted prime weights both get gradients
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outer_loss = model(x_outer, y_outer, prime_overrides=adapted)
    outer_loss.backward()

    # 4. Copy adapted gradients to prime init params
    for name, p in model.prime_named_params():
        g = adapted[name].grad
        if g is not None:
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

    return outer_loss.item()


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

    # Load train data
    train_files = sorted(glob.glob(os.path.join(DATA_PATH, "fineweb_train_*.bin")))
    train_tokens = torch.cat([load_data_shard(Path(f)) for f in train_files]).contiguous()
    total_train = train_tokens.numel() - 1
    print(f"val tokens: {val_tokens.numel() - 1}")
    print(f"train tokens: {total_train} ({len(train_files)} shards)")

    # Build model with prime MLPs from scratch
    model = GPT_TTT(
        vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, model_dim=MODEL_DIM,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, mlp_mult=MLP_MULT,
        logit_softcap=LOGIT_SOFTCAP, rope_base=ROPE_BASE, qk_gain_init=QK_GAIN_INIT,
        bigram_vocab_size=BIGRAM_VOCAB, bigram_dim=BIGRAM_DIM,
        xsa_last_n=XSA_LAST_N, rope_dims=ROPE_DIMS, ln_scale=True,
        ve_enabled=VE_ENABLED, ve_dim=VE_DIM, ve_layers=VE_LAYERS,
        prime_rank=PRIME_RANK, prime_layers=PRIME_LAYERS,
    ).to(device).bfloat16()

    total_params = sum(p.numel() for p in model.parameters())
    prime_count = sum(p.numel() for _, p in model.prime_named_params())
    print(f"model params: {total_params} (prime: {prime_count})")

    # If checkpoint exists, load for Phase 1 skip
    if os.path.exists(CHECKPOINT) and PHASE1_STEPS == 0:
        print(f"loading checkpoint: {CHECKPOINT}")
        sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model_sd = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded += 1
        model.load_state_dict(model_sd)
        print(f"loaded {loaded}/{len(sd)} keys")
    elif PHASE1_STEPS > 0:
        # ── Phase 1: Standard training ──
        print(f"\n=== Phase 1: Standard training ({PHASE1_STEPS} steps) ===")

        # All params trainable
        all_params = list(model.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=PHASE1_LR, weight_decay=0.04)

        warmdown_start = int(PHASE1_STEPS * (1 - WARMDOWN_FRAC))
        t0 = time.perf_counter()
        model.train()

        for step in range(PHASE1_STEPS):
            # Cosine LR with warmdown
            if step < 20:
                lr_mult = (step + 1) / 20
            elif step < warmdown_start:
                lr_mult = 1.0
            else:
                progress = (step - warmdown_start) / (PHASE1_STEPS - warmdown_start)
                lr_mult = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = PHASE1_LR * lr_mult

            offset = torch.randint(0, total_train - BATCH_SEQS * SEQ_LEN, (1,)).item()
            data = train_tokens[offset:offset + BATCH_SEQS * SEQ_LEN + 1].to(device=device, dtype=torch.int64)
            x = data[:-1].reshape(BATCH_SEQS, SEQ_LEN)
            y = data[1:].reshape(BATCH_SEQS, SEQ_LEN)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            if step % 200 == 0 or step == PHASE1_STEPS - 1:
                elapsed = time.perf_counter() - t0
                ms = 1000.0 * elapsed / (step + 1)
                print(f"  phase1 [{step+1}/{PHASE1_STEPS}] loss={loss.item():.4f} "
                      f"lr={PHASE1_LR * lr_mult:.5f} ms/step={ms:.0f}")

        print(f"Phase 1 done in {time.perf_counter() - t0:.0f}s")
        torch.save(model.state_dict(), "phase1_model.pt")

    # ── Baseline eval ──
    print("\n=== Baseline eval (after Phase 1, before Phase 2) ===")
    for p in model.parameters():
        p.requires_grad_(True)
    eval_baseline(model, val_tokens, device,
                  base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    # ── Phase 2: FOMAML with ALL params ──
    if PHASE2_STEPS > 0:
        print(f"\n=== Phase 2: FOMAML joint ({PHASE2_STEPS} steps) ===")
        print(f"  base_lr={PHASE2_BASE_LR} prime_lr={PHASE2_PRIME_LR} inner_lr={PHASE2_INNER_LR}")

        # Separate param groups: base model at low LR, prime MLPs at high LR
        prime_names = set(n for n, _ in model.prime_named_params())
        prime_norm_names = set()
        for n, _ in model.prime_norms.named_parameters():
            prime_norm_names.add(f"prime_norms.{n}")

        base_params = []
        prime_params = []
        for n, p in model.named_parameters():
            is_prime = any(pn in n for pn in ["prime_ups", "prime_downs", "prime_norms"])
            if is_prime:
                prime_params.append(p)
            else:
                base_params.append(p)

        optimizer = torch.optim.AdamW([
            {"params": base_params, "lr": PHASE2_BASE_LR},
            {"params": prime_params, "lr": PHASE2_PRIME_LR},
        ], weight_decay=0.0)

        t0 = time.perf_counter()
        model.train()

        for step in range(PHASE2_STEPS):
            # Cosine decay for Phase 2
            progress = step / max(PHASE2_STEPS - 1, 1)
            lr_mult = 0.5 * (1 + math.cos(math.pi * progress))
            for i, pg in enumerate(optimizer.param_groups):
                base_lr = PHASE2_BASE_LR if i == 0 else PHASE2_PRIME_LR
                pg["lr"] = base_lr * max(lr_mult, 0.1)

            # Sample 2x data for inner + outer
            needed = BATCH_SEQS * 2 * SEQ_LEN + 1
            offset = torch.randint(0, total_train - needed, (1,)).item()
            data = train_tokens[offset:offset + needed].to(device=device, dtype=torch.int64)

            half = BATCH_SEQS * SEQ_LEN
            x_inner = data[:half].reshape(BATCH_SEQS, SEQ_LEN)
            y_inner = data[1:half + 1].reshape(BATCH_SEQS, SEQ_LEN)
            x_outer = data[half:2 * half].reshape(BATCH_SEQS, SEQ_LEN)
            y_outer = data[half + 1:2 * half + 1].reshape(BATCH_SEQS, SEQ_LEN)

            optimizer.zero_grad(set_to_none=True)
            loss_val = fomaml_step_joint(model, x_inner, y_inner, x_outer, y_outer,
                                         inner_lr=PHASE2_INNER_LR, K=PHASE2_INNER_K)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 100 == 0 or step == PHASE2_STEPS - 1:
                elapsed = time.perf_counter() - t0
                ms = 1000.0 * elapsed / (step + 1)
                print(f"  phase2 [{step+1}/{PHASE2_STEPS}] loss={loss_val:.4f} ms/step={ms:.0f}")

        print(f"Phase 2 done in {time.perf_counter() - t0:.0f}s")
        torch.save(model.state_dict(), "e2e_model.pt")

    # ── Post-Phase2 eval ──
    print("\n=== Post-Phase2 eval (no TTT) ===")
    eval_baseline(model, val_tokens, device,
                  base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    # ── TTT eval ──
    print(f"\n=== TTT eval (lr={TTT_LR}, chunk={TTT_CHUNK}) ===")

    # Reset prime MLPs to meta-learned init
    if os.path.exists("e2e_model.pt"):
        sd = torch.load("e2e_model.pt", map_location="cpu", weights_only=True)
        prime_keys = [k for k in sd if "prime_" in k]
        with torch.no_grad():
            for k in prime_keys:
                model.state_dict()[k].copy_(sd[k])
        print(f"reset prime MLPs ({len(prime_keys)} keys)")

    # Only prime params trainable for TTT
    for p in model.parameters():
        p.requires_grad_(False)
    for _, p in model.prime_named_params():
        p.requires_grad_(True)

    eval_ttt(model, val_tokens, device, sp, TTT_LR, TTT_CHUNK,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    # ── Also run naive TTT for comparison (reset primes to zero) ──
    print(f"\n=== Naive TTT comparison (zero-init primes, lr={TTT_LR}) ===")
    with torch.no_grad():
        for n, p in model.prime_named_params():
            if "down" in n:
                p.zero_()
            else:
                p.data = torch.nn.init.orthogonal_(
                    torch.empty_like(p, dtype=torch.float32)).to(p.dtype)

    eval_ttt(model, val_tokens, device, sp, TTT_LR, TTT_CHUNK,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)


if __name__ == "__main__":
    main()
