"""Apply TENT patch to train_gpt_pr398.py -> train_gpt_ours.py"""

with open("train_gpt_ours.py", "r") as f:
    code = f.read()

# 1. Add TENT hyperparameters
old = '    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))'
new = '''    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_tent_enabled = bool(int(os.environ.get("TTT_TENT_ENABLED", "1")))
    ttt_tent_epochs = int(os.environ.get("TTT_TENT_EPOCHS", 30))
    ttt_tent_lr = float(os.environ.get("TTT_TENT_LR", 0.01))'''
code = code.replace(old, new)

# 2. Add tent_norm_recalib function before INT6 section
tent_fn = '''
NORM_PARAM_PATTERNS = ("attn_scale", "mlp_scale", "q_gain", "skip_weight")

def tent_norm_recalib(args, base_model, device, val_tokens, rank=0, world_size=1, log_fn=None):
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = args.ttt_batch_seqs
    for p in base_model.parameters():
        p.requires_grad_(False)
    norm_params = []
    for name, p in base_model.named_parameters():
        if any(k in name for k in NORM_PARAM_PATTERNS):
            p.requires_grad_(True)
            norm_params.append(p)
    if log_fn:
        log_fn(f"tent:start params={sum(p.numel() for p in norm_params)} epochs={args.ttt_tent_epochs}")
    optimizer = torch.optim.Adam(norm_params, lr=args.ttt_tent_lr)
    my_start = (total_seqs * rank) // world_size
    my_end = (total_seqs * (rank + 1)) // world_size
    base_model.train()
    t0 = time.perf_counter()
    for epoch in range(args.ttt_tent_epochs):
        for bs in range(my_start, my_end, batch_seqs):
            be = min(bs + batch_seqs, my_end)
            local = val_tokens[bs*seq_len:be*seq_len+1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            loss.backward()
            if world_size > 1:
                for p in norm_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            optimizer.step()
        if log_fn and (epoch+1) % max(1, args.ttt_tent_epochs//5) == 0:
            log_fn(f"tent_epoch:{epoch+1}/{args.ttt_tent_epochs} time:{time.perf_counter()-t0:.1f}s")
    for p in base_model.parameters():
        p.requires_grad_(True)
    if log_fn:
        log_fn(f"tent:done elapsed={time.perf_counter()-t0:.1f}s")

'''
marker = "# -----------------------------\n# INT6 MIXED QUANTIZATION"
code = code.replace(marker, tent_fn + marker)

# 3. Insert TENT call before ttt_adapt
old_call = "ttt_adapt(args, base_model, device, val_tokens, rank, world_size, log)"
new_call = """if args.ttt_tent_enabled:
            tent_norm_recalib(args, base_model, device, val_tokens, rank, world_size, log)
        ttt_adapt(args, base_model, device, val_tokens, rank, world_size, log)"""
code = code.replace(old_call, new_call)

with open("train_gpt_ours.py", "w") as f:
    f.write(code)
print("Patch applied OK")
