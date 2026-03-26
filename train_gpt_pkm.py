"""
Modified baseline train_gpt.py with PKM injection at the middle layer.
This tests whether PKM improves BPB within the 16MB budget.
"""
import sys
sys.path.insert(0, '.')

# Monkey-patch: inject PKM into the GPT model
from train_gpt import *
from pkm_layer import ProductKeyMemory

class GPT_PKM(GPT):
    """GPT with Product Key Memory injected after encoder layers."""
    def __init__(self, *args, pkm_subkeys=64, pkm_topk=8, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.tok_emb.weight.shape[1]
        self.pkm = ProductKeyMemory(
            d_model=dim, n_subkeys=pkm_subkeys, d_key=32, top_k=pkm_topk
        )
        print(f"PKM added: {self.pkm.n_values} slots, {self.pkm.param_count():,} params, ~{self.pkm.memory_mb(1):.1f}MB int8")

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)

        # PKM injection: right between encoder and decoder
        x = self.pkm(x)

        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

# Override model creation in main
original_main = main

def patched_main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a") as f: print(msg, file=f)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    # === USE PKM MODEL ===
    base_model = GPT_PKM(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        pkm_subkeys=64, pkm_topk=8,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear): module.float()
    restore_low_dim_params_to_fp32(base_model)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} (includes PKM)")

    # Use torch.compile but skip PKM (sparse ops don't compile well)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer: PKM values get their own Adam group
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n,p in block_named_params if p.ndim==2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n,p in block_named_params if p.ndim<2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)

    # PKM parameters
    pkm_params = list(base_model.pkm.parameters())
    pkm_matrix = [p for p in pkm_params if p.ndim == 2]
    pkm_other = [p for p in pkm_params if p.ndim < 2]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam([{"params":[base_model.tok_emb.weight],"lr":token_lr,"base_lr":token_lr}],
                                      betas=(args.beta1,args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in optimizer_muon.param_groups: g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam([{"params":scalar_params+pkm_other,"lr":args.scalar_lr,"base_lr":args.scalar_lr}],
                                         betas=(args.beta1,args.beta2), eps=args.adam_eps, fused=True)
    # PKM values get higher LR (they need to learn fast)
    optimizer_pkm = torch.optim.Adam([{"params":pkm_matrix,"lr":0.01,"base_lr":0.01}],
                                      betas=(0.9,0.99), eps=1e-8, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_pkm]

    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0*args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations-args.warmdown_iters,0)
            return max((args.iterations-step)/max(args.warmdown_iters,1),0.0) if ws<=step<args.iterations else 1.0
        step_ms = elapsed_ms/max(step,1)
        wd_ms = args.warmdown_iters*step_ms
        rem = max(max_wallclock_ms-elapsed_ms,0.0)
        return rem/max(wd_ms,1e-9) if rem<=wd_ms else 1.0

    # Warmup
    if args.warmup_steps > 0:
        init_state = {n:t.detach().cpu().clone() for n,t in base_model.state_dict().items()}
        init_opts = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms==grad_accum_steps-1
                x,y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16): loss = model(x,y)
                (loss*grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
        base_model.load_state_dict(init_state, strict=True)
        for o,s in zip(optimizers, init_opts): o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Main training loop
    training_time_ms = 0.0; stop_after_step = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step==args.iterations or (stop_after_step is not None and step>=stop_after_step)
        if last_step or (args.val_loss_every>0 and step%args.val_loss_every==0):
            torch.cuda.synchronize(); training_time_ms += 1000.0*(time.perf_counter()-t0)
            vl,vb = eval_val(args,model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_time_ms:.0f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step<args.iterations:
                log0(f"stopping_early: wallclock_cap step:{step}")
            break

        elapsed = training_time_ms + 1000.0*(time.perf_counter()-t0)
        scale = lr_mul(step, elapsed)
        zero_grad_all()
        train_loss = torch.zeros((),device=device)
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms==grad_accum_steps-1
            x,y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): loss = model(x,y)
            train_loss += loss.detach(); (loss*grad_scale).backward()
        train_loss /= grad_accum_steps

        for o in optimizers:
            for g in o.param_groups: g["lr"] = g["base_lr"]*scale
        for o in optimizers: o.step()
        zero_grad_all()
        step += 1

        approx_ms = training_time_ms + 1000.0*(time.perf_counter()-t0)
        if step<=10 or step%200==0:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed:
            t = torch.tensor(int(reached_cap),device=device); dist.all_reduce(t,op=dist.ReduceOp.MAX); reached_cap=bool(t.item())
        if stop_after_step is None and reached_cap: stop_after_step = step

    # Quantize and save
    log0(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")
    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO(); torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    if master_process:
        with open("final_model_pkm.int8.ptz","wb") as f: f.write(quant_blob)
        sz = os.path.getsize("final_model_pkm.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Artifact: {sz} bytes model + {code_bytes} bytes code = {sz+code_bytes} total")
        log0(f"Under 16MB: {'YES' if sz+code_bytes < 16_000_000 else 'NO - TOO BIG'}")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    patched_main()
