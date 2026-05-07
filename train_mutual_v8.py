"""
v8: Mutual Learning — Flat GPT + Fractal GPT co-training
=========================================================
Two architecturally diverse models train on the same data, teaching each
other via soft label exchange. At eval, ensemble their logits.

Model A (flat):    11L/512d/8H/4KV/3xMLP — our proven architecture
Model B (fractal): 4 unique layers × 3 loops = 12 effective depth, 512d/8H/4KV/4xMLP

Training protocol (each step):
  1. Forward both models on same batch
  2. Model A loss = CE + alpha * KL(B_logits || A_logits)
  3. Model B loss = CE + alpha * KL(A_logits || B_logits)
  4. Update both

Size budget: A (~10MB int5) + B (~5MB int5) = ~15MB < 16MB

Usage:
  PYTHONPATH=flash-attention/hopper:$PYTHONPATH torchrun --nproc_per_node=8 train_mutual_v8.py
"""
from __future__ import annotations
import copy, glob, io, math, os, random, subprocess, sys, time, uuid, zlib
from pathlib import Path
try:
    import zstandard; _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn_interface import flash_attn_func as flash_attn_3_func

# Import the full GPT and all components from our v1 base
# We exec the v1 script to get all class definitions
_v1_path = os.path.join(os.path.dirname(__file__), "train_gpt_v1.py")
_v1_code = open(_v1_path).read()
_v1_module = {}
exec(compile(_v1_code.split("def main():")[0], _v1_path, "exec"), _v1_module)

# Pull out the classes we need
Hyperparameters = _v1_module["Hyperparameters"]
GPT = _v1_module["GPT"]
Block = _v1_module["Block"]
RMSNorm = _v1_module["RMSNorm"]
CastedLinear = _v1_module["CastedLinear"]
Rotary = _v1_module["Rotary"]
SmearGate = _v1_module["SmearGate"]
Muon = _v1_module["Muon"]
BigramHashEmbedding = _v1_module["BigramHashEmbedding"]
ValueEmbedding = _v1_module["ValueEmbedding"]
CONTROL_TENSOR_NAME_PATTERNS = _v1_module["CONTROL_TENSOR_NAME_PATTERNS"]
restore_low_dim_params_to_fp32 = _v1_module["restore_low_dim_params_to_fp32"]
DistributedTokenLoader = _v1_module["DistributedTokenLoader"]
eval_val = _v1_module["eval_val"]
eval_val_sliding = _v1_module["eval_val_sliding"]
mixed_quantize_int6 = _v1_module["mixed_quantize_int6"]
dequantize_mixed_int6 = _v1_module["dequantize_mixed_int6"]
quantize_float_tensor = _v1_module["quantize_float_tensor"]


class FractalGPT(nn.Module):
    """Fractal model: N unique layers × M loops with loop position embeddings."""
    def __init__(self, vocab_size, num_layers, num_loops, model_dim, num_heads, num_kv_heads,
                 mlp_mult, rope_base=10000.0, logit_softcap=30.0, qk_gain_init=1.5,
                 rope_dims=16, ln_scale=True):
        super().__init__()
        self.num_loops = num_loops
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, std=0.005)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                  qk_gain_init, layer_idx=i, ln_scale=ln_scale)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            hd = model_dim // num_heads
            for b in self.blocks:
                b.attn.rope_dims = rope_dims
                b.attn.rotary = Rotary(hd, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.loop_pos = nn.Parameter(torch.randn(num_loops, model_dim) * 0.01)
        self.final_norm = RMSNorm()
        # Tied embeddings
        self.lm_head_weight = self.tok_emb.weight  # reference, not separate param

    def forward(self, input_ids, target_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.weight.size(-1),))
        x0 = x
        for loop in range(self.num_loops):
            x = x + self.loop_pos[loop]
            for block in self.blocks:
                x = block(x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = self.logit_softcap * torch.tanh(F.linear(x, self.tok_emb.weight) / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

    def forward_logits(self, input_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.weight.size(-1),))
        x0 = x
        for loop in range(self.num_loops):
            x = x + self.loop_pos[loop]
            for block in self.blocks:
                x = block(x, x0)
        x = self.final_norm(x)
        return self.logit_softcap * torch.tanh(F.linear(x, self.tok_emb.weight) / self.logit_softcap)


def setup_optimizers(model, args, is_fractal=False):
    """Create Muon + AdamW optimizer set for a model."""
    block_named_params = list(model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params
                     if p.ndim == 2 and not any(pt in name for pt in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for name, p in block_named_params
                     if p.ndim < 2 or any(pt in name for pt in CONTROL_TENSOR_NAME_PATTERNS)]

    if is_fractal:
        scalar_params.append(model.loop_pos)
    else:
        if model.skip_weights.numel() > 0:
            scalar_params.append(model.skip_weights)
        scalar_params.append(model.smear.gate)
        if model.bigram is not None:
            scalar_params.append(model.bigram.scale)

    token_lr = args.tied_embed_lr
    tok_params = [{"params": [model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]

    if not is_fractal:
        if model.bigram is not None:
            tok_params.append({"params": [model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
            if model.bigram.proj is not None:
                matrix_params.append(model.bigram.proj.weight)
        if model.ve_shared is not None:
            tok_params.append({"params": [model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
            if model.ve_shared.proj is not None:
                matrix_params.append(model.ve_shared.proj.weight)
            scalar_params.append(model.ve_shared.scale)
            for s in model.ve_layer_scales:
                scalar_params.append(s)

    opt_tok = torch.optim.AdamW(tok_params, betas=(args.beta1, args.beta2), eps=args.adam_eps,
                                 weight_decay=args.adam_wd, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for group in opt_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    return [opt_tok, opt_muon, opt_scalar]


def main():
    args = Hyperparameters()
    # Mutual learning params
    mutual_alpha = float(os.environ.get("MUTUAL_ALPHA", 0.3))
    mutual_temp = float(os.environ.get("MUTUAL_TEMP", 2.0))
    fractal_layers = int(os.environ.get("FRACTAL_LAYERS", 4))
    fractal_loops = int(os.environ.get("FRACTAL_LOOPS", 3))
    fractal_mlp = float(os.environ.get("FRACTAL_MLP_MULT", 4.0))

    distributed = int(os.environ.get("RANK", -1)) != -1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = rank = 0
        world_size = 1
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    master_process = (rank == 0)
    def log0(msg):
        if master_process:
            print(msg, flush=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer and val data (same as v1)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")

    val_files = sorted(glob.glob(args.val_files))
    val_tokens_list = []
    for vf in val_files:
        header = np.fromfile(vf, dtype="<i4", count=256)
        n = int(header[2])
        val_tokens_list.append(torch.from_numpy(np.fromfile(vf, dtype="<u2", offset=1024, count=n).astype(np.int32)))
    val_tokens = torch.cat(val_tokens_list).to(device)
    log0(f"val_loader:tokens:{val_tokens.numel()}")

    # Build BPB lookup tables
    base_bytes_lut = torch.zeros(args.vocab_size, device=device, dtype=torch.float32)
    has_leading_space_lut = torch.zeros(args.vocab_size, device=device, dtype=torch.bool)
    is_boundary_token_lut = torch.zeros(args.vocab_size, device=device, dtype=torch.bool)
    for tid in range(args.vocab_size):
        piece = sp.IdToPiece(tid)
        raw = piece.replace("\u2581", " ").encode("utf-8")
        base_bytes_lut[tid] = max(len(raw), 1)
        if piece.startswith("\u2581"):
            has_leading_space_lut[tid] = True
            is_boundary_token_lut[tid] = True
        if tid in (sp.bos_id(), sp.eos_id()) or tid == 0:
            is_boundary_token_lut[tid] = True

    grad_accum_steps = max(1, args.train_batch_tokens // (args.train_seq_len * world_size))
    grad_scale = 1.0 / grad_accum_steps

    # === Model A: Flat GPT (our SOTA) ===
    model_a = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in model_a.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model_a)

    # === Model B: Fractal GPT ===
    model_b = FractalGPT(
        vocab_size=args.vocab_size, num_layers=fractal_layers, num_loops=fractal_loops,
        model_dim=args.model_dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        mlp_mult=fractal_mlp, rope_base=args.rope_base, logit_softcap=args.logit_softcap,
        qk_gain_init=args.qk_gain_init, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    ).to(device).bfloat16()
    for m in model_b.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model_b)

    n_a = sum(p.numel() for p in model_a.parameters())
    n_b = sum(p.numel() for p in model_b.parameters())
    log0(f"model_a_params:{n_a} model_b_params:{n_b} total:{n_a + n_b}")

    # Compile both
    compiled_a = torch.compile(model_a, dynamic=False, fullgraph=True)
    compiled_b = torch.compile(model_b, dynamic=False, fullgraph=True)

    if distributed:
        ddp_a = DDP(compiled_a, device_ids=[local_rank], broadcast_buffers=False)
        ddp_b = DDP(compiled_b, device_ids=[local_rank], broadcast_buffers=False)
    else:
        ddp_a, ddp_b = compiled_a, compiled_b

    opts_a = setup_optimizers(model_a, args, is_fractal=False)
    opts_b = setup_optimizers(model_b, args, is_fractal=True)

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # EMA for both models
    ema_a = {name: t.detach().float().clone() for name, t in model_a.state_dict().items()}
    ema_b = {name: t.detach().float().clone() for name, t in model_b.state_dict().items()}
    ema_decay = 0.997

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        sms = elapsed_ms / max(step, 1)
        wms = args.warmdown_iters * sms
        rms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

    T = mutual_temp
    alpha = mutual_alpha

    log0(f"mutual_learning: alpha={alpha} temp={T} fractal={fractal_layers}x{fractal_loops} mlp={fractal_mlp}x")
    log0(f"seed:{args.seed}")

    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Set LR for both models
        for opts in [opts_a, opts_b]:
            for opt in opts:
                for group in opt.param_groups:
                    group["lr"] = group["base_lr"] * scale

        # Get batch
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)

        # Forward both models
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            # Model A: CE + KL from B
            logits_a = model_a.forward_logits(x)
            ce_a = F.cross_entropy(logits_a.reshape(-1, logits_a.size(-1)).float(), y.reshape(-1))
            with torch.no_grad():
                logits_b_detached = model_b.forward_logits(x)
            kl_a = F.kl_div(
                F.log_softmax(logits_a.float() / T, dim=-1),
                F.softmax(logits_b_detached.float() / T, dim=-1),
                reduction="batchmean") * (T * T)
            loss_a = (1.0 - alpha) * ce_a + alpha * kl_a

            # Model B: CE + KL from A
            logits_b = model_b.forward_logits(x)
            ce_b = F.cross_entropy(logits_b.reshape(-1, logits_b.size(-1)).float(), y.reshape(-1))
            with torch.no_grad():
                logits_a_detached = logits_a.detach()
            kl_b = F.kl_div(
                F.log_softmax(logits_b.float() / T, dim=-1),
                F.softmax(logits_a_detached.float() / T, dim=-1),
                reduction="batchmean") * (T * T)
            loss_b = (1.0 - alpha) * ce_b + alpha * kl_b

        # Backward + update A
        for opt in opts_a:
            opt.zero_grad(set_to_none=True)
        loss_a.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), args.grad_clip_norm)
        for opt in opts_a:
            opt.step()

        # Backward + update B
        for opt in opts_b:
            opt.zero_grad(set_to_none=True)
        loss_b.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), args.grad_clip_norm)
        for opt in opts_b:
            opt.step()

        # EMA update both
        with torch.no_grad():
            for name, t in model_a.state_dict().items():
                ema_a[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
            for name, t in model_b.state_dict().items():
                ema_b[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step} ce_a:{ce_a.item():.4f} ce_b:{ce_b.item():.4f} "
                 f"kl_a:{kl_a.item():.4f} kl_b:{kl_b.item():.4f} "
                 f"time:{approx_ms:.0f}ms avg:{approx_ms/step:.1f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"training_done steps:{step} time:{approx_ms:.0f}ms")

    # Apply EMA to both
    log0("ema:applying to both models")
    cs_a = model_a.state_dict()
    model_a.load_state_dict({n: t.to(dtype=cs_a[n].dtype) for n, t in ema_a.items()}, strict=True)
    cs_b = model_b.state_dict()
    model_b.load_state_dict({n: t.to(dtype=cs_b[n].dtype) for n, t in ema_b.items()}, strict=True)

    # Quantize both models
    sd_a = {k: v.detach().cpu() for k, v in model_a.state_dict().items() if "mtp_heads" not in k}
    sd_b = {k: v.detach().cpu() for k, v in model_b.state_dict().items()}

    q_a, m_a = mixed_quantize_int6(sd_a, {"mlp", "attn"})
    q_b, m_b = mixed_quantize_int6(sd_b, {"mlp", "attn"})

    # Save combined artifact
    combined = {"a_w": q_a, "a_m": m_a, "b_w": q_b, "b_m": m_b,
                "fractal_cfg": {"layers": fractal_layers, "loops": fractal_loops, "mlp": fractal_mlp}}
    buf = io.BytesIO()
    torch.save(combined, buf)
    raw = buf.getvalue()
    blob = zstandard.ZstdCompressor(level=22).compress(raw) if _COMPRESSOR == "zstd" else zlib.compress(raw, 9)

    if master_process:
        with open("final_ensemble.ptz", "wb") as f:
            f.write(blob)
        code_bytes = len(open(__file__).read().encode("utf-8"))
        log0(f"ensemble_artifact: {len(blob)} bytes code:{code_bytes} total:{len(blob)+code_bytes}")

    # Eval: ensemble logits
    model_a.eval()
    model_b.eval()

    log0("eval:ensemble sliding window")
    # Simple ensemble eval — average logits from both models
    seq_len = args.eval_seq_len or args.train_seq_len
    stride = args.eval_stride
    total_tokens = val_tokens.numel() - 1
    window_starts = list(range(0, total_tokens - seq_len + 1, stride))
    my_s = (len(window_starts) * rank) // world_size
    my_e = (len(window_starts) * (rank + 1)) // world_size

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model_a.eval()
    model_b.eval()
    t_eval = time.perf_counter()
    with torch.inference_mode():
        for wi in range(my_s, my_e, 32):
            batch_ws = list(range(wi, min(wi + 32, my_e)))
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            for i, wsi in enumerate(batch_ws):
                ws = window_starts[wsi]
                ct = val_tokens[ws:ws + seq_len + 1].to(dtype=torch.int64, device=device)
                x_batch[i] = ct[:-1]
                y_batch[i] = ct[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_a = model_a.forward_logits(x_batch)
                logits_b = model_b.forward_logits(x_batch)
                # Ensemble: average logits
                logits_ens = 0.5 * (logits_a + logits_b)
            nll = F.cross_entropy(
                logits_ens.reshape(-1, logits_ens.size(-1)).float(),
                y_batch.reshape(-1), reduction="none"
            ).reshape(bsz, seq_len)
            for i, wsi in enumerate(batch_ws):
                ws = window_starts[wsi]
                s = 0 if ws == 0 else max(seq_len - stride, 0)
                scored = nll[i, s:seq_len].to(torch.float64)
                loss_sum += scored.sum()
                token_count += float(seq_len - s)
                tgt = y_batch[i, s:seq_len]
                prev = x_batch[i, s:seq_len]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        for t in [loss_sum, token_count, byte_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    log0(f"ensemble_sliding val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
         f"stride:{stride} time:{1000*(time.perf_counter()-t_eval):.0f}ms")
    log0(f"ensemble_sliding_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
