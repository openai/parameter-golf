from __future__ import annotations
import os
import sys
import time
import uuid
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import sentencepiece as spm

# Modular Imports
from model import GPT, CastedLinear, RelaxedLinear
from data_utils import DistributedTokenLoader
from optimizer_utils import Muon
from eval_utils import eval_val, build_sentencepiece_luts, load_validation_tokens
from quant_utils import quantize_state_dict_int8, dequantize_state_dict_int8

# --- LOGGING ---
_LOG_FILE: str | None = None
_MASTER_PROCESS: bool = True

def log0(msg: str, console: bool = True) -> None:
    if not _MASTER_PROCESS:
        return
    if console:
        print(msg)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            print(msg, file=f)

# --- HYPERPARAMETERS ---
class Hyperparameters:
    # NOTE: When running from within a records/* folder, repo-root data lives at ../../../data/...
    # Users can always override with DATA_PATH/TOKENIZER_PATH env vars.
    data_path = os.environ.get("DATA_PATH", "../../../data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "../../../data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 65_536))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 50))
    train_batch_tokens = 524_288  # NON-NEGOTIABLE COMPETITION STANDARD
    micro_batch_tokens = 32_768   # 128 samples * 256 seq
    train_seq_len = 256
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 1024))
    num_heads = int(os.environ.get("NUM_HEADS", 16))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 10.0))

    matrix_lr = float(os.environ.get("MATRIX_LR", 0.012))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.015))
    embed_lr = float(os.environ.get("EMBED_LR", 0.7))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.06))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 100))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    
    lora_rank = int(os.environ.get("LORA_RANK", 16))
    num_steps = int(os.environ.get("NUM_STEPS", 12))

CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights").split(",")

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

def main() -> None:
    print("[debug] main() started")
    args = Hyperparameters()
    
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and os.environ.get("FORCE_SINGLE_GPU") != "1"
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # HARDCODED SUCCESS: 16x Accumulation for 524k tokens
    grad_accum_steps = 16
    grad_scale = 1.0 / grad_accum_steps
    
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = f"logs/{args.run_id}.txt" if master_process else None
    global _LOG_FILE, _MASTER_PROCESS
    _MASTER_PROCESS, _LOG_FILE = master_process, logfile
    if master_process: os.makedirs("logs", exist_ok=True)
    print("[debug] logging initialized")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    print("[debug] loading validation tokens...")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    print("[debug] initializing base_model...")
    base_model = GPT(
        vocab_size=args.vocab_size, num_steps=args.num_steps, model_dim=args.model_dim, 
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult, 
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std, 
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init, 
        lora_rank=args.lora_rank
    ).to(device).bfloat16()
    
    # Model Compilation: Block-level is already compiled in model.py.
    # On Windows, wrapping the whole model AGAIN for 12 steps often causes OOM/paging.
    # We rely on gradient checkpointing + block-level JIT for maximum stability.
    model = base_model
        
    if distributed:
        print("[debug] wrapping in DDP...")
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    # Optimizer splitting
    print("[debug] splitting params for optimizers...")
    matrix_params, scalar_params = [], []
    for name, p in base_model.named_parameters():
        if "tok_emb" in name or (base_model.lm_head is not None and "lm_head" in name): continue
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS) and "step_embeddings" not in name:
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    
    # MEGA-KERNEL OPTIMIZATION: AdamW with Aggressive Regularization (0.1 WD)
    print("[debug] initializing optimizers...")
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "target_lr": token_lr}], 
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True, weight_decay=0.1
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "target_lr": args.scalar_lr}], 
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True, weight_decay=0.1
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizers.insert(1, torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "target_lr": args.head_lr}], 
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True, weight_decay=0.1
        ))

    print("[debug] initializing data loader...")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    training_time_ms = 0.0
    t0 = time.perf_counter()
    step = 0
    print("[debug] initializing model EMA...")
    model_ema = {n.replace("module.", ""): p.clone().detach() for n, p in model.named_parameters()}

    print("[debug] entering training loop...")
    global_start_time = time.perf_counter()

    while True:
        elapsed_sec = time.perf_counter() - global_start_time
        last_step = step >= args.iterations or elapsed_sec >= args.max_wallclock_seconds
        if step > 0 and args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            torch.cuda.synchronize()
            original_params = {n: p.data.clone() for n, p in base_model.named_parameters()}
            for n, p in base_model.named_parameters():
                if n in model_ema: p.data.copy_(model_ema[n])
            # ELITE STRIDE: Mid-run speed, Final-step precision
            eval_stride = 64 if last_step else args.train_seq_len
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, max_steps=50, stride=eval_stride, ttt_lr=4e-4)
            for n, p in base_model.named_parameters(): p.data.copy_(original_params[n])
            log0(f"step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms{' [FINAL STRIDE 64]' if last_step else ''}")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        if last_step: break
        
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.micro_batch_tokens, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(x, y)
            (loss * grad_scale).backward()
            step_loss += loss.item() * grad_scale
            
        # Optimizer Logic: Elite Standard 14.0 (Cosine Wallclock Scheduler)
        # 1. 100-Step Maturity Ramp (Cold Start)
        # 2. 600-Second Cosine Decay (Generalization boost)
        import math
        elapsed_sec = time.perf_counter() - global_start_time
        global_ramp = min(step / 20, 1.0)
        global_decay = 0.5 * (1.0 + math.cos(min(elapsed_sec / 600.0, 1.0) * math.pi))
        total_scale = global_ramp * global_decay
        
        # Shorten Muon warmup to match the 20-step ramp
        frac = min(step / 20, 1.0)
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        
        # ELITE FIX: Universal Gradient Averaging
        # We divide ALL gradients by the number of recursive steps (12)
        # to stabilize the residual stream and prevent divergence.
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(args.num_steps)
            
        # Standard Elite Fix: Gradient Clipping (1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            
        # Update Muon groups with maturity ramp and wallclock decay
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
            group["lr"] = args.matrix_lr * total_scale
            
        # Update AdamW groups (Scalars and Embeddings)
        for opt in optimizers:
            if opt != optimizer_muon:
                for group in opt.param_groups:
                    group["lr"] = group["target_lr"] * total_scale
            opt.step()
        with torch.no_grad():
            for n, p in model.named_parameters():
                ema_n = n.replace("module.", "")
                if ema_n in model_ema: model_ema[ema_n].mul_(0.99).add_(p.data, alpha=0.01)
        
        # Add basic per-step logging with Data Transparency
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        training_time_ms += dt
        t0 = time.perf_counter()
        shard = train_loader.stream.file_idx
        pos = train_loader.stream.pos
        log0(f"step:{step} loss:{step_loss:.4f} dt:{dt:.2f}ms d:sh{shard}p{pos}")
        
        step += 1

if __name__ == "__main__":
    main()
