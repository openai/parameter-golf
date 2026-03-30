"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import train_gpt_lib.optim as optim_mod
from train_gpt_lib.config import Hyperparameters
from train_gpt_lib.comet_tracker import CometTracker
from train_gpt_lib.data import DistributedTokenLoader, build_sentencepiece_luts, load_validation_tokens
from train_gpt_lib.flash_attention import describe as flash_attn_describe, is_available as flash_attn_is_available
from train_gpt_lib.model import CastedLinear, GPT, restore_low_dim_params_to_fp32
from train_gpt_lib.optim import build_optimizers
from train_gpt_lib.serialization import save_and_validate_roundtrip
from train_gpt_lib.training import run_training


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    compile_muon = args.use_compile and args.use_compile_muon
    compile_model = args.use_compile and args.use_compile_model and (not args.use_mhc or args.allow_compile_with_mhc)
    if compile_muon:
        optim_mod.zeropower_via_newtonschulz5 = torch.compile(optim_mod.zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # explicit: keep kernel selection deterministic
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if master_process:
        cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<not set>")
        pythonhash = os.environ.get("PYTHONHASHSEED", "<not set>")
        log0(f"reproducibility: seed={args.seed} rank={rank} CUBLAS_WORKSPACE_CONFIG={cublas_cfg} PYTHONHASHSEED={pythonhash}", console=False)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    if args.experiment_name:
        log0(f"experiment_name:{args.experiment_name}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    flash_ver = args.flash_attn_version
    if flash_ver in (2, 3) and not flash_attn_is_available(flash_ver):
        raise RuntimeError(
            f"USE_FLASHATTENTION{flash_ver}=1 but flash_attn is not installed. "
            "Run: pip install flash-attn"
        )

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        hyper_conn_type=(args.mhc_type if args.use_mhc else "none"),
        hyper_conn_n=args.mhc_num_streams,
        flash_attn_version=flash_ver,
        mlp_proj_init=args.mlp_proj_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compile_fullgraph = args.compile_fullgraph
    compile_dynamic = args.compile_dynamic
    if compile_model and args.use_mhc:
        compile_fullgraph = False
        compile_dynamic = True
    train_model = (
        torch.compile(base_model, dynamic=compile_dynamic, fullgraph=compile_fullgraph) if compile_model else base_model
    )
    model: nn.Module = (
        DDP(
            train_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=args.use_mhc,
        )
        if distributed
        else train_model
    )

    optimizers, optimizer_muon = build_optimizers(args, base_model)
    n_params = sum(p.numel() for p in base_model.parameters())
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    log0(f"model_params:{n_params}")
    if args.use_compile and args.use_mhc and args.use_compile_model and not args.allow_compile_with_mhc:
        log0("compile:model_auto_disabled_for_mhc=True")
    if compile_model and args.use_mhc:
        log0("compile:model_mhc_mode=dynamic_fullgraph_off")
    log0(f"compile:global={args.use_compile} model={compile_model} muon={compile_muon}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"flash_attn:{flash_attn_describe()} active_version={flash_ver if flash_ver else 'torch_sdpa'}")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(f"mhc:enabled={args.use_mhc} type={args.mhc_type if args.use_mhc else 'none'} streams={args.mhc_num_streams}")
    log0("val_mode:full_precision_and_int8_quantized_dequantized_roundtrip")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    tracker = CometTracker(args, enabled=(master_process and args.comet_enable), log0=log0)

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    run_training(
        args=args,
        model=model,
        base_model=base_model,
        optimizer_muon=optimizer_muon,
        optimizers=optimizers,
        train_loader=train_loader,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        device=device,
        grad_accum_steps=grad_accum_steps,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        log0=log0,
        on_train_log=tracker.log_train,
        on_val_log=tracker.log_val,
    )

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    save_and_validate_roundtrip(
        args=args,
        base_model=base_model,
        model=model,
        rank=rank,
        world_size=world_size,
        device=device,
        grad_accum_steps=grad_accum_steps,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        code=code,
        log0=log0,
        master_process=master_process,
        on_model_size=tracker.log_model_size,
    )
    tracker.end()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
