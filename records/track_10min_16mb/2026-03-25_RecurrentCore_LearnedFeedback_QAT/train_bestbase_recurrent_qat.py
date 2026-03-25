"""Script 1: Recurrent core with full-rollout QAT, **no** error feedback.

This is the baseline recurrence experiment.  It answers: how much of the
recurrence-vs-quantization problem disappears if we simply train the real
quantised recurrent rollout starting from the current best recipe?

Forward pass:
  h_0 = Stem(x)
  h_{k+1} = f_{W_q}(h_k)          (no correction term)
  logits = LMHead(Tail(h_K))
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F

from model_recurrent_bestbase import (
    CastedLinear, RecurrentGPT, restore_low_dim_params_to_fp32,
)
from stability import RecurrentStabilizer
from train_utils_recurrent import (
    Hyperparameters, Muon, DistributedTokenLoader,
    build_sentencepiece_luts, load_validation_tokens,
    eval_val, eval_val_sliding, export_and_roundtrip,
    build_model, build_optimizers,
    add_common_args, apply_cli_overrides,
)
from ttt_recurrent import eval_val_sliding_ttt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recurrent bestbase with QAT only (no feedback)")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    args = apply_cli_overrides(args, cli)

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

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = (args.eval_seq_len
                              if args.eval_seq_len > 0 else args.train_seq_len)
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = build_model(args, device)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} stem:{base_model.num_stem} "
         f"core:{base_model.num_core} tail:{base_model.num_tail} "
         f"passes:{base_model.num_passes}")

    stabilizer = RecurrentStabilizer(
        clip_hidden=cli.clip_hidden, clip_value=cli.clip_value,
        jacobian_proxy_weight=cli.jacobian_proxy_weight)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = compiled_model

    optimizers, replicated_params = build_optimizers(base_model, args)
    optimizer_muon = optimizers[1]

    train_loader = DistributedTokenLoader(
        args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (1000.0 * args.max_wallclock_seconds
                        if args.max_wallclock_seconds > 0 else None)

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            wd_start = max(args.iterations - args.warmdown_iters, 0)
            if wd_start <= step < args.iterations:
                return max((args.iterations - step)
                           / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            return remaining_ms / max(warmdown_ms, 1e-9)
        return 1.0

    # warmup
    if args.warmup_steps > 0:
        init_state = {n: t.detach().cpu().clone()
                      for n, t in base_model.state_dict().items()}
        init_opts = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for _ in range(grad_accum_steps):
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len,
                    grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y, stabilizer=stabilizer)
                (loss * grad_scale).backward()
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(init_state, strict=True)
        for opt, st in zip(optimizers, init_opts, strict=True):
            opt.load_state_dict(st)
        zero_grad_all()
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device)

    ema_state = {n: t.detach().float().clone()
                 for n, t in base_model.state_dict().items()}
    ema_decay = 0.997
    swa_state = None
    swa_count = 0
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = (step == args.iterations
                     or (stop_after_step is not None
                         and step >= stop_after_step))
        should_validate = (last_step
                           or (args.val_loss_every > 0
                               and step % args.val_loss_every == 0))

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut,
                is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} "
                 f"val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms "
                 f"step_avg:{training_time_ms/max(step,1):.2f}ms")
            stabilizer.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        if (args.late_qat_threshold > 0 and scale < args.late_qat_threshold
                and not CastedLinear._qat_enabled):
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step}")

        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y, stabilizer=stabilizer)
            train_loss += loss.detach()
            (loss * grad_scale).backward()

        train_loss /= grad_accum_steps

        frac = (min(step / args.muon_momentum_warmup_steps, 1.0)
                if args.muon_momentum_warmup_steps > 0 else 1.0)
        for group in optimizer_muon.param_groups:
            group["momentum"] = ((1 - frac) * args.muon_momentum_warmup_start
                                 + frac * args.muon_momentum)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group.get("base_lr", group["lr"]) * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                base_model.parameters(), args.grad_clip_norm)

        optimizer_muon.launch_reduce_scatters()
        if distributed:
            for p in replicated_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        for opt in optimizers:
            if opt is not optimizer_muon:
                opt.step()
        optimizer_muon.step()
        zero_grad_all()

        with torch.no_grad():
            for n, t in base_model.state_dict().items():
                ema_state[n].mul_(ema_decay).add_(
                    t.detach().float(), alpha=1.0 - ema_decay)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {n: t.detach().cpu().clone()
                             for n, t in base_model.state_dict().items()}
                swa_count = 1
            else:
                for n, t in base_model.state_dict().items():
                    swa_state[n] += t.detach().cpu()
                swa_count += 1

        should_log = (args.train_log_every > 0
                      and (step <= 10 or step % args.train_log_every == 0))
        if should_log:
            log0(f"step:{step}/{args.iterations} "
                 f"train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms "
                 f"step_avg:{approx_ms/step:.2f}ms")

        reached_cap = (max_wallclock_ms is not None
                       and approx_ms >= max_wallclock_ms)
        if distributed and max_wallclock_ms is not None:
            cap_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    # apply EMA
    log0("ema:applying EMA weights")
    sd = base_model.state_dict()
    for n, t in sd.items():
        sd[n] = ema_state[n].to(dtype=t.dtype)
    base_model.load_state_dict(sd, strict=True)

    log0(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")

    # diagnostic
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_loss, diag_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut,
        is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"DIAGNOSTIC post_ema val_loss:{diag_loss:.4f} "
         f"val_bpb:{diag_bpb:.4f}")

    # export
    eval_model = export_and_roundtrip(
        base_model, args, log0, device, rank, world_size,
        grad_accum_steps, val_tokens, base_bytes_lut,
        has_leading_space_lut, is_boundary_token_lut)

    # TTT
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut,
            is_boundary_token_lut, stride=args.eval_stride, log0=log0,
            ttt_regime=cli.ttt_regime,
            ttt_recurrent_lr_scale=cli.ttt_recurrent_lr_scale)
        torch.cuda.synchronize()
        log0(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
             f"eval_time:{1000*(time.perf_counter()-t_ttt):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
