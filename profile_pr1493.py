"""
Short profiling run of PR #1493 training loop.
Runs real 8-GPU DDP training for a few steps, matching the actual workload.
"""
import os
import sys
import time

import torch
import torch.distributed as dist

from train_pr1493 import (
    Hyperparameters, GPT, ShuffledSequenceLoader, Optimizers,
    restore_fp32_params, set_logging_hparams, log,
)

PROFILE_STEPS = 3

def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.optimize_ddp = False

    h = Hyperparameters()
    set_logging_hparams(h)

    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)

    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(compiled_model, device_ids=[local_rank],
                    broadcast_buffers=False)
    else:
        model = compiled_model

    optimizers = Optimizers(h, base_model)
    loader = ShuffledSequenceLoader(h, device)

    # Warmup: compile + 1 step (unprofilied warm cache)
    model.train()
    optimizers.zero_grad_all()
    for micro in range(h.grad_accum_steps):
        if distributed:
            model.require_backward_grad_sync = (micro == h.grad_accum_steps - 1)
        x, y = loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        (loss / h.grad_accum_steps).backward()
    optimizers.step()
    torch.cuda.synchronize()
    if h.is_main_process:
        log(f"warmup done, starting {PROFILE_STEPS} profiled steps")

    # Profiled steps
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(PROFILE_STEPS):
        optimizers.zero_grad_all()
        for micro in range(h.grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = (micro == h.grad_accum_steps - 1)
            x, y = loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss / h.grad_accum_steps).backward()
        optimizers.step()
        torch.cuda.synchronize()
        if h.is_main_process:
            log(f"profile step {step} done, loss={loss.item():.4f}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if h.is_main_process:
        tok_per_sec = PROFILE_STEPS * h.train_batch_tokens / elapsed
        log(f"profiling done: {PROFILE_STEPS} steps in {elapsed:.2f}s, "
            f"{tok_per_sec:.0f} tok/s")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
