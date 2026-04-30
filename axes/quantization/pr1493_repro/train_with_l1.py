"""PR-1493 training with proximal L1 regularization during warmdown.

Adds soft-thresholding after each optimizer step during the last L1_FRAC
of training. This is the proximal gradient operator for L1, which pushes
weights TO exactly zero (unlike L2/WD which pushes TOWARD zero).

    soft_threshold(w, t) = sign(w) * max(|w| - t, 0)

The threshold is FIXED (not scaled by LR), so L1 pressure stays active
even as LR→0 — unlike WD which vanishes at the end.

Usage on RunPod 8×H100:
    L1_COEFF=0.0001 L1_START_FRAC=0.35 \
    BUNDLE_DIR=./bundle_l1 \
    torchrun --standalone --nproc_per_node=8 train_with_l1.py

Env vars:
    L1_COEFF          soft threshold per step (default 0.0001)
    L1_START_FRAC     training frac to start L1 (default 0.35 = last 65%)
    BUNDLE_DIR        where to save bundle
"""
import os, sys, math, time, copy
from pathlib import Path

os.environ.setdefault("BUNDLE_DIR", "bundle_l1")
os.environ.setdefault("TTT_ENABLED", "0")
os.environ.setdefault("ETLB_ENABLED", "0")

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_save_bundle as tsb


def train_model_with_l1(h, device, val_data):
    l1_coeff = float(os.environ.get("L1_COEFF", "0.0001"))
    l1_start_frac = float(os.environ.get("L1_START_FRAC", "0.35"))
    wd_final = float(os.environ.get("WD_FINAL", str(h.muon_wd)))  # default = no ramp
    wd_taper_start = float(os.environ.get("WD_TAPER_START_FRAC", str(1.0 - h.warmdown_frac)))
    tsb.log(f"l1 config: coeff={l1_coeff} start_frac={l1_start_frac}")
    tsb.log(f"wd taper: {h.muon_wd} -> {wd_final} starting at frac={wd_taper_start}, min_lr={h.min_lr}")

    base_model = tsb.GPT(h).to(device).bfloat16()
    tsb.restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model

    optimizers = tsb.Optimizers(h, base_model)
    train_loader = tsb.ShuffledSequenceLoader(h, device)
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = h.ema_decay

    max_wallclock_ms = None
    if h.max_wallclock_seconds > 0:
        gptq_reserve_ms = h.gptq_reserve_seconds * 1000
        max_wallclock_ms = h.max_wallclock_seconds * 1000 - gptq_reserve_ms
        tsb.log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-9)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac < (1.0 - h.warmdown_frac):
            return 1.0
        return max(h.min_lr, 1.0 - (frac - (1.0 - h.warmdown_frac)) / (h.warmdown_frac + 1e-9))

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    # Warmup
    if h.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                tsb.log(f"warmup_step: {warmup_step + 1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            tsb.log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    tsb.log(f"loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states):
            opt.load_state_dict(state)

    # Main training loop
    model.train()
    l1_active = False
    l1_applied_count = 0
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = tsb.eval_val(h, device, val_data, model)
            tsb.log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                tsb.log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)

        if h.num_loops > 0 and (not base_model.looping_active) and (frac >= h.enable_looping_at):
            base_model.looping_active = True
            tsb.log(f"layer_loop:enabled step:{step} frac:{frac:.3f}")

        # --- WD TAPER ---
        if wd_final != h.muon_wd and frac >= wd_taper_start:
            taper_progress = (frac - wd_taper_start) / (1.0 - wd_taper_start + 1e-9)
            current_wd = h.muon_wd + (wd_final - h.muon_wd) * taper_progress
            for group in optimizers.optimizer_muon.param_groups:
                group['weight_decay'] = current_wd

        train_loss = step_fn(step, scale)

        # --- L1 PROXIMAL STEP ---
        if frac >= l1_start_frac and l1_coeff > 0:
            if not l1_active:
                l1_active = True
                tsb.log(f"l1: activated at step={step} frac={frac:.3f}")
            with torch.no_grad():
                # Scale L1 threshold by LR (like WD does), so L1 pressure
                # stays proportional to the model's ability to adapt
                threshold = l1_coeff * scale  # scale = lr_mul(frac), goes 1→0
                for name, param in base_model.named_parameters():
                    if param.ndim == 2 and param.numel() > 65536 and "tok_emb" not in name:
                        param.data = torch.sign(param.data) * torch.clamp(
                            param.data.abs() - threshold, min=0
                        )
            l1_applied_count += 1
        # --- END L1 ---

        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1000.0)
            tsb.log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms / 60000:.1f}m tok/s: {tok_per_sec:.0f}")
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    tsb.log(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    tsb.log(f"l1: applied {l1_applied_count} times")

    # Apply EMA
    tsb.log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)

    # Count sparsity in EMA
    total_zeros = 0
    total_params = 0
    for name, param in base_model.named_parameters():
        if param.ndim == 2 and param.numel() > 65536 and "tok_emb" not in name:
            total_zeros += (param.data == 0).sum().item()
            total_params += param.numel()
    tsb.log(f"final_sparsity: {total_zeros:,}/{total_params:,} ({total_zeros/total_params:.1%} zeros in EMA)")

    return (base_model, compiled_model)


def train_and_eval(h, device):
    import random, numpy as np
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = tsb.ValidationData(h, device)
    tsb.log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    tsb.log(f"val_tokens: {val_data.val_tokens.numel() - 1}")
    base_model, compiled_model = train_model_with_l1(h, device, val_data)
    torch._dynamo.reset()
    tsb.timed_eval("pre-quantization post-ema", tsb.eval_val, h, device, val_data, compiled_model)
    tsb.save_bundle(h, base_model)
    tsb.log("done: bundle saved.")


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = tsb.Hyperparameters()
    tsb.set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        tsb.log("Hyperparameters:", console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                tsb.log(f"  {k}: {v}", console=True)
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
