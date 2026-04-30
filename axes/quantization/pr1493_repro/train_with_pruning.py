"""PR-1493 training with Hessian-aware magnitude pruning during warmdown.

Identical to train_save_bundle.py but adds gradual pruning during the
warmdown phase (last 72% of training). Uses importance = |w| * sqrt(H_diag_col)
from a pre-computed Hessian estimate (collected from a short calibration pass
at the start of warmdown).

Strategy:
  Steps 0-1300 (warmup + early training): normal training, no pruning
  Step ~1300 (warmdown begins): collect Hessian estimate from 16 calibration batches
  Steps 1300-4550 (warmdown): gradually increase sparsity 0% -> PRUNE_FINAL_SPARSITY
    - Cubic schedule: s(t) = s_final * (1 - (1 - t/T)^3)
    - After each optimizer step: zero out weights with lowest importance scores
    - importance = |w| * sqrt(H_diag_col) per tensor
    - Zeroed weights stay zeroed (thresholding, not masking — LR is decaying to 0)
  After training: save EMA bundle as usual (EMA tracks pruned weights → zeros converge)

Usage on RunPod 8×H100:
    PRUNE_FINAL_SPARSITY=0.3 \
    BUNDLE_DIR=./bundle_pruned_30 \
    torchrun --standalone --nproc_per_node=8 train_with_pruning.py

Env vars:
    PRUNE_FINAL_SPARSITY  target fraction of zeros at end of training (default 0.3)
    PRUNE_CALIB_BATCHES   batches for Hessian estimate at warmdown start (default 16)
    PRUNE_EVERY_N         apply pruning every N steps (default 50)
    BUNDLE_DIR            where to save the output bundle
"""

# This file is a thin wrapper that patches train_save_bundle.py's training loop.
# We import the full module and override train_model + train_and_eval.

import os
import sys
import math
import time
from pathlib import Path

# Set defaults before importing train_save_bundle
os.environ.setdefault("BUNDLE_DIR", "bundle_pruned")
os.environ.setdefault("TTT_ENABLED", "0")
os.environ.setdefault("ETLB_ENABLED", "0")

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_save_bundle as tsb


def collect_hessian_diags(model, train_loader, h, device, n_batches=16):
    """Collect diagonal of H = X^T X for each linear layer. Lightweight version
    of collect_hessians — only stores the diagonal, not the full matrix."""
    diags = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, tsb.CastedLinear) and module.weight.numel() > 65536:
            param_name = name + ".weight"
            def make_hook(pname):
                def hook_fn(mod, inp, out):
                    x = inp[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    # H_diag[j] = sum_i x[i,j]^2 = (x^2).sum(dim=0)
                    col_sq = (x ** 2).sum(dim=0)
                    if pname not in diags:
                        diags[pname] = torch.zeros(x.shape[1], dtype=torch.float32, device="cpu")
                    diags[pname] += col_sq.cpu()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(param_name)))

    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    for name in diags:
        diags[name] /= n_batches
    model.train()
    return diags


def apply_pruning(model, hessian_diags, target_sparsity):
    """Zero out weights with lowest importance = |w| * sqrt(H_diag_col).
    Applied per-tensor to the target sparsity fraction."""
    if target_sparsity <= 0:
        return 0, 0
    pruned = 0
    total = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim != 2 or param.numel() <= 65536 or "tok_emb" in name:
                continue
            # Find the matching Hessian diagonal
            h_key = None
            for candidate in [name, name.replace(".weight", "") + ".weight"]:
                if candidate in hessian_diags:
                    h_key = candidate
                    break
            if h_key is None:
                # Fallback: magnitude-only
                importance = param.data.abs().float()
            else:
                col_sens = hessian_diags[h_key].to(param.device).sqrt().clamp_min(1e-10)
                importance = param.data.abs().float() * col_sens.unsqueeze(0)

            threshold = torch.quantile(importance.flatten(), target_sparsity)
            mask = importance >= threshold
            param.data *= mask
            pruned += (~mask).sum().item()
            total += param.numel()
    return pruned, total


def train_model_with_pruning(h, device, val_data):
    """Modified train_model that adds Hessian-aware pruning during warmdown."""
    # Parse pruning config
    final_sparsity = float(os.environ.get("PRUNE_FINAL_SPARSITY", "0.3"))
    calib_batches = int(os.environ.get("PRUNE_CALIB_BATCHES", "16"))
    prune_every = int(os.environ.get("PRUNE_EVERY_N", "50"))

    tsb.log(f"pruning config: final_sparsity={final_sparsity:.0%} calib_batches={calib_batches} prune_every={prune_every}")

    # Standard model setup (copied from train_save_bundle.py train_model)
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

    # Pruning state
    hessian_diags = None
    warmdown_started = False
    warmdown_start_step = None

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
        if frac < h.warmdown_frac:
            return 1.0
        return max(h.min_lr, 1.0 - (frac - h.warmdown_frac) / (1.0 - h.warmdown_frac + 1e-9))

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

    # Warmup phase
    if h.warmup_steps > 0:
        import copy
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

        # Enable looping
        if h.num_loops > 0 and (not base_model.looping_active) and (frac >= h.enable_looping_at):
            base_model.looping_active = True
            tsb.log(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")

        # --- PRUNING LOGIC ---
        if frac >= h.warmdown_frac and not warmdown_started:
            warmdown_started = True
            warmdown_start_step = step
            # Collect Hessian diagonals for importance scoring
            tsb.log(f"pruning: warmdown started at step {step}, collecting Hessian diags...")
            hessian_diags = collect_hessian_diags(base_model, train_loader, h, device, n_batches=calib_batches)
            tsb.log(f"pruning: collected {len(hessian_diags)} Hessian diags")

        if warmdown_started and hessian_diags and step % prune_every == 0:
            # Use frac-based progress (respects wallclock cap, not h.iterations)
            warmdown_progress = (frac - h.warmdown_frac) / (1.0 - h.warmdown_frac + 1e-9)
            # Cubic schedule: slow start, accelerate
            current_sparsity = final_sparsity * (1 - (1 - warmdown_progress) ** 3)
            if current_sparsity > 0.001:
                pruned, total = apply_pruning(base_model, hessian_diags, current_sparsity)
                if step % (prune_every * 5) == 0:
                    tsb.log(f"pruning: step={step} target={current_sparsity:.1%} zeroed={pruned:,}/{total:,}")
        # --- END PRUNING ---

        train_loss = step_fn(step, scale)

        # Apply pruning threshold after optimizer step (keep pruned weights at zero)
        if warmdown_started and hessian_diags:
            # Use frac-based progress (respects wallclock cap, not h.iterations)
            warmdown_progress = (frac - h.warmdown_frac) / (1.0 - h.warmdown_frac + 1e-9)
            current_sparsity = final_sparsity * (1 - (1 - warmdown_progress) ** 3)
            if current_sparsity > 0.001:
                with torch.no_grad():
                    for name, param in base_model.named_parameters():
                        if param.ndim != 2 or param.numel() <= 65536 or "tok_emb" in name:
                            continue
                        h_key = None
                        for candidate in [name, name + ".weight"]:
                            if candidate in hessian_diags:
                                h_key = candidate
                                break
                        if h_key is not None:
                            col_sens = hessian_diags[h_key].to(param.device).sqrt().clamp_min(1e-10)
                            importance = param.data.abs().float() * col_sens.unsqueeze(0)
                        else:
                            importance = param.data.abs().float()
                        threshold = torch.quantile(importance.flatten(), current_sparsity)
                        param.data *= (importance >= threshold)

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

    # Apply EMA, then enforce sparsity on EMA (EMA lags behind pruning)
    tsb.log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)

    # Enforce final sparsity on EMA weights (EMA lags behind live pruning)
    if warmdown_started and hessian_diags and final_sparsity > 0:
        tsb.log(f"pruning: enforcing {final_sparsity:.0%} sparsity on EMA weights...")
        apply_pruning(base_model, hessian_diags, final_sparsity)

    # Log final sparsity
    total_zeros = 0
    total_params = 0
    for name, param in base_model.named_parameters():
        if param.ndim == 2 and param.numel() > 65536 and "tok_emb" not in name:
            total_zeros += (param.data == 0).sum().item()
            total_params += param.numel()
    tsb.log(f"final_sparsity: {total_zeros:,}/{total_params:,} ({total_zeros/total_params:.1%} zeros in EMA)")

    return (base_model, compiled_model)


def train_and_eval(h, device):
    import random
    import numpy as np
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = tsb.ValidationData(h, device)
    tsb.log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    tsb.log(f"val_tokens: {val_data.val_tokens.numel() - 1}")
    base_model, compiled_model = train_model_with_pruning(h, device, val_data)
    torch._dynamo.reset()
    tsb.timed_eval("pre-quantization post-ema", tsb.eval_val, h, device, val_data, compiled_model)
    tsb.save_bundle(h, base_model)
    tsb.log("done: bundle saved. Skipping quantization and post-quant eval.")


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
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
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = tsb.Hyperparameters()
    tsb.set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        tsb.log("=" * 100, console=False)
        tsb.log("Hyperparameters:", console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                tsb.log(f"  {k}: {v}", console=True)
        tsb.log("=" * 100, console=False)
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
