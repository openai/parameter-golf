from __future__ import annotations

import copy
import time
from typing import Callable

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .config import Hyperparameters
from .data import DistributedTokenLoader
from .eval import eval_val
from .lr_schedulers import get_scheduler
from .optim import Muon
from .serialization import dequantize_state_dict_int8, quantize_state_dict_int8
from .ternary import serialize_ternary_lzma, deserialize_ternary_lzma


def _optimizer_lr_abs_metrics(optimizers: list[torch.optim.Optimizer]) -> dict[str, float]:
    """Current |lr| per param group (after scheduler scale), plus max/min for dashboards."""
    out: dict[str, float] = {}
    for oi, opt in enumerate(optimizers):
        for gi, group in enumerate(opt.param_groups):
            out[f"lr_abs_opt{oi}_g{gi}"] = abs(float(group["lr"]))
    if out:
        vals = list(out.values())
        out["lr_abs_max"] = max(vals)
        out["lr_abs_min"] = min(vals)
    return out


def run_training(
    *,
    args: Hyperparameters,
    model: nn.Module,
    base_model: nn.Module,
    optimizer_muon: Muon,
    optimizers: list[torch.optim.Optimizer],
    train_loader: DistributedTokenLoader,
    rank: int,
    world_size: int,
    distributed: bool,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    log0: Callable[[str, bool], None],
    on_train_log: Callable[[int, float, float, float, dict[str, float] | None], None] | None = None,
    on_val_log: Callable[[int, float, float, float, float], None] | None = None,
) -> float:
    grad_scale = 1.0 / grad_accum_steps
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    _scheduler = get_scheduler(args.lr_schedule)

    def lr_mul(step: int, elapsed_ms: float) -> float:
        return _scheduler(step, args.iterations, args.warmdown_iters, elapsed_ms, max_wallclock_ms)

    def eval_val_full_precision() -> tuple[float, float]:
        return eval_val(
            args,
            model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )

    def eval_val_quantized_roundtrip() -> tuple[float, float]:
        # Validate on the quantized->dequantized weights that match submission-time path.
        fp_state = {name: tensor.detach().to("cpu").contiguous() for name, tensor in base_model.state_dict().items()}
        # MTP heads are training-only; exclude from quantization to match submission path.
        core_state = {k: v for k, v in fp_state.items() if not k.startswith("mtp_heads.")}
        mtp_state = {k: v for k, v in fp_state.items() if k.startswith("mtp_heads.")}

        if args.ternary_enabled:
            blob = serialize_ternary_lzma(core_state)
            q_state = deserialize_ternary_lzma(blob)
        else:
            quant_obj, _quant_stats = quantize_state_dict_int8(
                core_state,
                ptq_bits=args.ptq_bits,
                ptq_mlp_bits=args.ptq_mlp_bits,
                int6_layer_start=args.int6_layer_start,
                int6_layer_end=args.int6_layer_end,
            )
            q_state = dequantize_state_dict_int8(quant_obj)

        q_state.update(mtp_state)
        base_model.load_state_dict(q_state, strict=True)
        try:
            return eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
        finally:
            base_model.load_state_dict(fp_state, strict=True)

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss_fp, val_bpb_fp = eval_val_full_precision()
            val_loss_q, val_bpb_q = eval_val_quantized_roundtrip()
            log0(
                f"step:{step}/{args.iterations} "
                f"val_loss_full_precision:{val_loss_fp:.4f} val_bpb_full_precision:{val_bpb_fp:.4f} "
                f"val_loss_quantized_model:{val_loss_q:.4f} val_bpb_quantized_model:{val_bpb_q:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if on_val_log is not None:
                on_val_log(
                    step,
                    float(val_loss_fp),
                    float(val_bpb_fp),
                    float(val_loss_q),
                    float(val_bpb_q),
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        comet_every = max(args.comet_log_train_every, 1)
        act_norm_every = max(args.activation_norm_log_every, 1)
        will_collect_act_norms = (
            args.log_activation_norms
            and on_train_log is not None
            and args.comet_enable
            and rank == 0
            and (step + 1) % act_norm_every == 0
        )
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            base_model._activation_norm_collect = bool(
                will_collect_act_norms and micro_step == grad_accum_steps - 1
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        base_model._activation_norm_collect = False
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        else:
            grad_norm = torch.zeros((), device=device)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"grad_norm:{grad_norm.item():.4f} lr_scale:{scale:.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        if on_train_log is not None and step % comet_every == 0:
            extra_metrics: dict[str, float] = dict(_optimizer_lr_abs_metrics(optimizers))
            if will_collect_act_norms:
                act_norms = getattr(base_model, "_activation_norms", None)
                if act_norms:
                    extra_metrics.update(act_norms)
            on_train_log(step, float(train_loss.item()), scale, float(grad_norm.item()), extra_metrics)

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    return training_time_ms
