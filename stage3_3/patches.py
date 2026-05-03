"""
stage3_3 patches — state-dependent hyperparameters.

Base: 1.1233 frontier record train_gpt.py (all SOTA features built in).
Each patch makes a currently-static hyperparameter a function of training state.

All patches are Lane A (training dynamics). No export-only patches.
"""

from __future__ import annotations
import re


def _replace_unique(source: str, old: str, new: str, patch_name: str) -> str:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"[{patch_name}] target string not found:\n{old[:200]!r}")
    if count > 1:
        raise ValueError(f"[{patch_name}] ambiguous: found {count} occurrences:\n{old[:200]!r}")
    return source.replace(old, new)


PATCH_ORDER: dict[str, int] = {
    "step_lr": 10,
    "ns_steps_adaptive": 20,
    "phase_seq_len": 30,
    "per_family_decay": 40,
    "velocity_gate": 50,
    "muon_adam_switch": 60,
    "stochastic_depth": 70,
    "ema_distill": 80,
}


def apply_patches(source: str, patch_names: list[str]) -> str:
    ordered = sorted(patch_names, key=lambda n: PATCH_ORDER.get(n, 100))
    for name in ordered:
        fn = PATCHES[name]
        source = fn(source)
    return source


# ===========================================================================
# H1: STEP LR SCHEDULE (REGIME CHANGE)
#
# Problem: Smooth cosine warmdown spends thousands of steps at intermediate
# LRs (0.3x-0.7x) that are neither productive for learning nor low enough
# for convergence. Neither fish nor fowl.
#
# Mechanism: Full LR for 85% of wallclock, then sharp drop to 0.05x for
# the final 15%. Step schedules are standard in ViT/LLaMA training.
# ===========================================================================

def patch_step_lr(source: str) -> str:
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    step_lr = bool(int(os.environ.get("STEP_LR", "0")))\n'
        '    step_lr_bulk_frac = float(os.environ.get("STEP_LR_BULK_FRAC", "0.85"))\n'
        '    step_lr_final_scale = float(os.environ.get("STEP_LR_FINAL_SCALE", "0.05"))',
        "step_lr",
    )

    # Replace the lr_mul function with step-function version
    source = _replace_unique(
        source,
        '    def lr_mul(step: int, elapsed_ms: float) -> float:\n'
        '        if args.warmdown_iters <= 0:\n'
        '            return 1.0\n'
        '        if max_wallclock_ms is None:\n'
        '            warmdown_start = max(args.iterations - args.warmdown_iters, 0)\n'
        '            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0\n'
        '        step_ms = elapsed_ms / max(step, 1)\n'
        '        warmdown_ms = args.warmdown_iters * step_ms\n'
        '        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)\n'
        '        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0',
        '    def lr_mul(step: int, elapsed_ms: float) -> float:\n'
        '        if args.step_lr:\n'
        '            # Step schedule: full LR until bulk_frac of wallclock, then sharp drop\n'
        '            if max_wallclock_ms is not None:\n'
        '                frac_elapsed = elapsed_ms / max_wallclock_ms\n'
        '            else:\n'
        '                frac_elapsed = step / max(args.iterations, 1)\n'
        '            if frac_elapsed < args.step_lr_bulk_frac:\n'
        '                return 1.0\n'
        '            else:\n'
        '                return args.step_lr_final_scale\n'
        '        if args.warmdown_iters <= 0:\n'
        '            return 1.0\n'
        '        if max_wallclock_ms is None:\n'
        '            warmdown_start = max(args.iterations - args.warmdown_iters, 0)\n'
        '            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0\n'
        '        step_ms = elapsed_ms / max(step, 1)\n'
        '        warmdown_ms = args.warmdown_iters * step_ms\n'
        '        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)\n'
        '        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0',
        "step_lr",
    )
    return source


# ===========================================================================
# H2: THROUGHPUT-AWARE MUON NS STEPS
#
# Problem: backend_steps=5 constant. During warmdown LR is tiny so update
# direction quality matters less, but each NS iteration costs the same.
#
# Mechanism: bulk (scale>=0.5): 7 steps. warmdown (scale<0.5): 3 steps.
# Fewer NS steps = faster optimizer step = more total gradient updates.
# ===========================================================================

def patch_ns_steps_adaptive(source: str) -> str:
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    ns_adaptive = bool(int(os.environ.get("NS_ADAPTIVE", "0")))\n'
        '    ns_bulk_steps = int(os.environ.get("NS_BULK_STEPS", "7"))\n'
        '    ns_warmdown_steps = int(os.environ.get("NS_WARMDOWN_STEPS", "3"))\n'
        '    ns_switch_scale = float(os.environ.get("NS_SWITCH_SCALE", "0.5"))',
        "ns_steps_adaptive",
    )

    # Dynamically adjust backend_steps before the optimizer step
    source = _replace_unique(
        source,
        '        if args.grad_clip_norm > 0:\n'
        '            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)\n'
        '        for opt in optimizers:\n'
        '            opt.step()',
        '        if args.grad_clip_norm > 0:\n'
        '            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)\n'
        '        # Adaptive NS steps: more during bulk, fewer during warmdown\n'
        '        if args.ns_adaptive:\n'
        '            _ns = args.ns_bulk_steps if scale >= args.ns_switch_scale else args.ns_warmdown_steps\n'
        '            for group in optimizer_muon.param_groups:\n'
        '                group["backend_steps"] = _ns\n'
        '        for opt in optimizers:\n'
        '            opt.step()',
        "ns_steps_adaptive",
    )
    return source


# ===========================================================================
# H3: THREE-PHASE SEQUENCE LENGTH CURRICULUM
#
# Problem: train_seq_len=2048 constant. Attention is O(n²) — using 2048
# from step 1 wastes massive compute. Early training builds local features
# that don't need long context.
#
# Mechanism: 3-phase: 512 (early bulk) → 1024 (late bulk) → 2048 (warmdown).
# train_batch_tokens stays constant. 27% more total gradient updates.
# ===========================================================================

def patch_phase_seq_len(source: str) -> str:
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    phase_seq_len = bool(int(os.environ.get("PHASE_SEQ_LEN", "0")))\n'
        '    phase_seq_len_phase1 = int(os.environ.get("PHASE_SEQ_LEN_PHASE1", "512"))\n'
        '    phase_seq_len_phase2 = int(os.environ.get("PHASE_SEQ_LEN_PHASE2", "1024"))\n'
        '    phase_seq_len_phase3 = int(os.environ.get("PHASE_SEQ_LEN_PHASE3", "2048"))\n'
        '    phase_seq_len_switch1 = float(os.environ.get("PHASE_SEQ_LEN_SWITCH1", "0.7"))\n'
        '    phase_seq_len_switch2 = float(os.environ.get("PHASE_SEQ_LEN_SWITCH2", "0.3"))',
        "phase_seq_len",
    )

    # Dynamically adjust seq_len in the data loader call — 3-phase
    source = _replace_unique(
        source,
        '            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)\n'
        '            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):\n'
        '                loss = model(x, y)\n'
        '            train_loss += loss.detach()\n'
        '            (loss * grad_scale).backward()',
        '            _seq_len = args.train_seq_len\n'
        '            if args.phase_seq_len:\n'
        '                if scale >= args.phase_seq_len_switch1:\n'
        '                    _seq_len = args.phase_seq_len_phase1  # early bulk: short context\n'
        '                elif scale >= args.phase_seq_len_switch2:\n'
        '                    _seq_len = args.phase_seq_len_phase2  # late bulk: medium context\n'
        '                else:\n'
        '                    _seq_len = args.phase_seq_len_phase3  # warmdown: full context\n'
        '            x, y = train_loader.next_batch(args.train_batch_tokens, _seq_len, grad_accum_steps)\n'
        '            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):\n'
        '                loss = model(x, y)\n'
        '            train_loss += loss.detach()\n'
        '            (loss * grad_scale).backward()',
        "phase_seq_len",
    )
    return source


# ===========================================================================
# H4: PER-FAMILY WARMDOWN DECAY RATES
#
# Problem: All param families use the same lr *= scale. Embeddings, matrices,
# and scalars have different quantization sensitivity and different roles.
#
# Mechanism: embed lr *= scale^1.5 (converge first), matrix lr *= scale,
# scalar lr *= scale^0.5 (keep adapting).
# ===========================================================================

def patch_per_family_decay(source: str) -> str:
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    per_family_decay = bool(int(os.environ.get("PER_FAMILY_DECAY", "0")))\n'
        '    embed_decay_pow = float(os.environ.get("EMBED_DECAY_POW", "1.5"))\n'
        '    scalar_decay_pow = float(os.environ.get("SCALAR_DECAY_POW", "0.5"))',
        "per_family_decay",
    )

    # Replace the uniform LR update with per-family decay
    source = _replace_unique(
        source,
        '        for opt in optimizers:\n'
        '            for group in opt.param_groups:\n'
        '                group["lr"] = group["base_lr"] * scale',
        '        if args.per_family_decay and scale < 1.0:\n'
        '            # Per-family decay: embed decays faster, scalar decays slower\n'
        '            _embed_scale = max(scale, 0.0) ** args.embed_decay_pow\n'
        '            _scalar_scale = max(scale, 1e-9) ** args.scalar_decay_pow\n'
        '            for group in optimizer_tok.param_groups:\n'
        '                group["lr"] = group["base_lr"] * _embed_scale\n'
        '            for group in optimizer_muon.param_groups:\n'
        '                group["lr"] = group["base_lr"] * scale\n'
        '            for group in optimizer_scalar.param_groups:\n'
        '                group["lr"] = group["base_lr"] * _scalar_scale\n'
        '            if base_model.lm_head is not None:\n'
        '                for group in optimizer_head.param_groups:\n'
        '                    group["lr"] = group["base_lr"] * _embed_scale\n'
        '        else:\n'
        '            for opt in optimizers:\n'
        '                for group in opt.param_groups:\n'
        '                    group["lr"] = group["base_lr"] * scale',
        "per_family_decay",
    )
    return source


# ===========================================================================
# H5: LOSS-VELOCITY WARMDOWN GATING
#
# Problem: Warmdown starts at fixed wallclock fraction, ignoring whether the
# model is still rapidly improving. May waste productive bulk steps.
#
# Mechanism: Track EMA of loss velocity. While velocity is strongly negative
# (model still learning fast), hold scale=1.0 to delay warmdown. Cap delay
# at 20% of warmdown budget.
# ===========================================================================

def patch_velocity_gate(source: str) -> str:
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    velocity_gate = bool(int(os.environ.get("VELOCITY_GATE", "0")))\n'
        '    velocity_ema = float(os.environ.get("VELOCITY_EMA", "0.95"))\n'
        '    velocity_threshold = float(os.environ.get("VELOCITY_THRESHOLD", "0.001"))\n'
        '    velocity_max_delay_frac = float(os.environ.get("VELOCITY_MAX_DELAY_FRAC", "0.20"))',
        "velocity_gate",
    )

    # Add velocity tracking state
    source = _replace_unique(
        source,
        '    swa_state: dict[str, Tensor] | None = None\n'
        '    swa_count = 0',
        '    swa_state: dict[str, Tensor] | None = None\n'
        '    swa_count = 0\n'
        '    # Velocity gating state\n'
        '    _prev_loss: float | None = None\n'
        '    _loss_velocity: float = 0.0\n'
        '    _velocity_delay_steps: int = 0\n'
        '    _velocity_max_delay: int = int(args.warmdown_iters * args.velocity_max_delay_frac) if args.velocity_gate else 0',
        "velocity_gate",
    )

    # After computing scale, apply velocity gate
    source = _replace_unique(
        source,
        '        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n'
        '        scale = lr_mul(step, elapsed_ms)',
        '        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n'
        '        scale = lr_mul(step, elapsed_ms)\n'
        '        # Velocity gate: hold scale=1.0 while loss is still dropping fast\n'
        '        if args.velocity_gate and scale < 1.0 and _velocity_delay_steps < _velocity_max_delay:\n'
        '            if _loss_velocity < -args.velocity_threshold:\n'
        '                scale = 1.0\n'
        '                _velocity_delay_steps += 1\n'
        '                if _velocity_delay_steps == 1:\n'
        '                    log0(f"velocity_gate:delaying warmdown step:{step} velocity:{_loss_velocity:.6f}")',
        "velocity_gate",
    )

    # Update velocity tracker after each step's loss
    source = _replace_unique(
        source,
        '        train_loss /= grad_accum_steps',
        '        train_loss /= grad_accum_steps\n'
        '        # Update loss velocity tracker\n'
        '        if args.velocity_gate:\n'
        '            _cur_loss = train_loss.item()\n'
        '            if _prev_loss is not None:\n'
        '                _delta = _cur_loss - _prev_loss\n'
        '                _loss_velocity = args.velocity_ema * _loss_velocity + (1 - args.velocity_ema) * _delta\n'
        '            _prev_loss = _cur_loss',
        "velocity_gate",
    )
    return source


# ===========================================================================
# H6: MUON → ADAM WARMDOWN SWITCH (REGIME CHANGE)
#
# Problem: Muon (Newton-Schulz) for all matrix params for the whole run.
# Muon is designed for bulk learning — steepest descent in natural geometry.
# During deep warmdown under QAT, the task is fine convergence into a
# quantization-friendly basin. Adam's per-parameter adaptive rates are
# better for heterogeneous fine convergence.
#
# Mechanism: At scale < 0.3, replace Muon with fresh AdamW for matrix params.
# ===========================================================================

def patch_muon_adam_switch(source: str) -> str:
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    muon_adam_switch = bool(int(os.environ.get("MUON_ADAM_SWITCH", "0")))\n'
        '    muon_adam_switch_scale = float(os.environ.get("MUON_ADAM_SWITCH_SCALE", "0.3"))',
        "muon_adam_switch",
    )

    # Add state tracking and switch logic after the optimizer step
    source = _replace_unique(
        source,
        '        for opt in optimizers:\n'
        '            opt.step()\n'
        '        zero_grad_all()',
        '        for opt in optimizers:\n'
        '            opt.step()\n'
        '        zero_grad_all()\n'
        '        # Muon → Adam switch: replace Muon with AdamW during deep warmdown\n'
        '        if (args.muon_adam_switch and scale < args.muon_adam_switch_scale\n'
        '                and isinstance(optimizers[2 if base_model.lm_head is not None else 1], Muon)):\n'
        '            _muon_idx = 2 if base_model.lm_head is not None else 1\n'
        '            _muon_params = list(optimizers[_muon_idx].param_groups[0]["params"])\n'
        '            _adam_for_matrix = torch.optim.AdamW(\n'
        '                [{"params": _muon_params, "lr": args.matrix_lr * scale, "base_lr": args.matrix_lr}],\n'
        '                betas=(args.beta1, args.beta2),\n'
        '                eps=args.adam_eps,\n'
        '                weight_decay=args.muon_wd,\n'
        '                fused=True,\n'
        '            )\n'
        '            optimizers[_muon_idx] = _adam_for_matrix\n'
        '            log0(f"muon_adam_switch:activated step:{step} scale:{scale:.4f} matrix_params:{sum(p.numel() for p in _muon_params)}")\n'
        '            # Rebuild zero_grad_all to use new optimizer list\n'
        '            def zero_grad_all() -> None:\n'
        '                for opt in optimizers:\n'
        '                    opt.zero_grad(set_to_none=True)',
        "muon_adam_switch",
    )
    return source


# ===========================================================================
# H7: STOCHASTIC DEPTH DURING BULK (COMPUTE REALLOCATION)
#
# Problem: All 11 blocks execute every step. During bulk training, coarse
# features don't need all layers — running all 11 wastes compute.
#
# Mechanism: Each block randomly skips with p=0.3 during bulk, p=0.0
# during warmdown. ~25% faster steps, 12% more total gradient updates.
# ===========================================================================

def patch_stochastic_depth(source: str) -> str:
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    stoch_depth = bool(int(os.environ.get("STOCH_DEPTH", "0")))\n'
        '    stoch_depth_drop = float(os.environ.get("STOCH_DEPTH_DROP", "0.3"))\n'
        '    stoch_depth_switch = float(os.environ.get("STOCH_DEPTH_SWITCH", "0.5"))',
        "stochastic_depth",
    )

    # Add class variable to Block for drop probability
    source = _replace_unique(
        source,
        'class Block(nn.Module):\n'
        '    def __init__(',
        'class Block(nn.Module):\n'
        '    _stoch_drop_prob: float = 0.0  # set from training loop\n'
        '    def __init__(',
        "stochastic_depth",
    )

    # Add stochastic depth skip at the start of Block.forward
    source = _replace_unique(
        source,
        '    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:\n'
        '        mix = self.resid_mix.to(dtype=x.dtype)',
        '    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:\n'
        '        # Stochastic depth: skip this block with probability _stoch_drop_prob\n'
        '        if self.training and Block._stoch_drop_prob > 0 and torch.rand(1).item() < Block._stoch_drop_prob:\n'
        '            return x  # identity pass-through\n'
        '        mix = self.resid_mix.to(dtype=x.dtype)',
        "stochastic_depth",
    )

    # Set drop probability in the training loop based on scale
    source = _replace_unique(
        source,
        '        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n'
        '        scale = lr_mul(step, elapsed_ms)',
        '        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n'
        '        scale = lr_mul(step, elapsed_ms)\n'
        '        # Stochastic depth: drop blocks during bulk, use all during warmdown\n'
        '        if args.stoch_depth:\n'
        '            Block._stoch_drop_prob = args.stoch_depth_drop if scale >= args.stoch_depth_switch else 0.0',
        "stochastic_depth",
    )
    return source


# ===========================================================================
# H8: EMA SELF-DISTILLATION DURING WARMDOWN
#
# Problem: EMA is only used passively for checkpoint averaging. It
# represents a smoother trajectory but is never used as a teaching signal.
#
# Mechanism: During warmdown, add KL(student || EMA) loss on micro_step==0.
# The EMA guides the current model toward the averaged trajectory.
# ===========================================================================

def patch_ema_distill(source: str) -> str:
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    ema_distill = bool(int(os.environ.get("EMA_DISTILL", "0")))\n'
        '    ema_distill_weight = float(os.environ.get("EMA_DISTILL_WEIGHT", "0.5"))\n'
        '    ema_distill_switch = float(os.environ.get("EMA_DISTILL_SWITCH", "0.5"))',
        "ema_distill",
    )

    # Add self-distillation loss after the main forward+backward
    source = _replace_unique(
        source,
        '            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):\n'
        '                loss = model(x, y)\n'
        '            train_loss += loss.detach()\n'
        '            (loss * grad_scale).backward()',
        '            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):\n'
        '                loss = model(x, y)\n'
        '                # EMA self-distillation: add KL(student || EMA) during warmdown\n'
        '                if args.ema_distill and scale < args.ema_distill_switch and micro_step == 0:\n'
        '                    # Get student logits (WITH gradients for backprop)\n'
        '                    student_logits = base_model.forward_logits(x)\n'
        '                    # Temporarily load EMA state to get teacher logits (no grad)\n'
        '                    _orig_sd = {n: t.detach().clone() for n, t in base_model.state_dict().items()}\n'
        '                    _ema_sd = {n: t.to(dtype=_orig_sd[n].dtype) for n, t in ema_state.items()}\n'
        '                    base_model.load_state_dict(_ema_sd, strict=True)\n'
        '                    with torch.no_grad():\n'
        '                        teacher_logits = base_model.forward_logits(x).detach()\n'
        '                    base_model.load_state_dict(_orig_sd, strict=True)\n'
        '                    # KL divergence: student should match EMA distribution\n'
        '                    distill_loss = F.kl_div(\n'
        '                        F.log_softmax(student_logits.float().view(-1, student_logits.size(-1)), dim=-1),\n'
        '                        F.softmax(teacher_logits.float().view(-1, teacher_logits.size(-1)), dim=-1),\n'
        '                        reduction="batchmean",\n'
        '                    )\n'
        '                    loss = loss + args.ema_distill_weight * distill_loss\n'
        '            train_loss += loss.detach()\n'
        '            (loss * grad_scale).backward()',
        "ema_distill",
    )
    return source


# ===========================================================================
# Registry
# ===========================================================================

PATCHES: dict[str, callable] = {
    "step_lr": patch_step_lr,
    "ns_steps_adaptive": patch_ns_steps_adaptive,
    "phase_seq_len": patch_phase_seq_len,
    "per_family_decay": patch_per_family_decay,
    "velocity_gate": patch_velocity_gate,
    "muon_adam_switch": patch_muon_adam_switch,
    "stochastic_depth": patch_stochastic_depth,
    "ema_distill": patch_ema_distill,
}
