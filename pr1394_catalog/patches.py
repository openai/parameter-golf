"""
Executable first-order patch families for frontier_rebase/pr1394/train_gpt_human.py.

These are deliberately targeted at the real SP8192 frontier-rebase base, not the
older stage-local train_gpt variants. The goal is to support one honest family-
admission pack with six consequential hypotheses that can be materialized and run.
"""

from __future__ import annotations


def _replace_unique(source: str, old: str, new: str, patch_name: str) -> str:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"[{patch_name}] target string not found:\n{old[:200]!r}")
    if count > 1:
        raise ValueError(f"[{patch_name}] target string found {count} times:\n{old[:200]!r}")
    return source.replace(old, new)


def _insert_after_unique(source: str, old: str, addition: str, patch_name: str) -> str:
    return _replace_unique(source, old, old + addition, patch_name)


def patch_loop_ramp(source: str) -> str:
    name = "loop_ramp"
    source = _insert_after_unique(
        source,
        "    enable_looping_at = float(os.environ.get('ENABLE_LOOPING_AT', 0.5))\n",
        "    loop_ramp_start = float(os.environ.get('LOOP_RAMP_START', os.environ.get('ENABLE_LOOPING_AT', '0.5')))\n"
        "    loop_ramp_full = float(os.environ.get('LOOP_RAMP_FULL', '0.7'))\n",
        name,
    )
    source = _replace_unique(
        source,
        "        self.num_encoder_layers = h.num_layers // 2\n",
        "        self.num_layers = h.num_layers\n"
        "        self.num_encoder_layers = h.num_layers // 2\n",
        name,
    )
    source = _replace_unique(
        source,
        "        # Layer looping\n"
        "        self.looping_active: bool = False\n"
        "        if h.num_loops > 0:\n"
        "            loop_seg = list(range(h.loop_start, h.loop_end + 1))\n"
        "            all_indices = list(range(h.loop_start))\n"
        "            for _ in range(h.num_loops + 1):\n"
        "                all_indices.extend(loop_seg)\n"
        "            all_indices.extend(range(h.loop_end + 1, h.num_layers))\n"
        "            num_enc = len(all_indices) // 2\n"
        "            self.encoder_indices: list[int] = all_indices[:num_enc]\n"
        "            self.decoder_indices: list[int] = all_indices[num_enc:]\n"
        "        else:\n"
        "            self.encoder_indices = list(range(self.num_encoder_layers))\n"
        "            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))\n",
        "        # Layer looping\n"
        "        self.looping_active: bool = False\n"
        "        self.max_loop_repeats = h.num_loops\n"
        "        self.loop_start_idx = h.loop_start\n"
        "        self.loop_end_idx = h.loop_end\n"
        "        self.active_loop_repeats = 0\n"
        "        self.encoder_indices: list[int] = []\n"
        "        self.decoder_indices: list[int] = []\n"
        "        self.set_loop_repeats(0)\n",
        name,
    )
    source = _replace_unique(
        source,
        "    def _init_weights(self) -> None:\n",
        "    def set_loop_repeats(self, repeats: int) -> None:\n"
        "        repeats = max(0, min(int(repeats), self.max_loop_repeats))\n"
        "        self.active_loop_repeats = repeats\n"
        "        self.looping_active = repeats > 0\n"
        "        if repeats > 0:\n"
        "            loop_seg = list(range(self.loop_start_idx, self.loop_end_idx + 1))\n"
        "            all_indices = list(range(self.loop_start_idx))\n"
        "            for _ in range(repeats + 1):\n"
        "                all_indices.extend(loop_seg)\n"
        "            all_indices.extend(range(self.loop_end_idx + 1, self.num_layers))\n"
        "            num_enc = len(all_indices) // 2\n"
        "            self.encoder_indices = all_indices[:num_enc]\n"
        "            self.decoder_indices = all_indices[num_enc:]\n"
        "        else:\n"
        "            self.encoder_indices = list(range(self.num_encoder_layers))\n"
        "            self.decoder_indices = list(range(self.num_encoder_layers, self.num_layers))\n"
        "\n"
        "    def _init_weights(self) -> None:\n",
        name,
    )
    source = _replace_unique(
        source,
        "            base_model.looping_active = True\n"
        "            log(f\"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}\")\n",
        "            base_model.set_loop_repeats(h.num_loops)\n"
        "            log(\n"
        "                f\"loop_warmup:enabled repeats:{base_model.active_loop_repeats} \"\n"
        "                f\"encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}\"\n"
        "            )\n",
        name,
    )
    source = _replace_unique(
        source,
        "            base_model.looping_active = False\n",
        "            base_model.set_loop_repeats(0)\n",
        name,
    )
    source = _replace_unique(
        source,
        "        if h.num_loops > 0 and not base_model.looping_active and frac >= h.enable_looping_at:\n"
        "            base_model.looping_active = True\n"
        "            log(f\"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}\")\n"
        "        train_loss = step_fn(step, scale)\n",
        "        if h.num_loops > 0:\n"
        "            if frac < h.loop_ramp_start:\n"
        "                active_loops = 0\n"
        "            elif frac >= h.loop_ramp_full or h.num_loops <= 1:\n"
        "                active_loops = h.num_loops\n"
        "            else:\n"
        "                active_loops = max(1, h.num_loops // 2)\n"
        "            if active_loops != base_model.active_loop_repeats:\n"
        "                base_model.set_loop_repeats(active_loops)\n"
        "                log(\n"
        "                    f\"layer_loop:repeats={active_loops} step:{step} frac:{frac:.3f} \"\n"
        "                    f\"encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}\"\n"
        "                )\n"
        "        train_loss = step_fn(step, scale)\n",
        name,
    )
    source = _replace_unique(
        source,
        "        eval_model.looping_active = True\n",
        "        eval_model.set_loop_repeats(h.num_loops)\n",
        name,
    )
    return source


def patch_skip_gate_schedule(source: str) -> str:
    name = "skip_gate_schedule"
    source = _insert_after_unique(
        source,
        "    skip_gates_enabled = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))\n",
        "    skip_gate_bias_init = float(os.environ.get('SKIP_GATE_BIAS_INIT', '0.0'))\n"
        "    skip_gate_bias_end = float(os.environ.get('SKIP_GATE_BIAS_END', '0.0'))\n"
        "    skip_gate_bias_full_at = float(os.environ.get('SKIP_GATE_BIAS_FULL_AT', '0.5'))\n",
        name,
    )
    source = _replace_unique(
        source,
        "        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)) if h.skip_gates_enabled else None\n",
        "        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)) if h.skip_gates_enabled else None\n"
        "        self.skip_gate_bias = float(h.skip_gate_bias_init)\n",
        name,
    )
    source = _replace_unique(
        source,
        "                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]\n",
        "                    g = torch.sigmoid((self.skip_gates[skip_idx] + self.skip_gate_bias).to(dtype=x.dtype))[None, None, :]\n",
        name,
    )
    source = _replace_unique(
        source,
        "        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n"
        "        frac = training_frac(step, elapsed_ms)\n"
        "        scale = lr_mul(frac)\n",
        "        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n"
        "        frac = training_frac(step, elapsed_ms)\n"
        "        if base_model.skip_gates is not None:\n"
        "            if h.skip_gate_bias_full_at <= 0:\n"
        "                base_model.skip_gate_bias = h.skip_gate_bias_end\n"
        "            elif frac >= h.skip_gate_bias_full_at:\n"
        "                base_model.skip_gate_bias = h.skip_gate_bias_end\n"
        "            else:\n"
        "                bias_frac = frac / h.skip_gate_bias_full_at\n"
        "                base_model.skip_gate_bias = (\n"
        "                    (1.0 - bias_frac) * h.skip_gate_bias_init\n"
        "                    + bias_frac * h.skip_gate_bias_end\n"
        "                )\n"
        "        scale = lr_mul(frac)\n",
        name,
    )
    return source


def patch_ema_selector(source: str) -> str:
    name = "ema_selector"
    source = _insert_after_unique(
        source,
        "    ema_decay = float(os.environ.get('EMA_DECAY', 0.997))\n",
        "    ema_fast_decay = float(os.environ.get('EMA_FAST_DECAY', '0.0'))\n"
        "    ema_slow_decay = float(os.environ.get('EMA_SLOW_DECAY', '0.0'))\n",
        name,
    )
    source = _replace_unique(
        source,
        "    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}\n"
        "    ema_decay = h.ema_decay\n",
        "    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}\n"
        "    ema_decay = h.ema_decay\n"
        "    ema_fast_state = (\n"
        "        {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}\n"
        "        if h.ema_fast_decay > 0\n"
        "        else None\n"
        "    )\n"
        "    ema_slow_state = (\n"
        "        {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}\n"
        "        if h.ema_slow_decay > 0\n"
        "        else None\n"
        "    )\n",
        name,
    )
    source = _replace_unique(
        source,
        "        with torch.no_grad():\n"
        "            for name, t in base_model.state_dict().items():\n"
        "                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)\n",
        "        with torch.no_grad():\n"
        "            for name, t in base_model.state_dict().items():\n"
        "                t_float = t.detach().float()\n"
        "                ema_state[name].mul_(ema_decay).add_(t_float, alpha=1.0 - ema_decay)\n"
        "                if ema_fast_state is not None:\n"
        "                    ema_fast_state[name].mul_(h.ema_fast_decay).add_(t_float, alpha=1.0 - h.ema_fast_decay)\n"
        "                if ema_slow_state is not None:\n"
        "                    ema_slow_state[name].mul_(h.ema_slow_decay).add_(t_float, alpha=1.0 - h.ema_slow_decay)\n",
        name,
    )
    source = _replace_unique(
        source,
        "    # Weight averaging\n"
        "    log(\"ema:applying EMA weights\")\n"
        "    current_state = base_model.state_dict()\n"
        "    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}\n"
        "    base_model.load_state_dict(avg_state, strict=True)\n",
        "    # Weight averaging / selector\n"
        "    current_state = {name: t.detach().clone() for name, t in base_model.state_dict().items()}\n"
        "    candidate_states = [\n"
        "        (\"raw\", current_state),\n"
        "        (\"ema\", {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}),\n"
        "    ]\n"
        "    if ema_fast_state is not None:\n"
        "        candidate_states.append(\n"
        "            (\"ema_fast\", {name: t.to(dtype=current_state[name].dtype) for name, t in ema_fast_state.items()})\n"
        "        )\n"
        "    if ema_slow_state is not None:\n"
        "        candidate_states.append(\n"
        "            (\"ema_slow\", {name: t.to(dtype=current_state[name].dtype) for name, t in ema_slow_state.items()})\n"
        "        )\n"
        "    selector_model = model if h.distributed else compiled_model\n"
        "    best_name = \"raw\"\n"
        "    best_state = current_state\n"
        "    best_bpb = None\n"
        "    for candidate_name, candidate_state in candidate_states:\n"
        "        base_model.load_state_dict(candidate_state, strict=True)\n"
        "        val_loss_sel, val_bpb_sel = eval_val(h, device, val_data, selector_model)\n"
        "        log(f\"ema_selector:candidate={candidate_name} val_loss:{val_loss_sel:.6f} val_bpb:{val_bpb_sel:.6f}\")\n"
        "        if best_bpb is None or val_bpb_sel < best_bpb:\n"
        "            best_name = candidate_name\n"
        "            best_state = {name: t.detach().clone() for name, t in base_model.state_dict().items()}\n"
        "            best_bpb = val_bpb_sel\n"
        "    log(f\"ema_selector:chosen={best_name} val_bpb:{best_bpb:.6f}\")\n"
        "    base_model.load_state_dict(best_state, strict=True)\n",
        name,
    )
    return source


def patch_quant_clip_anneal(source: str) -> str:
    name = "quant_clip_anneal"
    source = _insert_after_unique(
        source,
        "    embed_clip_sigmas = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))\n",
        "    quant_clip_anneal_start = float(os.environ.get('QUANT_CLIP_ANNEAL_START', '0.55'))\n"
        "    quant_clip_anneal_full_at = float(os.environ.get('QUANT_CLIP_ANNEAL_FULL_AT', '0.9'))\n"
        "    quant_clip_anneal_alpha = float(os.environ.get('QUANT_CLIP_ANNEAL_ALPHA', '0.0'))\n"
        "    quant_embed_anneal_alpha = float(os.environ.get('QUANT_EMBED_ANNEAL_ALPHA', '0.0'))\n",
        name,
    )
    source = _replace_unique(
        source,
        "# ----------------------------------------\n"
        "# Optimization\n"
        "# ----------------------------------------\n",
        "def apply_quant_clip_anneal(model: nn.Module, h: Hyperparameters, strength: float) -> None:\n"
        "    if strength <= 0:\n"
        "        return\n"
        "    with torch.no_grad():\n"
        "        for name, param in model.named_parameters():\n"
        "            if param.ndim != 2 or not param.is_floating_point():\n"
        "                continue\n"
        "            if \"tok_emb\" in name:\n"
        "                alpha = min(h.quant_embed_anneal_alpha * strength, 1.0)\n"
        "                clip_sigmas = h.embed_clip_sigmas\n"
        "            elif any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):\n"
        "                continue\n"
        "            else:\n"
        "                alpha = min(h.quant_clip_anneal_alpha * strength, 1.0)\n"
        "                clip_sigmas = h.matrix_clip_sigmas\n"
        "            if alpha <= 0:\n"
        "                continue\n"
        "            t32 = param.data.float()\n"
        "            row_std = t32.std(dim=1, keepdim=True).clamp_min(1e-6)\n"
        "            row_clip = clip_sigmas * row_std\n"
        "            clamped = t32.clamp(min=-row_clip, max=row_clip)\n"
        "            param.data.copy_(torch.lerp(t32, clamped, alpha).to(param.dtype))\n"
        "\n"
        "# ----------------------------------------\n"
        "# Optimization\n"
        "# ----------------------------------------\n",
        name,
    )
    source = _replace_unique(
        source,
        "        train_loss = step_fn(step, scale)\n",
        "        train_loss = step_fn(step, scale)\n"
        "        if h.quant_clip_anneal_alpha > 0 or h.quant_embed_anneal_alpha > 0:\n"
        "            if frac >= h.quant_clip_anneal_start:\n"
        "                if h.quant_clip_anneal_full_at <= h.quant_clip_anneal_start:\n"
        "                    clip_strength = 1.0\n"
        "                else:\n"
        "                    clip_strength = min(\n"
        "                        (frac - h.quant_clip_anneal_start)\n"
        "                        / max(h.quant_clip_anneal_full_at - h.quant_clip_anneal_start, 1e-6),\n"
        "                        1.0,\n"
        "                    )\n"
        "                apply_quant_clip_anneal(base_model, h, clip_strength)\n",
        name,
    )
    return source


def patch_sensitivity_bits(source: str) -> str:
    name = "sensitivity_bits"
    source = _insert_after_unique(
        source,
        "    embed_bits = int(os.environ.get('EMBED_BITS', 8))\n",
        "    sensitivity_bits_enabled = bool(int(os.environ.get('SENSITIVITY_BITS_ENABLED', '0')))\n"
        "    sensitivity_topk = int(os.environ.get('SENSITIVITY_TOPK', '0'))\n"
        "    sensitivity_bottomk = int(os.environ.get('SENSITIVITY_BOTTOMK', '0'))\n"
        "    sensitivity_high_bits = int(os.environ.get('SENSITIVITY_HIGH_BITS', '7'))\n"
        "    sensitivity_low_bits = int(os.environ.get('SENSITIVITY_LOW_BITS', '5'))\n",
        name,
    )
    source = _replace_unique(
        source,
        "    meta: dict[str, object] = {}\n",
        "    meta: dict[str, object] = {}\n"
        "    matrix_bit_overrides: dict[str, int] = {}\n"
        "    if h.sensitivity_bits_enabled:\n"
        "        scored: list[tuple[float, str]] = []\n"
        "        for name, hessian in hessians.items():\n"
        "            if \"tok_emb\" in name:\n"
        "                continue\n"
        "            diag = hessian.diag().float()\n"
        "            scored.append((float(diag.mean().item()), name))\n"
        "        scored.sort(key=lambda item: item[0])\n"
        "        used: set[str] = set()\n"
        "        bottomk = max(h.sensitivity_bottomk, 0)\n"
        "        topk = max(h.sensitivity_topk, 0)\n"
        "        if bottomk > 0:\n"
        "            for _, matrix_name in scored[:bottomk]:\n"
        "                matrix_bit_overrides[matrix_name] = h.sensitivity_low_bits\n"
        "                used.add(matrix_name)\n"
        "        if topk > 0:\n"
        "            for _, matrix_name in reversed(scored[-topk:]):\n"
        "                if matrix_name in used:\n"
        "                    continue\n"
        "                matrix_bit_overrides[matrix_name] = h.sensitivity_high_bits\n"
        "        if matrix_bit_overrides:\n"
        "            log(\"Sensitivity-aware bits:\")\n"
        "            for matrix_name in sorted(matrix_bit_overrides):\n"
        "                log(f\"  {matrix_name}: int{matrix_bit_overrides[matrix_name]}\")\n",
        name,
    )
    source = _replace_unique(
        source,
        "        bits = h.embed_bits if \"tok_emb\" in name else h.matrix_bits\n",
        "        bits = h.embed_bits if \"tok_emb\" in name else matrix_bit_overrides.get(name, h.matrix_bits)\n",
        name,
    )
    return source


def patch_phase_split_optim(source: str) -> str:
    name = "phase_split_optim"
    source = _insert_after_unique(
        source,
        "    ema_decay = float(os.environ.get('EMA_DECAY', 0.997))\n",
        "    loop_matrix_lr_scale = float(os.environ.get('LOOP_MATRIX_LR_SCALE', '1.0'))\n"
        "    loop_scalar_lr_scale = float(os.environ.get('LOOP_SCALAR_LR_SCALE', '1.0'))\n"
        "    loop_token_lr_scale = float(os.environ.get('LOOP_TOKEN_LR_SCALE', '1.0'))\n"
        "    loop_head_lr_scale = float(os.environ.get('LOOP_HEAD_LR_SCALE', '1.0'))\n"
        "    loop_muon_wd_scale = float(os.environ.get('LOOP_MUON_WD_SCALE', '1.0'))\n",
        name,
    )
    source = _replace_unique(
        source,
        "        for opt in optimizers:\n"
        "            for group in opt.param_groups:\n"
        "                group[\"lr\"] = group[\"base_lr\"] * lr_scale\n",
        "        matrix_scale = h.loop_matrix_lr_scale if base_model.looping_active else 1.0\n"
        "        scalar_scale = h.loop_scalar_lr_scale if base_model.looping_active else 1.0\n"
        "        token_scale = h.loop_token_lr_scale if base_model.looping_active else 1.0\n"
        "        head_scale = h.loop_head_lr_scale if base_model.looping_active else 1.0\n"
        "        muon_wd_scale = h.loop_muon_wd_scale if base_model.looping_active else 1.0\n"
        "        for group in optimizers.optimizer_tok.param_groups:\n"
        "            group[\"lr\"] = group[\"base_lr\"] * lr_scale * token_scale\n"
        "        if optimizers.optimizer_head is not None:\n"
        "            for group in optimizers.optimizer_head.param_groups:\n"
        "                group[\"lr\"] = group[\"base_lr\"] * lr_scale * head_scale\n"
        "        for group in optimizers.optimizer_muon.param_groups:\n"
        "            group[\"lr\"] = group[\"base_lr\"] * lr_scale * matrix_scale\n"
        "            group[\"weight_decay\"] = h.muon_wd * muon_wd_scale\n"
        "        for group in optimizers.optimizer_scalar.param_groups:\n"
        "            group[\"lr\"] = group[\"base_lr\"] * lr_scale * scalar_scale\n",
        name,
    )
    return source
