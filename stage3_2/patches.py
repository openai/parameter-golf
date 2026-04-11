"""
stage3_2 patches — bounded state-conditioned controller policies.

The base for stage3_2 is the strong static frontier-like script in
stage3_2/base_train_gpt.py. This patch file injects a dynamic controller
that can vary a small set of actions based on training state.
"""

from __future__ import annotations


def _replace_unique(source: str, old: str, new: str, patch_name: str) -> str:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"[{patch_name}] target string not found:\n{old[:200]!r}")
    if count > 1:
        raise ValueError(f"[{patch_name}] ambiguous: found {count} occurrences:\n{old[:200]!r}")
    return source.replace(old, new)


def patch_state_controller(source: str) -> str:
    name = "state_controller"
    source = _replace_unique(
        source,
        "import copy\n"
        "import glob\n"
        "import io\n",
        "import copy\n"
        "import glob\n"
        "import io\n"
        "import json\n",
        name,
    )
    source = _replace_unique(
        source,
        "class CastedLinear(nn.Linear):\n"
        "    _qat_enabled: bool = False\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        w = self.weight.to(x.dtype)\n"
        "        if CastedLinear._qat_enabled and self.training and w.ndim == 2:\n"
        "            with torch.no_grad():\n"
        "                w32 = self.weight.float()\n"
        "                row_max = w32.abs().amax(dim=1)\n"
        "                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)\n"
        "                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)\n"
        "            w = w + (w_q - w).detach()\n"
        "        bias = self.bias.to(x.dtype) if self.bias is not None else None\n"
        "        return F.linear(x, w, bias)\n",
        "class CastedLinear(nn.Linear):\n"
        "    _qat_enabled: bool = False\n"
        "    _qat_alpha: float = 1.0\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        w = self.weight.to(x.dtype)\n"
        "        if CastedLinear._qat_enabled and self.training and w.ndim == 2:\n"
        "            with torch.no_grad():\n"
        "                w32 = self.weight.float()\n"
        "                row_max = w32.abs().amax(dim=1)\n"
        "                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)\n"
        "                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)\n"
        "            alpha = float(max(0.0, min(1.0, CastedLinear._qat_alpha)))\n"
        "            if alpha >= 1.0:\n"
        "                w = w + (w_q - w).detach()\n"
        "            elif alpha > 0.0:\n"
        "                w_mix = w + alpha * (w_q - w)\n"
        "                w = w + (w_mix - w).detach()\n"
        "        bias = self.bias.to(x.dtype) if self.bias is not None else None\n"
        "        return F.linear(x, w, bias)\n",
        name,
    )
    source = _replace_unique(
        source,
        "def main() -> None:\n",
        "def export_surrogate_penalty(model: nn.Module, device: torch.device) -> Tensor:\n"
        "    penalty = torch.zeros((), device=device)\n"
        "    count = 0\n"
        "    for param in model.parameters():\n"
        "        if param.ndim != 2:\n"
        "            continue\n"
        "        w32 = param.float()\n"
        "        row_max = w32.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0 / 31.0)\n"
        "        scale = row_max / 31.0\n"
        "        w_q = torch.clamp(torch.round(w32 / scale), -32, 31) * scale\n"
        "        penalty = penalty + (w32 - w_q).square().mean()\n"
        "        count += 1\n"
        "    return penalty / max(count, 1)\n"
        "\n"
        "\n"
        "def run_prequant_ttt(\n"
        "    base_model: nn.Module,\n"
        "    args: Hyperparameters,\n"
        "    device: torch.device,\n"
        "    val_tokens: Tensor,\n"
        "    freeze_blocks: int,\n"
        "    epochs: int,\n"
        "    lr: float,\n"
        "    stride: int,\n"
        "    max_chunks: int,\n"
        "    block_lr_decay: float,\n"
        "    grad_clip_norm: float,\n"
        "    log0,\n"
        ") -> None:\n"
        "    if epochs <= 0 or lr <= 0.0 or max_chunks <= 0:\n"
        "        return\n"
        "    total_blocks = len(base_model.blocks)\n"
        "    freeze_blocks = max(0, min(freeze_blocks, total_blocks))\n"
        "    original_requires_grad = {id(param): param.requires_grad for param in base_model.parameters()}\n"
        "    for param in base_model.parameters():\n"
        "        param.requires_grad_(False)\n"
        "    param_groups: list[dict[str, object]] = []\n"
        "    for block_idx, block in enumerate(base_model.blocks):\n"
        "        if block_idx < freeze_blocks:\n"
        "            continue\n"
        "        block_params = [param for param in block.parameters() if param.numel() > 0]\n"
        "        if not block_params:\n"
        "            continue\n"
        "        for param in block_params:\n"
        "            param.requires_grad_(True)\n"
        "        depth_from_top = max(total_blocks - 1 - block_idx, 0)\n"
        "        group_lr = lr * (block_lr_decay ** depth_from_top)\n"
        "        param_groups.append({\"params\": block_params, \"lr\": group_lr})\n"
        "    final_norm_params = [param for param in base_model.final_norm.parameters() if param.numel() > 0]\n"
        "    if final_norm_params:\n"
        "        for param in final_norm_params:\n"
        "            param.requires_grad_(True)\n"
        "        param_groups.append({\"params\": final_norm_params, \"lr\": lr})\n"
        "    if base_model.lm_head is not None:\n"
        "        head_params = [base_model.lm_head.weight]\n"
        "        for param in head_params:\n"
        "            param.requires_grad_(True)\n"
        "        param_groups.append({\"params\": head_params, \"lr\": lr})\n"
        "    if not param_groups:\n"
        "        for param in base_model.parameters():\n"
        "            param.requires_grad_(original_requires_grad[id(param)])\n"
        "        return\n"
        "    optimizer_ttt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=args.adam_eps, fused=True)\n"
        "    seq_len = args.train_seq_len\n"
        "    token_budget = max(val_tokens.numel() - seq_len - 1, 1)\n"
        "    stride = max(stride, 1)\n"
        "    total_chunks = min(max_chunks, max(token_budget // stride, 1))\n"
        "    was_training = base_model.training\n"
        "    base_model.train()\n"
        "    try:\n"
        "        for epoch_idx in range(epochs):\n"
        "            epoch_loss = 0.0\n"
        "            for chunk_idx in range(total_chunks):\n"
        "                start = min(chunk_idx * stride, max(val_tokens.numel() - seq_len - 1, 0))\n"
        "                local = val_tokens[start : start + seq_len + 1].to(device=device, dtype=torch.int64, non_blocking=True)\n"
        "                x = local[:-1].view(1, -1)\n"
        "                y = local[1:].view(1, -1)\n"
        "                optimizer_ttt.zero_grad(set_to_none=True)\n"
        "                with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "                    loss = base_model(x, y)\n"
        "                loss.backward()\n"
        "                if grad_clip_norm > 0:\n"
        "                    torch.nn.utils.clip_grad_norm_([param for group in param_groups for param in group[\"params\"]], grad_clip_norm)\n"
        "                optimizer_ttt.step()\n"
        "                epoch_loss += float(loss.detach().item())\n"
        "            log0(f\"state_controller:ttt epoch:{epoch_idx + 1}/{epochs} freeze_blocks:{freeze_blocks} chunks:{total_chunks} avg_loss:{epoch_loss / max(total_chunks, 1):.6f}\")\n"
        "    finally:\n"
        "        base_model.train(was_training)\n"
        "        for param in base_model.parameters():\n"
        "            param.requires_grad_(original_requires_grad[id(param)])\n"
        "\n"
        "\n"
        "def main() -> None:\n",
        name,
    )
    source = _replace_unique(
        source,
        "    optimizer_tok = torch.optim.AdamW(\n"
        "        tok_params,\n"
        "        betas=(args.beta1, args.beta2),\n"
        "        eps=args.adam_eps,\n"
        "        weight_decay=args.adam_wd,\n"
        "        fused=True,\n"
        "    )\n",
        "    optimizer_tok = torch.optim.AdamW(\n"
        "        tok_params,\n"
        "        betas=(args.beta1, args.beta2),\n"
        "        eps=args.adam_eps,\n"
        "        weight_decay=args.adam_wd,\n"
        "        fused=True,\n"
        "    )\n"
        "    for group in optimizer_tok.param_groups:\n"
        "        group[\"family\"] = \"token\"\n",
        name,
    )
    source = _replace_unique(
        source,
        "    for group in optimizer_muon.param_groups:\n"
        "        group[\"base_lr\"] = args.matrix_lr\n",
        "    for group in optimizer_muon.param_groups:\n"
        "        group[\"base_lr\"] = args.matrix_lr\n"
        "        group[\"family\"] = \"matrix\"\n",
        name,
    )
    source = _replace_unique(
        source,
        "    optimizer_scalar = torch.optim.AdamW(\n"
        "        [{\"params\": scalar_params, \"lr\": args.scalar_lr, \"base_lr\": args.scalar_lr}],\n"
        "        betas=(args.beta1, args.beta2),\n"
        "        eps=args.adam_eps,\n"
        "        weight_decay=args.adam_wd,\n"
        "        fused=True,\n"
        "    )\n",
        "    optimizer_scalar = torch.optim.AdamW(\n"
        "        [{\"params\": scalar_params, \"lr\": args.scalar_lr, \"base_lr\": args.scalar_lr}],\n"
        "        betas=(args.beta1, args.beta2),\n"
        "        eps=args.adam_eps,\n"
        "        weight_decay=args.adam_wd,\n"
        "        fused=True,\n"
        "    )\n"
        "    for group in optimizer_scalar.param_groups:\n"
        "        group[\"family\"] = \"scalar\"\n",
        name,
    )
    source = _replace_unique(
        source,
        "    if base_model.lm_head is not None:\n"
        "        optimizer_head = torch.optim.Adam(\n"
        "            [{\"params\": [base_model.lm_head.weight], \"lr\": args.head_lr, \"base_lr\": args.head_lr}],\n"
        "            betas=(args.beta1, args.beta2),\n"
        "            eps=args.adam_eps,\n"
        "            fused=True,\n"
        "        )\n"
        "        optimizers.insert(1, optimizer_head)\n",
        "    if base_model.lm_head is not None:\n"
        "        optimizer_head = torch.optim.Adam(\n"
        "            [{\"params\": [base_model.lm_head.weight], \"lr\": args.head_lr, \"base_lr\": args.head_lr}],\n"
        "            betas=(args.beta1, args.beta2),\n"
        "            eps=args.adam_eps,\n"
        "            fused=True,\n"
        "        )\n"
        "        for group in optimizer_head.param_groups:\n"
        "            group[\"family\"] = \"head\"\n"
        "        optimizers.insert(1, optimizer_head)\n",
        name,
    )
    source = _replace_unique(
        source,
        "    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n"
        "    def zero_grad_all() -> None:\n",
        "    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n"
        "    controller_spec = json.loads(os.environ.get(\"CTRL_SPEC_JSON\", \"{}\") or \"{}\")\n"
        "    controller_enable = bool(int(os.environ.get(\"CTRL_ENABLE\", \"0\"))) and bool(controller_spec)\n"
        "    ctrl_phase_bounds = controller_spec.get(\"phase_boundaries\", [0.6, 0.82])\n"
        "    ctrl_phases = controller_spec.get(\"phase_defaults\", {})\n"
        "    ctrl_gates = controller_spec.get(\"gates\", [])\n"
        "    ctrl_snapshot = controller_spec.get(\"snapshot\", {})\n"
        "    ctrl_pulse = controller_spec.get(\"pulse\", {})\n"
        "    train_loss_ema = None\n"
        "    prev_train_loss_ema = None\n"
        "    snapshot_candidates: list[tuple[str, dict[str, Tensor]]] = []\n"
        "\n"
        "    def zero_grad_all() -> None:\n",
        name,
    )
    source = _replace_unique(
        source,
        "    def lr_mul(step: int, elapsed_ms: float) -> float:\n"
        "        if args.warmdown_iters <= 0:\n"
        "            return 1.0\n"
        "        if max_wallclock_ms is None:\n"
        "            warmdown_start = max(args.iterations - args.warmdown_iters, 0)\n"
        "            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0\n"
        "        step_ms = elapsed_ms / max(step, 1)\n"
        "        warmdown_ms = args.warmdown_iters * step_ms\n"
        "        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)\n"
        "        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0\n",
        "    def lr_mul(step: int, elapsed_ms: float) -> float:\n"
        "        if args.warmdown_iters <= 0:\n"
        "            return 1.0\n"
        "        if max_wallclock_ms is None:\n"
        "            warmdown_start = max(args.iterations - args.warmdown_iters, 0)\n"
        "            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0\n"
        "        step_ms = elapsed_ms / max(step, 1)\n"
        "        warmdown_ms = args.warmdown_iters * step_ms\n"
        "        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)\n"
        "        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0\n"
        "\n"
        "    def ctrl_phase(progress: float) -> str:\n"
        "        if not controller_enable:\n"
        "            return \"base\"\n"
        "        b1 = float(ctrl_phase_bounds[0]) if len(ctrl_phase_bounds) > 0 else 0.6\n"
        "        b2 = float(ctrl_phase_bounds[1]) if len(ctrl_phase_bounds) > 1 else 0.82\n"
        "        if progress < b1:\n"
        "            return \"early\"\n"
        "        if progress < b2:\n"
        "            return \"mid\"\n"
        "        return \"late\"\n"
        "\n"
        "    def active_actions(progress: float, step_avg_ms: float, train_loss_slope: float, scale: float) -> dict[str, object]:\n"
        "        phase = ctrl_phase(progress)\n"
        "        actions = dict(ctrl_phases.get(phase, {}))\n"
        "        features = {\n"
        "            \"progress\": progress,\n"
        "            \"step_avg_ms\": step_avg_ms,\n"
        "            \"train_loss_slope\": train_loss_slope,\n"
        "            \"warmdown_frac\": 1.0 - scale,\n"
        "            \"scale\": scale,\n"
        "        }\n"
        "        for gate in ctrl_gates:\n"
        "            feature = str(gate.get(\"feature\", \"\"))\n"
        "            op = str(gate.get(\"op\", \">\"))\n"
        "            threshold = float(gate.get(\"threshold\", 0.0))\n"
        "            value = features.get(feature)\n"
        "            if value is None:\n"
        "                continue\n"
        "            active = (value > threshold) if op == \">\" else (value < threshold)\n"
        "            if active:\n"
        "                actions[str(gate[\"action\"])] = gate[\"value\"]\n"
        "        return actions\n"
        "\n"
        "    def clone_state_dict_cpu(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:\n"
        "        return {state_name: tensor.detach().cpu().clone() for state_name, tensor in state_dict.items()}\n"
        "\n"
        "    def maybe_capture_snapshot(current_step: int, progress: float, actions: dict[str, object]) -> None:\n"
        "        if not controller_enable:\n"
        "            return\n"
        "        every = int(actions.get(\"checkpoint_capture_rate\", ctrl_snapshot.get(\"every\", 0)) or 0)\n"
        "        start_frac = float(ctrl_snapshot.get(\"start_frac\", 0.72))\n"
        "        if every <= 0 or progress < start_frac:\n"
        "            return\n"
        "        if current_step % every != 0 and current_step != args.iterations:\n"
        "            return\n"
        "        snapshot_candidates.append((f\"step_{current_step}\", clone_state_dict_cpu(base_model.state_dict())))\n"
        "        keep_last_k = int(ctrl_snapshot.get(\"last_k\", 6))\n"
        "        if len(snapshot_candidates) > keep_last_k:\n"
        "            del snapshot_candidates[0 : len(snapshot_candidates) - keep_last_k]\n"
        "\n"
        "    def score_state_candidate(tag: str, state_dict: dict[str, Tensor], score_mode: str) -> tuple[float, float]:\n"
        "        current_state = clone_state_dict_cpu(base_model.state_dict())\n"
        "        try:\n"
        "            base_model.load_state_dict(state_dict, strict=True)\n"
        "            if score_mode == \"raw\":\n"
        "                cand_loss, cand_bpb = eval_val(\n"
        "                    args, compiled_model, rank, world_size, device, grad_accum_steps,\n"
        "                    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n"
        "                    eval_seq_len=effective_eval_seq_len,\n"
        "                )\n"
        "            else:\n"
        "                export_sd_local = {k: v for k, v in base_model.state_dict().items() if \"mtp_heads\" not in k}\n"
        "                sd_cpu_local = {k: v.detach().cpu() for k, v in export_sd_local.items()}\n"
        "                quant_result_local, quant_meta_local = mixed_quantize_int6(sd_cpu_local, {\"mlp\", \"attn\"})\n"
        "                deq_state_local = dequantize_mixed_int6(quant_result_local, quant_meta_local, sd_cpu_local)\n"
        "                eval_model_local = GPT(\n"
        "                    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,\n"
        "                    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,\n"
        "                    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,\n"
        "                    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,\n"
        "                    mtp_num_heads=0, mtp_loss_weight=0.0,\n"
        "                    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,\n"
        "                    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,\n"
        "                    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,\n"
        "                ).to(device).bfloat16()\n"
        "                for module in eval_model_local.modules():\n"
        "                    if isinstance(module, CastedLinear):\n"
        "                        module.float()\n"
        "                restore_low_dim_params_to_fp32(eval_model_local)\n"
        "                eval_model_local.load_state_dict(deq_state_local, strict=True)\n"
        "                compiled_eval_local = torch.compile(eval_model_local, dynamic=False, fullgraph=True)\n"
        "                cand_loss, cand_bpb = eval_val(\n"
        "                    args, compiled_eval_local, rank, world_size, device, grad_accum_steps,\n"
        "                    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n"
        "                    eval_seq_len=effective_eval_seq_len,\n"
        "                )\n"
        "            log0(f\"state_controller:snapshot:{tag} score_mode:{score_mode} val_bpb:{cand_bpb:.8f}\")\n"
        "            return cand_loss, cand_bpb\n"
        "        finally:\n"
        "            base_model.load_state_dict(current_state, strict=True)\n",
        name,
    )
    source = _replace_unique(
        source,
        "    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}\n"
        "    ema_decay = 0.997\n",
        "    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}\n"
        "    ema_decay = 0.997\n"
        "    current_actions: dict[str, object] = {}\n",
        name,
    )
    source = _replace_unique(
        source,
        "        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n"
        "        scale = lr_mul(step, elapsed_ms)\n"
        "        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:\n"
        "            CastedLinear._qat_enabled = True\n"
        "            log0(f\"late_qat:enabled step:{step} scale:{scale:.4f}\")\n",
        "        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n"
        "        scale = lr_mul(step, elapsed_ms)\n"
        "        progress = step / max(args.iterations, 1) if args.iterations > 0 else 1.0\n"
        "        step_avg_ms = elapsed_ms / max(step, 1)\n"
        "        train_loss_slope = 0.0 if train_loss_ema is None or prev_train_loss_ema is None else (train_loss_ema - prev_train_loss_ema)\n"
        "        current_actions = active_actions(progress, step_avg_ms, train_loss_slope, scale)\n"
        "        ema_decay = float(current_actions.get(\"ema_decay\", ema_decay))\n"
        "        qat_alpha = float(current_actions.get(\"qat_alpha\", 1.0 if (args.late_qat_threshold > 0 and scale < args.late_qat_threshold) else 0.0))\n"
        "        CastedLinear._qat_enabled = qat_alpha > 0.0\n"
        "        CastedLinear._qat_alpha = qat_alpha\n"
        "        if qat_alpha > 0.0 and step % max(args.train_log_every, 50) == 0:\n"
        "            log0(f\"state_controller:qat_alpha:{qat_alpha:.3f} step:{step} progress:{progress:.3f}\")\n",
        name,
    )
    source = _replace_unique(
        source,
        "        for opt in optimizers:\n"
        "            for group in opt.param_groups:\n"
        "                group[\"lr\"] = group[\"base_lr\"] * scale\n",
        "        token_lr_mult = float(current_actions.get(\"token_lr_mult\", 1.0))\n"
        "        matrix_lr_mult = float(current_actions.get(\"matrix_lr_mult\", 1.0))\n"
        "        scalar_lr_mult = float(current_actions.get(\"scalar_lr_mult\", 1.0))\n"
        "        head_lr_mult = float(current_actions.get(\"head_lr_mult\", 1.0))\n"
        "        freeze_token = bool(int(current_actions.get(\"freeze_token\", 0)))\n"
        "        freeze_head = bool(int(current_actions.get(\"freeze_head\", 0)))\n"
        "        for opt in optimizers:\n"
        "            for group in opt.param_groups:\n"
        "                family = group.get(\"family\", \"other\")\n"
        "                mult = 1.0\n"
        "                if family == \"token\":\n"
        "                    mult = 0.0 if freeze_token else token_lr_mult\n"
        "                elif family == \"matrix\":\n"
        "                    mult = matrix_lr_mult\n"
        "                elif family == \"scalar\":\n"
        "                    mult = scalar_lr_mult\n"
        "                elif family == \"head\":\n"
        "                    mult = 0.0 if freeze_head else head_lr_mult\n"
        "                group[\"lr\"] = group[\"base_lr\"] * scale * mult\n",
        name,
    )
    source = _replace_unique(
        source,
        "        for micro_step in range(grad_accum_steps):\n"
        "            if distributed:\n"
        "                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1\n"
        "            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)\n"
        "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "                loss = model(x, y)\n"
        "            train_loss += loss.detach()\n"
        "            (loss * grad_scale).backward()\n",
        "        pulse_every = int(ctrl_pulse.get(\"every\", 0) or 0)\n"
        "        pulse_start = float(ctrl_pulse.get(\"late_start\", 0.72))\n"
        "        pulse_mode = str(ctrl_pulse.get(\"mode\", \"\"))\n"
        "        pulse_weight = float(ctrl_pulse.get(\"weight\", 0.0))\n"
        "        pulse_active = pulse_every > 0 and progress >= pulse_start and (step % pulse_every == 0)\n"
        "        export_weight = float(current_actions.get(\"export_surrogate_weight\", 0.0))\n"
        "        for micro_step in range(grad_accum_steps):\n"
        "            if distributed:\n"
        "                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1\n"
        "            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)\n"
        "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "                loss = model(x, y)\n"
        "                if export_weight > 0.0:\n"
        "                    loss = loss + export_weight * export_surrogate_penalty(base_model, device).to(loss.dtype)\n"
        "                if pulse_active and pulse_mode in {\"export_surrogate\", \"late_qat\"}:\n"
        "                    loss = loss + pulse_weight * export_surrogate_penalty(base_model, device).to(loss.dtype)\n"
        "            train_loss += loss.detach()\n"
        "            (loss * grad_scale).backward()\n",
        name,
    )
    source = _replace_unique(
        source,
        "        for opt in optimizers:\n"
        "            opt.step()\n"
        "        zero_grad_all()\n"
        "        # EMA update\n"
        "        with torch.no_grad():\n"
        "            for name, t in base_model.state_dict().items():\n"
        "                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)\n"
        "        step += 1\n",
        "        for opt in optimizers:\n"
        "            opt.step()\n"
        "        zero_grad_all()\n"
        "        # EMA update\n"
        "        with torch.no_grad():\n"
        "            for name, t in base_model.state_dict().items():\n"
        "                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)\n"
        "        step += 1\n"
        "        prev_train_loss_ema = train_loss_ema\n"
        "        train_loss_ema = train_loss.item() if train_loss_ema is None else (0.9 * train_loss_ema + 0.1 * train_loss.item())\n"
        "        maybe_capture_snapshot(step, progress, current_actions)\n",
        name,
    )
    source = _replace_unique(
        source,
        "    # Apply EMA weights (better than SWA alone per PR#401)\n"
        "    log0(\"ema:applying EMA weights\")\n"
        "    current_state = base_model.state_dict()\n"
        "    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}\n"
        "    base_model.load_state_dict(avg_state, strict=True)\n",
        "    if controller_enable and ctrl_snapshot:\n"
        "        score_mode = str(ctrl_snapshot.get(\"score\", \"deployed\"))\n"
        "        selection_mode = str(current_actions.get(\"checkpoint_selection_mode\", ctrl_snapshot.get(\"mode\", \"ema\")))\n"
        "        snapshot_candidates.append((\"raw_final\", clone_state_dict_cpu(base_model.state_dict())))\n"
        "        if selection_mode in {\"ema\", \"best_deployed_last_k\", \"best_raw_last_k\"}:\n"
        "            snapshot_candidates.append((\"ema_final\", {state_name: tensor.detach().cpu().clone() for state_name, tensor in ema_state.items()}))\n"
        "        if selection_mode == \"last\":\n"
        "            chosen_tag, chosen_state = snapshot_candidates[-1]\n"
        "            log0(f\"state_controller:choose_snapshot mode:last tag:{chosen_tag}\")\n"
        "            base_model.load_state_dict(chosen_state, strict=True)\n"
        "        elif selection_mode == \"ema\":\n"
        "            chosen_state = {state_name: tensor.to(dtype=base_model.state_dict()[state_name].dtype) for state_name, tensor in ema_state.items()}\n"
        "            log0(\"state_controller:choose_snapshot mode:ema tag:ema_final\")\n"
        "            base_model.load_state_dict(chosen_state, strict=True)\n"
        "        else:\n"
        "            best_tag = None\n"
        "            best_state = None\n"
        "            best_bpb = None\n"
        "            candidate_score_mode = \"raw\" if selection_mode == \"best_raw_last_k\" else score_mode\n"
        "            for tag, candidate_state in snapshot_candidates:\n"
        "                _, candidate_bpb = score_state_candidate(tag, candidate_state, candidate_score_mode)\n"
        "                if best_bpb is None or candidate_bpb < best_bpb:\n"
        "                    best_bpb = candidate_bpb\n"
        "                    best_tag = tag\n"
        "                    best_state = candidate_state\n"
        "            if best_state is not None:\n"
        "                log0(f\"state_controller:choose_snapshot mode:{selection_mode} tag:{best_tag} val_bpb:{best_bpb:.8f}\")\n"
        "                base_model.load_state_dict(best_state, strict=True)\n"
        "        if bool(int(current_actions.get(\"ttt_enable\", 0))):\n"
        "            ttt_epochs = int(current_actions.get(\"ttt_epochs\", 0))\n"
        "            ttt_lr = float(current_actions.get(\"ttt_lr\", 0.0))\n"
        "            ttt_freeze_blocks = int(current_actions.get(\"ttt_freeze_blocks\", len(base_model.blocks)))\n"
        "            ttt_stride = int(current_actions.get(\"ttt_stride\", args.train_seq_len // 8 if args.train_seq_len >= 8 else 1))\n"
        "            ttt_chunks = int(current_actions.get(\"ttt_chunks\", 16))\n"
        "            ttt_block_lr_decay = float(current_actions.get(\"ttt_block_lr_decay\", 1.0))\n"
        "            ttt_grad_clip_norm = float(current_actions.get(\"ttt_grad_clip_norm\", args.grad_clip_norm))\n"
        "            log0(\n"
        "                f\"state_controller:ttt enable:1 epochs:{ttt_epochs} freeze_blocks:{ttt_freeze_blocks} lr:{ttt_lr:.6g} stride:{ttt_stride} chunks:{ttt_chunks} block_lr_decay:{ttt_block_lr_decay:.4f}\"\n"
        "            )\n"
        "            run_prequant_ttt(\n"
        "                base_model,\n"
        "                args,\n"
        "                device,\n"
        "                val_tokens,\n"
        "                freeze_blocks=ttt_freeze_blocks,\n"
        "                epochs=ttt_epochs,\n"
        "                lr=ttt_lr,\n"
        "                stride=ttt_stride,\n"
        "                max_chunks=ttt_chunks,\n"
        "                block_lr_decay=ttt_block_lr_decay,\n"
        "                grad_clip_norm=ttt_grad_clip_norm,\n"
        "                log0=log0,\n"
        "            )\n"
        "    else:\n"
        "        # Apply EMA weights (better than SWA alone per PR#401)\n"
        "        log0(\"ema:applying EMA weights\")\n"
        "        current_state = base_model.state_dict()\n"
        "        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}\n"
        "        base_model.load_state_dict(avg_state, strict=True)\n",
        name,
    )
    return source


PATCH_ORDER = {
    "state_controller": 10,
}


PATCHES = {
    "state_controller": patch_state_controller,
}


def apply_patches(source: str, patch_names: list[str]) -> str:
    ordered = sorted(patch_names, key=lambda n: PATCH_ORDER.get(n, 100))
    for patch_name in ordered:
        source = PATCHES[patch_name](source)
    return source
