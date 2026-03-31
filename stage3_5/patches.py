"""
stage3_5 patches — event-triggered branch tournament.

This stage is larger than stage3_4 in the same realm:
- branch on training state, not only on fixed time
- define late finishers through a small DSL
- choose the best export-state portfolio within each branch
"""

from __future__ import annotations


def _replace_unique(source: str, old: str, new: str, patch_name: str) -> str:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"[{patch_name}] target string not found:\n{old[:200]!r}")
    if count > 1:
        raise ValueError(f"[{patch_name}] ambiguous: found {count} occurrences:\n{old[:200]!r}")
    return source.replace(old, new)


def patch_event_branch_tournament(source: str) -> str:
    name = "event_branch_tournament"
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
        "def eval_val_sliding(\n",
        "def clone_state_dict_cpu(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:\n"
        "    return {state_name: tensor.detach().cpu().clone() for state_name, tensor in state_dict.items()}\n"
        "\n"
        "\n"
        "def capture_loader_state(loader: DistributedTokenLoader) -> dict[str, int]:\n"
        "    return {\"file_idx\": int(loader.stream.file_idx), \"pos\": int(loader.stream.pos)}\n"
        "\n"
        "\n"
        "def restore_loader_state(loader: DistributedTokenLoader, pattern: str, state: dict[str, int]) -> None:\n"
        "    loader.stream = TokenStream(pattern)\n"
        "    loader.stream.file_idx = int(state[\"file_idx\"]) % len(loader.stream.files)\n"
        "    loader.stream.tokens = load_data_shard(loader.stream.files[loader.stream.file_idx])\n"
        "    loader.stream.pos = int(state[\"pos\"])\n"
        "\n"
        "\n"
        "def branch_export_eval(\n"
        "    args: Hyperparameters,\n"
        "    state_dict_cpu: dict[str, Tensor],\n"
        "    rank: int,\n"
        "    world_size: int,\n"
        "    device: torch.device,\n"
        "    val_tokens: Tensor,\n"
        "    base_bytes_lut: Tensor,\n"
        "    has_leading_space_lut: Tensor,\n"
        "    is_boundary_token_lut: Tensor,\n"
        "    effective_eval_seq_len: int,\n"
        ") -> tuple[float, float]:\n"
        "    quant_result, quant_meta = mixed_quantize_int6(state_dict_cpu, {\"mlp\", \"attn\"})\n"
        "    deq_state = dequantize_mixed_int6(quant_result, quant_meta, state_dict_cpu)\n"
        "    eval_model = GPT(\n"
        "        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,\n"
        "        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,\n"
        "        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,\n"
        "        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,\n"
        "        mtp_num_heads=0, mtp_loss_weight=0.0,\n"
        "        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,\n"
        "        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,\n"
        "        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,\n"
        "    ).to(device).bfloat16()\n"
        "    for module in eval_model.modules():\n"
        "        if isinstance(module, CastedLinear):\n"
        "            module.float()\n"
        "    restore_low_dim_params_to_fp32(eval_model)\n"
        "    typed_state = {state_name: tensor.to(dtype=eval_model.state_dict()[state_name].dtype) for state_name, tensor in deq_state.items()}\n"
        "    eval_model.load_state_dict(typed_state, strict=True)\n"
        "    val_loss, val_bpb = eval_val_sliding(\n"
        "        args, eval_model, rank, world_size, device,\n"
        "        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n"
        "        stride=args.eval_stride, batch_seqs=32, eval_seq_len=effective_eval_seq_len,\n"
        "    )\n"
        "    del eval_model\n"
        "    return val_loss, val_bpb\n"
        "\n"
        "\n"
        "def branch_export_portfolio_eval(\n"
        "    args: Hyperparameters,\n"
        "    base_model: nn.Module,\n"
        "    ema_state: dict[str, Tensor],\n"
        "    export_modes: list[str],\n"
        "    rank: int,\n"
        "    world_size: int,\n"
        "    device: torch.device,\n"
        "    val_tokens: Tensor,\n"
        "    base_bytes_lut: Tensor,\n"
        "    has_leading_space_lut: Tensor,\n"
        "    is_boundary_token_lut: Tensor,\n"
        "    effective_eval_seq_len: int,\n"
        ") -> tuple[str, float, float, dict[str, Tensor]]:\n"
        "    current_state = clone_state_dict_cpu(base_model.state_dict())\n"
        "    best_mode = \"raw\"\n"
        "    best_loss = float(\"inf\")\n"
        "    best_bpb = float(\"inf\")\n"
        "    best_state = current_state\n"
        "    for export_mode in export_modes:\n"
        "        if export_mode == \"ema\":\n"
        "            state_cpu = {state_name: tensor.detach().cpu().clone() for state_name, tensor in ema_state.items()}\n"
        "        else:\n"
        "            state_cpu = clone_state_dict_cpu(base_model.state_dict())\n"
        "        export_sd = {k: v for k, v in state_cpu.items() if \"mtp_heads\" not in k}\n"
        "        val_loss, val_bpb = branch_export_eval(\n"
        "            args, export_sd, rank, world_size, device,\n"
        "            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n"
        "            effective_eval_seq_len,\n"
        "        )\n"
        "        if val_bpb < best_bpb:\n"
        "            best_mode = export_mode\n"
        "            best_loss = val_loss\n"
        "            best_bpb = val_bpb\n"
        "            best_state = state_cpu\n"
        "    return best_mode, best_loss, best_bpb, best_state\n"
        "\n"
        "\n"
        "def expand_finisher(branch_policy: dict[str, object]) -> dict[str, object]:\n"
        "    presets = {\n"
        "        \"ema_heavy\": {\"ema_decay\": 0.9995, \"qat_alpha\": 0.0, \"export_surrogate_weight\": 0.0, \"token_lr_mult\": 1.0, \"matrix_lr_mult\": 1.0, \"scalar_lr_mult\": 1.0, \"head_lr_mult\": 1.0, \"export_modes\": [\"ema\", \"raw\"]},\n"
        "        \"deploy_qat\": {\"ema_decay\": 0.9990, \"qat_alpha\": 0.55, \"export_surrogate_weight\": 0.0007, \"token_lr_mult\": 1.0, \"matrix_lr_mult\": 1.0, \"scalar_lr_mult\": 1.0, \"head_lr_mult\": 1.0, \"export_modes\": [\"ema\", \"raw\"]},\n"
        "        \"family_split\": {\"ema_decay\": 0.9985, \"qat_alpha\": 0.0, \"export_surrogate_weight\": 0.0, \"token_lr_mult\": 0.0, \"matrix_lr_mult\": 1.1, \"scalar_lr_mult\": 0.6, \"head_lr_mult\": 0.45, \"export_modes\": [\"ema\", \"raw\"]},\n"
        "        \"raw_finish\": {\"ema_decay\": 0.9970, \"qat_alpha\": 0.0, \"export_surrogate_weight\": 0.0, \"token_lr_mult\": 1.0, \"matrix_lr_mult\": 1.0, \"scalar_lr_mult\": 1.0, \"head_lr_mult\": 1.0, \"export_modes\": [\"raw\", \"ema\"]},\n"
        "        \"aggressive_qat\": {\"ema_decay\": 0.9990, \"qat_alpha\": 0.75, \"export_surrogate_weight\": 0.0010, \"token_lr_mult\": 1.0, \"matrix_lr_mult\": 1.0, \"scalar_lr_mult\": 1.0, \"head_lr_mult\": 1.0, \"export_modes\": [\"ema\", \"raw\"]},\n"
        "        \"matrix_push\": {\"ema_decay\": 0.9980, \"qat_alpha\": 0.0, \"export_surrogate_weight\": 0.0, \"token_lr_mult\": 0.0, \"matrix_lr_mult\": 1.2, \"scalar_lr_mult\": 0.7, \"head_lr_mult\": 0.35, \"export_modes\": [\"ema\", \"raw\"]},\n"
        "    }\n"
        "    kind = str(branch_policy.get(\"kind\", \"ema_heavy\"))\n"
        "    expanded = dict(presets.get(kind, presets[\"ema_heavy\"]))\n"
        "    expanded.update(branch_policy)\n"
        "    if \"name\" not in expanded:\n"
        "        expanded[\"name\"] = kind\n"
        "    if \"export_modes\" not in expanded:\n"
        "        expanded[\"export_modes\"] = [\"ema\", \"raw\"]\n"
        "    return expanded\n"
        "\n"
        "\n"
        "def eval_val_sliding(\n",
        name,
    )
    source = _replace_unique(
        source,
        "    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n"
        "    def zero_grad_all() -> None:\n",
        "    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n"
        "    branch_spec = json.loads(os.environ.get(\"BRANCH_SPEC_JSON\", \"{}\") or \"{}\")\n"
        "    branch_enabled = bool(int(os.environ.get(\"BRANCH_ENABLE\", \"0\"))) and bool(branch_spec)\n"
        "    branch_trigger_cfg = branch_spec.get(\"trigger\", {})\n"
        "    branch_finishers = [expand_finisher(item) for item in branch_spec.get(\"finishers\", [])]\n"
        "    branch_triggered = False\n"
        "    branch_best_state: dict[str, Tensor] | None = None\n"
        "    branch_best_name: str | None = None\n"
        "    branch_best_mode: str | None = None\n"
        "    trunk_model_state: dict[str, Tensor] | None = None\n"
        "    trunk_ema_state: dict[str, Tensor] | None = None\n"
        "    trunk_optimizer_states: list[dict[str, object]] | None = None\n"
        "    trunk_loader_state: dict[str, int] | None = None\n"
        "    trunk_training_time_ms: float | None = None\n"
        "    trunk_step: int | None = None\n"
        "    recent_losses: list[float] = []\n"
        "    def zero_grad_all() -> None:\n",
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
        "        CastedLinear._qat_enabled = args.late_qat_threshold > 0 and scale < args.late_qat_threshold\n"
        "        CastedLinear._qat_alpha = 1.0 if CastedLinear._qat_enabled else 0.0\n"
        "        if CastedLinear._qat_enabled and step % max(args.train_log_every, 50) == 0:\n"
        "            log0(f\"late_qat:enabled step:{step} scale:{scale:.4f}\")\n",
        name,
    )
    source = _replace_unique(
        source,
        "        train_loss /= grad_accum_steps\n",
        "        train_loss /= grad_accum_steps\n"
        "        recent_losses.append(float(train_loss.item()))\n"
        "        if len(recent_losses) > 8:\n"
        "            recent_losses.pop(0)\n",
        name,
    )
    source = _replace_unique(
        source,
        "        if stop_after_step is None and reached_cap:\n"
        "            stop_after_step = step\n",
        "        trigger_min_frac = float(branch_trigger_cfg.get(\"min_frac\", 0.65))\n"
        "        trigger_max_frac = float(branch_trigger_cfg.get(\"max_frac\", 0.84))\n"
        "        trigger_scale_below = float(branch_trigger_cfg.get(\"scale_below\", 0.16))\n"
        "        plateau_tol = float(branch_trigger_cfg.get(\"plateau_tol\", 0.010))\n"
        "        plateau_ready = False\n"
        "        if len(recent_losses) >= 6:\n"
        "            plateau_ready = (recent_losses[0] - recent_losses[-1]) < plateau_tol\n"
        "        branch_ready = (\n"
        "            branch_enabled and not branch_triggered and max_wallclock_ms is not None and branch_finishers\n"
        "            and approx_training_time_ms >= max_wallclock_ms * trigger_min_frac\n"
        "            and (scale <= trigger_scale_below or plateau_ready or approx_training_time_ms >= max_wallclock_ms * trigger_max_frac)\n"
        "        )\n"
        "        if branch_ready:\n"
        "            branch_triggered = True\n"
        "            trunk_model_state = clone_state_dict_cpu(base_model.state_dict())\n"
        "            trunk_ema_state = {state_name: tensor.detach().cpu().clone() for state_name, tensor in ema_state.items()}\n"
        "            trunk_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]\n"
        "            trunk_loader_state = capture_loader_state(train_loader)\n"
        "            trunk_training_time_ms = approx_training_time_ms\n"
        "            trunk_step = step\n"
        "            stop_after_step = step\n"
        "            log0(\n"
        "                f\"branching:event_trigger min_frac:{trigger_min_frac:.2f} max_frac:{trigger_max_frac:.2f} scale:{scale:.4f} plateau:{int(plateau_ready)} trunk_step:{trunk_step} trunk_ms:{trunk_training_time_ms:.0f} finishers:{len(branch_finishers)}\"\n"
        "            )\n"
        "        elif stop_after_step is None and reached_cap:\n"
        "            stop_after_step = step\n",
        name,
    )
    source = _replace_unique(
        source,
        "    log0(\n"
        "        f\"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB \"\n"
        "        f\"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB\"\n"
        "    )\n",
        "    if branch_triggered and trunk_model_state is not None and trunk_ema_state is not None and trunk_optimizer_states is not None and trunk_loader_state is not None and trunk_training_time_ms is not None and trunk_step is not None:\n"
        "        remaining_ms = max((max_wallclock_ms or trunk_training_time_ms) - trunk_training_time_ms, 0.0)\n"
        "        branch_budget_ms = remaining_ms / max(len(branch_finishers), 1)\n"
        "        best_branch_bpb = None\n"
        "        best_branch_loss = None\n"
        "        for branch_idx, finisher in enumerate(branch_finishers):\n"
        "            branch_name = str(finisher.get(\"name\", f\"branch_{branch_idx}\"))\n"
        "            base_model.load_state_dict(trunk_model_state, strict=True)\n"
        "            for opt, state in zip(optimizers, trunk_optimizer_states, strict=True):\n"
        "                opt.load_state_dict(copy.deepcopy(state))\n"
        "            ema_state = {state_name: tensor.detach().cpu().clone() for state_name, tensor in trunk_ema_state.items()}\n"
        "            restore_loader_state(train_loader, args.train_files, trunk_loader_state)\n"
        "            step = trunk_step\n"
        "            torch.cuda.synchronize()\n"
        "            t_branch = time.perf_counter()\n"
        "            branch_end_ms = trunk_training_time_ms + branch_budget_ms\n"
        "            while True:\n"
        "                if step >= args.iterations:\n"
        "                    break\n"
        "                elapsed_ms = trunk_training_time_ms + 1000.0 * (time.perf_counter() - t_branch)\n"
        "                if elapsed_ms >= branch_end_ms:\n"
        "                    break\n"
        "                scale = lr_mul(step, elapsed_ms)\n"
        "                branch_ema_decay = float(finisher.get(\"ema_decay\", 0.997))\n"
        "                branch_qat_alpha = float(finisher.get(\"qat_alpha\", 0.0))\n"
        "                branch_export_weight = float(finisher.get(\"export_surrogate_weight\", 0.0))\n"
        "                token_lr_mult = float(finisher.get(\"token_lr_mult\", 1.0))\n"
        "                matrix_lr_mult = float(finisher.get(\"matrix_lr_mult\", 1.0))\n"
        "                scalar_lr_mult = float(finisher.get(\"scalar_lr_mult\", 1.0))\n"
        "                head_lr_mult = float(finisher.get(\"head_lr_mult\", 1.0))\n"
        "                export_modes = [str(mode) for mode in finisher.get(\"export_modes\", [\"ema\", \"raw\"])]\n"
        "                CastedLinear._qat_enabled = branch_qat_alpha > 0.0\n"
        "                CastedLinear._qat_alpha = branch_qat_alpha\n"
        "                zero_grad_all()\n"
        "                train_loss = torch.zeros((), device=device)\n"
        "                for micro_step in range(grad_accum_steps):\n"
        "                    if distributed:\n"
        "                        model.require_backward_grad_sync = micro_step == grad_accum_steps - 1\n"
        "                    x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)\n"
        "                    with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "                        loss = model(x, y)\n"
        "                        if branch_export_weight > 0.0:\n"
        "                            tok_err = base_model.tok_emb.weight.float() - base_model.tok_emb.weight.float().round()\n"
        "                            loss = loss + branch_export_weight * tok_err.square().mean().to(loss.dtype)\n"
        "                    train_loss += loss.detach()\n"
        "                    (loss * grad_scale).backward()\n"
        "                train_loss /= grad_accum_steps\n"
        "                frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0\n"
        "                muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum\n"
        "                for group in optimizer_muon.param_groups:\n"
        "                    group[\"momentum\"] = muon_momentum\n"
        "                for opt in optimizers:\n"
        "                    for group in opt.param_groups:\n"
        "                        base_lr = group[\"base_lr\"]\n"
        "                        params = group[\"params\"]\n"
        "                        mult = matrix_lr_mult\n"
        "                        if params is tok_params:\n"
        "                            mult = token_lr_mult\n"
        "                        elif params is scalar_params:\n"
        "                            mult = scalar_lr_mult\n"
        "                        elif base_model.lm_head is not None and params == [base_model.lm_head.weight]:\n"
        "                            mult = head_lr_mult\n"
        "                        group[\"lr\"] = base_lr * scale * mult\n"
        "                if args.grad_clip_norm > 0:\n"
        "                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)\n"
        "                for opt in optimizers:\n"
        "                    opt.step()\n"
        "                zero_grad_all()\n"
        "                with torch.no_grad():\n"
        "                    for state_name, tensor in base_model.state_dict().items():\n"
        "                        ema_state[state_name].mul_(branch_ema_decay).add_(tensor.detach().float(), alpha=1.0 - branch_ema_decay)\n"
        "                step += 1\n"
        "            branch_mode, branch_val_loss, branch_val_bpb, branch_state = branch_export_portfolio_eval(\n"
        "                args, base_model, ema_state, export_modes,\n"
        "                rank, world_size, device,\n"
        "                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n"
        "                effective_eval_seq_len,\n"
        "            )\n"
        "            log0(\n"
        "                f\"branching:branch name:{branch_name} export_mode:{branch_mode} deployed_val_loss:{branch_val_loss:.4f} deployed_val_bpb:{branch_val_bpb:.8f} extra_steps:{step - trunk_step} branch_budget_ms:{branch_budget_ms:.0f}\"\n"
        "            )\n"
        "            if best_branch_bpb is None or branch_val_bpb < best_branch_bpb:\n"
        "                best_branch_bpb = branch_val_bpb\n"
        "                best_branch_loss = branch_val_loss\n"
        "                branch_best_name = branch_name\n"
        "                branch_best_mode = branch_mode\n"
        "                branch_best_state = branch_state\n"
        "        if branch_best_state is not None:\n"
        "            current_state = base_model.state_dict()\n"
        "            best_state_typed = {state_name: tensor.to(dtype=current_state[state_name].dtype) for state_name, tensor in branch_best_state.items()}\n"
        "            base_model.load_state_dict(best_state_typed, strict=True)\n"
        "            log0(f\"branching:choose_best name:{branch_best_name} mode:{branch_best_mode} deployed_val_loss:{best_branch_loss:.4f} deployed_val_bpb:{best_branch_bpb:.8f}\")\n"
        "    log0(\n"
        "        f\"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB \"\n"
        "        f\"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB\"\n"
        "    )\n",
        name,
    )
    source = _replace_unique(
        source,
        "    # Apply EMA weights (better than SWA alone per PR#401)\n"
        "    log0(\"ema:applying EMA weights\")\n"
        "    current_state = base_model.state_dict()\n"
        "    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}\n"
        "    base_model.load_state_dict(avg_state, strict=True)\n",
        "    if not branch_triggered:\n"
        "        # Apply EMA weights (better than SWA alone per PR#401)\n"
        "        log0(\"ema:applying EMA weights\")\n"
        "        current_state = base_model.state_dict()\n"
        "        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}\n"
        "        base_model.load_state_dict(avg_state, strict=True)\n",
        name,
    )
    return source


PATCH_ORDER = {
    "event_branch_tournament": 10,
}


PATCHES = {
    "event_branch_tournament": patch_event_branch_tournament,
}


def apply_patches(source: str, patch_names: list[str]) -> str:
    ordered = sorted(patch_names, key=lambda n: PATCH_ORDER.get(n, 100))
    for patch_name in ordered:
        source = PATCHES[patch_name](source)
    return source
