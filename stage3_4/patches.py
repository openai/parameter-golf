"""
stage3_4 patches — shared-trunk late branching.

This stage attacks a different false invariant from stage3_2:
the run should not commit to one late trajectory.
Instead, it trains a shared trunk, then spends the remaining budget on
multiple short late finishers and exports the best deployed branch.
"""

from __future__ import annotations


def _replace_unique(source: str, old: str, new: str, patch_name: str) -> str:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"[{patch_name}] target string not found:\n{old[:200]!r}")
    if count > 1:
        raise ValueError(f"[{patch_name}] ambiguous: found {count} occurrences:\n{old[:200]!r}")
    return source.replace(old, new)


def patch_late_branch_finishers(source: str) -> str:
    name = "late_branch_finishers"
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
        "    base_model: nn.Module,\n"
        "    rank: int,\n"
        "    world_size: int,\n"
        "    device: torch.device,\n"
        "    val_tokens: Tensor,\n"
        "    base_bytes_lut: Tensor,\n"
        "    has_leading_space_lut: Tensor,\n"
        "    is_boundary_token_lut: Tensor,\n"
        "    effective_eval_seq_len: int,\n"
        ") -> tuple[float, float]:\n"
        "    full_state_dict = base_model.state_dict()\n"
        "    export_sd = {k: v for k, v in full_state_dict.items() if \"mtp_heads\" not in k}\n"
        "    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n"
        "    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {\"mlp\", \"attn\"})\n"
        "    deq_state = dequantize_mixed_int6(quant_result, quant_meta, sd_cpu)\n"
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
        "    eval_model.load_state_dict(deq_state, strict=True)\n"
        "    val_loss, val_bpb = eval_val_sliding(\n"
        "        args, eval_model, rank, world_size, device,\n"
        "        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n"
        "        stride=args.eval_stride, batch_seqs=32, eval_seq_len=effective_eval_seq_len,\n"
        "    )\n"
        "    del eval_model\n"
        "    return val_loss, val_bpb\n"
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
        "    branch_start_frac = float(branch_spec.get(\"branch_start_frac\", 0.74))\n"
        "    branch_finishers = branch_spec.get(\"finishers\", [])\n"
        "    branch_triggered = False\n"
        "    branch_best_state: dict[str, Tensor] | None = None\n"
        "    branch_best_name: str | None = None\n"
        "    trunk_model_state: dict[str, Tensor] | None = None\n"
        "    trunk_ema_state: dict[str, Tensor] | None = None\n"
        "    trunk_optimizer_states: list[dict[str, object]] | None = None\n"
        "    trunk_loader_state: dict[str, int] | None = None\n"
        "    trunk_training_time_ms: float | None = None\n"
        "    trunk_step: int | None = None\n"
        "    def zero_grad_all() -> None:\n",
        name,
    )
    source = _replace_unique(
        source,
        "    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}\n"
        "    ema_decay = 0.997\n",
        "    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}\n"
        "    ema_decay = 0.997\n",
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
        "        for micro_step in range(grad_accum_steps):\n"
        "            if distributed:\n"
        "                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1\n"
        "            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)\n"
        "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "                loss = model(x, y)\n"
        "            train_loss += loss.detach()\n"
        "            (loss * grad_scale).backward()\n",
        "        for micro_step in range(grad_accum_steps):\n"
        "            if distributed:\n"
        "                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1\n"
        "            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)\n"
        "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "                loss = model(x, y)\n"
        "            train_loss += loss.detach()\n"
        "            (loss * grad_scale).backward()\n",
        name,
    )
    source = _replace_unique(
        source,
        "        if stop_after_step is None and reached_cap:\n"
        "            stop_after_step = step\n",
        "        branch_ready = (\n"
        "            branch_enabled and not branch_triggered and max_wallclock_ms is not None and branch_finishers\n"
        "            and approx_training_time_ms >= max_wallclock_ms * branch_start_frac\n"
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
        "                f\"branching:trigger start_frac:{branch_start_frac:.2f} trunk_step:{trunk_step} trunk_ms:{trunk_training_time_ms:.0f} finishers:{len(branch_finishers)}\"\n"
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
        "        for branch_idx, branch_policy in enumerate(branch_finishers):\n"
        "            branch_name = str(branch_policy.get(\"name\", f\"branch_{branch_idx}\"))\n"
        "            base_model.load_state_dict(trunk_model_state, strict=True)\n"
        "            for opt, state in zip(optimizers, trunk_optimizer_states, strict=True):\n"
        "                opt.load_state_dict(copy.deepcopy(state))\n"
        "            ema_state = {state_name: tensor.detach().cpu().clone() for state_name, tensor in trunk_ema_state.items()}\n"
        "            restore_loader_state(train_loader, args.train_files, trunk_loader_state)\n"
        "            step = trunk_step\n"
        "            branch_training_time_ms = trunk_training_time_ms\n"
        "            torch.cuda.synchronize()\n"
        "            t_branch = time.perf_counter()\n"
        "            branch_end_ms = trunk_training_time_ms + branch_budget_ms\n"
        "            while True:\n"
        "                if step >= args.iterations:\n"
        "                    break\n"
        "                elapsed_ms = branch_training_time_ms + 1000.0 * (time.perf_counter() - t_branch)\n"
        "                if elapsed_ms >= branch_end_ms:\n"
        "                    break\n"
        "                scale = lr_mul(step, elapsed_ms)\n"
        "                branch_ema_decay = float(branch_policy.get(\"ema_decay\", 0.997))\n"
        "                branch_qat_alpha = float(branch_policy.get(\"qat_alpha\", 0.0))\n"
        "                branch_export_weight = float(branch_policy.get(\"export_surrogate_weight\", 0.0))\n"
        "                token_lr_mult = float(branch_policy.get(\"token_lr_mult\", 1.0))\n"
        "                matrix_lr_mult = float(branch_policy.get(\"matrix_lr_mult\", 1.0))\n"
        "                scalar_lr_mult = float(branch_policy.get(\"scalar_lr_mult\", 1.0))\n"
        "                head_lr_mult = float(branch_policy.get(\"head_lr_mult\", 1.0))\n"
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
        "                            loss = loss + branch_export_weight * ((base_model.tok_emb.weight.float() - base_model.tok_emb.weight.float().round()).square().mean()).to(loss.dtype)\n"
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
        "            branch_training_time_ms = branch_end_ms\n"
        "            branch_use_ema = bool(int(branch_policy.get(\"use_ema\", 1)))\n"
        "            if branch_use_ema:\n"
        "                current_state = base_model.state_dict()\n"
        "                export_state = {state_name: tensor.to(dtype=current_state[state_name].dtype) for state_name, tensor in ema_state.items()}\n"
        "                base_model.load_state_dict(export_state, strict=True)\n"
        "            branch_val_loss, branch_val_bpb = branch_export_eval(\n"
        "                args, base_model, rank, world_size, device,\n"
        "                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n"
        "                effective_eval_seq_len,\n"
        "            )\n"
        "            log0(\n"
        "                f\"branching:branch name:{branch_name} deployed_val_loss:{branch_val_loss:.4f} deployed_val_bpb:{branch_val_bpb:.8f} extra_steps:{step - trunk_step} branch_budget_ms:{branch_budget_ms:.0f}\"\n"
        "            )\n"
        "            if best_branch_bpb is None or branch_val_bpb < best_branch_bpb:\n"
        "                best_branch_bpb = branch_val_bpb\n"
        "                branch_best_name = branch_name\n"
        "                branch_best_state = clone_state_dict_cpu(base_model.state_dict())\n"
        "        if branch_best_state is not None:\n"
        "            current_state = base_model.state_dict()\n"
        "            best_state_typed = {state_name: tensor.to(dtype=current_state[state_name].dtype) for state_name, tensor in branch_best_state.items()}\n"
        "            base_model.load_state_dict(best_state_typed, strict=True)\n"
        "            log0(f\"branching:choose_best name:{branch_best_name} deployed_val_bpb:{best_branch_bpb:.8f}\")\n"
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


# ===========================================================================
# N-GRAM LOGIT BIAS (EVAL-TIME)
#
# At eval time, compute bigram/trigram frequency tables from prior context.
# Add a weighted logit bias to model output. Zero training cost.
# ===========================================================================

def patch_ngram_bias(source: str) -> str:
    name = "ngram_bias"
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    ngram_bias = bool(int(os.environ.get("NGRAM_BIAS", "0")))\n'
        '    ngram_alpha = float(os.environ.get("NGRAM_ALPHA", "0.15"))  # bigram weight\n'
        '    ngram_beta = float(os.environ.get("NGRAM_BETA", "0.10"))   # trigram weight',
        name,
    )

    # Add ngram bias function before eval_val_sliding
    source = _replace_unique(
        source,
        'def eval_val_sliding(\n'
        '    args: Hyperparameters,\n'
        '    base_model: nn.Module,\n'
        '    rank: int,\n'
        '    world_size: int,\n'
        '    device: torch.device,\n'
        '    val_tokens: Tensor,\n'
        '    base_bytes_lut: Tensor,\n'
        '    has_leading_space_lut: Tensor,\n'
        '    is_boundary_token_lut: Tensor,\n'
        '    stride: int,\n'
        '    batch_seqs: int = 32,\n'
        '    eval_seq_len: int | None = None,\n'
        ') -> tuple[float, float]:',
        'def compute_ngram_bias(\n'
        '    tokens: Tensor, vocab_size: int, alpha: float, beta: float,\n'
        ') -> Tensor:\n'
        '    """Compute n-gram frequency-based logit bias from context tokens.\n'
        '    \n'
        '    For each position t, builds:\n'
        '      - bigram_counts[prev, v]: how many times token v followed prev in context[:t]\n'
        '      - trigram_counts[(pp, prev), v]: how many times v followed (pp, prev)\n'
        '    Then: bias[t, v] = alpha * log(1 + bigram_counts[tokens[t-1], v])\n'
        '                     + beta  * log(1 + trigram_counts[(tokens[t-2], tokens[t-1]), v])\n'
        '    \n'
        '    Returns a bias tensor of shape [len(tokens), vocab_size].\n'
        '    """\n'
        '    T = tokens.shape[0]\n'
        '    bias = torch.zeros(T, vocab_size, dtype=torch.float32, device=tokens.device)\n'
        '    if T < 2:\n'
        '        return bias\n'
        '    # Bigram table: for each context token c, count how often each v follows c\n'
        '    # Use dict of tensors to avoid V*V memory\n'
        '    bi_table: dict[int, Tensor] = {}  # bi_table[prev] = counts[V]\n'
        '    tri_table: dict[tuple[int, int], Tensor] = {}  # tri_table[(pp, prev)] = counts[V]\n'
        '    dev = tokens.device\n'
        '    for t in range(1, T):\n'
        '        prev = tokens[t - 1].item()\n'
        '        # Apply current bigram bias at position t\n'
        '        if alpha > 0 and prev in bi_table:\n'
        '            bias[t] += alpha * torch.log1p(bi_table[prev])\n'
        '        # Apply current trigram bias at position t\n'
        '        if beta > 0 and t >= 2:\n'
        '            pp = tokens[t - 2].item()\n'
        '            key = (pp, prev)\n'
        '            if key in tri_table:\n'
        '                bias[t] += beta * torch.log1p(tri_table[key])\n'
        '        # Update tables with the actual token at position t\n'
        '        # (this becomes available for future positions)\n'
        '        actual = tokens[t].item() if t < T else -1\n'
        '        if actual >= 0:\n'
        '            if prev not in bi_table:\n'
        '                bi_table[prev] = torch.zeros(vocab_size, dtype=torch.float32, device=dev)\n'
        '            bi_table[prev][actual] += 1.0\n'
        '            if t >= 2:\n'
        '                pp = tokens[t - 2].item()\n'
        '                key = (pp, prev)\n'
        '                if key not in tri_table:\n'
        '                    tri_table[key] = torch.zeros(vocab_size, dtype=torch.float32, device=dev)\n'
        '                tri_table[key][actual] += 1.0\n'
        '    return bias\n'
        '\n'
        '\n'
        'def eval_val_sliding(\n'
        '    args: Hyperparameters,\n'
        '    base_model: nn.Module,\n'
        '    rank: int,\n'
        '    world_size: int,\n'
        '    device: torch.device,\n'
        '    val_tokens: Tensor,\n'
        '    base_bytes_lut: Tensor,\n'
        '    has_leading_space_lut: Tensor,\n'
        '    is_boundary_token_lut: Tensor,\n'
        '    stride: int,\n'
        '    batch_seqs: int = 32,\n'
        '    eval_seq_len: int | None = None,\n'
        ') -> tuple[float, float]:',
        name,
    )

    # In eval_val_sliding, after computing logits, apply ngram bias before loss
    source = _replace_unique(
        source,
        '            nll = F.cross_entropy(\n'
        '                logits.reshape(-1, logits.size(-1)).float(),\n'
        '                y_batch.reshape(-1),\n'
        '                reduction="none",\n'
        '            ).reshape(bsz, seq_len)',
        '            if args.ngram_bias:\n'
        '                # Apply n-gram logit bias per sequence in batch\n'
        '                for _bi in range(bsz):\n'
        '                    wlen = wlens[_bi]\n'
        '                    if wlen > 1:\n'
        '                        _ctx = x_batch[_bi, :wlen]\n'
        '                        _bias = compute_ngram_bias(_ctx, logits.size(-1), args.ngram_alpha, args.ngram_beta)\n'
        '                        logits[_bi, :wlen] = logits[_bi, :wlen] + _bias.to(logits.dtype)\n'
        '            nll = F.cross_entropy(\n'
        '                logits.reshape(-1, logits.size(-1)).float(),\n'
        '                y_batch.reshape(-1),\n'
        '                reduction="none",\n'
        '            ).reshape(bsz, seq_len)',
        name,
    )
    return source


# ===========================================================================
# SLOT: PER-SAMPLE L-BFGS OPTIMIZATION (EVAL-TIME)
#
# Per-sample optimization of a logit bias vector at eval time. Each sample
# gets N L-BFGS steps to find a bias that minimizes its own loss.
# This is the paradigm from PR #1507 that reaches 0.2282 BPB.
# ===========================================================================

def patch_slot_basic(source: str) -> str:
    name = "slot_basic"
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    slot_enabled = bool(int(os.environ.get("SLOT_ENABLED", "0")))\n'
        '    slot_lbfgs_iters = int(os.environ.get("SLOT_LBFGS_ITERS", "20"))\n'
        '    slot_lr = float(os.environ.get("SLOT_LR", "0.1"))',
        name,
    )

    # Add SLOT optimization function before eval_val_sliding
    source = _replace_unique(
        source,
        'def eval_val_sliding(\n'
        '    args: Hyperparameters,\n'
        '    base_model: nn.Module,\n'
        '    rank: int,\n'
        '    world_size: int,\n'
        '    device: torch.device,\n'
        '    val_tokens: Tensor,\n'
        '    base_bytes_lut: Tensor,\n'
        '    has_leading_space_lut: Tensor,\n'
        '    is_boundary_token_lut: Tensor,\n'
        '    stride: int,\n'
        '    batch_seqs: int = 32,\n'
        '    eval_seq_len: int | None = None,\n'
        ') -> tuple[float, float]:',
        'def slot_optimize_bias(\n'
        '    model_fn, x: Tensor, y: Tensor, vocab_size: int,\n'
        '    max_iter: int = 20, lr: float = 0.1,\n'
        ') -> Tensor:\n'
        '    """Per-sample L-BFGS optimization of a logit bias vector.\n'
        '    \n'
        '    Optimizes delta in R^V to minimize CE(logits + delta, targets).\n'
        '    The model forward is frozen; only delta is optimized.\n'
        '    Returns final biased logits.\n'
        '    """\n'
        '    bsz, seq_len = x.shape\n'
        '    # Get base logits (frozen, no grad needed for model params)\n'
        '    with torch.no_grad():\n'
        '        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\n'
        '            base_logits = model_fn(x).float()\n'
        '    # Detach to ensure no graph connection to model\n'
        '    base_logits = base_logits.detach()\n'
        '    # Initialize bias to zero — this is the only optimized variable\n'
        '    delta = torch.zeros(vocab_size, dtype=torch.float32, device=x.device, requires_grad=True)\n'
        '    y_flat = y.reshape(-1)\n'
        '    def closure():\n'
        '        # L-BFGS calls closure multiple times per step; zero grad each time\n'
        '        if delta.grad is not None:\n'
        '            delta.grad.zero_()\n'
        '        biased = base_logits + delta[None, None, :]  # [B, T, V]\n'
        '        loss = F.cross_entropy(biased.reshape(-1, vocab_size), y_flat, reduction="mean")\n'
        '        loss.backward()\n'
        '        return loss\n'
        '    opt = torch.optim.LBFGS(\n'
        '        [delta], lr=lr, max_iter=max_iter,\n'
        '        line_search_fn="strong_wolfe",\n'
        '    )\n'
        '    opt.step(closure)\n'
        '    with torch.no_grad():\n'
        '        final_logits = base_logits + delta[None, None, :]\n'
        '    return final_logits\n'
        '\n'
        '\n'
        'def eval_val_sliding(\n'
        '    args: Hyperparameters,\n'
        '    base_model: nn.Module,\n'
        '    rank: int,\n'
        '    world_size: int,\n'
        '    device: torch.device,\n'
        '    val_tokens: Tensor,\n'
        '    base_bytes_lut: Tensor,\n'
        '    has_leading_space_lut: Tensor,\n'
        '    is_boundary_token_lut: Tensor,\n'
        '    stride: int,\n'
        '    batch_seqs: int = 32,\n'
        '    eval_seq_len: int | None = None,\n'
        ') -> tuple[float, float]:',
        name,
    )

    # In eval_val_sliding, apply SLOT optimization before computing NLL
    source = _replace_unique(
        source,
        '            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\n'
        '                logits = compiled_logits(x_batch)\n'
        '            nll = F.cross_entropy(',
        '            if args.slot_enabled:\n'
        '                # SLOT: per-batch L-BFGS optimization of logit bias\n'
        '                # enable_grad() needed because eval loop is inside inference_mode()\n'
        '                with torch.enable_grad():\n'
        '                    logits = slot_optimize_bias(\n'
        '                        base_model.forward_logits, x_batch, y_batch,\n'
        '                        vocab_size=args.vocab_size,\n'
        '                        max_iter=args.slot_lbfgs_iters, lr=args.slot_lr,\n'
        '                    )\n'
        '            else:\n'
        '                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\n'
        '                    logits = compiled_logits(x_batch)\n'
        '            nll = F.cross_entropy(',
        name,
    )
    return source


# ===========================================================================
# TTT AGGRESSIVE (POST-TRAINING, 10 EPOCHS)
#
# Same as stage3_1 ttt_aggressive but for the stage3_4 pipeline.
# Push TTT to 10 epochs with cosine LR, unfreeze block 1.
# ===========================================================================

def patch_ttt_aggressive(source: str) -> str:
    name = "ttt_aggressive"
    import re as _re
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))\n'
        '    ttt_epochs = int(os.environ.get("TTT_EPOCHS", "5"))\n'
        '    ttt_lr = float(os.environ.get("TTT_LR", "0.00045"))\n'
        '    ttt_freeze_blocks = os.environ.get("TTT_FREEZE_BLOCKS", "0,1")\n'
        '    ttt_cosine = bool(int(os.environ.get("TTT_COSINE", "0")))',
        name,
    )

    # Insert TTT loop after EMA weights are applied, before export quantization.
    # Use a longer anchor to avoid ambiguity with branch_export_eval's copy.
    source = _replace_unique(
        source,
        '    full_state_dict = base_model.state_dict()\n'
        '    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}\n'
        '    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)',
        '    # ---- TEST-TIME TRAINING (TTT) ----\n'
        '    if args.ttt_enabled:\n'
        '        import re as _re\n'
        '        freeze_blocks = set(int(b) for b in args.ttt_freeze_blocks.split(",") if b.strip())\n'
        '        ttt_params = []\n'
        '        frozen_count = 0\n'
        '        for pname, p in base_model.named_parameters():\n'
        '            m = _re.match(r"blocks\\.(\\d+)\\.", pname)\n'
        '            if m and int(m.group(1)) in freeze_blocks:\n'
        '                p.requires_grad_(False)\n'
        '                frozen_count += p.numel()\n'
        '            else:\n'
        '                p.requires_grad_(True)\n'
        '                ttt_params.append(p)\n'
        '        log0(f"ttt:start epochs:{args.ttt_epochs} lr:{args.ttt_lr} "\n'
        '             f"freeze:{freeze_blocks} cosine:{args.ttt_cosine} "\n'
        '             f"unfrozen:{sum(p.numel() for p in ttt_params)} frozen:{frozen_count}")\n'
        '        ttt_opt = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, weight_decay=0.01, fused=True)\n'
        '        base_model.train()\n'
        '        for ttt_epoch in range(args.ttt_epochs):\n'
        '            if args.ttt_cosine:\n'
        '                ttt_lr_now = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ttt_epoch / args.ttt_epochs))\n'
        '                for g in ttt_opt.param_groups:\n'
        '                    g["lr"] = ttt_lr_now\n'
        '            total_seqs = (val_tokens.numel() - 1) // args.train_seq_len\n'
        '            ttt_loss_sum = 0.0\n'
        '            ttt_steps = 0\n'
        '            for seq_start in range(0, total_seqs, 8):\n'
        '                seq_end = min(seq_start + 8, total_seqs)\n'
        '                raw_s = seq_start * args.train_seq_len\n'
        '                raw_e = seq_end * args.train_seq_len + 1\n'
        '                local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64)\n'
        '                x = local[:-1].reshape(-1, args.train_seq_len)\n'
        '                y = local[1:].reshape(-1, args.train_seq_len)\n'
        '                ttt_opt.zero_grad(set_to_none=True)\n'
        '                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):\n'
        '                    loss = base_model(x, y)\n'
        '                loss.backward()\n'
        '                torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)\n'
        '                ttt_opt.step()\n'
        '                ttt_loss_sum += loss.item()\n'
        '                ttt_steps += 1\n'
        '            log0(f"ttt:epoch:{ttt_epoch + 1}/{args.ttt_epochs} "\n'
        '                 f"avg_loss:{ttt_loss_sum / max(ttt_steps, 1):.4f}")\n'
        '        for p in base_model.parameters():\n'
        '            p.requires_grad_(True)\n'
        '        base_model.eval()\n'
        '        log0("ttt:done")\n'
        '    full_state_dict = base_model.state_dict()\n'
        '    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}\n'
        '    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)',
        name,
    )
    return source


PATCH_ORDER = {
    "late_branch_finishers": 10,
    "ngram_bias": 20,
    "slot_basic": 30,
    "ttt_aggressive": 40,
}


PATCHES = {
    "late_branch_finishers": patch_late_branch_finishers,
    "ngram_bias": patch_ngram_bias,
    "slot_basic": patch_slot_basic,
    "ttt_aggressive": patch_ttt_aggressive,
}


def apply_patches(source: str, patch_names: list[str]) -> str:
    ordered = sorted(patch_names, key=lambda n: PATCH_ORDER.get(n, 100))
    for patch_name in ordered:
        source = PATCHES[patch_name](source)
    return source
