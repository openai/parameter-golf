"""
Moonshot runtime patches for hailmary experiments.

This file keeps the base train_gpt.py untouched. Each patch rewrites the source
for one run directory, like the stage2_1 and stage3 flows.

The first wave here focuses on distinct, tractable moonshot patches:

- exact_overlap_eval
- ema_export
- late_qat_active
- countinit_bigram

For reuse, some patches delegate to the already-validated stage2_1 patch set.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_stage2_1_patches():
    path = Path(__file__).resolve().parents[1] / "stage2_1" / "patches.py"
    spec = importlib.util.spec_from_file_location("stage2_1_patches_for_hailmary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load stage2_1 patches from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_S21 = _load_stage2_1_patches()
_replace_any = _S21._replace_any
_replace_unique = _S21._replace_unique


def patch_exact_overlap_eval(source: str) -> str:
    # Reuse the validated sliding-window exact-tail implementation from stage2_1.
    return _S21.patch_sliding_eval(source)


def patch_ema_export(source: str) -> str:
    # Reuse the validated EMA export patch from stage2_1.
    return _S21.patch_ema(source)


def patch_xsa_all(source: str) -> str:
    # Reuse the generalized XSA patch and drive it with XSA_LAYERS=all.
    return _S21.patch_xsa_all(source)


def patch_xsa4(source: str) -> str:
    return _S21.patch_xsa4(source)


def patch_curriculum_shard_order(source: str) -> str:
    # Reuse the shard-order patch from stage2_1 for moonshot curriculum screens.
    return _S21.patch_curriculum_shard_order(source)


def patch_staged_curriculum(source: str) -> str:
    name = "staged_curriculum"
    source = _replace_unique(
        source,
        "def load_data_shard(file: Path) -> Tensor:\n"
        "    header_bytes = 256 * np.dtype(\"<i4\").itemsize\n"
        "    token_bytes = np.dtype(\"<u2\").itemsize\n"
        "    header = np.fromfile(file, dtype=\"<i4\", count=256)\n"
        "    # SHARD HEADER INTS & SHARD_MAGIC\n"
        "    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:\n"
        "        raise ValueError(f\"Unexpected shard header for {file}\")\n"
        "    num_tokens = int(header[2])\n"
        "    expected_size = header_bytes + num_tokens * token_bytes\n"
        "    if file.stat().st_size != expected_size:\n"
        "        raise ValueError(f\"Shard size mismatch for {file}: expected {expected_size} bytes\")\n"
        "    tokens_np = np.fromfile(file, dtype=\"<u2\", count=num_tokens, offset=header_bytes)\n"
        "    if tokens_np.size != num_tokens:\n"
        "        raise ValueError(f\"Short read for {file}\")\n"
        "    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))\n"
        "\n"
        "\n"
        "class TokenStream:\n",
        "def load_data_shard(file: Path) -> Tensor:\n"
        "    header_bytes = 256 * np.dtype(\"<i4\").itemsize\n"
        "    token_bytes = np.dtype(\"<u2\").itemsize\n"
        "    header = np.fromfile(file, dtype=\"<i4\", count=256)\n"
        "    # SHARD HEADER INTS & SHARD_MAGIC\n"
        "    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:\n"
        "        raise ValueError(f\"Unexpected shard header for {file}\")\n"
        "    num_tokens = int(header[2])\n"
        "    expected_size = header_bytes + num_tokens * token_bytes\n"
        "    if file.stat().st_size != expected_size:\n"
        "        raise ValueError(f\"Shard size mismatch for {file}: expected {expected_size} bytes\")\n"
        "    tokens_np = np.fromfile(file, dtype=\"<u2\", count=num_tokens, offset=header_bytes)\n"
        "    if tokens_np.size != num_tokens:\n"
        "        raise ValueError(f\"Short read for {file}\")\n"
        "    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))\n"
        "\n"
        "\n"
        "def order_shard_files_for_mode(files: list[Path], mode: str, source_path: str = \"\") -> list[Path]:\n"
        "    mode = mode.strip()\n"
        "    if not files or mode in {\"\", \"lexicographic\", \"sorted\"}:\n"
        "        return list(files)\n"
        "    if source_path:\n"
        "        order_path = Path(source_path)\n"
        "        if not order_path.exists():\n"
        "            raise FileNotFoundError(f\"STAGED_CURRICULUM_SOURCE not found: {order_path}\")\n"
        "        ordered_names = [line.strip() for line in order_path.read_text(encoding=\"utf-8\").splitlines() if line.strip()]\n"
        "        rank_map = {name: idx for idx, name in enumerate(ordered_names)}\n"
        "        default_rank = len(rank_map)\n"
        "        return sorted(files, key=lambda path: (rank_map.get(path.name, default_rank), path.name))\n"
        "    if mode == \"reverse\":\n"
        "        return list(reversed(files))\n"
        "    if mode in {\"shuffle\", \"random\"}:\n"
        "        seed = int(os.environ.get(\"SEED\", \"1337\"))\n"
        "        shuffled = list(files)\n"
        "        random.Random(seed).shuffle(shuffled)\n"
        "        return shuffled\n"
        "    if mode in {\"size_desc\", \"hard\"}:\n"
        "        return sorted(files, key=lambda path: (-path.stat().st_size, path.name))\n"
        "    if mode in {\"size_asc\", \"easy\"}:\n"
        "        return sorted(files, key=lambda path: (path.stat().st_size, path.name))\n"
        "    if mode == \"ranked\":\n"
        "        ranked_source = os.environ.get(\"STAGED_CURRICULUM_SOURCE\", \"\").strip()\n"
        "        if not ranked_source:\n"
        "            raise ValueError(\"STAGED_CURRICULUM_SOURCE is required when STAGED_CURRICULUM_PHASE*=ranked\")\n"
        "        return order_shard_files_for_mode(files, \"sorted\", ranked_source)\n"
        "    raise ValueError(f\"Unsupported staged curriculum mode: {mode!r}\")\n"
        "\n"
        "\n"
        "class TokenStream:\n",
        name,
    )
    source = _replace_unique(
        source,
        "    def __init__(self, pattern: str):\n"
        "        self.files = [Path(p) for p in sorted(glob.glob(pattern))]\n"
        "        if not self.files:\n"
        "            raise FileNotFoundError(f\"No files found for pattern: {pattern}\")\n"
        "        self.file_idx = 0\n"
        "        self.tokens = load_data_shard(self.files[0])\n"
        "        self.pos = 0\n",
        "    def __init__(self, pattern: str):\n"
        "        base_files = [Path(p) for p in sorted(glob.glob(pattern))]\n"
        "        if not base_files:\n"
        "            raise FileNotFoundError(f\"No files found for pattern: {pattern}\")\n"
        "        enabled = os.environ.get(\"STAGED_CURRICULUM_ENABLE\", \"0\") == \"1\"\n"
        "        source_path = os.environ.get(\"STAGED_CURRICULUM_SOURCE\", \"\").strip()\n"
        "        phase1_mode = os.environ.get(\"STAGED_CURRICULUM_PHASE1\", \"sorted\")\n"
        "        phase2_mode = os.environ.get(\"STAGED_CURRICULUM_PHASE2\", phase1_mode)\n"
        "        self.switch_frac = float(os.environ.get(\"STAGED_CURRICULUM_SWITCH_FRAC\", \"0.6\"))\n"
        "        self.phase1_files = order_shard_files_for_mode(base_files, phase1_mode, source_path if phase1_mode == \"ranked\" else \"\")\n"
        "        self.phase2_files = order_shard_files_for_mode(base_files, phase2_mode, source_path if phase2_mode == \"ranked\" else \"\")\n"
        "        self.active_files = self.phase1_files\n"
        "        self.files = self.active_files\n"
        "        self.file_idx = 0\n"
        "        self.tokens = load_data_shard(self.files[0])\n"
        "        self.pos = 0\n"
        "        self.staged_enabled = enabled\n",
        name,
    )
    source = _replace_unique(
        source,
        "    def _advance_file(self) -> None:\n"
        "        self.file_idx = (self.file_idx + 1) % len(self.files)\n"
        "        self.tokens = load_data_shard(self.files[self.file_idx])\n"
        "        self.pos = 0\n",
        "    def set_progress(self, frac: float) -> None:\n"
        "        if not self.staged_enabled:\n"
        "            return\n"
        "        target_files = self.phase2_files if frac >= self.switch_frac else self.phase1_files\n"
        "        if target_files is self.active_files:\n"
        "            return\n"
        "        self.active_files = target_files\n"
        "        self.files = self.active_files\n"
        "        self.file_idx = 0\n"
        "        self.tokens = load_data_shard(self.files[0])\n"
        "        self.pos = 0\n"
        "\n"
        "    def _advance_file(self) -> None:\n"
        "        self.file_idx = (self.file_idx + 1) % len(self.files)\n"
        "        self.tokens = load_data_shard(self.files[self.file_idx])\n"
        "        self.pos = 0\n",
        name,
    )
    source = _replace_unique(
        source,
        "class DistributedTokenLoader:\n"
        "    # Each call consumes a contiguous chunk from the shared token stream, then slices out\n"
        "    # one disjoint span per rank. The extra \"+1\" token lets us build (x, y) by shifting.\n"
        "    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):\n"
        "        self.rank = rank\n"
        "        self.world_size = world_size\n"
        "        self.device = device\n"
        "        self.stream = TokenStream(pattern)\n"
        "\n"
        "    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:\n",
        "class DistributedTokenLoader:\n"
        "    # Each call consumes a contiguous chunk from the shared token stream, then slices out\n"
        "    # one disjoint span per rank. The extra \"+1\" token lets us build (x, y) by shifting.\n"
        "    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):\n"
        "        self.rank = rank\n"
        "        self.world_size = world_size\n"
        "        self.device = device\n"
        "        self.stream = TokenStream(pattern)\n"
        "\n"
        "    def set_progress(self, frac: float) -> None:\n"
        "        self.stream.set_progress(frac)\n"
        "\n"
        "    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:\n",
        name,
    )
    source = _replace_unique(
        source,
        "        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n"
        "        scale = lr_mul(step, elapsed_ms)\n",
        "        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n"
        "        train_loader.set_progress(step / max(args.iterations, 1))\n"
        "        scale = lr_mul(step, elapsed_ms)\n",
        name,
    )
    return source


def patch_checkpoint_selection(source: str) -> str:
    name = "checkpoint_selection"
    source = _replace_unique(
        source,
        "import copy\n"
        "import glob\n",
        "import copy\n"
        "import glob\n",
        name,
    )
    source = _replace_unique(
        source,
        "    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None\n",
        "    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None\n"
        "    snapshot_select_enabled = os.environ.get(\"SNAPSHOT_SELECT_ENABLE\", \"0\") == \"1\"\n"
        "    snapshot_select_mode = os.environ.get(\"SNAPSHOT_SELECT_MODE\", \"deployed_score\")\n"
        "    snapshot_select_score = os.environ.get(\"SNAPSHOT_SELECT_SCORE\", \"deployed\")\n"
        "    snapshot_select_last_k = int(os.environ.get(\"SNAPSHOT_SELECT_LAST_K\", \"6\"))\n"
        "    snapshot_select_every = int(os.environ.get(\"SNAPSHOT_SELECT_EVERY\", \"250\"))\n"
        "    snapshot_select_start_frac = float(os.environ.get(\"SNAPSHOT_SELECT_START_FRAC\", \"0.75\"))\n"
        "    snapshot_candidates: list[tuple[str, dict[str, Tensor]]] = []\n"
        "\n"
        "    def clone_state_dict_cpu(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:\n"
        "        return {state_name: tensor.detach().cpu().clone() for state_name, tensor in state_dict.items()}\n"
        "\n"
        "    def maybe_capture_snapshot(current_step: int) -> None:\n"
        "        if not snapshot_select_enabled:\n"
        "            return\n"
        "        if current_step < int(snapshot_select_start_frac * args.iterations):\n"
        "            return\n"
        "        if snapshot_select_every > 1 and current_step % snapshot_select_every != 0 and current_step != args.iterations:\n"
        "            return\n"
        "        snapshot_candidates.append((f\"step_{current_step}\", clone_state_dict_cpu(base_model.state_dict())))\n"
        "        if len(snapshot_candidates) > snapshot_select_last_k:\n"
        "            del snapshot_candidates[0 : len(snapshot_candidates) - snapshot_select_last_k]\n"
        "\n"
        "    def score_state_candidate(tag: str, state_dict: dict[str, Tensor]) -> tuple[float, float]:\n"
        "        current_state = clone_state_dict_cpu(base_model.state_dict())\n"
        "        try:\n"
        "            base_model.load_state_dict(state_dict, strict=True)\n"
        "            if snapshot_select_score == \"raw\":\n"
        "                cand_loss, cand_bpb = eval_val(\n"
        "                    args,\n"
        "                    model,\n"
        "                    rank,\n"
        "                    world_size,\n"
        "                    device,\n"
        "                    grad_accum_steps,\n"
        "                    val_tokens,\n"
        "                    base_bytes_lut,\n"
        "                    has_leading_space_lut,\n"
        "                    is_boundary_token_lut,\n"
        "                )\n"
        "            else:\n"
        "                quant_obj_local, _ = quantize_state_dict_int8(base_model.state_dict())\n"
        "                base_model.load_state_dict(dequantize_state_dict_int8(quant_obj_local), strict=True)\n"
        "                cand_loss, cand_bpb = eval_val(\n"
        "                    args,\n"
        "                    model,\n"
        "                    rank,\n"
        "                    world_size,\n"
        "                    device,\n"
        "                    grad_accum_steps,\n"
        "                    val_tokens,\n"
        "                    base_bytes_lut,\n"
        "                    has_leading_space_lut,\n"
        "                    is_boundary_token_lut,\n"
        "                )\n"
        "            log0(f\"snapshot_select:candidate:{tag} score_mode:{snapshot_select_score} val_loss:{cand_loss:.8f} val_bpb:{cand_bpb:.8f}\")\n"
        "            return cand_loss, cand_bpb\n"
        "        finally:\n"
        "            base_model.load_state_dict(current_state, strict=True)\n",
        name,
    )
    source = _replace_unique(
        source,
        "        step += 1\n",
        "        step += 1\n"
        "        maybe_capture_snapshot(step)\n",
        name,
    )
    source = _replace_unique(
        source,
        "    # -----------------------------\n"
        "    # SERIALIZATION + ROUNDTRIP VALIDATION\n",
        "    if snapshot_select_enabled:\n"
        "        raw_final_state = clone_state_dict_cpu(base_model.state_dict())\n"
        "        snapshot_candidates.append((f\"final_step_{step}\", raw_final_state))\n"
        "        if snapshot_select_mode in {\"ema_vs_raw\", \"deployed_score\"} and \"ema_state\" in locals() and ema_state is not None:\n"
        "            snapshot_candidates.append((\"ema_final\", {state_name: tensor.detach().cpu().clone() for state_name, tensor in ema_state.items()}))\n"
        "        best_tag = None\n"
        "        best_state = None\n"
        "        best_bpb = None\n"
        "        for tag, candidate_state in snapshot_candidates:\n"
        "            _, candidate_bpb = score_state_candidate(tag, candidate_state)\n"
        "            if best_bpb is None or candidate_bpb < best_bpb:\n"
        "                best_bpb = candidate_bpb\n"
        "                best_tag = tag\n"
        "                best_state = candidate_state\n"
        "        if best_state is not None:\n"
        "            log0(f\"snapshot_select:chosen tag:{best_tag} val_bpb:{best_bpb:.8f} mode:{snapshot_select_mode} score_mode:{snapshot_select_score}\")\n"
        "            base_model.load_state_dict(best_state, strict=True)\n"
        "\n"
        "    # -----------------------------\n"
        "    # SERIALIZATION + ROUNDTRIP VALIDATION\n",
        name,
    )
    return source


def patch_alternating_objective(source: str) -> str:
    name = "alternating_objective"
    source = _replace_unique(
        source,
        "    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None\n",
        "    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None\n"
        "    alt_objective_enabled = os.environ.get(\"ALT_OBJECTIVE_ENABLE\", \"0\") == \"1\"\n"
        "    alt_objective_mode = os.environ.get(\"ALT_OBJECTIVE_MODE\", \"export_surrogate\")\n"
        "    alt_objective_every = int(os.environ.get(\"ALT_OBJECTIVE_EVERY\", \"8\"))\n"
        "    alt_objective_start_frac = float(os.environ.get(\"ALT_OBJECTIVE_START_FRAC\", \"0.75\"))\n"
        "    alt_objective_weight = float(os.environ.get(\"ALT_OBJECTIVE_WEIGHT\", \"0.05\"))\n"
        "    alt_weight_perturb_scale = float(os.environ.get(\"ALT_WEIGHT_PERTURB_SCALE\", os.environ.get(\"WEIGHT_PERTURB_SCALE\", \"0.01\")))\n"
        "\n"
        "    def active_alt_mode(current_step: int) -> str | None:\n"
        "        if not alt_objective_enabled:\n"
        "            return None\n"
        "        if current_step < int(alt_objective_start_frac * args.iterations):\n"
        "            return None\n"
        "        if alt_objective_every > 1 and current_step % alt_objective_every != 0:\n"
        "            return None\n"
        "        if alt_objective_mode != \"dual_pulse\":\n"
        "            return alt_objective_mode\n"
        "        pulse_idx = current_step // max(alt_objective_every, 1)\n"
        "        return \"weight_perturb\" if pulse_idx % 2 == 0 else \"export_surrogate\"\n"
        "\n"
        "    def export_surrogate_penalty() -> Tensor:\n"
        "        penalty = torch.zeros((), device=device)\n"
        "        count = 0\n"
        "        for param in base_model.parameters():\n"
        "            if param.ndim != 2:\n"
        "                continue\n"
        "            weight32 = param.float()\n"
        "            row_scale = weight32.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0 / 127.0) / 127.0\n"
        "            q = torch.clamp(torch.round(weight32 / row_scale), -127, 127)\n"
        "            recon = q * row_scale\n"
        "            penalty = penalty + (weight32 - recon).square().mean()\n"
        "            count += 1\n"
        "        return penalty / max(count, 1)\n"
        "\n"
        "    def apply_alt_weight_perturb() -> list[tuple[Tensor, Tensor]]:\n"
        "        applied: list[tuple[Tensor, Tensor]] = []\n"
        "        with torch.no_grad():\n"
        "            for param in base_model.parameters():\n"
        "                if param.ndim != 2:\n"
        "                    continue\n"
        "                noise = torch.randn_like(param) * alt_weight_perturb_scale\n"
        "                param.add_(noise)\n"
        "                applied.append((param, noise))\n"
        "        return applied\n"
        "\n"
        "    def restore_alt_weight_perturb(applied: list[tuple[Tensor, Tensor]]) -> None:\n"
        "        with torch.no_grad():\n"
        "            for param, noise in applied:\n"
        "                param.sub_(noise)\n",
        name,
    )
    source = _replace_unique(
        source,
        "        zero_grad_all()\n"
        "        train_loss = torch.zeros((), device=device)\n",
        "        zero_grad_all()\n"
        "        train_loss = torch.zeros((), device=device)\n"
        "        alt_mode = active_alt_mode(step)\n",
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
        "            alt_applied: list[tuple[Tensor, Tensor]] = []\n"
        "            if alt_mode == \"weight_perturb\":\n"
        "                alt_applied = apply_alt_weight_perturb()\n"
        "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "                loss = model(x, y)\n"
        "                if alt_mode in {\"late_qat\", \"export_surrogate\"}:\n"
        "                    loss = loss + alt_objective_weight * export_surrogate_penalty().to(loss.dtype)\n"
        "            train_loss += loss.detach()\n"
        "            (loss * grad_scale).backward()\n"
        "            if alt_applied:\n"
        "                restore_alt_weight_perturb(alt_applied)\n",
        name,
    )
    return source


def patch_leaky_relu_sq(source: str) -> str:
    return _S21.patch_leaky_relu_sq(source)


def patch_fa3(source: str) -> str:
    return _S21.patch_fa3(source)


def patch_muon_weight_decay(source: str) -> str:
    return _S21.patch_muon_weight_decay(source)


def patch_partial_rope(source: str) -> str:
    return _S21.patch_partial_rope(source)


def patch_ln_scale(source: str) -> str:
    return _S21.patch_ln_scale(source)


def patch_gptq_lite(source: str) -> str:
    return _S21.patch_gptq_lite(source)


def patch_zstd_export(source: str) -> str:
    return _S21.patch_zstd_export(source)


def patch_weight_perturbation(source: str) -> str:
    # Reuse the validated stage3-inspired flat-minima patch from stage2_1.
    return _S21.patch_weight_perturbation(source)


def patch_grad_centralization(source: str) -> str:
    # Reuse the validated stage3-inspired gradient preprocessing patch from stage2_1.
    return _S21.patch_grad_centralization(source)


def patch_full_gptq(source: str) -> str:
    name = "full_gptq"
    source = _replace_unique(
        source,
        "import io\n"
        "import math\n",
        "import io\n"
        "import lzma\n"
        "import math\n",
        name,
    )
    source = _replace_unique(
        source,
        "\n# -----------------------------\n# POST-TRAINING QUANTIZATION\n",
        "\n\ndef collect_hessians(\n"
        "    base_model: nn.Module,\n"
        "    train_loader: DistributedTokenLoader,\n"
        "    args: Hyperparameters,\n"
        "    device: torch.device,\n"
        "    grad_accum_steps: int,\n"
        "    num_batches: int = 256,\n"
        ") -> dict[str, Tensor]:\n"
        "    hessians: dict[str, Tensor] = {}\n"
        "    hooks: list[torch.utils.hooks.RemovableHandle] = []\n"
        "    was_training = base_model.training\n"
        "    for module_name, module in base_model.named_modules():\n"
        "        if isinstance(module, CastedLinear):\n"
        "            param_name = module_name + \".weight\"\n"
        "            cols = module.weight.shape[1]\n"
        "            hessians[param_name] = torch.zeros((cols, cols), dtype=torch.float32, device=\"cpu\")\n"
        "\n"
        "            def make_hook(pname: str):\n"
        "                def hook_fn(_module: nn.Module, module_input: tuple[Tensor, ...], _output: Tensor) -> None:\n"
        "                    x = module_input[0].detach().float()\n"
        "                    if x.ndim == 3:\n"
        "                        x = x.reshape(-1, x.shape[-1])\n"
        "                    hessians[pname].add_((x.T @ x).cpu())\n"
        "\n"
        "                return hook_fn\n"
        "\n"
        "            hooks.append(module.register_forward_hook(make_hook(param_name)))\n"
        "    base_model.eval()\n"
        "    try:\n"
        "        with torch.inference_mode(), torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "            for _ in range(num_batches):\n"
        "                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)\n"
        "                base_model(x, y)\n"
        "    finally:\n"
        "        for hook in hooks:\n"
        "            hook.remove()\n"
        "        base_model.train(was_training)\n"
        "    norm = float(max(num_batches, 1))\n"
        "    for pname, hessian in hessians.items():\n"
        "        hessians[pname] = (hessian / norm).contiguous()\n"
        "    return hessians\n"
        "\n"
        "# -----------------------------\n"
        "# POST-TRAINING QUANTIZATION\n",
        name,
    )
    source = _replace_unique(
        source,
        "CONTROL_TENSOR_NAME_PATTERNS = tuple(\n"
        "    pattern\n"
        "    for pattern in os.environ.get(\n"
        "        \"CONTROL_TENSOR_NAME_PATTERNS\",\n"
        "        \"attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights\",\n"
        "    ).split(\",\")\n"
        "    if pattern\n"
        ")\n"
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(\n"
        "    pattern\n"
        "    for pattern in os.environ.get(\n"
        "        \"INT8_KEEP_FLOAT_FP32_NAME_PATTERNS\",\n"
        "        \",\".join(CONTROL_TENSOR_NAME_PATTERNS),\n"
        "    ).split(\",\")\n"
        "    if pattern\n"
        ")\n"
        "INT8_KEEP_FLOAT_MAX_NUMEL = 65_536\n"
        "INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16\n"
        "INT8_PER_ROW_SCALE_DTYPE = torch.float16\n"
        "INT8_CLIP_PERCENTILE = 99.99984\n"
        "INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0\n"
        "\n"
        "def tensor_nbytes(t: Tensor) -> int:\n"
        "    return int(t.numel()) * int(t.element_size())\n"
        "\n"
        "def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:\n"
        "    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):\n"
        "        return t.float().contiguous()\n"
        "    if t.dtype in {torch.float32, torch.bfloat16}:\n"
        "        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()\n"
        "    return t\n"
        "\n"
        "def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:\n"
        "    t32 = t.float()\n"
        "    if t32.ndim == 2:\n"
        "        # Matrices get one scale per row, which usually tracks output-channel\n"
        "        # ranges much better than a single tensor-wide scale.\n"
        "        clip_abs = (\n"
        "            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)\n"
        "            if t32.numel()\n"
        "            else torch.empty((t32.shape[0],), dtype=torch.float32)\n"
        "        )\n"
        "        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])\n"
        "        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)\n"
        "        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()\n"
        "        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()\n"
        "\n"
        "    # Vectors / scalars use a simpler per-tensor scale.\n"
        "    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0\n"
        "    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)\n"
        "    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()\n"
        "    return q, scale\n"
        "\n"
        "def quantize_state_dict_int8(state_dict: dict[str, Tensor]):\n"
        "    # Single supported clean-script export format:\n"
        "    # - per-row int8 for 2D float tensors\n"
        "    # - per-tensor int8 for other float tensors\n"
        "    # - exact passthrough for non-floats\n"
        "    # - passthrough for small float tensors, stored as fp16 to save bytes\n"
        "    quantized: dict[str, Tensor] = {}\n"
        "    scales: dict[str, Tensor] = {}\n"
        "    dtypes: dict[str, str] = {}\n"
        "    passthrough: dict[str, Tensor] = {}\n"
        "    passthrough_orig_dtypes: dict[str, str] = {}\n"
        "    qmeta: dict[str, dict[str, object]] = {}\n"
        "    stats = dict.fromkeys(\n"
        "        (\"param_count\", \"num_tensors\", \"num_float_tensors\", \"num_nonfloat_tensors\", \"baseline_tensor_bytes\", \"int8_payload_bytes\"),\n"
        "        0,\n"
        "    )\n"
        "\n"
        "    for name, tensor in state_dict.items():\n"
        "        t = tensor.detach().to(\"cpu\").contiguous()\n"
        "        stats[\"param_count\"] += int(t.numel())\n"
        "        stats[\"num_tensors\"] += 1\n"
        "        stats[\"baseline_tensor_bytes\"] += tensor_nbytes(t)\n"
        "\n"
        "        if not t.is_floating_point():\n"
        "            stats[\"num_nonfloat_tensors\"] += 1\n"
        "            passthrough[name] = t\n"
        "            stats[\"int8_payload_bytes\"] += tensor_nbytes(t)\n"
        "            continue\n"
        "\n"
        "        # Small float tensors are cheap enough to keep directly. We still downcast\n"
        "        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.\n"
        "        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:\n"
        "            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)\n"
        "            passthrough[name] = kept\n"
        "            stats[\"int8_payload_bytes\"] += tensor_nbytes(kept)\n"
        "            continue\n"
        "\n"
        "        stats[\"num_float_tensors\"] += 1\n"
        "        q, s = quantize_float_tensor(t)\n"
        "        if s.ndim > 0:\n"
        "            qmeta[name] = {\"scheme\": \"per_row\", \"axis\": 0}\n"
        "        quantized[name] = q\n"
        "        scales[name] = s\n"
        "        dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        stats[\"int8_payload_bytes\"] += tensor_nbytes(q) + tensor_nbytes(s)\n"
        "\n"
        "    obj: dict[str, object] = {\n"
        "        \"__quant_format__\": \"int8_clean_per_row_v1\",\n"
        "        \"quantized\": quantized,\n"
        "        \"scales\": scales,\n"
        "        \"dtypes\": dtypes,\n"
        "        \"passthrough\": passthrough,\n"
        "    }\n"
        "    if qmeta:\n"
        "        obj[\"qmeta\"] = qmeta\n"
        "    if passthrough_orig_dtypes:\n"
        "        obj[\"passthrough_orig_dtypes\"] = passthrough_orig_dtypes\n"
        "    return obj, stats\n"
        "\n"
        "def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:\n"
        "    out: dict[str, Tensor] = {}\n"
        "    qmeta = obj.get(\"qmeta\", {})\n"
        "    passthrough_orig_dtypes = obj.get(\"passthrough_orig_dtypes\", {})\n"
        "    for name, q in obj[\"quantized\"].items():\n"
        "        dtype = getattr(torch, obj[\"dtypes\"][name])\n"
        "        s = obj[\"scales\"][name]\n"
        "        if qmeta.get(name, {}).get(\"scheme\") == \"per_row\" or s.ndim > 0:\n"
        "            s = s.to(dtype=torch.float32)\n"
        "            # Broadcast the saved row scale back across trailing dimensions.\n"
        "            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()\n"
        "        else:\n"
        "            scale = float(s.item())\n"
        "            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()\n"
        "    for name, t in obj[\"passthrough\"].items():\n"
        "        # Restore small tensors, undoing the temporary fp16 storage cast if needed.\n"
        "        out_t = t.detach().to(\"cpu\").contiguous()\n"
        "        orig_dtype = passthrough_orig_dtypes.get(name)\n"
        "        if isinstance(orig_dtype, str):\n"
        "            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()\n"
        "        out[name] = out_t\n"
        "    return out\n",
        "CONTROL_TENSOR_NAME_PATTERNS = tuple(\n"
        "    pattern\n"
        "    for pattern in os.environ.get(\n"
        "        \"CONTROL_TENSOR_NAME_PATTERNS\",\n"
        "        \"attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights\",\n"
        "    ).split(\",\")\n"
        "    if pattern\n"
        ")\n"
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(\n"
        "    pattern\n"
        "    for pattern in os.environ.get(\n"
        "        \"INT8_KEEP_FLOAT_FP32_NAME_PATTERNS\",\n"
        "        \",\".join(CONTROL_TENSOR_NAME_PATTERNS),\n"
        "    ).split(\",\")\n"
        "    if pattern\n"
        ")\n"
        "INT8_KEEP_FLOAT_MAX_NUMEL = 65_536\n"
        "INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16\n"
        "INT8_PER_ROW_SCALE_DTYPE = torch.float16\n"
        "INT8_CLIP_PERCENTILE = 99.99984\n"
        "INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0\n"
        "\n"
        "\n"
        "def tensor_nbytes(t: Tensor) -> int:\n"
        "    return int(t.numel()) * int(t.element_size())\n"
        "\n"
        "\n"
        "def _classify_param(name: str) -> str:\n"
        "    if \"tok_emb\" in name or \"lm_head\" in name:\n"
        "        return \"embed\"\n"
        "    if \".mlp.\" in name:\n"
        "        return \"mlp\"\n"
        "    if \".attn.\" in name or (\".proj.\" in name and \".mlp.\" not in name):\n"
        "        return \"attn\"\n"
        "    if \"count_bigram\" in name:\n"
        "        return \"bigram\"\n"
        "    return \"other\"\n"
        "\n"
        "\n"
        "def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:\n"
        "    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):\n"
        "        return t.float().contiguous()\n"
        "    if t.dtype in {torch.float32, torch.bfloat16}:\n"
        "        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()\n"
        "    return t\n"
        "\n"
        "\n"
        "def _quantize_int6_percentile(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:\n"
        "    t32 = t.float()\n"
        "    if t32.ndim == 2:\n"
        "        best_q = None\n"
        "        best_s = None\n"
        "        best_err = float(\"inf\")\n"
        "        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:\n"
        "            row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)\n"
        "            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)\n"
        "            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)\n"
        "            recon = q.float() * s.float()[:, None]\n"
        "            err = float((t32 - recon).pow(2).mean().item())\n"
        "            if err < best_err:\n"
        "                best_q = q\n"
        "                best_s = s\n"
        "                best_err = err\n"
        "        return best_q.contiguous(), best_s.contiguous()\n"
        "    amax = float(t32.abs().max().item()) if t32.numel() else 0.0\n"
        "    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)\n"
        "    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8).contiguous()\n"
        "    return q, scale\n"
        "\n"
        "\n"
        "def quantize_int6_gptq(\n"
        "    weight: Tensor,\n"
        "    hessian: Tensor | None = None,\n"
        "    clip_range: int = 31,\n"
        "    block_size: int = 128,\n"
        ") -> tuple[Tensor, Tensor]:\n"
        "    t32 = weight.float()\n"
        "    if t32.ndim != 2 or hessian is None:\n"
        "        return _quantize_int6_percentile(t32, clip_range)\n"
        "    rows, cols = t32.shape\n"
        "    H = hessian.float().clone()\n"
        "    dead = torch.diag(H) == 0\n"
        "    if dead.any():\n"
        "        H[dead, dead] = 1\n"
        "    damp = 0.01 * torch.mean(torch.diag(H)).clamp_min(1e-6)\n"
        "    H[torch.arange(cols), torch.arange(cols)] += damp\n"
        "    perm = torch.argsort(torch.diag(H), descending=True)\n"
        "    inv_perm = torch.argsort(perm)\n"
        "    W = t32[:, perm].clone()\n"
        "    if dead.any():\n"
        "        W[:, dead[perm]] = 0\n"
        "    H = H[perm][:, perm]\n"
        "    try:\n"
        "        Hinv = torch.linalg.cholesky(H)\n"
        "        Hinv = torch.cholesky_inverse(Hinv)\n"
        "        Hinv = torch.linalg.cholesky(Hinv, upper=True)\n"
        "    except torch.linalg.LinAlgError:\n"
        "        return _quantize_int6_percentile(t32, clip_range)\n"
        "    best_q = None\n"
        "    best_scale = None\n"
        "    best_err = float(\"inf\")\n"
        "    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:\n"
        "        row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)\n"
        "        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)\n"
        "        sf = s.float()\n"
        "        Q = torch.zeros_like(W, dtype=torch.int8)\n"
        "        W_work = W.clone()\n"
        "        for i1 in range(0, cols, block_size):\n"
        "            i2 = min(i1 + block_size, cols)\n"
        "            width = i2 - i1\n"
        "            W1 = W_work[:, i1:i2].clone()\n"
        "            Q1 = torch.zeros((rows, width), dtype=torch.int8)\n"
        "            Err1 = torch.zeros((rows, width), dtype=torch.float32)\n"
        "            Hinv1 = Hinv[i1:i2, i1:i2]\n"
        "            for i in range(width):\n"
        "                w = W1[:, i]\n"
        "                d = Hinv1[i, i].clamp_min(1e-8)\n"
        "                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)\n"
        "                Q1[:, i] = q\n"
        "                err = (w - q.float() * sf) / d\n"
        "                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)\n"
        "                Err1[:, i] = err\n"
        "            Q[:, i1:i2] = Q1\n"
        "            if i2 < cols:\n"
        "                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]\n"
        "        recon = Q.float() * sf[:, None]\n"
        "        mse = float((W - recon).pow(2).mean().item())\n"
        "        if mse < best_err:\n"
        "            best_q = Q[:, inv_perm]\n"
        "            best_scale = s\n"
        "            best_err = mse\n"
        "    return best_q.contiguous(), best_scale.contiguous()\n"
        "\n"
        "\n"
        "def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:\n"
        "    t32 = t.float()\n"
        "    if t32.ndim == 2:\n"
        "        clip_abs = (\n"
        "            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)\n"
        "            if t32.numel()\n"
        "            else torch.empty((t32.shape[0],), dtype=torch.float32)\n"
        "        )\n"
        "        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])\n"
        "        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)\n"
        "        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()\n"
        "        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()\n"
        "    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0\n"
        "    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)\n"
        "    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()\n"
        "    return q, scale\n"
        "\n"
        "\n"
        "def quantize_state_dict_int8(\n"
        "    state_dict: dict[str, Tensor],\n"
        "    hessians: dict[str, Tensor] | None = None,\n"
        "):\n"
        "    quantized: dict[str, Tensor] = {}\n"
        "    scales: dict[str, Tensor] = {}\n"
        "    dtypes: dict[str, str] = {}\n"
        "    passthrough: dict[str, Tensor] = {}\n"
        "    passthrough_orig_dtypes: dict[str, str] = {}\n"
        "    qmeta: dict[str, dict[str, object]] = {}\n"
        "    stats = dict.fromkeys(\n"
        "        (\"param_count\", \"num_tensors\", \"num_float_tensors\", \"num_nonfloat_tensors\", \"baseline_tensor_bytes\", \"int8_payload_bytes\"),\n"
        "        0,\n"
        "    )\n"
        "    gptq_enabled = os.environ.get(\"GPTQ_ENABLE\", \"0\") == \"1\"\n"
        "    gptq_block_size = int(os.environ.get(\"GPTQ_BLOCK_SIZE\", \"128\"))\n"
        "    int6_categories = {token.strip() for token in os.environ.get(\"GPTQ_INT6_CATEGORIES\", \"mlp,attn,bigram\").split(\",\") if token.strip()}\n"
        "    for name, tensor in state_dict.items():\n"
        "        t = tensor.detach().to(\"cpu\").contiguous()\n"
        "        stats[\"param_count\"] += int(t.numel())\n"
        "        stats[\"num_tensors\"] += 1\n"
        "        stats[\"baseline_tensor_bytes\"] += tensor_nbytes(t)\n"
        "        if not t.is_floating_point():\n"
        "            stats[\"num_nonfloat_tensors\"] += 1\n"
        "            passthrough[name] = t\n"
        "            stats[\"int8_payload_bytes\"] += tensor_nbytes(t)\n"
        "            continue\n"
        "        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:\n"
        "            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)\n"
        "            passthrough[name] = kept\n"
        "            stats[\"int8_payload_bytes\"] += tensor_nbytes(kept)\n"
        "            continue\n"
        "        stats[\"num_float_tensors\"] += 1\n"
        "        cat = _classify_param(name)\n"
        "        use_int6 = gptq_enabled and cat in int6_categories and t.ndim == 2\n"
        "        if use_int6:\n"
        "            q, s = quantize_int6_gptq(t, hessian=hessians.get(name) if hessians else None, clip_range=31, block_size=gptq_block_size)\n"
        "            qmeta[name] = {\"scheme\": \"per_row\", \"axis\": 0, \"bits\": 6}\n"
        "        else:\n"
        "            q, s = quantize_float_tensor(t)\n"
        "            if s.ndim > 0:\n"
        "                qmeta[name] = {\"scheme\": \"per_row\", \"axis\": 0, \"bits\": 8}\n"
        "        quantized[name] = q\n"
        "        scales[name] = s\n"
        "        dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        stats[\"int8_payload_bytes\"] += tensor_nbytes(q) + tensor_nbytes(s)\n"
        "    obj: dict[str, object] = {\n"
        "        \"__quant_format__\": \"gptq_int6_lzma_v1\" if gptq_enabled else \"int8_clean_per_row_v1\",\n"
        "        \"quantized\": quantized,\n"
        "        \"scales\": scales,\n"
        "        \"dtypes\": dtypes,\n"
        "        \"passthrough\": passthrough,\n"
        "    }\n"
        "    if qmeta:\n"
        "        obj[\"qmeta\"] = qmeta\n"
        "    if passthrough_orig_dtypes:\n"
        "        obj[\"passthrough_orig_dtypes\"] = passthrough_orig_dtypes\n"
        "    return obj, stats\n"
        "\n"
        "\n"
        "def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:\n"
        "    out: dict[str, Tensor] = {}\n"
        "    qmeta = obj.get(\"qmeta\", {})\n"
        "    passthrough_orig_dtypes = obj.get(\"passthrough_orig_dtypes\", {})\n"
        "    for name, q in obj[\"quantized\"].items():\n"
        "        dtype = getattr(torch, obj[\"dtypes\"][name])\n"
        "        s = obj[\"scales\"][name]\n"
        "        if qmeta.get(name, {}).get(\"scheme\") == \"per_row\" or s.ndim > 0:\n"
        "            s = s.to(dtype=torch.float32)\n"
        "            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()\n"
        "        else:\n"
        "            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()\n"
        "    for name, t in obj[\"passthrough\"].items():\n"
        "        out_t = t.detach().to(\"cpu\").contiguous()\n"
        "        orig_dtype = passthrough_orig_dtypes.get(name)\n"
        "        if isinstance(orig_dtype, str):\n"
        "            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()\n"
        "        out[name] = out_t\n"
        "    return out\n",
        name,
    )
    source = _replace_unique(
        source,
        "    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())\n"
        "    quant_buf = io.BytesIO()\n"
        "    torch.save(quant_obj, quant_buf)\n"
        "    quant_raw = quant_buf.getvalue()\n"
        "    quant_blob = zlib.compress(quant_raw, level=9)\n"
        "    quant_raw_bytes = len(quant_raw)\n"
        "    if master_process:\n"
        "        with open(\"final_model.int8.ptz\", \"wb\") as f:\n"
        "            f.write(quant_blob)\n"
        "        quant_file_bytes = os.path.getsize(\"final_model.int8.ptz\")\n"
        "        code_bytes = len(code.encode(\"utf-8\"))\n"
        "        ratio = quant_stats[\"baseline_tensor_bytes\"] / max(quant_stats[\"int8_payload_bytes\"], 1)\n"
        "        log0(\n"
        "            f\"Serialized model int8+zlib: {quant_file_bytes} bytes \"\n"
        "            f\"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)\"\n"
        "        )\n"
        "        log0(f\"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes\")\n"
        "\n"
        "    if distributed:\n"
        "        dist.barrier()\n"
        "    with open(\"final_model.int8.ptz\", \"rb\") as f:\n"
        "        quant_blob_disk = f.read()\n"
        "    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location=\"cpu\")\n"
        "    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)",
        "    gptq_enabled = os.environ.get(\"GPTQ_ENABLE\", \"0\") == \"1\"\n"
        "    if gptq_enabled:\n"
        "        gptq_calib_batches = int(os.environ.get(\"GPTQ_CALIB_BATCHES\", \"256\"))\n"
        "        log0(f\"gptq:calibrating with {gptq_calib_batches} batches...\")\n"
        "        calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n"
        "        hessians = collect_hessians(\n"
        "            base_model,\n"
        "            calib_loader,\n"
        "            args,\n"
        "            device,\n"
        "            grad_accum_steps,\n"
        "            num_batches=gptq_calib_batches,\n"
        "        )\n"
        "        log0(f\"gptq:collected hessians for {len(hessians)} layers\")\n"
        "        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict(), hessians=hessians)\n"
        "        quant_buf = io.BytesIO()\n"
        "        torch.save(quant_obj, quant_buf)\n"
        "        quant_raw = quant_buf.getvalue()\n"
        "        quant_blob = lzma.compress(quant_raw, preset=int(os.environ.get(\"GPTQ_LZMA_PRESET\", \"6\")))\n"
        "        quant_raw_bytes = len(quant_raw)\n"
        "        if master_process:\n"
        "            with open(\"final_model.int6.ptz\", \"wb\") as f:\n"
        "                f.write(quant_blob)\n"
        "            quant_file_bytes = os.path.getsize(\"final_model.int6.ptz\")\n"
        "            code_bytes = len(code.encode(\"utf-8\"))\n"
        "            ratio = quant_stats[\"baseline_tensor_bytes\"] / max(quant_stats[\"int8_payload_bytes\"], 1)\n"
        "            log0(\n"
        "                f\"Serialized model int6+lzma: {quant_file_bytes} bytes \"\n"
        "                f\"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)\"\n"
        "            )\n"
        "            log0(f\"Total submission size int6+lzma: {quant_file_bytes + code_bytes} bytes\")\n"
        "\n"
        "        if distributed:\n"
        "            dist.barrier()\n"
        "        with open(\"final_model.int6.ptz\", \"rb\") as f:\n"
        "            quant_blob_disk = f.read()\n"
        "        quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob_disk)), map_location=\"cpu\")\n"
        "        base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)\n"
        "    else:\n"
        "        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())\n"
        "        quant_buf = io.BytesIO()\n"
        "        torch.save(quant_obj, quant_buf)\n"
        "        quant_raw = quant_buf.getvalue()\n"
        "        quant_blob = zlib.compress(quant_raw, level=9)\n"
        "        quant_raw_bytes = len(quant_raw)\n"
        "        if master_process:\n"
        "            with open(\"final_model.int8.ptz\", \"wb\") as f:\n"
        "                f.write(quant_blob)\n"
        "            quant_file_bytes = os.path.getsize(\"final_model.int8.ptz\")\n"
        "            code_bytes = len(code.encode(\"utf-8\"))\n"
        "            ratio = quant_stats[\"baseline_tensor_bytes\"] / max(quant_stats[\"int8_payload_bytes\"], 1)\n"
        "            log0(\n"
        "                f\"Serialized model int8+zlib: {quant_file_bytes} bytes \"\n"
        "                f\"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)\"\n"
        "            )\n"
        "            log0(f\"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes\")\n"
        "\n"
        "        if distributed:\n"
        "            dist.barrier()\n"
        "        with open(\"final_model.int8.ptz\", \"rb\") as f:\n"
        "            quant_blob_disk = f.read()\n"
        "        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location=\"cpu\")\n"
        "        base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)",
        name,
    )
    source = _replace_unique(
        source,
        "    log0(\n"
        "        f\"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} \"\n"
        "        f\"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms\"\n"
        "    )\n"
        "    log0(f\"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}\")",
        "    quant_prefix = \"final_int6_lzma_roundtrip\" if os.environ.get(\"GPTQ_ENABLE\", \"0\") == \"1\" else \"final_int8_zlib_roundtrip\"\n"
        "    log0(\n"
        "        f\"{quant_prefix} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} \"\n"
        "        f\"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms\"\n"
        "    )\n"
        "    log0(f\"{quant_prefix}_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}\")",
        name,
    )
    return source


def patch_late_qat_active(source: str) -> str:
    name = "late_qat_active"
    source = _replace_unique(
        source,
        "class CastedLinear(nn.Linear):\n"
        "    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        bias = self.bias.to(x.dtype) if self.bias is not None else None\n"
        "        return F.linear(x, self.weight.to(x.dtype), bias)",
        "class _LateQATInt6STE(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx, w: Tensor) -> Tensor:\n"
        "        w32 = w.float()\n"
        "        abs_max = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)\n"
        "        scale = abs_max / 31.0\n"
        "        q = torch.clamp(torch.round(w32 / scale), -32, 31)\n"
        "        return (q * scale).to(w.dtype)\n"
        "\n"
        "    @staticmethod\n"
        "    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:\n"
        "        return (grad_output,)\n"
        "\n"
        "\n"
        "_late_qat_int6 = _LateQATInt6STE.apply\n"
        "\n"
        "\n"
        "class CastedLinear(nn.Linear):\n"
        "    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.\n"
        "    _late_qat_enabled = False\n"
        "\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        bias = self.bias.to(x.dtype) if self.bias is not None else None\n"
        "        w = self.weight.to(x.dtype)\n"
        "        if self.training and type(self)._late_qat_enabled and w.ndim == 2:\n"
        "            w = _late_qat_int6(w)\n"
        "        return F.linear(x, w, bias)",
        name,
    )
    source = _replace_unique(
        source,
        "        scale = lr_mul(step, elapsed_ms)\n"
        "        zero_grad_all()",
        "        scale = lr_mul(step, elapsed_ms)\n"
        "        if os.environ.get(\"LATE_QAT_ENABLE\", \"0\") == \"1\":\n"
        "            _threshold = float(os.environ.get(\"LATE_QAT_THRESHOLD\", \"0.15\"))\n"
        "            CastedLinear._late_qat_enabled = scale <= _threshold\n"
        "        else:\n"
        "            CastedLinear._late_qat_enabled = False\n"
        "        zero_grad_all()",
        name,
    )
    return source


def patch_countinit_bigram(source: str) -> str:
    name = "countinit_bigram"
    source = _replace_unique(
        source,
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.",
        "class CountInitBigramHead(nn.Module):\n"
        "    def __init__(self, vocab_size: int):\n"
        "        super().__init__()\n"
        "        self.table = nn.Parameter(torch.zeros(vocab_size, vocab_size, dtype=torch.float32))\n"
        "        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))\n"
        "\n"
        "    @torch.no_grad()\n"
        "    def initialize_from_tokens(self, tokens: Tensor, vocab_size: int, alpha: float = 0.1) -> None:\n"
        "        if tokens.numel() < 2:\n"
        "            return\n"
        "        prev_ids = tokens[:-1].to(dtype=torch.int64)\n"
        "        next_ids = tokens[1:].to(dtype=torch.int64)\n"
        "        flat = prev_ids * vocab_size + next_ids\n"
        "        counts = torch.bincount(flat, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size).float()\n"
        "        row_sum = counts.sum(dim=1, keepdim=True)\n"
        "        col_sum = counts.sum(dim=0, keepdim=True)\n"
        "        total = counts.sum().clamp_min(1.0)\n"
        "        cond = (counts + alpha) / (row_sum + alpha * vocab_size)\n"
        "        marg = (col_sum + alpha) / (total + alpha * vocab_size)\n"
        "        self.table.copy_((cond.log() - marg.log()).float())\n"
        "\n"
        "    def forward(self, prev_ids: Tensor) -> Tensor:\n"
        "        return self.scale.to(dtype=self.table.dtype) * self.table[prev_ids]\n"
        "\n"
        "\n"
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.",
        name,
    )
    source = _replace_unique(
        source,
        "        self.final_norm = RMSNorm()\n"
        "        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)",
        "        self.final_norm = RMSNorm()\n"
        "        self.count_bigram = CountInitBigramHead(vocab_size) if os.environ.get(\"COUNTINIT_BIGRAM\", \"0\") == \"1\" else None\n"
        "        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)",
        name,
    )
    source = _replace_unique(
        source,
        "        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)\n"
        "        return F.cross_entropy(logits.float(), targets, reduction=\"mean\")",
        "        if self.count_bigram is not None:\n"
        "            prev_flat = input_ids.reshape(-1)\n"
        "            logits_proj = logits_proj + self.count_bigram(prev_flat).to(dtype=logits_proj.dtype)\n"
        "        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)\n"
        "        return F.cross_entropy(logits.float(), targets, reduction=\"mean\")",
        name,
    )
    source = _replace_unique(
        source,
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)",
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)\n"
        "    if getattr(base_model, 'count_bigram', None) is not None:\n"
        "        scalar_params.extend(base_model.count_bigram.parameters())",
        name,
    )
    source = _replace_unique(
        source,
        "    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n"
        "\n"
        "    def zero_grad_all() -> None:\n",
        "    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n"
        "    if getattr(base_model, 'count_bigram', None) is not None:\n"
        "        _sample_tokens = int(os.environ.get(\"COUNTINIT_BIGRAM_TOKENS\", \"2000000\"))\n"
        "        _stream = TokenStream(args.train_files)\n"
        "        _tokens = _stream.take(_sample_tokens + 1).to(dtype=torch.int64)\n"
        "        base_model.count_bigram.initialize_from_tokens(_tokens.cpu(), args.vocab_size)\n"
        "        log0(f\"countinit_bigram:initialized sample_tokens:{_sample_tokens}\")\n"
        "\n"
        "    def zero_grad_all() -> None:\n",
        name,
    )
    return source


PATCH_REGISTRY = {
    "alternating_objective": patch_alternating_objective,
    "checkpoint_selection": patch_checkpoint_selection,
    "countinit_bigram": patch_countinit_bigram,
    "curriculum_shard_order": patch_curriculum_shard_order,
    "ema_export": patch_ema_export,
    "exact_overlap_eval": patch_exact_overlap_eval,
    "fa3": patch_fa3,
    "full_gptq": patch_full_gptq,
    "gptq_lite": patch_gptq_lite,
    "grad_centralization": patch_grad_centralization,
    "late_qat_active": patch_late_qat_active,
    "leaky_relu_sq": patch_leaky_relu_sq,
    "ln_scale": patch_ln_scale,
    "muon_weight_decay": patch_muon_weight_decay,
    "partial_rope": patch_partial_rope,
    "staged_curriculum": patch_staged_curriculum,
    "weight_perturbation": patch_weight_perturbation,
    "xsa4": patch_xsa4,
    "xsa_all": patch_xsa_all,
    "zstd_export": patch_zstd_export,
}

PATCH_ORDER = {
    "staged_curriculum": 5,
    "alternating_objective": 15,
    "partial_rope": 10,
    "xsa4": 20,
    "xsa_all": 20,
    "ln_scale": 30,
    "full_gptq": 40,
    "checkpoint_selection": 90,
}


def apply_patches(source: str, patch_names: list[str]) -> str:
    ordered_patch_names = sorted(
        enumerate(patch_names),
        key=lambda item: (PATCH_ORDER.get(item[1], 100), item[0]),
    )
    for _, patch_name in ordered_patch_names:
        try:
            patch_fn = PATCH_REGISTRY[patch_name]
        except KeyError as exc:
            raise ValueError(f"Unknown patch '{patch_name}'. Available: {sorted(PATCH_REGISTRY)}") from exc
        source = patch_fn(source)
    return source


def list_patches() -> list[str]:
    return sorted(PATCH_REGISTRY)
