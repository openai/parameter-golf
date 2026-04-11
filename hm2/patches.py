"""
Runtime code patches for hm2 bootstrap-to-handoff experiments.

hm2 is built to answer one question honestly:

    if a mechanism wins early but flattens late, should we keep it on?

The active stage exploits that by:
- bootstrapping with count-initialized bigram priors
- explicitly fading, freezing, or plateau-triggering their handoff
- pairing the late phase with deploy-side receivers like checkpoint selection,
  late QAT, or pre-quant TTT
- recording early/mid/late dynamics for every run
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {tag} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parents[1]
_HM = _load_module("hailmary_patches_for_hm2", _ROOT / "hailmary" / "patches.py")
_replace_unique = _HM._replace_unique


def patch_countinit_bigram(source: str) -> str:
    return _HM.patch_countinit_bigram(source)


def patch_checkpoint_selection(source: str) -> str:
    name = "checkpoint_selection"
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


def patch_pre_quant_ttt(source: str) -> str:
    name = "pre_quant_ttt"
    source = _replace_unique(
        source,
        "def main() -> None:\n",
        "def run_pre_quant_ttt(\n"
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
        "            log0(f\"pre_quant_ttt:epoch {epoch_idx + 1}/{epochs} freeze_blocks:{freeze_blocks} chunks:{total_chunks} avg_loss:{epoch_loss / max(total_chunks, 1):.6f}\")\n"
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
        "    # -----------------------------\n"
        "    # SERIALIZATION + ROUNDTRIP VALIDATION\n",
        "    if os.environ.get(\"PRE_QUANT_TTT_ENABLE\", \"0\") == \"1\":\n"
        "        ttt_epochs = int(os.environ.get(\"PRE_QUANT_TTT_EPOCHS\", \"3\"))\n"
        "        ttt_lr = float(os.environ.get(\"PRE_QUANT_TTT_LR\", \"5e-4\"))\n"
        "        ttt_freeze_blocks = int(os.environ.get(\"PRE_QUANT_TTT_FREEZE_BLOCKS\", str(max(len(base_model.blocks) - 2, 0))))\n"
        "        ttt_stride = int(os.environ.get(\"PRE_QUANT_TTT_STRIDE\", str(max(args.train_seq_len // 8, 1))))\n"
        "        ttt_chunks = int(os.environ.get(\"PRE_QUANT_TTT_CHUNKS\", \"32\"))\n"
        "        ttt_block_lr_decay = float(os.environ.get(\"PRE_QUANT_TTT_BLOCK_LR_DECAY\", \"1.0\"))\n"
        "        ttt_grad_clip_norm = float(os.environ.get(\"PRE_QUANT_TTT_GRAD_CLIP_NORM\", str(args.grad_clip_norm)))\n"
        "        log0(\n"
        "            f\"pre_quant_ttt:enable epochs:{ttt_epochs} freeze_blocks:{ttt_freeze_blocks} lr:{ttt_lr:.6g} stride:{ttt_stride} chunks:{ttt_chunks} block_lr_decay:{ttt_block_lr_decay:.4f}\"\n"
        "        )\n"
        "        run_pre_quant_ttt(\n"
        "            base_model,\n"
        "            args,\n"
        "            device,\n"
        "            val_tokens,\n"
        "            freeze_blocks=ttt_freeze_blocks,\n"
        "            epochs=ttt_epochs,\n"
        "            lr=ttt_lr,\n"
        "            stride=ttt_stride,\n"
        "            max_chunks=ttt_chunks,\n"
        "            block_lr_decay=ttt_block_lr_decay,\n"
        "            grad_clip_norm=ttt_grad_clip_norm,\n"
        "            log0=log0,\n"
        "        )\n"
        "\n"
        "    # -----------------------------\n"
        "    # SERIALIZATION + ROUNDTRIP VALIDATION\n",
        name,
    )
    return source


def patch_late_qat_active(source: str) -> str:
    return _HM.patch_late_qat_active(source)


def patch_phase_diagnostics(source: str) -> str:
    name = "phase_diagnostics"
    source = _replace_unique(
        source,
        "import copy\n"
        "import glob\n",
        "import copy\n"
        "import glob\n"
        "import json\n",
        name,
    )
    source = _replace_unique(
        source,
        "    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None\n",
        "    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None\n"
        "    phase_diag_enabled = os.environ.get(\"PHASE_DIAGNOSTICS_ENABLE\", \"1\") == \"1\"\n"
        "    phase_diag_early_frac = float(os.environ.get(\"PHASE_DIAG_EARLY_FRAC\", \"0.33\"))\n"
        "    phase_diag_mid_frac = float(os.environ.get(\"PHASE_DIAG_MID_FRAC\", \"0.66\"))\n"
        "    phase_diag_buckets = (\"early\", \"mid\", \"late\")\n"
        "    phase_diag_stats = {\n"
        "        bucket: {\n"
        "            \"count\": 0,\n"
        "            \"first_step\": None,\n"
        "            \"last_step\": None,\n"
        "            \"loss_start\": None,\n"
        "            \"loss_end\": None,\n"
        "            \"best_loss\": None,\n"
        "            \"step_avg_ms_start\": None,\n"
        "            \"step_avg_ms_end\": None,\n"
        "            \"last_val_bpb\": None,\n"
        "            \"last_val_loss\": None,\n"
        "        }\n"
        "        for bucket in phase_diag_buckets\n"
        "    }\n"
        "    phase_diag_events: list[dict[str, object]] = []\n"
        "\n"
        "    def phase_diag_bucket(progress: float) -> str:\n"
        "        if progress < phase_diag_early_frac:\n"
        "            return \"early\"\n"
        "        if progress < phase_diag_mid_frac:\n"
        "            return \"mid\"\n"
        "        return \"late\"\n"
        "\n"
        "    def record_phase_event(tag: str, current_step: int, payload: dict[str, object] | None = None) -> None:\n"
        "        if not phase_diag_enabled:\n"
        "            return\n"
        "        event = {\n"
        "            \"tag\": tag,\n"
        "            \"step\": int(current_step),\n"
        "            \"progress\": float(current_step / max(args.iterations, 1)),\n"
        "        }\n"
        "        if payload:\n"
        "            event.update(payload)\n"
        "        phase_diag_events.append(event)\n"
        "\n"
        "    def phase_diag_update(current_step: int, train_loss_value: float, step_avg_ms: float) -> None:\n"
        "        if not phase_diag_enabled or current_step <= 0:\n"
        "            return\n"
        "        bucket = phase_diag_bucket(current_step / max(args.iterations, 1))\n"
        "        stats = phase_diag_stats[bucket]\n"
        "        stats[\"count\"] += 1\n"
        "        stats[\"first_step\"] = current_step if stats[\"first_step\"] is None else stats[\"first_step\"]\n"
        "        stats[\"last_step\"] = current_step\n"
        "        if stats[\"loss_start\"] is None:\n"
        "            stats[\"loss_start\"] = train_loss_value\n"
        "            stats[\"step_avg_ms_start\"] = step_avg_ms\n"
        "        stats[\"loss_end\"] = train_loss_value\n"
        "        stats[\"step_avg_ms_end\"] = step_avg_ms\n"
        "        stats[\"best_loss\"] = train_loss_value if stats[\"best_loss\"] is None else min(stats[\"best_loss\"], train_loss_value)\n"
        "\n"
        "    def phase_diag_attach_val(current_step: int, val_loss: float, val_bpb: float) -> None:\n"
        "        if not phase_diag_enabled:\n"
        "            return\n"
        "        bucket = phase_diag_bucket(current_step / max(args.iterations, 1))\n"
        "        stats = phase_diag_stats[bucket]\n"
        "        stats[\"last_val_loss\"] = float(val_loss)\n"
        "        stats[\"last_val_bpb\"] = float(val_bpb)\n"
        "\n"
        "    def finalize_phase_diagnostics(final_post_quant_bpb: float | None) -> None:\n"
        "        if not phase_diag_enabled or not master_process:\n"
        "            return\n"
        "        summary = {\n"
        "            \"slot\": os.environ.get(\"PGOLF_SLOT\", \"\"),\n"
        "            \"phase\": os.environ.get(\"PGOLF_PHASE\", \"\"),\n"
        "            \"run_id\": os.environ.get(\"RUN_ID\", \"\"),\n"
        "            \"buckets\": {},\n"
        "            \"events\": phase_diag_events,\n"
        "            \"final_post_quant_bpb\": final_post_quant_bpb,\n"
        "        }\n"
        "        for bucket, stats in phase_diag_stats.items():\n"
        "            loss_delta = None\n"
        "            slope_per_100 = None\n"
        "            if stats[\"loss_start\"] is not None and stats[\"loss_end\"] is not None:\n"
        "                loss_delta = float(stats[\"loss_start\"] - stats[\"loss_end\"])\n"
        "            if stats[\"count\"] and stats[\"first_step\"] is not None and stats[\"last_step\"] is not None and stats[\"last_step\"] > stats[\"first_step\"]:\n"
        "                step_span = max(stats[\"last_step\"] - stats[\"first_step\"], 1)\n"
        "                slope_per_100 = float((stats[\"loss_start\"] - stats[\"loss_end\"]) * 100.0 / step_span)\n"
        "            summary[\"buckets\"][bucket] = {\n"
        "                **stats,\n"
        "                \"loss_delta\": loss_delta,\n"
        "                \"loss_slope_per_100_steps\": slope_per_100,\n"
        "            }\n"
        "        Path(\"phase_diagnostics.json\").write_text(json.dumps(summary, indent=2) + \"\\n\", encoding=\"utf-8\")\n"
        "        def _fmt(v: object) -> str:\n"
        "            return \"?\" if v is None else f\"{float(v):.4f}\"\n"
        "        log0(\n"
        "            \"phase_diagnostics:written \"\n"
        "            f\"early_delta:{_fmt(summary['buckets']['early']['loss_delta'])} \"\n"
        "            f\"mid_delta:{_fmt(summary['buckets']['mid']['loss_delta'])} \"\n"
        "            f\"late_delta:{_fmt(summary['buckets']['late']['loss_delta'])} \"\n"
        "            f\"post_quant:{_fmt(final_post_quant_bpb)} path:phase_diagnostics.json\"\n"
        "        )\n",
        name,
    )
    source = _replace_unique(
        source,
        "            log0(\n"
        "                f\"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} \"\n"
        "                f\"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms\"\n"
        "            )\n"
        "            torch.cuda.synchronize()\n",
        "            log0(\n"
        "                f\"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} \"\n"
        "                f\"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms\"\n"
        "            )\n"
        "            phase_diag_attach_val(step, float(val_loss), float(val_bpb))\n"
        "            torch.cuda.synchronize()\n",
        name,
    )
    source = _replace_unique(
        source,
        "        step += 1\n"
        "        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n",
        "        step += 1\n"
        "        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n"
        "        phase_diag_update(step, float(train_loss.item()), float(approx_training_time_ms / max(step, 1)))\n",
        name,
    )
    final_exact_lines = [
        (
            "    log0(f\"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}\")\n",
            "    finalize_phase_diagnostics(float(q_val_bpb))\n",
        ),
        (
            "        log0(f\"final_int8_zlib_roundtrip_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}\")\n",
            "        finalize_phase_diagnostics(float(sw_val_bpb))\n",
        ),
        (
            "        log0(f\"final_int8_zlib_roundtrip_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}\")\n",
            "        finalize_phase_diagnostics(float(sw64_val_bpb))\n",
        ),
        (
            "        log0(f\"final_int8_zlib_roundtrip_exact val_loss:{sv_loss:.8f} val_bpb:{sv_bpb:.8f}\")\n",
            "        finalize_phase_diagnostics(float(sv_bpb))\n",
        ),
    ]
    replacement_count = 0
    for old, tail in final_exact_lines:
        if old in source:
            source = source.replace(old, old + tail, 1)
            replacement_count += 1
    if replacement_count == 0:
        raise ValueError(f"Patch '{name}': target string not found")
    return source


def patch_bootstrap_handoff(source: str) -> str:
    name = "bootstrap_handoff"
    source = _replace_unique(
        source,
        "    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None\n",
        "    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None\n"
        "    bootstrap_handoff_enabled = os.environ.get(\"BOOTSTRAP_HANDOFF_ENABLE\", \"0\") == \"1\" and getattr(base_model, \"count_bigram\", None) is not None\n"
        "    bootstrap_handoff_mode = os.environ.get(\"BOOTSTRAP_HANDOFF_MODE\", \"fade\")\n"
        "    bootstrap_handoff_start_frac = float(os.environ.get(\"BOOTSTRAP_HANDOFF_START_FRAC\", \"0.35\"))\n"
        "    bootstrap_handoff_end_frac = float(os.environ.get(\"BOOTSTRAP_HANDOFF_END_FRAC\", \"0.65\"))\n"
        "    bootstrap_handoff_min_scale = float(os.environ.get(\"BOOTSTRAP_HANDOFF_MIN_SCALE\", \"0.0\"))\n"
        "    bootstrap_handoff_plateau_window = int(os.environ.get(\"BOOTSTRAP_HANDOFF_PLATEAU_WINDOW\", \"20\"))\n"
        "    bootstrap_handoff_plateau_eps = float(os.environ.get(\"BOOTSTRAP_HANDOFF_PLATEAU_EPS\", \"0.01\"))\n"
        "    bootstrap_late_receiver = os.environ.get(\"BOOTSTRAP_LATE_RECEIVER\", \"none\")\n"
        "    bootstrap_handoff_triggered = False\n"
        "    bootstrap_handoff_trigger_step = int(bootstrap_handoff_start_frac * args.iterations)\n"
        "    bootstrap_handoff_frozen = False\n"
        "    bootstrap_recent_losses: list[float] = []\n"
        "\n"
        "    def set_count_bigram_trainable(trainable: bool) -> None:\n"
        "        if getattr(base_model, \"count_bigram\", None) is None:\n"
        "            return\n"
        "        for param in base_model.count_bigram.parameters():\n"
        "            param.requires_grad_(trainable)\n"
        "\n"
        "    def maybe_trigger_bootstrap_handoff(current_step: int) -> None:\n"
        "        nonlocal bootstrap_handoff_triggered, bootstrap_handoff_trigger_step\n"
        "        if not bootstrap_handoff_enabled or bootstrap_handoff_triggered:\n"
        "            return\n"
        "        progress = current_step / max(args.iterations, 1)\n"
        "        if progress < bootstrap_handoff_start_frac:\n"
        "            return\n"
        "        if bootstrap_handoff_mode.startswith(\"plateau_\"):\n"
        "            if len(bootstrap_recent_losses) < max(bootstrap_handoff_plateau_window, 2):\n"
        "                return\n"
        "            recent = bootstrap_recent_losses[-bootstrap_handoff_plateau_window:]\n"
        "            if recent[0] - recent[-1] > bootstrap_handoff_plateau_eps:\n"
        "                return\n"
        "        bootstrap_handoff_triggered = True\n"
        "        bootstrap_handoff_trigger_step = current_step\n"
        "        if \"record_phase_event\" in locals():\n"
        "            record_phase_event(\n"
        "                \"bootstrap_handoff_trigger\",\n"
        "                current_step,\n"
        "                {\n"
        "                    \"mode\": bootstrap_handoff_mode,\n"
        "                    \"receiver\": bootstrap_late_receiver,\n"
        "                    \"recent_improvement\": None if len(bootstrap_recent_losses) < 2 else float(bootstrap_recent_losses[0] - bootstrap_recent_losses[-1]),\n"
        "                },\n"
        "            )\n"
        "\n"
        "    def apply_bootstrap_handoff(current_step: int) -> None:\n"
        "        nonlocal bootstrap_handoff_frozen, bootstrap_handoff_triggered, bootstrap_handoff_trigger_step\n"
        "        if not bootstrap_handoff_enabled or getattr(base_model, \"count_bigram\", None) is None:\n"
        "            return\n"
        "        if bootstrap_handoff_mode in {\"fade\", \"freeze\", \"zero\"} and not bootstrap_handoff_triggered and current_step / max(args.iterations, 1) >= bootstrap_handoff_start_frac:\n"
        "            bootstrap_handoff_triggered = True\n"
        "            bootstrap_handoff_trigger_step = current_step\n"
        "            if \"record_phase_event\" in locals():\n"
        "                record_phase_event(\n"
        "                    \"bootstrap_handoff_trigger\",\n"
        "                    current_step,\n"
        "                    {\"mode\": bootstrap_handoff_mode, \"receiver\": bootstrap_late_receiver},\n"
        "                )\n"
        "        maybe_trigger_bootstrap_handoff(current_step)\n"
        "        if bootstrap_handoff_mode in {\"fade\", \"plateau_fade\", \"zero\", \"plateau_zero\"}:\n"
        "            ramp_start = bootstrap_handoff_trigger_step / max(args.iterations, 1)\n"
        "            if not bootstrap_handoff_triggered:\n"
        "                target_scale = 1.0\n"
        "            else:\n"
        "                floor = bootstrap_handoff_min_scale if \"fade\" in bootstrap_handoff_mode else 0.0\n"
        "                denom = max(bootstrap_handoff_end_frac - ramp_start, 1e-6)\n"
        "                ramp = min(max((current_step / max(args.iterations, 1) - ramp_start) / denom, 0.0), 1.0)\n"
        "                target_scale = 1.0 - (1.0 - floor) * ramp\n"
        "            base_model.count_bigram.scale.data.fill_(float(target_scale))\n"
        "            if bootstrap_handoff_triggered and target_scale <= bootstrap_handoff_min_scale + 1e-6 and not bootstrap_handoff_frozen:\n"
        "                set_count_bigram_trainable(False)\n"
        "                bootstrap_handoff_frozen = True\n"
        "                if \"record_phase_event\" in locals():\n"
        "                    record_phase_event(\n"
        "                        \"bootstrap_handoff_freeze\",\n"
        "                        current_step,\n"
        "                        {\"mode\": bootstrap_handoff_mode, \"scale\": float(target_scale), \"receiver\": bootstrap_late_receiver},\n"
        "                    )\n"
        "        elif bootstrap_handoff_mode in {\"freeze\", \"plateau_freeze\"} and bootstrap_handoff_triggered and not bootstrap_handoff_frozen:\n"
        "            set_count_bigram_trainable(False)\n"
        "            bootstrap_handoff_frozen = True\n"
        "            if \"record_phase_event\" in locals():\n"
        "                record_phase_event(\n"
        "                    \"bootstrap_handoff_freeze\",\n"
        "                    current_step,\n"
        "                    {\"mode\": bootstrap_handoff_mode, \"scale\": float(base_model.count_bigram.scale.detach().item()), \"receiver\": bootstrap_late_receiver},\n"
        "                )\n",
        name,
    )
    source = _replace_unique(
        source,
        "        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n"
        "        scale = lr_mul(step, elapsed_ms)\n"
        "        zero_grad_all()\n",
        "        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)\n"
        "        scale = lr_mul(step, elapsed_ms)\n"
        "        apply_bootstrap_handoff(step)\n"
        "        zero_grad_all()\n",
        name,
    )
    source = _replace_unique(
        source,
        "        train_loss /= grad_accum_steps\n",
        "        train_loss /= grad_accum_steps\n"
        "        bootstrap_recent_losses.append(float(train_loss.item()))\n"
        "        if len(bootstrap_recent_losses) > max(bootstrap_handoff_plateau_window * 2, 64):\n"
        "            del bootstrap_recent_losses[0 : len(bootstrap_recent_losses) - max(bootstrap_handoff_plateau_window * 2, 64)]\n",
        name,
    )
    return source


PATCH_REGISTRY = {
    "bootstrap_handoff": patch_bootstrap_handoff,
    "checkpoint_selection": patch_checkpoint_selection,
    "countinit_bigram": patch_countinit_bigram,
    "late_qat_active": patch_late_qat_active,
    "phase_diagnostics": patch_phase_diagnostics,
    "pre_quant_ttt": patch_pre_quant_ttt,
}

PATCH_ORDER = {
    "countinit_bigram": 10,
    "phase_diagnostics": 20,
    "bootstrap_handoff": 30,
    "checkpoint_selection": 90,
    "late_qat_active": 95,
    "pre_quant_ttt": 100,
}


def list_patches() -> list[str]:
    return sorted(PATCH_REGISTRY)


def apply_patches(source: str, patch_names: list[str]) -> str:
    unknown = [name for name in patch_names if name not in PATCH_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown hm2 patch(es): {', '.join(sorted(unknown))}")
    ordered = sorted(dict.fromkeys(patch_names), key=lambda n: (PATCH_ORDER.get(n, 1000), n))
    for patch_name in ordered:
        source = PATCH_REGISTRY[patch_name](source)
    return source
