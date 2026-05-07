"""
JEPA-NTP Training Script
=========================

Wraps the baseline Parameter Golf train_gpt.py with JEPA-style auxiliary losses.

This script:
  1. Subclasses the baseline GPT model to return hidden states from the forward pass
  2. Adds spectral floor and/or cosine-MSE losses as configured
  3. Uses torch.compile for full training speed (no hooks!)
  4. Logs JEPA-specific metrics (effective rank, spectrum, curvature) to WandB

Key design: Hidden states are captured by returning them from forward() rather
than using hooks. This keeps the full compute graph traceable by torch.compile.
When capture_layers=None, the branches are dead-code-eliminated and you get
the exact same compiled graph as the baseline.

Usage:
    # Single experiment
    EXPERIMENT=exp3_combined RUN_ID=exp3_run1 \\
        torchrun --standalone --nproc_per_node=1 jepa_ntp/train_jepa_ntp.py
    
    # All experiments
    for exp in baseline exp1_spectral exp2_cosine_mse exp3_combined exp4_targeted; do
        EXPERIMENT=$exp RUN_ID=${exp}_run1 \\
            torchrun --standalone --nproc_per_node=1 jepa_ntp/train_jepa_ntp.py
    done

    # Full 8xH100 competition run
    EXPERIMENT=exp3_combined RUN_ID=competition_run \\
        torchrun --standalone --nproc_per_node=8 jepa_ntp/train_jepa_ntp.py
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# --- Add parent directory to path so we can import train_gpt ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_gpt import (
    GPT,
    Hyperparameters,
    Muon,
    CastedLinear,
    DistributedTokenLoader,
    TokenStream,
    build_sentencepiece_luts,
    eval_val,
    load_validation_tokens,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    restore_low_dim_params_to_fp32,
    zeropower_via_newtonschulz5,
    CONTROL_TENSOR_NAME_PATTERNS,
)

from config import (
    JEPAConfig,
    EXPERIMENT_CONFIGS,
)
from losses import spectral_variance_floor, cosine_mse_loss
from losses.cosine_mse import LatentPredictor
from metrics import compute_effective_rank, compute_singular_spectrum, compute_latent_curvature
from metrics.latent_smoothness import compute_cosine_smoothness


# ============================================================
# JEPA-GPT: SUBCLASS WITH INLINE HIDDEN STATE CAPTURE
# ============================================================
# Instead of hooks (which break torch.compile), we override forward()
# to optionally return intermediate hidden states as part of the output.
# When capture_layers is None, the capture branches are all False and
# torch.compile dead-code-eliminates them — same graph as baseline.

class JEPAGPT(GPT):
    """
    GPT subclass that can return intermediate hidden states from forward().
    Fully compatible with torch.compile — no hooks, no Python callbacks.

    IMPORTANT: forward() always returns (loss, captured_dict). The baseline
    eval_val() expects model(x, y) to return a single loss tensor, so we
    provide an EvalWrapper for validation calls.
    """

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        capture_layers: set[int] | None = None,
    ) -> tuple[Tensor, dict[int, Tensor]]:
        """
        Forward pass with optional hidden state capture.

        Args:
            input_ids: (batch, seq_len)
            target_ids: (batch, seq_len)
            capture_layers: Set of layer indices to capture. None = no capture.

        Returns:
            (ce_loss, captured) where captured is {layer_idx: (batch, seq, dim)}
            When capture_layers is None, captured is empty dict.
        """
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        captured: dict[int, Tensor] = {}

        # First half: encoder layers (store skips)
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
            if capture_layers is not None and i in capture_layers:
                captured[i] = x

        # Second half: decoder layers (consume skips in reverse)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            layer_idx = self.num_encoder_layers + i
            x = self.blocks[layer_idx](x, x0)
            if capture_layers is not None and layer_idx in capture_layers:
                captured[layer_idx] = x

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        return loss, captured


class EvalWrapper(nn.Module):
    """
    Thin wrapper that makes JEPAGPT forward() compatible with eval_val().
    eval_val() calls model(x, y).detach() — it expects a single tensor.
    This wrapper unpacks the (loss, captured) tuple and returns just the loss.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        loss, _captured = self.model(input_ids, target_ids)
        return loss

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self


def resolve_target_layers(
    num_layers: int,
    config_layers: list[int] | None,
) -> list[int]:
    """Compute target layer indices. Default: middle layers (25-60% of depth)."""
    if config_layers is not None:
        return config_layers
    start = max(1, int(num_layers * 0.25))
    end = min(num_layers - 1, int(num_layers * 0.60))
    return list(range(start, end + 1))


# ============================================================
# JEPA-AUGMENTED FORWARD PASS
# ============================================================

def jepa_forward(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    jepa_cfg: JEPAConfig,
    capture_layer_set: set[int] | None,
    predictor: LatentPredictor | None,
    step: int,
    total_steps: int,
) -> tuple[Tensor, dict[str, float], dict[int, Tensor]]:
    """
    Run forward pass with JEPA auxiliary losses.
    
    Returns:
        (total_loss, metrics_dict, captured_hidden_states)
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        ce_loss, captured = model(x, y, capture_layers=capture_layer_set)

    total_loss = ce_loss
    metrics = {"loss/ce": float(ce_loss.item())}

    if not captured:
        metrics["loss/total"] = float(total_loss.item())
        return total_loss, metrics, captured

    # Average across captured layers: (batch, seq, dim)
    h = torch.stack(list(captured.values()), dim=0).mean(dim=0)

    # --- Spectral Variance Floor ---
    if jepa_cfg.use_spectral:
        spec_loss = spectral_variance_floor(
            h,
            eps_floor=jepa_cfg.spectral_eps,
            use_deltas=jepa_cfg.spectral_use_deltas,
        )
        total_loss = total_loss + jepa_cfg.lambda_spec * spec_loss
        metrics["loss/spectral"] = float(spec_loss.item())

    # --- Cosine-MSE Prediction ---
    if jepa_cfg.use_cosine_mse and predictor is not None:
        alpha = jepa_cfg.get_alpha(step, total_steps)
        predicted = predictor(h)
        target = h[:, 1:, :]  # actual h_{t+1}
        cmse_loss = cosine_mse_loss(predicted, target)
        total_loss = total_loss + alpha * cmse_loss
        metrics["loss/cosine_mse"] = float(cmse_loss.item())
        metrics["schedule/alpha"] = alpha

    metrics["loss/total"] = float(total_loss.item())
    return total_loss, metrics, captured


# ============================================================
# METRIC LOGGING
# ============================================================

def log_jepa_metrics(
    captured: dict[int, Tensor],
    jepa_cfg: JEPAConfig,
    wandb_run,
    step: int,
):
    """Compute and log JEPA-specific diagnostics."""
    if not captured:
        return

    h = torch.stack(list(captured.values()), dim=0).mean(dim=0).detach()

    metrics = {}

    # Effective rank (primary anti-collapse metric)
    eff_rank = compute_effective_rank(h, use_deltas=True)
    metrics["diagnostics/effective_rank_delta"] = eff_rank
    eff_rank_abs = compute_effective_rank(h, use_deltas=False)
    metrics["diagnostics/effective_rank_abs"] = eff_rank_abs

    # Latent path curvature
    curvature = compute_latent_curvature(h)
    metrics["diagnostics/curvature"] = curvature
    cosine_smooth = compute_cosine_smoothness(h)
    metrics["diagnostics/cosine_smoothness"] = cosine_smooth

    # Singular value spectrum
    if jepa_cfg.log_spectrum:
        sv = compute_singular_spectrum(h, use_deltas=True, top_k=jepa_cfg.spectrum_top_k)
        for i, v in enumerate(sv):
            metrics[f"spectrum/sv_{i:03d}"] = float(v)

    if wandb_run is not None:
        wandb_run.log(metrics, step=step)
    else:
        # Fallback: print key metrics
        print(
            f"  [JEPA] eff_rank_Δ={eff_rank:.1f} eff_rank_abs={eff_rank_abs:.1f} "
            f"curvature={curvature:.4f} cos_smooth={cosine_smooth:.4f}"
        )


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main() -> None:
    global zeropower_via_newtonschulz5

    import sentencepiece as spm
    import subprocess
    import random
    import glob as glob_mod
    import io
    import zlib
    from torch.nn.parallel import DistributedDataParallel as DDP

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    _compile_disabled = (
        os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1"
        or os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1"
    )
    if not _compile_disabled:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # --- Load experiment config ---
    exp_name = os.environ.get("EXPERIMENT", "baseline")
    if exp_name not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment '{exp_name}'. Choose from: {list(EXPERIMENT_CONFIGS.keys())}"
        )
    jepa_cfg = EXPERIMENT_CONFIGS[exp_name]()
    print(f"[JEPA-NTP] Experiment: {exp_name}")
    print(f"[JEPA-NTP] Config: spectral={jepa_cfg.use_spectral}, cosine_mse={jepa_cfg.use_cosine_mse}")
    if jepa_cfg.target_layers:
        print(f"[JEPA-NTP] Target layers: {jepa_cfg.target_layers}")

    # --- Distributed + CUDA setup (same as baseline) ---
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)

    # --- Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Tokenizer + Validation ---
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # --- Model (JEPAGPT subclass with inline capture) ---
    base_model = JEPAGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # --- Resolve target layers ---
    target_layers = resolve_target_layers(args.num_layers, jepa_cfg.target_layers)
    needs_capture = jepa_cfg.use_spectral or jepa_cfg.use_cosine_mse
    capture_layer_set: set[int] | None = set(target_layers) if needs_capture else None
    log0(f"[JEPA-NTP] Target layers: {target_layers}, capture={'enabled' if needs_capture else 'disabled'}")

    # Set up predictor MLP (if using cosine-MSE)
    predictor: LatentPredictor | None = None
    if jepa_cfg.use_cosine_mse:
        predictor = LatentPredictor(
            dim=args.model_dim,
            hidden_mult=jepa_cfg.predictor_hidden_mult,
            dropout=jepa_cfg.predictor_dropout,
        ).to(device).bfloat16()
        log0(f"[JEPA-NTP] Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    # --- DDP + Compile ---
    # DDP must wrap the model BEFORE torch.compile (PyTorch DDP+compile guidance).
    # Predictor also needs DDP wrapping to keep weights synchronized across ranks.
    if distributed:
        model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False)
        if predictor is not None:
            predictor = DDP(predictor, device_ids=[local_rank], broadcast_buffers=False)
    else:
        model = base_model

    if not _compile_disabled:
        log0("[JEPA-NTP] torch.compile enabled (fullgraph=False for DDP compatibility)")
        model = torch.compile(model, dynamic=False, fullgraph=False)
    else:
        log0("[JEPA-NTP] torch.compile DISABLED")

    # EvalWrapper for eval_val() compatibility — eval_val expects model(x,y)
    # to return a single loss tensor, but JEPAGPT returns (loss, captured).
    eval_model = EvalWrapper(model)

    # --- Optimizers (same split as baseline) ---
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    # Predictor optimizer (separate Adam, higher LR for faster convergence)
    if predictor is not None:
        optimizer_pred = torch.optim.Adam(
            predictor.parameters(), lr=args.scalar_lr * 2, betas=(args.beta1, args.beta2),
        )
        optimizers.append(optimizer_pred)

    n_params = sum(p.numel() for p in base_model.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters()) if predictor else 0
    log0(f"model_params:{n_params} predictor_params:{pred_params} total:{n_params + pred_params}")

    # --- WandB ---
    wandb_run = None
    if jepa_cfg.use_wandb and master_process:
        try:
            import wandb
            wandb_run = wandb.init(
                project=jepa_cfg.wandb_project,
                group=jepa_cfg.wandb_group,
                tags=jepa_cfg.wandb_tags,
                name=f"{exp_name}_{args.run_id}",
                config={
                    "experiment": exp_name,
                    "model_dim": args.model_dim,
                    "num_layers": args.num_layers,
                    "vocab_size": args.vocab_size,
                    "use_spectral": jepa_cfg.use_spectral,
                    "use_cosine_mse": jepa_cfg.use_cosine_mse,
                    "alpha": jepa_cfg.alpha,
                    "lambda_spec": jepa_cfg.lambda_spec,
                    "target_layers": target_layers,
                    "total_params": n_params + pred_params,
                },
            )
            log0(f"[JEPA-NTP] WandB run: {wandb_run.url}")
        except ImportError:
            log0("[JEPA-NTP] wandb not installed, logging to stdout only")

    # --- Data loader ---
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # --- Warmup (same as baseline, resets state after) ---
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_pred_state = {n: t.detach().cpu().clone() for n, t in predictor.state_dict().items()} if predictor else None
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        if predictor:
            predictor.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                    if predictor is not None and isinstance(predictor, DDP):
                        predictor.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                loss, _, _ = jepa_forward(
                    model, x, y, jepa_cfg, capture_layer_set, predictor,
                    warmup_step, args.iterations,
                )
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        # Reset everything
        base_model.load_state_dict(initial_model_state, strict=True)
        if predictor and initial_pred_state:
            predictor.load_state_dict(initial_pred_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
            if predictor is not None and isinstance(predictor, DDP):
                predictor.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        log0(f"[JEPA-NTP] Warmup complete ({args.warmup_steps} steps)")

    # --- Main training loop ---
    training_time_ms = 0.0
    total_tokens_seen: int = 0  # Track tokens for Chinchilla-style loss curves
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model.train()
    if predictor:
        predictor.train()

    step = 0
    last_captured: dict[int, Tensor] = {}  # Keep last captured states for metric logging

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        # --- Validation ---
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, eval_model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"tokens:{total_tokens_seen:,} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if wandb_run:
                wandb_run.log({
                    "val/loss": val_loss,
                    "val/bpb": val_bpb,
                    "throughput/tokens_seen": total_tokens_seen,
                }, step=step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        # --- Training step ---
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        step_metrics: dict[str, float] = {}

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                if predictor is not None and isinstance(predictor, DDP):
                    predictor.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)

            loss, metrics, captured = jepa_forward(
                model, x, y, jepa_cfg, capture_layer_set, predictor,
                step, args.iterations,
            )
            train_loss += loss.detach()
            (loss * grad_scale).backward()
            step_metrics = metrics  # keep last micro-step metrics
            last_captured = captured  # keep for diagnostic logging

        train_loss /= grad_accum_steps

        # Track cumulative tokens seen (the real currency for loss curve comparison)
        total_tokens_seen += args.train_batch_tokens

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        # LR schedule
        for opt in optimizers:
            for group in opt.param_groups:
                if "base_lr" in group:
                    group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
            if predictor:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1

        # --- Logging ---
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            tokens_per_sec = total_tokens_seen / (approx_training_time_ms / 1000.0) if approx_training_time_ms > 0 else 0.0
            parts = [f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f}"]
            for k, v in step_metrics.items():
                if k.startswith("loss/") and k != "loss/total":
                    parts.append(f"{k.split('/')[-1]}:{v:.4f}")
            parts.append(f"tokens:{total_tokens_seen:,} tok/s:{tokens_per_sec:,.0f}")
            parts.append(f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")
            log0(" ".join(parts))

            if wandb_run:
                step_metrics["throughput/tokens_seen"] = total_tokens_seen
                step_metrics["throughput/tokens_per_sec"] = tokens_per_sec
                wandb_run.log(step_metrics, step=step)

        # --- JEPA diagnostic metrics ---
        should_log_jepa = (
            jepa_cfg.log_metrics_every > 0
            and step % jepa_cfg.log_metrics_every == 0
            and needs_capture
        )
        if should_log_jepa:
            log_jepa_metrics(last_captured, jepa_cfg, wandb_run, step)

        # --- Wallclock cap ---
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # --- Serialization (same as baseline) ---
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_size = len(code.encode("utf-8"))
        # NOTE: Predictor weights are NOT included in submission size —
        # they are only used during training, not inference.
        log0(f"Serialized model: {model_bytes} bytes (code: {code_size})")
        log0(f"Total submission size: {model_bytes + code_size} bytes")
        if predictor:
            torch.save(predictor.state_dict(), "final_predictor.pt")
            log0(f"Predictor saved (NOT part of submission): {os.path.getsize('final_predictor.pt')} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_size = len(code.encode("utf-8"))
        log0(f"Int8+zlib model: {quant_file_bytes} bytes, total: {quant_file_bytes + code_size} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, eval_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )

    if wandb_run:
        wandb_run.log({
            "final/val_loss": q_val_loss,
            "final/val_bpb": q_val_bpb,
            "final/model_size_bytes": os.path.getsize("final_model.int8.ptz") if master_process else 0,
        })
        wandb_run.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
