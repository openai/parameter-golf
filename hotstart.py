"""
Hotstart utilities for parameter golf.

Load a checkpoint and either:
  1. Resume training from that point
  2. Re-run EMA with different decay
  3. Re-quantize with different params
  4. Re-eval with different TTT/sliding window params

Usage:
  # Resume training from step 25 checkpoint
  python hotstart.py resume --ckpt /workspace/runs/test/checkpoints/ckpt_event_step25.pt

  # Re-quantize a pre-EMA checkpoint with different clip_sigmas
  python hotstart.py requant --ckpt /workspace/runs/test/checkpoints/ckpt_final_pre_ema_step50.pt --clip_sigmas 10.0

  # Re-eval a quantized model with TTT enabled
  python hotstart.py reeval --model final_model.int6.ptz --ttt

  # Re-run EMA with different decay on pre-EMA checkpoint
  python hotstart.py reema --ckpt /workspace/runs/test/checkpoints/ckpt_final_pre_ema_step50.pt --ema_decay 0.998
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import random
from pathlib import Path

# Import everything from the SOTA code
sys.path.insert(0, os.path.dirname(__file__))
from train_gpt_sota import (
    Hyperparameters,
    GPT,
    Optimizers,
    ValidationData,
    ShuffledSequenceLoader,
    set_logging_hparams,
    log,
    restore_fp32_params,
    eval_val,
    eval_val_sliding,
    eval_val_ttt,
    timed_eval,
    serialize,
    deserialize,
    collect_hessians,
    gptq_mixed_quantize,
    dequantize_mixed,
    _compress,
)


def load_checkpoint(ckpt_path, device):
    """Load a training checkpoint (model + optimizer + EMA state)."""
    log(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    log(f"  Step: {ckpt['step']}")
    log(f"  Keys: {list(ckpt.keys())}")
    return ckpt


def cmd_resume(args):
    """Resume training from a checkpoint."""
    h = Hyperparameters()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    set_logging_hparams(h)

    ckpt = load_checkpoint(args.ckpt, device)
    start_step = ckpt["step"]

    # Build model and load weights
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    base_model.load_state_dict(ckpt["model_state_dict"])

    # Build optimizers and load state
    optimizers = Optimizers(h, base_model)
    for opt, state in zip(optimizers, ckpt["optimizer_states"]):
        opt.load_state_dict(state)

    # Restore EMA state
    ema_state = {}
    if ckpt.get("ema_state"):
        ema_state = {k: v.to(device) for k, v in ckpt["ema_state"].items()}
    else:
        ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}

    # Check if recurrence should be active at this step
    frac = start_step / max(h.iterations, 1)
    if h.num_loops > 0 and frac >= h.enable_looping_at:
        base_model.looping_active = True
        log(f"Recurrence active (frac={frac:.3f} >= {h.enable_looping_at})")

    # Compile
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    val_data = ValidationData(h, device)
    train_loader = ShuffledSequenceLoader(h, device)

    # Resume training loop
    target_steps = args.steps or h.iterations
    log(f"Resuming from step {start_step} → {target_steps}")

    def training_frac(step, elapsed_ms):
        return step / max(h.iterations, 1)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    step = start_step
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while step < target_steps:
        frac = step / max(h.iterations, 1)
        scale = lr_mul(frac)

        if h.num_loops > 0 and not base_model.looping_active and frac >= h.enable_looping_at:
            base_model.looping_active = True
            log(f"Recurrence activated at step {step}")

        # Training step
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = compiled_model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps

        mom_frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - mom_frac) * h.muon_momentum_warmup_start + mom_frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        for opt in optimizers:
            opt.step()

        # EMA update
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(h.ema_decay).add_(t.detach().float(), alpha=1.0 - h.ema_decay)

        step += 1
        if step % 10 == 0 or step == target_steps:
            elapsed = time.perf_counter() - t0
            log(f"step:{step}/{target_steps} train_loss:{train_loss.item():.4f} time:{elapsed:.1f}s")

    # Eval
    val_loss, val_bpb = eval_val(h, device, val_data, compiled_model)
    log(f"Final (pre-EMA): val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")

    # Apply EMA
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)

    val_loss, val_bpb = eval_val(h, device, val_data, compiled_model)
    log(f"Final (post-EMA): val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")


def cmd_requant(args):
    """Re-quantize a checkpoint with different params."""
    h = Hyperparameters()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    set_logging_hparams(h)

    ckpt = load_checkpoint(args.ckpt, device)

    # Build model and load weights
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    base_model.load_state_dict(ckpt["model_state_dict"])

    # Apply EMA if checkpoint has it
    if ckpt.get("ema_state"):
        log("Applying EMA weights from checkpoint")
        current_state = base_model.state_dict()
        avg_state = {name: t.to(device=device, dtype=current_state[name].dtype) for name, t in ckpt["ema_state"].items()}
        base_model.load_state_dict(avg_state, strict=True)

    # Override quantization params
    if args.clip_sigmas is not None:
        h.matrix_clip_sigmas = args.clip_sigmas
        log(f"Using clip_sigmas={args.clip_sigmas}")
    if args.embed_clip_sigmas is not None:
        h.embed_clip_sigmas = args.embed_clip_sigmas
        log(f"Using embed_clip_sigmas={args.embed_clip_sigmas}")
    if args.calibration_batches is not None:
        h.gptq_calibration_batches = args.calibration_batches
        log(f"Using calibration_batches={args.calibration_batches}")

    # Quantize
    code = Path(os.path.join(os.path.dirname(__file__), "train_gpt_sota.py")).read_text(encoding="utf-8")
    serialize(h, base_model, code)

    # Eval quantized
    val_data = ValidationData(h, device)
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval("requant", eval_val, h, device, val_data, compiled_model)

    if h.sliding_window_enabled:
        timed_eval("requant_sliding", eval_val_sliding, h, device, val_data, eval_model)


def cmd_reeval(args):
    """Re-eval a quantized model with different TTT/eval params."""
    h = Hyperparameters()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    set_logging_hparams(h)

    # Override eval params
    if args.ttt_lr is not None:
        h.ttt_lr = args.ttt_lr
    if args.ttt_epochs is not None:
        h.ttt_epochs = args.ttt_epochs
    if args.ttt_chunk is not None:
        h.ttt_chunk_tokens = args.ttt_chunk

    val_data = ValidationData(h, device)

    model_path = args.model or h.quantized_model_path
    log(f"Loading quantized model: {model_path}")
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True

    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval("reeval", eval_val, h, device, val_data, compiled_model)

    if h.sliding_window_enabled:
        timed_eval("reeval_sliding", eval_val_sliding, h, device, val_data, eval_model)

    if args.ttt:
        h.ttt_enabled = True
        del eval_model, compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        timed_eval("reeval_ttt", eval_val_ttt, h, device, val_data, ttt_model)


def cmd_reema(args):
    """Re-run EMA with different decay on a pre-EMA checkpoint, then quantize and eval."""
    h = Hyperparameters()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    set_logging_hparams(h)

    ckpt = load_checkpoint(args.ckpt, device)

    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    base_model.load_state_dict(ckpt["model_state_dict"])

    if not ckpt.get("ema_state"):
        log("ERROR: Checkpoint has no EMA state to re-weight")
        return

    # The EMA state in the checkpoint was computed with the original decay.
    # We can't retroactively change the decay — we'd need the full training history.
    # But we CAN blend the raw weights with the EMA weights at a different ratio.
    ema_decay = args.ema_decay
    log(f"Blending raw weights with EMA at ratio {ema_decay}")

    current_state = base_model.state_dict()
    ema_state = {k: v.to(device) for k, v in ckpt["ema_state"].items()}

    # Blend: new = decay * ema + (1-decay) * raw
    blended = {}
    for name in current_state:
        raw = current_state[name].float()
        ema = ema_state[name].float()
        blended[name] = (ema_decay * ema + (1 - ema_decay) * raw).to(dtype=current_state[name].dtype)
    base_model.load_state_dict(blended, strict=True)

    # Eval
    val_data = ValidationData(h, device)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    timed_eval(f"reema_{ema_decay}", eval_val, h, device, val_data, compiled_model)

    # Quantize and eval
    code = Path(os.path.join(os.path.dirname(__file__), "train_gpt_sota.py")).read_text(encoding="utf-8")
    serialize(h, base_model, code)
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval(f"reema_{ema_decay}_quantized", eval_val, h, device, val_data, compiled_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotstart utilities for parameter golf")
    sub = parser.add_subparsers(dest="cmd")

    p_resume = sub.add_parser("resume", help="Resume training from checkpoint")
    p_resume.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    p_resume.add_argument("--steps", type=int, help="Target step count (default: h.iterations)")

    p_requant = sub.add_parser("requant", help="Re-quantize with different params")
    p_requant.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    p_requant.add_argument("--clip_sigmas", type=float, help="Matrix clip sigmas")
    p_requant.add_argument("--embed_clip_sigmas", type=float, help="Embedding clip sigmas")
    p_requant.add_argument("--calibration_batches", type=int, help="GPTQ calibration batches")

    p_reeval = sub.add_parser("reeval", help="Re-eval quantized model")
    p_reeval.add_argument("--model", help="Path to .int6.ptz file")
    p_reeval.add_argument("--ttt", action="store_true", help="Enable TTT")
    p_reeval.add_argument("--ttt_lr", type=float)
    p_reeval.add_argument("--ttt_epochs", type=int)
    p_reeval.add_argument("--ttt_chunk", type=int)

    p_reema = sub.add_parser("reema", help="Re-blend EMA weights")
    p_reema.add_argument("--ckpt", required=True, help="Path to pre-EMA checkpoint")
    p_reema.add_argument("--ema_decay", type=float, default=0.5, help="Blend ratio (1.0=full EMA, 0.0=raw)")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    {"resume": cmd_resume, "requant": cmd_requant, "reeval": cmd_reeval, "reema": cmd_reema}[args.cmd](args)
