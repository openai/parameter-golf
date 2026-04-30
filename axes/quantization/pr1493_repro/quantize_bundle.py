"""Apply PR-1493's GPTQ quantization to a saved bundle and evaluate.

Reproduces the merged-SOTA (PR-1493) quantization path on saved EMA weights
WITHOUT retraining. Runs pre-quant and post-quant eval (standard + sliding window).
No TTT, no ETLB.

Usage:
    cd axes/quantization/pr1493_repro
    BUNDLE_DIR=./local_bundle_seed42 \
    DATA_DIR=/path/to/parameter-golf/data \
    SEED=42 \
    torchrun --standalone --nproc_per_node=1 quantize_bundle.py

Prints:
    bundle_pre_quant_reference val_loss:... val_bpb:... eval_time:...ms
    bundle_pre_quant_sliding   val_loss:... val_bpb:... eval_time:...ms
    quant_artifact_brotli: <bytes>
    quantized_reference val_loss:... val_bpb:... eval_time:...ms
    quantized_sliding   val_loss:... val_bpb:... eval_time:...ms

The `quantized_sliding` BPB is the number to compare against PR-1493's reported
Sliding BPP of 1.0827. The `_reference` lines are the faster non-sliding eval.
"""

import io
import os
import sys
import time
from pathlib import Path

# Set defaults BEFORE importing train_save_bundle. Hyperparameters reads env at class
# definition time.
os.environ.setdefault("BUNDLE_DIR", "bundle")
os.environ.setdefault("VAL_LOSS_EVERY", "0")
os.environ.setdefault("TTT_ENABLED", "0")
os.environ.setdefault("ETLB_ENABLED", "0")
os.environ.setdefault("SLIDING_WINDOW_ENABLED", "0")  # skip sliding by default for fast iteration

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_save_bundle as tsb


def main():
    h = tsb.Hyperparameters()
    tsb.set_logging_hparams(h)
    # train_save_bundle.log() writes to h.logfile = f'logs/{run_id}.txt'; ensure dir exists.
    os.makedirs(os.path.dirname(h.logfile) or ".", exist_ok=True)
    log = tsb.log

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )
    # When flash_attn_3 is available, restrict to flash_sdp (PR-1493 default).
    # Otherwise enable all backends so the SDPA fallback can dispatch.
    _has_flash = hasattr(tsb, "_FLASH_ATTN_AVAILABLE") or "flash_attn_interface" in sys.modules
    enable_cudnn_sdp(False)
    enable_flash_sdp(not _has_flash)  # only use PyTorch built-in flash if no flash_attn_3
    enable_mem_efficient_sdp(not _has_flash)
    enable_math_sdp(not _has_flash)
    torch._dynamo.config.optimize_ddp = False

    bundle_dir = Path(h.bundle_dir)
    log(f"loading bundle from {bundle_dir.resolve()}")
    ema_weights = torch.load(bundle_dir / "ema_weights.pt", map_location="cpu")
    hessians = torch.load(bundle_dir / "hessians.pt", map_location="cpu")
    template_sd = torch.load(bundle_dir / "template_sd.pt", map_location="cpu")
    log(f"loaded: {len(ema_weights)} weights, {len(hessians)} hessians")

    # Build eval model and load EMA weights
    eval_model = tsb.GPT(h).to(device).bfloat16()
    tsb.restore_fp32_params(eval_model)
    current = eval_model.state_dict()
    load_sd = {k: v.to(current[k].dtype) for k, v in ema_weights.items() if k in current}
    missing = set(current) - set(load_sd)
    if missing:
        log(f"warning: {len(missing)} keys missing from EMA weights: {sorted(missing)[:5]}...")
    eval_model.load_state_dict(load_sd, strict=False)
    if h.num_loops > 0:
        eval_model.looping_active = True

    val_data = tsb.ValidationData(h, device)

    # Pre-quant eval (reference ceiling).
    # IMPORTANT: torch.compile is REQUIRED — the model gives wrong results in eager mode.
    # This is a known issue with this architecture (likely CastedLinear + autocast interaction).
    eval_model.eval()
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    tsb.timed_eval("bundle_pre_quant_reference", tsb.eval_val, h, device, val_data, compiled_model)
    if h.sliding_window_enabled:
        tsb.timed_eval(
            "bundle_pre_quant_sliding", tsb.eval_val_sliding, h, device, val_data, eval_model
        )
    torch._dynamo.reset()

    # Optional: post-hoc pruning before GPTQ
    prune_fraction = float(os.environ.get("PRUNE_FRACTION", "0.0"))
    prune_method = os.environ.get("PRUNE_METHOD", "magnitude")  # 'magnitude' or 'hessian'
    if prune_fraction > 0:
        log(f"pruning: method={prune_method} fraction={prune_fraction:.0%}")
        pruned_count = 0
        total_count = 0
        with torch.no_grad():
            for name, param in eval_model.named_parameters():
                if param.ndim == 2 and param.numel() > 65536 and "tok_emb" not in name:
                    w = param.data.float()
                    if prune_method == "hessian" and name + ".weight" in hessians:
                        h_key = name + ".weight"
                    elif prune_method == "hessian" and name in hessians:
                        h_key = name
                    else:
                        h_key = None

                    if prune_method == "hessian" and h_key is not None:
                        # importance = |w| * sqrt(H_diag_col)
                        H_diag = hessians[h_key].float().diag().to(w.device)
                        col_sens = H_diag.sqrt().clamp_min(1e-10)
                        importance = w.abs() * col_sens.unsqueeze(0)  # (rows, cols)
                    elif prune_method == "large":
                        # INVERSE: prune the LARGEST weights (counterintuitive control)
                        importance = -w.abs()  # negative so largest get lowest "importance"
                    elif prune_method == "random":
                        # Random pruning (control baseline)
                        importance = torch.rand_like(w)
                    else:
                        importance = w.abs()

                    threshold = torch.quantile(importance.flatten(), prune_fraction)
                    mask = importance >= threshold
                    pruned_count += (~mask).sum().item()
                    total_count += param.numel()
                    param.data *= mask
        log(f"pruning: zeroed {pruned_count:,} / {total_count:,} values ({pruned_count/total_count:.1%})")

    # Quantize via PR-1493's gptq_mixed_quantize
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    log("gptq: applying PR-1493 mixed quantize (int6 matrices k=12.85, int8 tok_emb k=20.0)...")
    t0 = time.perf_counter()
    quant_result, quant_meta = tsb.gptq_mixed_quantize(sd_cpu, hessians, h)
    log(f"gptq: done in {time.perf_counter() - t0:.1f}s")

    # Serialize + compress — verify artifact size
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = tsb._compress(quant_raw, h.compressor)
    artifact_bytes = len(quant_blob)
    log(f"quant_artifact_{h.compressor}: {artifact_bytes} bytes ({artifact_bytes / 1e6:.3f} MB)")

    # Dequantize and load into model
    quant_state = torch.load(
        io.BytesIO(tsb._decompress(quant_blob, h.compressor)),
        map_location="cpu",
    )
    dequant_sd = tsb.dequantize_mixed(quant_state["w"], quant_state["m"], template_sd)
    current = eval_model.state_dict()
    restored = {k: v.to(current[k].dtype) for k, v in dequant_sd.items() if k in current}
    eval_model.load_state_dict(restored, strict=False)
    if h.num_loops > 0:
        eval_model.looping_active = True

    # Post-quant eval
    eval_model.eval()
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    tsb.timed_eval("quantized_reference", tsb.eval_val, h, device, val_data, compiled_model)
    if h.sliding_window_enabled:
        tsb.timed_eval("quantized_sliding", tsb.eval_val_sliding, h, device, val_data, eval_model)

    log("done: quantize + eval complete.")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
