"""
Standalone eval-only script for sliding window + phased LoRA TTT evaluation.

Usage:
    torchrun --standalone --nproc_per_node=4 eval_only.py

Loads a quantized model from final_model.int6.ptz and runs:
  1. Sliding window eval (eval_val_sliding)
  2. Phased LoRA TTT eval (eval_val_ttt_phased) if TTT_ENABLED=1

All classes/functions are imported from train_gpt_readable.py via exec().
No training code is executed.
"""

import math, os, random, sys, time
import numpy as np
import torch
import torch.distributed as dist

# Force EVAL_ONLY so train_and_eval() would skip training if called directly
os.environ.setdefault("EVAL_ONLY", "1")

# ---- Load all definitions from train_gpt_readable.py via exec ----
_script_dir = os.path.dirname(os.path.abspath(__file__))
_train_script = os.path.join(_script_dir, "train_gpt_readable.py")
_ns = {"__name__": "_train_gpt_readable_imported", "__file__": _train_script}
with open(_train_script, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _train_script, "exec"), _ns)

# Pull out everything we need
Hyperparameters = _ns["Hyperparameters"]
ValidationData = _ns["ValidationData"]
GPT = _ns["GPT"]
deserialize = _ns["deserialize"]
eval_val = _ns["eval_val"]
eval_val_sliding = _ns["eval_val_sliding"]
eval_val_ttt_phased = _ns["eval_val_ttt_phased"]
BatchedTTTLoRA = _ns["BatchedTTTLoRA"]
BatchedLinearLoRA = _ns["BatchedLinearLoRA"]
set_logging_hparams = _ns["set_logging_hparams"]
log = _ns["log"]
timed_eval = _ns["timed_eval"]
restore_fp32_params = _ns["restore_fp32_params"]


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError("bad world_size")
    if 8 % world_size != 0:
        raise ValueError("world_size must divide 8")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp,
        enable_math_sdp, enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(True)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False

    # ---- Hyperparameters ----
    h = Hyperparameters()
    set_logging_hparams(h)

    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log("=" * 60)
        log("eval_only.py — standalone evaluation")
        log("=" * 60)
        log("Hyperparameters:")
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}")
        log("=" * 60)

    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)

    # ---- Validation data ----
    val_data = ValidationData(h, device)
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    # ---- Deserialize quantized model ----
    if distributed:
        dist.barrier()
    eval_model = deserialize(h, device)

    # Enable looping if the model has depth recurrence indices
    if len(eval_model.encoder_indices) != eval_model.num_encoder_layers:
        eval_model.looping_active = True
        log(f"looping_active=True encoder:{eval_model.encoder_indices} decoder:{eval_model.decoder_indices}")

    # ---- Standard quantized eval (non-sliding) ----
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval("quantized", eval_val, h, device, val_data, compiled_model)

    # ---- Sliding window eval ----
    if h.sliding_window_enabled:
        timed_eval("quantized_sliding_window", eval_val_sliding, h, device, val_data, eval_model)

    # ---- Phased LoRA TTT eval ----
    if h.ttt_enabled and h.ttt_lora_rank > 0:
        ttt_model = deserialize(h, device)
        if len(ttt_model.encoder_indices) != ttt_model.num_encoder_layers:
            ttt_model.looping_active = True

        # Warm up rotary caches for TTT eval seq len
        for block in ttt_model.blocks:
            block.attn.rotary._cos_cached = None
            block.attn.rotary._sin_cached = None
            block.attn.rotary._seq_len_cached = 0
            block.attn.rotary(h.ttt_eval_seq_len, device, torch.bfloat16)

        # Build compiled forward_ttt function
        def _fwd_ttt_inner(input_ids, target_ids, lora):
            return ttt_model.forward_ttt(input_ids, target_ids, lora=lora)

        _fwd_ttt_compiled_inner = None

        def _fwd_ttt(input_ids, target_ids, lora):
            nonlocal _fwd_ttt_compiled_inner
            if _fwd_ttt_compiled_inner is None:
                _fwd_ttt_compiled_inner = torch.compile(_fwd_ttt_inner, dynamic=True)
            return _fwd_ttt_compiled_inner(input_ids, target_ids, lora=lora)

        fwd_ttt_compiled = _fwd_ttt

        # Compile warmup
        log("ttt_lora:warming up compile")
        t_warmup = time.perf_counter()
        for bsz_w in [h.ttt_batch_size]:
            wl = BatchedTTTLoRA(
                bsz_w, ttt_model, h.ttt_lora_rank,
                k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
            ).to(device)
            wo = torch.optim.AdamW(
                wl.parameters(), lr=h.ttt_lora_lr,
                betas=(h.ttt_beta1, h.ttt_beta2), eps=1e-10,
                weight_decay=h.ttt_weight_decay, fused=True,
            )
            for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
                xw = torch.randint(0, h.vocab_size, (bsz_w, ctx_len), device=device, dtype=torch.int64)
                yw = torch.randint(0, h.vocab_size, (bsz_w, ctx_len), device=device, dtype=torch.int64)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = fwd_ttt_compiled(xw, yw, lora=wl)
                ptl[:, :min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
                wo.step()
                wo.zero_grad(set_to_none=True)
            del wl, wo
        torch.cuda.empty_cache()
        log(f"ttt_lora:compile warmup done ({time.perf_counter() - t_warmup:.1f}s)")
        log(f"ttt_lora_alpha: {BatchedLinearLoRA._ALPHA}")
        log(f"ttt_warm_start_a: {BatchedLinearLoRA._WARM_START_A}")

        # Run TTT eval
        log("beginning TTT eval timer")
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb = eval_val_ttt_phased(
            h, ttt_model, device, val_data, forward_ttt_train=fwd_ttt_compiled,
        )
        torch.cuda.synchronize()
        ttt_eval_elapsed = time.perf_counter() - t_ttt
        log(
            f"quantized_ttt_phased val_loss:{ttt_val_loss:.8f} "
            f"val_bpb:{ttt_val_bpb:.8f} eval_time:{ttt_eval_elapsed * 1e3:.0f}ms"
        )

    log("eval_only.py done")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
