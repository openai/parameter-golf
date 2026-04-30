"""Re-quantize a saved final_model.pt and evaluate q / q_sw / q_ttt.

Usage:
  CKPT_PATH=final_model.pt OUT_PTZ=requant.int6.ptz \
  GPTQ_DAMP=0.01 GPTQ_BLOCK_SIZE=128 GPTQ_CALIBRATION_BATCHES=64 \
  TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
  SEED=42 QK_GAIN_INIT=5.25 \
  WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 \
  RUN_ID=requant_test \
  ./safe_launch.sh torchrun --standalone --nproc_per_node=8 requant_eval.py

The training-time flags (WD/PAIRED) only set Hyperparameters fields — no training
runs. They must match the recipe used to produce final_model.pt so the model
graph matches the state_dict (paired-head Muon affects no graph topology, so
this is mostly defensive; iha would require IHA_ENABLED=1 too).
"""
import io, os, sys, time
from pathlib import Path
import torch, torch.distributed as dist

import train_pr1493 as T


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False

    h = T.Hyperparameters()
    T.set_logging_hparams(h)

    ckpt_path = os.environ.get("CKPT_PATH", "final_model.pt")
    out_ptz = os.environ.get("OUT_PTZ", "requant.int6.ptz")
    h.quantized_model_path = out_ptz
    h.model_path = ckpt_path  # so any "Serialized model" log refers to source

    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        T.log(100 * "=", console=False)
        T.log("requant_eval Hyperparameters:")
        for k in sorted(vars(type(h))):
            if not k.startswith("_"):
                T.log(f"  {k}: {getattr(type(h), k)}", console=True)
        T.log(f"  CKPT_PATH={ckpt_path}", console=True)
        T.log(f"  OUT_PTZ={out_ptz}", console=True)
        T.log(f"  GPTQ_DAMP={h.gptq_damp} GPTQ_BLOCK_SIZE={h.gptq_block_size}", console=True)
        T.log(f"  GPTQ_CALIBRATION_BATCHES={h.gptq_calibration_batches}", console=True)
        T.log("=" * 100, console=False)

    # Build the model, restore fp32 controls, load checkpoint.
    base_model = T.GPT(h).to(device).bfloat16()
    T.restore_fp32_params(base_model)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = base_model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        T.log(f"load:missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            T.log(f"  missing[0..3]={missing[:3]}")
        if unexpected:
            T.log(f"  unexpected[0..3]={unexpected[:3]}")
    base_model.to(device)

    val_data = T.ValidationData(h, device)
    T.log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    # Match end-of-training state: looping_active=True with h.num_loops loops.
    T.configure_eval_model(base_model, h, "pre-quant")
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    T.timed_eval("pre-quantization (loaded ckpt)", T.eval_val, h, device, val_data, compiled_model)

    # GPTQ + brotli serialize. This will OVERWRITE OUT_PTZ but the FP ckpt is at CKPT_PATH.
    T.fold_iha_mixes(base_model)
    code = Path(T.__file__).read_text(encoding="utf-8")
    bytes_total, quant_file_bytes = T.serialize(h, base_model, code)
    if distributed:
        dist.barrier()

    # Quantized eval.
    eval_model = T.configure_eval_model(T.deserialize(h, device), h, "quantized")
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    T.timed_eval("quantized", T.eval_val, h, device, val_data, compiled_eval)
    if h.sliding_window_enabled:
        T.timed_eval("quantized_sliding_window", T.eval_val_sliding, h, device, val_data, eval_model)
    if h.ttt_enabled and h.sliding_window_enabled:
        del eval_model, compiled_eval
        torch._dynamo.reset(); torch.cuda.empty_cache()
        ttt_model = T.configure_eval_model(T.deserialize(h, device), h, "ttt")
        T.timed_eval("quantized_ttt", T.eval_val_ttt, h, device, val_data, ttt_model)
        del ttt_model

    if h.is_main_process:
        T.log(f"requant_eval done: bytes_total={bytes_total} quant_file_bytes={quant_file_bytes}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
