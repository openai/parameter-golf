"""Eval-only script: loads a pre-quantized model and runs TTT eval."""
import os, sys, time, torch, torch.distributed as dist
sys.path.insert(0, "experiments")
from train_gpt_v1 import (Hyperparameters, set_logging_hparams, log,
                           ValidationData, deserialize, 
                           eval_val, eval_val_sliding, eval_val_ttt, timed_eval)

def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    h = Hyperparameters()
    set_logging_hparams(h)
    os.makedirs("logs", exist_ok=True)
    
    if h.is_main_process:
        log(f"Eval-only mode: loading quantized model")
        log(f"ttt_adaptive={h.ttt_adaptive} ttt_max_epochs={h.ttt_max_epochs} ttt_min_epochs={h.ttt_min_epochs}")
    
    val_data = ValidationData(h, device)
    log(f"val_tokens: {val_data.val_tokens.numel()-1}")
    
    # Load quantized model
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    
    # Run quantized eval
    compiled = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval("quantized", eval_val, h, device, val_data, compiled)
    
    # Sliding window
    if h.sliding_window_enabled:
        timed_eval("quantized_sliding_window", eval_val_sliding, h, device, val_data, eval_model)
    
    # TTT
    if h.ttt_enabled and h.sliding_window_enabled:
        del compiled
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        timed_eval("quantized_ttt", eval_val_ttt, h, device, val_data, ttt_model)
    
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
