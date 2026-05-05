"""Minimal TTT-only eval: loads quantized model and runs only TTT."""
import os, sys, time, torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt import (Hyperparameters, set_logging_hparams, log,
                        ValidationData, deserialize, eval_val_ttt, timed_eval)

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
        log(f"TTT-only eval: adaptive={h.ttt_adaptive} max={h.ttt_max_epochs} min={h.ttt_min_epochs}")
    val_data = ValidationData(h, device)
    log(f"val_tokens: {val_data.val_tokens.numel()-1}")
    ttt_model = deserialize(h, device)
    if h.num_loops > 0:
        ttt_model.looping_active = True
    timed_eval("quantized_ttt", eval_val_ttt, h, device, val_data, ttt_model)
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
