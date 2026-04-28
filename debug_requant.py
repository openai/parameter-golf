"""Quick debug: compare original deserialize vs manual requant path."""
import io, os, time, torch, torch.distributed as dist

from train_pr1493 import (
    Hyperparameters, GPT, ValidationData, ShuffledSequenceLoader,
    collect_hessians, gptq_quantize_weight, gptq_mixed_quantize,
    dequantize_mixed, _compress, _decompress,
    restore_fp32_params, eval_val_sliding, set_logging_hparams, log,
    CastedLinear, classify_param, deserialize,
)

def main():
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    h = Hyperparameters()
    set_logging_hparams(h)
    val_data = ValidationData(h, device)

    # Test 1: original deserialize path
    log("=== Test 1: original deserialize() ===")
    eval_model_orig = deserialize(h, device)
    torch._dynamo.reset()
    val_loss, val_bpb = eval_val_sliding(h, device, val_data, eval_model_orig)
    log(f"Original deserialize: val_bpb={val_bpb:.8f}")
    del eval_model_orig
    torch.cuda.empty_cache()

    # Test 2: manual requant with same sigmas as original
    log("\n=== Test 2: manual requant (same sigmas 12.85/20.0) ===")
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    sd_fp32 = torch.load(h.model_path, map_location='cpu', weights_only=True)
    base_model.load_state_dict(sd_fp32)
    sd_cpu = {k: v.detach().cpu() for k, v in sd_fp32.items()}

    log("Collecting hessians...")
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(base_model, calib_loader, h, device,
                                n_calibration_batches=h.gptq_calibration_batches)
    log(f"Collected {len(hessians)} hessians")

    # Use the original gptq_mixed_quantize (known-good path)
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    deq_state = dequantize_mixed(quant_result, quant_meta, sd_cpu)

    eval_model2 = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model2)
    eval_model2.load_state_dict(deq_state, strict=True)
    torch._dynamo.reset()
    val_loss, val_bpb = eval_val_sliding(h, device, val_data, eval_model2)
    log(f"Manual requant (gptq_mixed_quantize): val_bpb={val_bpb:.8f}")
    del eval_model2
    torch.cuda.empty_cache()

    # Test 3: same as test 2 but serialize/deserialize through brotli roundtrip
    log("\n=== Test 3: brotli roundtrip ===")
    quant_buf = io.BytesIO()
    torch.save({'w': quant_result, 'm': quant_meta}, quant_buf)
    quant_blob = _compress(quant_buf.getvalue(), h.compressor)
    log(f"Compressed size: {len(quant_blob):,} bytes")

    quant_state = torch.load(io.BytesIO(_decompress(quant_blob, h.compressor)), map_location='cpu')
    deq_state2 = dequantize_mixed(quant_state['w'], quant_state['m'], sd_cpu)

    eval_model3 = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model3)
    eval_model3.load_state_dict(deq_state2, strict=True)
    torch._dynamo.reset()
    val_loss, val_bpb = eval_val_sliding(h, device, val_data, eval_model3)
    log(f"Brotli roundtrip: val_bpb={val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
