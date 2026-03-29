"""Standalone TTT eval: loads quantized model and runs TTT with configurable epochs."""
import os, sys, time, math, io, lzma, torch, torch.distributed as dist
os.environ.setdefault("PYTHONUNBUFFERED", "1")

sys.path.insert(0, os.path.dirname(__file__))
from train_gpt import (
    Hyperparameters, GPT, ResidualScale, CastedLinear,
    dequantize_mixed_int6, _rebank_state_dict, _unbank_state_dict,
    load_validation_tokens, build_sentencepiece_luts,
    eval_val_sliding_ttt, restore_low_dim_params_to_fp32,
)
import sentencepiece as spm

def main():
    args = Hyperparameters()
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_flash_sdp, enable_cudnn_sdp, enable_mem_efficient_sdp, enable_math_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    def log0(msg):
        if rank == 0: print(msg)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    eval_passes = int(os.environ.get("EVAL_PASSES", "4"))
    residual_scale_init = float(os.environ.get("RESIDUAL_SCALE_INIT", "0.5"))

    with open("final_model.int6.ptz", "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")

    sd_cpu = torch.load("final_model.pt", map_location="cpu")
    unbanked_template = _unbank_state_dict(sd_cpu, args.num_layers)
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_template)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        core_start=args.core_start, core_end=args.core_end,
        num_passes=eval_passes, interpass_rmsnorm=False,
    ).to(device).bfloat16()

    eval_rs = ResidualScale(eval_passes, residual_scale_init).to(device)
    eval_model.residual_scale = eval_rs
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)

    torch.compile(eval_model, dynamic=False, fullgraph=True)

    log0(f"TTT_EPOCHS={args.ttt_epochs} EVAL_PASSES={eval_passes}")
    t0 = time.perf_counter()
    ttt_loss, ttt_bpb = eval_val_sliding_ttt(
        args, eval_model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.eval_stride, log0=log0,
    )
    elapsed = time.perf_counter() - t0
    log0(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} eval_time:{elapsed*1000:.0f}ms")
    log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
